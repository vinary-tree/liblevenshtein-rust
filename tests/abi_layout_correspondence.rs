//! Three-way ABI layout agreement for the vinary-tree interop resource ABI.
//!
//! INVARIANT-HOOK: VT-ABI-1..6 — manifest ↔ compiler ↔ SMT three-way agreement;
//! spec: docs/verification/smt/abi_layout.smt2
//!
//! `docs/verification/ABI_LAYOUT_MANIFEST.tsv` is the single source of truth
//! for the published layout. Three artifacts must agree, and each agreement
//! edge has exactly one owner:
//!
//! - manifest ↔ compiler — this file, via `offset_of!`/`size_of`/`align_of`
//!   on the running target (the LP64 columns on 64-bit hosts, the 32arm
//!   columns on CI's `linux-armv7-experimental` leg), alongside
//!   `vinary-tree-interop/tests/layout_contract.rs`, which pins the compiler
//!   against the same numbers upstream;
//! - manifest ↔ SMT — this file, via a mechanical scan of the
//!   `(define-fun …)` constant table in `docs/verification/smt/abi_layout.smt2`
//!   (the constant names are mechanical — `o64_VtResource_context` and
//!   friends — precisely so this scan can be exact);
//! - SMT self-consistency (no field overlap, alignment, tail padding,
//!   discriminant contiguity, flag-bit disjointness, identifier distinctness)
//!   — `docs/verification/smt/abi_layout.smt2`, gated dual-solver (Z3 and
//!   cvc5, every `(check-sat)` UNSAT).
//!
//! Wildcard-free enum sweeps are deliberately NOT repeated here:
//! `vinary-tree-interop/tests/discriminant_pins.rs` owns them. This file
//! asserts manifest agreement only, so no law has two executable homes.

#![cfg(feature = "bindings-core")]

use core::ffi::c_void;
use core::mem::{align_of, offset_of, size_of};
use std::collections::{BTreeMap, BTreeSet};

use vinary_tree_interop::{
    dictionary_flags, wfst_flags, VtDictionaryEdge, VtDictionaryVTable, VtInterfaceId,
    VtOptionalU64, VtResource, VtResourceVTable, VtStatus, VtUnitDomain, VtValueDomain,
    VtWeightDomain, VtWfstArc, VtWfstVTable, VT_ABI_VERSION, VT_DICTIONARY_INTERFACE_ID,
    VT_DICTIONARY_INTERFACE_VERSION, VT_RECOMMENDED_ARC_BATCH, VT_RECOMMENDED_EDGE_BATCH,
    VT_WFST_INTERFACE_ID, VT_WFST_INTERFACE_VERSION,
};

const MANIFEST: &str = include_str!("../docs/verification/ABI_LAYOUT_MANIFEST.tsv");
const SMT_MIRROR: &str = include_str!("../docs/verification/smt/abi_layout.smt2");

/// Representative optional C function pointer. All Rust function-pointer
/// types share one size and alignment on every supported target, and the
/// interop ABI additionally relies on the guaranteed null-pointer
/// optimization for `Option<extern "C" fn>` (VT-ABI-6), so one representative
/// type stands in for every vtable slot — the same simplification
/// `layout_contract.rs` makes with its `fn_ptr_width()`.
type FnPtr = Option<unsafe extern "C" fn(*mut c_void)>;

// ── manifest parsing ─────────────────────────────────────────────────────────

/// One data row of `ABI_LAYOUT_MANIFEST.tsv`.
#[derive(Debug)]
struct Row {
    kind: &'static str,
    ty: &'static str,
    member: &'static str,
    offset64: Option<u64>,
    size64: Option<u64>,
    offset32: Option<u64>,
    size32: Option<u64>,
    value: &'static str,
}

fn parse_cell(cell: &str, line_number: usize, column: &str) -> Option<u64> {
    if cell == "-" {
        None
    } else {
        Some(cell.parse::<u64>().unwrap_or_else(|error| {
            panic!("manifest line {line_number}: column {column} = {cell:?} is not a u64: {error}")
        }))
    }
}

fn parse_manifest() -> Vec<Row> {
    let mut rows = Vec::with_capacity(96);
    for (index, line) in MANIFEST.lines().enumerate() {
        let line_number = index + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            columns.len(),
            9,
            "manifest line {line_number}: expected 9 tab-separated columns, found {}",
            columns.len()
        );
        rows.push(Row {
            kind: columns[0],
            ty: columns[1],
            member: columns[2],
            offset64: parse_cell(columns[3], line_number, "offset64"),
            size64: parse_cell(columns[4], line_number, "size64"),
            offset32: parse_cell(columns[5], line_number, "offset32arm"),
            size32: parse_cell(columns[6], line_number, "size32arm"),
            value: columns[7],
        });
    }
    rows
}

// ── SMT mirror parsing ───────────────────────────────────────────────────────

/// Scan the SMT mirror for its mechanical zero-arity integer constants,
/// `(define-fun NAME () Int VALUE)` (an optional trailing `; comment` is
/// ignored). Helper predicates (`disj`, `aligned`, `inside`, `pow2`) take
/// parameters, so the `"() Int "` prefix filters them out exactly.
fn parse_smt_constants() -> BTreeMap<String, u64> {
    let mut constants = BTreeMap::new();
    for (index, raw_line) in SMT_MIRROR.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        let Some(rest) = line.strip_prefix("(define-fun ") else {
            continue;
        };
        let Some((name, rest)) = rest.split_once(' ') else {
            continue;
        };
        let Some(rest) = rest.strip_prefix("() Int ") else {
            continue;
        };
        let value_text = rest
            .split(')')
            .next()
            .expect("split always yields at least one piece")
            .trim();
        let value = value_text.parse::<u64>().unwrap_or_else(|error| {
            panic!("SMT mirror line {line_number}: constant {name} has non-u64 value {value_text:?}: {error}")
        });
        assert!(
            constants.insert(name.to_string(), value).is_none(),
            "SMT mirror line {line_number}: constant {name} is defined twice"
        );
    }
    constants
}

// ── live compiler facts ──────────────────────────────────────────────────────

/// `(offset, size, alignment)` of one published struct field on the running
/// target. Field widths/alignments use the field's own Rust type, with the
/// representative [`FnPtr`] standing in for every vtable slot.
fn live_field(ty: &str, member: &str) -> Option<(usize, usize, usize)> {
    macro_rules! entry {
        ($container:ty, $field:ident, $field_ty:ty) => {
            Some((
                offset_of!($container, $field),
                size_of::<$field_ty>(),
                align_of::<$field_ty>(),
            ))
        };
    }
    match (ty, member) {
        ("VtInterfaceId", "bytes") => entry!(VtInterfaceId, bytes, [u8; 16]),
        ("VtResource", "context") => entry!(VtResource, context, *mut c_void),
        ("VtResource", "vtable") => entry!(VtResource, vtable, *const VtResourceVTable),
        ("VtResourceVTable", "struct_size") => entry!(VtResourceVTable, struct_size, usize),
        ("VtResourceVTable", "abi_version") => entry!(VtResourceVTable, abi_version, u32),
        ("VtResourceVTable", "reserved") => entry!(VtResourceVTable, reserved, u32),
        ("VtResourceVTable", "retain") => entry!(VtResourceVTable, retain, FnPtr),
        ("VtResourceVTable", "release") => entry!(VtResourceVTable, release, FnPtr),
        ("VtResourceVTable", "query_interface") => entry!(VtResourceVTable, query_interface, FnPtr),
        ("VtOptionalU64", "value") => entry!(VtOptionalU64, value, u64),
        ("VtOptionalU64", "has_value") => entry!(VtOptionalU64, has_value, u8),
        ("VtOptionalU64", "reserved") => entry!(VtOptionalU64, reserved, [u8; 7]),
        ("VtDictionaryEdge", "label") => entry!(VtDictionaryEdge, label, u64),
        ("VtDictionaryEdge", "node") => entry!(VtDictionaryEdge, node, u64),
        ("VtDictionaryVTable", "struct_size") => entry!(VtDictionaryVTable, struct_size, usize),
        ("VtDictionaryVTable", "interface_version") => {
            entry!(VtDictionaryVTable, interface_version, u32)
        }
        ("VtDictionaryVTable", "unit_domain") => {
            entry!(VtDictionaryVTable, unit_domain, VtUnitDomain)
        }
        ("VtDictionaryVTable", "value_domain") => {
            entry!(VtDictionaryVTable, value_domain, VtValueDomain)
        }
        ("VtDictionaryVTable", "flags") => entry!(VtDictionaryVTable, flags, u64),
        ("VtDictionaryVTable", "snapshot") => entry!(VtDictionaryVTable, snapshot, FnPtr),
        ("VtDictionaryVTable", "root") => entry!(VtDictionaryVTable, root, FnPtr),
        ("VtDictionaryVTable", "len") => entry!(VtDictionaryVTable, len, FnPtr),
        ("VtDictionaryVTable", "node_is_final") => entry!(VtDictionaryVTable, node_is_final, FnPtr),
        ("VtDictionaryVTable", "node_value_u64") => {
            entry!(VtDictionaryVTable, node_value_u64, FnPtr)
        }
        ("VtDictionaryVTable", "node_transition") => {
            entry!(VtDictionaryVTable, node_transition, FnPtr)
        }
        ("VtDictionaryVTable", "node_edges") => entry!(VtDictionaryVTable, node_edges, FnPtr),
        ("VtWfstArc", "input_label") => entry!(VtWfstArc, input_label, u64),
        ("VtWfstArc", "output_label") => entry!(VtWfstArc, output_label, u64),
        ("VtWfstArc", "target_state") => entry!(VtWfstArc, target_state, u64),
        ("VtWfstArc", "weight") => entry!(VtWfstArc, weight, f64),
        ("VtWfstArc", "has_input") => entry!(VtWfstArc, has_input, u8),
        ("VtWfstArc", "has_output") => entry!(VtWfstArc, has_output, u8),
        ("VtWfstArc", "reserved") => entry!(VtWfstArc, reserved, [u8; 6]),
        ("VtWfstVTable", "struct_size") => entry!(VtWfstVTable, struct_size, usize),
        ("VtWfstVTable", "interface_version") => entry!(VtWfstVTable, interface_version, u32),
        ("VtWfstVTable", "unit_domain") => entry!(VtWfstVTable, unit_domain, VtUnitDomain),
        ("VtWfstVTable", "weight_domain") => entry!(VtWfstVTable, weight_domain, VtWeightDomain),
        ("VtWfstVTable", "reserved") => entry!(VtWfstVTable, reserved, u32),
        ("VtWfstVTable", "flags") => entry!(VtWfstVTable, flags, u64),
        ("VtWfstVTable", "snapshot") => entry!(VtWfstVTable, snapshot, FnPtr),
        ("VtWfstVTable", "start") => entry!(VtWfstVTable, start, FnPtr),
        ("VtWfstVTable", "num_states") => entry!(VtWfstVTable, num_states, FnPtr),
        ("VtWfstVTable", "state_info") => entry!(VtWfstVTable, state_info, FnPtr),
        ("VtWfstVTable", "state_arcs") => entry!(VtWfstVTable, state_arcs, FnPtr),
        _ => None,
    }
}

/// `(size, alignment)` of one published struct on the running target.
fn live_struct(ty: &str) -> Option<(usize, usize)> {
    match ty {
        "VtInterfaceId" => Some((size_of::<VtInterfaceId>(), align_of::<VtInterfaceId>())),
        "VtResource" => Some((size_of::<VtResource>(), align_of::<VtResource>())),
        "VtResourceVTable" => Some((
            size_of::<VtResourceVTable>(),
            align_of::<VtResourceVTable>(),
        )),
        "VtOptionalU64" => Some((size_of::<VtOptionalU64>(), align_of::<VtOptionalU64>())),
        "VtDictionaryEdge" => Some((
            size_of::<VtDictionaryEdge>(),
            align_of::<VtDictionaryEdge>(),
        )),
        "VtDictionaryVTable" => Some((
            size_of::<VtDictionaryVTable>(),
            align_of::<VtDictionaryVTable>(),
        )),
        "VtWfstArc" => Some((size_of::<VtWfstArc>(), align_of::<VtWfstArc>())),
        "VtWfstVTable" => Some((size_of::<VtWfstVTable>(), align_of::<VtWfstVTable>())),
        _ => None,
    }
}

/// Live discriminant of one published enum variant (all enums are repr(u32)).
fn live_discriminant(ty: &str, member: &str) -> Option<u64> {
    let discriminant = match (ty, member) {
        ("VtStatus", "Ok") => VtStatus::Ok as u64,
        ("VtStatus", "End") => VtStatus::End as u64,
        ("VtStatus", "InvalidArgument") => VtStatus::InvalidArgument as u64,
        ("VtStatus", "NullPointer") => VtStatus::NullPointer as u64,
        ("VtStatus", "Unsupported") => VtStatus::Unsupported as u64,
        ("VtStatus", "IoError") => VtStatus::IoError as u64,
        ("VtStatus", "Closed") => VtStatus::Closed as u64,
        ("VtStatus", "LimitExceeded") => VtStatus::LimitExceeded as u64,
        ("VtStatus", "ProviderError") => VtStatus::ProviderError as u64,
        ("VtUnitDomain", "Byte") => VtUnitDomain::Byte as u64,
        ("VtUnitDomain", "UnicodeScalar") => VtUnitDomain::UnicodeScalar as u64,
        ("VtUnitDomain", "U64") => VtUnitDomain::U64 as u64,
        ("VtValueDomain", "Unit") => VtValueDomain::Unit as u64,
        ("VtValueDomain", "OptionalU64") => VtValueDomain::OptionalU64 as u64,
        ("VtValueDomain", "Bytes") => VtValueDomain::Bytes as u64,
        ("VtWeightDomain", "TropicalF64") => VtWeightDomain::TropicalF64 as u64,
        ("VtWeightDomain", "LogF64") => VtWeightDomain::LogF64 as u64,
        ("VtWeightDomain", "ProbabilityF64") => VtWeightDomain::ProbabilityF64 as u64,
        ("VtWeightDomain", "ArcticF64") => VtWeightDomain::ArcticF64 as u64,
        ("VtWeightDomain", "SignedTropicalF64") => VtWeightDomain::SignedTropicalF64 as u64,
        ("VtWeightDomain", "CountF64") => VtWeightDomain::CountF64 as u64,
        ("VtWeightDomain", "BooleanF64") => VtWeightDomain::BooleanF64 as u64,
        _ => return None,
    };
    Some(discriminant)
}

/// Live value of one published flag bit.
fn live_flag(module: &str, member: &str) -> Option<u64> {
    match (module, member) {
        ("dictionary_flags", "PARALLEL_REENTRANT") => Some(dictionary_flags::PARALLEL_REENTRANT),
        ("dictionary_flags", "SUFFIX_BASED") => Some(dictionary_flags::SUFFIX_BASED),
        ("dictionary_flags", "IMMUTABLE") => Some(dictionary_flags::IMMUTABLE),
        ("wfst_flags", "PARALLEL_REENTRANT") => Some(wfst_flags::PARALLEL_REENTRANT),
        ("wfst_flags", "IMMUTABLE") => Some(wfst_flags::IMMUTABLE),
        ("wfst_flags", "LAZY") => Some(wfst_flags::LAZY),
        ("wfst_flags", "ACYCLIC") => Some(wfst_flags::ACYCLIC),
        _ => None,
    }
}

/// Live value of one published numeric constant.
fn live_constant(name: &str) -> Option<u64> {
    match name {
        "VT_ABI_VERSION" => Some(u64::from(VT_ABI_VERSION)),
        "VT_DICTIONARY_INTERFACE_VERSION" => Some(u64::from(VT_DICTIONARY_INTERFACE_VERSION)),
        "VT_WFST_INTERFACE_VERSION" => Some(u64::from(VT_WFST_INTERFACE_VERSION)),
        "VT_RECOMMENDED_EDGE_BATCH" => Some(VT_RECOMMENDED_EDGE_BATCH as u64),
        "VT_RECOMMENDED_ARC_BATCH" => Some(VT_RECOMMENDED_ARC_BATCH as u64),
        _ => None,
    }
}

/// Live bytes and SMT-name prefix of one published interface identifier.
fn live_interface_id(name: &str) -> Option<(&'static [u8; 16], &'static str)> {
    match name {
        "VT_DICTIONARY_INTERFACE_ID" => Some((&VT_DICTIONARY_INTERFACE_ID.bytes, "dictionary")),
        "VT_WFST_INTERFACE_ID" => Some((&VT_WFST_INTERFACE_ID.bytes, "wfst")),
        _ => None,
    }
}

// ── target tier selection ────────────────────────────────────────────────────

/// The two exactly-pinned tiers of the published ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tier {
    /// Required LP64 tier (x86_64/aarch64): the `offset64`/`size64` columns.
    Lp64,
    /// 32-bit ARM EABI tier (CI `linux-armv7-experimental`): the
    /// `offset32arm`/`size32arm` columns.
    Arm32,
}

fn current_tier() -> Option<Tier> {
    if cfg!(target_pointer_width = "64") {
        Some(Tier::Lp64)
    } else if cfg!(all(target_pointer_width = "32", target_arch = "arm")) {
        Some(Tier::Arm32)
    } else {
        None
    }
}

fn tier_offset_size(row: &Row, tier: Tier) -> (Option<u64>, Option<u64>) {
    match tier {
        Tier::Lp64 => (row.offset64, row.size64),
        Tier::Arm32 => (row.offset32, row.size32),
    }
}

// ── manifest shape ───────────────────────────────────────────────────────────

const EXPECTED_FIELD_COUNTS: &[(&str, usize)] = &[
    ("VtInterfaceId", 1),
    ("VtResource", 2),
    ("VtResourceVTable", 6),
    ("VtOptionalU64", 3),
    ("VtDictionaryEdge", 2),
    ("VtDictionaryVTable", 12),
    ("VtWfstArc", 7),
    ("VtWfstVTable", 11),
];

const EXPECTED_ENUM_COUNTS: &[(&str, usize)] = &[
    ("VtStatus", 9),
    ("VtUnitDomain", 3),
    ("VtValueDomain", 3),
    ("VtWeightDomain", 7),
];

const EXPECTED_FLAG_COUNTS: &[(&str, usize)] = &[("dictionary_flags", 3), ("wfst_flags", 4)];

const EXPECTED_NUMERIC_CONSTANTS: usize = 5;
const EXPECTED_INTERFACE_IDS: usize = 2;

#[test]
fn manifest_shape_is_complete_and_unique() {
    let rows = parse_manifest();

    let mut seen = BTreeSet::new();
    let mut field_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut total_rows: BTreeMap<&str, (bool, bool)> = BTreeMap::new();
    let mut enum_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut flag_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut numeric_constants = 0usize;
    let mut interface_ids = 0usize;

    for row in &rows {
        assert!(
            seen.insert((row.kind, row.ty, row.member)),
            "duplicate manifest row: {} {} {}",
            row.kind,
            row.ty,
            row.member
        );
        match row.kind {
            "struct" => {
                assert_eq!(row.value, "-", "struct rows carry no value: {row:?}");
                match row.member {
                    "__size" | "__align" => {
                        assert!(
                            row.offset64.is_none() && row.offset32.is_none(),
                            "total rows carry no offsets: {row:?}"
                        );
                        assert!(
                            row.size64.is_some() && row.size32.is_some(),
                            "total rows carry both tier values: {row:?}"
                        );
                        let entry = total_rows.entry(row.ty).or_default();
                        if row.member == "__size" {
                            entry.0 = true;
                        } else {
                            entry.1 = true;
                        }
                    }
                    _ => {
                        assert!(
                            row.offset64.is_some()
                                && row.size64.is_some()
                                && row.offset32.is_some()
                                && row.size32.is_some(),
                            "field rows carry offsets and sizes for both tiers: {row:?}"
                        );
                        *field_counts.entry(row.ty).or_default() += 1;
                    }
                }
            }
            "enum" | "flag" | "const" => {
                assert!(
                    row.offset64.is_none()
                        && row.size64.is_none()
                        && row.offset32.is_none()
                        && row.size32.is_none(),
                    "{} rows carry no offset/size columns: {row:?}",
                    row.kind
                );
                match row.kind {
                    "enum" => *enum_counts.entry(row.ty).or_default() += 1,
                    "flag" => *flag_counts.entry(row.ty).or_default() += 1,
                    _ => {
                        if row.ty == "VtInterfaceId" {
                            interface_ids += 1;
                        } else {
                            numeric_constants += 1;
                        }
                    }
                }
            }
            other => panic!("unknown manifest row kind {other:?}: {row:?}"),
        }
    }

    for (ty, expected) in EXPECTED_FIELD_COUNTS {
        assert_eq!(
            field_counts.get(ty),
            Some(expected),
            "field-row count for {ty}"
        );
        assert_eq!(
            total_rows.get(ty),
            Some(&(true, true)),
            "{ty} must carry __size and __align rows"
        );
    }
    assert_eq!(field_counts.len(), EXPECTED_FIELD_COUNTS.len());
    assert_eq!(total_rows.len(), EXPECTED_FIELD_COUNTS.len());
    for (ty, expected) in EXPECTED_ENUM_COUNTS {
        assert_eq!(
            enum_counts.get(ty),
            Some(expected),
            "variant count for {ty}"
        );
    }
    assert_eq!(enum_counts.len(), EXPECTED_ENUM_COUNTS.len());
    for (module, expected) in EXPECTED_FLAG_COUNTS {
        assert_eq!(
            flag_counts.get(module),
            Some(expected),
            "flag count for {module}"
        );
    }
    assert_eq!(flag_counts.len(), EXPECTED_FLAG_COUNTS.len());
    assert_eq!(numeric_constants, EXPECTED_NUMERIC_CONSTANTS);
    assert_eq!(interface_ids, EXPECTED_INTERFACE_IDS);
}

// ── manifest ↔ compiler ──────────────────────────────────────────────────────

#[test]
fn optional_fn_pointer_niche_matches_pointer_width() {
    // VT-ABI-6 anchor for this file's use of one representative fn-pointer
    // type for every vtable slot's width and alignment.
    assert_eq!(size_of::<FnPtr>(), size_of::<*const c_void>());
    assert_eq!(align_of::<FnPtr>(), align_of::<*const c_void>());
}

#[test]
fn manifest_matches_live_layout_on_this_target() {
    let Some(tier) = current_tier() else {
        eprintln!("skipping: no exactly-pinned manifest tier for this target");
        return;
    };
    for row in parse_manifest() {
        if row.kind != "struct" {
            continue;
        }
        let (tsv_offset, tsv_size) = tier_offset_size(&row, tier);
        match row.member {
            "__size" => {
                let (live_size, _) = live_struct(row.ty)
                    .unwrap_or_else(|| panic!("manifest names unknown struct {}", row.ty));
                assert_eq!(
                    tsv_size,
                    Some(live_size as u64),
                    "{} total size on {tier:?}",
                    row.ty
                );
            }
            "__align" => {
                let (_, live_align) = live_struct(row.ty)
                    .unwrap_or_else(|| panic!("manifest names unknown struct {}", row.ty));
                assert_eq!(
                    tsv_size,
                    Some(live_align as u64),
                    "{} alignment on {tier:?}",
                    row.ty
                );
            }
            member => {
                let (live_offset, live_size, _) = live_field(row.ty, member)
                    .unwrap_or_else(|| panic!("manifest names unknown field {}.{member}", row.ty));
                assert_eq!(
                    tsv_offset,
                    Some(live_offset as u64),
                    "{}.{member} offset on {tier:?}",
                    row.ty
                );
                assert_eq!(
                    tsv_size,
                    Some(live_size as u64),
                    "{}.{member} width on {tier:?}",
                    row.ty
                );
            }
        }
    }
}

#[test]
fn manifest_matches_live_discriminants_flags_and_constants() {
    for row in parse_manifest() {
        match row.kind {
            "enum" => {
                let live = live_discriminant(row.ty, row.member).unwrap_or_else(|| {
                    panic!(
                        "manifest names unknown enum variant {}::{}",
                        row.ty, row.member
                    )
                });
                let manifest: u64 = row.value.parse().expect("enum value column is a u64");
                assert_eq!(manifest, live, "{}::{} discriminant", row.ty, row.member);
            }
            "flag" => {
                let live = live_flag(row.ty, row.member).unwrap_or_else(|| {
                    panic!("manifest names unknown flag {}::{}", row.ty, row.member)
                });
                let manifest: u64 = row.value.parse().expect("flag value column is a u64");
                assert_eq!(manifest, live, "{}::{} bit value", row.ty, row.member);
            }
            "const" if row.ty == "VtInterfaceId" => {
                let (live_bytes, _) = live_interface_id(row.member).unwrap_or_else(|| {
                    panic!("manifest names unknown interface id {}", row.member)
                });
                assert_eq!(
                    row.value.len(),
                    16,
                    "{} value column must be exactly 16 ASCII bytes",
                    row.member
                );
                assert_eq!(
                    row.value.as_bytes(),
                    live_bytes,
                    "{} identifier bytes",
                    row.member
                );
            }
            "const" => {
                let live = live_constant(row.member)
                    .unwrap_or_else(|| panic!("manifest names unknown constant {}", row.member));
                let manifest: u64 = row.value.parse().expect("const value column is a u64");
                assert_eq!(manifest, live, "{} value", row.member);
            }
            _ => {}
        }
    }
}

// ── manifest ↔ SMT mirror ────────────────────────────────────────────────────

#[test]
fn smt_mirror_matches_manifest_cell_by_cell() {
    let rows = parse_manifest();
    let smt = parse_smt_constants();
    let mut expected_names: BTreeSet<String> = ["ptr64".to_string(), "ptr32".to_string()]
        .into_iter()
        .collect();

    let lookup = |name: &str| -> u64 {
        *smt.get(name)
            .unwrap_or_else(|| panic!("SMT mirror is missing constant {name}"))
    };

    assert_eq!(lookup("ptr64"), 8, "LP64 pointer width");
    assert_eq!(lookup("ptr32"), 4, "32-bit ARM pointer width");

    for row in &rows {
        match row.kind {
            "struct" => match row.member {
                "__size" => {
                    for (prefix, cell) in [("sz64", row.size64), ("sz32", row.size32)] {
                        let name = format!("{prefix}_{}", row.ty);
                        assert_eq!(Some(lookup(&name)), cell, "{name} vs manifest");
                        expected_names.insert(name);
                    }
                }
                "__align" => {
                    for (prefix, cell) in [("al64", row.size64), ("al32", row.size32)] {
                        let name = format!("{prefix}_{}", row.ty);
                        assert_eq!(Some(lookup(&name)), cell, "{name} vs manifest");
                        expected_names.insert(name);
                    }
                }
                member => {
                    for (prefix, cell) in [
                        ("o64", row.offset64),
                        ("s64", row.size64),
                        ("o32", row.offset32),
                        ("s32", row.size32),
                    ] {
                        let name = format!("{prefix}_{}_{member}", row.ty);
                        assert_eq!(Some(lookup(&name)), cell, "{name} vs manifest");
                        expected_names.insert(name);
                    }
                    // The alignment mirrors are not manifest cells; they are
                    // AAPCS/LP64 natural-alignment claims proved consistent
                    // with the offsets by SMT property (c) and checked live
                    // against the compiler on each pinned tier below.
                    for prefix in ["a64", "a32"] {
                        let name = format!("{prefix}_{}_{member}", row.ty);
                        lookup(&name);
                        expected_names.insert(name);
                    }
                }
            },
            "enum" => {
                let name = format!("d_{}_{}", row.ty, row.member);
                let manifest: u64 = row.value.parse().expect("enum value column is a u64");
                assert_eq!(lookup(&name), manifest, "{name} vs manifest");
                expected_names.insert(name);
            }
            "flag" => {
                let name = format!("f_{}_{}", row.ty, row.member);
                let manifest: u64 = row.value.parse().expect("flag value column is a u64");
                assert_eq!(lookup(&name), manifest, "{name} vs manifest");
                expected_names.insert(name);
            }
            "const" if row.ty == "VtInterfaceId" => {
                let (_, prefix) = live_interface_id(row.member).unwrap_or_else(|| {
                    panic!("manifest names unknown interface id {}", row.member)
                });
                let length_name = format!("id_{prefix}_len");
                assert_eq!(lookup(&length_name), 16, "{length_name} vs manifest");
                expected_names.insert(length_name);
                for (index, byte) in row.value.bytes().enumerate() {
                    let name = format!("id_{prefix}_{index}");
                    assert_eq!(lookup(&name), u64::from(byte), "{name} vs manifest");
                    expected_names.insert(name);
                }
            }
            "const" => {
                let name = format!("c_{}", row.member);
                let manifest: u64 = row.value.parse().expect("const value column is a u64");
                assert_eq!(lookup(&name), manifest, "{name} vs manifest");
                expected_names.insert(name);
            }
            other => panic!("unknown manifest row kind {other:?}"),
        }
    }

    // Live check of the tier's alignment mirrors (a64_* on LP64 hosts, a32_*
    // on the armv7 CI leg) against the compiler.
    if let Some(tier) = current_tier() {
        let prefix = match tier {
            Tier::Lp64 => "a64",
            Tier::Arm32 => "a32",
        };
        for row in &rows {
            if row.kind != "struct" || row.member.starts_with("__") {
                continue;
            }
            let (_, _, live_align) = live_field(row.ty, row.member).unwrap_or_else(|| {
                panic!("manifest names unknown field {}.{}", row.ty, row.member)
            });
            let name = format!("{prefix}_{}_{}", row.ty, row.member);
            assert_eq!(
                lookup(&name),
                live_align as u64,
                "{name} vs live field alignment on {tier:?}"
            );
        }
    }

    // No orphans: every mechanical constant in the SMT mirror corresponds to
    // a manifest row (bidirectional set equality).
    let actual_names: BTreeSet<String> = smt.keys().cloned().collect();
    let missing: Vec<&String> = expected_names.difference(&actual_names).collect();
    let orphaned: Vec<&String> = actual_names.difference(&expected_names).collect();
    assert!(
        missing.is_empty() && orphaned.is_empty(),
        "SMT mirror and manifest disagree on the constant universe:\n  missing from SMT: {missing:?}\n  orphaned in SMT: {orphaned:?}"
    );
}
