//! Discriminant and flag-bit pins for every enum and bitset the interop ABI
//! publishes.
//!
//! Each enum is pinned twice: an exact numeric assertion per variant (the ABI
//! value C sees), and an exhaustive wildcard-free `match` so that ADDING a
//! variant fails compilation here — forcing the author to extend the pins and,
//! per the ABI-evolution policy, to justify the addition as a compatible
//! change.
//!
//! INVARIANT-HOOK: VT-ABI-5 — reserved fields default to zero and enum
//! discriminants are contiguous, unique, and pinned.
//!
//! Formal home: `docs/verification/smt/abi_layout.smt2` proves the manifest's
//! discriminant table is contiguous and collision-free; this file pins that
//! the Rust definitions match the manifest.

use vinary_tree_interop::{
    dictionary_flags, wfst_flags, VtDictionaryEdge, VtOptionalU64, VtStatus, VtUnitDomain,
    VtValueDomain, VtWeightDomain, VtWfstArc,
};

// ── VtStatus ────────────────────────────────────────────────────────────────

#[test]
fn status_discriminants_are_pinned() {
    assert_eq!(VtStatus::Ok as u32, 0);
    assert_eq!(VtStatus::End as u32, 1);
    assert_eq!(VtStatus::InvalidArgument as u32, 2);
    assert_eq!(VtStatus::NullPointer as u32, 3);
    assert_eq!(VtStatus::Unsupported as u32, 4);
    assert_eq!(VtStatus::IoError as u32, 5);
    assert_eq!(VtStatus::Closed as u32, 6);
    assert_eq!(VtStatus::LimitExceeded as u32, 7);
    assert_eq!(VtStatus::ProviderError as u32, 8);
}

#[test]
fn status_universe_is_exactly_nine_values() {
    // Wildcard-free: a new variant fails compilation here by design.
    let universe = [
        VtStatus::Ok,
        VtStatus::End,
        VtStatus::InvalidArgument,
        VtStatus::NullPointer,
        VtStatus::Unsupported,
        VtStatus::IoError,
        VtStatus::Closed,
        VtStatus::LimitExceeded,
        VtStatus::ProviderError,
    ];
    let mut seen = [false; 9];
    for status in universe {
        let value = match status {
            VtStatus::Ok => 0u32,
            VtStatus::End => 1,
            VtStatus::InvalidArgument => 2,
            VtStatus::NullPointer => 3,
            VtStatus::Unsupported => 4,
            VtStatus::IoError => 5,
            VtStatus::Closed => 6,
            VtStatus::LimitExceeded => 7,
            VtStatus::ProviderError => 8,
        };
        assert_eq!(
            value, status as u32,
            "match table drifted from discriminant"
        );
        assert!(
            !seen[value as usize],
            "duplicate discriminant {value} — two variants share an ABI value"
        );
        seen[value as usize] = true;
    }
    assert!(
        seen.iter().all(|&hit| hit),
        "discriminants are not contiguous 0..=8"
    );
}

/// The wire codec is a bijection onto the published range: every in-range
/// raw value decodes to the status that re-encodes to it, and nothing
/// outside 0..=8 decodes at all (the from_raw half of the LLEV-B6 wire
/// hardening — an out-of-range provider status is a VALUE, never UB).
#[test]
fn wire_round_trip_covers_exactly_the_published_range() {
    for raw in 0u32..=8 {
        let status = VtStatus::from_raw(raw).expect("in-range wire value decodes");
        assert_eq!(status.to_raw(), raw);
    }
    for raw in [9u32, 10, 42, u32::MAX] {
        assert!(
            VtStatus::from_raw(raw).is_none(),
            "out-of-range {raw} must not decode"
        );
    }
}

#[test]
fn is_ok_holds_exactly_for_ok() {
    let universe = [
        VtStatus::Ok,
        VtStatus::End,
        VtStatus::InvalidArgument,
        VtStatus::NullPointer,
        VtStatus::Unsupported,
        VtStatus::IoError,
        VtStatus::Closed,
        VtStatus::LimitExceeded,
        VtStatus::ProviderError,
    ];
    for status in universe {
        assert_eq!(
            status.is_ok(),
            matches!(status, VtStatus::Ok),
            "is_ok() must hold exactly for Ok, got a different answer for {status:?}"
        );
    }
}

// ── unit, value, and weight domains ─────────────────────────────────────────

#[test]
fn unit_domain_discriminants_are_pinned() {
    assert_eq!(VtUnitDomain::Byte as u32, 1);
    assert_eq!(VtUnitDomain::UnicodeScalar as u32, 2);
    assert_eq!(VtUnitDomain::U64 as u32, 3);
    for domain in [
        VtUnitDomain::Byte,
        VtUnitDomain::UnicodeScalar,
        VtUnitDomain::U64,
    ] {
        let value = match domain {
            VtUnitDomain::Byte => 1u32,
            VtUnitDomain::UnicodeScalar => 2,
            VtUnitDomain::U64 => 3,
        };
        assert_eq!(value, domain as u32);
    }
}

#[test]
fn value_domain_discriminants_are_pinned() {
    assert_eq!(VtValueDomain::Unit as u32, 0);
    assert_eq!(VtValueDomain::OptionalU64 as u32, 1);
    assert_eq!(VtValueDomain::Bytes as u32, 2);
    for domain in [
        VtValueDomain::Unit,
        VtValueDomain::OptionalU64,
        VtValueDomain::Bytes,
    ] {
        let value = match domain {
            VtValueDomain::Unit => 0u32,
            VtValueDomain::OptionalU64 => 1,
            VtValueDomain::Bytes => 2,
        };
        assert_eq!(value, domain as u32);
    }
}

#[test]
fn weight_domain_discriminants_are_pinned() {
    assert_eq!(VtWeightDomain::TropicalF64 as u32, 1);
    assert_eq!(VtWeightDomain::LogF64 as u32, 2);
    assert_eq!(VtWeightDomain::ProbabilityF64 as u32, 3);
    assert_eq!(VtWeightDomain::ArcticF64 as u32, 4);
    assert_eq!(VtWeightDomain::SignedTropicalF64 as u32, 5);
    assert_eq!(VtWeightDomain::CountF64 as u32, 6);
    assert_eq!(VtWeightDomain::BooleanF64 as u32, 7);
    let universe = [
        VtWeightDomain::TropicalF64,
        VtWeightDomain::LogF64,
        VtWeightDomain::ProbabilityF64,
        VtWeightDomain::ArcticF64,
        VtWeightDomain::SignedTropicalF64,
        VtWeightDomain::CountF64,
        VtWeightDomain::BooleanF64,
    ];
    let mut seen = [false; 8];
    for domain in universe {
        let value = match domain {
            VtWeightDomain::TropicalF64 => 1u32,
            VtWeightDomain::LogF64 => 2,
            VtWeightDomain::ProbabilityF64 => 3,
            VtWeightDomain::ArcticF64 => 4,
            VtWeightDomain::SignedTropicalF64 => 5,
            VtWeightDomain::CountF64 => 6,
            VtWeightDomain::BooleanF64 => 7,
        };
        assert_eq!(value, domain as u32);
        assert!(
            !seen[value as usize],
            "duplicate weight-domain value {value}"
        );
        seen[value as usize] = true;
    }
    assert!(
        seen[1..].iter().all(|&hit| hit),
        "weight-domain discriminants are not contiguous 1..=7"
    );
}

// ── capability flag bits ────────────────────────────────────────────────────

#[test]
fn dictionary_flag_bits_are_pinned_and_disjoint() {
    assert_eq!(dictionary_flags::PARALLEL_REENTRANT, 1 << 0);
    assert_eq!(dictionary_flags::SUFFIX_BASED, 1 << 1);
    assert_eq!(dictionary_flags::IMMUTABLE, 1 << 2);
    let all = dictionary_flags::PARALLEL_REENTRANT
        | dictionary_flags::SUFFIX_BASED
        | dictionary_flags::IMMUTABLE;
    assert_eq!(all.count_ones(), 3, "dictionary flag bits overlap");
}

#[test]
fn wfst_flag_bits_are_pinned_and_disjoint() {
    assert_eq!(wfst_flags::PARALLEL_REENTRANT, 1 << 0);
    assert_eq!(wfst_flags::IMMUTABLE, 1 << 1);
    assert_eq!(wfst_flags::LAZY, 1 << 2);
    assert_eq!(wfst_flags::ACYCLIC, 1 << 3);
    let all = wfst_flags::PARALLEL_REENTRANT
        | wfst_flags::IMMUTABLE
        | wfst_flags::LAZY
        | wfst_flags::ACYCLIC;
    assert_eq!(all.count_ones(), 4, "wfst flag bits overlap");
}

// ── zeroed defaults (the consumer-side half of reserved-must-be-zero) ───────

#[test]
fn defaults_are_all_zero_including_reserved_fields() {
    let optional = VtOptionalU64::default();
    assert_eq!(optional.value, 0);
    assert_eq!(optional.has_value, 0);
    assert_eq!(optional.reserved, [0u8; 7]);

    let edge = VtDictionaryEdge::default();
    assert_eq!(edge.label, 0);
    assert_eq!(edge.node, 0);

    let arc = VtWfstArc::default();
    assert_eq!(arc.input_label, 0);
    assert_eq!(arc.output_label, 0);
    assert_eq!(arc.target_state, 0);
    assert_eq!(arc.weight, 0.0);
    assert_eq!(arc.has_input, 0);
    assert_eq!(arc.has_output, 0);
    assert_eq!(arc.reserved, [0u8; 6]);
}
