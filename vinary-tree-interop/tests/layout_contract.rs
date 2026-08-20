//! Layout contract for every `#[repr(C)]` type the interop ABI publishes.
//!
//! These pins are the executable half of the ABI-stability obligation: a field
//! reorder, width change, or padding shift in `vinary-tree-interop` is an ABI
//! break for every independently built producer and consumer, so it must fail
//! here first. Exact tables are pinned for the required 64-bit tier and for the
//! 32-bit ARM EABI tier exercised by CI's `linux-armv7-experimental` leg;
//! target-independent structural laws (declaration-order monotonicity, no
//! overlap, tail padding) are asserted everywhere.
//!
//! INVARIANT-HOOK: VT-ABI-1 — struct sizes/alignments match the published ABI.
//! INVARIANT-HOOK: VT-ABI-2 — field offsets match the published ABI.
//! INVARIANT-HOOK: VT-ABI-3 — `VtResource` is exactly two machine words.
//! INVARIANT-HOOK: VT-ABI-4 — interface identifiers are exact 16-byte strings.
//! INVARIANT-HOOK: VT-ABI-6 — `Option<unsafe extern "C" fn>` has the null niche
//! (an absent vtable entry is a null function pointer, byte-for-byte).
//!
//! Formal home: `docs/verification/ABI_LAYOUT_MANIFEST.tsv` +
//! `docs/verification/smt/abi_layout.smt2` (manifest self-consistency, wave W1);
//! this file pins what SMT cannot: the actual compiler-produced layout.

use core::ffi::c_void;
use core::mem::{align_of, offset_of, size_of};

use vinary_tree_interop::{
    VtDictionaryEdge, VtDictionaryEntriesCursor, VtDictionaryEntriesInfo,
    VtDictionaryEntriesVTable, VtDictionaryEntry, VtDictionaryEntryBatchLimits,
    VtDictionaryEntryBatchView, VtDictionaryGraphEdge, VtDictionaryGraphNode,
    VtDictionaryGraphVTable, VtDictionaryGraphView, VtDictionaryVTable, VtDictionaryVisitVTable,
    VtInterfaceId, VtOptionalU64, VtResource, VtResourceVTable, VtSnapshotIdentity,
    VtSnapshotIdentityVTable, VtWfstArc, VtWfstVTable, VT_ABI_VERSION,
    VT_DICTIONARY_ENTRIES_INTERFACE_ID, VT_DICTIONARY_ENTRIES_INTERFACE_VERSION,
    VT_DICTIONARY_GRAPH_INTERFACE_ID, VT_DICTIONARY_GRAPH_INTERFACE_VERSION,
    VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION, VT_DICTIONARY_VISIT_INTERFACE_ID,
    VT_DICTIONARY_VISIT_INTERFACE_VERSION, VT_RECOMMENDED_ARC_BATCH, VT_RECOMMENDED_EDGE_BATCH,
    VT_SNAPSHOT_IDENTITY_INTERFACE_ID, VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION,
    VT_WFST_INTERFACE_ID, VT_WFST_INTERFACE_VERSION,
};

/// One machine word.
const WORD: usize = size_of::<usize>();

// ── target-independent structural laws ──────────────────────────────────────

/// Assert that `offsets` (declaration order) are strictly increasing, that no
/// field of the given widths overlaps its successor, and that the struct size
/// is the last field's end rounded up to the alignment.
fn assert_packed_in_order(
    type_name: &str,
    size: usize,
    align: usize,
    fields: &[(&str, usize, usize)],
) {
    let mut previous_end = 0usize;
    for (name, offset, width) in fields {
        assert!(
            *offset >= previous_end,
            "{type_name}.{name}: offset {offset} overlaps the previous field (ends at {previous_end})"
        );
        previous_end = offset + width;
    }
    assert!(
        previous_end <= size,
        "{type_name}: last field ends at {previous_end}, beyond size {size}"
    );
    assert_eq!(
        size % align,
        0,
        "{type_name}: size {size} is not a multiple of alignment {align}"
    );
    assert!(
        size - previous_end < align,
        "{type_name}: {} bytes of tail padding exceed one alignment unit — a field \
         was removed or the layout drifted",
        size - previous_end
    );
}

fn fn_ptr_width() -> usize {
    size_of::<Option<unsafe extern "C" fn(*mut c_void)>>()
}

#[test]
fn every_struct_is_packed_in_declaration_order() {
    let ptr = WORD;
    let fnp = fn_ptr_width();
    assert_packed_in_order(
        "VtResource",
        size_of::<VtResource>(),
        align_of::<VtResource>(),
        &[
            ("context", offset_of!(VtResource, context), ptr),
            ("vtable", offset_of!(VtResource, vtable), ptr),
        ],
    );
    assert_packed_in_order(
        "VtResourceVTable",
        size_of::<VtResourceVTable>(),
        align_of::<VtResourceVTable>(),
        &[
            (
                "struct_size",
                offset_of!(VtResourceVTable, struct_size),
                WORD,
            ),
            ("abi_version", offset_of!(VtResourceVTable, abi_version), 4),
            ("reserved", offset_of!(VtResourceVTable, reserved), 4),
            ("retain", offset_of!(VtResourceVTable, retain), fnp),
            ("release", offset_of!(VtResourceVTable, release), fnp),
            (
                "query_interface",
                offset_of!(VtResourceVTable, query_interface),
                fnp,
            ),
        ],
    );
    assert_packed_in_order(
        "VtOptionalU64",
        size_of::<VtOptionalU64>(),
        align_of::<VtOptionalU64>(),
        &[
            ("value", offset_of!(VtOptionalU64, value), 8),
            ("has_value", offset_of!(VtOptionalU64, has_value), 1),
            ("reserved", offset_of!(VtOptionalU64, reserved), 7),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEdge",
        size_of::<VtDictionaryEdge>(),
        align_of::<VtDictionaryEdge>(),
        &[
            ("label", offset_of!(VtDictionaryEdge, label), 8),
            ("node", offset_of!(VtDictionaryEdge, node), 8),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryGraphNode",
        size_of::<VtDictionaryGraphNode>(),
        align_of::<VtDictionaryGraphNode>(),
        &[
            (
                "edge_start",
                offset_of!(VtDictionaryGraphNode, edge_start),
                8,
            ),
            ("edge_len", offset_of!(VtDictionaryGraphNode, edge_len), 8),
            (
                "value_cursor",
                offset_of!(VtDictionaryGraphNode, value_cursor),
                8,
            ),
            ("is_final", offset_of!(VtDictionaryGraphNode, is_final), 1),
            ("reserved", offset_of!(VtDictionaryGraphNode, reserved), 7),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryGraphEdge",
        size_of::<VtDictionaryGraphEdge>(),
        align_of::<VtDictionaryGraphEdge>(),
        &[
            ("label", offset_of!(VtDictionaryGraphEdge, label), 8),
            ("target", offset_of!(VtDictionaryGraphEdge, target), 8),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryGraphView",
        size_of::<VtDictionaryGraphView>(),
        align_of::<VtDictionaryGraphView>(),
        &[
            ("nodes", offset_of!(VtDictionaryGraphView, nodes), ptr),
            (
                "node_count",
                offset_of!(VtDictionaryGraphView, node_count),
                WORD,
            ),
            ("edges", offset_of!(VtDictionaryGraphView, edges), ptr),
            (
                "edge_count",
                offset_of!(VtDictionaryGraphView, edge_count),
                WORD,
            ),
            ("root", offset_of!(VtDictionaryGraphView, root), 8),
            ("reserved", offset_of!(VtDictionaryGraphView, reserved), 8),
        ],
    );
    assert_packed_in_order(
        "VtSnapshotIdentity",
        size_of::<VtSnapshotIdentity>(),
        align_of::<VtSnapshotIdentity>(),
        &[
            ("producer", offset_of!(VtSnapshotIdentity, producer), 8),
            ("revision", offset_of!(VtSnapshotIdentity, revision), 8),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEntry",
        size_of::<VtDictionaryEntry>(),
        align_of::<VtDictionaryEntry>(),
        &[
            (
                "unit_offset",
                offset_of!(VtDictionaryEntry, unit_offset),
                WORD,
            ),
            ("unit_len", offset_of!(VtDictionaryEntry, unit_len), WORD),
            (
                "value_offset",
                offset_of!(VtDictionaryEntry, value_offset),
                WORD,
            ),
            ("value_len", offset_of!(VtDictionaryEntry, value_len), WORD),
            ("reserved", offset_of!(VtDictionaryEntry, reserved), 8),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEntryBatchLimits",
        size_of::<VtDictionaryEntryBatchLimits>(),
        align_of::<VtDictionaryEntryBatchLimits>(),
        &[
            (
                "max_entries",
                offset_of!(VtDictionaryEntryBatchLimits, max_entries),
                WORD,
            ),
            (
                "max_units",
                offset_of!(VtDictionaryEntryBatchLimits, max_units),
                WORD,
            ),
            (
                "max_values",
                offset_of!(VtDictionaryEntryBatchLimits, max_values),
                WORD,
            ),
            (
                "reserved",
                offset_of!(VtDictionaryEntryBatchLimits, reserved),
                8,
            ),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEntryBatchView",
        size_of::<VtDictionaryEntryBatchView>(),
        align_of::<VtDictionaryEntryBatchView>(),
        &[
            (
                "entries",
                offset_of!(VtDictionaryEntryBatchView, entries),
                ptr,
            ),
            (
                "entry_count",
                offset_of!(VtDictionaryEntryBatchView, entry_count),
                WORD,
            ),
            ("units", offset_of!(VtDictionaryEntryBatchView, units), ptr),
            (
                "unit_count",
                offset_of!(VtDictionaryEntryBatchView, unit_count),
                WORD,
            ),
            (
                "values",
                offset_of!(VtDictionaryEntryBatchView, values),
                ptr,
            ),
            (
                "value_count",
                offset_of!(VtDictionaryEntryBatchView, value_count),
                WORD,
            ),
            (
                "generation",
                offset_of!(VtDictionaryEntryBatchView, generation),
                8,
            ),
            (
                "reserved",
                offset_of!(VtDictionaryEntryBatchView, reserved),
                8,
            ),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEntriesInfo",
        size_of::<VtDictionaryEntriesInfo>(),
        align_of::<VtDictionaryEntriesInfo>(),
        &[
            (
                "unit_domain",
                offset_of!(VtDictionaryEntriesInfo, unit_domain),
                4,
            ),
            (
                "value_domain",
                offset_of!(VtDictionaryEntriesInfo, value_domain),
                4,
            ),
            ("order", offset_of!(VtDictionaryEntriesInfo, order), 4),
            (
                "reserved0",
                offset_of!(VtDictionaryEntriesInfo, reserved0),
                4,
            ),
            ("flags", offset_of!(VtDictionaryEntriesInfo, flags), 8),
            (
                "exact_len",
                offset_of!(VtDictionaryEntriesInfo, exact_len),
                WORD,
            ),
            (
                "identity",
                offset_of!(VtDictionaryEntriesInfo, identity),
                size_of::<VtSnapshotIdentity>(),
            ),
            (
                "reserved",
                offset_of!(VtDictionaryEntriesInfo, reserved),
                16,
            ),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEntriesCursor",
        size_of::<VtDictionaryEntriesCursor>(),
        align_of::<VtDictionaryEntriesCursor>(),
        &[
            (
                "context",
                offset_of!(VtDictionaryEntriesCursor, context),
                ptr,
            ),
            ("vtable", offset_of!(VtDictionaryEntriesCursor, vtable), ptr),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryVTable",
        size_of::<VtDictionaryVTable>(),
        align_of::<VtDictionaryVTable>(),
        &[
            (
                "struct_size",
                offset_of!(VtDictionaryVTable, struct_size),
                WORD,
            ),
            (
                "interface_version",
                offset_of!(VtDictionaryVTable, interface_version),
                4,
            ),
            (
                "unit_domain",
                offset_of!(VtDictionaryVTable, unit_domain),
                4,
            ),
            (
                "value_domain",
                offset_of!(VtDictionaryVTable, value_domain),
                4,
            ),
            ("flags", offset_of!(VtDictionaryVTable, flags), 8),
            ("snapshot", offset_of!(VtDictionaryVTable, snapshot), fnp),
            ("root", offset_of!(VtDictionaryVTable, root), fnp),
            ("len", offset_of!(VtDictionaryVTable, len), fnp),
            (
                "node_is_final",
                offset_of!(VtDictionaryVTable, node_is_final),
                fnp,
            ),
            (
                "node_value_u64",
                offset_of!(VtDictionaryVTable, node_value_u64),
                fnp,
            ),
            (
                "node_transition",
                offset_of!(VtDictionaryVTable, node_transition),
                fnp,
            ),
            (
                "node_edges",
                offset_of!(VtDictionaryVTable, node_edges),
                fnp,
            ),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryVisitVTable",
        size_of::<VtDictionaryVisitVTable>(),
        align_of::<VtDictionaryVisitVTable>(),
        &[
            (
                "struct_size",
                offset_of!(VtDictionaryVisitVTable, struct_size),
                WORD,
            ),
            (
                "interface_version",
                offset_of!(VtDictionaryVisitVTable, interface_version),
                4,
            ),
            ("reserved", offset_of!(VtDictionaryVisitVTable, reserved), 4),
            (
                "node_visit",
                offset_of!(VtDictionaryVisitVTable, node_visit),
                fnp,
            ),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryGraphVTable",
        size_of::<VtDictionaryGraphVTable>(),
        align_of::<VtDictionaryGraphVTable>(),
        &[
            (
                "struct_size",
                offset_of!(VtDictionaryGraphVTable, struct_size),
                WORD,
            ),
            (
                "interface_version",
                offset_of!(VtDictionaryGraphVTable, interface_version),
                4,
            ),
            ("reserved", offset_of!(VtDictionaryGraphVTable, reserved), 4),
            ("graph", offset_of!(VtDictionaryGraphVTable, graph), fnp),
            (
                "node_value_u64",
                offset_of!(VtDictionaryGraphVTable, node_value_u64),
                fnp,
            ),
        ],
    );
    assert_packed_in_order(
        "VtSnapshotIdentityVTable",
        size_of::<VtSnapshotIdentityVTable>(),
        align_of::<VtSnapshotIdentityVTable>(),
        &[
            (
                "struct_size",
                offset_of!(VtSnapshotIdentityVTable, struct_size),
                WORD,
            ),
            (
                "interface_version",
                offset_of!(VtSnapshotIdentityVTable, interface_version),
                4,
            ),
            (
                "reserved",
                offset_of!(VtSnapshotIdentityVTable, reserved),
                4,
            ),
            (
                "identity",
                offset_of!(VtSnapshotIdentityVTable, identity),
                fnp,
            ),
        ],
    );
    assert_packed_in_order(
        "VtDictionaryEntriesVTable",
        size_of::<VtDictionaryEntriesVTable>(),
        align_of::<VtDictionaryEntriesVTable>(),
        &[
            (
                "struct_size",
                offset_of!(VtDictionaryEntriesVTable, struct_size),
                WORD,
            ),
            (
                "interface_version",
                offset_of!(VtDictionaryEntriesVTable, interface_version),
                4,
            ),
            (
                "reserved",
                offset_of!(VtDictionaryEntriesVTable, reserved),
                4,
            ),
            ("open", offset_of!(VtDictionaryEntriesVTable, open), fnp),
            (
                "next_batch",
                offset_of!(VtDictionaryEntriesVTable, next_batch),
                fnp,
            ),
            (
                "release_batch",
                offset_of!(VtDictionaryEntriesVTable, release_batch),
                fnp,
            ),
            ("reduce", offset_of!(VtDictionaryEntriesVTable, reduce), fnp),
            ("cancel", offset_of!(VtDictionaryEntriesVTable, cancel), fnp),
            ("close", offset_of!(VtDictionaryEntriesVTable, close), fnp),
        ],
    );
    assert_packed_in_order(
        "VtWfstArc",
        size_of::<VtWfstArc>(),
        align_of::<VtWfstArc>(),
        &[
            ("input_label", offset_of!(VtWfstArc, input_label), 8),
            ("output_label", offset_of!(VtWfstArc, output_label), 8),
            ("target_state", offset_of!(VtWfstArc, target_state), 8),
            ("weight", offset_of!(VtWfstArc, weight), 8),
            ("has_input", offset_of!(VtWfstArc, has_input), 1),
            ("has_output", offset_of!(VtWfstArc, has_output), 1),
            ("reserved", offset_of!(VtWfstArc, reserved), 6),
        ],
    );
    assert_packed_in_order(
        "VtWfstVTable",
        size_of::<VtWfstVTable>(),
        align_of::<VtWfstVTable>(),
        &[
            ("struct_size", offset_of!(VtWfstVTable, struct_size), WORD),
            (
                "interface_version",
                offset_of!(VtWfstVTable, interface_version),
                4,
            ),
            ("unit_domain", offset_of!(VtWfstVTable, unit_domain), 4),
            ("weight_domain", offset_of!(VtWfstVTable, weight_domain), 4),
            ("reserved", offset_of!(VtWfstVTable, reserved), 4),
            ("flags", offset_of!(VtWfstVTable, flags), 8),
            ("snapshot", offset_of!(VtWfstVTable, snapshot), fnp),
            ("start", offset_of!(VtWfstVTable, start), fnp),
            ("num_states", offset_of!(VtWfstVTable, num_states), fnp),
            ("state_info", offset_of!(VtWfstVTable, state_info), fnp),
            ("state_arcs", offset_of!(VtWfstVTable, state_arcs), fnp),
        ],
    );
}

// ── exact pins: required 64-bit tier ────────────────────────────────────────

#[cfg(target_pointer_width = "64")]
mod exact_64 {
    use super::*;

    #[test]
    fn vt_resource_layout() {
        assert_eq!(size_of::<VtResource>(), 16);
        assert_eq!(align_of::<VtResource>(), 8);
        assert_eq!(offset_of!(VtResource, context), 0);
        assert_eq!(offset_of!(VtResource, vtable), 8);
    }

    #[test]
    fn vt_resource_vtable_layout() {
        assert_eq!(size_of::<VtResourceVTable>(), 40);
        assert_eq!(align_of::<VtResourceVTable>(), 8);
        assert_eq!(offset_of!(VtResourceVTable, struct_size), 0);
        assert_eq!(offset_of!(VtResourceVTable, abi_version), 8);
        assert_eq!(offset_of!(VtResourceVTable, reserved), 12);
        assert_eq!(offset_of!(VtResourceVTable, retain), 16);
        assert_eq!(offset_of!(VtResourceVTable, release), 24);
        assert_eq!(offset_of!(VtResourceVTable, query_interface), 32);
    }

    #[test]
    fn vt_dictionary_vtable_layout() {
        assert_eq!(size_of::<VtDictionaryVTable>(), 88);
        assert_eq!(align_of::<VtDictionaryVTable>(), 8);
        assert_eq!(offset_of!(VtDictionaryVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryVTable, interface_version), 8);
        assert_eq!(offset_of!(VtDictionaryVTable, unit_domain), 12);
        assert_eq!(offset_of!(VtDictionaryVTable, value_domain), 16);
        assert_eq!(offset_of!(VtDictionaryVTable, flags), 24);
        assert_eq!(offset_of!(VtDictionaryVTable, snapshot), 32);
        assert_eq!(offset_of!(VtDictionaryVTable, root), 40);
        assert_eq!(offset_of!(VtDictionaryVTable, len), 48);
        assert_eq!(offset_of!(VtDictionaryVTable, node_is_final), 56);
        assert_eq!(offset_of!(VtDictionaryVTable, node_value_u64), 64);
        assert_eq!(offset_of!(VtDictionaryVTable, node_transition), 72);
        assert_eq!(offset_of!(VtDictionaryVTable, node_edges), 80);
    }

    #[test]
    fn vt_dictionary_visit_vtable_layout() {
        assert_eq!(size_of::<VtDictionaryVisitVTable>(), 24);
        assert_eq!(align_of::<VtDictionaryVisitVTable>(), 8);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, interface_version), 8);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, reserved), 12);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, node_visit), 16);
    }

    #[test]
    fn vt_dictionary_graph_vtable_layout() {
        assert_eq!(size_of::<VtDictionaryGraphVTable>(), 32);
        assert_eq!(align_of::<VtDictionaryGraphVTable>(), 8);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, interface_version), 8);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, reserved), 12);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, graph), 16);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, node_value_u64), 24);
    }

    #[test]
    fn vt_dictionary_graph_view_layout() {
        assert_eq!(size_of::<VtDictionaryGraphView>(), 48);
        assert_eq!(align_of::<VtDictionaryGraphView>(), 8);
        assert_eq!(offset_of!(VtDictionaryGraphView, nodes), 0);
        assert_eq!(offset_of!(VtDictionaryGraphView, node_count), 8);
        assert_eq!(offset_of!(VtDictionaryGraphView, edges), 16);
        assert_eq!(offset_of!(VtDictionaryGraphView, edge_count), 24);
        assert_eq!(offset_of!(VtDictionaryGraphView, root), 32);
        assert_eq!(offset_of!(VtDictionaryGraphView, reserved), 40);
    }

    #[test]
    fn vt_snapshot_identity_vtable_layout() {
        assert_eq!(size_of::<VtSnapshotIdentityVTable>(), 24);
        assert_eq!(align_of::<VtSnapshotIdentityVTable>(), 8);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, struct_size), 0);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, interface_version), 8);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, reserved), 12);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, identity), 16);
    }

    #[test]
    fn vt_dictionary_entries_layouts() {
        assert_eq!(size_of::<VtDictionaryEntry>(), 40);
        assert_eq!(align_of::<VtDictionaryEntry>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntry, unit_offset), 0);
        assert_eq!(offset_of!(VtDictionaryEntry, unit_len), 8);
        assert_eq!(offset_of!(VtDictionaryEntry, value_offset), 16);
        assert_eq!(offset_of!(VtDictionaryEntry, value_len), 24);
        assert_eq!(offset_of!(VtDictionaryEntry, reserved), 32);

        assert_eq!(size_of::<VtDictionaryEntryBatchLimits>(), 32);
        assert_eq!(align_of::<VtDictionaryEntryBatchLimits>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, max_entries), 0);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, max_units), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, max_values), 16);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, reserved), 24);

        assert_eq!(size_of::<VtDictionaryEntryBatchView>(), 64);
        assert_eq!(align_of::<VtDictionaryEntryBatchView>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, entries), 0);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, entry_count), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, units), 16);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, unit_count), 24);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, values), 32);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, value_count), 40);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, generation), 48);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, reserved), 56);

        assert_eq!(size_of::<VtDictionaryEntriesInfo>(), 64);
        assert_eq!(align_of::<VtDictionaryEntriesInfo>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, unit_domain), 0);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, value_domain), 4);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, order), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, reserved0), 12);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, flags), 16);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, exact_len), 24);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, identity), 32);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, reserved), 48);

        assert_eq!(size_of::<VtDictionaryEntriesCursor>(), 16);
        assert_eq!(align_of::<VtDictionaryEntriesCursor>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesCursor, context), 0);
        assert_eq!(offset_of!(VtDictionaryEntriesCursor, vtable), 8);

        assert_eq!(size_of::<VtDictionaryEntriesVTable>(), 64);
        assert_eq!(align_of::<VtDictionaryEntriesVTable>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, interface_version), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, reserved), 12);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, open), 16);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, next_batch), 24);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, release_batch), 32);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, reduce), 40);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, cancel), 48);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, close), 56);
    }

    #[test]
    fn vt_wfst_vtable_layout() {
        assert_eq!(size_of::<VtWfstVTable>(), 72);
        assert_eq!(align_of::<VtWfstVTable>(), 8);
        assert_eq!(offset_of!(VtWfstVTable, struct_size), 0);
        assert_eq!(offset_of!(VtWfstVTable, interface_version), 8);
        assert_eq!(offset_of!(VtWfstVTable, unit_domain), 12);
        assert_eq!(offset_of!(VtWfstVTable, weight_domain), 16);
        assert_eq!(offset_of!(VtWfstVTable, reserved), 20);
        assert_eq!(offset_of!(VtWfstVTable, flags), 24);
        assert_eq!(offset_of!(VtWfstVTable, snapshot), 32);
        assert_eq!(offset_of!(VtWfstVTable, start), 40);
        assert_eq!(offset_of!(VtWfstVTable, num_states), 48);
        assert_eq!(offset_of!(VtWfstVTable, state_info), 56);
        assert_eq!(offset_of!(VtWfstVTable, state_arcs), 64);
    }
}

// ── exact pins: 32-bit ARM EABI tier (CI linux-armv7-experimental) ──────────

#[cfg(all(target_pointer_width = "32", target_arch = "arm"))]
mod exact_32_arm {
    use super::*;

    #[test]
    fn vt_resource_layout() {
        assert_eq!(size_of::<VtResource>(), 8);
        assert_eq!(align_of::<VtResource>(), 4);
        assert_eq!(offset_of!(VtResource, context), 0);
        assert_eq!(offset_of!(VtResource, vtable), 4);
    }

    #[test]
    fn vt_resource_vtable_layout() {
        assert_eq!(size_of::<VtResourceVTable>(), 24);
        assert_eq!(align_of::<VtResourceVTable>(), 4);
        assert_eq!(offset_of!(VtResourceVTable, struct_size), 0);
        assert_eq!(offset_of!(VtResourceVTable, abi_version), 4);
        assert_eq!(offset_of!(VtResourceVTable, reserved), 8);
        assert_eq!(offset_of!(VtResourceVTable, retain), 12);
        assert_eq!(offset_of!(VtResourceVTable, release), 16);
        assert_eq!(offset_of!(VtResourceVTable, query_interface), 20);
    }

    #[test]
    fn vt_dictionary_vtable_layout() {
        assert_eq!(size_of::<VtDictionaryVTable>(), 56);
        assert_eq!(align_of::<VtDictionaryVTable>(), 8);
        assert_eq!(offset_of!(VtDictionaryVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryVTable, interface_version), 4);
        assert_eq!(offset_of!(VtDictionaryVTable, unit_domain), 8);
        assert_eq!(offset_of!(VtDictionaryVTable, value_domain), 12);
        assert_eq!(offset_of!(VtDictionaryVTable, flags), 16);
        assert_eq!(offset_of!(VtDictionaryVTable, snapshot), 24);
        assert_eq!(offset_of!(VtDictionaryVTable, root), 28);
        assert_eq!(offset_of!(VtDictionaryVTable, len), 32);
        assert_eq!(offset_of!(VtDictionaryVTable, node_is_final), 36);
        assert_eq!(offset_of!(VtDictionaryVTable, node_value_u64), 40);
        assert_eq!(offset_of!(VtDictionaryVTable, node_transition), 44);
        assert_eq!(offset_of!(VtDictionaryVTable, node_edges), 48);
    }

    #[test]
    fn vt_dictionary_visit_vtable_layout() {
        assert_eq!(size_of::<VtDictionaryVisitVTable>(), 16);
        assert_eq!(align_of::<VtDictionaryVisitVTable>(), 4);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, interface_version), 4);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, reserved), 8);
        assert_eq!(offset_of!(VtDictionaryVisitVTable, node_visit), 12);
    }

    #[test]
    fn vt_dictionary_graph_vtable_layout() {
        assert_eq!(size_of::<VtDictionaryGraphVTable>(), 20);
        assert_eq!(align_of::<VtDictionaryGraphVTable>(), 4);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, interface_version), 4);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, reserved), 8);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, graph), 12);
        assert_eq!(offset_of!(VtDictionaryGraphVTable, node_value_u64), 16);
    }

    #[test]
    fn vt_dictionary_graph_view_layout() {
        assert_eq!(size_of::<VtDictionaryGraphView>(), 32);
        assert_eq!(align_of::<VtDictionaryGraphView>(), 8);
        assert_eq!(offset_of!(VtDictionaryGraphView, nodes), 0);
        assert_eq!(offset_of!(VtDictionaryGraphView, node_count), 4);
        assert_eq!(offset_of!(VtDictionaryGraphView, edges), 8);
        assert_eq!(offset_of!(VtDictionaryGraphView, edge_count), 12);
        assert_eq!(offset_of!(VtDictionaryGraphView, root), 16);
        assert_eq!(offset_of!(VtDictionaryGraphView, reserved), 24);
    }

    #[test]
    fn vt_snapshot_identity_vtable_layout() {
        assert_eq!(size_of::<VtSnapshotIdentityVTable>(), 16);
        assert_eq!(align_of::<VtSnapshotIdentityVTable>(), 4);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, struct_size), 0);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, interface_version), 4);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, reserved), 8);
        assert_eq!(offset_of!(VtSnapshotIdentityVTable, identity), 12);
    }

    #[test]
    fn vt_dictionary_entries_layouts() {
        assert_eq!(size_of::<VtDictionaryEntry>(), 24);
        assert_eq!(align_of::<VtDictionaryEntry>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntry, unit_offset), 0);
        assert_eq!(offset_of!(VtDictionaryEntry, unit_len), 4);
        assert_eq!(offset_of!(VtDictionaryEntry, value_offset), 8);
        assert_eq!(offset_of!(VtDictionaryEntry, value_len), 12);
        assert_eq!(offset_of!(VtDictionaryEntry, reserved), 16);

        assert_eq!(size_of::<VtDictionaryEntryBatchLimits>(), 24);
        assert_eq!(align_of::<VtDictionaryEntryBatchLimits>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, max_entries), 0);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, max_units), 4);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, max_values), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchLimits, reserved), 16);

        assert_eq!(size_of::<VtDictionaryEntryBatchView>(), 40);
        assert_eq!(align_of::<VtDictionaryEntryBatchView>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, entries), 0);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, entry_count), 4);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, units), 8);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, unit_count), 12);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, values), 16);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, value_count), 20);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, generation), 24);
        assert_eq!(offset_of!(VtDictionaryEntryBatchView, reserved), 32);

        assert_eq!(size_of::<VtDictionaryEntriesInfo>(), 64);
        assert_eq!(align_of::<VtDictionaryEntriesInfo>(), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, unit_domain), 0);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, value_domain), 4);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, order), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, reserved0), 12);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, flags), 16);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, exact_len), 24);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, identity), 32);
        assert_eq!(offset_of!(VtDictionaryEntriesInfo, reserved), 48);

        assert_eq!(size_of::<VtDictionaryEntriesCursor>(), 8);
        assert_eq!(align_of::<VtDictionaryEntriesCursor>(), 4);
        assert_eq!(offset_of!(VtDictionaryEntriesCursor, context), 0);
        assert_eq!(offset_of!(VtDictionaryEntriesCursor, vtable), 4);

        assert_eq!(size_of::<VtDictionaryEntriesVTable>(), 36);
        assert_eq!(align_of::<VtDictionaryEntriesVTable>(), 4);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, struct_size), 0);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, interface_version), 4);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, reserved), 8);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, open), 12);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, next_batch), 16);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, release_batch), 20);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, reduce), 24);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, cancel), 28);
        assert_eq!(offset_of!(VtDictionaryEntriesVTable, close), 32);
    }

    #[test]
    fn vt_wfst_vtable_layout() {
        assert_eq!(size_of::<VtWfstVTable>(), 56);
        assert_eq!(align_of::<VtWfstVTable>(), 8);
        assert_eq!(offset_of!(VtWfstVTable, struct_size), 0);
        assert_eq!(offset_of!(VtWfstVTable, interface_version), 4);
        assert_eq!(offset_of!(VtWfstVTable, unit_domain), 8);
        assert_eq!(offset_of!(VtWfstVTable, weight_domain), 12);
        assert_eq!(offset_of!(VtWfstVTable, reserved), 16);
        assert_eq!(offset_of!(VtWfstVTable, flags), 24);
        assert_eq!(offset_of!(VtWfstVTable, snapshot), 32);
        assert_eq!(offset_of!(VtWfstVTable, start), 36);
        assert_eq!(offset_of!(VtWfstVTable, num_states), 40);
        assert_eq!(offset_of!(VtWfstVTable, state_info), 44);
        assert_eq!(offset_of!(VtWfstVTable, state_arcs), 48);
    }
}

// ── pointer-width-independent exact pins ────────────────────────────────────

#[test]
fn pod_payload_layouts_are_word_size_independent() {
    // Every field of these three types is u64/f64/u8, so the layout is the
    // same on every supported target.
    assert_eq!(size_of::<VtInterfaceId>(), 16);
    assert_eq!(align_of::<VtInterfaceId>(), 1);
    assert_eq!(offset_of!(VtInterfaceId, bytes), 0);

    assert_eq!(size_of::<VtOptionalU64>(), 16);
    assert_eq!(offset_of!(VtOptionalU64, value), 0);
    assert_eq!(offset_of!(VtOptionalU64, has_value), 8);
    assert_eq!(offset_of!(VtOptionalU64, reserved), 9);

    assert_eq!(size_of::<VtDictionaryEdge>(), 16);
    assert_eq!(offset_of!(VtDictionaryEdge, label), 0);
    assert_eq!(offset_of!(VtDictionaryEdge, node), 8);

    assert_eq!(size_of::<VtDictionaryGraphNode>(), 32);
    assert_eq!(offset_of!(VtDictionaryGraphNode, edge_start), 0);
    assert_eq!(offset_of!(VtDictionaryGraphNode, edge_len), 8);
    assert_eq!(offset_of!(VtDictionaryGraphNode, value_cursor), 16);
    assert_eq!(offset_of!(VtDictionaryGraphNode, is_final), 24);
    assert_eq!(offset_of!(VtDictionaryGraphNode, reserved), 25);

    assert_eq!(size_of::<VtDictionaryGraphEdge>(), 16);
    assert_eq!(offset_of!(VtDictionaryGraphEdge, label), 0);
    assert_eq!(offset_of!(VtDictionaryGraphEdge, target), 8);

    assert_eq!(size_of::<VtSnapshotIdentity>(), 16);
    assert_eq!(offset_of!(VtSnapshotIdentity, producer), 0);
    assert_eq!(offset_of!(VtSnapshotIdentity, revision), 8);

    assert_eq!(size_of::<VtWfstArc>(), 40);
    assert_eq!(offset_of!(VtWfstArc, input_label), 0);
    assert_eq!(offset_of!(VtWfstArc, output_label), 8);
    assert_eq!(offset_of!(VtWfstArc, target_state), 16);
    assert_eq!(offset_of!(VtWfstArc, weight), 24);
    assert_eq!(offset_of!(VtWfstArc, has_input), 32);
    assert_eq!(offset_of!(VtWfstArc, has_output), 33);
    assert_eq!(offset_of!(VtWfstArc, reserved), 34);
}

#[test]
fn vt_resource_is_exactly_two_machine_words() {
    // VT-ABI-3: the law every language binding repeats (the JVM asserts it at
    // runtime in BindingArchitectureTest.sharedResourceIsExactlyTwoWords).
    assert_eq!(size_of::<VtResource>(), 2 * WORD);
    assert_eq!(align_of::<VtResource>(), align_of::<usize>());
}

#[test]
fn vt_entries_cursor_is_exactly_two_machine_words() {
    assert_eq!(size_of::<VtDictionaryEntriesCursor>(), 2 * WORD);
    assert_eq!(align_of::<VtDictionaryEntriesCursor>(), align_of::<usize>());
}

// ── null semantics and interface identifiers ────────────────────────────────

#[test]
fn null_resource_semantics_are_either_word() {
    assert!(VtResource::NULL.is_null());
    let vtable = VtResourceVTable {
        struct_size: size_of::<VtResourceVTable>(),
        abi_version: VT_ABI_VERSION,
        reserved: 0,
        retain: None,
        release: None,
        query_interface: None,
    };
    let context_null = VtResource {
        context: core::ptr::null_mut(),
        vtable: &vtable,
    };
    assert!(
        context_null.is_null(),
        "a null context must make the resource null even with a valid vtable"
    );
    let vtable_null = VtResource {
        context: core::ptr::dangling_mut::<c_void>(),
        vtable: core::ptr::null(),
    };
    assert!(
        vtable_null.is_null(),
        "a null vtable must make the resource null even with a non-null context"
    );
}

#[test]
fn null_entries_cursor_semantics_are_either_word() {
    assert!(VtDictionaryEntriesCursor::NULL.is_null());
    let vtable = VtDictionaryEntriesVTable {
        struct_size: size_of::<VtDictionaryEntriesVTable>(),
        interface_version: VT_DICTIONARY_ENTRIES_INTERFACE_VERSION,
        reserved: 0,
        open: None,
        next_batch: None,
        release_batch: None,
        reduce: None,
        cancel: None,
        close: None,
    };
    let context_null = VtDictionaryEntriesCursor {
        context: core::ptr::null_mut(),
        vtable: &vtable,
    };
    assert!(context_null.is_null());
    let vtable_null = VtDictionaryEntriesCursor {
        context: core::ptr::dangling_mut::<c_void>(),
        vtable: core::ptr::null(),
    };
    assert!(vtable_null.is_null());
}

#[test]
fn interface_identifiers_are_exact_sixteen_byte_strings() {
    // VT-ABI-4: identifiers are compared byte-for-byte across the ABI; all
    // published identifiers are exactly 16 bytes with no NUL terminator.
    assert_eq!(&VT_DICTIONARY_INTERFACE_ID.bytes, b"vt.dictionary.v1");
    assert_eq!(&VT_DICTIONARY_VISIT_INTERFACE_ID.bytes, b"vt.dict.visit.v1");
    assert_eq!(&VT_DICTIONARY_GRAPH_INTERFACE_ID.bytes, b"vt.dict.graph.v1");
    assert_eq!(
        &VT_DICTIONARY_ENTRIES_INTERFACE_ID.bytes,
        b"vt.dict.entry.v1"
    );
    assert_eq!(
        &VT_SNAPSHOT_IDENTITY_INTERFACE_ID.bytes,
        b"vt.snapshot.id.1"
    );
    assert_eq!(&VT_WFST_INTERFACE_ID.bytes, b"vt.scalar-wfst.1");
    let ids = [
        VT_DICTIONARY_INTERFACE_ID,
        VT_DICTIONARY_VISIT_INTERFACE_ID,
        VT_DICTIONARY_GRAPH_INTERFACE_ID,
        VT_DICTIONARY_ENTRIES_INTERFACE_ID,
        VT_SNAPSHOT_IDENTITY_INTERFACE_ID,
        VT_WFST_INTERFACE_ID,
    ];
    for (index, left) in ids.iter().enumerate() {
        for right in &ids[index + 1..] {
            assert_ne!(left, right, "published interface IDs must be unique");
        }
    }
    assert_eq!(VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_ID);
}

#[test]
fn published_constants_are_pinned() {
    assert_eq!(VT_ABI_VERSION, 1);
    assert_eq!(VT_DICTIONARY_INTERFACE_VERSION, 1);
    assert_eq!(VT_DICTIONARY_VISIT_INTERFACE_VERSION, 1);
    assert_eq!(VT_DICTIONARY_GRAPH_INTERFACE_VERSION, 1);
    assert_eq!(VT_DICTIONARY_ENTRIES_INTERFACE_VERSION, 1);
    assert_eq!(VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION, 1);
    assert_eq!(VT_WFST_INTERFACE_VERSION, 1);
    assert_eq!(VT_RECOMMENDED_EDGE_BATCH, 256);
    assert_eq!(VT_RECOMMENDED_ARC_BATCH, 256);
}

// ── the load-bearing Option<extern fn> null niche ───────────────────────────

#[test]
fn optional_function_pointers_have_the_null_niche() {
    // VT-ABI-6: every vtable slot is `Option<unsafe extern "C" fn ...>`. The C
    // side fills absent operations with NULL, so Rust's guaranteed null-pointer
    // optimization for Option-of-fn is what makes the two representations one
    // ABI. If either assertion ever fails, the entire vtable ABI is broken.
    type Retain = Option<unsafe extern "C" fn(*mut c_void)>;
    assert_eq!(size_of::<Retain>(), size_of::<*const c_void>());
    // SAFETY: transmuting an Option fn-pointer to usize is defined for reading
    // the discriminant representation; None is guaranteed to be the null
    // pointer under the NPO for `Option<extern "C" fn>`.
    let none_bits: usize = unsafe { core::mem::transmute::<Retain, usize>(None) };
    assert_eq!(
        none_bits, 0,
        "None must be the all-zero (NULL) function pointer"
    );
}
