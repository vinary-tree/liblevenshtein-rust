package io.vinarytree.interop;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.MemoryLayout;

/** FFM layouts shared verbatim by all vinary-tree JVM artifacts. */
public final class InteropLayouts {
    /** Two-word {@code VtResource}: provider context followed by base vtable. */
    public static final MemoryLayout RESOURCE = MemoryLayout.structLayout(
            ADDRESS.withName("context"), ADDRESS.withName("vtable"));
    /** Base retained-resource vtable. */
    public static final MemoryLayout RESOURCE_VTABLE = MemoryLayout.structLayout(
            JAVA_LONG.withName("struct_size"),
            JAVA_INT.withName("abi_version"),
            JAVA_INT.withName("reserved"),
            ADDRESS.withName("retain"),
            ADDRESS.withName("release"),
            ADDRESS.withName("query_interface"));
    /** Version-1 dictionary traversal vtable on supported 64-bit JVMs. */
    public static final MemoryLayout DICTIONARY_VTABLE = MemoryLayout.structLayout(
            JAVA_LONG.withName("struct_size"),
            JAVA_INT.withName("interface_version"),
            JAVA_INT.withName("unit_domain"),
            JAVA_INT.withName("value_domain"),
            MemoryLayout.paddingLayout(4),
            JAVA_LONG.withName("flags"),
            ADDRESS.withName("snapshot"),
            ADDRESS.withName("root"),
            ADDRESS.withName("len"),
            ADDRESS.withName("node_is_final"),
            ADDRESS.withName("node_value_u64"),
            ADDRESS.withName("node_transition"),
            ADDRESS.withName("node_edges"));
    /** Optional unsigned 64-bit value layout. */
    public static final MemoryLayout OPTIONAL_U64 = MemoryLayout.structLayout(
            JAVA_LONG.withName("value"),
            JAVA_BYTE.withName("has_value"),
            MemoryLayout.sequenceLayout(7, JAVA_BYTE).withName("reserved"));
    /** One contiguous dictionary edge. */
    public static final MemoryLayout DICTIONARY_EDGE = MemoryLayout.structLayout(
            JAVA_LONG.withName("label"), JAVA_LONG.withName("node"));
    /** One dictionary-entry descriptor into the parallel arenas. */
    public static final MemoryLayout DICTIONARY_ENTRY = MemoryLayout.structLayout(
            JAVA_LONG.withName("unit_offset"),
            JAVA_LONG.withName("unit_len"),
            JAVA_LONG.withName("value_offset"),
            JAVA_LONG.withName("value_len"),
            JAVA_LONG.withName("reserved"));
    /** Hard bounds for one entries-v1 batch. */
    public static final MemoryLayout DICTIONARY_ENTRY_LIMITS = MemoryLayout.structLayout(
            JAVA_LONG.withName("max_entries"),
            JAVA_LONG.withName("max_units"),
            JAVA_LONG.withName("max_values"),
            JAVA_LONG.withName("reserved"));
    /** Cursor-owned entries-v1 batch view. */
    public static final MemoryLayout DICTIONARY_ENTRY_BATCH = MemoryLayout.structLayout(
            ADDRESS.withName("entries"),
            JAVA_LONG.withName("entry_count"),
            ADDRESS.withName("units"),
            JAVA_LONG.withName("unit_count"),
            ADDRESS.withName("values"),
            JAVA_LONG.withName("value_count"),
            JAVA_LONG.withName("generation"),
            JAVA_LONG.withName("reserved"));
    /** Immutable metadata captured when an entries-v1 cursor opens. */
    public static final MemoryLayout DICTIONARY_ENTRIES_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("unit_domain"),
            JAVA_INT.withName("value_domain"),
            JAVA_INT.withName("order"),
            JAVA_INT.withName("reserved0"),
            JAVA_LONG.withName("flags"),
            JAVA_LONG.withName("exact_len"),
            JAVA_LONG.withName("identity_producer"),
            JAVA_LONG.withName("identity_revision"),
            MemoryLayout.sequenceLayout(2, JAVA_LONG).withName("reserved"));
    /** Move-only two-word dictionary entries cursor. */
    public static final MemoryLayout DICTIONARY_ENTRIES_CURSOR = MemoryLayout.structLayout(
            ADDRESS.withName("context"), ADDRESS.withName("vtable"));
    /** Complete version-1 dictionary entries vtable on supported 64-bit JVMs. */
    public static final MemoryLayout DICTIONARY_ENTRIES_VTABLE = MemoryLayout.structLayout(
            JAVA_LONG.withName("struct_size"),
            JAVA_INT.withName("interface_version"),
            JAVA_INT.withName("reserved"),
            ADDRESS.withName("open"),
            ADDRESS.withName("next_batch"),
            ADDRESS.withName("release_batch"),
            ADDRESS.withName("reduce"),
            ADDRESS.withName("cancel"),
            ADDRESS.withName("close"));
    /** Current shared base ABI version. */
    public static final int ABI_VERSION = 1;
    /** Current dictionary interface version. */
    public static final int DICTIONARY_INTERFACE_VERSION = 1;
    /** Current optional dictionary entries interface version. */
    public static final int DICTIONARY_ENTRIES_INTERFACE_VERSION = 1;

    private InteropLayouts() {}
}
