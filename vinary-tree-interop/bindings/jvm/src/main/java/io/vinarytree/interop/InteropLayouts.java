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
    /** Current shared base ABI version. */
    public static final int ABI_VERSION = 1;
    /** Current dictionary interface version. */
    public static final int DICTIONARY_INTERFACE_VERSION = 1;

    private InteropLayouts() {}
}
