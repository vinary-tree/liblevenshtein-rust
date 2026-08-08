package io.vinarytree.interop;

import java.util.List;
import java.util.OptionalLong;

/** Immutable Unicode-scalar node graph captured in O(1) by a host provider. */
public interface UnicodeDictionarySnapshot {
    /** Root node identifier. */
    long root();

    /** Number of final terms, or empty when computing it is not O(1). */
    OptionalLong size();

    /** Whether {@code node} terminates a term. */
    boolean isFinal(long node);

    /** Optional unsigned-64 value stored at {@code node}. */
    OptionalLong value(long node);

    /** Stable outgoing edges for {@code node}. */
    List<Edge> edges(long node);

    /** One Unicode-scalar edge and its child node. */
    record Edge(int scalar, long node) {
        /** Validate that the edge label is a Unicode scalar value. */
        public Edge {
            if (!Character.isValidCodePoint(scalar)
                    || scalar >= Character.MIN_SURROGATE
                    && scalar <= Character.MAX_SURROGATE) {
                throw new IllegalArgumentException("edge label is not a Unicode scalar");
            }
        }
    }
}
