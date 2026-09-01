package io.vinarytree.liblevenshtein;

/** Lazy result ordering.
 * <p>Generated from bindings/api.json; do not edit numeric values manually.
 */
public enum QueryOrder {
    /** Provider traversal order with bounded buffering. */
    TRAVERSAL(0),
    /** Distance then term, buffering at most one distance layer. */
    DISTANCE_THEN_TERM(1);

    private final int nativeValue;

    QueryOrder(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    int nativeValue() {
        return nativeValue;
    }
}
