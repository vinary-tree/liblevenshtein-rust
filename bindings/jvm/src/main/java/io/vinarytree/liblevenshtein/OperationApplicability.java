package io.vinarytree.liblevenshtein;

/** Runtime generalized-operation applicability predicate.
 * <p>Generated from bindings/api.json; do not edit numeric values manually.
 */
public enum OperationApplicability {
    /** Apply without inspecting consumed units. */
    ANY(0),
    /** Apply only when the consumed source and target slices are equal. */
    EQUAL(1),
    /** Apply only to an adjacent two-unit transposition. */
    ADJACENT_TRANSPOSE(2),
    /** Apply only to a configured directional source/target pair. */
    LISTED(3);

    private final int nativeValue;

    OperationApplicability(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    int nativeValue() {
        return nativeValue;
    }
}
