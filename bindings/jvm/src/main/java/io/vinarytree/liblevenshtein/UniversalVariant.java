package io.vinarytree.liblevenshtein;

/** Universal edit-automaton variant.
 * <p>Generated from bindings/api.json; do not edit numeric values manually.
 */
public enum UniversalVariant {
    /** Standard insert/delete/substitute universal automaton. */
    STANDARD(0),
    /** Universal automaton with adjacent transposition. */
    TRANSPOSITION(1),
    /** Universal automaton with merge-and-split edits. */
    MERGE_AND_SPLIT(2);

    private final int nativeValue;

    UniversalVariant(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    int nativeValue() {
        return nativeValue;
    }
}
