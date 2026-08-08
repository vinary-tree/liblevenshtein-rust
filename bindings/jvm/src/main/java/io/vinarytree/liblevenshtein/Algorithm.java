package io.vinarytree.liblevenshtein;

/** Edit-distance algorithm. */
public enum Algorithm {
    /** Standard insert/delete/substitute distance. */
    STANDARD(0),
    /** Optimal string alignment with adjacent transposition. */
    TRANSPOSITION(1),
    /** Merge-and-split edit distance. */
    MERGE_AND_SPLIT(2),
    /** Unrestricted Damerau-Levenshtein distance. */
    DAMERAU_LEVENSHTEIN(3);

    private final int nativeValue;
    Algorithm(int nativeValue) { this.nativeValue = nativeValue; }
    int nativeValue() { return nativeValue; }
}
