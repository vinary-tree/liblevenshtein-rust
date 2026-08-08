package io.vinarytree.liblevenshtein;

/** Built-in phonetic rewrite-rule set. */
public enum PhoneticRuleSetKind {
    /** English orthography normalization. */
    ENGLISH_ORTHOGRAPHY(0),
    /** English phonetic transformation. */
    ENGLISH_PHONETIC(1);

    private final int nativeValue;
    PhoneticRuleSetKind(int nativeValue) { this.nativeValue = nativeValue; }
    int nativeValue() { return nativeValue; }
}
