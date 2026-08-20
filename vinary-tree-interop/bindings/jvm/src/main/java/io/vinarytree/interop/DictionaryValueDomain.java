package io.vinarytree.interop;

/** Value representation attached to each present dictionary key. */
public enum DictionaryValueDomain {
    /** Set semantics: every present key has no mapped value. */
    UNIT(0),
    /** Map semantics: a present key may optionally carry one unsigned-64 value. */
    OPTIONAL_U64(1);

    final int wireValue;

    DictionaryValueDomain(int wireValue) {
        this.wireValue = wireValue;
    }

    static DictionaryValueDomain fromWire(int value) {
        return switch (value) {
            case 0 -> UNIT;
            case 1 -> OPTIONAL_U64;
            default -> throw new DictionaryInteropException(
                    8, "unsupported dictionary value-domain value " + Integer.toUnsignedString(value));
        };
    }
}
