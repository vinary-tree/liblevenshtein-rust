package io.vinarytree.interop;

/** Element representation used by dictionary keys. */
public enum DictionaryUnitDomain {
    /** Arbitrary bytes. */
    BYTE(1),
    /** Unicode scalar values. */
    UNICODE_SCALAR(2),
    /** Unsigned 64-bit tokens. */
    U64(3);

    final int wireValue;

    DictionaryUnitDomain(int wireValue) {
        this.wireValue = wireValue;
    }

    static DictionaryUnitDomain fromWire(int value) {
        return switch (value) {
            case 1 -> BYTE;
            case 2 -> UNICODE_SCALAR;
            case 3 -> U64;
            default -> throw new DictionaryInteropException(
                    8, "unknown dictionary unit-domain value " + Integer.toUnsignedString(value));
        };
    }
}
