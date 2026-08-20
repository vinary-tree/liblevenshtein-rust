package io.vinarytree.interop;

import java.util.Arrays;
import java.util.Objects;

/** Immutable, value-equal dictionary key preserving its native unit domain. */
public final class DictionaryKey implements Comparable<DictionaryKey> {
    private final DictionaryUnitDomain domain;
    private final byte[] bytes;
    private final int[] scalars;
    private final long[] u64;

    private DictionaryKey(
            DictionaryUnitDomain domain, byte[] bytes, int[] scalars, long[] u64) {
        this.domain = domain;
        this.bytes = bytes;
        this.scalars = scalars;
        this.u64 = u64;
    }

    /** Copy an arbitrary byte key. */
    public static DictionaryKey bytes(byte[] value) {
        return new DictionaryKey(
                DictionaryUnitDomain.BYTE,
                Objects.requireNonNull(value, "value").clone(),
                null,
                null);
    }

    /** Copy and validate a Unicode-scalar key. */
    public static DictionaryKey unicodeScalars(int[] value) {
        Objects.requireNonNull(value, "value");
        int[] copy = value.clone();
        for (int scalar : copy) {
            if (!Character.isValidCodePoint(scalar)
                    || scalar >= Character.MIN_SURROGATE
                            && scalar <= Character.MAX_SURROGATE) {
                throw new IllegalArgumentException("key contains a non-scalar code point");
            }
        }
        return new DictionaryKey(DictionaryUnitDomain.UNICODE_SCALAR, null, copy, null);
    }

    /** Convert a Java string to its exact Unicode-scalar sequence. */
    public static DictionaryKey unicode(String value) {
        Objects.requireNonNull(value, "value");
        return unicodeScalars(value.codePoints().toArray());
    }

    /** Copy unsigned-64 token bits without signed conversion. */
    public static DictionaryKey u64(long[] value) {
        return new DictionaryKey(
                DictionaryUnitDomain.U64,
                null,
                null,
                Objects.requireNonNull(value, "value").clone());
    }

    /** Native unit domain. */
    public DictionaryUnitDomain domain() {
        return domain;
    }

    /** Number of native units without copying key storage. */
    public int unitCount() {
        return switch (domain) {
            case BYTE -> bytes.length;
            case UNICODE_SCALAR -> scalars.length;
            case U64 -> u64.length;
        };
    }

    /** Return a defensive copy of a byte key. */
    public byte[] bytes() {
        require(DictionaryUnitDomain.BYTE);
        return bytes.clone();
    }

    /** Return a defensive copy of Unicode scalar values. */
    public int[] unicodeScalars() {
        require(DictionaryUnitDomain.UNICODE_SCALAR);
        return scalars.clone();
    }

    /** Convert a Unicode-scalar key to a Java string. */
    public String unicode() {
        require(DictionaryUnitDomain.UNICODE_SCALAR);
        StringBuilder result = new StringBuilder(scalars.length);
        for (int scalar : scalars) result.appendCodePoint(scalar);
        return result.toString();
    }

    /** Return a defensive copy of unsigned-64 token bits. */
    public long[] u64() {
        require(DictionaryUnitDomain.U64);
        return u64.clone();
    }

    private void require(DictionaryUnitDomain expected) {
        if (domain != expected) {
            throw new IllegalStateException("key domain is " + domain + ", not " + expected);
        }
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (!(other instanceof DictionaryKey key) || domain != key.domain) return false;
        return switch (domain) {
            case BYTE -> Arrays.equals(bytes, key.bytes);
            case UNICODE_SCALAR -> Arrays.equals(scalars, key.scalars);
            case U64 -> Arrays.equals(u64, key.u64);
        };
    }

    @Override
    public int hashCode() {
        int content = switch (domain) {
            case BYTE -> Arrays.hashCode(bytes);
            case UNICODE_SCALAR -> Arrays.hashCode(scalars);
            case U64 -> Arrays.hashCode(u64);
        };
        return 31 * domain.hashCode() + content;
    }

    /** Compare by domain and then unsigned native lexicographic order. */
    @Override
    public int compareTo(DictionaryKey other) {
        int domainOrder = Integer.compare(domain.wireValue, other.domain.wireValue);
        if (domainOrder != 0) return domainOrder;
        return switch (domain) {
            case BYTE -> compareBytes(bytes, other.bytes);
            case UNICODE_SCALAR -> compareInts(scalars, other.scalars);
            case U64 -> compareLongs(u64, other.u64);
        };
    }

    private static int compareBytes(byte[] left, byte[] right) {
        int common = Math.min(left.length, right.length);
        for (int i = 0; i < common; i++) {
            int compared = Integer.compare(Byte.toUnsignedInt(left[i]), Byte.toUnsignedInt(right[i]));
            if (compared != 0) return compared;
        }
        return Integer.compare(left.length, right.length);
    }

    private static int compareInts(int[] left, int[] right) {
        int common = Math.min(left.length, right.length);
        for (int i = 0; i < common; i++) {
            int compared = Integer.compareUnsigned(left[i], right[i]);
            if (compared != 0) return compared;
        }
        return Integer.compare(left.length, right.length);
    }

    private static int compareLongs(long[] left, long[] right) {
        int common = Math.min(left.length, right.length);
        for (int i = 0; i < common; i++) {
            int compared = Long.compareUnsigned(left[i], right[i]);
            if (compared != 0) return compared;
        }
        return Integer.compare(left.length, right.length);
    }

    @Override
    public String toString() {
        return switch (domain) {
            case BYTE -> Arrays.toString(bytes);
            case UNICODE_SCALAR -> unicode();
            case U64 -> Arrays.stream(u64)
                    .mapToObj(Long::toUnsignedString)
                    .toList()
                    .toString();
        };
    }
}
