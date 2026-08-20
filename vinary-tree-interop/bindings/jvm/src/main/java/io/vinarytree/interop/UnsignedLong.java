package io.vinarytree.interop;

/** Lossless unsigned-64 value represented by its raw Java {@code long} bits. */
public record UnsignedLong(long bits) implements Comparable<UnsignedLong> {
    /** Parse an unsigned decimal value. */
    public static UnsignedLong parse(String value) {
        return new UnsignedLong(Long.parseUnsignedLong(value));
    }

    /** Return the unsigned decimal representation. */
    @Override
    public String toString() {
        return Long.toUnsignedString(bits);
    }

    /** Compare as unsigned 64-bit integers. */
    @Override
    public int compareTo(UnsignedLong other) {
        return Long.compareUnsigned(bits, other.bits);
    }
}
