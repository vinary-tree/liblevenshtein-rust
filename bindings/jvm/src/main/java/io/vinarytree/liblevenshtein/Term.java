package io.vinarytree.liblevenshtein;

/** A matched key in its provider's native unit domain. */
public sealed interface Term permits Term.Utf8, Term.Bytes, Term.U64 {
    /** Valid Unicode text. */
    record Utf8(String value) implements Term {}

    /** Raw byte key. */
    record Bytes(byte[] value) implements Term {
        public Bytes { value = value.clone(); }
        @Override public byte[] value() { return value.clone(); }
    }

    /** Raw u64 tokens stored in Java {@code long} bit patterns. */
    record U64(long[] value) implements Term {
        public U64 { value = value.clone(); }
        @Override public long[] value() { return value.clone(); }
    }
}
