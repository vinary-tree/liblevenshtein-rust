package bench;

/**
 * Normative checksum primitives (PROTOCOL.md §8).
 *
 * All arithmetic is wrapping 64-bit; Java's signed {@code long} produces
 * bit-identical results to unsigned implementations, and only equality is
 * ever compared. Hex serialization uses the unsigned form.
 */
public final class Fnv {
    public static final long OFFSET = 0xcbf29ce484222325L;
    public static final long PRIME = 0x00000100000001b3L;

    private Fnv() {}

    public static long update(long hash, byte value) {
        return (hash ^ (value & 0xffL)) * PRIME;
    }

    public static long fnv1a64(byte[] data) {
        long hash = OFFSET;
        for (byte b : data) {
            hash = update(hash, b);
        }
        return hash;
    }

    /** entry(term, distance) over utf8(term) ‖ 0x00 ‖ LE64(distance). */
    public static long entry(byte[] termUtf8, long distance) {
        long hash = OFFSET;
        for (byte b : termUtf8) {
            hash = update(hash, b);
        }
        hash = update(hash, (byte) 0x00);
        for (int i = 0; i < 8; i++) {
            hash = update(hash, (byte) ((distance >>> (8 * i)) & 0xff));
        }
        return hash;
    }

    /**
     * ASCII fast path used inside gate passes: for this workload every term
     * is ASCII, so hashing the chars as bytes equals hashing UTF-8 bytes.
     * Falls back to real UTF-8 encoding the moment a non-ASCII char appears.
     */
    public static long entryAscii(String term, long distance) {
        int length = term.length();
        long hash = OFFSET;
        for (int i = 0; i < length; i++) {
            char c = term.charAt(i);
            if (c > 0x7f) {
                return entry(term.getBytes(java.nio.charset.StandardCharsets.UTF_8), distance);
            }
            hash = update(hash, (byte) c);
        }
        hash = update(hash, (byte) 0x00);
        for (int i = 0; i < 8; i++) {
            hash = update(hash, (byte) ((distance >>> (8 * i)) & 0xff));
        }
        return hash;
    }

    /** Startup self-test (PROTOCOL.md §2); throws on any mismatch. */
    public static void selfTest() {
        expect(fnv1a64(new byte[0]), 0xcbf29ce484222325L, "fnv1a64(\"\")");
        expect(fnv1a64(new byte[] {'a'}), 0xaf63dc4c8601ec8cL, "fnv1a64(\"a\")");
        expect(entryAscii("cat", 1), 0x9697fa3e50464bc4L, "entry(cat,1)");
        expect(entryAscii("cat", 0), 0xb592c1475b3595e5L, "entry(cat,0)");
        expect(entryAscii("cot", 1), 0xb8acc5d3816bcdeaL, "entry(cot,1)");
        expect(entryAscii("cat", 0) + entryAscii("cot", 1), 0x6e3f871adca163cfL, "checksum{2}");
    }

    private static void expect(long actual, long expected, String label) {
        if (actual != expected) {
            throw new IllegalStateException(
                "checksum self-test failed for " + label + ": got "
                    + Long.toHexString(actual) + ", want " + Long.toHexString(expected));
        }
    }
}
