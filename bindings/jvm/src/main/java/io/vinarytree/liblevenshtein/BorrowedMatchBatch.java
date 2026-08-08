package io.vinarytree.liblevenshtein;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.OptionalLong;

/** A zero-copy batch view valid only during a {@link BorrowedBatchConsumer}. */
public final class BorrowedMatchBatch {
    private static final long DATA = Native.MATCH.byteOffset(groupElement("term_data"));
    private static final long TERM_LEN = Native.MATCH.byteOffset(groupElement("term_len"));
    private static final long BYTE_LEN = Native.MATCH.byteOffset(groupElement("byte_len"));
    private static final long DISTANCE = Native.MATCH.byteOffset(groupElement("distance"));
    private static final long ID = Native.MATCH.byteOffset(groupElement("id"));
    private static final long DOMAIN = Native.MATCH.byteOffset(groupElement("unit_domain"));
    private static final long HAS_ID = Native.MATCH.byteOffset(groupElement("has_id"));

    private final MemorySegment descriptors;
    private final int size;
    private boolean active = true;

    BorrowedMatchBatch(MemorySegment descriptors, int size) {
        this.descriptors = descriptors;
        this.size = size;
    }

    /** Number of match descriptors. */
    public int size() { ensureActive(); return size; }

    /** Borrow one descriptor without decoding or copying its term. */
    public BorrowedMatch get(int index) {
        ensureActive();
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException(index);
        return new BorrowedMatch(descriptors.asSlice(index * Native.MATCH.byteSize(), Native.MATCH.byteSize()));
    }

    void invalidate() { active = false; }

    private void ensureActive() {
        if (!active) throw new IllegalStateException("borrowed batch lease has ended");
    }

    /** One borrowed match descriptor with lazy term decoding. */
    public final class BorrowedMatch {
        private final MemorySegment descriptor;
        BorrowedMatch(MemorySegment descriptor) { this.descriptor = descriptor; }

        /** Exact edit distance. */
        public long distance() { ensureActive(); return descriptor.get(JAVA_LONG, DISTANCE); }

        /** Optional provider value. */
        public OptionalLong id() {
            ensureActive();
            return descriptor.get(JAVA_BYTE, HAS_ID) == 0
                    ? OptionalLong.empty()
                    : OptionalLong.of(descriptor.get(JAVA_LONG, ID));
        }

        /** Native unit domain (1 byte, 2 Unicode scalar, 3 u64). */
        public int unitDomain() { ensureActive(); return descriptor.get(JAVA_INT, DOMAIN); }

        /** Borrow raw bytes for byte or Unicode terms without copying. */
        public MemorySegment bytes() {
            ensureActive();
            if (unitDomain() == Native.DOMAIN_U64) throw new IllegalStateException("u64 term has no byte view");
            long length = descriptor.get(JAVA_LONG, BYTE_LEN);
            return descriptor.get(ADDRESS, DATA).reinterpret(length);
        }

        /** Decode a Unicode term only when requested. */
        public String utf8() {
            if (unitDomain() != Native.DOMAIN_UNICODE) throw new IllegalStateException("term is not Unicode");
            return new String(bytes().toArray(JAVA_BYTE), StandardCharsets.UTF_8);
        }

        /** Borrow aligned u64 tokens without copying. */
        public MemorySegment u64() {
            ensureActive();
            if (unitDomain() != Native.DOMAIN_U64) throw new IllegalStateException("term is not u64");
            long length = descriptor.get(JAVA_LONG, TERM_LEN);
            return descriptor.get(ADDRESS, DATA).reinterpret(length * JAVA_LONG.byteSize());
        }

        Match materialize() {
            Term term = switch (unitDomain()) {
                case Native.DOMAIN_BYTE -> new Term.Bytes(bytes().toArray(JAVA_BYTE));
                case Native.DOMAIN_UNICODE -> new Term.Utf8(utf8());
                case Native.DOMAIN_U64 -> new Term.U64(u64().toArray(JAVA_LONG));
                default -> throw new IllegalStateException("unknown unit domain");
            };
            return new Match(term, distance(), id());
        }
    }
}
