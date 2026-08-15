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
        return new BorrowedMatch(descriptor(index));
    }

    /** Exact edit distance without allocating a {@link BorrowedMatch} wrapper. */
    public long distance(int index) {
        return descriptors.get(JAVA_LONG, descriptorOffset(index) + DISTANCE);
    }

    /** Number of logical units in the term. */
    public long termLength(int index) {
        return descriptors.get(JAVA_LONG, descriptorOffset(index) + TERM_LEN);
    }

    /** Number of bytes occupied by a byte or Unicode term. */
    public long byteLength(int index) {
        return descriptors.get(JAVA_LONG, descriptorOffset(index) + BYTE_LEN);
    }

    /** Optional provider value without allocating a {@link BorrowedMatch} wrapper. */
    public OptionalLong id(int index) {
        long offset = descriptorOffset(index);
        return descriptors.get(JAVA_BYTE, offset + HAS_ID) == 0
                ? OptionalLong.empty()
                : OptionalLong.of(descriptors.get(JAVA_LONG, offset + ID));
    }

    /** Native unit domain (1 byte, 2 Unicode scalar, 3 u64). */
    public int unitDomain(int index) {
        return descriptors.get(JAVA_INT, descriptorOffset(index) + DOMAIN);
    }

    /** Borrow raw bytes without copying or allocating a per-match wrapper. */
    public MemorySegment bytes(int index) {
        long offset = descriptorOffset(index);
        int domain = descriptors.get(JAVA_INT, offset + DOMAIN);
        if (domain == Native.DOMAIN_U64) {
            throw new IllegalStateException("u64 term has no byte view");
        }
        long length = descriptors.get(JAVA_LONG, offset + BYTE_LEN);
        return descriptors.get(ADDRESS, offset + DATA).reinterpret(length);
    }

    /** Decode one Unicode term without allocating a per-match wrapper. */
    public String utf8(int index) {
        if (unitDomain(index) != Native.DOMAIN_UNICODE) {
            throw new IllegalStateException("term is not Unicode");
        }
        return new String(bytes(index).toArray(JAVA_BYTE), StandardCharsets.UTF_8);
    }

    /** Borrow aligned u64 tokens without copying or allocating a per-match wrapper. */
    public MemorySegment u64(int index) {
        long offset = descriptorOffset(index);
        if (descriptors.get(JAVA_INT, offset + DOMAIN) != Native.DOMAIN_U64) {
            throw new IllegalStateException("term is not u64");
        }
        long length = descriptors.get(JAVA_LONG, offset + TERM_LEN);
        return descriptors.get(ADDRESS, offset + DATA)
                .reinterpret(Math.multiplyExact(length, JAVA_LONG.byteSize()));
    }

    void invalidate() { active = false; }

    private void ensureActive() {
        if (!active) throw new IllegalStateException("borrowed batch lease has ended");
    }

    private long descriptorOffset(int index) {
        ensureActive();
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException(index);
        return Math.multiplyExact((long) index, Native.MATCH.byteSize());
    }

    private MemorySegment descriptor(int index) {
        return descriptors.asSlice(descriptorOffset(index), Native.MATCH.byteSize());
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
            return descriptor.get(ADDRESS, DATA)
                    .reinterpret(Math.multiplyExact(length, JAVA_LONG.byteSize()));
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
