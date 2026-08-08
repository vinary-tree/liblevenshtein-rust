package io.vinarytree.liblevenshtein;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.vinarytree.interop.DictionaryResource;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/** Levenshtein automaton configuration over a retained dictionary resource. */
public final class Transducer implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    private final State state;
    private final Cleaner.Cleanable cleanable;

    /** Retain a dictionary in O(1), using standard Levenshtein distance. */
    public Transducer(DictionaryResource dictionary) {
        this(dictionary, Algorithm.STANDARD);
    }

    /** Retain a dictionary in O(1) with the selected algorithm. */
    public Transducer(DictionaryResource dictionary, Algorithm algorithm) {
        Objects.requireNonNull(dictionary, "dictionary");
        Objects.requireNonNull(algorithm, "algorithm");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.transducerNew(
                    dictionary.resourceSegment(), algorithm.nativeValue(), out));
            state = new State(out.get(ADDRESS, 0));
        }
        cleanable = CLEANER.register(this, state);
    }

    /** Start a traversal-order Unicode query. */
    public QueryCursor query(String query, long maximumDistance) {
        return query(query, maximumDistance, QueryOrder.TRAVERSAL);
    }

    /** Start a Unicode query and capture the current dictionary revision. */
    public QueryCursor query(String query, long maximumDistance, QueryOrder order) {
        ensureOpen();
        byte[] bytes = Objects.requireNonNull(query, "query").getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = bytes(arena, bytes);
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryUtf8(state.handle(), input, bytes.length, maximumDistance,
                    Objects.requireNonNull(order, "order").nativeValue(), out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
    }

    /** Start a raw-byte query. */
    public QueryCursor query(byte[] query, long maximumDistance) {
        ensureOpen();
        Objects.requireNonNull(query, "query");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryBytes(state.handle(), bytes(arena, query), query.length,
                    maximumDistance, QueryOrder.TRAVERSAL.nativeValue(), out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
    }

    /** Start a raw u64-token query using Java long bit patterns. */
    public QueryCursor query(long[] query, long maximumDistance) {
        ensureOpen();
        Objects.requireNonNull(query, "query");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(
                    Math.max(1, query.length) * JAVA_LONG.byteSize(), JAVA_LONG.byteAlignment());
            for (int index = 0; index < query.length; index++) {
                input.setAtIndex(JAVA_LONG, index, query[index]);
            }
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryU64(state.handle(), input, query.length, maximumDistance,
                    QueryOrder.TRAVERSAL.nativeValue(), out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
    }

    /** Start a dictionary × phonetic-language product query. */
    public QueryCursor query(PhoneticPattern pattern, int maximumDistance) {
        ensureOpen();
        if (maximumDistance < 0 || maximumDistance > 255) {
            throw new IllegalArgumentException("maximumDistance must be 0..255");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryPattern(state.handle(),
                    Objects.requireNonNull(pattern, "pattern").handle(),
                    (byte) maximumDistance, out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
    }

    private static MemorySegment bytes(Arena arena, byte[] bytes) {
        MemorySegment result = arena.allocate(Math.max(1, bytes.length), 1);
        if (bytes.length != 0) {
            result.asSlice(0, bytes.length).copyFrom(MemorySegment.ofArray(bytes));
        }
        return result;
    }

    private void ensureOpen() {
        state.handle();
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static final class State implements Runnable {
        private MemorySegment handle;

        State(MemorySegment handle) {
            this.handle = Objects.requireNonNull(handle, "handle");
        }

        synchronized MemorySegment handle() {
            if (handle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("transducer is closed");
            }
            return handle;
        }

        @Override
        public synchronized void run() {
            if (!handle.equals(MemorySegment.NULL)) {
                Native.transducerFree(handle);
                handle = MemorySegment.NULL;
            }
        }
    }
}
