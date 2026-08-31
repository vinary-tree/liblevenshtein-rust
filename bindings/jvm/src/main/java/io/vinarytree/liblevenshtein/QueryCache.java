package io.vinarytree.liblevenshtein;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Exclusive synchronization-free cache for complete repeated query results.
 *
 * <p>Limits apply independently to traversal and distance-then-term order.
 * Create one cache per worker for parallel workloads. Each returned cursor is
 * independent and may outlive this cache.
 */
public final class QueryCache implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    private final State state;
    private final Cleaner.Cleanable cleanable;

    /** Retain a transducer with practical default hard bounds. */
    public QueryCache(Transducer transducer) {
        this(transducer, 1024, 64L * 1024 * 1024);
    }

    /** Retain a transducer and configure hard per-order bounds. */
    public QueryCache(Transducer transducer, long maximumEntries, long maximumWeight) {
        Objects.requireNonNull(transducer, "transducer");
        if (maximumEntries < 0 || maximumWeight < 0) {
            throw new IllegalArgumentException("cache limits must be nonnegative");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryCacheNew(
                    transducer.handle(), maximumEntries, maximumWeight, out));
            state = new State(out.get(ADDRESS, 0));
        }
        cleanable = CLEANER.register(this, state);
    }

    /** Copy aggregate policy counters and current residency. */
    public QueryCacheStats stats() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(Native.QUERY_CACHE_STATS);
            Native.check(Native.queryCacheStats(state.handle(), out));
            return new QueryCacheStats(
                    out.get(JAVA_LONG, 0), out.get(JAVA_LONG, 8),
                    out.get(JAVA_LONG, 16), out.get(JAVA_LONG, 24),
                    out.get(JAVA_LONG, 32), out.get(JAVA_LONG, 40),
                    out.get(JAVA_LONG, 48), out.get(JAVA_LONG, 56));
        }
    }

    /** Drop resident results while preserving policy counters. */
    public void clear() {
        Native.check(Native.queryCacheClear(state.handle()));
    }

    /** Reset counters while preserving residency and frequency state. */
    public void resetStats() {
        Native.check(Native.queryCacheResetStats(state.handle()));
    }

    /** Query Unicode text through the bounded complete-result cache. */
    public QueryCursor query(String query, long maximumDistance) {
        return query(query, maximumDistance, QueryOrder.TRAVERSAL);
    }

    /** Query Unicode text with an explicit result order. */
    public QueryCursor query(String query, long maximumDistance, QueryOrder order) {
        Transducer.requireDistance(maximumDistance);
        byte[] encoded = Objects.requireNonNull(query, "query").getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryCacheUtf8(
                    state.handle(), Transducer.bytes(arena, encoded), encoded.length,
                    maximumDistance, Objects.requireNonNull(order, "order").nativeValue(), out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
    }

    /** Query exact bytes with an explicit result order. */
    public QueryCursor query(byte[] query, long maximumDistance, QueryOrder order) {
        Transducer.requireDistance(maximumDistance);
        Objects.requireNonNull(query, "query");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryCacheBytes(
                    state.handle(), Transducer.bytes(arena, query), query.length,
                    maximumDistance, Objects.requireNonNull(order, "order").nativeValue(), out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
    }

    /** Query exact u64 tokens represented by Java long bit patterns. */
    public QueryCursor query(long[] query, long maximumDistance, QueryOrder order) {
        Transducer.requireDistance(maximumDistance);
        Objects.requireNonNull(query, "query");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(
                    Math.multiplyExact(Math.max(1L, query.length), JAVA_LONG.byteSize()),
                    JAVA_LONG.byteAlignment());
            for (int index = 0; index < query.length; index++) {
                input.setAtIndex(JAVA_LONG, index, query[index]);
            }
            MemorySegment out = arena.allocate(ADDRESS);
            Native.check(Native.queryCacheU64(
                    state.handle(), input, query.length, maximumDistance,
                    Objects.requireNonNull(order, "order").nativeValue(), out));
            return new QueryCursor(out.get(ADDRESS, 0));
        }
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

        MemorySegment handle() {
            if (handle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("query cache is closed");
            }
            return handle;
        }

        @Override
        public void run() {
            if (!handle.equals(MemorySegment.NULL)) {
                Native.queryCacheFree(handle);
                handle = MemorySegment.NULL;
            }
        }
    }
}
