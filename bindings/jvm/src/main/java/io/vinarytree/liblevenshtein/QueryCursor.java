package io.vinarytree.liblevenshtein;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;

/** One-shot lazy query stream retaining its query-start dictionary snapshot. */
public final class QueryCursor implements Iterator<Match>, Iterable<Match>, AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();
    private static final long MATCHES = Native.BATCH.byteOffset(groupElement("matches"));
    private static final long LEN = Native.BATCH.byteOffset(groupElement("len"));
    private static final long GENERATION = Native.BATCH.byteOffset(groupElement("generation"));

    private final Arena arena = Arena.ofShared();
    private final MemorySegment batchOut = arena.allocate(Native.BATCH);
    private final State state;
    private final Cleaner.Cleanable cleanable;
    private final ArrayDeque<Match> materialized = new ArrayDeque<>();
    private boolean claimed;
    private boolean exhausted;

    QueryCursor(MemorySegment handle) {
        state = new State(handle, arena);
        cleanable = CLEANER.register(this, state);
    }

    /** Consume native batches without allocating managed match objects. */
    public void forEachBatch(BorrowedBatchConsumer consumer) {
        Objects.requireNonNull(consumer, "consumer");
        claim();
        try {
            while (withNextBatch(consumer)) {}
        } finally {
            close();
        }
    }

    private boolean withNextBatch(BorrowedBatchConsumer consumer) {
        MemorySegment handle = state.handle();
        int status = Native.nextBatch(handle, GeneratedAbi.DEFAULT_MATCH_BATCH, batchOut);
        if (status == Native.END) { exhausted = true; return false; }
        Native.check(status);
        long length = batchOut.get(JAVA_LONG, LEN);
        long generation = batchOut.get(JAVA_LONG, GENERATION);
        MemorySegment pointer = batchOut.get(ADDRESS, MATCHES);
        BorrowedMatchBatch batch = new BorrowedMatchBatch(
                pointer.reinterpret(length * Native.MATCH.byteSize()), Math.toIntExact(length));
        try {
            consumer.accept(batch);
        } finally {
            batch.invalidate();
            Native.check(Native.releaseBatch(handle, generation));
        }
        return true;
    }

    private void fillMaterialized() {
        if (exhausted || !materialized.isEmpty()) return;
        withNextBatch(batch -> {
            for (int index = 0; index < batch.size(); index++) {
                materialized.addLast(batch.get(index).materialize());
            }
        });
        if (exhausted) close();
    }

    @Override public boolean hasNext() {
        if (!claimed) claimed = true;
        fillMaterialized();
        return !materialized.isEmpty();
    }

    @Override public Match next() {
        if (!hasNext()) throw new NoSuchElementException();
        return materialized.removeFirst();
    }

    @Override public Iterator<Match> iterator() {
        claim();
        return this;
    }

    private void claim() {
        if (claimed) throw new IllegalStateException("query cursor may only be consumed once");
        claimed = true;
    }

    @Override public void close() {
        int status = state.closeExplicitly();
        cleanable.clean();
        if (status != GeneratedAbi.STATUS_OK) {
            Native.check(status);
        }
    }

    private static final class State implements Runnable {
        private MemorySegment handle;
        private Arena arena;

        State(MemorySegment handle, Arena arena) {
            this.handle = Objects.requireNonNull(handle, "handle");
            this.arena = Objects.requireNonNull(arena, "arena");
        }

        synchronized MemorySegment handle() {
            if (handle.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("query cursor is closed");
            }
            return handle;
        }

        synchronized int closeExplicitly() {
            if (handle.equals(MemorySegment.NULL)) return GeneratedAbi.STATUS_OK;
            int status = Native.cursorFree(handle);
            handle = MemorySegment.NULL;
            arena.close();
            arena = null;
            return status;
        }

        @Override
        public synchronized void run() {
            if (!handle.equals(MemorySegment.NULL)) {
                Native.cursorFree(handle);
                handle = MemorySegment.NULL;
                arena.close();
                arena = null;
            }
        }
    }
}
