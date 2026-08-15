package io.vinarytree.liblevenshtein;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.MemorySegment;

/** Shared native-batch lease mechanics for lazy and lexical query consumers. */
final class NativeQueryBatches {
    private static final long MATCHES = Native.BATCH.byteOffset(groupElement("matches"));
    private static final long LEN = Native.BATCH.byteOffset(groupElement("len"));
    private static final long GENERATION = Native.BATCH.byteOffset(groupElement("generation"));

    private NativeQueryBatches() {}

    /**
     * Borrow one native page for the dynamic extent of {@code consumer}.
     *
     * @return {@code false} only when the cursor is exhausted
     */
    static boolean withNextBatch(
            MemorySegment cursor,
            MemorySegment batchOut,
            BorrowedBatchConsumer consumer) {
        int status = Native.nextBatch(cursor, GeneratedAbi.DEFAULT_MATCH_BATCH, batchOut);
        if (status == Native.END) {
            return false;
        }
        Native.check(status);
        long length = batchOut.get(JAVA_LONG, LEN);
        long generation = batchOut.get(JAVA_LONG, GENERATION);
        MemorySegment pointer = batchOut.get(ADDRESS, MATCHES);
        BorrowedMatchBatch batch = new BorrowedMatchBatch(
                pointer.reinterpret(Math.multiplyExact(length, Native.MATCH.byteSize())),
                Math.toIntExact(length));
        try {
            consumer.accept(batch);
        } finally {
            batch.invalidate();
            Native.check(Native.releaseBatch(cursor, generation));
        }
        return true;
    }
}
