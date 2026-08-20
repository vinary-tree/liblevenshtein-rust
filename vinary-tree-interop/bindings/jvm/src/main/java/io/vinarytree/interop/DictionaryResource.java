package io.vinarytree.interop;

import java.lang.foreign.MemorySegment;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/** A live two-word {@code VtResource} implementing dictionary interface v1. */
public interface DictionaryResource extends AutoCloseable {
    /** Borrow the native resource struct for one synchronous FFM call. */
    MemorySegment resourceSegment();

    /** Open a bounded streaming iterator over the current immutable revision. */
    default DictionaryEntryIterator entryIterator() {
        return entryIterator(DictionaryBatchLimits.DEFAULT);
    }

    /** Open a streaming iterator with explicit per-batch hard limits. */
    default DictionaryEntryIterator entryIterator(DictionaryBatchLimits limits) {
        return DictionaryEntryIterator.open(this, limits);
    }

    /** Open an unsplittable closeable spliterator over the captured revision. */
    default DictionaryEntrySpliterator entrySpliterator() {
        return entrySpliterator(DictionaryBatchLimits.DEFAULT);
    }

    /** Open a closeable spliterator with explicit per-batch hard limits. */
    default DictionaryEntrySpliterator entrySpliterator(DictionaryBatchLimits limits) {
        return new DictionaryEntrySpliterator(entryIterator(limits));
    }

    /**
     * Open a sequential closeable stream.
     *
     * <p>Use the returned stream in try-with-resources when a terminal operation
     * may stop before exhaustion; {@link Stream#close()} cancels the cursor.
     */
    default Stream<DictionaryEntry> entryStream() {
        return entryStream(DictionaryBatchLimits.DEFAULT);
    }

    /** Open a sequential closeable stream with explicit per-batch hard limits. */
    default Stream<DictionaryEntry> entryStream(DictionaryBatchLimits limits) {
        DictionaryEntrySpliterator source = entrySpliterator(limits);
        return StreamSupport.stream(source, false).onClose(source::close);
    }

    /** Materialize immutable Set/Map views of the revision visible now. */
    default DictionarySnapshot entriesSnapshot() {
        return entriesSnapshot(DictionaryBatchLimits.DEFAULT);
    }

    /** Materialize immutable Set/Map views using explicit native batch limits. */
    default DictionarySnapshot entriesSnapshot(DictionaryBatchLimits limits) {
        return DictionarySnapshot.materialize(this, limits);
    }

    /** Release this facade's owned resource retain. */
    @Override
    void close();
}
