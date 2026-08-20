package io.vinarytree.interop;

import java.util.Objects;
import java.util.Spliterator;
import java.util.function.Consumer;

/** Closeable, unsplittable view of one native dictionary entries cursor. */
public final class DictionaryEntrySpliterator
        implements Spliterator<DictionaryEntry>, AutoCloseable {
    private static final int CHARACTERISTICS =
            ORDERED | DISTINCT | NONNULL | IMMUTABLE;

    private final DictionaryEntryIterator iterator;
    private long estimate;

    DictionaryEntrySpliterator(DictionaryEntryIterator iterator) {
        this.iterator = Objects.requireNonNull(iterator, "iterator");
        estimate = iterator.metadata().exactLength().orElse(Long.MAX_VALUE);
    }

    @Override
    public boolean tryAdvance(Consumer<? super DictionaryEntry> action) {
        Objects.requireNonNull(action, "action");
        try {
            if (!iterator.hasNext()) return false;
            action.accept(iterator.next());
            if (estimate != Long.MAX_VALUE) estimate--;
            return true;
        } catch (RuntimeException | Error failure) {
            try {
                close();
            } catch (RuntimeException cleanup) {
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    @Override
    public Spliterator<DictionaryEntry> trySplit() {
        return null;
    }

    @Override
    public long estimateSize() {
        return estimate;
    }

    @Override
    public int characteristics() {
        return CHARACTERISTICS
                | (iterator.metadata().exactLength().isPresent() ? SIZED | SUBSIZED : 0);
    }

    /** Cancel any unread entries and close the cursor. */
    @Override
    public void close() {
        iterator.close();
    }
}
