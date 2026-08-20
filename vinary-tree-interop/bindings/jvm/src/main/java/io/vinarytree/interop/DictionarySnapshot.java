package io.vinarytree.interop;

import java.util.AbstractCollection;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;

/** Materialized immutable Set/Map views of one captured dictionary revision. */
public final class DictionarySnapshot extends AbstractCollection<DictionaryEntry> {
    private final DictionaryEntriesMetadata metadata;
    private final List<DictionaryEntry> orderedEntries;
    private final Map<DictionaryKey, java.util.Optional<UnsignedLong>> entries;

    static DictionarySnapshot materialize(
            DictionaryResource resource, DictionaryBatchLimits limits) {
        try (DictionaryEntryIterator iterator = DictionaryEntryIterator.open(resource, limits)) {
            DictionaryEntriesMetadata metadata = iterator.metadata();
            if (metadata.exactLength().isPresent()
                    && metadata.exactLength().getAsLong() > Integer.MAX_VALUE) {
                throw new DictionaryInteropException(
                        7, "snapshot is too large for a materialized JVM collection");
            }
            int expected = metadata.exactLength().isPresent()
                    ? Math.toIntExact(metadata.exactLength().getAsLong())
                    : 16;
            List<DictionaryEntry> ordered = new ArrayList<>(expected);
            iterator.forEachRemaining(ordered::add);
            return new DictionarySnapshot(metadata, ordered);
        }
    }

    private DictionarySnapshot(
            DictionaryEntriesMetadata metadata,
            List<DictionaryEntry> ordered) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        // `ordered` is private materializer state. Wrapping it transfers that
        // ownership without copying the backing array a second time.
        this.orderedEntries = Collections.unmodifiableList(ordered);
        this.entries = new OrderedEntryMap(orderedEntries);
    }

    /** Metadata captured with this snapshot. */
    public DictionaryEntriesMetadata metadata() {
        return metadata;
    }

    /** Immutable value-equal Set view in native lexicographic iteration order. */
    public Set<DictionaryKey> keys() {
        return entries.keySet();
    }

    /** Immutable Map view preserving absent versus mapped unsigned-64 values. */
    public Map<DictionaryKey, java.util.Optional<UnsignedLong>> entries() {
        return entries;
    }

    /** Immutable entry records in native lexicographic order. */
    public List<DictionaryEntry> orderedEntries() {
        return orderedEntries;
    }

    /** Number of entries, bounded by the JVM collection range. */
    public int size() {
        return orderedEntries.size();
    }

    /** Iterate in strict native lexicographic order. */
    @Override
    public java.util.Iterator<DictionaryEntry> iterator() {
        return orderedEntries.iterator();
    }

    /**
     * Read-only Map/Set/Collection views over the one sorted entry array.
     *
     * <p>Lookups use binary search and iteration follows native order. This
     * avoids a second output-sized hash table, lazy initialization, and locks.
     */
    private static final class OrderedEntryMap
            extends AbstractMap<DictionaryKey, java.util.Optional<UnsignedLong>> {
        private final List<DictionaryEntry> entries;
        private final Set<Map.Entry<DictionaryKey, java.util.Optional<UnsignedLong>>> entrySet;

        OrderedEntryMap(List<DictionaryEntry> entries) {
            this.entries = entries;
            this.entrySet = new OrderedEntrySet(entries);
        }

        @Override
        public java.util.Optional<UnsignedLong> get(Object key) {
            int index = find(key);
            return index < 0 ? null : entries.get(index).value();
        }

        @Override
        public boolean containsKey(Object key) {
            return find(key) >= 0;
        }

        @Override
        public int size() {
            return entries.size();
        }

        @Override
        public Set<Map.Entry<DictionaryKey, java.util.Optional<UnsignedLong>>> entrySet() {
            return entrySet;
        }

        private int find(Object candidate) {
            if (!(candidate instanceof DictionaryKey key)) return -1;
            int low = 0;
            int high = entries.size() - 1;
            while (low <= high) {
                int middle = (low + high) >>> 1;
                int comparison = entries.get(middle).key().compareTo(key);
                if (comparison < 0) {
                    low = middle + 1;
                } else if (comparison > 0) {
                    high = middle - 1;
                } else {
                    return middle;
                }
            }
            return -1;
        }
    }

    private static final class OrderedEntrySet
            extends AbstractSet<Map.Entry<DictionaryKey, java.util.Optional<UnsignedLong>>> {
        private final List<DictionaryEntry> entries;

        OrderedEntrySet(List<DictionaryEntry> entries) {
            this.entries = entries;
        }

        @Override
        public Iterator<Map.Entry<DictionaryKey, java.util.Optional<UnsignedLong>>> iterator() {
            Iterator<DictionaryEntry> source = entries.iterator();
            return new Iterator<>() {
                @Override
                public boolean hasNext() {
                    return source.hasNext();
                }

                @Override
                public Map.Entry<DictionaryKey, java.util.Optional<UnsignedLong>> next() {
                    if (!source.hasNext()) throw new NoSuchElementException();
                    DictionaryEntry entry = source.next();
                    return Map.entry(entry.key(), entry.value());
                }
            };
        }

        @Override
        public int size() {
            return entries.size();
        }
    }
}
