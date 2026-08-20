package io.vinarytree.liblevenshtein;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.vinarytree.interop.DictionaryBatchLimits;
import io.vinarytree.interop.DictionaryEntry;
import io.vinarytree.interop.DictionaryKey;
import io.vinarytree.interop.DictionaryResource;
import io.vinarytree.interop.DictionarySnapshot;
import io.vinarytree.interop.DictionaryUnitDomain;
import io.vinarytree.interop.DictionaryValueDomain;
import io.vinarytree.interop.InteropLayouts;
import io.vinarytree.interop.UnsignedLong;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Executable entries-v1 collection conformance against the native provider. */
final class DictionaryCollectionsTest {
    private static final DictionaryBatchLimits SMALL_BATCH =
            new DictionaryBatchLimits(2, 16, 2);

    @Test
    void immutableSetAndMapViewsCaptureOneUnicodeRevision() {
        try (NativeDictionary dictionary = new NativeDictionary(DictionaryUnitDomain.UNICODE_SCALAR)) {
            dictionary.putBytes(new byte[0], Optional.empty());
            dictionary.putUtf8("café", Optional.of(new UnsignedLong(-1)));
            dictionary.putUtf8("cat", Optional.empty());

            DictionarySnapshot snapshot = dictionary.entriesSnapshot(SMALL_BATCH);
            assertEquals(0, snapshot.orderedEntries().get(0).key().unitCount());
            assertEquals(DictionaryUnitDomain.UNICODE_SCALAR, snapshot.metadata().unitDomain());
            assertEquals(DictionaryValueDomain.OPTIONAL_U64, snapshot.metadata().valueDomain());
            assertEquals(3, snapshot.metadata().exactLength().orElseThrow());
            assertTrue(snapshot.metadata().snapshotIdentity().isPresent());
            assertEquals(List.of("", "café", "cat"), snapshot.keys().stream()
                    .map(DictionaryKey::unicode)
                    .toList());
            assertEquals(
                    Optional.of(new UnsignedLong(-1)),
                    snapshot.entries().get(DictionaryKey.unicode("café")));
            assertEquals(Optional.empty(), snapshot.entries().get(DictionaryKey.unicode("cat")));
            assertFalse(snapshot.entries().containsKey(DictionaryKey.unicode("dog")));
            assertThrows(
                    UnsupportedOperationException.class,
                    () -> snapshot.keys().add(DictionaryKey.unicode("dog")));
            assertThrows(
                    UnsupportedOperationException.class,
                    () -> snapshot.entries().put(DictionaryKey.unicode("dog"), Optional.empty()));

            dictionary.putUtf8("dog", Optional.of(new UnsignedLong(7)));
            assertEquals(3, snapshot.size());
            assertFalse(snapshot.keys().contains(DictionaryKey.unicode("dog")));
            DictionarySnapshot fresh = dictionary.entriesSnapshot(SMALL_BATCH);
            assertEquals(4, fresh.size());
            assertNotEquals(
                    snapshot.metadata().snapshotIdentity(), fresh.metadata().snapshotIdentity());
        }
    }

    @Test
    void arbitraryBytesAndUnsignedTokensRemainLosslessAndOrdered() {
        try (NativeDictionary bytes = new NativeDictionary(DictionaryUnitDomain.BYTE)) {
            bytes.putBytes(new byte[0], Optional.empty());
            bytes.putBytes(new byte[] {0}, Optional.of(new UnsignedLong(0)));
            bytes.putBytes(new byte[] {0, (byte) 0xff}, Optional.of(new UnsignedLong(-1)));
            bytes.putBytes(new byte[] {(byte) 0xff}, Optional.empty());
            List<DictionaryEntry> entries = bytes.entriesSnapshot(SMALL_BATCH).orderedEntries();
            assertEquals(0, entries.get(0).key().unitCount());
            assertEquals(1, entries.get(1).key().unitCount());
            assertArrayEquals(new byte[0], entries.get(0).key().bytes());
            assertArrayEquals(new byte[] {0}, entries.get(1).key().bytes());
            assertArrayEquals(new byte[] {0, (byte) 0xff}, entries.get(2).key().bytes());
            assertArrayEquals(new byte[] {(byte) 0xff}, entries.get(3).key().bytes());
            assertEquals("18446744073709551615", entries.get(2).value().orElseThrow().toString());
        }

        try (NativeDictionary tokens = new NativeDictionary(DictionaryUnitDomain.U64)) {
            tokens.putU64(new long[] {0}, Optional.empty());
            tokens.putU64(new long[] {Long.MIN_VALUE}, Optional.of(new UnsignedLong(Long.MIN_VALUE)));
            tokens.putU64(new long[] {-1}, Optional.of(new UnsignedLong(-1)));
            List<DictionaryEntry> entries = tokens.entriesSnapshot(SMALL_BATCH).orderedEntries();
            assertTrue(entries.stream().allMatch(entry -> entry.key().unitCount() == 1));
            assertArrayEquals(new long[] {0}, entries.get(0).key().u64());
            assertArrayEquals(new long[] {Long.MIN_VALUE}, entries.get(1).key().u64());
            assertArrayEquals(new long[] {-1}, entries.get(2).key().u64());
        }
    }

    @Test
    void closeableIteratorSpliteratorAndStreamCancelEarly() {
        try (NativeDictionary dictionary = new NativeDictionary(DictionaryUnitDomain.UNICODE_SCALAR)) {
            for (String value : List.of("a", "b", "c", "d")) {
                dictionary.putUtf8(value, Optional.empty());
            }
            try (var iterator = dictionary.entryIterator(SMALL_BATCH)) {
                assertTrue(iterator.hasNext());
                assertEquals("a", iterator.next().key().unicode());
                iterator.cancel();
                assertFalse(iterator.hasNext());
            }
            try (var stream = dictionary.entryStream(SMALL_BATCH)) {
                assertEquals(List.of("a", "b"), stream.limit(2)
                        .map(entry -> entry.key().unicode())
                        .toList());
            }
            try (var spliterator = dictionary.entrySpliterator(SMALL_BATCH)) {
                var first = new java.util.ArrayList<String>();
                assertTrue(spliterator.tryAdvance(entry -> first.add(entry.key().unicode())));
                assertEquals(List.of("a"), first);
            }
        }
    }

    private static final class NativeDictionary implements DictionaryResource {
        private static final Linker LINKER = Linker.nativeLinker();
        private static final Arena LIBRARY_ARENA = Arena.global();
        private static final SymbolLookup SYMBOLS = SymbolLookup.libraryLookup(
                Path.of(
                        System.getProperty("libdictenstein.nativeDir"),
                        System.mapLibraryName("libdictenstein")),
                LIBRARY_ARENA);
        private static final MethodHandle CREATE = downcall(
                "ldict_dynamic_dawg_new", FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS));
        private static final MethodHandle FREE = downcall(
                "ldict_dictionary_free", FunctionDescriptor.ofVoid(ADDRESS));
        private static final MethodHandle RESOURCE = downcall(
                "ldict_dictionary_resource", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
        private static final MethodHandle PUT_TEXT = downcall(
                "ldict_dictionary_insert_text_value",
                FunctionDescriptor.of(
                        JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_BYTE, ADDRESS));
        private static final MethodHandle PUT_U64 = downcall(
                "ldict_dictionary_insert_u64_value",
                FunctionDescriptor.of(
                        JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_BYTE, ADDRESS));

        private final Arena arena = Arena.ofShared();
        private final MemorySegment resource = arena.allocate(InteropLayouts.RESOURCE);
        private MemorySegment handle;

        NativeDictionary(DictionaryUnitDomain domain) {
            try (Arena call = Arena.ofConfined()) {
                MemorySegment output = call.allocate(ADDRESS);
                check(callInt(CREATE, switch (domain) {
                    case BYTE -> 1;
                    case UNICODE_SCALAR -> 2;
                    case U64 -> 3;
                }, output));
                handle = output.get(ADDRESS, 0);
                check(callInt(RESOURCE, handle, resource));
            }
        }

        void putUtf8(String value, Optional<UnsignedLong> mapped) {
            putBytes(value.getBytes(StandardCharsets.UTF_8), mapped);
        }

        void putBytes(byte[] value, Optional<UnsignedLong> mapped) {
            try (Arena call = Arena.ofConfined()) {
                MemorySegment data = value.length == 0
                        ? MemorySegment.NULL
                        : call.allocateFrom(JAVA_BYTE, value);
                MemorySegment inserted = call.allocate(JAVA_BYTE);
                check(callInt(
                        PUT_TEXT,
                        handle,
                        data,
                        (long) value.length,
                        mapped.map(UnsignedLong::bits).orElse(0L),
                        (byte) (mapped.isPresent() ? 1 : 0),
                        inserted));
                assertEquals(1, Byte.toUnsignedInt(inserted.get(JAVA_BYTE, 0)));
            }
        }

        void putU64(long[] value, Optional<UnsignedLong> mapped) {
            try (Arena call = Arena.ofConfined()) {
                MemorySegment data = value.length == 0
                        ? MemorySegment.NULL
                        : call.allocateFrom(JAVA_LONG, value);
                MemorySegment inserted = call.allocate(JAVA_BYTE);
                check(callInt(
                        PUT_U64,
                        handle,
                        data,
                        (long) value.length,
                        mapped.map(UnsignedLong::bits).orElse(0L),
                        (byte) (mapped.isPresent() ? 1 : 0),
                        inserted));
                assertEquals(1, Byte.toUnsignedInt(inserted.get(JAVA_BYTE, 0)));
            }
        }

        @Override
        public MemorySegment resourceSegment() {
            if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("closed");
            return resource;
        }

        @Override
        public void close() {
            if (!handle.equals(MemorySegment.NULL)) {
                run(FREE, handle);
                handle = MemorySegment.NULL;
                arena.close();
            }
        }

        private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
            return LINKER.downcallHandle(SYMBOLS.find(name).orElseThrow(), descriptor);
        }

        private static int callInt(MethodHandle handle, Object... arguments) {
            try {
                return (int) handle.invokeWithArguments(arguments);
            } catch (Throwable throwable) {
                throw new RuntimeException(throwable);
            }
        }

        private static void run(MethodHandle handle, Object... arguments) {
            try {
                handle.invokeWithArguments(arguments);
            } catch (Throwable throwable) {
                throw new RuntimeException(throwable);
            }
        }

        private static void check(int status) {
            if (status != 0) throw new AssertionError("libdictenstein status " + status);
        }
    }
}
