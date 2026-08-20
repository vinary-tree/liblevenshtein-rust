package io.vinarytree.interop;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.MemoryLayout.PathElement.sequenceElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Single-consumer, closeable iterator over one captured dictionary revision.
 *
 * <p>Each refill performs one entries-v1 {@code next_batch} call, validates and
 * copies the complete bounded native batch, then releases its generation before
 * exposing any managed entry. Closing early sends {@code cancel} before
 * {@code close}.
 */
public final class DictionaryEntryIterator
        implements java.util.Iterator<DictionaryEntry>, AutoCloseable {
    private static final int OK = 0;
    private static final int END = 1;
    private static final int UNSUPPORTED = 4;
    private static final int PROVIDER_ERROR = 8;
    private static final int BATCH_IN_USE = 9;
    private static final long INFO_EXACT_LEN = 1;
    private static final long INFO_SNAPSHOT_IDENTITY = 2;
    private static final int ORDER_LEXICOGRAPHIC = 1;
    private static final byte[] ENTRIES_ID =
            "vt.dict.entry.v1".getBytes(StandardCharsets.US_ASCII);
    private static final Cleaner CLEANER = Cleaner.create();
    private static final Linker LINKER = Linker.nativeLinker();

    private static final long RESOURCE_CONTEXT =
            InteropLayouts.RESOURCE.byteOffset(groupElement("context"));
    private static final long RESOURCE_VTABLE =
            InteropLayouts.RESOURCE.byteOffset(groupElement("vtable"));
    private static final long INFO_UNIT_DOMAIN =
            InteropLayouts.DICTIONARY_ENTRIES_INFO.byteOffset(groupElement("unit_domain"));
    private static final long INFO_VALUE_DOMAIN =
            InteropLayouts.DICTIONARY_ENTRIES_INFO.byteOffset(groupElement("value_domain"));
    private static final long INFO_ORDER =
            InteropLayouts.DICTIONARY_ENTRIES_INFO.byteOffset(groupElement("order"));
    private static final long INFO_RESERVED0 =
            InteropLayouts.DICTIONARY_ENTRIES_INFO.byteOffset(groupElement("reserved0"));
    private static final long INFO_FLAGS =
            InteropLayouts.DICTIONARY_ENTRIES_INFO.byteOffset(groupElement("flags"));
    private static final long INFO_EXACT_LENGTH =
            InteropLayouts.DICTIONARY_ENTRIES_INFO.byteOffset(groupElement("exact_len"));
    private static final long INFO_IDENTITY_PRODUCER = InteropLayouts.DICTIONARY_ENTRIES_INFO
            .byteOffset(groupElement("identity_producer"));
    private static final long INFO_IDENTITY_REVISION = InteropLayouts.DICTIONARY_ENTRIES_INFO
            .byteOffset(groupElement("identity_revision"));
    private static final long INFO_RESERVED_0 = InteropLayouts.DICTIONARY_ENTRIES_INFO
            .byteOffset(groupElement("reserved"), sequenceElement(0));
    private static final long INFO_RESERVED_1 = InteropLayouts.DICTIONARY_ENTRIES_INFO
            .byteOffset(groupElement("reserved"), sequenceElement(1));
    private static final long CURSOR_CONTEXT =
            InteropLayouts.DICTIONARY_ENTRIES_CURSOR.byteOffset(groupElement("context"));
    private static final long CURSOR_VTABLE =
            InteropLayouts.DICTIONARY_ENTRIES_CURSOR.byteOffset(groupElement("vtable"));
    private static final long BATCH_ENTRIES =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("entries"));
    private static final long BATCH_ENTRY_COUNT =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("entry_count"));
    private static final long BATCH_UNITS =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("units"));
    private static final long BATCH_UNIT_COUNT =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("unit_count"));
    private static final long BATCH_VALUES =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("values"));
    private static final long BATCH_VALUE_COUNT =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("value_count"));
    private static final long BATCH_GENERATION =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("generation"));
    private static final long BATCH_RESERVED =
            InteropLayouts.DICTIONARY_ENTRY_BATCH.byteOffset(groupElement("reserved"));
    private static final long ENTRY_UNIT_OFFSET =
            InteropLayouts.DICTIONARY_ENTRY.byteOffset(groupElement("unit_offset"));
    private static final long ENTRY_UNIT_LENGTH =
            InteropLayouts.DICTIONARY_ENTRY.byteOffset(groupElement("unit_len"));
    private static final long ENTRY_VALUE_OFFSET =
            InteropLayouts.DICTIONARY_ENTRY.byteOffset(groupElement("value_offset"));
    private static final long ENTRY_VALUE_LENGTH =
            InteropLayouts.DICTIONARY_ENTRY.byteOffset(groupElement("value_len"));
    private static final long ENTRY_RESERVED =
            InteropLayouts.DICTIONARY_ENTRY.byteOffset(groupElement("reserved"));

    private final DictionaryBatchLimits limits;
    private final DictionaryEntriesMetadata metadata;
    private final State state;
    private final Cleaner.Cleanable cleanable;
    private final ArrayDeque<DictionaryEntry> buffered = new ArrayDeque<>();
    private DictionaryKey previous;
    private long delivered;
    private boolean exhausted;

    private DictionaryEntryIterator(
            DictionaryBatchLimits limits,
            DictionaryEntriesMetadata metadata,
            State state) {
        this.limits = limits;
        this.metadata = metadata;
        this.state = state;
        this.cleanable = CLEANER.register(this, state);
    }

    /** Open entries-v1 on a borrowed resource and capture its current revision. */
    public static DictionaryEntryIterator open(
            DictionaryResource resource, DictionaryBatchLimits limits) {
        Objects.requireNonNull(resource, "resource");
        Objects.requireNonNull(limits, "limits");
        Arena arena = Arena.ofShared();
        MemorySegment cursor = arena.allocate(InteropLayouts.DICTIONARY_ENTRIES_CURSOR);
        MemorySegment info = arena.allocate(InteropLayouts.DICTIONARY_ENTRIES_INFO);
        cursor.fill((byte) 0);
        info.fill((byte) 0);
        State state = null;
        MethodHandle fallbackClose = null;
        boolean cursorOpened = false;
        try (Arena callArena = Arena.ofConfined()) {
            MemorySegment source = resource.resourceSegment().reinterpret(
                    InteropLayouts.RESOURCE.byteSize());
            MemorySegment resourceContext = source.get(ADDRESS, RESOURCE_CONTEXT);
            MemorySegment baseAddress = source.get(ADDRESS, RESOURCE_VTABLE);
            requireAddress(resourceContext, "resource context");
            requireAddress(baseAddress, "resource vtable");
            MemorySegment base = baseAddress.reinterpret(InteropLayouts.RESOURCE_VTABLE.byteSize());
            validateBaseVtable(base);

            MemorySegment id = callArena.allocate(ENTRIES_ID.length, 1);
            MemorySegment.copy(ENTRIES_ID, 0, id, JAVA_BYTE, 0, ENTRIES_ID.length);
            MemorySegment outVtable = callArena.allocate(ADDRESS);
            outVtable.set(ADDRESS, 0, MemorySegment.NULL);
            MethodHandle query = downcall(
                    base.get(ADDRESS, 32),
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
            int queryStatus = callInt(
                    query,
                    resourceContext,
                    id,
                    InteropLayouts.DICTIONARY_ENTRIES_INTERFACE_VERSION,
                    outVtable);
            if (queryStatus == UNSUPPORTED) {
                throw new UnsupportedOperationException(
                        "resource does not implement vt.dict.entry.v1");
            }
            check(queryStatus, "query dictionary entries interface");
            MemorySegment interfaceAddress = outVtable.get(ADDRESS, 0);
            MemorySegment entriesVtable = validatedEntriesVtable(interfaceAddress);
            fallbackClose = downcall(
                    entriesVtable.get(ADDRESS, 56), FunctionDescriptor.of(JAVA_INT, ADDRESS));
            MethodHandle open = downcall(
                    entriesVtable.get(ADDRESS, 16),
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
            int openStatus = callInt(open, resourceContext, cursor, info);
            check(openStatus, "open dictionary entries cursor");
            cursorOpened = true;

            MemorySegment cursorContext = cursor.get(ADDRESS, CURSOR_CONTEXT);
            MemorySegment cursorVtableAddress = cursor.get(ADDRESS, CURSOR_VTABLE);
            requireAddress(cursorContext, "entries cursor context");
            MemorySegment cursorVtable = validatedEntriesVtable(cursorVtableAddress);
            state = new State(arena, cursor, cursorVtable);
            DictionaryEntriesMetadata metadata = validateMetadata(info);
            return new DictionaryEntryIterator(limits, metadata, state);
        } catch (Throwable failure) {
            if (state != null) {
                try {
                    state.close(false);
                } catch (RuntimeException cleanup) {
                    failure.addSuppressed(cleanup);
                }
            } else {
                if (cursorOpened && fallbackClose != null) {
                    try {
                        callInt(fallbackClose, cursor);
                    } catch (RuntimeException cleanup) {
                        failure.addSuppressed(cleanup);
                    }
                }
                arena.close();
            }
            throw rethrow(failure);
        }
    }

    /** Metadata captured at the same linearization point as this cursor. */
    public DictionaryEntriesMetadata metadata() {
        return metadata;
    }

    @Override
    public boolean hasNext() {
        if (buffered.isEmpty() && !exhausted) {
            try {
                refill();
            } catch (RuntimeException | Error failure) {
                if (!state.isClosed()) {
                    try {
                        state.close(true);
                    } catch (RuntimeException cleanup) {
                        failure.addSuppressed(cleanup);
                    }
                }
                cleanable.clean();
                exhausted = true;
                buffered.clear();
                throw failure;
            }
        }
        return !buffered.isEmpty();
    }

    @Override
    public DictionaryEntry next() {
        if (!hasNext()) throw new NoSuchElementException();
        return buffered.removeFirst();
    }

    /** Cancel the remaining native traversal and close this cursor. */
    public void cancel() {
        close();
    }

    @Override
    public void close() {
        exhausted = true;
        buffered.clear();
        try {
            state.close(true);
        } finally {
            cleanable.clean();
        }
    }

    private void refill() {
        MemorySegment batch = state.batch();
        batch.fill((byte) 0);
        int status = state.next(limits);
        if (status == END) {
            validateEmptyBatch(batch);
            exhausted = true;
            verifyExactLength();
            state.close(false);
            cleanable.clean();
            return;
        }
        if (status != OK) {
            failAndClose(new DictionaryInteropException(
                    status, "dictionary entries next_batch returned status "
                            + Integer.toUnsignedString(status)));
        }

        long generation = batch.get(JAVA_LONG, BATCH_GENERATION);
        RuntimeException failure = null;
        try {
            copyBatch(batch);
        } catch (RuntimeException exception) {
            failure = exception;
        }
        try {
            check(state.release(generation), "release dictionary entries batch");
        } catch (RuntimeException releaseFailure) {
            if (failure == null) failure = releaseFailure;
            else failure.addSuppressed(releaseFailure);
        }
        if (failure != null) failAndClose(failure);
    }

    private void copyBatch(MemorySegment batch) {
        long entryCount = positiveCount(batch.get(JAVA_LONG, BATCH_ENTRY_COUNT), "entry_count");
        long unitCount = nonnegative(batch.get(JAVA_LONG, BATCH_UNIT_COUNT), "unit_count");
        long valueCount = nonnegative(batch.get(JAVA_LONG, BATCH_VALUE_COUNT), "value_count");
        long generation = batch.get(JAVA_LONG, BATCH_GENERATION);
        if (entryCount > limits.maxEntries()
                || unitCount > limits.maxUnits()
                || valueCount > limits.maxValues()) {
            malformed("batch exceeds the requested limits");
        }
        if (generation == 0 || batch.get(JAVA_LONG, BATCH_RESERVED) != 0) {
            malformed("batch generation/reserved fields are invalid");
        }

        MemorySegment entriesAddress = batch.get(ADDRESS, BATCH_ENTRIES);
        MemorySegment unitsAddress = batch.get(ADDRESS, BATCH_UNITS);
        MemorySegment valuesAddress = batch.get(ADDRESS, BATCH_VALUES);
        requireAddress(entriesAddress, "nonempty entry descriptors");
        requirePointerCount(unitsAddress, unitCount, "unit arena");
        requirePointerCount(valuesAddress, valueCount, "value arena");
        if (entriesAddress.address() % InteropLayouts.DICTIONARY_ENTRY.byteAlignment() != 0) {
            malformed("entry descriptor arena is misaligned");
        }
        if (valueCount != 0 && valuesAddress.address() % JAVA_LONG.byteAlignment() != 0) {
            malformed("value arena is misaligned");
        }

        long descriptorBytes = Math.multiplyExact(
                entryCount, InteropLayouts.DICTIONARY_ENTRY.byteSize());
        MemorySegment entries = entriesAddress.reinterpret(descriptorBytes);
        MemorySegment units = reinterpretUnits(unitsAddress, unitCount);
        MemorySegment values = valueCount == 0
                ? MemorySegment.NULL
                : valuesAddress.reinterpret(Math.multiplyExact(valueCount, JAVA_LONG.byteSize()));

        long nextUnit = 0;
        long nextValue = 0;
        for (long index = 0; index < entryCount; index++) {
            MemorySegment descriptor = entries.asSlice(
                    index * InteropLayouts.DICTIONARY_ENTRY.byteSize(),
                    InteropLayouts.DICTIONARY_ENTRY.byteSize());
            long unitOffset = nonnegative(
                    descriptor.get(JAVA_LONG, ENTRY_UNIT_OFFSET), "unit offset");
            long unitLength = nonnegative(
                    descriptor.get(JAVA_LONG, ENTRY_UNIT_LENGTH), "unit length");
            long valueOffset = nonnegative(
                    descriptor.get(JAVA_LONG, ENTRY_VALUE_OFFSET), "value offset");
            long valueLength = nonnegative(
                    descriptor.get(JAVA_LONG, ENTRY_VALUE_LENGTH), "value length");
            if (descriptor.get(JAVA_LONG, ENTRY_RESERVED) != 0
                    || unitOffset != nextUnit
                    || valueOffset != nextValue
                    || Math.addExact(unitOffset, unitLength) > unitCount
                    || Math.addExact(valueOffset, valueLength) > valueCount) {
                malformed("entry descriptor arenas are not canonical and packed");
            }
            if (metadata.valueDomain() == DictionaryValueDomain.UNIT && valueLength != 0
                    || metadata.valueDomain() == DictionaryValueDomain.OPTIONAL_U64
                            && valueLength > 1) {
                malformed("entry value length does not match the value domain");
            }

            DictionaryKey key = decodeKey(units, unitOffset, unitLength);
            if (previous != null && previous.compareTo(key) >= 0) {
                malformed("entries are not globally strict lexicographic order");
            }
            previous = key;
            Optional<UnsignedLong> value = valueLength == 0
                    ? Optional.empty()
                    : Optional.of(new UnsignedLong(values.getAtIndex(JAVA_LONG, valueOffset)));
            buffered.addLast(new DictionaryEntry(key, value));
            nextUnit = Math.addExact(unitOffset, unitLength);
            nextValue = Math.addExact(valueOffset, valueLength);
        }
        if (nextUnit != unitCount || nextValue != valueCount) {
            malformed("entry descriptors do not consume their complete arenas");
        }
        delivered = Math.addExact(delivered, entryCount);
        if (metadata.exactLength().isPresent()
                && delivered > metadata.exactLength().getAsLong()) {
            malformed("cursor yielded more than its advertised exact length");
        }
    }

    private MemorySegment reinterpretUnits(MemorySegment address, long count) {
        if (count == 0) return MemorySegment.NULL;
        long elementSize = switch (metadata.unitDomain()) {
            case BYTE -> 1;
            case UNICODE_SCALAR -> 4;
            case U64 -> 8;
        };
        if (address.address() % elementSize != 0) malformed("unit arena is misaligned");
        return address.reinterpret(Math.multiplyExact(count, elementSize));
    }

    private DictionaryKey decodeKey(MemorySegment units, long offset, long length) {
        int size = Math.toIntExact(length);
        return switch (metadata.unitDomain()) {
            case BYTE -> {
                byte[] result = size == 0
                        ? new byte[0]
                        : units.asSlice(offset, length).toArray(JAVA_BYTE);
                yield DictionaryKey.bytes(result);
            }
            case UNICODE_SCALAR -> {
                int[] result = new int[size];
                for (int i = 0; i < size; i++) {
                    int scalar = units.getAtIndex(JAVA_INT, Math.addExact(offset, i));
                    if (!Character.isValidCodePoint(scalar)
                            || scalar >= Character.MIN_SURROGATE
                                    && scalar <= Character.MAX_SURROGATE) {
                        malformed("Unicode entry contains a non-scalar code point");
                    }
                    result[i] = scalar;
                }
                yield DictionaryKey.unicodeScalars(result);
            }
            case U64 -> {
                long[] result = new long[size];
                for (int i = 0; i < size; i++) {
                    result[i] = units.getAtIndex(JAVA_LONG, Math.addExact(offset, i));
                }
                yield DictionaryKey.u64(result);
            }
        };
    }

    private void verifyExactLength() {
        if (metadata.exactLength().isPresent()
                && delivered != metadata.exactLength().getAsLong()) {
            malformed("cursor exhausted before its advertised exact length");
        }
    }

    private void failAndClose(RuntimeException failure) {
        try {
            state.close(true);
        } catch (RuntimeException cleanup) {
            failure.addSuppressed(cleanup);
        }
        cleanable.clean();
        exhausted = true;
        buffered.clear();
        throw failure;
    }

    private static DictionaryEntriesMetadata validateMetadata(MemorySegment info) {
        if (info.get(JAVA_INT, INFO_ORDER) != ORDER_LEXICOGRAPHIC
                || info.get(JAVA_INT, INFO_RESERVED0) != 0
                || info.get(JAVA_LONG, INFO_RESERVED_0) != 0
                || info.get(JAVA_LONG, INFO_RESERVED_1) != 0) {
            malformed("entries metadata order/reserved fields are invalid");
        }
        DictionaryUnitDomain unit =
                DictionaryUnitDomain.fromWire(info.get(JAVA_INT, INFO_UNIT_DOMAIN));
        DictionaryValueDomain value =
                DictionaryValueDomain.fromWire(info.get(JAVA_INT, INFO_VALUE_DOMAIN));
        long flags = info.get(JAVA_LONG, INFO_FLAGS);
        long exact = info.get(JAVA_LONG, INFO_EXACT_LENGTH);
        long producer = info.get(JAVA_LONG, INFO_IDENTITY_PRODUCER);
        long revision = info.get(JAVA_LONG, INFO_IDENTITY_REVISION);
        OptionalLong exactLength;
        if ((flags & INFO_EXACT_LEN) != 0) {
            if (exact < 0) malformed("exact entry length exceeds the JVM range");
            exactLength = OptionalLong.of(exact);
        } else {
            if (exact != 0) malformed("absent exact length must be zero");
            exactLength = OptionalLong.empty();
        }
        Optional<SnapshotIdentity> identity;
        if ((flags & INFO_SNAPSHOT_IDENTITY) != 0) {
            identity = Optional.of(new SnapshotIdentity(producer, revision));
        } else {
            if (producer != 0 || revision != 0) malformed("absent snapshot identity must be zero");
            identity = Optional.empty();
        }
        return new DictionaryEntriesMetadata(unit, value, exactLength, identity);
    }

    private static void validateBaseVtable(MemorySegment table) {
        if (table.get(JAVA_LONG, 0) < InteropLayouts.RESOURCE_VTABLE.byteSize()
                || table.get(JAVA_INT, 8) != InteropLayouts.ABI_VERSION
                || table.get(JAVA_INT, 12) != 0) {
            malformed("resource base vtable header is incompatible");
        }
        requireAddress(table.get(ADDRESS, 16), "resource retain operation");
        requireAddress(table.get(ADDRESS, 24), "resource release operation");
        requireAddress(table.get(ADDRESS, 32), "resource query_interface operation");
    }

    private static MemorySegment validatedEntriesVtable(MemorySegment address) {
        requireAddress(address, "dictionary entries vtable");
        MemorySegment table = address.reinterpret(InteropLayouts.DICTIONARY_ENTRIES_VTABLE.byteSize());
        if (table.get(JAVA_LONG, 0) < InteropLayouts.DICTIONARY_ENTRIES_VTABLE.byteSize()
                || table.get(JAVA_INT, 8) < InteropLayouts.DICTIONARY_ENTRIES_INTERFACE_VERSION
                || table.get(JAVA_INT, 12) != 0) {
            malformed("dictionary entries vtable header is incompatible");
        }
        for (long offset = 16; offset <= 56; offset += 8) {
            requireAddress(table.get(ADDRESS, offset), "dictionary entries operation");
        }
        return table;
    }

    private static void validateEmptyBatch(MemorySegment batch) {
        if (!batch.get(ADDRESS, BATCH_ENTRIES).equals(MemorySegment.NULL)
                || batch.get(JAVA_LONG, BATCH_ENTRY_COUNT) != 0
                || !batch.get(ADDRESS, BATCH_UNITS).equals(MemorySegment.NULL)
                || batch.get(JAVA_LONG, BATCH_UNIT_COUNT) != 0
                || !batch.get(ADDRESS, BATCH_VALUES).equals(MemorySegment.NULL)
                || batch.get(JAVA_LONG, BATCH_VALUE_COUNT) != 0
                || batch.get(JAVA_LONG, BATCH_GENERATION) != 0
                || batch.get(JAVA_LONG, BATCH_RESERVED) != 0) {
            malformed("End did not return the canonical empty batch");
        }
    }

    private static long positiveCount(long value, String name) {
        if (value <= 0) malformed(name + " must be positive");
        return value;
    }

    private static long nonnegative(long value, String name) {
        if (value < 0) malformed(name + " exceeds the JVM range");
        return value;
    }

    private static void requirePointerCount(MemorySegment pointer, long count, String name) {
        if (pointer.equals(MemorySegment.NULL) != (count == 0)) {
            malformed(name + " pointer/count mismatch");
        }
    }

    private static void requireAddress(MemorySegment address, String name) {
        if (address.equals(MemorySegment.NULL)) malformed(name + " is null");
    }

    private static void check(int status, String operation) {
        if (status != OK) {
            throw new DictionaryInteropException(
                    status,
                    operation + " returned status " + Integer.toUnsignedString(status));
        }
    }

    private static void malformed(String message) {
        throw new DictionaryInteropException(PROVIDER_ERROR, message);
    }

    private static MethodHandle downcall(MemorySegment address, FunctionDescriptor descriptor) {
        requireAddress(address, "native operation");
        return LINKER.downcallHandle(address, descriptor);
    }

    private static int callInt(MethodHandle handle, Object... arguments) {
        try {
            return (int) handle.invokeWithArguments(arguments);
        } catch (Throwable throwable) {
            throw rethrow(throwable);
        }
    }

    private static RuntimeException rethrow(Throwable throwable) {
        return throwable instanceof RuntimeException runtime
                ? runtime
                : new RuntimeException(throwable);
    }

    private static final class State implements Runnable {
        private Arena arena;
        private final MemorySegment cursor;
        private final MemorySegment batch;
        private final MethodHandle next;
        private final MethodHandle release;
        private final MethodHandle cancel;
        private final MethodHandle close;
        private boolean closed;

        State(Arena arena, MemorySegment cursor, MemorySegment vtable) {
            this.arena = arena;
            this.cursor = cursor;
            this.batch = arena.allocate(InteropLayouts.DICTIONARY_ENTRY_BATCH);
            this.next = downcall(
                    vtable.get(ADDRESS, 24),
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
            this.release = downcall(
                    vtable.get(ADDRESS, 32),
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
            this.cancel = downcall(
                    vtable.get(ADDRESS, 48), FunctionDescriptor.of(JAVA_INT, ADDRESS));
            this.close = downcall(
                    vtable.get(ADDRESS, 56), FunctionDescriptor.of(JAVA_INT, ADDRESS));
        }

        synchronized MemorySegment batch() {
            ensureOpen();
            return batch;
        }

        synchronized boolean isClosed() {
            return closed;
        }

        synchronized int next(DictionaryBatchLimits values) {
            ensureOpen();
            try (Arena callArena = Arena.ofConfined()) {
                MemorySegment limits = callArena.allocate(InteropLayouts.DICTIONARY_ENTRY_LIMITS);
                limits.set(JAVA_LONG, 0, values.maxEntries());
                limits.set(JAVA_LONG, 8, values.maxUnits());
                limits.set(JAVA_LONG, 16, values.maxValues());
                limits.set(JAVA_LONG, 24, 0);
                return callInt(next, cursor, limits, batch);
            }
        }

        synchronized int release(long generation) {
            ensureOpen();
            return callInt(release, cursor, generation);
        }

        synchronized void close(boolean cancelFirst) {
            if (closed) return;
            RuntimeException failure = null;
            if (cancelFirst) {
                int status = callInt(cancel, cursor);
                if (status != OK && status != BATCH_IN_USE) {
                    failure = new DictionaryInteropException(
                            status, "cancel dictionary entries cursor failed");
                }
            }
            int closeStatus = callInt(close, cursor);
            if (closeStatus != OK) {
                RuntimeException closeFailure = new DictionaryInteropException(
                        closeStatus, "close dictionary entries cursor failed");
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
            if (closeStatus == OK) {
                closed = true;
                arena.close();
                arena = null;
            }
            if (failure != null) throw failure;
        }

        private void ensureOpen() {
            if (closed) throw new IllegalStateException("dictionary entries cursor is closed");
        }

        @Override
        public synchronized void run() {
            if (closed) return;
            try {
                callInt(cancel, cursor);
                if (callInt(close, cursor) == OK) {
                    closed = true;
                    arena.close();
                    arena = null;
                }
            } catch (RuntimeException ignored) {
                // Cleaner is leak containment; explicit close reports failures.
            }
        }
    }
}
