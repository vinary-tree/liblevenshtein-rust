package io.vinarytree.interop;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/** FFM upcall adapter for a Java-defined Unicode dictionary provider.
 *
 * <p>The capture callback must return an immutable revision in O(1). Consumer
 * calls are serialized unless {@code parallelReentrant} is explicitly enabled.
 */
public final class UnicodeDictionaryResource implements DictionaryResource {
    private static final int OK = 0;
    private static final int NULL_POINTER = 3;
    private static final int UNSUPPORTED = 4;
    private static final int CLOSED = 6;
    private static final int PROVIDER_ERROR = 8;
    private static final int UNIT_UNICODE = 2;
    private static final int VALUE_OPTIONAL_U64 = 1;
    private static final long FLAG_PARALLEL_REENTRANT = 1;
    private static final long FLAG_IMMUTABLE = 4;
    private static final byte[] DICTIONARY_ID =
            "vt.dictionary.v1".getBytes(StandardCharsets.US_ASCII);

    private static final Linker LINKER = Linker.nativeLinker();
    private static final Arena GLOBAL = Arena.global();
    private static final AtomicLong NEXT_KEY = new AtomicLong(1);
    private static final ConcurrentHashMap<Long, Holder> HOLDERS = new ConcurrentHashMap<>();

    private static final long RESOURCE_CONTEXT =
            InteropLayouts.RESOURCE.byteOffset(groupElement("context"));
    private static final long RESOURCE_VTABLE =
            InteropLayouts.RESOURCE.byteOffset(groupElement("vtable"));
    private static final long OPTIONAL_VALUE =
            InteropLayouts.OPTIONAL_U64.byteOffset(groupElement("value"));
    private static final long OPTIONAL_PRESENT =
            InteropLayouts.OPTIONAL_U64.byteOffset(groupElement("has_value"));
    private static final long EDGE_LABEL =
            InteropLayouts.DICTIONARY_EDGE.byteOffset(groupElement("label"));
    private static final long EDGE_NODE =
            InteropLayouts.DICTIONARY_EDGE.byteOffset(groupElement("node"));

    private static final MemorySegment RETAIN = upcall("retain",
            MethodType.methodType(void.class, MemorySegment.class),
            FunctionDescriptor.ofVoid(ADDRESS));
    private static final MemorySegment RELEASE = upcall("release",
            MethodType.methodType(void.class, MemorySegment.class),
            FunctionDescriptor.ofVoid(ADDRESS));
    private static final MemorySegment QUERY_INTERFACE = upcall("queryInterface",
            MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class,
                    int.class, MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
    private static final MemorySegment SNAPSHOT = upcall("snapshot",
            MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    private static final MemorySegment ROOT = upcall("root",
            MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    private static final MemorySegment LEN = upcall("len",
            MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class,
                    MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    private static final MemorySegment IS_FINAL = upcall("isFinal",
            MethodType.methodType(int.class, MemorySegment.class, long.class,
                    MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MemorySegment VALUE = upcall("value",
            MethodType.methodType(int.class, MemorySegment.class, long.class,
                    MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MemorySegment TRANSITION = upcall("transition",
            MethodType.methodType(int.class, MemorySegment.class, long.class, long.class,
                    MemorySegment.class, MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS));
    private static final MemorySegment EDGES = upcall("edges",
            MethodType.methodType(int.class, MemorySegment.class, long.class, long.class,
                    MemorySegment.class, long.class, MemorySegment.class, MemorySegment.class),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS,
                    JAVA_LONG, ADDRESS, ADDRESS));

    private static final MemorySegment BASE_VTABLE = makeBaseVtable();
    private static final MemorySegment MUTABLE_SERIAL = makeDictionaryVtable(false, false);
    private static final MemorySegment MUTABLE_PARALLEL = makeDictionaryVtable(false, true);
    private static final MemorySegment IMMUTABLE_SERIAL = makeDictionaryVtable(true, false);
    private static final MemorySegment IMMUTABLE_PARALLEL = makeDictionaryVtable(true, true);

    private final Arena arena = Arena.ofShared();
    private final Holder holder;
    private final MemorySegment resource;
    private long key;

    /** Create a serialized provider over an O(1) immutable-revision capture. */
    public UnicodeDictionaryResource(Supplier<? extends UnicodeDictionarySnapshot> capture) {
        this(capture, false);
    }

    /** Create a provider, optionally allowing concurrent and reentrant calls. */
    public UnicodeDictionaryResource(
            Supplier<? extends UnicodeDictionarySnapshot> capture,
            boolean parallelReentrant) {
        Objects.requireNonNull(capture, "capture");
        holder = new Holder(() -> Objects.requireNonNull(capture.get(), "captured snapshot"),
                false, parallelReentrant, new Fault());
        key = register(holder);
        resource = arena.allocate(InteropLayouts.RESOURCE);
        writeResource(resource, key);
    }

    /** Last exception converted to {@code VT_STATUS_PROVIDER_ERROR}, if any. */
    public Throwable lastCallbackError() {
        return holder.fault.lastError;
    }

    /** Borrow the two-word resource for one synchronous FFM call. */
    @Override
    public MemorySegment resourceSegment() {
        if (key == 0) {
            throw new IllegalStateException("dictionary resource is closed");
        }
        return resource;
    }

    /** Release this facade's retain; native consumers keep their own retains. */
    @Override
    public void close() {
        if (key != 0) {
            release(MemorySegment.ofAddress(key));
            key = 0;
            arena.close();
        }
    }

    private static long register(Holder holder) {
        long candidate = NEXT_KEY.getAndIncrement();
        if (candidate == 0) {
            candidate = NEXT_KEY.getAndIncrement();
        }
        HOLDERS.put(candidate, holder);
        return candidate;
    }

    private static void retain(MemorySegment context) {
        Holder value = HOLDERS.get(context.address());
        if (value != null) {
            value.references.incrementAndGet();
        }
    }

    private static void release(MemorySegment context) {
        long contextKey = context.address();
        HOLDERS.computeIfPresent(contextKey, (_key, value) ->
                value.references.decrementAndGet() == 0 ? null : value);
    }

    private static int queryInterface(
            MemorySegment context, MemorySegment interfaceId, int minimumVersion,
            MemorySegment output) {
        if (interfaceId.equals(MemorySegment.NULL) || output.equals(MemorySegment.NULL)) {
            return NULL_POINTER;
        }
        Holder value = HOLDERS.get(context.address());
        if (value == null) return CLOSED;
        try {
            byte[] actual = interfaceId.reinterpret(DICTIONARY_ID.length).toArray(JAVA_BYTE);
            if (!Arrays.equals(actual, DICTIONARY_ID)
                    || minimumVersion > InteropLayouts.DICTIONARY_INTERFACE_VERSION) {
                return UNSUPPORTED;
            }
            output.reinterpret(ADDRESS.byteSize()).set(
                    ADDRESS, 0, dictionaryVtable(value.immutable, value.parallel));
            return OK;
        } catch (Throwable throwable) {
            value.fault.lastError = throwable;
            return PROVIDER_ERROR;
        }
    }

    private static int snapshot(MemorySegment context, MemorySegment output) {
        Holder parent = HOLDERS.get(context.address());
        if (parent == null) return CLOSED;
        return status(parent, () -> {
            UnicodeDictionarySnapshot captured = parent.snapshot();
            Holder child = new Holder(() -> captured, true, parent.parallel, parent.fault);
            writeResource(output.reinterpret(InteropLayouts.RESOURCE.byteSize()), register(child));
        });
    }

    private static int root(MemorySegment context, MemorySegment output) {
        Holder value = HOLDERS.get(context.address());
        if (value == null) return CLOSED;
        return status(value, () -> output.reinterpret(JAVA_LONG.byteSize()).set(
                JAVA_LONG, 0, value.snapshot().root()));
    }

    private static int len(MemorySegment context, MemorySegment output, MemorySegment known) {
        Holder value = HOLDERS.get(context.address());
        if (value == null) return CLOSED;
        return status(value, () -> {
            OptionalLong size = value.snapshot().size();
            output.reinterpret(JAVA_LONG.byteSize()).set(JAVA_LONG, 0, size.orElse(0));
            known.reinterpret(JAVA_BYTE.byteSize()).set(JAVA_BYTE, 0,
                    (byte) (size.isPresent() ? 1 : 0));
        });
    }

    private static int isFinal(MemorySegment context, long node, MemorySegment output) {
        Holder value = HOLDERS.get(context.address());
        if (value == null) return CLOSED;
        return status(value, () -> output.reinterpret(JAVA_BYTE.byteSize()).set(
                JAVA_BYTE, 0, (byte) (value.snapshot().isFinal(node) ? 1 : 0)));
    }

    private static int value(MemorySegment context, long node, MemorySegment output) {
        Holder resource = HOLDERS.get(context.address());
        if (resource == null) return CLOSED;
        return status(resource, () -> {
            OptionalLong value = resource.snapshot().value(node);
            MemorySegment target = output.reinterpret(InteropLayouts.OPTIONAL_U64.byteSize());
            target.fill((byte) 0);
            target.set(JAVA_LONG, OPTIONAL_VALUE, value.orElse(0));
            target.set(JAVA_BYTE, OPTIONAL_PRESENT, (byte) (value.isPresent() ? 1 : 0));
        });
    }

    private static int transition(
            MemorySegment context, long node, long label, MemorySegment child,
            MemorySegment found) {
        Holder value = HOLDERS.get(context.address());
        if (value == null) return CLOSED;
        return status(value, () -> {
            long result = 0;
            boolean present = false;
            for (UnicodeDictionarySnapshot.Edge edge : value.snapshot().edges(node)) {
                if (Integer.toUnsignedLong(edge.scalar()) == label) {
                    result = edge.node();
                    present = true;
                    break;
                }
            }
            child.reinterpret(JAVA_LONG.byteSize()).set(JAVA_LONG, 0, result);
            found.reinterpret(JAVA_BYTE.byteSize()).set(JAVA_BYTE, 0,
                    (byte) (present ? 1 : 0));
        });
    }

    private static int edges(
            MemorySegment context, long node, long start, MemorySegment output,
            long capacity, MemorySegment written, MemorySegment total) {
        Holder value = HOLDERS.get(context.address());
        if (value == null) return CLOSED;
        return status(value, () -> {
            List<UnicodeDictionarySnapshot.Edge> values = value.snapshot().edges(node);
            long count = Math.min(Math.max(0, values.size() - start), capacity);
            MemorySegment target = count == 0 || output.equals(MemorySegment.NULL)
                    ? MemorySegment.NULL
                    : output.reinterpret(Math.multiplyExact(count,
                            InteropLayouts.DICTIONARY_EDGE.byteSize()));
            for (long index = 0; index < count; index++) {
                UnicodeDictionarySnapshot.Edge edge = values.get(Math.toIntExact(start + index));
                MemorySegment slot = target.asSlice(
                        index * InteropLayouts.DICTIONARY_EDGE.byteSize(),
                        InteropLayouts.DICTIONARY_EDGE.byteSize());
                slot.set(JAVA_LONG, EDGE_LABEL, Integer.toUnsignedLong(edge.scalar()));
                slot.set(JAVA_LONG, EDGE_NODE, edge.node());
            }
            written.reinterpret(JAVA_LONG.byteSize()).set(JAVA_LONG, 0, count);
            total.reinterpret(JAVA_LONG.byteSize()).set(JAVA_LONG, 0, values.size());
        });
    }

    private static int status(Holder holder, ThrowingRunnable operation) {
        try {
            operation.run();
            return OK;
        } catch (Throwable throwable) {
            holder.fault.lastError = throwable;
            return PROVIDER_ERROR;
        }
    }

    private static MemorySegment upcall(
            String name, MethodType type, FunctionDescriptor descriptor) {
        try {
            MethodHandle handle = MethodHandles.lookup().findStatic(
                    UnicodeDictionaryResource.class, name, type);
            return LINKER.upcallStub(handle, descriptor, GLOBAL);
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }

    private static MemorySegment makeBaseVtable() {
        MemorySegment table = GLOBAL.allocate(InteropLayouts.RESOURCE_VTABLE);
        table.set(JAVA_LONG, 0, InteropLayouts.RESOURCE_VTABLE.byteSize());
        table.set(JAVA_INT, 8, InteropLayouts.ABI_VERSION);
        table.set(JAVA_INT, 12, 0);
        table.set(ADDRESS, 16, RETAIN);
        table.set(ADDRESS, 24, RELEASE);
        table.set(ADDRESS, 32, QUERY_INTERFACE);
        return table;
    }

    private static MemorySegment makeDictionaryVtable(boolean immutable, boolean parallel) {
        MemorySegment table = GLOBAL.allocate(InteropLayouts.DICTIONARY_VTABLE);
        table.fill((byte) 0);
        table.set(JAVA_LONG, 0, InteropLayouts.DICTIONARY_VTABLE.byteSize());
        table.set(JAVA_INT, 8, InteropLayouts.DICTIONARY_INTERFACE_VERSION);
        table.set(JAVA_INT, 12, UNIT_UNICODE);
        table.set(JAVA_INT, 16, VALUE_OPTIONAL_U64);
        table.set(JAVA_LONG, 24,
                (immutable ? FLAG_IMMUTABLE : 0) | (parallel ? FLAG_PARALLEL_REENTRANT : 0));
        table.set(ADDRESS, 32, SNAPSHOT);
        table.set(ADDRESS, 40, ROOT);
        table.set(ADDRESS, 48, LEN);
        table.set(ADDRESS, 56, IS_FINAL);
        table.set(ADDRESS, 64, VALUE);
        table.set(ADDRESS, 72, TRANSITION);
        table.set(ADDRESS, 80, EDGES);
        return table;
    }

    private static MemorySegment dictionaryVtable(boolean immutable, boolean parallel) {
        if (immutable) {
            return parallel ? IMMUTABLE_PARALLEL : IMMUTABLE_SERIAL;
        }
        return parallel ? MUTABLE_PARALLEL : MUTABLE_SERIAL;
    }

    private static void writeResource(MemorySegment target, long context) {
        target.set(ADDRESS, RESOURCE_CONTEXT, MemorySegment.ofAddress(context));
        target.set(ADDRESS, RESOURCE_VTABLE, BASE_VTABLE);
    }

    private static final class Holder {
        private final Supplier<UnicodeDictionarySnapshot> capture;
        private final boolean immutable;
        private final boolean parallel;
        private final Fault fault;
        private final AtomicInteger references = new AtomicInteger(1);

        private Holder(
                Supplier<UnicodeDictionarySnapshot> capture,
                boolean immutable,
                boolean parallel,
                Fault fault) {
            this.capture = capture;
            this.immutable = immutable;
            this.parallel = parallel;
            this.fault = fault;
        }

        private UnicodeDictionarySnapshot snapshot() {
            return capture.get();
        }
    }

    private static final class Fault {
        private volatile Throwable lastError;
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Throwable;
    }
}
