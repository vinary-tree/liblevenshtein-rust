package io.vinarytree.liblevenshtein;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.vinarytree.interop.UnicodeDictionaryResource;
import io.vinarytree.interop.UnicodeDictionarySnapshot;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

/**
 * C9 leak-discipline suite for the JVM facade.
 *
 * <p>A >=10,000-cycle create/use/free loop must reach a memory steady state.
 * The facade frees native handles deterministically at {@link AutoCloseable}
 * {@code close()} (there is no native leak regardless of GC), so these tests
 * target the managed side: that closed {@link Transducer}/{@link QueryCursor}
 * wrappers become unreachable and are reclaimed (via their {@code Cleaner}
 * registrations and {@link WeakReference} tracking after {@code System.gc()}),
 * and that the JVM heap does not drift upward across the cycles.
 */
final class LeakDisciplineTest {
    private static final int CYCLES = 10_000;
    private static final int WARMUP = 2_000;
    // A per-cycle handle/retain leak would accrue hundreds of MiB over 10k
    // cycles; this ceiling is generous against GC/heap-measurement noise while
    // still catching a gross leak.
    private static final long MAX_HEAP_GROWTH_BYTES = 32L * 1024 * 1024;

    @Test
    void tenThousandClosedCyclesLeaveNoLiveWrappers() {
        TrieSnapshot snapshot = fixture();
        List<WeakReference<Object>> refs = new ArrayList<>(2 * CYCLES);
        for (int i = 0; i < CYCLES; i++) {
            UnicodeDictionaryResource dictionary =
                    new UnicodeDictionaryResource(() -> snapshot);
            Transducer transducer = new Transducer(dictionary);
            QueryCursor cursor = transducer.query("cat", 2);
            cursor.forEachRemaining(match -> {});
            cursor.close();
            transducer.close();
            dictionary.close();
            refs.add(new WeakReference<>(transducer));
            refs.add(new WeakReference<>(cursor));
        }
        long live = countLiveAfterGc(refs);
        assertEquals(0L, live,
                live + " facade wrappers survived close()+GC across " + CYCLES + " cycles");
    }

    @Test
    void tenThousandIteratorCyclesReachHeapSteadyState() {
        TrieSnapshot snapshot = fixture();
        Runnable cycle = () -> {
            try (UnicodeDictionaryResource dictionary =
                            new UnicodeDictionaryResource(() -> snapshot);
                    Transducer transducer = new Transducer(dictionary);
                    QueryCursor cursor = transducer.query("cat", 2)) {
                cursor.forEachRemaining(match -> {});
            }
        };
        assertHeapSteadyState(cycle);
    }

    @Test
    void tenThousandReducerCyclesReachHeapSteadyState() {
        TrieSnapshot snapshot = fixture();
        Runnable cycle = () -> {
            try (UnicodeDictionaryResource dictionary =
                            new UnicodeDictionaryResource(() -> snapshot);
                    Transducer transducer = new Transducer(dictionary)) {
                int[] total = {0};
                transducer.query("cat", 2).forEachBatch(batch -> total[0] += batch.size());
            }
        };
        assertHeapSteadyState(cycle);
    }

    private static void assertHeapSteadyState(Runnable cycle) {
        for (int i = 0; i < WARMUP; i++) {
            cycle.run();
        }
        long base = settledUsedHeap();
        for (int i = 0; i < CYCLES; i++) {
            cycle.run();
        }
        long growth = settledUsedHeap() - base;
        assertTrue(growth < MAX_HEAP_GROWTH_BYTES,
                "heap grew " + growth + " bytes over " + CYCLES + " cycles");
    }

    private static long settledUsedHeap() {
        long used = Long.MAX_VALUE;
        for (int i = 0; i < 6; i++) {
            System.gc();
            sleep(20);
            Runtime runtime = Runtime.getRuntime();
            used = Math.min(used, runtime.totalMemory() - runtime.freeMemory());
        }
        return used;
    }

    private static long countLiveAfterGc(List<WeakReference<Object>> refs) {
        long live = refs.size();
        for (int spin = 0; spin < 50 && live > 0; spin++) {
            System.gc();
            // Allocation pressure nudges the collector to process weak refs.
            byte[] pressure = new byte[1 << 20];
            pressure[0] = 1;
            sleep(10);
            live = refs.stream().filter(reference -> reference.get() != null).count();
        }
        return live;
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
        }
    }

    private static TrieSnapshot fixture() {
        Map<String, OptionalLong> values = new LinkedHashMap<>();
        values.put("cat", OptionalLong.of(1));
        values.put("cot", OptionalLong.of(2));
        values.put("cut", OptionalLong.of(3));
        values.put("scat", OptionalLong.empty());
        return new TrieSnapshot(values);
    }

    private static final class TrieSnapshot implements UnicodeDictionarySnapshot {
        private final List<Node> nodes = new ArrayList<>();
        private final long terms;

        private TrieSnapshot(Map<String, OptionalLong> entries) {
            nodes.add(new Node());
            entries.forEach((term, value) -> {
                int node = 0;
                for (int scalar : term.codePoints().toArray()) {
                    Integer child = nodes.get(node).edges.get(scalar);
                    if (child == null) {
                        child = nodes.size();
                        nodes.get(node).edges.put(scalar, child);
                        nodes.add(new Node());
                    }
                    node = child;
                }
                nodes.get(node).terminal = true;
                nodes.get(node).value = value;
            });
            terms = entries.size();
        }

        @Override
        public long root() {
            return 0;
        }

        @Override
        public OptionalLong size() {
            return OptionalLong.of(terms);
        }

        @Override
        public boolean isFinal(long node) {
            return nodes.get(Math.toIntExact(node)).terminal;
        }

        @Override
        public OptionalLong value(long node) {
            return nodes.get(Math.toIntExact(node)).value;
        }

        @Override
        public List<Edge> edges(long node) {
            return nodes.get(Math.toIntExact(node)).edges.entrySet().stream()
                    .map(entry -> new Edge(entry.getKey(), entry.getValue()))
                    .toList();
        }
    }

    private static final class Node {
        private final TreeMap<Integer, Integer> edges = new TreeMap<>();
        private boolean terminal;
        private OptionalLong value = OptionalLong.empty();
    }
}
