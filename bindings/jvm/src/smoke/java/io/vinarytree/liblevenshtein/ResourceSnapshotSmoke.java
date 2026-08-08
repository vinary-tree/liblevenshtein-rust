package io.vinarytree.liblevenshtein;

import io.vinarytree.interop.UnicodeDictionaryResource;
import io.vinarytree.interop.UnicodeDictionarySnapshot;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicReference;

/** Standalone and JUnit-reused FFM provider snapshot conformance fixture. */
public final class ResourceSnapshotSmoke {
    private ResourceSnapshotSmoke() {}

    /** Run the long-lived cursor fixture, throwing on any contract violation. */
    public static void verify() {
        AtomicReference<TrieSnapshot> current = new AtomicReference<>(initial());
        List<Match> frozen;
        Match first;
        QueryCursor longLived;

        try (UnicodeDictionaryResource dictionary =
                        new UnicodeDictionaryResource(current::get);
                Transducer transducer = new Transducer(dictionary)) {
            frozen = sorted(drain(transducer.query("cat", 2)));
            longLived = transducer.query("cat", 2);
            first = longLived.next();

            // Publish remove, value update, insert, and compact/checkpoint-
            // equivalent immutable revisions after partial consumption.
            current.set(mutated());
            current.set(mutated().compactedCopy());
        }

        List<Match> observed = new ArrayList<>();
        observed.add(first);
        observed.addAll(drain(longLived));
        if (!frozen.equals(sorted(observed))) {
            throw new AssertionError("long-lived cursor changed after mutation");
        }

        try (UnicodeDictionaryResource dictionary =
                        new UnicodeDictionaryResource(current::get);
                Transducer transducer = new Transducer(dictionary)) {
            if (frozen.equals(sorted(drain(transducer.query("cat", 2))))) {
                throw new AssertionError("fresh cursor did not observe the new revision");
            }
        }
    }

    /** Run from CI without requiring a test framework on the smoke classpath. */
    public static void main(String[] arguments) {
        verify();
    }

    private static List<Match> drain(QueryCursor cursor) {
        try (cursor) {
            List<Match> result = new ArrayList<>();
            cursor.forEachRemaining(result::add);
            return result;
        }
    }

    private static List<Match> sorted(List<Match> values) {
        return values.stream().sorted(Comparator.comparing(ResourceSnapshotSmoke::text)).toList();
    }

    private static String text(Match match) {
        return ((Term.Utf8) match.term()).value();
    }

    private static TrieSnapshot initial() {
        Map<String, OptionalLong> values = new LinkedHashMap<>();
        values.put("cat", OptionalLong.of(1));
        values.put("cot", OptionalLong.of(2));
        values.put("cut", OptionalLong.of(3));
        values.put("scat", OptionalLong.empty());
        return new TrieSnapshot(values);
    }

    private static TrieSnapshot mutated() {
        Map<String, OptionalLong> values = new LinkedHashMap<>();
        values.put("cat", OptionalLong.of(1));
        values.put("cit", OptionalLong.of(5));
        values.put("cut", OptionalLong.of(30));
        values.put("scat", OptionalLong.empty());
        return new TrieSnapshot(values);
    }

    private static final class TrieSnapshot implements UnicodeDictionarySnapshot {
        private final List<Node> nodes = new ArrayList<>();
        private final long terms;

        private TrieSnapshot(Map<String, OptionalLong> entries) {
            nodes.add(new Node());
            entries.forEach((term, id) -> {
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
                nodes.get(node).value = id;
            });
            terms = entries.size();
        }

        private TrieSnapshot compactedCopy() {
            Map<String, OptionalLong> entries = new LinkedHashMap<>();
            collect(0, new StringBuilder(), entries);
            return new TrieSnapshot(entries);
        }

        private void collect(long node, StringBuilder prefix, Map<String, OptionalLong> output) {
            Node current = nodes.get(Math.toIntExact(node));
            if (current.terminal) {
                output.put(prefix.toString(), current.value);
            }
            current.edges.forEach((scalar, child) -> {
                int length = prefix.length();
                prefix.appendCodePoint(scalar);
                collect(child, prefix, output);
                prefix.setLength(length);
            });
        }

        @Override public long root() { return 0; }
        @Override public OptionalLong size() { return OptionalLong.of(terms); }
        @Override public boolean isFinal(long node) {
            return nodes.get(Math.toIntExact(node)).terminal;
        }
        @Override public OptionalLong value(long node) {
            return nodes.get(Math.toIntExact(node)).value;
        }
        @Override public List<Edge> edges(long node) {
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
