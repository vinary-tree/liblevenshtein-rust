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

    /** Run the complete public-facade fixture, throwing on any contract violation. */
    public static void verify() {
        verifySelectorsErrorsPhoneticsAndLifecycle();
        verifySnapshotIsolation();
        verifyQueryCacheRequiresStableSnapshotIdentity();
    }

    private static void verifyQueryCacheRequiresStableSnapshotIdentity() {
        try (UnicodeDictionaryResource dictionary =
                        new UnicodeDictionaryResource(() -> initial());
                Transducer transducer = new Transducer(dictionary);
                QueryCache cache = new QueryCache(transducer, 4, 4096)) {
            QueryCacheStats empty = cache.stats();
            require(empty.requests() == 0 && empty.residentEntries() == 0,
                    "new query cache did not start empty");
            try (QueryCursor ignored = cache.query("cat", 1)) {
                throw new AssertionError(
                        "cache accepted a provider without stable snapshot identity");
            } catch (NativeException failure) {
                require(failure.status() == Status.UNSUPPORTED,
                        "identity-less provider returned status " + failure.status());
            }
            cache.clear();
            cache.resetStats();
            require(cache.stats().requests() == 0,
                    "query cache counters did not reset");
        }
    }

    private static void verifySnapshotIsolation() {
        AtomicReference<TrieSnapshot> current = new AtomicReference<>(initial());
        List<Match> frozen;
        Match first;
        QueryCursor longLived;

        try (UnicodeDictionaryResource dictionary =
                        new UnicodeDictionaryResource(current::get);
                Transducer transducer = new Transducer(dictionary)) {
            frozen = sorted(drain(transducer.query("cat", 2)));
            // C5: the borrowed-batch reducer (forEachBatch) drains the same
            // query-start snapshot to the same matches as the pull iterator.
            if (!frozen.equals(sorted(reduceDrain(transducer.query("cat", 2))))) {
                throw new AssertionError("reducer drain disagrees with iterator drain");
            }
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

    private static void verifySelectorsErrorsPhoneticsAndLifecycle() {
        AtomicReference<TrieSnapshot> current = new AtomicReference<>(selectorFixture());
        try (UnicodeDictionaryResource dictionary =
                new UnicodeDictionaryResource(current::get)) {
            List<Match> standard = query(
                    dictionary, Algorithm.STANDARD, "ba", 1, QueryOrder.TRAVERSAL);
            require(!contains(standard, "ab", 1),
                    "standard distance unexpectedly accepted a transposition");

            List<Match> transposition = query(
                    dictionary, Algorithm.TRANSPOSITION, "ba", 1, QueryOrder.TRAVERSAL);
            require(contains(transposition, "ab", 1),
                    "transposition algorithm did not accept an adjacent swap");

            List<Match> mergeAndSplit = query(
                    dictionary, Algorithm.MERGE_AND_SPLIT, "ab", 1, QueryOrder.TRAVERSAL);
            require(contains(mergeAndSplit, "c", 1),
                    "merge-and-split algorithm did not accept its distinguishing edit");

            List<Match> damerau = query(
                    dictionary,
                    Algorithm.DAMERAU_LEVENSHTEIN,
                    "ca",
                    2,
                    QueryOrder.TRAVERSAL);
            require(contains(damerau, "abc", 2),
                    "unrestricted Damerau-Levenshtein did not accept its distinguishing edit");

            List<Match> traversal = query(
                    dictionary, Algorithm.STANDARD, "cat", 1, QueryOrder.TRAVERSAL);
            List<Match> ranked = query(
                    dictionary,
                    Algorithm.STANDARD,
                    "cat",
                    1,
                    QueryOrder.DISTANCE_THEN_TERM);
            require(describe(traversal).equals(List.of("bat:1", "cat:0", "cats:1")),
                    "traversal query order changed: " + describe(traversal));
            require(describe(ranked).equals(List.of("cat:0", "bat:1", "cats:1")),
                    "distance-then-term query order changed: " + describe(ranked));

            try (PhoneticPattern regex = PhoneticPattern.compileRegex("c[ao]t")) {
                require(regex.matches("cat"), "regex pattern rejected cat");
                require(!regex.matches("cut"), "regex pattern accepted cut");
            }
            try (PhoneticPattern llre = PhoneticPattern.compileLlre(
                    "@name \"Greeting\"\n^hello$")) {
                require(llre.matches("hello"), "LLRE pattern rejected hello");
                require(!llre.matches("world"), "LLRE pattern accepted world");
            }

            try (PhoneticRuleSet parsed =
                    PhoneticRuleSet.parse("ph -> f\ngh ->\n")) {
                require(parsed.size() == 2, "parsed rule count changed");
                require(parsed.apply("phgh").equals("f"), "parsed rewrite result changed");
            }
            for (PhoneticRuleSetKind kind : PhoneticRuleSetKind.values()) {
                try (PhoneticRuleSet builtin = PhoneticRuleSet.builtin(kind)) {
                    require(builtin.size() > 0, "built-in phonetic rule set is empty: " + kind);
                    require(!builtin.apply("phone").isEmpty(),
                            "built-in phonetic rewrite returned an empty result: " + kind);
                }
            }

            verifyTypedNativeError();
            verifyExplicitLifecycleGuards(dictionary);
        }
    }

    private static List<Match> query(
            UnicodeDictionaryResource dictionary,
            Algorithm algorithm,
            String input,
            long maximumDistance,
            QueryOrder order) {
        try (Transducer transducer = new Transducer(dictionary, algorithm)) {
            return drain(transducer.query(input, maximumDistance, order));
        }
    }

    private static boolean contains(List<Match> matches, String term, long distance) {
        return matches.stream().anyMatch(match ->
                text(match).equals(term) && match.distance() == distance);
    }

    private static List<String> describe(List<Match> matches) {
        return matches.stream()
                .map(match -> text(match) + ":" + match.distance())
                .toList();
    }

    private static void verifyTypedNativeError() {
        try (PhoneticPattern ignored = PhoneticPattern.compileRegex("(")) {
            throw new AssertionError("invalid regex unexpectedly compiled");
        } catch (NativeException failure) {
            require(failure.status() == Status.INVALID_ARGUMENT,
                    "invalid regex returned status " + failure.status());
            require(failure.statusCode() == Status.INVALID_ARGUMENT.code(),
                    "typed and raw native statuses disagree");
            require(failure.getMessage() != null && !failure.getMessage().isBlank(),
                    "native failure omitted its copied diagnostic");
        }
    }

    private static void verifyExplicitLifecycleGuards(
            UnicodeDictionaryResource dictionary) {
        Transducer transducer = new Transducer(dictionary, Algorithm.STANDARD);
        QueryCursor cursor = transducer.query("cat", 1);
        cursor.close();
        cursor.close();
        expectClosed(cursor::hasNext, "query cursor");
        transducer.close();
        transducer.close();
        expectClosed(() -> transducer.query("cat", 1), "transducer");

        PhoneticPattern pattern = PhoneticPattern.compileRegex("cat");
        pattern.close();
        pattern.close();
        expectClosed(() -> pattern.matches("cat"), "phonetic pattern");

        PhoneticRuleSet rules =
                PhoneticRuleSet.builtin(PhoneticRuleSetKind.ENGLISH_ORTHOGRAPHY);
        rules.close();
        rules.close();
        expectClosed(rules::size, "phonetic rule set");
    }

    private static void expectClosed(Runnable operation, String resource) {
        try {
            operation.run();
            throw new AssertionError(resource + " accepted use after close");
        } catch (IllegalStateException expected) {
            require(expected.getMessage() != null && expected.getMessage().contains("closed"),
                    resource + " closed-handle diagnostic is not actionable");
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
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

    /** Drain a cursor through the borrowed-batch reducer, materializing matches. */
    private static List<Match> reduceDrain(QueryCursor cursor) {
        List<Match> result = new ArrayList<>();
        cursor.forEachBatch(batch -> {
            for (int index = 0; index < batch.size(); index++) {
                result.add(batch.get(index).materialize());
            }
        });
        return result;
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

    private static TrieSnapshot selectorFixture() {
        Map<String, OptionalLong> values = new LinkedHashMap<>();
        values.put("ab", OptionalLong.of(1));
        values.put("c", OptionalLong.of(2));
        values.put("abc", OptionalLong.of(3));
        values.put("bat", OptionalLong.of(4));
        values.put("cat", OptionalLong.of(5));
        values.put("cats", OptionalLong.of(6));
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
