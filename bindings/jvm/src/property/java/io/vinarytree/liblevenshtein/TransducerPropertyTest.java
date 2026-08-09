package io.vinarytree.liblevenshtein;

import io.vinarytree.interop.UnicodeDictionaryResource;
import io.vinarytree.interop.UnicodeDictionarySnapshot;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.TreeMap;
import net.jqwik.api.Arbitraries;
import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;
import net.jqwik.api.constraints.IntRange;

/**
 * C8 property-based tests for the JVM facade, driven by jqwik.
 *
 * <p>Every property is checked against an in-language brute-force Levenshtein
 * oracle over a random dictionary, so a native regression -- or a marshalling
 * bug in this facade -- shrinks to a minimal counterexample instead of drifting
 * silently. The facade is transducer-only (it exposes no standalone distance
 * entry point), so {@code d(a,b)} is realized through a singleton-dictionary
 * query whose bound {@code max(|a|,|b|)} provably admits the peer.
 *
 * <p>jqwik runs on an isolated JUnit Platform 1.x source set ({@code property})
 * because the Jupiter {@code test} task is pinned to JUnit Platform 6, which
 * jqwik does not yet support. Each property carries a fixed {@code seed} for
 * reproducibility; {@code jqwik.properties} disables the on-disk sample database
 * so no stray corpus file is produced.
 */
class TransducerPropertyTest {

    // "é" (U+00E9) is one Unicode scalar but two UTF-8 bytes: mixing it in
    // proves the facade counts scalars, not bytes, end to end.
    private static final char[] ALPHABET = {'a', 'b', 'c', 'é'};

    @Provide
    Arbitrary<String> terms() {
        return Arbitraries.strings().withChars(ALPHABET).ofMinLength(0).ofMaxLength(6);
    }

    @Provide
    Arbitrary<Map<String, Optional<Long>>> dictionaries() {
        return Arbitraries.maps(terms(), Arbitraries.longs().optional()).ofMaxSize(8);
    }

    // --- (b) result set equals the brute-force oracle -----------------------

    @Property(tries = 200, seed = "20260809")
    void queryResultSetEqualsOracle(
            @ForAll("dictionaries") Map<String, Optional<Long>> dictionary,
            @ForAll("terms") String query,
            @ForAll @IntRange(min = 0, max = 3) int k) {
        Map<String, OptionalLong> entries = toEntries(dictionary);
        Map<String, OptionalLong> expected = new LinkedHashMap<>();
        entries.forEach((term, value) -> {
            if (levenshtein(query, term) <= k) {
                expected.put(term, value);
            }
        });

        Map<String, Match> got = query(entries, query, k, QueryOrder.TRAVERSAL);

        check(got.keySet().equals(expected.keySet()),
                "result set " + got.keySet() + " != oracle " + expected.keySet());
        got.forEach((term, match) -> {
            check(match.distance() == levenshtein(query, term),
                    "distance for " + term + ": " + match.distance()
                            + " != " + levenshtein(query, term)); // (a) exact distance
            check(match.id().equals(expected.get(term)),
                    "value for " + term + ": " + match.id() + " != " + expected.get(term)); // (c)
        });
    }

    // --- (a) symmetry and identity ------------------------------------------

    @Property(tries = 200, seed = "20260809")
    void distanceIsSymmetricWithZeroIdentity(
            @ForAll("terms") String a, @ForAll("terms") String b) {
        long forward = transducerDistance(a, b);
        long backward = transducerDistance(b, a);
        check(forward == backward, "asymmetry: d(" + a + "," + b + ")=" + forward
                + " but d(" + b + "," + a + ")=" + backward);
        check(forward == levenshtein(a, b),
                "facade " + forward + " != oracle " + levenshtein(a, b));
        check(transducerDistance(a, a) == 0, "identity d(a,a) != 0 for " + a);
    }

    // --- (a) threshold consistency ------------------------------------------

    @Property(tries = 200, seed = "20260809")
    void thresholdIsConsistentWithDistance(
            @ForAll("terms") String a,
            @ForAll("terms") String b,
            @ForAll @IntRange(min = 0, max = 3) int k) {
        int distance = levenshtein(a, b);
        Map<String, OptionalLong> one = new LinkedHashMap<>();
        one.put(b, OptionalLong.of(0));
        Map<String, Match> got = query(one, a, k, QueryOrder.TRAVERSAL);
        if (distance <= k) {
            check(got.containsKey(b) && got.get(b).distance() == distance,
                    "within-threshold peer " + b + " missing/wrong at k=" + k);
        } else {
            check(!got.containsKey(b), "over-threshold peer " + b + " leaked at k=" + k);
        }
    }

    // --- (c) u64 value channel round-trips (edge cases exercised by jqwik) ---

    @Property(tries = 200, seed = "20260809")
    void u64ValueRoundTrips(@ForAll long value) {
        // jqwik's default long edge cases include 0, 1, -1 (u64 MAX),
        // Long.MIN_VALUE (u64 2^63) and Long.MAX_VALUE, so the boundaries are
        // always covered; the native id is a raw 64-bit pattern.
        Map<String, OptionalLong> one = new LinkedHashMap<>();
        one.put("term", OptionalLong.of(value));
        Map<String, Match> got = query(one, "term", 0, QueryOrder.TRAVERSAL);
        check(got.size() == 1, "expected exactly one match, got " + got.size());
        check(got.get("term").id().equals(OptionalLong.of(value)),
                "value " + value + " did not round-trip: " + got.get("term").id());
    }

    // --- bonus: distance-then-term ordering is monotone ---------------------

    @Property(tries = 200, seed = "20260809")
    void distanceThenTermOrderingIsMonotone(
            @ForAll("dictionaries") Map<String, Optional<Long>> dictionary,
            @ForAll("terms") String query,
            @ForAll @IntRange(min = 0, max = 3) int k) {
        Map<String, OptionalLong> entries = toEntries(dictionary);
        TrieSnapshot snapshot = new TrieSnapshot(entries);
        List<long[]> observed = new ArrayList<>();
        List<String> observedTerms = new ArrayList<>();
        try (UnicodeDictionaryResource dictionaryResource =
                        new UnicodeDictionaryResource(() -> snapshot);
                Transducer transducer = new Transducer(dictionaryResource);
                QueryCursor cursor =
                        transducer.query(query, k, QueryOrder.DISTANCE_THEN_TERM)) {
            for (Match match : cursor) {
                observed.add(new long[] {match.distance()});
                observedTerms.add(text(match));
            }
        }
        for (int i = 1; i < observed.size(); i++) {
            long previous = observed.get(i - 1)[0];
            long current = observed.get(i)[0];
            check(previous <= current
                            || previous == current
                                    && observedTerms.get(i - 1).compareTo(observedTerms.get(i)) <= 0,
                    "ordering violated at index " + i);
        }
    }

    // --- helpers ------------------------------------------------------------

    private static Map<String, OptionalLong> toEntries(Map<String, Optional<Long>> dictionary) {
        Map<String, OptionalLong> entries = new LinkedHashMap<>();
        dictionary.forEach((term, value) ->
                entries.put(term, value.map(OptionalLong::of).orElse(OptionalLong.empty())));
        return entries;
    }

    private static Map<String, Match> query(
            Map<String, OptionalLong> entries, String query, long k, QueryOrder order) {
        TrieSnapshot snapshot = new TrieSnapshot(entries);
        Map<String, Match> got = new LinkedHashMap<>();
        try (UnicodeDictionaryResource dictionary =
                        new UnicodeDictionaryResource(() -> snapshot);
                Transducer transducer = new Transducer(dictionary);
                QueryCursor cursor = transducer.query(query, k, order)) {
            for (Match match : cursor) {
                got.put(text(match), match);
            }
        }
        return got;
    }

    private static long transducerDistance(String a, String b) {
        Map<String, OptionalLong> one = new LinkedHashMap<>();
        one.put(b, OptionalLong.of(0));
        long bound = Math.max(a.length(), b.length());
        Map<String, Match> got = query(one, a, bound, QueryOrder.TRAVERSAL);
        Match match = got.get(b);
        if (match == null) {
            throw new AssertionError("peer " + b + " not within bound " + bound + " of " + a);
        }
        return match.distance();
    }

    private static String text(Match match) {
        return ((Term.Utf8) match.term()).value();
    }

    private static int levenshtein(String a, String b) {
        if (a.equals(b)) {
            return 0;
        }
        int la = a.length();
        int lb = b.length();
        if (la == 0) {
            return lb;
        }
        if (lb == 0) {
            return la;
        }
        int[] previous = new int[lb + 1];
        for (int j = 0; j <= lb; j++) {
            previous[j] = j;
        }
        for (int i = 1; i <= la; i++) {
            int[] current = new int[lb + 1];
            current[0] = i;
            char ca = a.charAt(i - 1);
            for (int j = 1; j <= lb; j++) {
                int cost = ca == b.charAt(j - 1) ? 0 : 1;
                current[j] = Math.min(Math.min(previous[j] + 1, current[j - 1] + 1),
                        previous[j - 1] + cost);
            }
            previous = current;
        }
        return previous[lb];
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    /** Minimal in-language trie oracle over Unicode scalars. */
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

        private Node() {
            Objects.requireNonNull(edges);
        }
    }
}
