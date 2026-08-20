package bench;

import com.github.liblevenshtein.collection.dictionary.SortedDawg;
import com.github.liblevenshtein.transducer.Algorithm;
import com.github.liblevenshtein.transducer.Candidate;
import com.github.liblevenshtein.transducer.ITransducer;
import com.github.liblevenshtein.transducer.factory.TransducerBuilder;
import java.util.List;

/**
 * Legacy side: com.github.universal-automata:liblevenshtein:3.0.0 exactly as
 * a 2016-era user runs it — TransducerBuilder with the isSorted=true fast
 * path (legal because the workload's sortedness is asserted at load), lazy
 * Iterable&lt;Candidate&gt; drained fully.
 *
 * The legacy build() couples the transducer to the dictionary, so
 * "construct" here times the whole builder pipeline minus candidate wiring —
 * documented as the legacy dictionary construction cost (SortedDawg build
 * dominates it).
 */
public final class LegacyAdapter implements HarnessAdapter {
    private List<String> terms;
    private SortedDawg dawg;
    private ITransducer<Candidate> transducer;

    @Override
    public void buildDictionary(List<String> terms, String backend) {
        if (!"own".equals(backend)) {
            throw new IllegalArgumentException(
                "legacy target only supports backend 'own', got: " + backend);
        }
        this.terms = terms;
        // The REAL legacy dictionary construction (Daciuk-style incremental
        // SortedDawg over the sorted collection) — this is what construct
        // mode times. new SortedDawg(Collection) is the public API a user
        // calls directly; TransducerBuilder performs the equivalent build
        // internally for the query path.
        this.dawg = new SortedDawg(terms);
    }

    @Override
    public void freeDictionary() {
        dawg = null;         // GC-managed; the legacy API has no close()
        transducer = null;
    }

    @Override
    public ConstructionProof validateDictionary(List<String> expectedTerms) {
        if (dawg == null) {
            throw new IllegalStateException("dictionary has not been built");
        }
        if (dawg.size() != expectedTerms.size()) {
            throw new IllegalStateException(
                "semantic validation failed: dictionary size " + dawg.size()
                    + ", expected " + expectedTerms.size());
        }
        for (String term : expectedTerms) {
            if (!dawg.contains(term)) {
                throw new IllegalStateException(
                    "semantic validation failed: constructed dictionary lost " + term);
            }
        }
        return new ConstructionProof(
            expectedTerms.size(), expectedTerms.size(), Fnv.semanticTerms(expectedTerms));
    }

    @Override
    public void createTransducer(String algorithm) {
        // defaultMaxDistance is irrelevant for the transduce(term, n)
        // overload used by pass(); build with a sane default.
        transducer = buildFor(algorithm, 2);
    }

    /** The full legacy builder pipeline (SortedDawg + automaton wiring). */
    public ITransducer<Candidate> buildFor(String algorithm, int defaultMaxDistance) {
        return new TransducerBuilder()
            .dictionary(terms, /* isSorted= */ true)
            .algorithm(parseAlgorithm(algorithm))
            .defaultMaxDistance(defaultMaxDistance)
            .includeDistance(true)
            .build();
    }

    public static Algorithm parseAlgorithm(String algorithm) {
        switch (algorithm) {
            case "standard":
                return Algorithm.STANDARD;
            case "transposition":
                return Algorithm.TRANSPOSITION;
            case "merge_and_split":
                return Algorithm.MERGE_AND_SPLIT;
            default:
                throw new IllegalArgumentException(
                    "legacy 3.0.0 does not support algorithm: " + algorithm);
        }
    }

    @Override
    public PassResult pass(List<String> queries, int maxDistance, boolean withChecksum) {
        long matches = 0;
        long bytes = 0;
        long distanceSum = 0;
        long checksum = 0;
        for (String query : queries) {
            for (Candidate candidate : transducer.transduce(query, maxDistance)) {
                String term = candidate.term();       // published jar: Lombok fluent accessors
                long distance = candidate.distance();
                matches++;
                bytes += term.length();   // ASCII workload: length == UTF-8 bytes
                distanceSum += distance;
                if (withChecksum) {
                    checksum += Fnv.entryAscii(term, distance);
                }
            }
        }
        return new PassResult(matches, bytes, distanceSum, checksum);
    }

    @Override
    public TargetInfo targetInfo(String backend) {
        return new TargetInfo(
            "legacy",
            "jvm-legacy-jar",
            "3.0.0",
            "maven",
            "com.github.universal-automata:liblevenshtein:3.0.0",
            "legacy_sorted_dawg",
            "jvm_string",
            null);
    }
}
