import com.github.liblevenshtein.transducer.Algorithm;
import com.github.liblevenshtein.transducer.Candidate;
import com.github.liblevenshtein.transducer.ITransducer;
import com.github.liblevenshtein.transducer.UnsubsumeFunction;
import com.github.liblevenshtein.transducer.factory.TransducerBuilder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Counts actual published-Java subsumption calls on a specified workload. */
public final class LegacySubsumptionProbe {
    private LegacySubsumptionProbe() {}

    private static Algorithm algorithm(String value) {
        switch (value) {
            case "standard":
                return Algorithm.STANDARD;
            case "transposition":
                return Algorithm.TRANSPOSITION;
            case "merge_and_split":
                return Algorithm.MERGE_AND_SPLIT;
            default:
                throw new IllegalArgumentException("unsupported algorithm: " + value);
        }
    }

    public static void main(String[] argv) throws Exception {
        if (argv.length != 4) {
            throw new IllegalArgumentException(
                "usage: LegacySubsumptionProbe <sorted-dictionary> <queries> <algorithm> <distance>");
        }

        List<String> terms = Files.readAllLines(Path.of(argv[0]));
        terms.removeIf(String::isEmpty);
        List<String> queries = Files.readAllLines(Path.of(argv[1]));
        queries.removeIf(String::isEmpty);
        int distance = Integer.parseInt(argv[3]);
        ITransducer<Candidate> transducer = new TransducerBuilder()
            .dictionary(terms, true)
            .algorithm(algorithm(argv[2]))
            .defaultMaxDistance(distance)
            .includeDistance(true)
            .build();

        UnsubsumeFunction.resetCounters();
        long matches = 0;
        long termBytes = 0;
        long distanceSum = 0;
        for (String query : queries) {
            for (Candidate candidate : transducer.transduce(query, distance)) {
                matches++;
                termBytes += candidate.term().getBytes(StandardCharsets.UTF_8).length;
                distanceSum += candidate.distance();
            }
        }

        System.out.printf(
            "{\n"
                + "  \"schema\": \"liblevenshtein.legacy-java-subsumption.v1\",\n"
                + "  \"algorithm\": \"%s\",\n"
                + "  \"max_distance\": %d,\n"
                + "  \"term_count\": %d,\n"
                + "  \"query_count\": %d,\n"
                + "  \"matches\": %d,\n"
                + "  \"term_bytes\": %d,\n"
                + "  \"distance_sum\": %d,\n"
                + "  \"unsubsume_calls\": %d,\n"
                + "  \"outer_positions\": %d,\n"
                + "  \"subsumption_comparisons\": %d,\n"
                + "  \"removed_positions\": %d\n"
                + "}\n",
            argv[2], distance, terms.size(), queries.size(), matches, termBytes, distanceSum,
            UnsubsumeFunction.calls(), UnsubsumeFunction.outerPositions(),
            UnsubsumeFunction.comparisons(), UnsubsumeFunction.removals());
    }
}
