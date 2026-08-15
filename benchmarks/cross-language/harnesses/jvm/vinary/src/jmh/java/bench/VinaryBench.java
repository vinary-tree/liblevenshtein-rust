package bench;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * JMH benchmark for the vinary-tree JVM binding. One @Benchmark invocation =
 * one full pass over the query set with every cursor fully drained (the same
 * sample definition as every self-timed harness). The folded triple is
 * returned so the JIT cannot eliminate the drain.
 *
 * The runner drives one cell per JMH invocation via -p parameter overrides;
 * the annotation defaults below only define the valid value spaces.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(2)
public class VinaryBench {

    @Param({"standard", "transposition", "merge_and_split", "damerau_levenshtein"})
    public String algorithm;

    @Param({"1", "2", "3"})
    public int distance;

    @Param({"hits", "std-d1", "std-d2", "std-d3", "tr-d1", "tr-d2", "tr-d3", "oov"})
    public String queryset;

    @Param({"dynamic_dawg", "double_array_trie"})
    public String backend;

    /** Causal result-transport selector; parity runs retain managed materialization. */
    @Param({"materialized"})
    public String resultMode;

    private VinaryAdapter adapter;
    private List<String> queries;
    private PassResult reference;

    @Setup(Level.Trial)
    public void setup() {
        Fnv.selfTest();
        Path workload = Path.of(System.getProperty("xl.workload"));
        Path dictionaryPath = workload.resolve("dictionary.txt");
        List<String> terms = Workload.readLines(dictionaryPath);
        Workload.assertStrictlySorted(terms, dictionaryPath);
        queries = Workload.readLines(workload.resolve("queries").resolve(queryset + ".txt"));
        adapter = new VinaryAdapter();
        adapter.buildDictionary(terms, backend);
        adapter.createTransducer(algorithm);
        reference = adapter.pass(queries, distance, true);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        adapter.freeDictionary();
    }

    @Benchmark
    public long fullPass() {
        PassResult result = switch (resultMode) {
            case "borrowed" -> adapter.passBorrowed(queries, distance, false);
            case "materialized" -> adapter.pass(queries, distance, false);
            default -> throw new IllegalArgumentException("unknown result mode: " + resultMode);
        };
        if (!result.tripleEquals(reference)) {
            throw new IllegalStateException("nondeterministic result during measurement");
        }
        return result.matches() ^ result.termBytes() ^ result.distanceSum();
    }
}
