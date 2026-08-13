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
import org.openjdk.jmh.annotations.Warmup;

/**
 * JMH benchmark for legacy liblevenshtein-java 3.0.0. Identical shape to
 * VinaryBench: one invocation = one full pass, lazy Iterables fully drained,
 * folded triple returned against dead-code elimination.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(2)
public class LegacyBench {

    @Param({"standard", "transposition", "merge_and_split"})
    public String algorithm;

    @Param({"1", "2", "3"})
    public int distance;

    @Param({"hits", "std-d1", "std-d2", "std-d3", "tr-d1", "tr-d2", "tr-d3", "oov"})
    public String queryset;

    private LegacyAdapter adapter;
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
        adapter = new LegacyAdapter();
        adapter.buildDictionary(terms, "own");
        adapter.createTransducer(algorithm);
        reference = adapter.pass(queries, distance, true);
    }

    @Benchmark
    public long fullPass() {
        PassResult result = adapter.pass(queries, distance, false);
        if (!result.tripleEquals(reference)) {
            throw new IllegalStateException("nondeterministic result during measurement");
        }
        return result.matches() ^ result.termBytes() ^ result.distanceSum();
    }
}
