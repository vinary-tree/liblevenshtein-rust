package bench;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The shared PROTOCOL.md implementation driving a {@link HarnessAdapter}:
 * CLI contract (§1), startup self-test (§2), input loading (§3), gate pass +
 * triple (§5), all four modes (§6), and the --cells batch driver (§7).
 * JMH owns the pair's primary query timing; the self-timed query mode here
 * completes the CLI contract and backs the gate/construct/memory cells.
 */
public final class ProtocolRunner {
    private static final double WALL_CAP_SECONDS = 300.0;
    private static final String SAMPLE_DEFINITION =
        "one full pass over the query set; every cursor fully drained and "
            + "(term, distance) materialized";

    private final HarnessAdapter adapter;

    public ProtocolRunner(HarnessAdapter adapter) {
        this.adapter = adapter;
    }

    // ------------------------------------------------------------------
    // CLI
    // ------------------------------------------------------------------

    private static final class Args {
        String mode = "";
        String algorithm = null;
        int maxDistance = -1;
        Path dictionary = null;
        Path queries = null;
        String backend = "";
        Path out = null;
        int samples = 30;
        double warmupSeconds = 3.0;
        int gateLimit = 200;
        int reps = 10;
        Path cells = null;
    }

    private static Args parseArgs(String[] argv) {
        Args args = new Args();
        for (int i = 0; i + 1 < argv.length; i += 2) {
            String flag = argv[i];
            String value = argv[i + 1];
            switch (flag) {
                case "--mode" -> args.mode = value;
                case "--algorithm" -> args.algorithm = value;
                case "--max-distance" -> args.maxDistance = Integer.parseInt(value);
                case "--dictionary" -> args.dictionary = Path.of(value);
                case "--queries" -> args.queries = Path.of(value);
                case "--backend" -> args.backend = value;
                case "--out" -> args.out = Path.of(value);
                case "--samples" -> args.samples = Integer.parseInt(value);
                case "--warmup-seconds" -> args.warmupSeconds = Double.parseDouble(value);
                case "--gate-limit" -> args.gateLimit = Integer.parseInt(value);
                case "--reps" -> args.reps = Integer.parseInt(value);
                case "--cells" -> args.cells = Path.of(value);
                default -> throw new IllegalArgumentException("unknown flag: " + flag);
            }
        }
        if (args.mode.isEmpty() || args.dictionary == null || args.backend.isEmpty()) {
            throw new IllegalArgumentException("--mode, --dictionary, --backend are required");
        }
        return args;
    }

    // ------------------------------------------------------------------
    // Entry point
    // ------------------------------------------------------------------

    public int run(String[] argv) {
        Fnv.selfTest();
        Args args = parseArgs(argv);

        List<String> terms = Workload.readLines(args.dictionary);
        Workload.assertStrictlySorted(terms, args.dictionary);

        switch (args.mode) {
            case "construct" -> {
                requireNonNull(args.out, "--out");
                runConstruct(args, terms);
            }
            case "query", "verify", "memory-child" -> {
                long buildStart = System.nanoTime();
                adapter.buildDictionary(terms, args.backend);
                long constructNs = System.nanoTime() - buildStart;
                if (args.cells != null) {
                    runCellsBatch(args, terms, constructNs);
                } else {
                    requireNonNull(args.algorithm, "--algorithm");
                    requireNonNull(args.queries, "--queries");
                    requireNonNull(args.out, "--out");
                    if (args.maxDistance < 0) {
                        throw new IllegalArgumentException("--max-distance is required");
                    }
                    adapter.createTransducer(args.algorithm);
                    List<String> queries = Workload.readLines(args.queries);
                    runOneCell(args, terms, queries, args.algorithm, args.maxDistance,
                        args.queries, args.out, constructNs);
                }
            }
            default -> throw new IllegalArgumentException("unknown mode: " + args.mode);
        }
        return 0;
    }

    private static void requireNonNull(Object value, String flag) {
        if (value == null) {
            throw new IllegalArgumentException(flag + " is required");
        }
    }

    // ------------------------------------------------------------------
    // Modes
    // ------------------------------------------------------------------

    private void runConstruct(Args args, List<String> terms) {
        adapter.buildDictionary(terms, args.backend);   // warmup build
        HarnessAdapter.ConstructionProof constructionProof =
            adapter.validateDictionary(terms);
        adapter.freeDictionary();
        long[] timesNs = new long[args.reps];
        for (int r = 0; r < args.reps; r++) {
            long start = System.nanoTime();
            adapter.buildDictionary(terms, args.backend);
            timesNs[r] = System.nanoTime() - start;
            adapter.freeDictionary();
        }
        Path queries = args.queries != null ? args.queries : Path.of("workload/queries/hits.txt");
        writeResult(args.out, args, "construct", "standard", 1, queries, 1,
            terms.size(), null, 1, new long[0], null, 0L, timesNs, constructionProof,
            "ok", List.of(
                "construct mode: timed region is the build from the pre-sorted in-memory list only"));
    }

    private void runCellsBatch(Args args, List<String> terms, long constructNs) {
        final List<String[]> rows;
        try {
            rows = new ArrayList<>();
            for (String line : Files.readAllLines(args.cells, StandardCharsets.UTF_8)) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                String[] fields = trimmed.split("\t");
                if (fields.length != 4) {
                    throw new IllegalArgumentException("cells row needs 4 fields: " + line);
                }
                rows.add(fields);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + args.cells, e);
        }
        for (String[] row : rows) {
            String algorithm = row[0];
            int maxDistance = Integer.parseInt(row[1]);
            Path queriesPath = Path.of(row[2]);
            Path outPath = Path.of(row[3]);
            adapter.createTransducer(algorithm);
            List<String> queries = Workload.readLines(queriesPath);
            runOneCell(args, terms, queries, algorithm, maxDistance, queriesPath, outPath,
                constructNs);
        }
    }

    private void runOneCell(Args args, List<String> terms, List<String> queries,
                            String algorithm, int maxDistance, Path queriesPath, Path outPath,
                            long constructNs) {
        switch (args.mode) {
            case "verify" -> {
                int limit = Math.min(args.gateLimit, queries.size());
                List<String> subset = queries.subList(0, limit);
                PassResult gateLimited = adapter.pass(subset, maxDistance, true);
                writeResult(outPath, args, "verify", algorithm, maxDistance, queriesPath, limit,
                    terms.size(), constructNs, 0, new long[0], gateLimited,
                    gateLimited.checksum(), null, null, "ok", List.of());
            }
            case "memory-child" -> {
                PassResult gate = adapter.pass(queries, maxDistance, true);
                writeResult(outPath, args, "memory", algorithm, maxDistance,
                    queriesPath, queries.size(), terms.size(), constructNs, 0, new long[0], gate,
                    gate.checksum(), null, null, "ok", List.of());
            }
            case "query" -> {
                PassResult gate = adapter.pass(queries, maxDistance, true);
                long warmStart = System.nanoTime();
                long warmupNs = (long) (args.warmupSeconds * 1e9);
                int warmupPasses = 0;
                long lastPassNs = 0;
                while (System.nanoTime() - warmStart < warmupNs || warmupPasses < 2) {
                    long t0 = System.nanoTime();
                    PassResult triple = adapter.pass(queries, maxDistance, false);
                    lastPassNs = System.nanoTime() - t0;
                    assertTriple(triple, gate);
                    warmupPasses++;
                }
                int sampleCount = args.samples;
                String status = "ok";
                List<String> notes = new ArrayList<>(2);
                double lastPassSeconds = lastPassNs / 1e9;
                if (sampleCount * lastPassSeconds > WALL_CAP_SECONDS) {
                    int reduced = Math.max(10, (int) (WALL_CAP_SECONDS / lastPassSeconds));
                    notes.add(String.format(Locale.ROOT,
                        "samples reduced from %d to %d by the %.0fs wall cap (estimated pass %.3fs)",
                        sampleCount, reduced, WALL_CAP_SECONDS, lastPassSeconds));
                    sampleCount = reduced;
                    status = "degraded";
                }
                long[] samplesNs = new long[sampleCount];
                for (int i = 0; i < sampleCount; i++) {
                    long t0 = System.nanoTime();
                    PassResult triple = adapter.pass(queries, maxDistance, false);
                    samplesNs[i] = System.nanoTime() - t0;
                    assertTriple(triple, gate);
                }
                writeResult(outPath, args, "query", algorithm, maxDistance, queriesPath,
                    queries.size(), terms.size(), constructNs, warmupPasses, samplesNs, gate,
                    gate.checksum(), null, null, status, notes);
            }
            default -> throw new IllegalStateException("unreachable mode " + args.mode);
        }
    }

    private static void assertTriple(PassResult actual, PassResult reference) {
        if (!actual.tripleEquals(reference)) {
            throw new IllegalStateException(
                "nondeterministic result: " + actual + " != " + reference);
        }
    }

    // ------------------------------------------------------------------
    // Result JSON (runner post-fills the rest per PROTOCOL.md §11)
    // ------------------------------------------------------------------

    @SuppressWarnings("checkstyle:ParameterNumber")
    private void writeResult(Path outPath, Args args, String mode, String algorithm,
                             int maxDistance, Path queriesPath, int queryCount, int termCount,
                             Long constructNs, int warmupPasses, long[] samplesNs,
                             PassResult gate, long checksum, long[] constructTimes,
                             HarnessAdapter.ConstructionProof constructionProof,
                             String status, List<String> notes) {
        HarnessAdapter.TargetInfo info = adapter.targetInfo(args.backend);
        String queryset = queriesPath.getFileName().toString().replaceFirst("\\.txt$", "");
        StringBuilder json = new StringBuilder(4096);
        json.append("{\n");
        json.append("  \"schema_version\": \"1.0.0\",\n");
        json.append("  \"suite\": \"cross-language-v1\",\n");
        json.append("  \"timestamp_utc\": \"")
            .append(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ROOT)
                .withZone(ZoneOffset.UTC).format(Instant.now()))
            .append("\",\n");
        json.append("  \"target\": {\n");
        json.append("    \"language\": \"java\",\n");
        json.append("    \"implementation\": \"").append(info.implementation()).append("\",\n");
        json.append("    \"backend\": \"").append(info.backendLabel()).append("\",\n");
        json.append("    \"runtime_version\": \"").append(escape(
            System.getProperty("java.vm.name") + " " + System.getProperty("java.version")))
            .append("\",\n");
        json.append("    \"library_version\": \"").append(escape(info.libraryVersion()))
            .append("\",\n");
        json.append("    \"artifact\": { \"kind\": \"").append(info.artifactKind())
            .append("\", \"id\": \"").append(escape(info.artifactId())).append("\" }\n");
        json.append("  },\n");
        json.append("  \"dictionary\": {\n");
        json.append("    \"file\": \"").append(escape(args.dictionary.toString())).append("\",\n");
        json.append("    \"term_count\": ").append(termCount).append(",\n");
        json.append("    \"structure\": \"").append(info.dictionaryStructure()).append("\",\n");
        json.append("    \"unit_domain\": \"").append(info.unitDomain()).append("\"");
        if (constructNs != null) {
            json.append(",\n    \"construct_ns\": ").append(constructNs).append("\n");
        } else {
            json.append("\n");
        }
        json.append("  },\n");
        json.append("  \"workload\": {\n");
        json.append("    \"queryset\": \"").append(escape(queryset)).append("\",\n");
        json.append("    \"file\": \"").append(escape(queriesPath.toString())).append("\",\n");
        json.append("    \"query_count\": ").append(queryCount).append("\n");
        json.append("  },\n");
        json.append("  \"algorithm\": \"").append(algorithm).append("\",\n");
        json.append("  \"max_distance\": ").append(maxDistance).append(",\n");
        json.append("  \"mode\": \"").append(mode).append("\",\n");
        json.append("  \"protocol\": {\n");
        json.append("    \"timer\": \"monotonic\",\n");
        json.append("    \"harness\": \"self-timed\",\n");
        json.append("    \"warmup_seconds_min\": ").append(args.warmupSeconds).append(",\n");
        json.append("    \"warmup_passes\": ").append(warmupPasses).append(",\n");
        json.append("    \"samples_requested\": ")
            .append(mode.equals("construct") ? args.reps : (mode.equals("query") ? args.samples : 0))
            .append(",\n");
        json.append("    \"sample_definition\": \"").append(escape(SAMPLE_DEFINITION))
            .append("\",\n");
        json.append("    \"batch_size\": ").append(
            info.batchSize() == null ? "null" : info.batchSize().toString()).append(",\n");
        json.append("    \"wall_cap_seconds\": ").append((long) WALL_CAP_SECONDS).append("\n");
        json.append("  },\n");
        if (constructTimes != null) {
            json.append("  \"construct\": {\n");
            json.append("    \"reps\": ").append(constructTimes.length).append(",\n");
            json.append("    \"times_ns\": [").append(joinLongs(constructTimes)).append("],\n");
            json.append("    \"term_count\": ").append(termCount).append(",\n");
            json.append("    \"semantic_term_count\": ")
                .append(constructionProof.termCount()).append(",\n");
            json.append("    \"semantic_membership_checks\": ")
                .append(constructionProof.membershipChecks()).append(",\n");
            json.append("    \"semantic_checksum_hex\": \"")
                .append(String.format(Locale.ROOT, "%016x", constructionProof.checksum()))
                .append("\"\n");
            json.append("  },\n");
        } else {
            json.append("  \"measurements\": {\n");
            json.append("    \"samples_ns\": [").append(joinLongs(samplesNs)).append("],\n");
            json.append("    \"sample_count\": ").append(samplesNs.length).append(",\n");
            json.append("    \"matches_per_pass\": ").append(gate.matches()).append(",\n");
            json.append("    \"term_bytes_per_pass\": ").append(gate.termBytes()).append(",\n");
            json.append("    \"distance_sum_per_pass\": ").append(gate.distanceSum()).append(",\n");
            json.append("    \"checksum_hex\": \"")
                .append(String.format(Locale.ROOT, "%016x", checksum)).append("\"\n");
            json.append("  },\n");
        }
        json.append("  \"status\": \"").append(status).append("\",\n");
        json.append("  \"notes\": [");
        for (int i = 0; i < notes.size(); i++) {
            if (i > 0) {
                json.append(", ");
            }
            json.append("\"").append(escape(notes.get(i))).append("\"");
        }
        json.append("]\n}\n");

        try {
            if (outPath.getParent() != null) {
                Files.createDirectories(outPath.getParent());
            }
            Files.writeString(outPath, json.toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot write " + outPath, e);
        }
    }

    private static String joinLongs(long[] values) {
        StringBuilder sb = new StringBuilder(values.length * 12);
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(values[i]);
        }
        return sb.toString();
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }
}
