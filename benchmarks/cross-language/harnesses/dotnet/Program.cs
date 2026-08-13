// .NET harness for the cross-language benchmark program.
//
// Implements harnesses/common/PROTOCOL.md over the P/Invoke facades
// (VinaryTree.Liblevenshtein + VinaryTree.Libdictenstein, both consuming the
// shared VinaryTree.Interop resource ABI). Fairness notes (PROTOCOL.md §10):
// workstation GC, default tiering (DOTNET_gcServer=0 default, recorded);
// InvariantGlobalization=true is a build-time deviation, recorded in notes.
//
// Clock (PROTOCOL.md §9): System.Diagnostics.Stopwatch —
// Stopwatch.GetTimestamp() ticks converted with ns = ticks * (1e9 / Frequency)
// (on Linux Frequency is 1e9, so the conversion is the identity).

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using VinaryTree.Libdictenstein;
using VinaryTree.Liblevenshtein;

namespace VinaryTree.BenchCross;

internal static class Program
{
    private const ulong FnvOffset = 0xcbf29ce484222325UL;
    private const ulong FnvPrime = 0x100000001b3UL;
    private const int BatchSize = 256; // facade NextBatch capacity (LLEV_DEFAULT_MATCH_BATCH)
    private const double WallCapSeconds = 300.0;
    private const string SampleDefinition =
        "one full pass over the query set; every cursor fully drained and " +
        "(term, distance) materialized";
    private const string LibraryVersion = "0.10.0";

    private static readonly double NsPerTick = 1e9 / Stopwatch.Frequency;

    private static long NowNs()
    {
        long ticks = Stopwatch.GetTimestamp();
        return Stopwatch.Frequency == 1_000_000_000L
            ? ticks
            : (long)Math.Round(ticks * NsPerTick);
    }

    // ------------------------------------------------------------------
    // Checksum primitives (PROTOCOL.md §8) + startup self-test (§2).
    // C# ulong arithmetic is unchecked (wrapping) by default; `unchecked`
    // blocks make the intent explicit.
    // ------------------------------------------------------------------

    private static ulong FnvUpdate(ulong hash, byte value)
    {
        unchecked { return (hash ^ value) * FnvPrime; }
    }

    private static ulong Fnv1a64(ReadOnlySpan<byte> data)
    {
        ulong hash = FnvOffset;
        foreach (byte b in data) hash = FnvUpdate(hash, b);
        return hash;
    }

    private static ulong EntryHash(string term, ulong distance)
    {
        ulong hash = FnvOffset;
        for (int i = 0; i < term.Length; i++)
        {
            char code = term[i];
            if (code > 0x7f) return EntryHashBytes(Encoding.UTF8.GetBytes(term), distance);
            hash = FnvUpdate(hash, (byte)code);
        }
        return FinishEntry(hash, distance);
    }

    private static ulong EntryHashBytes(ReadOnlySpan<byte> utf8, ulong distance)
    {
        ulong hash = FnvOffset;
        foreach (byte b in utf8) hash = FnvUpdate(hash, b);
        return FinishEntry(hash, distance);
    }

    private static ulong FinishEntry(ulong hash, ulong distance)
    {
        hash = FnvUpdate(hash, 0x00); // separator
        for (int i = 0; i < 8; i++) hash = FnvUpdate(hash, (byte)((distance >> (8 * i)) & 0xff)); // LE64
        return hash;
    }

    private static void SelfTest()
    {
        static void Expect(ulong actual, ulong wanted, string label)
        {
            if (actual != wanted)
            {
                throw new InvalidOperationException(
                    $"checksum self-test failed for {label}: got {actual:x16}, want {wanted:x16}");
            }
        }

        Expect(Fnv1a64(ReadOnlySpan<byte>.Empty), 0xcbf29ce484222325UL, "fnv1a64(\"\")");
        Expect(Fnv1a64("a"u8), 0xaf63dc4c8601ec8cUL, "fnv1a64(\"a\")");
        Expect(EntryHash("cat", 1), 0x9697fa3e50464bc4UL, "entry(cat,1)");
        Expect(EntryHash("cat", 0), 0xb592c1475b3595e5UL, "entry(cat,0)");
        Expect(EntryHash("cot", 1), 0xb8acc5d3816bcdeaUL, "entry(cot,1)");
        unchecked
        {
            Expect(EntryHash("cat", 0) + EntryHash("cot", 1), 0x6e3f871adca163cfUL, "checksum{2}");
        }
        Expect(0UL, 0x0000000000000000UL, "checksum{}");
    }

    // ------------------------------------------------------------------
    // CLI (PROTOCOL.md §1)
    // ------------------------------------------------------------------

    private sealed class Args
    {
        public string Mode = "";
        public string? Algorithm;
        public int MaxDistance = -1;
        public string? Dictionary;
        public string? Queries;
        public string Backend = "";
        public string? Out;
        public int Samples = 30;
        public double WarmupSeconds = 3.0;
        public int GateLimit = 200;
        public int Reps = 10;
        public string? Cells;
    }

    private static void Die(string message)
    {
        Console.Error.WriteLine($"bench-cross-dotnet: {message}");
        Environment.Exit(2);
    }

    private static Args ParseArgs(string[] argv)
    {
        var args = new Args();
        for (int i = 0; i < argv.Length; i += 2)
        {
            if (i + 1 >= argv.Length) Die($"flag requires a value: {argv[i]}");
            string flag = argv[i];
            string value = argv[i + 1];
            switch (flag)
            {
                case "--mode": args.Mode = value; break;
                case "--algorithm": args.Algorithm = value; break;
                case "--max-distance": args.MaxDistance = int.Parse(value, CultureInfo.InvariantCulture); break;
                case "--dictionary": args.Dictionary = value; break;
                case "--queries": args.Queries = value; break;
                case "--backend": args.Backend = value; break;
                case "--out": args.Out = value; break;
                case "--samples": args.Samples = int.Parse(value, CultureInfo.InvariantCulture); break;
                case "--warmup-seconds": args.WarmupSeconds = double.Parse(value, CultureInfo.InvariantCulture); break;
                case "--gate-limit": args.GateLimit = int.Parse(value, CultureInfo.InvariantCulture); break;
                case "--reps": args.Reps = int.Parse(value, CultureInfo.InvariantCulture); break;
                case "--cells": args.Cells = value; break;
                default: Die($"unknown flag: {flag}"); break;
            }
        }
        if (args.Mode.Length == 0 || args.Dictionary is null || args.Backend.Length == 0)
        {
            Die("--mode, --dictionary, --backend are required");
        }
        return args;
    }

    private static Algorithm ParseAlgorithm(string name) => name switch
    {
        "standard" => Algorithm.Standard,
        "transposition" => Algorithm.Transposition,
        "merge_and_split" => Algorithm.MergeAndSplit,
        "damerau_levenshtein" => Algorithm.DamerauLevenshtein,
        _ => throw new ArgumentException($"unknown algorithm: {name}"),
    };

    // ------------------------------------------------------------------
    // Input loading (PROTOCOL.md §3)
    // ------------------------------------------------------------------

    private static string[] ReadLines(string path)
    {
        string raw = File.ReadAllText(path, Encoding.UTF8);
        string[] lines = raw.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        if (lines.Length == 0) Die($"{path} contains no lines");
        return lines;
    }

    private static void AssertStrictlySorted(string[] lines, string path)
    {
        for (int i = 0; i + 1 < lines.Length; i++)
        {
            // Ordinal comparison over the ASCII workload equals byte order.
            if (string.CompareOrdinal(lines[i], lines[i + 1]) >= 0)
            {
                Die($"{path} is not strictly byte-sorted at line {i + 1}: " +
                    $"\"{lines[i]}\" >= \"{lines[i + 1]}\"");
            }
        }
    }

    // ------------------------------------------------------------------
    // Dictionary + transducer side (PROTOCOL.md §4–5)
    // ------------------------------------------------------------------

    private readonly record struct Triple(long Matches, long Bytes, long DistanceSum);

    private sealed class Side
    {
        private Libdictenstein.Dictionary? dictionary;
        private Transducer? transducer;
        private KeyValuePair<string, ulong?>[]? preparedEntries;

        public void BuildDictionary(string[] terms, string backend)
        {
            if (preparedEntries is null)
            {
                preparedEntries = new KeyValuePair<string, ulong?>[terms.Length]; // preallocated once
                for (int i = 0; i < terms.Length; i++)
                {
                    preparedEntries[i] = new KeyValuePair<string, ulong?>(terms[i], null);
                }
            }
            switch (backend)
            {
                case "dynamic_dawg":
                {
                    var dawg = new DynamicDawg(UnitDomain.UnicodeScalar);
                    nuint inserted = dawg.PutAll(preparedEntries); // ONE batch call (§4)
                    if (inserted != (nuint)terms.Length)
                    {
                        Die($"batch insert count mismatch: {inserted} != {terms.Length}");
                    }
                    dictionary = dawg;
                    break;
                }
                case "double_array_trie":
                    dictionary = new DoubleArrayTrie(preparedEntries, UnitDomain.UnicodeScalar);
                    break;
                default:
                    Die($"unknown backend: {backend}");
                    break;
            }
        }

        public void FreeDictionary()
        {
            transducer?.Dispose();
            transducer = null;
            dictionary?.Dispose();
            dictionary = null;
        }

        public void CreateTransducer(string algorithm)
        {
            transducer?.Dispose();
            transducer = new Transducer(dictionary!, ParseAlgorithm(algorithm));
        }

        public Triple FullPass(string[] queries, int limit, int maxDistance, ref ulong checksum,
                               bool withChecksum)
        {
            long matches = 0;
            long bytes = 0;
            long distanceSum = 0;
            Transducer machine = transducer!;
            for (int i = 0; i < limit; i++)
            {
                using Query cursor = machine.Query(queries[i], (nuint)maxDistance);
                foreach (Match match in cursor)
                {
                    string term = (string)match.Term;
                    matches += 1;
                    bytes += term.Length; // ASCII workload: UTF-16 length == UTF-8 byte length
                    distanceSum += (long)match.Distance;
                    if (withChecksum)
                    {
                        unchecked { checksum += EntryHash(term, match.Distance); }
                    }
                }
            }
            return new Triple(matches, bytes, distanceSum);
        }
    }

    // ------------------------------------------------------------------
    // Result JSON (PROTOCOL.md §11: harness-filled fields only)
    // ------------------------------------------------------------------

    private sealed class CellOutput
    {
        public required string Mode;
        public required string Structure;
        public required string Algorithm;
        public required int MaxDistance;
        public required string DictionaryFile;
        public required int TermCount;
        public long? ConstructNs;
        public required string QueriesFile;
        public required int QueryCount;
        public int WarmupPasses;
        public required int SamplesRequested;
        public required double WarmupSeconds;
        public long[] SamplesNs = Array.Empty<long>();
        public Triple Triple;
        public ulong Checksum;
        public long[]? ConstructTimes;
        public required string Status;
        public required List<string> Notes;
    }

    private static void WriteResult(string outPath, CellOutput cell)
    {
        string? parent = Path.GetDirectoryName(outPath);
        if (!string.IsNullOrEmpty(parent)) Directory.CreateDirectory(parent);
        using var stream = File.Create(outPath);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

        writer.WriteStartObject();
        writer.WriteString("schema_version", "1.0.0");
        writer.WriteString("suite", "cross-language-v1");
        writer.WriteString("timestamp_utc",
            DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture));

        writer.WriteStartObject("target");
        writer.WriteString("language", "dotnet");
        writer.WriteString("implementation", "vinary-tree");
        writer.WriteString("backend", "pinvoke");
        writer.WriteString("runtime_version", RuntimeInformation.FrameworkDescription);
        writer.WriteString("library_version", LibraryVersion);
        writer.WriteStartObject("artifact");
        writer.WriteString("kind", "local-build");
        writer.WriteString("id", $"VinaryTree.Liblevenshtein@{LibraryVersion}");
        writer.WriteEndObject();
        writer.WriteEndObject();

        writer.WriteStartObject("dictionary");
        writer.WriteString("file", cell.DictionaryFile);
        writer.WriteNumber("term_count", cell.TermCount);
        writer.WriteString("structure", cell.Structure);
        writer.WriteString("unit_domain", "unicode_scalar");
        if (cell.ConstructNs is long constructNs) writer.WriteNumber("construct_ns", constructNs);
        writer.WriteEndObject();

        writer.WriteStartObject("workload");
        writer.WriteString("queryset", Path.GetFileNameWithoutExtension(cell.QueriesFile));
        writer.WriteString("file", cell.QueriesFile);
        writer.WriteNumber("query_count", cell.QueryCount);
        writer.WriteEndObject();

        writer.WriteString("algorithm", cell.Algorithm);
        writer.WriteNumber("max_distance", cell.MaxDistance);
        writer.WriteString("mode", cell.Mode == "memory-child" ? "memory" : cell.Mode);

        writer.WriteStartObject("protocol");
        writer.WriteString("timer", "monotonic");
        writer.WriteString("harness", "self-timed");
        writer.WriteNumber("warmup_seconds_min", cell.WarmupSeconds);
        writer.WriteNumber("warmup_passes", cell.WarmupPasses);
        writer.WriteNumber("samples_requested", cell.SamplesRequested);
        writer.WriteString("sample_definition", SampleDefinition);
        writer.WriteNumber("batch_size", BatchSize);
        writer.WriteNumber("wall_cap_seconds", WallCapSeconds);
        writer.WriteEndObject();

        if (cell.ConstructTimes is long[] constructTimes)
        {
            writer.WriteStartObject("construct");
            writer.WriteNumber("reps", constructTimes.Length);
            writer.WriteStartArray("times_ns");
            foreach (long value in constructTimes) writer.WriteNumberValue(value);
            writer.WriteEndArray();
            writer.WriteNumber("term_count", cell.TermCount);
            writer.WriteEndObject();
        }
        else
        {
            writer.WriteStartObject("measurements");
            writer.WriteStartArray("samples_ns");
            foreach (long value in cell.SamplesNs) writer.WriteNumberValue(value);
            writer.WriteEndArray();
            writer.WriteNumber("sample_count", cell.SamplesNs.Length);
            writer.WriteNumber("matches_per_pass", cell.Triple.Matches);
            writer.WriteNumber("term_bytes_per_pass", cell.Triple.Bytes);
            writer.WriteNumber("distance_sum_per_pass", cell.Triple.DistanceSum);
            writer.WriteString("checksum_hex", cell.Checksum.ToString("x16", CultureInfo.InvariantCulture));
            writer.WriteEndObject();
        }

        writer.WriteString("status", cell.Status);
        writer.WriteStartArray("notes");
        foreach (string note in cell.Notes) writer.WriteStringValue(note);
        writer.WriteEndArray();
        writer.WriteEndObject();
        writer.Flush();
        stream.WriteByte((byte)'\n');
    }

    // ------------------------------------------------------------------
    // Modes (PROTOCOL.md §6)
    // ------------------------------------------------------------------

    private static List<string> BaseNotes(string backend)
    {
        var notes = new List<string>(8)
        {
            "P/Invoke facade (LibraryImport) over the shared C ABI",
            "workstation GC defaults (DOTNET_gcServer=0 default); default tiering",
            "InvariantGlobalization=true (build-time deviation from template defaults)",
            "clock: Stopwatch.GetTimestamp(); ns = ticks * (1e9 / Stopwatch.Frequency)",
        };
        if (backend == "dynamic_dawg")
        {
            notes.Add("dynamic_dawg populated with ONE facade PutAll batch call");
        }
        return notes;
    }

    private static void RunQueryCell(Side side, Args args, string[] queries, string algorithm,
                                     int maxDistance, string queriesPath, string outPath,
                                     string[] terms, long constructNs)
    {
        ulong checksum = 0;
        Triple gate = side.FullPass(queries, queries.Length, maxDistance, ref checksum, true);

        long warmStart = NowNs();
        long warmupNs = (long)(args.WarmupSeconds * 1e9);
        int warmupPasses = 0;
        long lastPassNs = 0;
        ulong ignored = 0;
        while (NowNs() - warmStart < warmupNs || warmupPasses < 2)
        {
            long t0 = NowNs();
            Triple triple = side.FullPass(queries, queries.Length, maxDistance, ref ignored, false);
            lastPassNs = NowNs() - t0;
            if (triple != gate) Die("nondeterministic result during warmup");
            warmupPasses++;
        }

        int sampleCount = args.Samples;
        string status = "ok";
        List<string> notes = BaseNotes(args.Backend);
        double lastPassSeconds = lastPassNs / 1e9;
        if (sampleCount * lastPassSeconds > WallCapSeconds)
        {
            int reduced = Math.Max(10, (int)(WallCapSeconds / lastPassSeconds));
            notes.Add(string.Create(CultureInfo.InvariantCulture,
                $"samples reduced from {sampleCount} to {reduced} by the {WallCapSeconds:0}s wall cap " +
                $"(estimated pass {lastPassSeconds:0.000}s)"));
            sampleCount = reduced;
            status = "degraded";
        }

        long[] samplesNs = new long[sampleCount]; // preallocated (§3.4)
        for (int i = 0; i < sampleCount; i++)
        {
            long t0 = NowNs();
            Triple triple = side.FullPass(queries, queries.Length, maxDistance, ref ignored, false);
            samplesNs[i] = NowNs() - t0;
            if (triple != gate) Die("nondeterministic result during measurement");
        }

        WriteResult(outPath, new CellOutput
        {
            Mode = "query",
            Structure = args.Backend,
            Algorithm = algorithm,
            MaxDistance = maxDistance,
            DictionaryFile = args.Dictionary!,
            TermCount = terms.Length,
            ConstructNs = constructNs,
            QueriesFile = queriesPath,
            QueryCount = queries.Length,
            WarmupPasses = warmupPasses,
            SamplesRequested = args.Samples,
            WarmupSeconds = args.WarmupSeconds,
            SamplesNs = samplesNs,
            Triple = gate,
            Checksum = checksum,
            Status = status,
            Notes = notes,
        });
    }

    private static int Main(string[] argv)
    {
        SelfTest();
        Args args = ParseArgs(argv);

        string[] terms = ReadLines(args.Dictionary!);
        AssertStrictlySorted(terms, args.Dictionary!);
        var side = new Side();

        if (args.Mode == "construct")
        {
            if (args.Out is null) Die("--out is required for construct mode");
            side.BuildDictionary(terms, args.Backend); // warmup build (also runs allocator/JIT warm)
            side.FreeDictionary();
            long[] times = new long[args.Reps]; // preallocated
            for (int r = 0; r < args.Reps; r++)
            {
                long t0 = NowNs();
                side.BuildDictionary(terms, args.Backend);
                times[r] = NowNs() - t0;
                side.FreeDictionary();
            }
            List<string> notes = BaseNotes(args.Backend);
            notes.Add("construct mode: timed region is the build from the pre-sorted in-memory list only");
            WriteResult(args.Out!, new CellOutput
            {
                Mode = "construct",
                Structure = args.Backend,
                Algorithm = "standard",
                MaxDistance = 1,
                DictionaryFile = args.Dictionary!,
                TermCount = terms.Length,
                ConstructNs = null,
                QueriesFile = args.Queries ?? "workload/queries/hits.txt",
                QueryCount = 1,
                WarmupPasses = 1,
                SamplesRequested = args.Reps,
                WarmupSeconds = args.WarmupSeconds,
                ConstructTimes = times,
                Status = "ok",
                Notes = notes,
            });
            return 0;
        }

        long buildStart = NowNs();
        side.BuildDictionary(terms, args.Backend);
        long constructNs = NowNs() - buildStart;

        void RunOne(string algorithm, int maxDistance, string queriesPath, string outPath)
        {
            side.CreateTransducer(algorithm);
            string[] queries = ReadLines(queriesPath);
            switch (args.Mode)
            {
                case "verify":
                {
                    int limit = Math.Min(args.GateLimit, queries.Length);
                    ulong checksum = 0;
                    Triple triple = side.FullPass(queries, limit, maxDistance, ref checksum, true);
                    WriteResult(outPath, new CellOutput
                    {
                        Mode = "verify",
                        Structure = args.Backend,
                        Algorithm = algorithm,
                        MaxDistance = maxDistance,
                        DictionaryFile = args.Dictionary!,
                        TermCount = terms.Length,
                        ConstructNs = constructNs,
                        QueriesFile = queriesPath,
                        QueryCount = limit,
                        WarmupPasses = 0,
                        SamplesRequested = 0,
                        WarmupSeconds = args.WarmupSeconds,
                        Triple = triple,
                        Checksum = checksum,
                        Status = "ok",
                        Notes = BaseNotes(args.Backend),
                    });
                    break;
                }
                case "memory-child":
                {
                    ulong checksum = 0;
                    Triple triple = side.FullPass(queries, queries.Length, maxDistance, ref checksum, true);
                    WriteResult(outPath, new CellOutput
                    {
                        Mode = "memory-child",
                        Structure = args.Backend,
                        Algorithm = algorithm,
                        MaxDistance = maxDistance,
                        DictionaryFile = args.Dictionary!,
                        TermCount = terms.Length,
                        ConstructNs = constructNs,
                        QueriesFile = queriesPath,
                        QueryCount = queries.Length,
                        WarmupPasses = 0,
                        SamplesRequested = 0,
                        WarmupSeconds = args.WarmupSeconds,
                        Triple = triple,
                        Checksum = checksum,
                        Status = "ok",
                        Notes = BaseNotes(args.Backend),
                    });
                    break;
                }
                case "query":
                    RunQueryCell(side, args, queries, algorithm, maxDistance, queriesPath, outPath,
                                 terms, constructNs);
                    break;
                default:
                    Die($"unknown mode: {args.Mode}");
                    break;
            }
        }

        if (args.Cells is string cellsPath)
        {
            foreach (string rawLine in File.ReadAllLines(cellsPath))
            {
                string line = rawLine.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                string[] fields = line.Split('\t');
                if (fields.Length != 4) Die($"cells row needs 4 tab-separated fields: {line}");
                RunOne(fields[0], int.Parse(fields[1], CultureInfo.InvariantCulture), fields[2], fields[3]);
            }
        }
        else
        {
            if (args.Algorithm is null || args.MaxDistance < 0 || args.Queries is null || args.Out is null)
            {
                Die("--algorithm, --max-distance, --queries, --out are required");
            }
            RunOne(args.Algorithm!, args.MaxDistance, args.Queries!, args.Out!);
        }

        side.FreeDictionary();
        return 0;
    }
}
