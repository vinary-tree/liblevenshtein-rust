// Go harness for the cross-language benchmark program.
//
// Implements harnesses/common/PROTOCOL.md over the cgo facades
// (bindings/go liblevenshtein + libdictenstein sibling), resolved through the
// harness-local go.work. The runner provides CGO_LDFLAGS at build time and
// LD_LIBRARY_PATH + GOMAXPROCS=1 + GOGC=100 at run time (fairness rule 5).
//
// Dictionary population uses the facade's single batch call
// (DynamicDawg.PutAll -> ldict_dictionary_insert_text_batch).
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	libdictenstein "github.com/vinary-tree/libdictenstein/bindings/go/v4"
	liblevenshtein "github.com/vinary-tree/liblevenshtein-rust/bindings/go/v4"
)

const (
	fnvOffset        = uint64(0xcbf29ce484222325)
	fnvPrime         = uint64(0x100000001b3)
	batchSize        = 256
	wallCapSeconds   = 300.0
	sampleDefinition = "one full pass over the query set; every cursor fully drained and " +
		"(term, distance) materialized"
)

var algorithms = map[string]liblevenshtein.Algorithm{
	"standard":            liblevenshtein.Standard,
	"transposition":       liblevenshtein.Transposition,
	"merge_and_split":     liblevenshtein.MergeAndSplit,
	"damerau_levenshtein": liblevenshtein.DamerauLevenshtein,
}

// ---------------------------------------------------------------------------
// Checksum primitives (PROTOCOL.md section 8) + self-test (section 2)
// ---------------------------------------------------------------------------

func fnvUpdate(hash uint64, b byte) uint64 { return (hash ^ uint64(b)) * fnvPrime }

func fnv1a64(data []byte) uint64 {
	hash := fnvOffset
	for _, b := range data {
		hash = fnvUpdate(hash, b)
	}
	return hash
}

func entryHash(term string, distance uint64) uint64 {
	hash := fnvOffset
	for i := 0; i < len(term); i++ { // Go string bytes ARE the UTF-8 bytes
		hash = fnvUpdate(hash, term[i])
	}
	hash = fnvUpdate(hash, 0x00) // separator
	for i := 0; i < 8; i++ {     // LE64(distance)
		hash = fnvUpdate(hash, byte(distance>>(8*uint(i))))
	}
	return hash
}

func selfTest() {
	expect := func(actual, wanted uint64, label string) {
		if actual != wanted {
			die(fmt.Sprintf("checksum self-test failed for %s: got %016x, want %016x",
				label, actual, wanted))
		}
	}
	expect(fnv1a64(nil), 0xcbf29ce484222325, `fnv1a64("")`)
	expect(fnv1a64([]byte("a")), 0xaf63dc4c8601ec8c, `fnv1a64("a")`)
	expect(entryHash("cat", 1), 0x9697fa3e50464bc4, "entry(cat,1)")
	expect(entryHash("cat", 0), 0xb592c1475b3595e5, "entry(cat,0)")
	expect(entryHash("cot", 1), 0xb8acc5d3816bcdea, "entry(cot,1)")
	expect(entryHash("cat", 0)+entryHash("cot", 1), 0x6e3f871adca163cf, "checksum{2}")
	var empty uint64
	expect(empty, 0x0000000000000000, "checksum{}")
}

// ---------------------------------------------------------------------------
// CLI (PROTOCOL.md section 1)
// ---------------------------------------------------------------------------

func die(message string) {
	fmt.Fprintf(os.Stderr, "bench-cross-go: %s\n", message)
	os.Exit(2)
}

type cliArgs struct {
	mode          string
	algorithm     string
	maxDistance   int
	dictionary    string
	queries       string
	backend       string
	out           string
	samples       int
	warmupSeconds float64
	gateLimit     int
	reps          int
	cells         string
}

func parseArgs(argv []string) cliArgs {
	args := cliArgs{maxDistance: -1, samples: 30, warmupSeconds: 3.0, gateLimit: 200, reps: 10}
	parseInt := func(flag, value string) int {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			die(fmt.Sprintf("%s expects an integer, got %q", flag, value))
		}
		return parsed
	}
	for i := 0; i < len(argv); i += 2 {
		if i+1 >= len(argv) {
			die("flag requires a value: " + argv[i])
		}
		flag, value := argv[i], argv[i+1]
		switch flag {
		case "--mode":
			args.mode = value
		case "--algorithm":
			args.algorithm = value
		case "--max-distance":
			args.maxDistance = parseInt(flag, value)
		case "--dictionary":
			args.dictionary = value
		case "--queries":
			args.queries = value
		case "--backend":
			args.backend = value
		case "--out":
			args.out = value
		case "--samples":
			args.samples = parseInt(flag, value)
		case "--warmup-seconds":
			parsed, err := strconv.ParseFloat(value, 64)
			if err != nil {
				die(fmt.Sprintf("--warmup-seconds expects a number, got %q", value))
			}
			args.warmupSeconds = parsed
		case "--gate-limit":
			args.gateLimit = parseInt(flag, value)
		case "--reps":
			args.reps = parseInt(flag, value)
		case "--cells":
			args.cells = value
		default:
			die("unknown flag: " + flag)
		}
	}
	if args.mode == "" || args.dictionary == "" || args.backend == "" {
		die("--mode, --dictionary, --backend are required")
	}
	return args
}

// ---------------------------------------------------------------------------
// Input loading (PROTOCOL.md section 3)
// ---------------------------------------------------------------------------

func readLines(path string) []string {
	raw, err := os.ReadFile(path)
	if err != nil {
		die("cannot read " + path + ": " + err.Error())
	}
	text := string(raw)
	lines := make([]string, 0, strings.Count(text, "\n")+1) // preallocated once
	start := 0
	for i := 0; i <= len(text); i++ {
		if i == len(text) || text[i] == '\n' {
			if i > start {
				lines = append(lines, text[start:i])
			}
			start = i + 1
		}
	}
	if len(lines) == 0 {
		die(path + " contains no lines")
	}
	return lines
}

func assertStrictlySorted(lines []string, path string) {
	for i := 0; i+1 < len(lines); i++ {
		// Go string comparison is bytewise: exactly the required byte order.
		if lines[i] >= lines[i+1] {
			die(fmt.Sprintf("%s is not strictly byte-sorted at line %d: %q >= %q",
				path, i+1, lines[i], lines[i+1]))
		}
	}
}

// ---------------------------------------------------------------------------
// Dictionary + transducer side (PROTOCOL.md section 4)
// ---------------------------------------------------------------------------

type side struct {
	dictionary      *libdictenstein.Dictionary
	transducer      *liblevenshtein.Transducer
	preparedEntries []libdictenstein.Entry // built once, reused across rebuilds
}

func (s *side) entries(terms []string) []libdictenstein.Entry {
	if s.preparedEntries == nil {
		s.preparedEntries = make([]libdictenstein.Entry, len(terms)) // preallocated once
		for i, term := range terms {
			s.preparedEntries[i] = libdictenstein.Entry{Term: term}
		}
	}
	return s.preparedEntries
}

func (s *side) buildDictionary(terms []string, backend string) {
	switch backend {
	case "dynamic_dawg":
		dawg, err := libdictenstein.NewDynamicDawg(libdictenstein.UnicodeScalarDomain)
		if err != nil {
			die("NewDynamicDawg failed: " + err.Error())
		}
		inserted, err := dawg.PutAll(s.entries(terms)) // ONE batch call (section 4)
		if err != nil {
			die("PutAll failed: " + err.Error())
		}
		if int(inserted) != len(terms) {
			die(fmt.Sprintf("batch insert count mismatch: %d != %d", inserted, len(terms)))
		}
		s.dictionary = dawg.Dictionary
	case "double_array_trie":
		trie, err := libdictenstein.NewDoubleArrayTrie(s.entries(terms), libdictenstein.UnicodeScalarDomain)
		if err != nil {
			die("NewDoubleArrayTrie failed: " + err.Error())
		}
		s.dictionary = trie.Dictionary
	default:
		die("unknown backend: " + backend)
	}
}

func (s *side) freeDictionary() {
	if s.transducer != nil {
		if err := s.transducer.Close(); err != nil {
			die("transducer close failed: " + err.Error())
		}
		s.transducer = nil
	}
	if s.dictionary != nil {
		if err := s.dictionary.Close(); err != nil {
			die("dictionary close failed: " + err.Error())
		}
		s.dictionary = nil
	}
}

func (s *side) createTransducer(algorithm string) {
	if s.transducer != nil {
		if err := s.transducer.Close(); err != nil {
			die("transducer close failed: " + err.Error())
		}
	}
	mapped, ok := algorithms[algorithm]
	if !ok {
		die("unknown algorithm: " + algorithm)
	}
	transducer, err := liblevenshtein.NewTransducer(s.dictionary, mapped)
	if err != nil {
		die("NewTransducer failed: " + err.Error())
	}
	s.transducer = transducer
}

// ---------------------------------------------------------------------------
// Passes (PROTOCOL.md section 5)
// ---------------------------------------------------------------------------

type triple struct{ matches, bytes, distSum uint64 }

// fullPass drains every cursor completely, materializing (term, distance) for
// each match. Timed passes accumulate only the O(1) triple; the checksum is
// computed exclusively in untimed gate/verify contexts.
func (s *side) fullPass(queries []string, maxDistance int, withChecksum bool) (triple, uint64) {
	var result triple
	var checksum uint64
	for _, query := range queries {
		iterator, err := s.transducer.Query(query, uint(maxDistance), liblevenshtein.Traversal)
		if err != nil {
			die("query failed: " + err.Error())
		}
		for {
			match, ok, err := iterator.Next()
			if err != nil {
				_ = iterator.Close()
				die("cursor next failed: " + err.Error())
			}
			if !ok {
				break
			}
			term := match.Text // materialized Go string (UTF-8 bytes)
			result.matches++
			result.bytes += uint64(len(term)) // UTF-8 byte length
			result.distSum += uint64(match.Distance)
			if withChecksum {
				checksum += entryHash(term, uint64(match.Distance))
			}
		}
		if err := iterator.Close(); err != nil {
			die("cursor close failed: " + err.Error())
		}
	}
	return result, checksum
}

// ---------------------------------------------------------------------------
// Result JSON (PROTOCOL.md section 11 — runner post-fills run_id, sha256s,
// cell_snapshot, environment_ref, and the memory object)
// ---------------------------------------------------------------------------

type artifactJSON struct {
	Kind string `json:"kind"`
	ID   string `json:"id"`
}

type targetJSON struct {
	Language       string       `json:"language"`
	Implementation string       `json:"implementation"`
	Backend        string       `json:"backend"`
	RuntimeVersion string       `json:"runtime_version"`
	LibraryVersion string       `json:"library_version"`
	Artifact       artifactJSON `json:"artifact"`
}

type dictionaryJSON struct {
	File        string `json:"file"`
	TermCount   int    `json:"term_count"`
	Structure   string `json:"structure"`
	UnitDomain  string `json:"unit_domain"`
	ConstructNs *int64 `json:"construct_ns,omitempty"`
}

type workloadJSON struct {
	Queryset   string `json:"queryset"`
	File       string `json:"file"`
	QueryCount int    `json:"query_count"`
}

type protocolJSON struct {
	Timer            string  `json:"timer"`
	Harness          string  `json:"harness"`
	WarmupSecondsMin float64 `json:"warmup_seconds_min"`
	WarmupPasses     int     `json:"warmup_passes"`
	SamplesRequested int     `json:"samples_requested"`
	SampleDefinition string  `json:"sample_definition"`
	BatchSize        int     `json:"batch_size"`
	WallCapSeconds   float64 `json:"wall_cap_seconds"`
}

type measurementsJSON struct {
	SamplesNs          []int64 `json:"samples_ns"`
	SampleCount        int     `json:"sample_count"`
	MatchesPerPass     uint64  `json:"matches_per_pass"`
	TermBytesPerPass   uint64  `json:"term_bytes_per_pass"`
	DistanceSumPerPass uint64  `json:"distance_sum_per_pass"`
	ChecksumHex        string  `json:"checksum_hex"`
}

type constructJSON struct {
	Reps      int     `json:"reps"`
	TimesNs   []int64 `json:"times_ns"`
	TermCount int     `json:"term_count"`
}

type resultJSON struct {
	SchemaVersion string            `json:"schema_version"`
	Suite         string            `json:"suite"`
	TimestampUTC  string            `json:"timestamp_utc"`
	Target        targetJSON        `json:"target"`
	Dictionary    dictionaryJSON    `json:"dictionary"`
	Workload      workloadJSON      `json:"workload"`
	Algorithm     string            `json:"algorithm"`
	MaxDistance   int               `json:"max_distance"`
	Mode          string            `json:"mode"`
	Protocol      protocolJSON      `json:"protocol"`
	Construct     *constructJSON    `json:"construct,omitempty"`
	Measurements  *measurementsJSON `json:"measurements,omitempty"`
	Status        string            `json:"status"`
	Notes         []string          `json:"notes"`
}

type cellMeta struct {
	termCount   int
	structure   string
	constructNs *int64
	notes       []string
}

func querysetStem(path string) string {
	return strings.TrimSuffix(filepath.Base(path), ".txt")
}

func renderResult(args cliArgs, mode, algorithm string, maxDistance int, queriesPath string,
	queryCount int, meta cellMeta, warmupPasses int, samplesNs []int64, ref triple,
	checksum uint64, constructTimes []int64, status string, notes []string) resultJSON {
	samplesRequested := 0
	switch mode {
	case "construct":
		samplesRequested = args.reps
	case "query":
		samplesRequested = args.samples
	}
	jsonMode := mode
	if mode == "memory-child" {
		jsonMode = "memory"
	}
	result := resultJSON{
		SchemaVersion: "1.0.0",
		Suite:         "cross-language-v1",
		TimestampUTC:  time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Target: targetJSON{
			Language:       "go",
			Implementation: "vinary-tree",
			Backend:        "cgo",
			RuntimeVersion: runtime.Version(),
			LibraryVersion: "0.10.0",
			Artifact:       artifactJSON{Kind: "local-build", ID: "vinary-tree-liblevenshtein@0.10.0"},
		},
		Dictionary: dictionaryJSON{
			File:        args.dictionary,
			TermCount:   meta.termCount,
			Structure:   meta.structure,
			UnitDomain:  "unicode_scalar",
			ConstructNs: meta.constructNs,
		},
		Workload: workloadJSON{
			Queryset:   querysetStem(queriesPath),
			File:       queriesPath,
			QueryCount: queryCount,
		},
		Algorithm:   algorithm,
		MaxDistance: maxDistance,
		Mode:        jsonMode,
		Protocol: protocolJSON{
			Timer:            "monotonic",
			Harness:          "self-timed",
			WarmupSecondsMin: args.warmupSeconds,
			WarmupPasses:     warmupPasses,
			SamplesRequested: samplesRequested,
			SampleDefinition: sampleDefinition,
			BatchSize:        batchSize,
			WallCapSeconds:   wallCapSeconds,
		},
		Status: status,
		Notes:  notes,
	}
	if constructTimes != nil {
		result.Construct = &constructJSON{
			Reps:      len(constructTimes),
			TimesNs:   constructTimes,
			TermCount: meta.termCount,
		}
	} else {
		result.Measurements = &measurementsJSON{
			SamplesNs:          samplesNs,
			SampleCount:        len(samplesNs),
			MatchesPerPass:     ref.matches,
			TermBytesPerPass:   ref.bytes,
			DistanceSumPerPass: ref.distSum,
			ChecksumHex:        fmt.Sprintf("%016x", checksum),
		}
	}
	return result
}

func writeResult(outPath string, result resultJSON) {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		die("cannot create output directory: " + err.Error())
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		die("JSON encoding failed: " + err.Error())
	}
	if err := os.WriteFile(outPath, append(encoded, '\n'), 0o644); err != nil {
		die("cannot write " + outPath + ": " + err.Error())
	}
}

// ---------------------------------------------------------------------------
// Modes (PROTOCOL.md sections 6-7)
// ---------------------------------------------------------------------------

func runQueryCell(s *side, args cliArgs, queries []string, algorithm string, maxDistance int,
	queriesPath, outPath string, meta cellMeta) {
	ref, checksum := s.fullPass(queries, maxDistance, true) // untimed gate pass

	warmStart := time.Now()
	warmupBudget := time.Duration(args.warmupSeconds * float64(time.Second))
	warmupPasses := 0
	var lastPassNs int64
	for time.Since(warmStart) < warmupBudget || warmupPasses < 2 {
		t0 := time.Now()
		observed, _ := s.fullPass(queries, maxDistance, false)
		lastPassNs = time.Since(t0).Nanoseconds()
		if observed != ref {
			die("nondeterministic result during warmup")
		}
		warmupPasses++
	}

	sampleCount := args.samples
	status := "ok"
	notes := append([]string(nil), meta.notes...)
	lastPassSeconds := float64(lastPassNs) / 1e9
	if float64(sampleCount)*lastPassSeconds > wallCapSeconds {
		reduced := int(wallCapSeconds / lastPassSeconds)
		if reduced < 10 {
			reduced = 10
		}
		notes = append(notes, fmt.Sprintf(
			"samples reduced from %d to %d by the %.0fs wall cap (estimated pass %.3fs)",
			sampleCount, reduced, wallCapSeconds, lastPassSeconds))
		sampleCount = reduced
		status = "degraded"
	}

	samplesNs := make([]int64, sampleCount) // preallocated (section 3.4)
	for i := 0; i < sampleCount; i++ {
		t0 := time.Now()
		observed, _ := s.fullPass(queries, maxDistance, false)
		samplesNs[i] = time.Since(t0).Nanoseconds()
		if observed != ref {
			die("nondeterministic result during measurement")
		}
	}

	writeResult(outPath, renderResult(args, "query", algorithm, maxDistance, queriesPath,
		len(queries), meta, warmupPasses, samplesNs, ref, checksum, nil, status, notes))
}

func main() {
	selfTest()
	args := parseArgs(os.Args[1:])

	terms := readLines(args.dictionary)
	assertStrictlySorted(terms, args.dictionary)

	s := &side{}
	baseNotes := []string{
		"cgo facade (bindings/go) over the release cdylibs",
		fmt.Sprintf("GOMAXPROCS=%d, GOGC=%s (fairness rule 5: explicit defaults, recorded)",
			runtime.GOMAXPROCS(0), gogcSetting()),
	}
	meta := cellMeta{termCount: len(terms), structure: args.backend, notes: baseNotes}

	if args.mode == "construct" {
		if args.out == "" {
			die("--out is required for construct mode")
		}
		s.buildDictionary(terms, args.backend) // warmup build (also gate per section 6.2)
		s.freeDictionary()
		times := make([]int64, args.reps) // preallocated
		for r := 0; r < args.reps; r++ {
			t0 := time.Now()
			s.buildDictionary(terms, args.backend)
			times[r] = time.Since(t0).Nanoseconds()
			s.freeDictionary()
		}
		notes := append(append([]string(nil), baseNotes...),
			"construct mode: timed region is the build from the pre-sorted in-memory list only")
		queriesPath := args.queries
		if queriesPath == "" {
			queriesPath = "workload/queries/hits.txt"
		}
		writeResult(args.out, renderResult(args, "construct", "standard", 1, queriesPath, 1,
			meta, 1, nil, triple{}, 0, times, "ok", notes))
		return
	}

	buildStart := time.Now()
	s.buildDictionary(terms, args.backend)
	constructNs := time.Since(buildStart).Nanoseconds()
	meta.constructNs = &constructNs

	runOne := func(algorithm string, maxDistance int, queriesPath, outPath string) {
		s.createTransducer(algorithm)
		queries := readLines(queriesPath)
		switch args.mode {
		case "verify":
			limit := args.gateLimit
			if limit > len(queries) {
				limit = len(queries)
			}
			ref, checksum := s.fullPass(queries[:limit], maxDistance, true)
			writeResult(outPath, renderResult(args, "verify", algorithm, maxDistance,
				queriesPath, limit, meta, 0, []int64{}, ref, checksum, nil, "ok", meta.notes))
		case "memory-child":
			ref, checksum := s.fullPass(queries, maxDistance, true)
			writeResult(outPath, renderResult(args, "memory-child", algorithm, maxDistance,
				queriesPath, len(queries), meta, 0, []int64{}, ref, checksum, nil, "ok", meta.notes))
		case "query":
			runQueryCell(s, args, queries, algorithm, maxDistance, queriesPath, outPath, meta)
		default:
			die("unknown mode: " + args.mode)
		}
	}

	if args.cells != "" {
		raw, err := os.ReadFile(args.cells)
		if err != nil {
			die("cannot read cells file: " + err.Error())
		}
		for _, line := range strings.Split(string(raw), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			fields := strings.Split(line, "\t")
			if len(fields) != 4 {
				die("cells row needs 4 tab-separated fields: " + line)
			}
			distance, err := strconv.Atoi(fields[1])
			if err != nil {
				die("cells row max_distance must be an integer: " + line)
			}
			runOne(fields[0], distance, fields[2], fields[3])
		}
	} else {
		if args.algorithm == "" || args.maxDistance < 0 || args.queries == "" || args.out == "" {
			die("--algorithm, --max-distance, --queries, --out are required")
		}
		runOne(args.algorithm, args.maxDistance, args.queries, args.out)
	}
}

func gogcSetting() string {
	if value := os.Getenv("GOGC"); value != "" {
		return value
	}
	return "100"
}
