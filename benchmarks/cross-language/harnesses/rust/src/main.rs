//! Rust oracle harness for the cross-language benchmark program.
//!
//! Implements `harnesses/common/PROTOCOL.md` exactly. This binary is both
//! the "raw Rust core" anchor of the atlas and the correctness ORACLE the
//! gate compares every other target against.
//!
//! Anchor semantics note: dictionaries here are the idiomatic byte-domain
//! `DynamicDawg`/`DoubleArrayTrie` (the natural Rust choice for an ASCII
//! lexicon). Language bindings construct unicode-scalar-domain dictionaries
//! through the C ABI; for this all-ASCII workload the RESULTS are identical
//! (the gate proves it), and the domain choice is accounted as part of each
//! binding's measured cost.

use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::{Dictionary, DictionaryNode};
use liblevenshtein::transducer::{Algorithm, SubstitutionPolicyFor, Transducer, Unrestricted};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const WALL_CAP_SECONDS: f64 = 300.0;
const SAMPLE_DEFINITION: &str =
    "one full pass over the query set; every cursor fully drained and (term, distance) materialized";

// ---------------------------------------------------------------------------
// Checksum primitives (PROTOCOL.md §8) + startup self-test (§2)
// ---------------------------------------------------------------------------

#[inline]
fn fnv_update(h: u64, b: u8) -> u64 {
    (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h = fnv_update(h, b);
    }
    h
}

fn entry_hash(term: &str, distance: u64) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in term.as_bytes() {
        h = fnv_update(h, b);
    }
    h = fnv_update(h, 0x00);
    for i in 0..8 {
        h = fnv_update(h, ((distance >> (8 * i)) & 0xff) as u8);
    }
    h
}

fn self_test() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325, "FNV offset vector");
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c, "FNV 'a' vector");
    assert_eq!(entry_hash("cat", 1), 0x9697_fa3e_5046_4bc4, "entry(cat,1)");
    assert_eq!(entry_hash("cat", 0), 0xb592_c147_5b35_95e5, "entry(cat,0)");
    assert_eq!(entry_hash("cot", 1), 0xb8ac_c5d3_816b_cdea, "entry(cot,1)");
    let combined = entry_hash("cat", 0).wrapping_add(entry_hash("cot", 1));
    assert_eq!(combined, 0x6e3f_871a_dca1_63cf, "checksum{{(cat,0),(cot,1)}}");
}

// ---------------------------------------------------------------------------
// CLI (PROTOCOL.md §1)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    mode: String,
    algorithm: Option<String>,
    max_distance: Option<usize>,
    dictionary: PathBuf,
    queries: Option<PathBuf>,
    backend: String,
    out: Option<PathBuf>,
    samples: usize,
    warmup_seconds: f64,
    gate_limit: usize,
    reps: usize,
    cells: Option<PathBuf>,
}

fn die(message: &str) -> ! {
    eprintln!("bench-cross-rust: {message}");
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        mode: String::new(),
        algorithm: None,
        max_distance: None,
        dictionary: PathBuf::new(),
        queries: None,
        backend: String::new(),
        out: None,
        samples: 30,
        warmup_seconds: 3.0,
        gate_limit: 200,
        reps: 10,
        cells: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let value = argv
            .get(i + 1)
            .unwrap_or_else(|| die(&format!("flag {flag} requires a value")));
        match flag {
            "--mode" => args.mode = value.clone(),
            "--algorithm" => args.algorithm = Some(value.clone()),
            "--max-distance" => {
                args.max_distance =
                    Some(value.parse().unwrap_or_else(|_| die("bad --max-distance")))
            }
            "--dictionary" => args.dictionary = PathBuf::from(value),
            "--queries" => args.queries = Some(PathBuf::from(value)),
            "--backend" => args.backend = value.clone(),
            "--out" => args.out = Some(PathBuf::from(value)),
            "--samples" => args.samples = value.parse().unwrap_or_else(|_| die("bad --samples")),
            "--warmup-seconds" => {
                args.warmup_seconds = value.parse().unwrap_or_else(|_| die("bad --warmup-seconds"))
            }
            "--gate-limit" => {
                args.gate_limit = value.parse().unwrap_or_else(|_| die("bad --gate-limit"))
            }
            "--reps" => args.reps = value.parse().unwrap_or_else(|_| die("bad --reps")),
            "--cells" => args.cells = Some(PathBuf::from(value)),
            other => die(&format!("unknown flag: {other}")),
        }
        i += 2;
    }
    if args.mode.is_empty() {
        die("--mode is required");
    }
    if args.dictionary.as_os_str().is_empty() {
        die("--dictionary is required");
    }
    if args.backend.is_empty() {
        die("--backend is required");
    }
    args
}

fn parse_algorithm(name: &str) -> Algorithm {
    match name {
        "standard" => Algorithm::Standard,
        "transposition" => Algorithm::Transposition,
        "merge_and_split" => Algorithm::MergeAndSplit,
        "damerau_levenshtein" => Algorithm::DamerauLevenshtein,
        other => die(&format!("unknown algorithm: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Input loading (PROTOCOL.md §3)
// ---------------------------------------------------------------------------

fn read_lines(path: &Path) -> Vec<String> {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {e}", path.display())));
    let estimate = data.bytes().filter(|&b| b == b'\n').count() + 1;
    let mut lines = Vec::with_capacity(estimate);
    for line in data.split('\n') {
        if !line.is_empty() {
            lines.push(line.to_string());
        }
    }
    if lines.is_empty() {
        die(&format!("{} contains no lines", path.display()));
    }
    lines
}

fn assert_strictly_sorted(lines: &[String], path: &Path) {
    for i in 0..lines.len().saturating_sub(1) {
        if lines[i].as_bytes() >= lines[i + 1].as_bytes() {
            die(&format!(
                "{} is not strictly byte-sorted at line {}: {:?} >= {:?}",
                path.display(),
                i + 1,
                lines[i],
                lines[i + 1]
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Passes (PROTOCOL.md §5)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Triple {
    matches: u64,
    bytes: u64,
    dist: u64,
}

fn full_pass<D>(transducer: &Transducer<D>, queries: &[String], max_distance: usize) -> Triple
where
    D: Dictionary,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    let mut matches = 0u64;
    let mut bytes = 0u64;
    let mut dist = 0u64;
    for query in queries {
        for candidate in transducer.query_with_distance(query, max_distance) {
            matches += 1;
            bytes += candidate.term.len() as u64;
            dist += candidate.distance as u64;
        }
    }
    Triple {
        matches,
        bytes,
        dist,
    }
}

fn gate_pass<D>(
    transducer: &Transducer<D>,
    queries: &[String],
    max_distance: usize,
) -> (Triple, u64)
where
    D: Dictionary,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    let mut matches = 0u64;
    let mut bytes = 0u64;
    let mut dist = 0u64;
    let mut checksum = 0u64;
    for query in queries {
        for candidate in transducer.query_with_distance(query, max_distance) {
            matches += 1;
            bytes += candidate.term.len() as u64;
            dist += candidate.distance as u64;
            checksum = checksum.wrapping_add(entry_hash(
                &candidate.term,
                candidate.distance as u64,
            ));
        }
    }
    (
        Triple {
            matches,
            bytes,
            dist,
        },
        checksum,
    )
}

// ---------------------------------------------------------------------------
// JSON emission (hand-rolled; PROTOCOL.md §11 — runner post-fills the rest)
// ---------------------------------------------------------------------------

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn iso8601_utc_now() -> String {
    // Civil-from-days (Howard Hinnant's algorithm); no external time crates.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_secs();
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

struct CellContext {
    backend: String,
    structure: String,
    algorithm_name: String,
    max_distance: usize,
    mode: String,
    dictionary_file: String,
    term_count: usize,
    queries_file: String,
    query_count: usize,
    construct_ns: Option<u64>,
    notes: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn render_result_json(
    ctx: &CellContext,
    warmup_passes: usize,
    samples_requested: usize,
    warmup_seconds: f64,
    samples_ns: &[u64],
    triple: Triple,
    checksum: u64,
    construct_times: Option<(usize, &[u64])>,
    status: &str,
) -> String {
    let queryset = Path::new(&ctx.queries_file)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let mut out = String::with_capacity(4096);
    out.push_str("{\n");
    out.push_str("  \"schema_version\": \"1.0.0\",\n");
    out.push_str("  \"suite\": \"cross-language-v1\",\n");
    out.push_str(&format!(
        "  \"timestamp_utc\": \"{}\",\n",
        iso8601_utc_now()
    ));
    out.push_str("  \"target\": {\n");
    out.push_str("    \"language\": \"rust\",\n");
    out.push_str("    \"implementation\": \"rust-core\",\n");
    out.push_str(&format!(
        "    \"backend\": \"{}\",\n",
        json_escape(&ctx.backend)
    ));
    out.push_str(&format!(
        "    \"runtime_version\": \"{}\",\n",
        json_escape(env!("BENCH_RUSTC_VERSION"))
    ));
    out.push_str(&format!(
        "    \"library_version\": \"{}\",\n",
        json_escape(env!("BENCH_LIBLEV_VERSION"))
    ));
    out.push_str("    \"artifact\": { \"kind\": \"local-build\", \"id\": \"bench-cross-rust\" }\n");
    out.push_str("  },\n");
    out.push_str("  \"dictionary\": {\n");
    out.push_str(&format!(
        "    \"file\": \"{}\",\n",
        json_escape(&ctx.dictionary_file)
    ));
    out.push_str(&format!("    \"term_count\": {},\n", ctx.term_count));
    out.push_str(&format!(
        "    \"structure\": \"{}\",\n",
        json_escape(&ctx.structure)
    ));
    out.push_str("    \"unit_domain\": \"byte\"");
    if let Some(ns) = ctx.construct_ns {
        out.push_str(&format!(",\n    \"construct_ns\": {ns}\n"));
    } else {
        out.push('\n');
    }
    out.push_str("  },\n");
    out.push_str("  \"workload\": {\n");
    out.push_str(&format!("    \"queryset\": \"{}\",\n", json_escape(&queryset)));
    out.push_str(&format!(
        "    \"file\": \"{}\",\n",
        json_escape(&ctx.queries_file)
    ));
    out.push_str(&format!("    \"query_count\": {}\n", ctx.query_count));
    out.push_str("  },\n");
    out.push_str(&format!(
        "  \"algorithm\": \"{}\",\n",
        json_escape(&ctx.algorithm_name)
    ));
    out.push_str(&format!("  \"max_distance\": {},\n", ctx.max_distance));
    out.push_str(&format!("  \"mode\": \"{}\",\n", json_escape(&ctx.mode)));
    out.push_str("  \"protocol\": {\n");
    out.push_str("    \"timer\": \"monotonic\",\n");
    out.push_str("    \"harness\": \"self-timed\",\n");
    out.push_str(&format!(
        "    \"warmup_seconds_min\": {warmup_seconds},\n"
    ));
    out.push_str(&format!("    \"warmup_passes\": {warmup_passes},\n"));
    out.push_str(&format!(
        "    \"samples_requested\": {samples_requested},\n"
    ));
    out.push_str(&format!(
        "    \"sample_definition\": \"{}\",\n",
        json_escape(SAMPLE_DEFINITION)
    ));
    out.push_str("    \"batch_size\": null,\n");
    out.push_str(&format!("    \"wall_cap_seconds\": {WALL_CAP_SECONDS}\n"));
    out.push_str("  },\n");
    if let Some((reps, times)) = construct_times {
        out.push_str("  \"construct\": {\n");
        out.push_str(&format!("    \"reps\": {reps},\n"));
        let joined: Vec<String> = times.iter().map(|t| t.to_string()).collect();
        out.push_str(&format!("    \"times_ns\": [{}],\n", joined.join(", ")));
        out.push_str(&format!("    \"term_count\": {}\n", ctx.term_count));
        out.push_str("  },\n");
    } else {
        out.push_str("  \"measurements\": {\n");
        let joined: Vec<String> = samples_ns.iter().map(|t| t.to_string()).collect();
        out.push_str(&format!("    \"samples_ns\": [{}],\n", joined.join(", ")));
        out.push_str(&format!("    \"sample_count\": {},\n", samples_ns.len()));
        out.push_str(&format!("    \"matches_per_pass\": {},\n", triple.matches));
        out.push_str(&format!("    \"term_bytes_per_pass\": {},\n", triple.bytes));
        out.push_str(&format!(
            "    \"distance_sum_per_pass\": {},\n",
            triple.dist
        ));
        out.push_str(&format!(
            "    \"checksum_hex\": \"{checksum:016x}\"\n"
        ));
        out.push_str("  },\n");
    }
    out.push_str(&format!("  \"status\": \"{}\",\n", json_escape(status)));
    let notes: Vec<String> = ctx
        .notes
        .iter()
        .map(|n| format!("\"{}\"", json_escape(n)))
        .collect();
    out.push_str(&format!("  \"notes\": [{}]\n", notes.join(", ")));
    out.push_str("}\n");
    out
}

fn write_out(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| die(&format!("cannot create {}: {e}", parent.display())));
    }
    std::fs::write(path, content)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {e}", path.display())));
}

// ---------------------------------------------------------------------------
// Modes (PROTOCOL.md §6) — generic over the dictionary backend
// ---------------------------------------------------------------------------

fn base_notes() -> Vec<String> {
    vec![
        "anchor uses idiomatic byte-domain dictionaries; bindings use the ABI's unicode-scalar domain (identical results for this ASCII workload, verified by the gate)".to_string(),
    ]
}

#[allow(clippy::too_many_arguments)]
fn run_query_cell<D>(
    transducer: &Transducer<D>,
    queries: &[String],
    args_samples: usize,
    warmup_seconds: f64,
    ctx: &mut CellContext,
) -> (Vec<u64>, Triple, u64, usize, String)
where
    D: Dictionary,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    let max_distance = ctx.max_distance;
    let (ref_triple, checksum) = gate_pass(transducer, queries, max_distance);

    let warm_start = Instant::now();
    let mut warmup_passes = 0usize;
    let mut last_pass_seconds = 0f64;
    while warm_start.elapsed().as_secs_f64() < warmup_seconds || warmup_passes < 2 {
        let t0 = Instant::now();
        let triple = full_pass(transducer, queries, max_distance);
        last_pass_seconds = t0.elapsed().as_secs_f64();
        if triple != ref_triple {
            die("nondeterministic result during warmup");
        }
        warmup_passes += 1;
    }

    let mut sample_count = args_samples;
    let mut status = "ok".to_string();
    if sample_count as f64 * last_pass_seconds > WALL_CAP_SECONDS {
        sample_count = std::cmp::max(10, (WALL_CAP_SECONDS / last_pass_seconds) as usize);
        status = "degraded".to_string();
        ctx.notes.push(format!(
            "samples reduced from {args_samples} to {sample_count} by the {WALL_CAP_SECONDS}s wall cap (estimated pass {last_pass_seconds:.3}s)"
        ));
    }

    let mut samples_ns = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let t0 = Instant::now();
        let triple = full_pass(transducer, queries, max_distance);
        samples_ns.push(t0.elapsed().as_nanos() as u64);
        if triple != ref_triple {
            die("nondeterministic result during measurement");
        }
    }
    (samples_ns, ref_triple, checksum, warmup_passes, status)
}

struct CellRow {
    algorithm: String,
    max_distance: usize,
    queries_path: PathBuf,
    out_path: PathBuf,
}

fn parse_cells_file(path: &Path) -> Vec<CellRow> {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {e}", path.display())));
    let mut rows = Vec::with_capacity(data.lines().count());
    for (lineno, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 4 {
            die(&format!(
                "{}:{} expected 4 tab-separated fields, got {}",
                path.display(),
                lineno + 1,
                fields.len()
            ));
        }
        rows.push(CellRow {
            algorithm: fields[0].to_string(),
            max_distance: fields[1]
                .parse()
                .unwrap_or_else(|_| die(&format!("bad distance at line {}", lineno + 1))),
            queries_path: PathBuf::from(fields[2]),
            out_path: PathBuf::from(fields[3]),
        });
    }
    if rows.is_empty() {
        die(&format!("{} contains no cells", path.display()));
    }
    rows
}

fn run_with_backend<D, F>(args: &Args, terms: &[String], build: F, structure: &str) -> ExitCode
where
    D: Dictionary + Clone,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    F: Fn(&[String]) -> D,
{
    let dictionary_file = args.dictionary.display().to_string();

    match args.mode.as_str() {
        "construct" => {
            // One warmup build (also serves as the gate dictionary), then
            // `reps` timed builds with the drop outside the timed window.
            let warm = build(terms);
            drop(warm);
            let mut times_ns = Vec::with_capacity(args.reps);
            for _ in 0..args.reps {
                let t0 = Instant::now();
                let dict = build(terms);
                times_ns.push(t0.elapsed().as_nanos() as u64);
                drop(dict);
            }
            let out = args
                .out
                .clone()
                .unwrap_or_else(|| die("--out is required for construct mode"));
            let mut ctx = CellContext {
                backend: "native".to_string(),
                structure: structure.to_string(),
                algorithm_name: "standard".to_string(),
                max_distance: 1,
                mode: "construct".to_string(),
                dictionary_file,
                term_count: terms.len(),
                queries_file: args
                    .queries
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "workload/queries/hits.txt".to_string()),
                query_count: 1,
                construct_ns: None,
                notes: base_notes(),
            };
            ctx.notes.push(
                "construct mode: timed region is the build from the pre-sorted in-memory list only"
                    .to_string(),
            );
            let json = render_result_json(
                &ctx,
                1,
                args.reps,
                args.warmup_seconds,
                &[],
                Triple {
                    matches: 0,
                    bytes: 0,
                    dist: 0,
                },
                0,
                Some((args.reps, &times_ns)),
                "ok",
            );
            write_out(&out, &json);
            ExitCode::SUCCESS
        }
        "query" | "verify" | "memory-child" => {
            let build_t0 = Instant::now();
            let dict = build(terms);
            let construct_ns = build_t0.elapsed().as_nanos() as u64;

            let run_single = |algorithm_name: &str,
                              max_distance: usize,
                              queries_path: &Path,
                              out_path: &Path|
             -> ExitCode {
                let queries = read_lines(queries_path);
                let algorithm = parse_algorithm(algorithm_name);
                let transducer = Transducer::new(dict.clone(), algorithm);
                let mut ctx = CellContext {
                    backend: "native".to_string(),
                    structure: structure.to_string(),
                    algorithm_name: algorithm_name.to_string(),
                    max_distance,
                    // Schema mode enum has "memory", not "memory-child" (the
                    // -child suffix only distinguishes the CLI entry point).
                    mode: if args.mode == "memory-child" {
                        "memory".to_string()
                    } else {
                        args.mode.clone()
                    },
                    dictionary_file: dictionary_file.clone(),
                    term_count: terms.len(),
                    queries_file: queries_path.display().to_string(),
                    query_count: queries.len(),
                    construct_ns: Some(construct_ns),
                    notes: base_notes(),
                };
                match args.mode.as_str() {
                    "verify" => {
                        let limit = std::cmp::min(args.gate_limit, queries.len());
                        let subset = &queries[..limit];
                        ctx.query_count = limit;
                        let (triple, checksum) = gate_pass(&transducer, subset, max_distance);
                        let json = render_result_json(
                            &ctx,
                            0,
                            0,
                            args.warmup_seconds,
                            &[],
                            triple,
                            checksum,
                            None,
                            "ok",
                        );
                        write_out(out_path, &json);
                        ExitCode::SUCCESS
                    }
                    "memory-child" => {
                        let (triple, checksum) = gate_pass(&transducer, &queries, max_distance);
                        let json = render_result_json(
                            &ctx,
                            0,
                            0,
                            args.warmup_seconds,
                            &[],
                            triple,
                            checksum,
                            None,
                            "ok",
                        );
                        write_out(out_path, &json);
                        ExitCode::SUCCESS
                    }
                    _ => {
                        let (samples_ns, triple, checksum, warmup_passes, status) = run_query_cell(
                            &transducer,
                            &queries,
                            args.samples,
                            args.warmup_seconds,
                            &mut ctx,
                        );
                        let json = render_result_json(
                            &ctx,
                            warmup_passes,
                            args.samples,
                            args.warmup_seconds,
                            &samples_ns,
                            triple,
                            checksum,
                            None,
                            &status,
                        );
                        write_out(out_path, &json);
                        ExitCode::SUCCESS
                    }
                }
            };

            if let Some(cells_path) = &args.cells {
                let rows = parse_cells_file(cells_path);
                for row in &rows {
                    let code = run_single(
                        &row.algorithm,
                        row.max_distance,
                        &row.queries_path,
                        &row.out_path,
                    );
                    if code != ExitCode::SUCCESS {
                        return code;
                    }
                }
                ExitCode::SUCCESS
            } else {
                let algorithm = args
                    .algorithm
                    .clone()
                    .unwrap_or_else(|| die("--algorithm is required"));
                let max_distance = args
                    .max_distance
                    .unwrap_or_else(|| die("--max-distance is required"));
                let queries = args
                    .queries
                    .clone()
                    .unwrap_or_else(|| die("--queries is required"));
                let out = args
                    .out
                    .clone()
                    .unwrap_or_else(|| die("--out is required"));
                run_single(&algorithm, max_distance, &queries, &out)
            }
        }
        other => die(&format!("unknown mode: {other}")),
    }
}

fn main() -> ExitCode {
    self_test();
    let args = parse_args();

    let terms = read_lines(&args.dictionary);
    assert_strictly_sorted(&terms, &args.dictionary);

    match args.backend.as_str() {
        "dynamic_dawg" => run_with_backend(
            &args,
            &terms,
            |t| DynamicDawg::<()>::from_terms(t.iter().map(String::as_str)),
            "dynamic_dawg",
        ),
        "double_array_trie" => run_with_backend(
            &args,
            &terms,
            |t| DoubleArrayTrie::<()>::from_terms(t.iter().map(String::as_str)),
            "double_array_trie",
        ),
        other => die(&format!("unknown backend: {other}")),
    }
}
