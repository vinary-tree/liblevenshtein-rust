//! Same-binary driver for phonetic iterator-retention experiments.
//!
//! The phonetic NFA is compiled from `--pattern` once. The argument passed to
//! `query` is intentionally the same string even though the current phonetic
//! API takes its language from the precompiled NFA rather than reparsing that
//! argument per iterator.

use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use liblevenshtein::phonetic::nfa::compile;
use liblevenshtein::phonetic::regex::parse;
use liblevenshtein::transducer::{PhoneticCandidate, PhoneticTransducerChar};
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

const LEGACY_RETENTION_ENV: &str = "LIBLEVENSHTEIN_CAUSAL_RETAIN_LEGACY_PHONETIC_STATE";
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Args {
    dictionary: PathBuf,
    pattern: String,
    max_distance: u8,
    iterations: usize,
    workload: Workload,
}

#[derive(Clone, Copy, Debug)]
enum Workload {
    ConstructDrop,
    Retained,
    First,
    Full,
}

impl Workload {
    fn parse(value: &str) -> Self {
        match value {
            "construct-drop" => Self::ConstructDrop,
            "retained" => Self::Retained,
            "first" => Self::First,
            "full" => Self::Full,
            _ => fail("--workload must be construct-drop, retained, first, or full"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::ConstructDrop => "construct-drop",
            Self::Retained => "retained",
            Self::First => "first",
            Self::Full => "full",
        }
    }

    fn timing_scope(self) -> &'static str {
        match self {
            Self::ConstructDrop => "construction-and-drop",
            Self::Retained => "construction-with-drop-excluded",
            Self::First => "construction-first-result-and-drop",
            Self::Full => "construction-full-consumption-and-drop",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SimpleTotals {
    matches: u64,
    term_bytes: u64,
    distance: u64,
}

impl SimpleTotals {
    #[inline]
    fn consume(&mut self, candidate: &PhoneticCandidate) {
        self.matches = self.matches.saturating_add(1);
        self.term_bytes = self.term_bytes.saturating_add(candidate.term.len() as u64);
        self.distance = self
            .distance
            .saturating_add(u64::from(candidate.edit_distance));
        black_box(candidate);
    }

    fn repeated(self, iterations: usize) -> Self {
        let iterations = iterations as u64;
        Self {
            matches: self.matches.saturating_mul(iterations),
            term_bytes: self.term_bytes.saturating_mul(iterations),
            distance: self.distance.saturating_mul(iterations),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GateTotals {
    simple: SimpleTotals,
    checksum: u64,
    order_checksum: u64,
}

impl GateTotals {
    fn consume(&mut self, candidate: &PhoneticCandidate) {
        self.simple.consume(candidate);

        let mut entry = FNV_OFFSET;
        for byte in candidate.term.bytes() {
            entry = (entry ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        entry = (entry ^ u64::from(candidate.edit_distance)).wrapping_mul(FNV_PRIME);
        for byte in candidate.phonetic_cost.to_bits().to_le_bytes() {
            entry = (entry ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        self.checksum = self.checksum.wrapping_add(entry);
        self.order_checksum = self
            .order_checksum
            .wrapping_mul(0x9E37_79B1_85EB_CA87)
            .wrapping_add(entry);
    }
}

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("causal_phonetic_profile: {}", message.as_ref());
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut dictionary = None;
    let mut pattern = None;
    let mut max_distance = None;
    let mut iterations = None;
    let mut workload = None;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv
            .next()
            .unwrap_or_else(|| fail(format!("{flag} requires a value")));
        match flag.as_str() {
            "--dictionary" => dictionary = Some(PathBuf::from(value)),
            "--pattern" => pattern = Some(value),
            "--max-distance" => {
                max_distance =
                    Some(value.parse().unwrap_or_else(|_| {
                        fail("--max-distance must be an integer from 0 to 255")
                    }))
            }
            "--iterations" => {
                iterations = Some(
                    value
                        .parse()
                        .unwrap_or_else(|_| fail("--iterations must be a positive integer")),
                )
            }
            "--workload" => workload = Some(Workload::parse(&value)),
            _ => fail(format!("unknown flag {flag:?}")),
        }
    }

    let iterations = iterations.unwrap_or_else(|| fail("--iterations is required"));
    if iterations == 0 {
        fail("--iterations must be positive");
    }

    Args {
        dictionary: dictionary.unwrap_or_else(|| fail("--dictionary is required")),
        pattern: pattern.unwrap_or_else(|| fail("--pattern is required")),
        max_distance: max_distance.unwrap_or_else(|| fail("--max-distance is required")),
        iterations,
        workload: workload.unwrap_or_else(|| fail("--workload is required")),
    }
}

fn read_terms(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| fail(format!("cannot read {}: {error}", path.display())));
    let terms: Vec<_> = text
        .lines()
        .filter(|term| !term.is_empty())
        .map(str::to_owned)
        .collect();
    if terms.is_empty() {
        fail(format!("{} contains no non-empty terms", path.display()));
    }
    terms
}

fn gate<I>(query: I, first_only: bool) -> GateTotals
where
    I: Iterator<Item = PhoneticCandidate>,
{
    let mut totals = GateTotals::default();
    if first_only {
        if let Some(candidate) = query.into_iter().next() {
            totals.consume(&candidate);
        }
    } else {
        for candidate in query {
            totals.consume(&candidate);
        }
    }
    totals
}

fn run_profile<I, F>(
    args: &Args,
    term_count: usize,
    retention_mode: &'static str,
    mut make_query: F,
) where
    I: Iterator<Item = PhoneticCandidate>,
    F: FnMut() -> I,
{
    // Equal steady-state policy for every workload and arm. Besides allocator
    // warm-up, this resolves any process-local lazy initialization before the
    // measured region.
    drop(black_box(make_query()));

    // Rich checksums are deliberately outside the timer. Both gates are
    // emitted even for construction-only workloads so every process proves
    // first-result and full-order equivalence.
    let first_gate = gate(make_query(), true);
    let full_gate = gate(make_query(), false);

    let start = Instant::now();
    let timed = match args.workload {
        Workload::ConstructDrop => {
            for _ in 0..args.iterations {
                drop(black_box(make_query()));
            }
            SimpleTotals::default()
        }
        Workload::Retained => {
            let mut queries = Vec::with_capacity(args.iterations);
            for _ in 0..args.iterations {
                queries.push(make_query());
            }
            black_box(&queries);
            let elapsed = start.elapsed().as_nanos();
            print_measurement::<I>(
                args,
                term_count,
                retention_mode,
                elapsed,
                first_gate,
                full_gate,
                SimpleTotals::default(),
            );
            return;
        }
        Workload::First => {
            let mut totals = SimpleTotals::default();
            for _ in 0..args.iterations {
                if let Some(candidate) = make_query().next() {
                    totals.consume(&candidate);
                }
            }
            assert_eq!(totals, first_gate.simple.repeated(args.iterations));
            totals
        }
        Workload::Full => {
            let mut totals = SimpleTotals::default();
            for _ in 0..args.iterations {
                for candidate in make_query() {
                    totals.consume(&candidate);
                }
            }
            assert_eq!(totals, full_gate.simple.repeated(args.iterations));
            totals
        }
    };
    let elapsed = start.elapsed().as_nanos();
    black_box(timed);
    print_measurement::<I>(
        args,
        term_count,
        retention_mode,
        elapsed,
        first_gate,
        full_gate,
        timed,
    );
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if control <= '\u{1f}' => {
                write!(&mut out, "\\u{:04x}", u32::from(control)).unwrap();
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

#[allow(clippy::too_many_arguments)]
fn print_measurement<I>(
    args: &Args,
    term_count: usize,
    retention_mode: &str,
    elapsed_ns: u128,
    first_gate: GateTotals,
    full_gate: GateTotals,
    timed: SimpleTotals,
) {
    let iterator_inline_bytes = std::mem::size_of::<I>();
    let retained_inline_bytes = iterator_inline_bytes.saturating_mul(args.iterations);
    println!("{{");
    println!("  \"schema\": \"liblevenshtein.causal-phonetic.v2\",");
    println!(
        "  \"phonetic_retention_mode\": {},",
        json_string(retention_mode)
    );
    println!("  \"workload\": {},", json_string(args.workload.name()));
    println!(
        "  \"timing_scope\": {},",
        json_string(args.workload.timing_scope())
    );
    println!("  \"term_count\": {term_count},");
    println!("  \"pattern\": {},", json_string(&args.pattern));
    println!("  \"max_distance\": {},", args.max_distance);
    println!("  \"iterations\": {},", args.iterations);
    println!("  \"iterator_inline_bytes\": {iterator_inline_bytes},");
    println!("  \"retained_inline_bytes\": {retained_inline_bytes},");
    println!("  \"elapsed_ns\": {elapsed_ns},");
    println!("  \"first_gate_matches\": {},", first_gate.simple.matches);
    println!(
        "  \"first_gate_term_bytes\": {},",
        first_gate.simple.term_bytes
    );
    println!(
        "  \"first_gate_distance_sum\": {},",
        first_gate.simple.distance
    );
    println!("  \"first_gate_checksum_u64\": {},", first_gate.checksum);
    println!(
        "  \"first_gate_order_checksum_u64\": {},",
        first_gate.order_checksum
    );
    println!("  \"full_gate_matches\": {},", full_gate.simple.matches);
    println!(
        "  \"full_gate_term_bytes\": {},",
        full_gate.simple.term_bytes
    );
    println!(
        "  \"full_gate_distance_sum\": {},",
        full_gate.simple.distance
    );
    println!("  \"full_gate_checksum_u64\": {},", full_gate.checksum);
    println!(
        "  \"full_gate_order_checksum_u64\": {},",
        full_gate.order_checksum
    );
    println!("  \"timed_matches\": {},", timed.matches);
    println!("  \"timed_term_bytes\": {},", timed.term_bytes);
    println!("  \"timed_distance_sum\": {}", timed.distance);
    println!("}}");
}

fn main() {
    let args = parse_args();
    let terms = read_terms(&args.dictionary);
    let parsed = parse(&args.pattern)
        .unwrap_or_else(|error| fail(format!("cannot parse phonetic pattern: {error}")));
    let nfa = compile(&parsed)
        .unwrap_or_else(|error| fail(format!("cannot compile phonetic pattern: {error}")));
    let dictionary = DoubleArrayTrieChar::from_terms(terms.iter().map(String::as_str));
    let transducer = PhoneticTransducerChar::new(dictionary, nfa, args.max_distance);
    let term_count = terms.len();

    // Select once, outside every measured region. Separate generic
    // monomorphizations keep the production iterator and the historical
    // layout control out of a common enum or inline optional payload.
    if std::env::var_os(LEGACY_RETENTION_ENV).is_some() {
        run_profile(&args, term_count, "legacy-retained-state", || {
            transducer.query_legacy_unit_cost_retention_control(&args.pattern)
        });
    } else {
        run_profile(&args, term_count, "mode-specific-state", || {
            transducer.query(&args.pattern)
        });
    }
}
