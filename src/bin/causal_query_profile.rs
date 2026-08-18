//! Work-counter driver for the Java performance-parity investigation.

use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgU64};
use libdictenstein::{
    causal_construction_stats, reset_causal_construction_stats, CausalConstructionStats,
    Dictionary, DictionaryNode,
};
use liblevenshtein::transducer::{Algorithm, SubstitutionPolicyFor, Transducer, Unrestricted};
use liblevenshtein::{causal_perf_stats, reset_causal_perf_stats, CausalPerfStats};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Args {
    dictionary: PathBuf,
    queries: PathBuf,
    domain: String,
    constructor: String,
    algorithm: Algorithm,
    max_distance: usize,
    passes: usize,
}

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("causal_query_profile: {}", message.as_ref());
    std::process::exit(2);
}

fn parse_algorithm(value: &str) -> Algorithm {
    match value {
        "standard" => Algorithm::Standard,
        "transposition" => Algorithm::Transposition,
        "merge_and_split" => Algorithm::MergeAndSplit,
        "damerau_levenshtein" => Algorithm::DamerauLevenshtein,
        _ => fail(format!("unknown algorithm {value:?}")),
    }
}

fn parse_args() -> Args {
    let mut dictionary = None;
    let mut queries = None;
    let mut domain = "byte".to_owned();
    let mut constructor = "from_terms".to_owned();
    let mut algorithm = Algorithm::Standard;
    let mut max_distance = None;
    let mut passes = 1usize;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv
            .next()
            .unwrap_or_else(|| fail(format!("{flag} requires a value")));
        match flag.as_str() {
            "--dictionary" => dictionary = Some(PathBuf::from(value)),
            "--queries" => queries = Some(PathBuf::from(value)),
            "--domain" => domain = value,
            "--constructor" => constructor = value,
            "--algorithm" => algorithm = parse_algorithm(&value),
            "--max-distance" => {
                max_distance = Some(
                    value
                        .parse()
                        .unwrap_or_else(|_| fail("--max-distance must be an integer")),
                )
            }
            "--passes" => {
                passes = value
                    .parse()
                    .unwrap_or_else(|_| fail("--passes must be a positive integer"))
            }
            _ => fail(format!("unknown flag {flag:?}")),
        }
    }
    if passes == 0 {
        fail("--passes must be positive");
    }
    Args {
        dictionary: dictionary.unwrap_or_else(|| fail("--dictionary is required")),
        queries: queries.unwrap_or_else(|| fail("--queries is required")),
        domain,
        constructor,
        algorithm,
        max_distance: max_distance.unwrap_or_else(|| fail("--max-distance is required")),
        passes,
    }
}

fn read_lines(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| fail(format!("cannot read {}: {error}", path.display())));
    let lines: Vec<_> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect();
    if lines.is_empty() {
        fail(format!("{} has no non-empty lines", path.display()));
    }
    lines
}

#[inline]
fn hash_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
}

fn entry_hash(term: &str, distance: usize) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in term.bytes() {
        hash = hash_byte(hash, byte);
    }
    hash = hash_byte(hash, 0);
    for byte in (distance as u64).to_le_bytes() {
        hash = hash_byte(hash, byte);
    }
    hash
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResultTotals {
    matches: u64,
    bytes: u64,
    distance: u64,
    checksum: u64,
}

struct Measurement {
    term_count: usize,
    query_count: usize,
    build_ns: u128,
    query_ns: u128,
    totals: ResultTotals,
    construction: CausalConstructionStats,
    stats: CausalPerfStats,
}

fn query_pass<D>(
    transducer: &Transducer<D>,
    queries: &[String],
    max_distance: usize,
) -> ResultTotals
where
    D: Dictionary,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    let mut totals = ResultTotals::default();
    for query in queries {
        for candidate in transducer.query_with_distance(query, max_distance) {
            totals.matches = totals.matches.saturating_add(1);
            totals.bytes = totals.bytes.saturating_add(candidate.term.len() as u64);
            totals.distance = totals.distance.saturating_add(candidate.distance as u64);
            totals.checksum = totals
                .checksum
                .wrapping_add(entry_hash(&candidate.term, candidate.distance));
        }
    }
    totals
}

fn run<D, F>(args: &Args, terms: &[String], queries: &[String], build: F)
where
    D: Dictionary,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    F: FnOnce(&[String]) -> D,
{
    reset_causal_construction_stats();
    let build_start = Instant::now();
    let dictionary = build(terms);
    let build_ns = build_start.elapsed().as_nanos();
    let construction = causal_construction_stats();
    let transducer = Transducer::new(dictionary, args.algorithm);

    reset_causal_perf_stats();
    let query_start = Instant::now();
    let expected = query_pass(&transducer, queries, args.max_distance);
    for _ in 1..args.passes {
        let actual = query_pass(&transducer, queries, args.max_distance);
        assert_eq!(actual, expected, "query passes must be deterministic");
    }
    let query_ns = query_start.elapsed().as_nanos();
    let stats = causal_perf_stats();
    print_json(
        args,
        &Measurement {
            term_count: terms.len(),
            query_count: queries.len(),
            build_ns,
            query_ns,
            totals: expected,
            construction,
            stats,
        },
    );
}

fn print_json(args: &Args, measurement: &Measurement) {
    let Measurement {
        term_count,
        query_count,
        build_ns,
        query_ns,
        totals,
        construction,
        stats,
    } = measurement;
    let mut json = String::with_capacity(4096);
    writeln!(&mut json, "{{").unwrap();
    writeln!(
        &mut json,
        "  \"schema\": \"liblevenshtein.causal-work.v1\","
    )
    .unwrap();
    writeln!(&mut json, "  \"domain\": {:?},", args.domain).unwrap();
    writeln!(&mut json, "  \"constructor\": {:?},", args.constructor).unwrap();
    writeln!(
        &mut json,
        "  \"algorithm\": {:?},",
        format!("{:?}", args.algorithm)
    )
    .unwrap();
    writeln!(&mut json, "  \"max_distance\": {},", args.max_distance).unwrap();
    writeln!(&mut json, "  \"passes\": {},", args.passes).unwrap();
    writeln!(&mut json, "  \"term_count\": {term_count},").unwrap();
    writeln!(&mut json, "  \"query_count\": {query_count},").unwrap();
    writeln!(&mut json, "  \"build_ns\": {build_ns},").unwrap();
    writeln!(&mut json, "  \"query_ns\": {query_ns},").unwrap();
    writeln!(&mut json, "  \"matches\": {},", totals.matches).unwrap();
    writeln!(&mut json, "  \"term_bytes\": {},", totals.bytes).unwrap();
    writeln!(&mut json, "  \"distance_sum\": {},", totals.distance).unwrap();
    writeln!(&mut json, "  \"checksum_u64\": {},", totals.checksum).unwrap();
    writeln!(&mut json, "  \"construction_work\": {{").unwrap();
    let construction_fields = [
        ("term_insert_attempts", construction.term_insert_attempts),
        ("input_units", construction.input_units),
        ("version_loads", construction.version_loads),
        ("path_units_walked", construction.path_units_walked),
        ("edge_lists_cloned", construction.edge_lists_cloned),
        ("edge_arcs_cloned", construction.edge_arcs_cloned),
        ("nodes_created", construction.nodes_created),
        ("nodes_dropped", construction.nodes_dropped),
        (
            "graph_versions_created",
            construction.graph_versions_created,
        ),
        ("cas_publications", construction.cas_publications),
        ("cas_retries", construction.cas_retries),
        ("batch_sort_calls", construction.batch_sort_calls),
        ("batch_sort_terms", construction.batch_sort_terms),
        ("batch_sort_units", construction.batch_sort_units),
    ];
    for (index, (name, value)) in construction_fields.iter().enumerate() {
        let comma = if index + 1 == construction_fields.len() {
            ""
        } else {
            ","
        };
        writeln!(&mut json, "    \"{name}\": {value}{comma}").unwrap();
    }
    writeln!(&mut json, "  }},").unwrap();
    writeln!(&mut json, "  \"work\": {{").unwrap();
    let fields = [
        ("dictionary_intersections", stats.dictionary_intersections),
        ("final_checks", stats.final_checks),
        ("edges_enumerated", stats.edges_enumerated),
        ("transition_attempts", stats.transition_attempts),
        ("transition_accepted", stats.transition_accepted),
        ("packed_standard_queries", stats.packed_standard_queries),
        ("positional_unit_queries", stats.positional_unit_queries),
        (
            "packed_standard_transition_attempts",
            stats.packed_standard_transition_attempts,
        ),
        (
            "packed_standard_transition_dead",
            stats.packed_standard_transition_dead,
        ),
        ("packed_dfa_queries", stats.packed_dfa_queries),
        (
            "packed_dfa_transition_hits",
            stats.packed_dfa_transition_hits,
        ),
        (
            "packed_dfa_transition_misses",
            stats.packed_dfa_transition_misses,
        ),
        (
            "packed_dfa_states_interned",
            stats.packed_dfa_states_interned,
        ),
        ("generated_transition_hits", stats.generated_transition_hits),
        (
            "generated_transition_misses",
            stats.generated_transition_misses,
        ),
        (
            "generated_product_expansions",
            stats.generated_product_expansions,
        ),
        (
            "generated_product_identity_expansions",
            stats.generated_product_identity_expansions,
        ),
        (
            "generated_product_unique_expansions",
            stats.generated_product_unique_expansions,
        ),
        (
            "generated_product_repeated_expansions",
            stats.generated_product_repeated_expansions,
        ),
        ("epsilon_input_positions", stats.epsilon_input_positions),
        ("epsilon_output_positions", stats.epsilon_output_positions),
        ("characteristic_vectors", stats.characteristic_vectors),
        ("characteristic_units", stats.characteristic_units),
        ("successor_candidates", stats.successor_candidates),
        ("state_insert_attempts", stats.state_insert_attempts),
        ("state_insert_retained", stats.state_insert_retained),
        ("subsumption_checks", stats.subsumption_checks),
        ("state_copy_calls", stats.state_copy_calls),
        ("state_positions_copied", stats.state_positions_copied),
        ("state_bytes_copied", stats.state_bytes_copied),
        ("state_positions_enqueued", stats.state_positions_enqueued),
        ("state_bytes_enqueued", stats.state_bytes_enqueued),
        ("pending_queue_peak", stats.pending_queue_peak),
        ("pool_acquires", stats.pool_acquires),
        ("pool_reuses", stats.pool_reuses),
        ("pool_misses", stats.pool_misses),
        ("pool_releases", stats.pool_releases),
        ("matches_materialized", stats.matches_materialized),
        (
            "foreign_is_final_callbacks",
            stats.foreign_is_final_callbacks,
        ),
        ("foreign_edge_callbacks", stats.foreign_edge_callbacks),
        ("foreign_edge_pages", stats.foreign_edge_pages),
        ("foreign_edge_descriptors", stats.foreign_edge_descriptors),
        ("ffi_matches_packed", stats.ffi_matches_packed),
        ("ffi_bytes_packed", stats.ffi_bytes_packed),
    ];
    for (index, (name, value)) in fields.iter().enumerate() {
        let comma = if index + 1 == fields.len() { "" } else { "," };
        writeln!(&mut json, "    \"{name}\": {value}{comma}").unwrap();
    }
    writeln!(&mut json, "  }}").unwrap();
    writeln!(&mut json, "}}").unwrap();
    print!("{json}");
}

fn main() {
    let args = parse_args();
    let terms = read_lines(&args.dictionary);
    let queries = read_lines(&args.queries);
    match (args.domain.as_str(), args.constructor.as_str()) {
        ("byte", "from_terms") => run(&args, &terms, &queries, |items| {
            DynamicDawg::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("byte", "from_sorted_terms") => run(&args, &terms, &queries, |items| {
            DynamicDawg::<()>::from_sorted_terms(items.iter().map(String::as_str))
        }),
        ("byte", "stream") => run(&args, &terms, &queries, |items| {
            let dictionary = DynamicDawg::<()>::new();
            for item in items {
                dictionary.insert(item);
            }
            dictionary
        }),
        ("unicode", "from_terms") => run(&args, &terms, &queries, |items| {
            DynamicDawgChar::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("unicode", "from_sorted_terms") => run(&args, &terms, &queries, |items| {
            DynamicDawgChar::<()>::from_sorted_terms(items.iter().map(String::as_str))
        }),
        ("unicode", "stream") => run(&args, &terms, &queries, |items| {
            let dictionary = DynamicDawgChar::<()>::new();
            for item in items {
                dictionary.insert(item);
            }
            dictionary
        }),
        ("u64", "from_terms") => run(&args, &terms, &queries, |items| {
            DynamicDawgU64::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("u64", "from_sorted_terms") => run(&args, &terms, &queries, |items| {
            DynamicDawgU64::<()>::from_sorted_terms(items.iter().map(String::as_str))
        }),
        ("u64", "stream") => run(&args, &terms, &queries, |items| {
            let dictionary = DynamicDawgU64::<()>::new();
            for item in items {
                dictionary.insert(item);
            }
            dictionary
        }),
        ("byte" | "unicode" | "u64", _) => {
            fail("--constructor must be from_terms, from_sorted_terms, or stream")
        }
        _ => fail("--domain must be byte, unicode, or u64"),
    }
}
