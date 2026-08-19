//! Work-counter driver for the Java performance-parity investigation.

use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgU64};
use libdictenstein::{
    causal_construction_stats, reset_causal_construction_stats, CausalConstructionStats, CharUnit,
    Dictionary, DictionaryNode,
};
use liblevenshtein::cache::eviction::Noop;
use liblevenshtein::transducer::{
    Algorithm, PrefixQueryIterator, PriorityQueryIterator, SubsequenceQueryIterator,
    SubstitutionPolicyFor, Transducer, Unrestricted,
};
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
    scheduler: String,
    max_distance: usize,
    passes: usize,
    limit: Option<usize>,
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
    let mut scheduler = "unordered".to_owned();
    let mut max_distance = None;
    let mut passes = 1usize;
    let mut limit = None;
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
            "--scheduler" => {
                if !matches!(
                    value.as_str(),
                    "unordered" | "ordered" | "priority" | "prefix" | "subsequence"
                ) {
                    fail(
                        "--scheduler must be unordered, ordered, priority, prefix, or subsequence",
                    );
                }
                scheduler = value;
            }
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
            "--limit" => {
                let parsed = value
                    .parse()
                    .unwrap_or_else(|_| fail("--limit must be a positive integer"));
                if parsed == 0 {
                    fail("--limit must be positive");
                }
                limit = Some(parsed);
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
        scheduler,
        max_distance: max_distance.unwrap_or_else(|| fail("--max-distance is required")),
        passes,
        limit,
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
    order_checksum: u64,
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

fn query_pass<D, const WITH_CHECKSUM: bool>(
    transducer: &Transducer<D>,
    queries: &[String],
    max_distance: usize,
    scheduler: &str,
    limit: Option<usize>,
) -> ResultTotals
where
    D: Dictionary,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
{
    let mut totals = ResultTotals::default();
    let limit = limit.unwrap_or(usize::MAX);
    for query in queries {
        let mut consume = |term: &str, distance: usize| {
            totals.matches = totals.matches.saturating_add(1);
            totals.bytes = totals.bytes.saturating_add(term.len() as u64);
            totals.distance = totals.distance.saturating_add(distance as u64);
            if WITH_CHECKSUM {
                let entry = entry_hash(term, distance);
                totals.checksum = totals.checksum.wrapping_add(entry);
                totals.order_checksum = totals
                    .order_checksum
                    .wrapping_mul(0x9E37_79B1_85EB_CA87)
                    .wrapping_add(entry);
            }
        };
        match scheduler {
            "unordered" => {
                for candidate in transducer
                    .query_with_distance(query, max_distance)
                    .take(limit)
                {
                    consume(&candidate.term, candidate.distance);
                }
            }
            "ordered" => {
                for candidate in transducer.query_ordered(query, max_distance).take(limit) {
                    consume(&candidate.term, candidate.distance);
                }
            }
            "priority" => {
                for candidate in PriorityQueryIterator::new(
                    transducer.dictionary().root(),
                    query,
                    max_distance,
                    transducer.algorithm(),
                )
                .take(limit)
                {
                    consume(&candidate.term, candidate.distance);
                }
            }
            "prefix" => {
                let query_units = <D::Node as DictionaryNode>::Unit::from_str(query);
                for candidate in PrefixQueryIterator::from_dictionary(
                    transducer.dictionary(),
                    query_units,
                    max_distance,
                    transducer.algorithm(),
                )
                .take(limit)
                {
                    let term = <D::Node as DictionaryNode>::Unit::to_string(&candidate.units);
                    consume(&term, candidate.distance);
                }
            }
            "subsequence" => {
                let query_units = <D::Node as DictionaryNode>::Unit::from_str(query);
                for candidate in
                    SubsequenceQueryIterator::from_dictionary(transducer.dictionary(), query_units)
                        .take(limit)
                {
                    let term = <D::Node as DictionaryNode>::Unit::to_string(&candidate.units);
                    consume(&term, 0);
                }
            }
            _ => unreachable!("scheduler was validated during argument parsing"),
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

    // Match the normative cross-language protocol: establish the exact result
    // checksum in an untimed gate, then profile only materialization plus the
    // O(1)-per-match count/byte/distance triple. Per-byte hashing inside the
    // timer otherwise overwhelms and misattributes the query hot path.
    let gate = query_pass::<D, true>(
        &transducer,
        queries,
        args.max_distance,
        &args.scheduler,
        args.limit,
    );

    reset_causal_perf_stats();
    let query_start = Instant::now();
    let expected = query_pass::<D, false>(
        &transducer,
        queries,
        args.max_distance,
        &args.scheduler,
        args.limit,
    );
    assert_eq!(
        (expected.matches, expected.bytes, expected.distance),
        (gate.matches, gate.bytes, gate.distance),
        "timed query result triple must match the checksum gate",
    );
    for _ in 1..args.passes {
        let actual = query_pass::<D, false>(
            &transducer,
            queries,
            args.max_distance,
            &args.scheduler,
            args.limit,
        );
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
            totals: gate,
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
    let is_dat_constructor = matches!(
        args.constructor.as_str(),
        "double_array_trie_from_terms" | "triple_noop_double_array_trie_from_terms"
    );
    let dat_cursor_mode = if is_dat_constructor {
        if std::env::var_os("LIBDICTENSTEIN_CAUSAL_USE_CHECKED_DAT_CURSOR_EDGES").is_some() {
            "checked-revalidation"
        } else {
            "construction-proven"
        }
    } else {
        "not-applicable"
    };
    writeln!(&mut json, "  \"dat_cursor_mode\": {dat_cursor_mode:?},").unwrap();
    let parent_arena_requested = cfg!(feature = "perf-instrumentation")
        && std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_CURSOR_KEY_RECONSTRUCTION").is_some();
    let cursor_key_mode = if is_dat_constructor {
        if parent_arena_requested {
            "parent-arena"
        } else {
            "cursor-native"
        }
    } else {
        "parent-arena"
    };
    writeln!(&mut json, "  \"cursor_key_mode\": {cursor_key_mode:?},").unwrap();
    let priority_path_mode = if args.scheduler == "priority" {
        if cfg!(feature = "benchmark-controls")
            && std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_CLONED_PRIORITY_PATHS").is_some()
        {
            "cloned-vectors"
        } else {
            "parent-arena"
        }
    } else {
        "not-applicable"
    };
    writeln!(
        &mut json,
        "  \"priority_path_mode\": {priority_path_mode:?},"
    )
    .unwrap();
    let dfs_edge_mode = if matches!(args.scheduler.as_str(), "prefix" | "subsequence") {
        if cfg!(feature = "benchmark-controls")
            && std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_DFS_EDGE_PAGING").is_some()
        {
            "eager".to_owned()
        } else {
            let capacity = if cfg!(feature = "benchmark-controls") {
                std::env::var("LIBLEVENSHTEIN_CAUSAL_DFS_EDGE_PAGE_CAPACITY")
                    .unwrap_or_else(|_| "8".to_owned())
            } else {
                "8".to_owned()
            };
            format!("paged-{capacity}")
        }
    } else {
        "not-applicable".to_owned()
    };
    writeln!(&mut json, "  \"dfs_edge_mode\": {dfs_edge_mode:?},").unwrap();
    let packed_source_row_mode = "source-fixed";
    writeln!(
        &mut json,
        "  \"packed_source_row_mode\": {packed_source_row_mode:?},"
    )
    .unwrap();
    writeln!(&mut json, "  \"scheduler\": {:?},", args.scheduler).unwrap();
    writeln!(
        &mut json,
        "  \"algorithm\": {:?},",
        format!("{:?}", args.algorithm)
    )
    .unwrap();
    writeln!(&mut json, "  \"max_distance\": {},", args.max_distance).unwrap();
    writeln!(&mut json, "  \"passes\": {},", args.passes).unwrap();
    match args.limit {
        Some(limit) => writeln!(&mut json, "  \"limit\": {limit},").unwrap(),
        None => writeln!(&mut json, "  \"limit\": null,").unwrap(),
    }
    writeln!(&mut json, "  \"term_count\": {term_count},").unwrap();
    writeln!(&mut json, "  \"query_count\": {query_count},").unwrap();
    writeln!(&mut json, "  \"build_ns\": {build_ns},").unwrap();
    writeln!(&mut json, "  \"query_ns\": {query_ns},").unwrap();
    writeln!(&mut json, "  \"matches\": {},", totals.matches).unwrap();
    writeln!(&mut json, "  \"term_bytes\": {},", totals.bytes).unwrap();
    writeln!(&mut json, "  \"distance_sum\": {},", totals.distance).unwrap();
    writeln!(&mut json, "  \"checksum_u64\": {},", totals.checksum).unwrap();
    writeln!(
        &mut json,
        "  \"order_checksum_u64\": {},",
        totals.order_checksum
    )
    .unwrap();
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
        (
            "owned_traversal_arena_insertions",
            stats.owned_traversal_arena_insertions,
        ),
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
        ("packed_osa_queries", stats.packed_osa_queries),
        (
            "packed_osa_fallback_policy",
            stats.packed_osa_fallback_policy,
        ),
        (
            "packed_osa_fallback_prefix",
            stats.packed_osa_fallback_prefix,
        ),
        ("packed_osa_fallback_width", stats.packed_osa_fallback_width),
        (
            "packed_osa_transition_attempts",
            stats.packed_osa_transition_attempts,
        ),
        (
            "packed_osa_transition_dead",
            stats.packed_osa_transition_dead,
        ),
        (
            "packed_merge_split_queries",
            stats.packed_merge_split_queries,
        ),
        (
            "packed_merge_split_fallback_policy",
            stats.packed_merge_split_fallback_policy,
        ),
        (
            "packed_merge_split_fallback_prefix",
            stats.packed_merge_split_fallback_prefix,
        ),
        (
            "packed_merge_split_fallback_width",
            stats.packed_merge_split_fallback_width,
        ),
        (
            "packed_merge_split_transition_attempts",
            stats.packed_merge_split_transition_attempts,
        ),
        (
            "packed_merge_split_transition_dead",
            stats.packed_merge_split_transition_dead,
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
        (
            "packed_dfa_source_rows_prepared",
            stats.packed_dfa_source_rows_prepared,
        ),
        (
            "packed_dfa_class_zero_probes",
            stats.packed_dfa_class_zero_probes,
        ),
        (
            "packed_dfa_class_zero_reusable_probes",
            stats.packed_dfa_class_zero_reusable_probes,
        ),
        (
            "packed_dfa_physical_target_probes",
            stats.packed_dfa_physical_target_probes,
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
        ("parent_path_queries", stats.parent_path_queries),
        ("cursor_key_queries", stats.cursor_key_queries),
        ("path_arena_nodes_created", stats.path_arena_nodes_created),
        (
            "cursor_key_reconstructions",
            stats.cursor_key_reconstructions,
        ),
        ("cursor_key_reverse_steps", stats.cursor_key_reverse_steps),
        ("term_units_materialized", stats.term_units_materialized),
        ("dfs_nodes_paged", stats.dfs_nodes_paged),
        ("dfs_nodes_eager", stats.dfs_nodes_eager),
        ("dfs_edge_page_requests", stats.dfs_edge_page_requests),
        ("dfs_edges_fetched", stats.dfs_edges_fetched),
        ("dfs_edges_consumed", stats.dfs_edges_consumed),
        ("dfs_edge_buffer_peak", stats.dfs_edge_buffer_peak),
        ("dfs_edge_buffer_spills", stats.dfs_edge_buffer_spills),
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
        ("byte", "triple_noop_from_sorted_terms") => run(&args, &terms, &queries, |items| {
            Noop::new(Noop::new(Noop::new(DynamicDawg::<()>::from_sorted_terms(
                items.iter().map(String::as_str),
            ))))
        }),
        ("byte", "double_array_trie_from_terms") => run(&args, &terms, &queries, |items| {
            DoubleArrayTrie::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("byte", "triple_noop_double_array_trie_from_terms") => {
            run(&args, &terms, &queries, |items| {
                Noop::new(Noop::new(Noop::new(DoubleArrayTrie::<()>::from_terms(
                    items.iter().map(String::as_str),
                ))))
            })
        }
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
        ("unicode", "double_array_trie_from_terms") => run(&args, &terms, &queries, |items| {
            DoubleArrayTrieChar::<()>::from_terms(items.iter().map(String::as_str))
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
        ("byte" | "unicode" | "u64", _) => fail(
            "--constructor must be from_terms, from_sorted_terms, \
                 triple_noop_from_sorted_terms (byte only), \
                 triple_noop_double_array_trie_from_terms (byte only), \
                 double_array_trie_from_terms (byte or unicode), or stream",
        ),
        _ => fail("--domain must be byte, unicode, or u64"),
    }
}
