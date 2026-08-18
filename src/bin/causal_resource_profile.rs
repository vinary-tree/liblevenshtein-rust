//! Work-counter driver for the dictionary-resource query path.

use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};
use libdictenstein::{
    causal_construction_stats, reset_causal_construction_stats, CausalConstructionStats,
};
use liblevenshtein::bindings::{MatchBatch, MatchTerm, QueryOrder, ResourceTransducer};
use liblevenshtein::transducer::Algorithm;
use liblevenshtein::{causal_perf_stats, reset_causal_perf_stats, CausalPerfStats};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Args {
    dictionary: PathBuf,
    queries: PathBuf,
    max_distance: usize,
    batch_size: usize,
    passes: usize,
    constructor: Constructor,
    algorithm: Algorithm,
}

#[derive(Clone, Copy)]
enum Constructor {
    Incremental,
    Batch,
}

impl Constructor {
    fn parse(value: &str) -> Self {
        match value {
            "incremental" => Self::Incremental,
            "batch" => Self::Batch,
            _ => fail("--constructor must be incremental or batch"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Incremental => "incremental",
            Self::Batch => "batch",
        }
    }
}

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("causal_resource_profile: {}", message.as_ref());
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut dictionary = None;
    let mut queries = None;
    let mut max_distance = None;
    let mut batch_size = 256usize;
    let mut passes = 1usize;
    let mut constructor = Constructor::Incremental;
    let mut algorithm = Algorithm::Standard;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv
            .next()
            .unwrap_or_else(|| fail(format!("{flag} requires a value")));
        match flag.as_str() {
            "--dictionary" => dictionary = Some(PathBuf::from(value)),
            "--queries" => queries = Some(PathBuf::from(value)),
            "--max-distance" => {
                max_distance = Some(
                    value
                        .parse()
                        .unwrap_or_else(|_| fail("--max-distance must be an integer")),
                )
            }
            "--batch-size" => {
                batch_size = value
                    .parse()
                    .unwrap_or_else(|_| fail("--batch-size must be a positive integer"))
            }
            "--passes" => {
                passes = value
                    .parse()
                    .unwrap_or_else(|_| fail("--passes must be a positive integer"))
            }
            "--constructor" => constructor = Constructor::parse(&value),
            "--algorithm" => {
                algorithm = value
                    .parse()
                    .unwrap_or_else(|error| fail(format!("invalid --algorithm: {error}")))
            }
            _ => fail(format!("unknown flag {flag:?}")),
        }
    }
    if batch_size == 0 {
        fail("--batch-size must be positive");
    }
    if passes == 0 {
        fail("--passes must be positive");
    }
    Args {
        dictionary: dictionary.unwrap_or_else(|| fail("--dictionary is required")),
        queries: queries.unwrap_or_else(|| fail("--queries is required")),
        max_distance: max_distance.unwrap_or_else(|| fail("--max-distance is required")),
        batch_size,
        passes,
        constructor,
        algorithm,
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

fn main() {
    let args = parse_args();
    let terms = read_lines(&args.dictionary);
    let queries = read_lines(&args.queries);

    let build_start = Instant::now();
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    match args.constructor {
        Constructor::Incremental => {
            for term in &terms {
                dictionary
                    .insert_text(term.as_bytes(), None)
                    .unwrap_or_else(|error| fail(format!("dictionary insertion failed: {error}")));
            }
        }
        Constructor::Batch => {
            dictionary
                .insert_text_batch(terms.iter().map(|term| (term.as_bytes(), None)))
                .unwrap_or_else(|error| fail(format!("batch construction failed: {error}")));
        }
    }
    let build_ns = build_start.elapsed().as_nanos();
    let resource = dictionary.resource();
    let live_transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), args.algorithm)
            .unwrap_or_else(|error| fail(format!("resource validation failed: {error}")))
    };

    reset_causal_perf_stats();
    reset_causal_construction_stats();
    let transducer = live_transducer
        .snapshot()
        .unwrap_or_else(|error| fail(format!("resource snapshot failed: {error}")));
    let query_start = Instant::now();
    let mut batch = MatchBatch::default();
    let mut matches = 0u64;
    let mut term_bytes = 0u64;
    let mut distance_sum = 0u64;
    let mut checksum = 0u64;
    let mut order_checksum = FNV_OFFSET;
    let mut batches = 0u64;
    for pass in 0..args.passes {
        for query in &queries {
            for byte in (pass as u64).to_le_bytes() {
                order_checksum = hash_byte(order_checksum, byte);
            }
            for byte in query.bytes() {
                order_checksum = hash_byte(order_checksum, byte);
            }
            order_checksum = hash_byte(order_checksum, 0xff);
            let mut cursor = transducer
                .query_utf8(query, args.max_distance, QueryOrder::Traversal)
                .unwrap_or_else(|error| fail(format!("query creation failed: {error}")));
            loop {
                let count = cursor
                    .next_batch(&mut batch, args.batch_size)
                    .unwrap_or_else(|error| fail(format!("query traversal failed: {error}")));
                if count == 0 {
                    break;
                }
                batches = batches.saturating_add(1);
                for item in batch.as_slice() {
                    let MatchTerm::Utf8(term) = &item.term else {
                        fail("Unicode resource returned a non-UTF-8 term");
                    };
                    matches = matches.saturating_add(1);
                    term_bytes = term_bytes.saturating_add(term.len() as u64);
                    distance_sum = distance_sum.saturating_add(item.distance as u64);
                    let hash = entry_hash(term, item.distance);
                    checksum = checksum.wrapping_add(hash);
                    for byte in hash.to_le_bytes() {
                        order_checksum = hash_byte(order_checksum, byte);
                    }
                }
            }
        }
    }
    let query_ns = query_start.elapsed().as_nanos();
    // Reclamation is deliberately outside `query_ns`: release every producer
    // and consumer owner so the provider's synchronous arena-drop counters are
    // observable without folding teardown into query latency.
    drop(transducer);
    drop(live_transducer);
    drop(resource);
    drop(dictionary);
    print_json(
        &args,
        terms.len(),
        queries.len(),
        build_ns,
        query_ns,
        matches,
        term_bytes,
        distance_sum,
        checksum,
        order_checksum,
        batches,
        causal_perf_stats(),
        causal_construction_stats(),
    );
}

#[allow(clippy::too_many_arguments)]
fn print_json(
    args: &Args,
    term_count: usize,
    query_count: usize,
    build_ns: u128,
    query_ns: u128,
    matches: u64,
    term_bytes: u64,
    distance_sum: u64,
    checksum: u64,
    order_checksum: u64,
    batches: u64,
    consumer: CausalPerfStats,
    provider: CausalConstructionStats,
) {
    let mut json = String::with_capacity(4096);
    writeln!(&mut json, "{{").unwrap();
    writeln!(
        &mut json,
        "  \"schema\": \"liblevenshtein.causal-resource-work.v2\","
    )
    .unwrap();
    writeln!(&mut json, "  \"term_count\": {term_count},").unwrap();
    writeln!(&mut json, "  \"query_count\": {query_count},").unwrap();
    writeln!(&mut json, "  \"max_distance\": {},", args.max_distance).unwrap();
    writeln!(&mut json, "  \"batch_size\": {},", args.batch_size).unwrap();
    writeln!(&mut json, "  \"passes\": {},", args.passes).unwrap();
    writeln!(
        &mut json,
        "  \"constructor\": \"{}\",",
        args.constructor.name()
    )
    .unwrap();
    writeln!(&mut json, "  \"algorithm\": \"{}\",", args.algorithm.name()).unwrap();
    writeln!(&mut json, "  \"build_ns\": {build_ns},").unwrap();
    writeln!(&mut json, "  \"query_ns\": {query_ns},").unwrap();
    writeln!(&mut json, "  \"matches\": {matches},").unwrap();
    writeln!(&mut json, "  \"term_bytes\": {term_bytes},").unwrap();
    writeln!(&mut json, "  \"distance_sum\": {distance_sum},").unwrap();
    writeln!(&mut json, "  \"checksum_u64\": {checksum},").unwrap();
    writeln!(&mut json, "  \"order_checksum_u64\": {order_checksum},").unwrap();
    writeln!(&mut json, "  \"nonempty_batches\": {batches},").unwrap();
    writeln!(&mut json, "  \"consumer_work\": {{").unwrap();
    let consumer_fields = [
        (
            "dictionary_intersections",
            consumer.dictionary_intersections,
        ),
        ("final_checks", consumer.final_checks),
        ("edges_enumerated", consumer.edges_enumerated),
        ("transition_attempts", consumer.transition_attempts),
        ("transition_accepted", consumer.transition_accepted),
        ("packed_standard_queries", consumer.packed_standard_queries),
        ("positional_unit_queries", consumer.positional_unit_queries),
        (
            "packed_standard_transition_attempts",
            consumer.packed_standard_transition_attempts,
        ),
        (
            "packed_standard_transition_dead",
            consumer.packed_standard_transition_dead,
        ),
        ("packed_dfa_queries", consumer.packed_dfa_queries),
        (
            "packed_dfa_transition_hits",
            consumer.packed_dfa_transition_hits,
        ),
        (
            "packed_dfa_transition_misses",
            consumer.packed_dfa_transition_misses,
        ),
        (
            "packed_dfa_states_interned",
            consumer.packed_dfa_states_interned,
        ),
        (
            "generated_transition_hits",
            consumer.generated_transition_hits,
        ),
        (
            "generated_transition_misses",
            consumer.generated_transition_misses,
        ),
        (
            "generated_product_expansions",
            consumer.generated_product_expansions,
        ),
        (
            "generated_product_identity_expansions",
            consumer.generated_product_identity_expansions,
        ),
        (
            "generated_product_unique_expansions",
            consumer.generated_product_unique_expansions,
        ),
        (
            "generated_product_repeated_expansions",
            consumer.generated_product_repeated_expansions,
        ),
        ("characteristic_vectors", consumer.characteristic_vectors),
        ("characteristic_units", consumer.characteristic_units),
        ("state_bytes_copied", consumer.state_bytes_copied),
        ("state_bytes_enqueued", consumer.state_bytes_enqueued),
        ("pool_misses", consumer.pool_misses),
        ("matches_materialized", consumer.matches_materialized),
        (
            "foreign_is_final_callbacks",
            consumer.foreign_is_final_callbacks,
        ),
        ("foreign_edge_callbacks", consumer.foreign_edge_callbacks),
        ("foreign_edge_pages", consumer.foreign_edge_pages),
        (
            "foreign_edge_descriptors",
            consumer.foreign_edge_descriptors,
        ),
        ("foreign_node_cache_hits", consumer.foreign_node_cache_hits),
        (
            "foreign_node_cache_misses",
            consumer.foreign_node_cache_misses,
        ),
        (
            "foreign_pending_cursors_emitted",
            consumer.foreign_pending_cursors_emitted,
        ),
        (
            "foreign_ready_cursors_emitted",
            consumer.foreign_ready_cursors_emitted,
        ),
        (
            "foreign_edge_cursor_promotions",
            consumer.foreign_edge_cursor_promotions,
        ),
        (
            "foreign_pending_descriptor_loads",
            consumer.foreign_pending_descriptor_loads,
        ),
        (
            "foreign_ready_descriptor_reads",
            consumer.foreign_ready_descriptor_reads,
        ),
        (
            "foreign_child_directory_lookups",
            consumer.foreign_child_directory_lookups,
        ),
        ("foreign_leaf_expansions", consumer.foreign_leaf_expansions),
        ("foreign_graph_decodes", consumer.foreign_graph_decodes),
    ];
    write_fields(&mut json, &consumer_fields);
    writeln!(&mut json, "  }},").unwrap();
    writeln!(&mut json, "  \"provider_work\": {{").unwrap();
    let provider_fields = [
        ("snapshots_created", provider.resource_snapshots_created),
        ("arena_locks", provider.resource_arena_locks),
        ("is_final_calls", provider.resource_is_final_calls),
        ("value_calls", provider.resource_value_calls),
        ("graph_projections", provider.resource_graph_projections),
        ("graph_calls", provider.resource_graph_calls),
        ("graph_value_calls", provider.resource_graph_value_calls),
        ("edges_calls", provider.resource_edges_calls),
        ("edge_cache_misses", provider.resource_edge_cache_misses),
        (
            "native_edges_enumerated",
            provider.resource_native_edges_enumerated,
        ),
        ("nodes_materialized", provider.resource_nodes_materialized),
        ("descriptors_cloned", provider.resource_descriptors_cloned),
        ("nodes_reclaimed", provider.resource_nodes_reclaimed),
        ("reclaim_nanos", provider.resource_reclaim_nanos),
        ("reclaim_max_nanos", provider.resource_reclaim_max_nanos),
    ];
    write_fields(&mut json, &provider_fields);
    writeln!(&mut json, "  }}").unwrap();
    writeln!(&mut json, "}}").unwrap();
    print!("{json}");
}

fn write_fields(json: &mut String, fields: &[(&str, u64)]) {
    for (index, (name, value)) in fields.iter().enumerate() {
        let comma = if index + 1 == fields.len() { "" } else { "," };
        writeln!(json, "    \"{name}\": {value}{comma}").unwrap();
    }
}
