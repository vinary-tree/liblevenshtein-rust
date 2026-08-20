//! Deterministic all-backend no-regression evidence matrix.
//!
//! Every applicable production `DictionaryNode` family is constructed and
//! queried through one unit-generic kernel. Unsupported family/unit cells are
//! emitted as explicit `inapplicable` rows rather than disappearing from the
//! evidence set.

use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieChar};
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgChar, DynamicDawgU64};
use libdictenstein::pathmap::{PathMapDictionary, PathMapDictionaryChar};
use libdictenstein::scdawg::{Scdawg, ScdawgChar};
use libdictenstein::suffix_automaton::{SuffixAutomaton, SuffixAutomatonChar};
use libdictenstein::{
    CharUnit, Dictionary, DictionaryNode, DictionaryTraversalRoot, PersistentARTrie,
    PersistentARTrieChar, PersistentARTrieU64, PersistentScdawg, PersistentScdawgChar,
    PersistentSuffixAutomaton, PersistentSuffixAutomatonChar, PersistentSuffixTree,
    PersistentSuffixTreeChar, PersistentVocabARTrie, SyncStrategy,
};
use liblevenshtein::transducer::substitution_policy::{SubstitutionPolicyFor, Unrestricted};
use liblevenshtein::transducer::{Algorithm, Transducer};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const SCHEMA: &str = "liblevenshtein.backend-propagation-matrix.v1";
const EXPECTED_ROWS: usize = Family::ALL.len() * Domain::ALL.len() * 5;
const DEFAULT_TERMS: usize = 256;
const DEFAULT_QUERIES: usize = 64;
const DEFAULT_REPETITIONS: usize = 1;
const DEFAULT_DISTANCE: usize = 2;
const DEFAULT_MAX_ALLOCATED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_RSS_BYTES: u64 = 8 * 1024 * 1024 * 1024;

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

fn update_peak(candidate: u64) {
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while candidate > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

// SAFETY: all allocation operations preserve the pointer/layout contract by
// delegating to `System`; relaxed counters are observational only.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed)
                + layout.size() as u64;
            update_peak(live);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed)
                + layout.size() as u64;
            update_peak(live);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        System.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = System.realloc(pointer, layout, new_size);
        if !replacement.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            let live = if new_size >= layout.size() {
                LIVE_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed)
                    + (new_size - layout.size()) as u64
            } else {
                LIVE_BYTES.fetch_sub((layout.size() - new_size) as u64, Ordering::Relaxed)
                    - (layout.size() - new_size) as u64
            };
            update_peak(live);
        }
        replacement
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug)]
struct AllocSnapshot {
    allocations: u64,
    allocated_bytes: u64,
    deallocations: u64,
    deallocated_bytes: u64,
    live_bytes: u64,
}

impl AllocSnapshot {
    fn read() -> Self {
        Self {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
            deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
            live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StageMetrics {
    elapsed: Duration,
    allocations: u64,
    allocated_bytes: u64,
    deallocations: u64,
    deallocated_bytes: u64,
    peak_live_growth: u64,
    rss_bytes: Option<u64>,
}

struct StageStart {
    instant: Instant,
    allocations: AllocSnapshot,
}

fn begin_stage() -> StageStart {
    let allocations = AllocSnapshot::read();
    PEAK_LIVE_BYTES.store(allocations.live_bytes, Ordering::Relaxed);
    StageStart {
        instant: Instant::now(),
        allocations,
    }
}

fn finish_stage(start: StageStart) -> StageMetrics {
    let elapsed = start.instant.elapsed();
    let finish = AllocSnapshot::read();
    StageMetrics {
        elapsed,
        allocations: finish.allocations - start.allocations.allocations,
        allocated_bytes: finish.allocated_bytes - start.allocations.allocated_bytes,
        deallocations: finish.deallocations - start.allocations.deallocations,
        deallocated_bytes: finish.deallocated_bytes - start.allocations.deallocated_bytes,
        peak_live_growth: PEAK_LIVE_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(start.allocations.live_bytes),
        rss_bytes: current_rss_bytes(),
    }
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))?
        .split_ascii_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kib.checked_mul(1024)
}

#[cfg(not(target_os = "linux"))]
fn current_rss_bytes() -> Option<u64> {
    None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Domain {
    Byte,
    Char,
    U64,
}

impl Domain {
    const ALL: [Self; 3] = [Self::Byte, Self::Char, Self::U64];

    const fn name(self) -> &'static str {
        match self {
            Self::Byte => "byte",
            Self::Char => "char",
            Self::U64 => "u64",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family {
    DynamicDawg,
    DoubleArrayTrie,
    PathMap,
    SuffixAutomaton,
    Scdawg,
    PersistentSuffixAutomaton,
    PersistentSuffixTree,
    PersistentScdawg,
    PersistentArtrie,
    PersistentVocabulary,
}

impl Family {
    const ALL: [Self; 10] = [
        Self::DynamicDawg,
        Self::DoubleArrayTrie,
        Self::PathMap,
        Self::SuffixAutomaton,
        Self::Scdawg,
        Self::PersistentSuffixAutomaton,
        Self::PersistentSuffixTree,
        Self::PersistentScdawg,
        Self::PersistentArtrie,
        Self::PersistentVocabulary,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::DynamicDawg => "dynamic-dawg",
            Self::DoubleArrayTrie => "double-array-trie",
            Self::PathMap => "pathmap",
            Self::SuffixAutomaton => "suffix-automaton",
            Self::Scdawg => "scdawg",
            Self::PersistentSuffixAutomaton => "persistent-suffix-automaton",
            Self::PersistentSuffixTree => "persistent-suffix-tree",
            Self::PersistentScdawg => "persistent-scdawg",
            Self::PersistentArtrie => "persistent-artrie",
            Self::PersistentVocabulary => "persistent-vocabulary",
        }
    }

    const fn supports(self, domain: Domain) -> bool {
        match self {
            Self::DynamicDawg | Self::PersistentArtrie => true,
            Self::PersistentVocabulary => matches!(domain, Domain::Char),
            _ => !matches!(domain, Domain::U64),
        }
    }

    const fn suffix_based(self) -> bool {
        matches!(
            self,
            Self::SuffixAutomaton
                | Self::Scdawg
                | Self::PersistentSuffixAutomaton
                | Self::PersistentSuffixTree
                | Self::PersistentScdawg
        )
    }
}

#[derive(Clone, Copy)]
struct AlgorithmSpec {
    name: &'static str,
    algorithm: Algorithm,
}

impl AlgorithmSpec {
    const ALL: [Self; 4] = [
        Self {
            name: "standard",
            algorithm: Algorithm::Standard,
        },
        Self {
            name: "osa-transposition",
            algorithm: Algorithm::Transposition,
        },
        Self {
            name: "merge-and-split",
            algorithm: Algorithm::MergeAndSplit,
        },
        Self {
            name: "damerau-levenshtein",
            algorithm: Algorithm::DamerauLevenshtein,
        },
    ];
}

#[derive(Debug)]
struct Corpus<U: CharUnit> {
    terms: Vec<String>,
    queries: Vec<Vec<U>>,
    checksum: u64,
    max_term_units: usize,
}

impl<U: CharUnit> Corpus<U> {
    fn new(term_count: usize, query_count: usize) -> Self {
        let terms = generate_terms(term_count);
        let queries = generate_queries(&terms, query_count)
            .iter()
            .map(|query| U::from_str(query))
            .collect();
        let mut checksum = 0xcbf2_9ce4_8422_2325;
        let mut max_term_units = 0;
        for term in &terms {
            let units = U::from_str(term);
            max_term_units = max_term_units.max(units.len());
            checksum = mix_units(checksum, &units);
        }
        Self {
            terms,
            queries,
            checksum,
            max_term_units,
        }
    }
}

fn generate_terms(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| {
            let stem = match index % 7 {
                0 => "alpha",
                1 => "café",
                2 => "東京",
                3 => "naïve",
                4 => "delta",
                5 => "λambda",
                _ => "emoji🙂",
            };
            format!("{stem}-{:03}-{:06}", index % 97, index)
        })
        .collect()
}

fn generate_queries(terms: &[String], count: usize) -> Vec<String> {
    (0..count)
        .map(|index| {
            let term = &terms[(index.wrapping_mul(7919)) % terms.len()];
            if index % 3 == 0 {
                return term.clone();
            }
            let mut chars: Vec<char> = term.chars().collect();
            if index % 3 == 1 && chars.len() > 2 {
                let middle = chars.len() / 2;
                chars.swap(middle - 1, middle);
            } else if let Some(last) = chars.last_mut() {
                *last = if *last == 'x' { 'y' } else { 'x' };
            }
            chars.into_iter().collect()
        })
        .collect()
}

#[inline]
fn mix(checksum: u64, value: u64) -> u64 {
    checksum
        .wrapping_mul(0x0000_0100_0000_01b3)
        .wrapping_add(value)
}

fn mix_units<U: CharUnit>(mut checksum: u64, units: &[U]) -> u64 {
    checksum = mix(checksum, units.len() as u64);
    for unit in units {
        checksum = mix(checksum, unit.hash_to_u64());
    }
    checksum
}

/// Borrow a dictionary while preserving its native traversal-root override.
/// This lets all algorithms share one constructed backend without falling back
/// from snapshot cursors to an owned root.
struct BorrowedDictionary<'a, D>(&'a D);

impl<D: Dictionary> Dictionary for BorrowedDictionary<'_, D> {
    type Node = D::Node;

    fn root(&self) -> Self::Node {
        self.0.root()
    }

    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        self.0.traversal_root()
    }

    fn contains(&self, term: &str) -> bool {
        self.0.contains(term)
    }

    fn len(&self) -> Option<usize> {
        self.0.len()
    }

    fn sync_strategy(&self) -> SyncStrategy {
        self.0.sync_strategy()
    }

    fn is_suffix_based(&self) -> bool {
        self.0.is_suffix_based()
    }
}

#[derive(Clone, Debug)]
struct Row {
    replicate: usize,
    arm: String,
    profile: String,
    cell_order: usize,
    algorithm_order: Option<usize>,
    family: &'static str,
    backend: String,
    domain: &'static str,
    algorithm: &'static str,
    stage: &'static str,
    applicability: &'static str,
    reason: &'static str,
    terms: usize,
    operations: usize,
    result_count: usize,
    checksum: u64,
    metrics: Option<StageMetrics>,
    max_terms: usize,
    max_operations: usize,
    max_results: usize,
    max_allocated_bytes: u64,
    max_rss_bytes: u64,
    binary_sha: String,
}

impl Row {
    fn print_csv(&self) {
        let (
            elapsed_ns,
            ns_per_operation,
            allocations,
            allocated_bytes,
            deallocations,
            deallocated_bytes,
            peak_live_growth,
            rss_bytes,
        ) = match self.metrics {
            Some(metrics) => (
                metrics.elapsed.as_nanos().to_string(),
                if self.operations == 0 {
                    String::new()
                } else {
                    format!(
                        "{:.6}",
                        metrics.elapsed.as_nanos() as f64 / self.operations as f64
                    )
                },
                metrics.allocations.to_string(),
                metrics.allocated_bytes.to_string(),
                metrics.deallocations.to_string(),
                metrics.deallocated_bytes.to_string(),
                metrics.peak_live_growth.to_string(),
                metrics
                    .rss_bytes
                    .map_or_else(String::new, |value| value.to_string()),
            ),
            None => (
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ),
        };
        println!(
            "{SCHEMA},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.replicate,
            self.arm,
            self.profile,
            self.cell_order,
            self.algorithm_order.map_or_else(String::new, |value| value.to_string()),
            self.family,
            self.backend,
            self.domain,
            self.algorithm,
            self.stage,
            self.applicability,
            self.reason,
            self.terms,
            self.operations,
            self.result_count,
            elapsed_ns,
            ns_per_operation,
            allocations,
            allocated_bytes,
            deallocations,
            deallocated_bytes,
            peak_live_growth,
            rss_bytes,
            self.checksum,
            self.max_terms,
            self.max_operations,
            self.max_results,
            self.max_allocated_bytes,
            self.max_rss_bytes,
            self.binary_sha,
            EXPECTED_ROWS,
        );
    }
}

#[derive(Clone, Debug)]
struct Config {
    replicate: usize,
    arm: String,
    profile: String,
    terms: usize,
    queries: usize,
    repetitions: usize,
    distance: usize,
    max_allocated_bytes: u64,
    max_rss_bytes: u64,
    binary_sha: String,
    print_header: bool,
    header_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            replicate: 1,
            arm: "treatment".to_owned(),
            profile: "treatment".to_owned(),
            terms: DEFAULT_TERMS,
            queries: DEFAULT_QUERIES,
            repetitions: DEFAULT_REPETITIONS,
            distance: DEFAULT_DISTANCE,
            max_allocated_bytes: DEFAULT_MAX_ALLOCATED_BYTES,
            max_rss_bytes: DEFAULT_MAX_RSS_BYTES,
            binary_sha: "unrecorded".to_owned(),
            print_header: true,
            header_only: false,
        }
    }
}

fn validate_metrics(metrics: StageMetrics, config: &Config, stage: &str) {
    assert!(
        metrics.allocated_bytes <= config.max_allocated_bytes,
        "{stage} allocated {} bytes, above hard bound {}",
        metrics.allocated_bytes,
        config.max_allocated_bytes
    );
    if let Some(rss) = metrics.rss_bytes {
        assert!(
            rss <= config.max_rss_bytes,
            "{stage} process RSS {rss} is above hard bound {}",
            config.max_rss_bytes
        );
    }
}

fn measure_backend<D, U, F>(
    family: Family,
    domain: Domain,
    cell_order: usize,
    corpus: &Corpus<U>,
    config: &Config,
    factory: F,
) -> Vec<Row>
where
    D: Dictionary,
    D::Node: DictionaryNode<Unit = U>,
    U: CharUnit,
    Unrestricted: SubstitutionPolicyFor<U>,
    F: FnOnce() -> D,
{
    let backend = format!("{}-{}", family.name(), domain.name());
    let start = begin_stage();
    let dictionary = factory();
    let construction = finish_stage(start);
    validate_metrics(construction, config, &format!("{backend}/construction"));
    assert_eq!(
        dictionary.len(),
        Some(corpus.terms.len()),
        "{backend} length"
    );
    for term in &corpus.terms {
        assert!(dictionary.contains(term), "{backend} lost term {term:?}");
    }

    let max_operations = config
        .queries
        .checked_mul(config.repetitions)
        .expect("query operation bound fits usize");
    let mut rows = Vec::with_capacity(5);
    rows.push(Row {
        replicate: config.replicate,
        arm: config.arm.clone(),
        profile: config.profile.clone(),
        cell_order,
        algorithm_order: None,
        family: family.name(),
        backend: backend.clone(),
        domain: domain.name(),
        algorithm: "none",
        stage: "construction",
        applicability: "applicable",
        reason: "native-production-constructor",
        terms: corpus.terms.len(),
        operations: corpus.terms.len(),
        result_count: corpus.terms.len(),
        checksum: corpus.checksum,
        metrics: Some(construction),
        max_terms: config.terms,
        max_operations: config.terms,
        max_results: config.terms,
        max_allocated_bytes: config.max_allocated_bytes,
        max_rss_bytes: config.max_rss_bytes,
        binary_sha: config.binary_sha.clone(),
    });

    let algorithm_offset = (config.replicate + cell_order) % AlgorithmSpec::ALL.len();
    for algorithm_order in 0..AlgorithmSpec::ALL.len() {
        let spec =
            AlgorithmSpec::ALL[(algorithm_offset + algorithm_order) % AlgorithmSpec::ALL.len()];
        let transducer = Transducer::new(BorrowedDictionary(&dictionary), spec.algorithm);
        let mut result_count = 0usize;
        let mut checksum = 0xcbf2_9ce4_8422_2325;
        let start = begin_stage();
        for repetition in 0..config.repetitions {
            for (query_index, query) in corpus.queries.iter().enumerate() {
                checksum = mix(checksum, repetition as u64);
                checksum = mix(checksum, query_index as u64);
                for candidate in transducer.query_units_with_distance(query, config.distance) {
                    assert!(candidate.distance <= config.distance);
                    assert!(candidate.term.len() <= corpus.max_term_units);
                    result_count = result_count
                        .checked_add(1)
                        .expect("result count remains bounded");
                    checksum = mix(checksum, candidate.distance as u64);
                    checksum = mix_units(checksum, &candidate.term);
                }
            }
        }
        let metrics = finish_stage(start);
        validate_metrics(
            metrics,
            config,
            &format!("{backend}/{}/{}/query", domain.name(), spec.name),
        );
        let max_results = max_operations
            .saturating_mul(corpus.terms.len())
            .saturating_mul(corpus.max_term_units.max(1))
            .saturating_mul(corpus.max_term_units.max(1));
        assert!(
            result_count <= max_results,
            "{backend}/{}/{} results exceeded bound: {result_count} > {max_results}",
            domain.name(),
            spec.name,
        );
        rows.push(Row {
            replicate: config.replicate,
            arm: config.arm.clone(),
            profile: config.profile.clone(),
            cell_order,
            algorithm_order: Some(algorithm_order),
            family: family.name(),
            backend: backend.clone(),
            domain: domain.name(),
            algorithm: spec.name,
            stage: "query",
            applicability: "applicable",
            reason: if family.suffix_based() {
                "suffix-language-semantics"
            } else {
                "finite-term-language-semantics"
            },
            terms: corpus.terms.len(),
            operations: max_operations,
            result_count,
            checksum: black_box(checksum),
            metrics: Some(metrics),
            max_terms: config.terms,
            max_operations,
            max_results,
            max_allocated_bytes: config.max_allocated_bytes,
            max_rss_bytes: config.max_rss_bytes,
            binary_sha: config.binary_sha.clone(),
        });
    }
    rows
}

fn inapplicable_rows(
    family: Family,
    domain: Domain,
    cell_order: usize,
    config: &Config,
) -> Vec<Row> {
    let reason = match family {
        Family::PersistentVocabulary => "vocabulary-keys-are-unicode-scalars",
        _ => "no-production-u64-dictionary-node",
    };
    let backend = format!("{}-{}", family.name(), domain.name());
    std::iter::once(("none", "construction"))
        .chain(
            AlgorithmSpec::ALL
                .iter()
                .map(|algorithm| (algorithm.name, "query")),
        )
        .enumerate()
        .map(|(index, (algorithm, stage))| Row {
            replicate: config.replicate,
            arm: config.arm.clone(),
            profile: config.profile.clone(),
            cell_order,
            algorithm_order: (stage == "query").then(|| index - 1),
            family: family.name(),
            backend: backend.clone(),
            domain: domain.name(),
            algorithm,
            stage,
            applicability: "inapplicable",
            reason,
            terms: config.terms,
            operations: 0,
            result_count: 0,
            checksum: 0,
            metrics: None,
            max_terms: config.terms,
            max_operations: config.queries.saturating_mul(config.repetitions),
            max_results: 0,
            max_allocated_bytes: config.max_allocated_bytes,
            max_rss_bytes: config.max_rss_bytes,
            binary_sha: config.binary_sha.clone(),
        })
        .collect()
}

struct MatrixRun<'a> {
    byte: &'a Corpus<u8>,
    chars: &'a Corpus<char>,
    u64s: &'a Corpus<u64>,
    scratch: &'a Path,
    config: &'a Config,
}

fn run_supported(
    family: Family,
    domain: Domain,
    cell_order: usize,
    run: &MatrixRun<'_>,
) -> Vec<Row> {
    let MatrixRun {
        byte,
        chars,
        u64s,
        scratch,
        config,
    } = run;
    match (family, domain) {
        (Family::DynamicDawg, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                DynamicDawg::<()>::from_terms(&byte.terms)
            })
        }
        (Family::DynamicDawg, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                DynamicDawgChar::<()>::from_terms(&chars.terms)
            })
        }
        (Family::DynamicDawg, Domain::U64) => {
            measure_backend(family, domain, cell_order, u64s, config, || {
                DynamicDawgU64::<()>::from_terms(&u64s.terms)
            })
        }
        (Family::DoubleArrayTrie, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                DoubleArrayTrie::from_terms(&byte.terms)
            })
        }
        (Family::DoubleArrayTrie, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                DoubleArrayTrieChar::from_terms(&chars.terms)
            })
        }
        (Family::PathMap, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                PathMapDictionary::<()>::from_terms(&byte.terms)
            })
        }
        (Family::PathMap, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                PathMapDictionaryChar::<()>::from_terms(&chars.terms)
            })
        }
        (Family::SuffixAutomaton, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                SuffixAutomaton::<()>::from_texts(&byte.terms)
            })
        }
        (Family::SuffixAutomaton, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                SuffixAutomatonChar::<()>::from_texts(&chars.terms)
            })
        }
        (Family::Scdawg, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                Scdawg::<()>::from_terms(&byte.terms)
            })
        }
        (Family::Scdawg, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                ScdawgChar::<()>::from_terms(&chars.terms)
            })
        }
        (Family::PersistentSuffixAutomaton, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                PersistentSuffixAutomaton::<()>::from_texts(&byte.terms)
            })
        }
        (Family::PersistentSuffixAutomaton, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                PersistentSuffixAutomatonChar::<()>::from_texts(&chars.terms)
            })
        }
        (Family::PersistentSuffixTree, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                PersistentSuffixTree::<()>::from_texts(&byte.terms)
            })
        }
        (Family::PersistentSuffixTree, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                PersistentSuffixTreeChar::<()>::from_texts(&chars.terms)
            })
        }
        (Family::PersistentScdawg, Domain::Byte) => {
            measure_backend(family, domain, cell_order, byte, config, || {
                PersistentScdawg::<()>::from_terms(&byte.terms)
            })
        }
        (Family::PersistentScdawg, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                PersistentScdawgChar::<()>::from_terms(&chars.terms)
            })
        }
        (Family::PersistentArtrie, Domain::Byte) => measure_backend(
            family,
            domain,
            cell_order,
            byte,
            config,
            #[allow(deprecated)]
            || {
                let dictionary = PersistentARTrie::<()>::new();
                for term in &byte.terms {
                    assert!(dictionary.insert(term));
                }
                dictionary
            },
        ),
        (Family::PersistentArtrie, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                chars
                    .terms
                    .iter()
                    .map(String::as_str)
                    .collect::<PersistentARTrieChar<()>>()
            })
        }
        (Family::PersistentArtrie, Domain::U64) => {
            measure_backend(family, domain, cell_order, u64s, config, || {
                PersistentARTrieU64::<()>::from_terms(&u64s.terms)
            })
        }
        (Family::PersistentVocabulary, Domain::Char) => {
            measure_backend(family, domain, cell_order, chars, config, || {
                let path = scratch.join("persistent-vocabulary.vocab");
                let dictionary = PersistentVocabARTrie::create(path)
                    .expect("create persistent vocabulary evidence backend");
                for term in &chars.terms {
                    dictionary
                        .insert(term)
                        .expect("insert vocabulary evidence term");
                }
                dictionary
            })
        }
        _ => unreachable!("run_supported called for an unsupported cell"),
    }
}

const CONTROL_KNOBS: &[&str] = &[
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_SNAPSHOT_CURSORS",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_DFS_EDGE_PAGING",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_TRAVERSAL_BUFFER_REUSE",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_CURSOR_KEY_RECONSTRUCTION",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_CLASS_ZERO_ROW_CACHE",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_EXACT_COST_PACKED_LANES",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_DFA",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_WORD_KERNEL",
    "LIBLEVENSHTEIN_CAUSAL_USE_LEGACY_PACKED_MASKS",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_STATIC_PACKED_DISPATCH",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_STANDARD",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_STATIC_PACKED_ROWS",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_MERGE_SPLIT",
    "LIBLEVENSHTEIN_CAUSAL_DISABLE_PACKED_OSA",
    "LIBLEVENSHTEIN_CAUSAL_USE_MONOLITHIC_UNIT_STEP",
    "LIBLEVENSHTEIN_CAUSAL_USE_JAGGED_GENERATED_TARGETS",
    "LIBLEVENSHTEIN_CAUSAL_USE_DICTIONARY_LABEL_CHARACTERISTIC_INDEX",
    "LIBLEVENSHTEIN_CAUSAL_USE_GLOBAL_SUBSUMPTION_SCAN",
    "LIBDICTENSTEIN_CAUSAL_USE_CHECKED_DAT_CURSOR_EDGES",
];

fn configure_profile(profile: &str) {
    match profile {
        "treatment" => {
            for variable in CONTROL_KNOBS {
                std::env::remove_var(variable);
            }
        }
        "legacy-shared-kernels" => {
            for variable in CONTROL_KNOBS {
                std::env::set_var(variable, "1");
            }
        }
        "inherited" => {}
        _ => panic!("unknown profile: {profile}"),
    }
}

fn run_matrix(config: &Config, scratch: &Path) -> Vec<Row> {
    assert!(config.terms > 0);
    assert!(config.queries > 0 && config.queries <= config.terms);
    let byte = Corpus::<u8>::new(config.terms, config.queries);
    let chars = Corpus::<char>::new(config.terms, config.queries);
    let u64s = Corpus::<u64>::new(config.terms, config.queries);
    let run = MatrixRun {
        byte: &byte,
        chars: &chars,
        u64s: &u64s,
        scratch,
        config,
    };
    let cell_count = Family::ALL.len() * Domain::ALL.len();
    let offset = config.replicate % cell_count;
    let mut rows = Vec::with_capacity(EXPECTED_ROWS);
    for cell_order in 0..cell_count {
        let cell = (offset + cell_order) % cell_count;
        let family = Family::ALL[cell / Domain::ALL.len()];
        let domain = Domain::ALL[cell % Domain::ALL.len()];
        if family.supports(domain) {
            rows.extend(run_supported(family, domain, cell_order, &run));
        } else {
            rows.extend(inapplicable_rows(family, domain, cell_order, config));
        }
    }
    assert_eq!(rows.len(), EXPECTED_ROWS);
    rows
}

fn positive_usize(flag: &str, value: String) -> usize {
    value
        .parse::<usize>()
        .ok()
        .filter(|&value| value > 0)
        .unwrap_or_else(|| panic!("{flag} requires a positive integer"))
}

fn parse_config() -> Config {
    let mut config = Config::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--replicate" => {
                config.replicate = positive_usize(
                    "--replicate",
                    arguments.next().expect("--replicate requires a value"),
                )
            }
            "--arm" => config.arm = arguments.next().expect("--arm requires a value"),
            "--profile" => config.profile = arguments.next().expect("--profile requires a value"),
            "--terms" => {
                config.terms = positive_usize(
                    "--terms",
                    arguments.next().expect("--terms requires a value"),
                )
            }
            "--queries" => {
                config.queries = positive_usize(
                    "--queries",
                    arguments.next().expect("--queries requires a value"),
                )
            }
            "--repetitions" => {
                config.repetitions = positive_usize(
                    "--repetitions",
                    arguments.next().expect("--repetitions requires a value"),
                )
            }
            "--distance" => {
                config.distance = arguments
                    .next()
                    .expect("--distance requires a value")
                    .parse()
                    .expect("--distance requires a non-negative integer")
            }
            "--max-allocated-bytes" => {
                config.max_allocated_bytes = arguments
                    .next()
                    .expect("--max-allocated-bytes requires a value")
                    .parse()
                    .expect("--max-allocated-bytes requires an integer")
            }
            "--max-rss-bytes" => {
                config.max_rss_bytes = arguments
                    .next()
                    .expect("--max-rss-bytes requires a value")
                    .parse()
                    .expect("--max-rss-bytes requires an integer")
            }
            "--binary-sha" => {
                config.binary_sha = arguments.next().expect("--binary-sha requires a value")
            }
            "--no-header" => config.print_header = false,
            "--header-only" => config.header_only = true,
            "--help" | "-h" => {
                println!("usage: backend_propagation_matrix [--replicate N] [--arm LABEL] [--profile treatment|legacy-shared-kernels|inherited] [--terms N] [--queries N] [--repetitions N] [--distance N] [--max-allocated-bytes N] [--max-rss-bytes N] [--binary-sha HEX] [--no-header|--header-only]");
                std::process::exit(0);
            }
            unknown => panic!("unknown argument: {unknown}"),
        }
    }
    assert!(!config.arm.contains(','), "--arm cannot contain a comma");
    assert!(
        config.queries <= config.terms,
        "queries cannot exceed terms"
    );
    assert!(config.distance <= Algorithm::MAX_DAMERAU_DISTANCE);
    assert!(config.max_allocated_bytes > 0 && config.max_rss_bytes > 0);
    config
}

fn print_header() {
    println!("schema,replicate,arm,profile,cell_order,algorithm_order,backend_family,backend,unit_domain,algorithm,stage,applicability,reason,terms,operations,result_count,elapsed_ns,ns_per_operation,allocations,allocated_bytes,deallocations,deallocated_bytes,peak_live_growth_bytes,rss_bytes,checksum_u64,max_terms,max_operations,max_results,max_allocated_bytes,max_rss_bytes,binary_sha256,expected_rows");
}

fn main() {
    let config = parse_config();
    if config.print_header || config.header_only {
        print_header();
    }
    if config.header_only {
        return;
    }
    configure_profile(&config.profile);
    let scratch = tempfile::tempdir().expect("create backend matrix scratch directory");
    let rows = run_matrix(&config, scratch.path());
    for row in &rows {
        row.print_csv();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_is_total_and_explicit_about_inapplicable_cells() {
        let config = Config {
            terms: 8,
            queries: 3,
            repetitions: 1,
            max_allocated_bytes: DEFAULT_MAX_ALLOCATED_BYTES,
            max_rss_bytes: DEFAULT_MAX_RSS_BYTES,
            ..Config::default()
        };
        configure_profile("treatment");
        let scratch = tempfile::tempdir().expect("matrix test scratch");
        let rows = run_matrix(&config, scratch.path());
        assert_eq!(rows.len(), EXPECTED_ROWS);
        assert_eq!(
            rows.iter()
                .filter(|row| row.applicability == "applicable")
                .count(),
            105
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.applicability == "inapplicable")
                .count(),
            45
        );
        assert!(rows.iter().filter_map(|row| row.metrics).all(|metrics| {
            metrics.allocated_bytes <= config.max_allocated_bytes
                && metrics
                    .rss_bytes
                    .is_none_or(|rss| rss <= config.max_rss_bytes)
        }));
        assert!(rows
            .iter()
            .filter(|row| row.stage == "query" && row.applicability == "applicable")
            .all(|row| row.operations == config.queries && row.checksum != 0));
    }

    #[test]
    fn corpus_is_deterministic_in_every_unit_domain() {
        assert_eq!(
            Corpus::<u8>::new(16, 4).checksum,
            Corpus::<u8>::new(16, 4).checksum
        );
        assert_eq!(
            Corpus::<char>::new(16, 4).checksum,
            Corpus::<char>::new(16, 4).checksum
        );
        assert_eq!(
            Corpus::<u64>::new(16, 4).checksum,
            Corpus::<u64>::new(16, 4).checksum
        );
    }
}
