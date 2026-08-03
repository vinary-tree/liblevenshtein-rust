//! Phase-0 decision benchmark for specialized Hamming and indel trie walkers.
//!
//! The baseline is deliberately conservative and complete: run the shipped
//! standard-Levenshtein intersection, filter its candidates by the target
//! metric's structural constraints, and recompute the target distance.  The
//! candidate arms traverse the same dictionary directly and prune with the
//! target metric.  `WalkStats` therefore counts real dictionary nodes and
//! outgoing edges considered; it is not inferred from elapsed time.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::{Dictionary, DictionaryNode, SyncStrategy};
use liblevenshtein::transducer::{Algorithm, Transducer};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DICTIONARY_SIZES: [usize; 3] = [1_000, 10_000, 100_000];
const QUERY_LENGTHS: [usize; 3] = [4, 8, 16];
const BUDGETS: [usize; 4] = [0, 1, 2, 3];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WalkStats {
    matches: usize,
    nodes_visited: usize,
    edges_enumerated: usize,
}

fn fixed_width_word(mut ordinal: usize, width: usize) -> String {
    let mut bytes = vec![b'a'; width];
    for byte in bytes.iter_mut().rev() {
        *byte = b'a' + (ordinal % 26) as u8;
        ordinal /= 26;
    }
    String::from_utf8(bytes).expect("lowercase ASCII is UTF-8")
}

fn corpus(size: usize, width: usize) -> Vec<String> {
    (0..size).map(|i| fixed_width_word(i, width)).collect()
}

#[derive(Default)]
struct TraversalCounters {
    nodes: AtomicUsize,
    edges: AtomicUsize,
}

impl TraversalCounters {
    fn reset(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        self.edges.store(0, Ordering::Relaxed);
    }

    fn read(&self, matches: usize) -> WalkStats {
        WalkStats {
            matches,
            nodes_visited: self.nodes.load(Ordering::Relaxed),
            edges_enumerated: self.edges.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
struct CountingDictionary<D> {
    inner: D,
    counters: Arc<TraversalCounters>,
}

impl<D> CountingDictionary<D> {
    fn new(inner: D) -> Self {
        Self {
            inner,
            counters: Arc::new(TraversalCounters::default()),
        }
    }
}

#[derive(Clone)]
struct CountingNode<N> {
    inner: N,
    counters: Arc<TraversalCounters>,
}

impl<N: DictionaryNode> DictionaryNode for CountingNode<N> {
    type Unit = N::Unit;

    fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        self.inner.transition(label).map(|inner| Self {
            inner,
            counters: Arc::clone(&self.counters),
        })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        self.counters.nodes.fetch_add(1, Ordering::Relaxed);
        let counters = Arc::clone(&self.counters);
        Box::new(self.inner.edges().map(move |(label, inner)| {
            counters.edges.fetch_add(1, Ordering::Relaxed);
            (
                label,
                Self {
                    inner,
                    counters: Arc::clone(&counters),
                },
            )
        }))
    }

    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<D: Dictionary> Dictionary for CountingDictionary<D> {
    type Node = CountingNode<D::Node>;

    fn root(&self) -> Self::Node {
        CountingNode {
            inner: self.inner.root(),
            counters: Arc::clone(&self.counters),
        }
    }

    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }

    fn is_suffix_based(&self) -> bool {
        self.inner.is_suffix_based()
    }
}

fn hamming_distance(left: &str, right: &str) -> Option<usize> {
    let mut left = left.chars();
    let mut right = right.chars();
    let mut mismatches = 0;
    loop {
        match (left.next(), right.next()) {
            (Some(a), Some(b)) => mismatches += usize::from(a != b),
            (None, None) => return Some(mismatches),
            _ => return None,
        }
    }
}

fn indel_distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let mut previous: Vec<usize> = (0..=left.len()).collect();
    let infinity = left
        .len()
        .saturating_add(right.chars().count())
        .saturating_add(1);

    for (row, right_char) in right.chars().enumerate() {
        let mut current = vec![infinity; left.len() + 1];
        current[0] = row + 1;
        for column in 1..=left.len() {
            current[column] = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1));
            if left[column - 1] == right_char {
                current[column] = current[column].min(previous[column - 1]);
            }
        }
        previous = current;
    }
    previous[left.len()]
}

fn baseline_hamming(
    transducer: &Transducer<DoubleArrayTrieChar<()>>,
    query: &str,
    budget: usize,
) -> WalkStats {
    let mut stats = WalkStats::default();
    for candidate in transducer.query_with_distance(query, budget) {
        stats.nodes_visited += 1;
        if hamming_distance(query, &candidate.term).is_some_and(|d| d <= budget) {
            stats.matches += 1;
        }
    }
    stats
}

fn baseline_indel(
    transducer: &Transducer<DoubleArrayTrieChar<()>>,
    query: &str,
    budget: usize,
) -> WalkStats {
    let mut stats = WalkStats::default();
    for candidate in transducer.query_with_distance(query, budget) {
        stats.nodes_visited += 1;
        if indel_distance(query, &candidate.term) <= budget {
            stats.matches += 1;
        }
    }
    stats
}

fn counted_baseline_hamming(
    transducer: &Transducer<CountingDictionary<DoubleArrayTrieChar<()>>>,
    counters: &TraversalCounters,
    query: &str,
    budget: usize,
) -> WalkStats {
    counters.reset();
    let matches = transducer
        .query_with_distance(query, budget)
        .filter(|candidate| hamming_distance(query, &candidate.term).is_some_and(|d| d <= budget))
        .count();
    counters.read(matches)
}

fn counted_baseline_indel(
    transducer: &Transducer<CountingDictionary<DoubleArrayTrieChar<()>>>,
    counters: &TraversalCounters,
    query: &str,
    budget: usize,
) -> WalkStats {
    counters.reset();
    let matches = transducer
        .query_with_distance(query, budget)
        .filter(|candidate| indel_distance(query, &candidate.term) <= budget)
        .count();
    counters.read(matches)
}

fn walk_hamming_node<N: DictionaryNode<Unit = char>>(
    node: N,
    query: &[char],
    depth: usize,
    mismatches: usize,
    budget: usize,
    stats: &mut WalkStats,
) {
    stats.nodes_visited += 1;
    if depth == query.len() {
        stats.matches += usize::from(node.is_final() && mismatches <= budget);
        return;
    }

    for (label, child) in node.edges() {
        stats.edges_enumerated += 1;
        let child_mismatches = mismatches + usize::from(label != query[depth]);
        if child_mismatches <= budget {
            walk_hamming_node(child, query, depth + 1, child_mismatches, budget, stats);
        }
    }
}

fn walk_hamming(dictionary: &DoubleArrayTrieChar<()>, query: &str, budget: usize) -> WalkStats {
    let query: Vec<char> = query.chars().collect();
    let mut stats = WalkStats::default();
    walk_hamming_node(dictionary.root(), &query, 0, 0, budget, &mut stats);
    stats
}

fn walk_indel_node<N: DictionaryNode<Unit = char>>(
    node: N,
    query: &[char],
    previous: &[usize],
    depth: usize,
    budget: usize,
    stats: &mut WalkStats,
) {
    stats.nodes_visited += 1;
    if node.is_final() && previous[query.len()] <= budget {
        stats.matches += 1;
    }

    let infinity = query.len().saturating_add(depth).saturating_add(budget + 2);
    for (label, child) in node.edges() {
        stats.edges_enumerated += 1;
        let mut current = vec![infinity; query.len() + 1];
        current[0] = depth + 1;
        for column in 1..=query.len() {
            current[column] = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1));
            if query[column - 1] == label {
                current[column] = current[column].min(previous[column - 1]);
            }
        }
        if current
            .iter()
            .copied()
            .min()
            .is_some_and(|cost| cost <= budget)
        {
            walk_indel_node(child, query, &current, depth + 1, budget, stats);
        }
    }
}

fn walk_indel(dictionary: &DoubleArrayTrieChar<()>, query: &str, budget: usize) -> WalkStats {
    let query: Vec<char> = query.chars().collect();
    let initial: Vec<usize> = (0..=query.len()).collect();
    let mut stats = WalkStats::default();
    walk_indel_node(dictionary.root(), &query, &initial, 0, budget, &mut stats);
    stats
}

fn benchmark_degenerate_walkers(c: &mut Criterion) {
    for size in DICTIONARY_SIZES {
        for width in QUERY_LENGTHS {
            let terms = corpus(size, width);
            let dictionary = DoubleArrayTrieChar::from_terms(&terms);
            let transducer = Transducer::new(dictionary.clone(), Algorithm::Standard);
            let counting_dictionary = CountingDictionary::new(dictionary.clone());
            let counters = Arc::clone(&counting_dictionary.counters);
            let counting_transducer = Transducer::new(counting_dictionary, Algorithm::Standard);
            let query = terms[size / 2].clone();
            assert!(
                dictionary.contains(&query),
                "benchmark query must be indexed"
            );

            for budget in BUDGETS {
                let hamming_baseline = baseline_hamming(&transducer, &query, budget);
                let hamming_counted =
                    counted_baseline_hamming(&counting_transducer, &counters, &query, budget);
                let hamming_candidate = walk_hamming(&dictionary, &query, budget);
                assert_eq!(hamming_candidate.matches, hamming_baseline.matches);
                assert_eq!(hamming_counted.matches, hamming_baseline.matches);

                let indel_baseline = baseline_indel(&transducer, &query, budget);
                let indel_counted =
                    counted_baseline_indel(&counting_transducer, &counters, &query, budget);
                let indel_candidate = walk_indel(&dictionary, &query, budget);
                assert_eq!(indel_candidate.matches, indel_baseline.matches);
                assert_eq!(indel_counted.matches, indel_baseline.matches);

                eprintln!(
                    "counter metric=hamming size={size} qlen={width} k={budget} baseline_nodes={} baseline_edges={} candidate_nodes={} candidate_edges={}",
                    hamming_counted.nodes_visited,
                    hamming_counted.edges_enumerated,
                    hamming_candidate.nodes_visited,
                    hamming_candidate.edges_enumerated,
                );
                eprintln!(
                    "counter metric=indel size={size} qlen={width} k={budget} baseline_nodes={} baseline_edges={} candidate_nodes={} candidate_edges={}",
                    indel_counted.nodes_visited,
                    indel_counted.edges_enumerated,
                    indel_candidate.nodes_visited,
                    indel_candidate.edges_enumerated,
                );

                let parameter = format!("size={size}/qlen={width}/k={budget}");
                let mut group = c.benchmark_group("degenerate_hamming");
                group.sample_size(10);
                group.warm_up_time(Duration::from_millis(200));
                group.measurement_time(Duration::from_millis(500));
                group.bench_with_input(
                    BenchmarkId::new("honest_baseline", &parameter),
                    &(),
                    |b, _| {
                        b.iter(|| {
                            black_box(baseline_hamming(&transducer, black_box(&query), budget))
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("candidate_walker", &parameter),
                    &(),
                    |b, _| {
                        b.iter(|| black_box(walk_hamming(&dictionary, black_box(&query), budget)))
                    },
                );
                group.finish();

                let mut group = c.benchmark_group("degenerate_indel");
                group.sample_size(10);
                group.warm_up_time(Duration::from_millis(200));
                group.measurement_time(Duration::from_millis(500));
                group.bench_with_input(
                    BenchmarkId::new("honest_baseline", &parameter),
                    &(),
                    |b, _| {
                        b.iter(|| black_box(baseline_indel(&transducer, black_box(&query), budget)))
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("candidate_walker", &parameter),
                    &(),
                    |b, _| b.iter(|| black_box(walk_indel(&dictionary, black_box(&query), budget))),
                );
                group.finish();
            }
        }
    }
}

criterion_group!(benches, benchmark_degenerate_walkers);
criterion_main!(benches);
