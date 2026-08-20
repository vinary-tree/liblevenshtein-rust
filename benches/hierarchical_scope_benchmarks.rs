use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;
use std::collections::HashSet;
use std::hint::black_box;

// ============================================================================
// Data Generation Helpers
// ============================================================================

/// Generate a dictionary with terms mapped to scope ID sets.
/// Each term can appear in multiple scopes (simulating visibility across nested contexts).
fn create_hierarchical_dict(
    num_terms: usize,
    num_scopes: usize,
    avg_scopes_per_term: usize,
) -> PathMapDictionary<HashSet<u32>> {
    let terms: Vec<(String, HashSet<u32>)> = (0..num_terms)
        .map(|i| {
            let term = format!("term{:04}", i);

            // Generate scope set for this term
            let mut scopes = HashSet::new();
            let base_scope = (i % num_scopes) as u32;
            scopes.insert(base_scope);

            // Add parent scopes (simulating lexical inheritance)
            let num_additional = (i % avg_scopes_per_term) + 1;
            for j in 1..num_additional {
                let parent = (base_scope + j as u32) % num_scopes as u32;
                scopes.insert(parent);
            }

            (term, scopes)
        })
        .collect();

    PathMapDictionary::from_terms_with_values(terms)
}

/// Generate a query scope set (visible scopes from root to current context)
fn create_query_scopes(current_scope: u32, depth: usize) -> HashSet<u32> {
    let mut scopes = HashSet::new();
    scopes.insert(0); // Always include global scope

    // Add parent scopes up the hierarchy
    for i in 0..depth {
        scopes.insert(current_scope.saturating_sub(i as u32));
    }

    scopes
}

// ============================================================================
// Scenario Configurations
// ============================================================================

struct Scenario {
    name: &'static str,
    num_terms: usize,
    num_scopes: usize,
    avg_scopes_per_term: usize,
    query_scope_depth: usize,
}

const SCENARIOS: &[Scenario] = &[
    // Shallow hierarchy: Typical local variables in a function
    Scenario {
        name: "shallow_hierarchy",
        num_terms: 10_000,
        num_scopes: 10,
        avg_scopes_per_term: 3,
        query_scope_depth: 3,
    },
    // Deep hierarchy: Nested classes/modules in large codebases
    Scenario {
        name: "deep_hierarchy",
        num_terms: 10_000,
        num_scopes: 100,
        avg_scopes_per_term: 8,
        query_scope_depth: 10,
    },
    // Wide hierarchy: Large flat namespace (e.g., many sibling modules)
    Scenario {
        name: "wide_hierarchy",
        num_terms: 10_000,
        num_scopes: 1000,
        avg_scopes_per_term: 2,
        query_scope_depth: 2,
    },
    // Dense graph: Many imports/inherited symbols (high scope multiplicity)
    Scenario {
        name: "dense_graph",
        num_terms: 10_000,
        num_scopes: 50,
        avg_scopes_per_term: 15,
        query_scope_depth: 5,
    },
];

// ============================================================================
// Approach 1: HashSet Filtering (Current Implementation)
// ============================================================================

fn approach1_hashset(
    transducer: &Transducer<PathMapDictionary<HashSet<u32>>>,
    prefix: &str,
    max_distance: usize,
    visible_scopes: &HashSet<u32>,
) -> Vec<String> {
    transducer
        .query_filtered(black_box(prefix), black_box(max_distance), |term_scopes| {
            !term_scopes.is_disjoint(visible_scopes)
        })
        .map(|candidate| candidate.term)
        .collect()
}

// ============================================================================
// Approach 2: Bitmask Optimization (≤64 scopes)
// ============================================================================

/// Convert scope IDs to a bitmask (works for scope IDs < 64)
fn scopes_to_bitmask(scopes: &HashSet<u32>) -> u64 {
    let mut mask = 0u64;
    for &scope_id in scopes {
        if scope_id < 64 {
            mask |= 1u64 << scope_id;
        }
    }
    mask
}

/// Check if two bitmasks intersect
#[inline(always)]
fn bitmasks_intersect(mask_a: u64, mask_b: u64) -> bool {
    (mask_a & mask_b) != 0
}

fn create_bitmask_dict(
    num_terms: usize,
    num_scopes: usize,
    avg_scopes_per_term: usize,
) -> PathMapDictionary<u64> {
    // Only works for scopes < 64
    assert!(num_scopes <= 64, "Bitmask approach limited to 64 scopes");

    let terms: Vec<(String, u64)> = (0..num_terms)
        .map(|i| {
            let term = format!("term{:04}", i);

            // Generate scope set
            let mut scopes = HashSet::new();
            let base_scope = (i % num_scopes) as u32;
            scopes.insert(base_scope);

            let num_additional = (i % avg_scopes_per_term) + 1;
            for j in 1..num_additional {
                let parent = (base_scope + j as u32) % num_scopes as u32;
                scopes.insert(parent);
            }

            // Convert to bitmask
            let mask = scopes_to_bitmask(&scopes);
            (term, mask)
        })
        .collect();

    PathMapDictionary::from_terms_with_values(terms)
}

fn approach2_bitmask(
    transducer: &Transducer<PathMapDictionary<u64>>,
    prefix: &str,
    max_distance: usize,
    visible_mask: u64,
) -> Vec<String> {
    transducer
        .query_filtered(black_box(prefix), black_box(max_distance), |term_mask| {
            bitmasks_intersect(*term_mask, visible_mask)
        })
        .map(|candidate| candidate.term)
        .collect()
}

// ============================================================================
// Approach 3: Hybrid (Bitmask + HashSet fallback)
// ============================================================================

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "serialization", feature = "persistent-artrie"),
    derive(serde::Serialize, serde::Deserialize)
)]
enum HybridScopeData {
    Mask(u64),         // For scopes 0-63
    Set(HashSet<u32>), // For overflow or when >64 scopes exist
}

impl Default for HybridScopeData {
    fn default() -> Self {
        HybridScopeData::Mask(0)
    }
}

// Implement DictionaryValue trait for HybridScopeData
impl libdictenstein::DictionaryValue for HybridScopeData {}

impl HybridScopeData {
    fn from_scopes(scopes: &HashSet<u32>) -> Self {
        // Check if all scopes fit in u64
        if scopes.iter().all(|&id| id < 64) {
            HybridScopeData::Mask(scopes_to_bitmask(scopes))
        } else {
            HybridScopeData::Set(scopes.clone())
        }
    }

    #[inline]
    fn intersects(&self, query_scopes: &HashSet<u32>, query_mask: u64) -> bool {
        match self {
            HybridScopeData::Mask(mask) => bitmasks_intersect(*mask, query_mask),
            HybridScopeData::Set(set) => !set.is_disjoint(query_scopes),
        }
    }
}

fn create_hybrid_dict(
    num_terms: usize,
    num_scopes: usize,
    avg_scopes_per_term: usize,
) -> PathMapDictionary<HybridScopeData> {
    let terms: Vec<(String, HybridScopeData)> = (0..num_terms)
        .map(|i| {
            let term = format!("term{:04}", i);

            // Generate scope set
            let mut scopes = HashSet::new();
            let base_scope = (i % num_scopes) as u32;
            scopes.insert(base_scope);

            let num_additional = (i % avg_scopes_per_term) + 1;
            for j in 1..num_additional {
                let parent = (base_scope + j as u32) % num_scopes as u32;
                scopes.insert(parent);
            }

            // Convert to hybrid representation
            let hybrid = HybridScopeData::from_scopes(&scopes);
            (term, hybrid)
        })
        .collect();

    PathMapDictionary::from_terms_with_values(terms)
}

fn approach3_hybrid(
    transducer: &Transducer<PathMapDictionary<HybridScopeData>>,
    prefix: &str,
    max_distance: usize,
    visible_scopes: &HashSet<u32>,
    visible_mask: u64,
) -> Vec<String> {
    transducer
        .query_filtered(black_box(prefix), black_box(max_distance), |term_data| {
            term_data.intersects(visible_scopes, visible_mask)
        })
        .map(|candidate| candidate.term)
        .collect()
}

// ============================================================================
// Approach 4: Sorted Vector Intersection
// ============================================================================

/// Two-pointer intersection check with early termination
#[inline]
fn sorted_vecs_intersect(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            return true; // Early termination
        }
        if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    false
}

fn create_sorted_vec_dict(
    num_terms: usize,
    num_scopes: usize,
    avg_scopes_per_term: usize,
) -> PathMapDictionary<Vec<u32>> {
    let terms: Vec<(String, Vec<u32>)> = (0..num_terms)
        .map(|i| {
            let term = format!("term{:04}", i);

            // Generate scope set
            let mut scopes = HashSet::new();
            let base_scope = (i % num_scopes) as u32;
            scopes.insert(base_scope);

            let num_additional = (i % avg_scopes_per_term) + 1;
            for j in 1..num_additional {
                let parent = (base_scope + j as u32) % num_scopes as u32;
                scopes.insert(parent);
            }

            // Convert to sorted vector
            let mut vec: Vec<u32> = scopes.into_iter().collect();
            vec.sort_unstable();
            (term, vec)
        })
        .collect();

    PathMapDictionary::from_terms_with_values(terms)
}

fn approach4_sorted_vec(
    transducer: &Transducer<PathMapDictionary<Vec<u32>>>,
    prefix: &str,
    max_distance: usize,
    visible_scopes_vec: &[u32],
) -> Vec<String> {
    transducer
        .query_filtered(black_box(prefix), black_box(max_distance), |term_scopes| {
            sorted_vecs_intersect(term_scopes, visible_scopes_vec)
        })
        .map(|candidate| candidate.term)
        .collect()
}

// ============================================================================
// Approach 5: Post-filtering (Baseline Comparison)
// ============================================================================

fn approach5_post_filter(
    transducer: &Transducer<PathMapDictionary<HashSet<u32>>>,
    prefix: &str,
    max_distance: usize,
    visible_scopes: &HashSet<u32>,
) -> Vec<String> {
    let dict = transducer.dictionary();

    transducer
        .query(black_box(prefix), black_box(max_distance))
        .filter(|term| {
            if let Some(term_scopes) = dict.get_value(term) {
                !term_scopes.is_disjoint(visible_scopes)
            } else {
                false
            }
        })
        .collect()
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn bench_approach1_hashset(c: &mut Criterion) {
    let mut group = c.benchmark_group("approach1_hashset");

    for scenario in SCENARIOS {
        let dict = create_hierarchical_dict(
            scenario.num_terms,
            scenario.num_scopes,
            scenario.avg_scopes_per_term,
        );
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let query_scopes = create_query_scopes(5, scenario.query_scope_depth);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario.name),
            scenario,
            |b, _scenario| b.iter(|| approach1_hashset(&transducer, "term", 2, &query_scopes)),
        );
    }

    group.finish();
}

fn bench_approach2_bitmask(c: &mut Criterion) {
    let mut group = c.benchmark_group("approach2_bitmask");

    // Only benchmark scenarios with ≤64 scopes
    for scenario in SCENARIOS.iter().filter(|s| s.num_scopes <= 64) {
        let dict = create_bitmask_dict(
            scenario.num_terms,
            scenario.num_scopes,
            scenario.avg_scopes_per_term,
        );
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let query_scopes = create_query_scopes(5, scenario.query_scope_depth);
        let query_mask = scopes_to_bitmask(&query_scopes);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario.name),
            scenario,
            |b, _scenario| b.iter(|| approach2_bitmask(&transducer, "term", 2, query_mask)),
        );
    }

    group.finish();
}

fn bench_approach3_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("approach3_hybrid");

    for scenario in SCENARIOS {
        let dict = create_hybrid_dict(
            scenario.num_terms,
            scenario.num_scopes,
            scenario.avg_scopes_per_term,
        );
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let query_scopes = create_query_scopes(5, scenario.query_scope_depth);
        let query_mask = scopes_to_bitmask(&query_scopes);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario.name),
            scenario,
            |b, _scenario| {
                b.iter(|| approach3_hybrid(&transducer, "term", 2, &query_scopes, query_mask))
            },
        );
    }

    group.finish();
}

fn bench_approach4_sorted_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("approach4_sorted_vec");

    for scenario in SCENARIOS {
        let dict = create_sorted_vec_dict(
            scenario.num_terms,
            scenario.num_scopes,
            scenario.avg_scopes_per_term,
        );
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let query_scopes = create_query_scopes(5, scenario.query_scope_depth);
        let mut query_scopes_vec: Vec<u32> = query_scopes.into_iter().collect();
        query_scopes_vec.sort_unstable();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario.name),
            scenario,
            |b, _scenario| {
                b.iter(|| approach4_sorted_vec(&transducer, "term", 2, &query_scopes_vec))
            },
        );
    }

    group.finish();
}

fn bench_approach5_post_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("approach5_post_filter");

    for scenario in SCENARIOS {
        let dict = create_hierarchical_dict(
            scenario.num_terms,
            scenario.num_scopes,
            scenario.avg_scopes_per_term,
        );
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let query_scopes = create_query_scopes(5, scenario.query_scope_depth);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario.name),
            scenario,
            |b, _scenario| b.iter(|| approach5_post_filter(&transducer, "term", 2, &query_scopes)),
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_approach1_hashset,
    bench_approach2_bitmask,
    bench_approach3_hybrid,
    bench_approach4_sorted_vec,
    bench_approach5_post_filter,
);

criterion_main!(benches);
