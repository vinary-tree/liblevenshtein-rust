//! Same-binary causal benchmark for the bounded query-result cache.
//!
//! The control reproduces the former owned-tuple lookup exactly. The treatment
//! uses the public bounded cache with unit weights. Both arms are prewarmed
//! with the same keys, measured in alternating order, and emit 51 samples after
//! three warm-ups for pgmcp experiment 259.
//!
//! Run pinned to one physical core:
//!
//! ```text
//! taskset -c 2 cargo run --release --example query_cache_policy_experiment
//! ```

use liblevenshtein::transducer::{QueryCacheLimits, QueryCacheWeigher, VersionedQueryCache};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const HOT_KEYS: usize = 128;
const LOOKUPS_PER_SAMPLE: usize = 100_000;
const WARMUPS: usize = 3;
const SAMPLES: usize = 51;

struct LegacyCache<V> {
    entries: HashMap<(String, usize), Arc<[V]>>,
}

impl<V> LegacyCache<V> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    #[inline]
    fn get_or_compute<F>(&mut self, query: &str, max_distance: usize, compute: F) -> Arc<[V]>
    where
        F: FnOnce() -> Vec<V>,
    {
        // This is intentionally the exact former production hit path.
        if let Some(hit) = self.entries.get(&(query.to_owned(), max_distance)) {
            return Arc::clone(hit);
        }
        let result: Arc<[V]> = compute().into();
        self.entries
            .insert((query.to_owned(), max_distance), Arc::clone(&result));
        result
    }
}

fn unit_weigher(_query: &str, _distance: usize, _results: &[u64]) -> usize {
    1
}

fn treatment_cache(
    capacity: usize,
) -> VersionedQueryCache<u64, impl QueryCacheWeigher<u64> + Clone> {
    VersionedQueryCache::with_limits_and_weigher(
        QueryCacheLimits::new(capacity, capacity),
        unit_weigher,
    )
}

fn measure_control(cache: &mut LegacyCache<u64>, queries: &[String]) -> f64 {
    let start = Instant::now();
    let mut checksum = 0u64;
    for index in 0..LOOKUPS_PER_SAMPLE {
        let query = &queries[(index / 4) % queries.len()];
        let result = cache.get_or_compute(query, index & 3, || panic!("prewarmed control hit"));
        checksum = checksum.wrapping_add(result[0]);
    }
    black_box(checksum);
    start.elapsed().as_nanos() as f64 / LOOKUPS_PER_SAMPLE as f64
}

fn measure_treatment<W>(cache: &mut VersionedQueryCache<u64, W>, queries: &[String]) -> f64
where
    W: QueryCacheWeigher<u64>,
{
    let start = Instant::now();
    let mut checksum = 0u64;
    for index in 0..LOOKUPS_PER_SAMPLE {
        let query = &queries[(index / 4) % queries.len()];
        let result =
            cache.get_or_compute(query, index & 3, 0, || panic!("prewarmed treatment hit"));
        checksum = checksum.wrapping_add(result[0]);
    }
    black_box(checksum);
    start.elapsed().as_nanos() as f64 / LOOKUPS_PER_SAMPLE as f64
}

fn comma_separated(samples: &[f64]) -> String {
    samples
        .iter()
        .map(|sample| format!("{sample:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let queries: Vec<String> = (0..HOT_KEYS)
        .map(|index| format!("repeated-query-{index:04}"))
        .collect();
    let capacity = HOT_KEYS * 4; // four max-distance variants per query
    let mut control = LegacyCache::new();
    let mut treatment = treatment_cache(capacity);

    for (query_index, query) in queries.iter().enumerate() {
        for distance in 0..4 {
            let expected = ((query_index * 4 + distance) as u64).wrapping_mul(17);
            let control_result = control.get_or_compute(query, distance, || vec![expected]);
            let treatment_result = treatment.get_or_compute(query, distance, 0, || vec![expected]);
            assert_eq!(control_result, treatment_result);
        }
    }
    assert_eq!(treatment.len(), capacity);

    let total_rounds = WARMUPS + SAMPLES;
    let mut control_samples = Vec::with_capacity(SAMPLES);
    let mut treatment_samples = Vec::with_capacity(SAMPLES);
    for round in 0..total_rounds {
        // Alternate order to distribute thermal/frequency drift symmetrically.
        let (control_ns, treatment_ns) = if round & 1 == 0 {
            (
                measure_control(&mut control, &queries),
                measure_treatment(&mut treatment, &queries),
            )
        } else {
            let treatment_ns = measure_treatment(&mut treatment, &queries);
            let control_ns = measure_control(&mut control, &queries);
            (control_ns, treatment_ns)
        };
        if round >= WARMUPS {
            control_samples.push(control_ns);
            treatment_samples.push(treatment_ns);
        }
    }

    // A preregistered policy gate: a prewarmed set must survive a scan ten
    // times the cache capacity. These lookups are outside the timed samples.
    let mut scan_cache = treatment_cache(HOT_KEYS);
    for key in 0..HOT_KEYS {
        let query = format!("hot-{key}");
        let _ = scan_cache.get_or_compute(&query, 1, 0, || vec![key as u64]);
    }
    for _ in 0..12 {
        for key in 0..HOT_KEYS {
            let query = format!("hot-{key}");
            let _ = scan_cache.get_or_compute(&query, 1, 0, || panic!("hot hit"));
        }
    }
    for key in 0..(10 * HOT_KEYS) {
        let query = format!("scan-{key}");
        let _ = scan_cache.get_or_compute(&query, 1, 0, || vec![key as u64]);
    }
    let retained = (0..HOT_KEYS)
        .filter(|key| scan_cache.contains(&format!("hot-{key}"), 1))
        .count();
    assert!(retained * 100 >= HOT_KEYS * 95);

    println!("control_hot_hit_ns={}", comma_separated(&control_samples));
    println!(
        "treatment_hot_hit_ns={}",
        comma_separated(&treatment_samples)
    );
    println!("scan_hot_retained={retained}/{HOT_KEYS}");
    println!(
        "treatment_residency=entries:{},weight:{},max_entries:{},max_weight:{}",
        treatment.len(),
        treatment.resident_weight(),
        treatment.limits().max_entries(),
        treatment.limits().max_weight()
    );
}
