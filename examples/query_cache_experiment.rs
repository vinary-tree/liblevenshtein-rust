//! Phase 6 novel optimization #4 — pre-registered measurement harness.
//!
//! Compares a version-tied cross-query cache (treatment) against the uncached
//! transducer (control) on a repeated-query workload, emitting raw per-query
//! latency samples for the pre-registered Welch t-test. Correctness-gated:
//! cached results must equal uncached before any timing.
//!
//! Run: `cargo run --release --example query_cache_experiment --features pathmap-backend`

#[cfg(feature = "pathmap-backend")]
fn main() {
    use libdictenstein::pathmap::PathMapDictionary;
    use liblevenshtein::prelude::*;
    use liblevenshtein::transducer::VersionedQueryCache;
    use std::time::Instant;

    let dict_terms: Vec<String> = (0..1000).map(|i| format!("word{i:04}")).collect();
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(dict_terms.iter().map(String::as_str).collect::<Vec<_>>());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Repeated-query workload: a fixed set of 50 queries re-issued every
    // replicate — the canonical spell-check / autocomplete / batch-rescan shape.
    let queries: Vec<String> = (0..50)
        .map(|i| format!("word{:04}", (i * 17) % 1000))
        .collect();
    let max_distance = 2usize;
    let replicates = 60usize; // first 3 are warm-up (discarded) → 57 measured (>= protocol 51)

    // Correctness gate: cached results must equal uncached for every query.
    {
        let mut cache: VersionedQueryCache<String> = VersionedQueryCache::new();
        for q in &queries {
            let mut uncached: Vec<String> = transducer.query(q, max_distance).collect();
            let mut cached: Vec<String> = cache
                .get_or_compute(q, max_distance, 0, || {
                    transducer.query(q, max_distance).collect()
                })
                .to_vec();
            uncached.sort();
            cached.sort();
            assert_eq!(uncached, cached, "cache correctness mismatch for {q:?}");
        }
        eprintln!(
            "correctness: OK ({} queries identical cached vs uncached)",
            queries.len()
        );
    }

    // Control arm: uncached.
    let mut control = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let start = Instant::now();
        let mut acc = 0usize;
        for q in &queries {
            let r: Vec<String> = transducer.query(q, max_distance).collect();
            acc = acc.wrapping_add(r.len());
        }
        std::hint::black_box(acc);
        control.push(start.elapsed().as_nanos() as f64 / queries.len() as f64);
    }

    // Treatment arm: version-tied cache (static dict ⇒ version 0 throughout).
    let mut cache: VersionedQueryCache<String> = VersionedQueryCache::new();
    let mut treatment = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let start = Instant::now();
        let mut acc = 0usize;
        for q in &queries {
            let r = cache.get_or_compute(q, max_distance, 0, || {
                transducer.query(q, max_distance).collect()
            });
            acc = acc.wrapping_add(r.len());
        }
        std::hint::black_box(acc);
        treatment.push(start.elapsed().as_nanos() as f64 / queries.len() as f64);
    }

    let fmt = |v: &[f64]| {
        v[3..]
            .iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    println!("control_ns_per_query={}", fmt(&control));
    println!("treatment_ns_per_query={}", fmt(&treatment));
}

#[cfg(not(feature = "pathmap-backend"))]
fn main() {
    eprintln!("query_cache_experiment requires --features pathmap-backend");
}
