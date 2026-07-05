//! Phase 6 novel option #3 / SOTA — measurement harness (pgmcp experiment 143).
//!
//! Parallel batch fuzzy-query evaluation: a batch (query distribution) of
//! independent queries evaluated with rayon over a *shared, read-only*
//! `&Transducer` (safe because `PathMapDictionary` is `Arc<ArcSwap>`-backed and
//! `query()` takes `&self`), versus sequential iteration. Emits raw whole-batch
//! wall-time samples for the pre-registered Welch t-test. Correctness-gated:
//! parallel per-query result counts must equal sequential.
//!
//! Run: `cargo run --release --example query_batch_parallel_experiment --features "pathmap-backend rayon"`

#[cfg(all(feature = "pathmap-backend", feature = "rayon"))]
fn main() {
    use libdictenstein::pathmap::PathMapDictionary;
    use liblevenshtein::prelude::*;
    use rayon::prelude::*;
    use std::time::Instant;

    let dict_terms: Vec<String> = (0..2000).map(|i| format!("word{i:04}")).collect();
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(dict_terms.iter().map(String::as_str).collect::<Vec<_>>());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // A batch (query distribution) of 512 distinct queries at max_distance 2.
    let queries: Vec<String> = (0..512).map(|i| format!("word{:04}", (i * 7) % 2000)).collect();
    let d = 2usize;
    let replicates = 60usize; // first 3 warm-up → 57 measured (>= protocol 51)

    // Correctness gate: parallel per-query result counts must equal sequential.
    let seq_counts: Vec<usize> = queries.iter().map(|q| transducer.query(q, d).count()).collect();
    let par_counts: Vec<usize> = queries.par_iter().map(|q| transducer.query(q, d).count()).collect();
    assert_eq!(seq_counts, par_counts, "parallel batch result-count mismatch");
    eprintln!(
        "correctness: OK (batch={}, rayon threads={})",
        queries.len(),
        rayon::current_num_threads()
    );

    let run = |parallel: bool| -> f64 {
        let start = Instant::now();
        let total: usize = if parallel {
            queries.par_iter().map(|q| transducer.query(q, d).count()).sum()
        } else {
            queries.iter().map(|q| transducer.query(q, d).count()).sum()
        };
        std::hint::black_box(total);
        start.elapsed().as_secs_f64() * 1000.0 // whole-batch ms
    };

    let mut control = Vec::with_capacity(replicates); // sequential
    for _ in 0..replicates {
        control.push(run(false));
    }
    let mut treatment = Vec::with_capacity(replicates); // parallel
    for _ in 0..replicates {
        treatment.push(run(true));
    }

    let fmt = |v: &[f64]| v[3..].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
    println!("control_ms={}", fmt(&control));
    println!("treatment_ms={}", fmt(&treatment));
}

#[cfg(not(all(feature = "pathmap-backend", feature = "rayon")))]
fn main() {
    eprintln!("query_batch_parallel_experiment requires --features \"pathmap-backend rayon\"");
}
