//! Causal driver for concurrent live-resource snapshot and cold graph capture.
//!
//! Build with `--features causal-resource-profiling`. Set
//! `LIBLEVENSHTEIN_CAUSAL_USE_LEGACY_SNAPSHOT_LOCKS=1` for the preregistered
//! control arm; omit it for the lock-free treatment.

use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};
use liblevenshtein::bindings::{MatchBatch, MatchTerm, QueryOrder, ResourceTransducer};
use liblevenshtein::transducer::Algorithm;
use std::sync::{Arc, Barrier};
use std::time::Instant;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Args {
    samples: usize,
    warmups: usize,
    threads: usize,
    terms: usize,
    queries_per_thread: usize,
}

fn args() -> Args {
    let mut parsed = Args {
        samples: 30,
        warmups: 3,
        threads: 16,
        terms: 4096,
        queries_per_thread: 64,
    };
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .unwrap_or_else(|| panic!("{flag} requires a value"));
        let value = value
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{flag} requires a positive integer"));
        assert!(value != 0, "{flag} requires a positive integer");
        match flag.as_str() {
            "--samples" => parsed.samples = value,
            "--warmups" => parsed.warmups = value,
            "--threads" => parsed.threads = value,
            "--terms" => parsed.terms = value,
            "--queries-per-thread" => parsed.queries_per_thread = value,
            _ => panic!("unknown argument {flag}"),
        }
    }
    parsed
}

fn hash_match(mut hash: u64, term: &str, distance: usize, id: Option<u64>) -> u64 {
    for byte in term.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    for byte in (distance as u64).to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    for byte in id.unwrap_or(u64::MAX).to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

fn one_sample(
    transducer: &ResourceTransducer,
    threads: usize,
    queries_per_thread: usize,
) -> (u128, u64, u64) {
    std::thread::scope(|scope| {
        let barrier = Arc::new(Barrier::new(threads + 1));
        let workers = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    let mut batch = MatchBatch::default();
                    let mut matches = 0u64;
                    let mut checksum = FNV_OFFSET;
                    for _ in 0..queries_per_thread {
                        let mut cursor = transducer
                            .query_utf8("term-0010", 0, QueryOrder::Traversal)
                            .expect("create concurrent query");
                        loop {
                            let written = cursor.next_batch(&mut batch, 256).expect("drain query");
                            if written == 0 {
                                break;
                            }
                            for item in batch.as_slice() {
                                let MatchTerm::Utf8(term) = &item.term else {
                                    panic!("Unicode query returned a non-Unicode term");
                                };
                                matches += 1;
                                checksum = hash_match(checksum, term, item.distance, item.id);
                            }
                        }
                    }
                    (matches, checksum)
                })
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        barrier.wait();
        let mut matches = 0u64;
        let mut checksum = 0u64;
        for worker in workers {
            let (worker_matches, worker_checksum) = worker.join().expect("query worker");
            matches += worker_matches;
            checksum = checksum.wrapping_add(worker_checksum);
        }
        (start.elapsed().as_nanos(), matches, checksum)
    })
}

fn main() {
    let args = args();
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    let terms = (0..args.terms)
        .map(|index| (format!("term-{index:04}"), Some(index as u64)))
        .collect::<Vec<_>>();
    dictionary
        .insert_text_batch(terms.iter().map(|(term, value)| (term.as_bytes(), *value)))
        .expect("build dictionary");
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard)
            .expect("retain resource")
    };

    for sample in 0..args.warmups + args.samples {
        let marker = format!("revision-marker-{sample:04}");
        dictionary
            .insert_text(marker.as_bytes(), Some(sample as u64))
            .expect("advance resource revision");
        let (elapsed_ns, matches, checksum) =
            one_sample(&transducer, args.threads, args.queries_per_thread);
        if sample >= args.warmups {
            println!(
                "{{\"sample\":{},\"elapsed_ns\":{},\"matches\":{},\"checksum\":{}}}",
                sample - args.warmups,
                elapsed_ns,
                matches,
                checksum
            );
        }
    }
}
