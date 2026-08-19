//! Same-binary pressure-miss benchmark for query-cache victim planning.

use liblevenshtein::transducer::{QueryCacheLimits, QueryCacheWeigher, VersionedQueryCache};
use std::hint::black_box;
use std::time::Instant;

const ALLOCATING_PLAN_ENV: &str = "LIBLEVENSHTEIN_CAUSAL_ALLOCATING_QUERY_CACHE_VICTIM_PLAN";
const FIXED_HASH_SEED_ENV: &str = "LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED";

#[derive(Clone, Copy)]
struct Args {
    capacity: usize,
    requests: usize,
}

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("causal_query_cache_profile: {}", message.as_ref());
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut capacity = 256usize;
    let mut requests = 4_096usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        if flag == "--help" {
            println!("usage: causal_query_cache_profile [--capacity N] [--requests N]");
            std::process::exit(0);
        }
        let value = arguments
            .next()
            .unwrap_or_else(|| fail(format!("{flag} requires a value")));
        match flag.as_str() {
            "--capacity" => {
                capacity = value
                    .parse()
                    .unwrap_or_else(|_| fail("--capacity must be a positive integer"));
            }
            "--requests" => {
                requests = value
                    .parse()
                    .unwrap_or_else(|_| fail("--requests must be a positive integer"));
            }
            _ => fail(format!("unknown flag {flag:?}")),
        }
    }
    if capacity == 0 || requests == 0 {
        fail("--capacity and --requests must be positive");
    }
    Args { capacity, requests }
}

fn unit_weigher(_query: &str, _distance: usize, _results: &[u64]) -> usize {
    1
}

fn cache(capacity: usize) -> VersionedQueryCache<u64, impl QueryCacheWeigher<u64> + Clone> {
    VersionedQueryCache::with_limits_and_weigher(
        QueryCacheLimits::new(capacity, capacity),
        unit_weigher,
    )
}

fn main() {
    let args = parse_args();
    let allocating_plan = std::env::var_os(ALLOCATING_PLAN_ENV).is_some();
    let fixed_hash_seed = std::env::var_os(FIXED_HASH_SEED_ENV).is_some();
    let hot = (0..args.capacity)
        .map(|index| format!("hot-{index:08}"))
        .collect::<Vec<_>>();
    let pressure = (0..(args.requests + args.capacity))
        .map(|index| format!("pressure-{index:08}"))
        .collect::<Vec<_>>();
    let mut cache = cache(args.capacity);

    for (index, query) in hot.iter().enumerate() {
        let value = cache.get_or_compute(query, 1, 0, || vec![index as u64]);
        assert_eq!(value[0], index as u64);
    }
    for _ in 0..16 {
        for (index, query) in hot.iter().enumerate() {
            let value = cache.get_or_compute(query, 1, 0, Vec::new);
            assert_eq!(value[0], index as u64);
        }
    }

    // Reach steady pressure behavior outside the timer. Unique one-hit keys
    // exercise rejection planning without becoming reusable residents.
    for (index, query) in pressure.iter().take(args.capacity).enumerate() {
        let expected = index as u64 ^ 0xa5a5_a5a5_a5a5_a5a5;
        let value = cache.get_or_compute(query, 1, 0, || vec![expected]);
        assert_eq!(value[0], expected);
    }

    cache.reset_stats();
    let start = Instant::now();
    let mut checksum = 0u64;
    for (index, query) in pressure
        .iter()
        .skip(args.capacity)
        .take(args.requests)
        .enumerate()
    {
        let expected = index as u64 ^ 0x5a5a_5a5a_5a5a_5a5a;
        let value = cache.get_or_compute(query, 1, 0, || vec![expected]);
        assert_eq!(value[0], expected);
        checksum = checksum.wrapping_add(value[0]);
    }
    let elapsed_ns = start.elapsed().as_nanos();
    black_box(checksum);

    let retained = hot.iter().filter(|query| cache.contains(query, 1)).count();
    assert!(
        retained * 100 >= args.capacity * 95,
        "one-hit pressure must not evict the hot set: {retained}/{}",
        args.capacity
    );
    let stats = cache.stats();
    println!("{{");
    println!("  \"schema\": \"liblevenshtein.causal-query-cache.v1\",");
    println!(
        "  \"victim_plan\": \"{}\",",
        if allocating_plan {
            "allocating-transactional"
        } else {
            "reused-in-place"
        }
    );
    println!("  \"capacity\": {},", args.capacity);
    println!("  \"requests\": {},", args.requests);
    println!("  \"fixed_hash_seed\": {fixed_hash_seed},");
    println!("  \"elapsed_ns\": {elapsed_ns},");
    println!("  \"checksum_u64\": {checksum},");
    println!("  \"hot_retained\": {retained},");
    println!("  \"resident_entries\": {},", cache.len());
    println!("  \"resident_weight\": {},", cache.resident_weight());
    println!("  \"hits\": {},", stats.hits());
    println!("  \"misses\": {},", stats.misses());
    println!("  \"admissions\": {},", stats.admissions());
    println!("  \"rejections\": {},", stats.rejections());
    println!("  \"evictions\": {}", stats.evictions());
    println!("}}");
}
