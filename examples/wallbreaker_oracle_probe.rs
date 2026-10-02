//! Deterministic WallBreaker-versus-exhaustive performance control.
//!
//! Run each arm in a fresh, CPU-pinned process. Both arms return the same
//! members and exact distances; an assertion checks that contract before any
//! timing. Query allocations are counted separately from timed iterations.
//!
//! Example:
//! `cargo run --release --example wallbreaker_oracle_probe -- selective wallbreaker 17 500`

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use libdictenstein::scdawg::ScdawgChar;
use liblevenshtein::distance::standard_distance_bounded;
use liblevenshtein::wallbreaker::WallBreaker;

struct CountedSystem;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

fn count_allocation(size: usize, allocated: bool) {
    if allocated && COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        count_allocation(layout.size(), !pointer.is_null());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        count_allocation(layout.size(), !pointer.is_null());
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, size) };
        count_allocation(size, !new_pointer.is_null());
        new_pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedSystem = CountedSystem;

fn corpus(seed: u64) -> Vec<String> {
    const ALPHABET: &[char] = &[
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'é', '猫',
    ];
    let mut state = seed | 1;
    let mut distinct = BTreeSet::new();
    while distinct.len() < 512 {
        let mut term = String::new();
        for _ in 0..16 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            term.push(ALPHABET[(state as usize) % ALPHABET.len()]);
        }
        distinct.insert(term);
    }
    let mut terms = vec![String::new(), "a".into(), "é".into(), "猫".into()];
    terms.extend(distinct);
    terms
}

fn exhaustive(terms: &[String], query: &str, bound: usize) -> BTreeMap<String, usize> {
    terms
        .iter()
        .filter_map(|term| {
            standard_distance_bounded(query, term, bound).map(|distance| (term.clone(), distance))
        })
        .collect()
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let case = arguments.next().expect("case: selective|short");
    let arm = arguments.next().expect("arm: wallbreaker|exhaustive");
    let seed: u64 = arguments.next().expect("seed").parse().expect("u64 seed");
    let iterations: u64 = arguments
        .next()
        .expect("iterations")
        .parse()
        .expect("positive iteration count");
    assert!(iterations > 0 && arguments.next().is_none());
    assert!(matches!(arm.as_str(), "wallbreaker" | "exhaustive"));

    let terms = corpus(seed);
    let dictionary = ScdawgChar::<()>::from_terms(terms.iter().map(String::as_str));
    let (query, bound) = match case.as_str() {
        "selective" => {
            let mut chars: Vec<char> = terms[4].chars().collect();
            chars[3] = if chars[3] == 'é' { '猫' } else { 'é' };
            chars[11] = if chars[11] == '猫' { 'é' } else { '猫' };
            (chars.into_iter().collect::<String>(), 2)
        }
        "short" => ("é".to_owned(), 2),
        _ => panic!("unknown case"),
    };
    let matcher = WallBreaker::new(&dictionary, bound);
    let expected = exhaustive(&terms, &query, bound);
    let observed: BTreeMap<_, _> = matcher
        .query(&query)
        .map(|result| (result.term, result.distance))
        .collect();
    assert_eq!(observed, expected, "candidate and oracle results differ");

    let run = || -> usize {
        match arm.as_str() {
            "wallbreaker" => matcher.query(black_box(&query)).count(),
            "exhaustive" => terms
                .iter()
                .filter(|term| {
                    standard_distance_bounded(black_box(&query), black_box(term), bound).is_some()
                })
                .count(),
            _ => unreachable!(),
        }
    };
    for _ in 0..16 {
        black_box(run());
    }
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(run());
    }
    let ns_per_query = start.elapsed().as_nanos() as f64 / iterations as f64;

    const ALLOCATION_REPEATS: u64 = 16;
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..ALLOCATION_REPEATS {
        black_box(run());
    }
    COUNTING.store(false, Ordering::Relaxed);
    let allocation_count = ALLOCATIONS.load(Ordering::Relaxed) as f64 / ALLOCATION_REPEATS as f64;
    let allocated_bytes =
        ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / ALLOCATION_REPEATS as f64;

    println!(
        "case={case} arm={arm} seed={seed} iterations={iterations} result_count={} ns_per_query={ns_per_query:.3} allocations_per_query={allocation_count:.3} bytes_per_query={allocated_bytes:.3}",
        expected.len()
    );
}
