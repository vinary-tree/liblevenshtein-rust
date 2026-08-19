//! Deterministic admission/eviction quality matrix for bounded query caches.

use liblevenshtein::transducer::{QueryCacheLimits, VersionedQueryCache};
use std::collections::{HashMap, HashSet, VecDeque};

const CAPACITY: usize = 128;
const KEY_SPACE: usize = 4_096;

type UnitWeigher = fn(&str, usize, &[usize]) -> usize;
type UnitWeighedQueryCache = VersionedQueryCache<usize, UnitWeigher>;

trait Policy {
    fn access(&mut self, key: usize) -> bool;
    fn contains(&self, key: usize) -> bool;
}

struct Fifo {
    capacity: usize,
    members: HashSet<usize>,
    order: VecDeque<usize>,
}

impl Fifo {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            members: HashSet::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }
}

impl Policy for Fifo {
    fn access(&mut self, key: usize) -> bool {
        if self.members.contains(&key) {
            return true;
        }
        if self.members.len() == self.capacity {
            let victim = self.order.pop_front().expect("full FIFO has a head");
            assert!(self.members.remove(&victim));
        }
        assert!(self.members.insert(key));
        self.order.push_back(key);
        false
    }

    fn contains(&self, key: usize) -> bool {
        self.members.contains(&key)
    }
}

#[derive(Clone, Copy)]
struct Recency {
    stamp: u64,
}

struct Lru {
    capacity: usize,
    clock: u64,
    entries: HashMap<usize, Recency>,
}

impl Lru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            clock: 0,
            entries: HashMap::with_capacity(capacity),
        }
    }
}

impl Policy for Lru {
    fn access(&mut self, key: usize) -> bool {
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.stamp = self.clock;
            return true;
        }
        if self.entries.len() == self.capacity {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.stamp)
                .map(|(&key, _)| key)
                .expect("full LRU has a victim");
            self.entries.remove(&victim);
        }
        self.entries.insert(key, Recency { stamp: self.clock });
        false
    }

    fn contains(&self, key: usize) -> bool {
        self.entries.contains_key(&key)
    }
}

#[derive(Clone, Copy)]
struct SieveEntry {
    key: usize,
    visited: bool,
}

struct Sieve {
    capacity: usize,
    slots: Vec<SieveEntry>,
    index: HashMap<usize, usize>,
    hand: usize,
}

impl Sieve {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            slots: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            hand: 0,
        }
    }
}

impl Policy for Sieve {
    fn access(&mut self, key: usize) -> bool {
        if let Some(&slot) = self.index.get(&key) {
            self.slots[slot].visited = true;
            return true;
        }
        if self.slots.len() < self.capacity {
            let slot = self.slots.len();
            self.slots.push(SieveEntry {
                key,
                visited: false,
            });
            self.index.insert(key, slot);
            return false;
        }

        loop {
            let slot = self.hand;
            self.hand = (self.hand + 1) % self.slots.len();
            if self.slots[slot].visited {
                self.slots[slot].visited = false;
                continue;
            }
            let victim = self.slots[slot].key;
            assert_eq!(self.index.remove(&victim), Some(slot));
            self.slots[slot] = SieveEntry {
                key,
                visited: false,
            };
            self.index.insert(key, slot);
            return false;
        }
    }

    fn contains(&self, key: usize) -> bool {
        self.index.contains_key(&key)
    }
}

#[derive(Clone, Copy)]
struct LfuEntry {
    frequency: u16,
    stamp: u64,
}

struct AgingExactLfu {
    capacity: usize,
    accesses: usize,
    clock: u64,
    entries: HashMap<usize, LfuEntry>,
}

impl AgingExactLfu {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            accesses: 0,
            clock: 0,
            entries: HashMap::with_capacity(capacity),
        }
    }

    fn age_if_needed(&mut self) {
        self.accesses += 1;
        if self.accesses < self.capacity * 10 {
            return;
        }
        for entry in self.entries.values_mut() {
            entry.frequency = (entry.frequency >> 1).max(1);
        }
        self.accesses = 0;
    }
}

impl Policy for AgingExactLfu {
    fn access(&mut self, key: usize) -> bool {
        self.age_if_needed();
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.frequency = entry.frequency.saturating_add(1);
            entry.stamp = self.clock;
            return true;
        }
        if self.entries.len() == self.capacity {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| (entry.frequency, entry.stamp))
                .map(|(&key, _)| key)
                .expect("full LFU has a victim");
            self.entries.remove(&victim);
        }
        self.entries.insert(
            key,
            LfuEntry {
                frequency: 1,
                stamp: self.clock,
            },
        );
        false
    }

    fn contains(&self, key: usize) -> bool {
        self.entries.contains_key(&key)
    }
}

fn unit_weigher(_query: &str, _distance: usize, _results: &[usize]) -> usize {
    1
}

struct TinyLfuSieve {
    queries: Vec<String>,
    cache: UnitWeighedQueryCache,
}

impl TinyLfuSieve {
    fn new(capacity: usize, key_space: usize) -> Self {
        Self {
            queries: (0..key_space)
                .map(|key| format!("policy-key-{key:08}"))
                .collect(),
            cache: VersionedQueryCache::with_limits_and_weigher(
                QueryCacheLimits::new(capacity, capacity),
                unit_weigher as UnitWeigher,
            ),
        }
    }
}

impl Policy for TinyLfuSieve {
    fn access(&mut self, key: usize) -> bool {
        let query = &self.queries[key];
        let hit = self.cache.contains(query, 0);
        let result = self.cache.get_or_compute(query, 0, 0, || vec![key]);
        assert_eq!(result[0], key);
        hit
    }

    fn contains(&self, key: usize) -> bool {
        self.cache.contains(&self.queries[key], 0)
    }
}

type Factory = fn() -> Box<dyn Policy>;

fn fifo() -> Box<dyn Policy> {
    Box::new(Fifo::new(CAPACITY))
}

fn lru() -> Box<dyn Policy> {
    Box::new(Lru::new(CAPACITY))
}

fn sieve() -> Box<dyn Policy> {
    Box::new(Sieve::new(CAPACITY))
}

fn exact_lfu() -> Box<dyn Policy> {
    Box::new(AgingExactLfu::new(CAPACITY))
}

fn tiny_lfu_sieve() -> Box<dyn Policy> {
    Box::new(TinyLfuSieve::new(CAPACITY, KEY_SPACE))
}

fn warm_hot(policy: &mut dyn Policy, offset: usize) {
    for _ in 0..16 {
        for key in offset..(offset + CAPACITY) {
            policy.access(key);
        }
    }
}

fn scan_retention(factory: Factory) -> usize {
    let mut policy = factory();
    warm_hot(policy.as_mut(), 0);
    for key in CAPACITY..(CAPACITY * 11) {
        policy.access(key);
    }
    (0..CAPACITY).filter(|&key| policy.contains(key)).count()
}

fn phase_shift(factory: Factory) -> (usize, f64) {
    let mut policy = factory();
    warm_hot(policy.as_mut(), 0);
    let mut hits = 0usize;
    let mut requests = 0usize;
    let mut rounds_to_95 = 65usize;
    for round in 1..=64 {
        for key in CAPACITY..(2 * CAPACITY) {
            hits += usize::from(policy.access(key));
            requests += 1;
        }
        let residents = (CAPACITY..(2 * CAPACITY))
            .filter(|&key| policy.contains(key))
            .count();
        if rounds_to_95 == 65 && residents * 100 >= CAPACITY * 95 {
            rounds_to_95 = round;
        }
    }
    (rounds_to_95, hits as f64 / requests as f64)
}

fn zipf_trace(length: usize) -> Vec<usize> {
    let mut cumulative = Vec::with_capacity(KEY_SPACE);
    let mut total = 0.0f64;
    for rank in 1..=KEY_SPACE {
        total += 1.0 / (rank as f64).powf(1.1);
        cumulative.push(total);
    }
    for value in &mut cumulative {
        *value /= total;
    }

    let mut state = 0x4d59_5df4_d0f3_3173u64;
    (0..length)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let sample = (state >> 11) as f64 / ((1u64 << 53) as f64);
            cumulative.partition_point(|&cutoff| cutoff < sample)
        })
        .collect()
}

fn zipf_hit_rate(factory: Factory, trace: &[usize]) -> f64 {
    let mut policy = factory();
    let warmup = trace.len() / 10;
    let mut hits = 0usize;
    for &key in &trace[..warmup] {
        policy.access(key);
    }
    for &key in &trace[warmup..] {
        hits += usize::from(policy.access(key));
    }
    hits as f64 / (trace.len() - warmup) as f64
}

fn main() {
    // This diagnostic runs before any worker threads or cache construction.
    // Fixing only the benchmark-control hash seed makes policy comparisons
    // byte-reproducible while production caches retain per-instance keys.
    std::env::set_var("LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED", "1");
    let trace = zipf_trace(200_000);
    println!("policy,scan_hot_retained,phase_rounds_to_95,phase_hit_rate,zipf_hit_rate");
    for (name, factory) in [
        ("fifo", fifo as Factory),
        ("lru", lru as Factory),
        ("sieve", sieve as Factory),
        ("aging-exact-lfu", exact_lfu as Factory),
        ("tinylfu-sieve", tiny_lfu_sieve as Factory),
    ] {
        let retained = scan_retention(factory);
        let (rounds, phase_hits) = phase_shift(factory);
        let zipf_hits = zipf_hit_rate(factory, &trace);
        println!("{name},{retained},{rounds},{phase_hits:.6},{zipf_hits:.6}");
    }
}
