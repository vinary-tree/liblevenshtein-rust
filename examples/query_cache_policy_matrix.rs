//! Deterministic, allocation-counted policy matrix for bounded query caches.
//!
//! The production TinyLFU+SIEVE cache and four purpose-built bounded controls
//! receive identical keys, result allocations, logical weights, and traces.
//! Control membership is a preallocated dense directory because hashing is not
//! intrinsic to FIFO, LRU, SIEVE, or exact LFU. Policy construction and trace
//! generation are outside every timed/allocation-counted interval.

use liblevenshtein::transducer::{QueryCacheLimits, VersionedQueryCache};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::VecDeque;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SCHEMA: &str = "liblevenshtein.causal-query-cache-policy-matrix.v2";
const CAPACITY: usize = 128;
const MAX_WEIGHT: usize = CAPACITY * 3;
const KEY_SPACE: usize = 4_096;
const HOT_ROUNDS: usize = 16;
const PHASE_ROUNDS: usize = 64;
const DEFAULT_HOT_OPERATIONS: usize = 100_000;
const DEFAULT_ZIPF_OPERATIONS: usize = 200_000;
const ABSENT: usize = usize::MAX;

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: every operation delegates to the process System allocator with the
// original pointer/layout contract. The additional relaxed atomics are
// observational counters and do not influence allocation or reclamation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = System.realloc(pointer, layout, new_size);
        if !replacement.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        replacement
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AllocationSnapshot {
    allocations: u64,
    allocated_bytes: u64,
    deallocations: u64,
    deallocated_bytes: u64,
}

impl AllocationSnapshot {
    fn reset() {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        DEALLOCATIONS.store(0, Ordering::Relaxed);
        DEALLOCATED_BYTES.store(0, Ordering::Relaxed);
    }

    fn read() -> Self {
        Self {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
            deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Limits {
    entries: usize,
    weight: usize,
}

impl Limits {
    const fn standard() -> Self {
        Self {
            entries: CAPACITY,
            weight: MAX_WEIGHT,
        }
    }
}

#[derive(Debug)]
struct Universe {
    queries: Vec<String>,
}

impl Universe {
    fn new() -> Self {
        Self {
            queries: (0..KEY_SPACE)
                .map(|key| format!("policy-key-{key:08}"))
                .collect(),
        }
    }

    fn query(&self, key: usize) -> &str {
        &self.queries[key]
    }
}

fn logical_weight(key: usize) -> usize {
    // One through four logical units, with a stable nonsequential pattern.
    1 + (((key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 62) as usize)
}

fn benchmark_weigher(_query: &str, _distance: usize, results: &[usize]) -> usize {
    logical_weight(results[0])
}

#[derive(Debug)]
struct Resident {
    key: usize,
    weight: usize,
    // Store the same owned key/result shapes as the production cache. The dense
    // directory replaces only policy-extrinsic control hashing.
    _query: Arc<str>,
    result: Arc<[usize]>,
}

#[derive(Debug)]
struct DenseStore {
    slots: Vec<Option<Resident>>,
    directory: Vec<usize>,
    free: Vec<usize>,
    len: usize,
    resident_weight: usize,
    limits: Limits,
}

impl DenseStore {
    fn new(limits: Limits) -> Self {
        let mut free: Vec<usize> = (0..limits.entries).collect();
        free.reverse();
        Self {
            slots: std::iter::repeat_with(|| None)
                .take(limits.entries)
                .collect(),
            directory: vec![ABSENT; KEY_SPACE],
            free,
            len: 0,
            resident_weight: 0,
            limits,
        }
    }

    #[inline]
    fn slot_for(&self, key: usize) -> Option<usize> {
        let slot = self.directory[key];
        (slot != ABSENT).then_some(slot)
    }

    #[inline]
    fn contains(&self, key: usize) -> bool {
        self.slot_for(key).is_some()
    }

    #[inline]
    fn hit(&self, key: usize) -> Option<(usize, usize)> {
        let slot = self.slot_for(key)?;
        let result = Arc::clone(&self.slots[slot].as_ref()?.result);
        Some((slot, result[0]))
    }

    fn can_admit(&self, weight: usize) -> bool {
        self.len < self.limits.entries
            && weight <= self.limits.weight.saturating_sub(self.resident_weight)
    }

    fn insert(
        &mut self,
        universe: &Universe,
        key: usize,
        weight: usize,
        result: Arc<[usize]>,
    ) -> usize {
        assert!(self.can_admit(weight));
        assert!(!self.contains(key));
        let slot = self.free.pop().expect("admissible store has a free slot");
        self.slots[slot] = Some(Resident {
            key,
            weight,
            _query: Arc::from(universe.query(key)),
            result,
        });
        self.directory[key] = slot;
        self.len += 1;
        self.resident_weight = self
            .resident_weight
            .checked_add(weight)
            .expect("logical weight is bounded");
        slot
    }

    fn remove(&mut self, slot: usize) {
        let resident = self.slots[slot].take().expect("victim is resident");
        assert_eq!(self.directory[resident.key], slot);
        self.directory[resident.key] = ABSENT;
        self.len -= 1;
        self.resident_weight -= resident.weight;
        self.free.push(slot);
    }

    fn resident_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, resident)| resident.as_ref().map(|_| slot))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Access {
    hit: bool,
    value: usize,
}

trait Policy {
    fn access(&mut self, universe: &Universe, key: usize) -> Access;
    fn contains(&self, universe: &Universe, key: usize) -> bool;
    fn len(&self) -> usize;
    fn resident_weight(&self) -> usize;
    fn limits(&self) -> Limits;
}

fn computed_result(key: usize) -> Arc<[usize]> {
    vec![key].into()
}

#[derive(Debug)]
struct Fifo {
    store: DenseStore,
    order: VecDeque<usize>,
}

impl Fifo {
    fn new(limits: Limits) -> Self {
        Self {
            store: DenseStore::new(limits),
            order: VecDeque::with_capacity(limits.entries),
        }
    }
}

impl Policy for Fifo {
    fn access(&mut self, universe: &Universe, key: usize) -> Access {
        if let Some((_, value)) = self.store.hit(key) {
            return Access { hit: true, value };
        }
        let result = computed_result(key);
        let weight = logical_weight(key);
        if weight > self.store.limits.weight {
            return Access {
                hit: false,
                value: result[0],
            };
        }
        while !self.store.can_admit(weight) {
            let victim = self.order.pop_front().expect("full FIFO has a head");
            self.store.remove(victim);
        }
        let slot = self
            .store
            .insert(universe, key, weight, Arc::clone(&result));
        self.order.push_back(slot);
        Access {
            hit: false,
            value: result[0],
        }
    }

    fn contains(&self, _universe: &Universe, key: usize) -> bool {
        self.store.contains(key)
    }

    fn len(&self) -> usize {
        self.store.len
    }

    fn resident_weight(&self) -> usize {
        self.store.resident_weight
    }

    fn limits(&self) -> Limits {
        self.store.limits
    }
}

#[derive(Debug)]
struct Lru {
    store: DenseStore,
    previous: Vec<usize>,
    next: Vec<usize>,
    head: usize,
    tail: usize,
}

impl Lru {
    fn new(limits: Limits) -> Self {
        Self {
            store: DenseStore::new(limits),
            previous: vec![ABSENT; limits.entries],
            next: vec![ABSENT; limits.entries],
            head: ABSENT,
            tail: ABSENT,
        }
    }

    fn detach(&mut self, slot: usize) {
        let previous = self.previous[slot];
        let next = self.next[slot];
        if previous == ABSENT {
            self.head = next;
        } else {
            self.next[previous] = next;
        }
        if next == ABSENT {
            self.tail = previous;
        } else {
            self.previous[next] = previous;
        }
        self.previous[slot] = ABSENT;
        self.next[slot] = ABSENT;
    }

    fn append(&mut self, slot: usize) {
        self.previous[slot] = self.tail;
        self.next[slot] = ABSENT;
        if self.tail == ABSENT {
            self.head = slot;
        } else {
            self.next[self.tail] = slot;
        }
        self.tail = slot;
    }

    fn touch(&mut self, slot: usize) {
        if self.tail != slot {
            self.detach(slot);
            self.append(slot);
        }
    }
}

impl Policy for Lru {
    fn access(&mut self, universe: &Universe, key: usize) -> Access {
        if let Some((slot, value)) = self.store.hit(key) {
            self.touch(slot);
            return Access { hit: true, value };
        }
        let result = computed_result(key);
        let weight = logical_weight(key);
        if weight > self.store.limits.weight {
            return Access {
                hit: false,
                value: result[0],
            };
        }
        while !self.store.can_admit(weight) {
            let victim = self.head;
            assert_ne!(victim, ABSENT, "full LRU has a head");
            self.detach(victim);
            self.store.remove(victim);
        }
        let slot = self
            .store
            .insert(universe, key, weight, Arc::clone(&result));
        self.append(slot);
        Access {
            hit: false,
            value: result[0],
        }
    }

    fn contains(&self, _universe: &Universe, key: usize) -> bool {
        self.store.contains(key)
    }

    fn len(&self) -> usize {
        self.store.len
    }

    fn resident_weight(&self) -> usize {
        self.store.resident_weight
    }

    fn limits(&self) -> Limits {
        self.store.limits
    }
}

#[derive(Debug)]
struct Sieve {
    store: DenseStore,
    visited: Vec<bool>,
    hand: usize,
}

impl Sieve {
    fn new(limits: Limits) -> Self {
        Self {
            store: DenseStore::new(limits),
            visited: vec![false; limits.entries],
            hand: 0,
        }
    }

    fn evict_one(&mut self) {
        loop {
            let slot = self.hand;
            self.hand = (self.hand + 1) % self.store.limits.entries;
            if self.store.slots[slot].is_none() {
                continue;
            }
            if self.visited[slot] {
                self.visited[slot] = false;
                continue;
            }
            self.store.remove(slot);
            return;
        }
    }
}

impl Policy for Sieve {
    fn access(&mut self, universe: &Universe, key: usize) -> Access {
        if let Some((slot, value)) = self.store.hit(key) {
            self.visited[slot] = true;
            return Access { hit: true, value };
        }
        let result = computed_result(key);
        let weight = logical_weight(key);
        if weight > self.store.limits.weight {
            return Access {
                hit: false,
                value: result[0],
            };
        }
        while !self.store.can_admit(weight) {
            self.evict_one();
        }
        let slot = self
            .store
            .insert(universe, key, weight, Arc::clone(&result));
        self.visited[slot] = false;
        Access {
            hit: false,
            value: result[0],
        }
    }

    fn contains(&self, _universe: &Universe, key: usize) -> bool {
        self.store.contains(key)
    }

    fn len(&self) -> usize {
        self.store.len
    }

    fn resident_weight(&self) -> usize {
        self.store.resident_weight
    }

    fn limits(&self) -> Limits {
        self.store.limits
    }
}

#[derive(Debug)]
struct AgingExactLfu {
    store: DenseStore,
    frequency: Vec<u16>,
    stamp: Vec<u64>,
    accesses: usize,
    clock: u64,
}

impl AgingExactLfu {
    fn new(limits: Limits) -> Self {
        Self {
            store: DenseStore::new(limits),
            frequency: vec![0; limits.entries],
            stamp: vec![0; limits.entries],
            accesses: 0,
            clock: 0,
        }
    }

    fn age_if_needed(&mut self) {
        self.accesses += 1;
        if self.accesses < self.store.limits.entries * 10 {
            return;
        }
        for slot in self.store.resident_slots() {
            self.frequency[slot] = (self.frequency[slot] >> 1).max(1);
        }
        self.accesses = 0;
    }

    fn evict_one(&mut self) {
        let victim = self
            .store
            .resident_slots()
            .min_by_key(|&slot| (self.frequency[slot], self.stamp[slot]))
            .expect("full exact LFU has a victim");
        self.store.remove(victim);
        self.frequency[victim] = 0;
        self.stamp[victim] = 0;
    }
}

impl Policy for AgingExactLfu {
    fn access(&mut self, universe: &Universe, key: usize) -> Access {
        self.age_if_needed();
        self.clock = self.clock.wrapping_add(1);
        if let Some((slot, value)) = self.store.hit(key) {
            self.frequency[slot] = self.frequency[slot].saturating_add(1);
            self.stamp[slot] = self.clock;
            return Access { hit: true, value };
        }
        let result = computed_result(key);
        let weight = logical_weight(key);
        if weight > self.store.limits.weight {
            return Access {
                hit: false,
                value: result[0],
            };
        }
        while !self.store.can_admit(weight) {
            self.evict_one();
        }
        let slot = self
            .store
            .insert(universe, key, weight, Arc::clone(&result));
        self.frequency[slot] = 1;
        self.stamp[slot] = self.clock;
        Access {
            hit: false,
            value: result[0],
        }
    }

    fn contains(&self, _universe: &Universe, key: usize) -> bool {
        self.store.contains(key)
    }

    fn len(&self) -> usize {
        self.store.len
    }

    fn resident_weight(&self) -> usize {
        self.store.resident_weight
    }

    fn limits(&self) -> Limits {
        self.store.limits
    }
}

type BenchmarkWeigher = fn(&str, usize, &[usize]) -> usize;

struct TinyLfuSieve {
    cache: VersionedQueryCache<usize, BenchmarkWeigher>,
    limits: Limits,
}

impl TinyLfuSieve {
    fn new(limits: Limits) -> Self {
        Self {
            cache: VersionedQueryCache::with_limits_and_weigher(
                QueryCacheLimits::new(limits.entries, limits.weight),
                benchmark_weigher as BenchmarkWeigher,
            ),
            limits,
        }
    }
}

impl Policy for TinyLfuSieve {
    fn access(&mut self, universe: &Universe, key: usize) -> Access {
        let before = self.cache.stats().hits();
        let result = self
            .cache
            .get_or_compute(universe.query(key), 0, 0, || vec![key]);
        Access {
            hit: self.cache.stats().hits() != before,
            value: result[0],
        }
    }

    fn contains(&self, universe: &Universe, key: usize) -> bool {
        self.cache.contains(universe.query(key), 0)
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn resident_weight(&self) -> usize {
        self.cache.resident_weight()
    }

    fn limits(&self) -> Limits {
        self.limits
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PolicyKind {
    Fifo,
    Lru,
    Sieve,
    AgingExactLfu,
    TinyLfuSieve,
}

impl PolicyKind {
    const ALL: [Self; 5] = [
        Self::Fifo,
        Self::Lru,
        Self::Sieve,
        Self::AgingExactLfu,
        Self::TinyLfuSieve,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Fifo => "fifo",
            Self::Lru => "lru",
            Self::Sieve => "sieve",
            Self::AgingExactLfu => "aging-exact-lfu",
            Self::TinyLfuSieve => "tinylfu-sieve",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkloadKind {
    HotHits,
    Scan,
    PhaseShift,
    Zipf,
}

impl WorkloadKind {
    const ALL: [Self; 4] = [Self::HotHits, Self::Scan, Self::PhaseShift, Self::Zipf];

    const fn name(self) -> &'static str {
        match self {
            Self::HotHits => "hot-hits",
            Self::Scan => "scan",
            Self::PhaseShift => "phase-shift",
            Self::Zipf => "zipf",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TimedAccesses {
    operations: usize,
    elapsed: Duration,
    allocations: AllocationSnapshot,
    hits: usize,
    misses: usize,
    checksum: u64,
}

#[derive(Clone, Debug)]
struct Measurement {
    replicate: usize,
    policy_order: usize,
    workload_order: usize,
    policy: &'static str,
    workload: &'static str,
    timed: TimedAccesses,
    resident_entries: usize,
    resident_weight: usize,
    limits: Limits,
    scan_hot_retained: Option<usize>,
    phase_rounds_to_95: Option<usize>,
    binary_sha: String,
}

impl Measurement {
    fn print_csv(&self) {
        let operations = self.timed.operations as f64;
        let elapsed_ns = self.timed.elapsed.as_nanos();
        let hit_rate = self.timed.hits as f64 / operations;
        println!(
            "{SCHEMA},{},{},{},{},{},{},{},{:.6},{},{:.9},{},{:.9},{},{},{},{},{:.9},{},{},{},{},{},{},{},{}",
            self.replicate,
            self.policy_order,
            self.workload_order,
            self.policy,
            self.workload,
            self.timed.operations,
            elapsed_ns,
            elapsed_ns as f64 / operations,
            self.timed.allocations.allocations,
            self.timed.allocations.allocations as f64 / operations,
            self.timed.allocations.allocated_bytes,
            self.timed.allocations.allocated_bytes as f64 / operations,
            self.timed.allocations.deallocations,
            self.timed.allocations.deallocated_bytes,
            self.timed.hits,
            self.timed.misses,
            hit_rate,
            self.resident_entries,
            self.resident_weight,
            self.limits.entries,
            self.limits.weight,
            optional_usize(self.scan_hot_retained),
            optional_usize(self.phase_rounds_to_95),
            self.timed.checksum,
            self.binary_sha,
        );
    }
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn mix_checksum(checksum: u64, value: usize, hit: bool) -> u64 {
    // Hit status is deliberately excluded: policies may have different hit
    // rates, but all must return the exact same value sequence.
    let _ = hit;
    checksum
        .wrapping_mul(0x0000_0100_0000_01b3)
        .wrapping_add(value as u64)
}

fn warm_hot(policy: &mut impl Policy, universe: &Universe, offset: usize) {
    for _ in 0..HOT_ROUNDS {
        for key in offset..(offset + CAPACITY) {
            let access = policy.access(universe, key);
            assert_eq!(access.value, key);
        }
    }
}

fn begin_measurement() -> Instant {
    AllocationSnapshot::reset();
    Instant::now()
}

fn finish_measurement(
    start: Instant,
    operations: usize,
    hits: usize,
    checksum: u64,
) -> TimedAccesses {
    let elapsed = start.elapsed();
    let allocations = AllocationSnapshot::read();
    TimedAccesses {
        operations,
        elapsed,
        allocations,
        hits,
        misses: operations - hits,
        checksum: black_box(checksum),
    }
}

fn hot_hit_workload(
    policy: &mut impl Policy,
    universe: &Universe,
    operations: usize,
) -> (TimedAccesses, Option<usize>, Option<usize>) {
    warm_hot(policy, universe, 0);
    let mut hits = 0usize;
    let mut checksum = 0u64;
    let start = begin_measurement();
    for operation in 0..operations {
        let key = operation % CAPACITY;
        let access = policy.access(universe, key);
        assert_eq!(access.value, key);
        hits += usize::from(access.hit);
        checksum = mix_checksum(checksum, access.value, access.hit);
    }
    (
        finish_measurement(start, operations, hits, checksum),
        None,
        None,
    )
}

fn scan_workload(
    policy: &mut impl Policy,
    universe: &Universe,
) -> (TimedAccesses, Option<usize>, Option<usize>) {
    warm_hot(policy, universe, 0);
    let operations = CAPACITY * 10;
    let mut hits = 0usize;
    let mut checksum = 0u64;
    let start = begin_measurement();
    for key in CAPACITY..(CAPACITY + operations) {
        let access = policy.access(universe, key);
        assert_eq!(access.value, key);
        hits += usize::from(access.hit);
        checksum = mix_checksum(checksum, access.value, access.hit);
    }
    let timed = finish_measurement(start, operations, hits, checksum);
    let retained = (0..CAPACITY)
        .filter(|&key| policy.contains(universe, key))
        .count();
    (timed, Some(retained), None)
}

fn phase_shift_workload(
    policy: &mut impl Policy,
    universe: &Universe,
) -> (TimedAccesses, Option<usize>, Option<usize>) {
    warm_hot(policy, universe, 0);
    let operations = CAPACITY * PHASE_ROUNDS;
    let mut elapsed = Duration::ZERO;
    let mut hits = 0usize;
    let mut checksum = 0u64;
    let mut rounds_to_95 = None;
    AllocationSnapshot::reset();
    for round in 1..=PHASE_ROUNDS {
        let start = Instant::now();
        for key in CAPACITY..(2 * CAPACITY) {
            let access = policy.access(universe, key);
            assert_eq!(access.value, key);
            hits += usize::from(access.hit);
            checksum = mix_checksum(checksum, access.value, access.hit);
        }
        elapsed += start.elapsed();
        if rounds_to_95.is_none() {
            let residents = (CAPACITY..(2 * CAPACITY))
                .filter(|&key| policy.contains(universe, key))
                .count();
            if residents * 100 >= CAPACITY * 95 {
                rounds_to_95 = Some(round);
            }
        }
    }
    let allocations = AllocationSnapshot::read();
    (
        TimedAccesses {
            operations,
            elapsed,
            allocations,
            hits,
            misses: operations - hits,
            checksum: black_box(checksum),
        },
        None,
        rounds_to_95,
    )
}

fn zipf_workload(
    policy: &mut impl Policy,
    universe: &Universe,
    trace: &[usize],
) -> (TimedAccesses, Option<usize>, Option<usize>) {
    let warmup = trace.len() / 10;
    for &key in &trace[..warmup] {
        let access = policy.access(universe, key);
        assert_eq!(access.value, key);
    }
    let measured = &trace[warmup..];
    let mut hits = 0usize;
    let mut checksum = 0u64;
    let start = begin_measurement();
    for &key in measured {
        let access = policy.access(universe, key);
        assert_eq!(access.value, key);
        hits += usize::from(access.hit);
        checksum = mix_checksum(checksum, access.value, access.hit);
    }
    (
        finish_measurement(start, measured.len(), hits, checksum),
        None,
        None,
    )
}

fn run_workload(
    policy: &mut impl Policy,
    universe: &Universe,
    workload: WorkloadKind,
    hot_operations: usize,
    zipf_trace: &[usize],
) -> (TimedAccesses, Option<usize>, Option<usize>) {
    match workload {
        WorkloadKind::HotHits => hot_hit_workload(policy, universe, hot_operations),
        WorkloadKind::Scan => scan_workload(policy, universe),
        WorkloadKind::PhaseShift => phase_shift_workload(policy, universe),
        WorkloadKind::Zipf => zipf_workload(policy, universe, zipf_trace),
    }
}

struct PolicyRun<'a> {
    policy_order: usize,
    config: &'a Config,
    universe: &'a Universe,
    zipf_trace: &'a [usize],
}

fn run_policy<P, F>(policy_name: &'static str, run: &PolicyRun<'_>, factory: F) -> Vec<Measurement>
where
    P: Policy,
    F: Fn() -> P,
{
    let mut measurements = Vec::with_capacity(WorkloadKind::ALL.len());
    let workload_offset = (run.config.replicate + run.policy_order) % WorkloadKind::ALL.len();
    for workload_order in 0..WorkloadKind::ALL.len() {
        let workload =
            WorkloadKind::ALL[(workload_offset + workload_order) % WorkloadKind::ALL.len()];
        let mut policy = factory();
        let (timed, scan_hot_retained, phase_rounds_to_95) = run_workload(
            &mut policy,
            run.universe,
            workload,
            run.config.hot_operations,
            run.zipf_trace,
        );
        let limits = policy.limits();
        assert!(policy.len() <= limits.entries);
        assert!(policy.resident_weight() <= limits.weight);
        measurements.push(Measurement {
            replicate: run.config.replicate,
            policy_order: run.policy_order,
            workload_order,
            policy: policy_name,
            workload: workload.name(),
            timed,
            resident_entries: policy.len(),
            resident_weight: policy.resident_weight(),
            limits,
            scan_hot_retained,
            phase_rounds_to_95,
            binary_sha: run.config.binary_sha.clone(),
        });
    }
    measurements
}

fn run_policy_kind(
    kind: PolicyKind,
    policy_order: usize,
    config: &Config,
    universe: &Universe,
    zipf_trace: &[usize],
) -> Vec<Measurement> {
    let limits = Limits::standard();
    let run = PolicyRun {
        policy_order,
        config,
        universe,
        zipf_trace,
    };
    match kind {
        PolicyKind::Fifo => run_policy(kind.name(), &run, || Fifo::new(limits)),
        PolicyKind::Lru => run_policy(kind.name(), &run, || Lru::new(limits)),
        PolicyKind::Sieve => run_policy(kind.name(), &run, || Sieve::new(limits)),
        PolicyKind::AgingExactLfu => run_policy(kind.name(), &run, || AgingExactLfu::new(limits)),
        PolicyKind::TinyLfuSieve => run_policy(kind.name(), &run, || TinyLfuSieve::new(limits)),
    }
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

fn print_header() {
    println!(
        "schema,replicate,policy_order,workload_order,policy,workload,operations,elapsed_ns,ns_per_operation,allocations,allocations_per_operation,allocated_bytes,allocated_bytes_per_operation,deallocations,deallocated_bytes,hits,misses,hit_rate,resident_entries,resident_logical_weight,max_entries,max_logical_weight,scan_hot_retained,phase_rounds_to_95,checksum_u64,binary_sha256"
    );
}

#[derive(Clone, Debug)]
struct Config {
    replicate: usize,
    hot_operations: usize,
    zipf_operations: usize,
    binary_sha: String,
    print_header: bool,
    header_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            replicate: 1,
            hot_operations: DEFAULT_HOT_OPERATIONS,
            zipf_operations: DEFAULT_ZIPF_OPERATIONS,
            binary_sha: "unrecorded".to_owned(),
            print_header: true,
            header_only: false,
        }
    }
}

fn positive_usize(flag: &str, value: String) -> usize {
    value
        .parse::<usize>()
        .ok()
        .filter(|&value| value != 0)
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
                );
            }
            "--hot-operations" => {
                config.hot_operations = positive_usize(
                    "--hot-operations",
                    arguments.next().expect("--hot-operations requires a value"),
                );
            }
            "--zipf-operations" => {
                config.zipf_operations = positive_usize(
                    "--zipf-operations",
                    arguments
                        .next()
                        .expect("--zipf-operations requires a value"),
                );
                assert!(
                    config.zipf_operations >= 10,
                    "--zipf-operations must leave a measured tail after 10% warmup"
                );
            }
            "--binary-sha" => {
                config.binary_sha = arguments.next().expect("--binary-sha requires a value");
                assert!(
                    config.binary_sha == "unrecorded"
                        || config
                            .binary_sha
                            .bytes()
                            .all(|byte| byte.is_ascii_hexdigit()),
                    "--binary-sha must be a hexadecimal digest"
                );
            }
            "--no-header" => config.print_header = false,
            "--header-only" => config.header_only = true,
            "--help" | "-h" => {
                println!(
                    "usage: query_cache_policy_matrix [--replicate N] [--hot-operations N] [--zipf-operations N] [--binary-sha HEX] [--no-header|--header-only]"
                );
                std::process::exit(0);
            }
            unknown => panic!("unknown argument: {unknown}"),
        }
    }
    config
}

fn main() {
    let config = parse_config();
    if config.print_header || config.header_only {
        print_header();
    }
    if config.header_only {
        return;
    }

    // This diagnostic runs before cache construction. Fixing only the
    // benchmark-control hash seed makes policy comparisons reproducible while
    // production caches retain independent randomized keys.
    std::env::set_var("LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED", "1");
    let universe = Universe::new();
    let trace = zipf_trace(config.zipf_operations);
    let policy_offset = config.replicate % PolicyKind::ALL.len();
    let mut measurements = Vec::with_capacity(PolicyKind::ALL.len() * WorkloadKind::ALL.len());
    for policy_order in 0..PolicyKind::ALL.len() {
        let kind = PolicyKind::ALL[(policy_offset + policy_order) % PolicyKind::ALL.len()];
        measurements.extend(run_policy_kind(
            kind,
            policy_order,
            &config,
            &universe,
            &trace,
        ));
    }
    for measurement in &measurements {
        measurement.print_csv();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exercise_bounds(policy: &mut impl Policy, universe: &Universe) {
        for round in 0..8 {
            for key in 0..KEY_SPACE {
                let key = (key * 37 + round * 101) % KEY_SPACE;
                let access = policy.access(universe, key);
                assert_eq!(access.value, key);
                assert!(policy.len() <= policy.limits().entries);
                assert!(policy.resident_weight() <= policy.limits().weight);
            }
        }
    }

    #[test]
    fn every_policy_preserves_exact_values_and_hard_limits() {
        std::env::set_var("LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED", "1");
        let universe = Universe::new();
        let limits = Limits::standard();
        exercise_bounds(&mut Fifo::new(limits), &universe);
        exercise_bounds(&mut Lru::new(limits), &universe);
        exercise_bounds(&mut Sieve::new(limits), &universe);
        exercise_bounds(&mut AgingExactLfu::new(limits), &universe);
        exercise_bounds(&mut TinyLfuSieve::new(limits), &universe);
    }

    #[test]
    fn deterministic_traces_return_policy_independent_checksums() {
        std::env::set_var("LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED", "1");
        let universe = Universe::new();
        let trace = zipf_trace(4_096);
        let limits = Limits::standard();
        let mut checksums = Vec::new();
        macro_rules! checksum {
            ($policy:expr) => {{
                let mut policy = $policy;
                let (timed, _, _) = zipf_workload(&mut policy, &universe, &trace);
                checksums.push(timed.checksum);
            }};
        }
        checksum!(Fifo::new(limits));
        checksum!(Lru::new(limits));
        checksum!(Sieve::new(limits));
        checksum!(AgingExactLfu::new(limits));
        checksum!(TinyLfuSieve::new(limits));
        assert!(checksums.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn intrusive_lru_updates_recency_without_duplicate_residents() {
        let universe = Universe::new();
        let limits = Limits {
            entries: 3,
            weight: usize::MAX,
        };
        let mut lru = Lru::new(limits);
        for key in 0..3 {
            lru.access(&universe, key);
        }
        assert!(lru.access(&universe, 0).hit);
        lru.access(&universe, 3);
        assert!(lru.contains(&universe, 0));
        assert!(!lru.contains(&universe, 1));
        assert!(lru.contains(&universe, 2));
        assert!(lru.contains(&universe, 3));
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn scan_and_phase_quality_fields_are_workload_specific() {
        let universe = Universe::new();
        let limits = Limits::standard();
        let (_, retained, rounds) = scan_workload(&mut Fifo::new(limits), &universe);
        assert!(retained.is_some());
        assert!(rounds.is_none());
        let (_, retained, rounds) = phase_shift_workload(&mut Fifo::new(limits), &universe);
        assert!(retained.is_none());
        assert_eq!(rounds, Some(1));
    }
}
