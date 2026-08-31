//! Bounded, version-tied cross-query result cache.
//!
//! Fuzzy query evaluation is dominated by the automaton/dictionary product
//! walk. Applications that repeat queries can skip that work by memoizing
//! `(query, max_distance) -> results`, but an unbounded result cache turns an
//! adversarial or merely diverse query stream into unbounded memory growth.
//!
//! This cache separates two decisions:
//!
//! 1. An aging, approximate-frequency sketch decides whether a new result is
//!    valuable enough to displace resident data. This is the TinyLFU admission
//!    model: a Bloom-style doorkeeper suppresses one-hit noise and four rows of
//!    saturating 4-bit counters estimate recent frequency in fixed space.
//! 2. A SIEVE hand and one reference bit per resident select eviction
//!    candidates. Hits only set that bit; they never reorder a linked list.
//!
//! Approximation can change *which* reusable result remains resident, but never
//! query correctness: a miss always recomputes the exact value. Entry count and
//! logical weight are hard bounds, independent of the sketch. Frequency is
//! periodically halved so a formerly hot workload cannot pin entries forever.
//!
//! The admission design follows Einziger, Friedman, and Manes,
//! [TinyLFU](https://doi.org/10.1145/3149371). The eviction hand follows Zhang
//! et al., [SIEVE](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo).
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::{QueryCacheLimits, VersionedQueryCache};
//!
//! let limits = QueryCacheLimits::new(128, 8 * 1024 * 1024);
//! let mut cache: VersionedQueryCache<String> =
//!     VersionedQueryCache::with_limits(limits);
//! let mut version = 0u64;
//!
//! let a = cache.get_or_compute("foo", 2, version, || vec!["foo".to_string()]);
//! let b = cache.get_or_compute("foo", 2, version, || panic!("cache hit"));
//! assert_eq!(a, b);
//!
//! // A dictionary mutation changes the version and invalidates every result.
//! version += 1;
//! let c = cache.get_or_compute("foo", 2, version, || {
//!     vec!["foo".to_string(), "food".to_string()]
//! });
//! assert_eq!(c.len(), 2);
//! assert!(cache.len() <= limits.max_entries());
//! assert!(cache.resident_weight() <= limits.max_weight());
//! ```

use ahash::RandomState as AHashRandomState;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::hash::BuildHasher;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

const DEFAULT_MAX_ENTRIES: usize = 1024;
const DEFAULT_MAX_WEIGHT: usize = 64 * 1024 * 1024;
const SKETCH_ROWS: usize = 4;
const MIN_SKETCH_WIDTH: usize = 64;
const MAX_SKETCH_WIDTH: usize = 1 << 20;
const ACCESSES_PER_ENTRY_BEFORE_AGING: usize = 10;

/// Hard residency limits for [`VersionedQueryCache`].
///
/// A zero entry or weight limit disables admission while preserving exact
/// miss computation. Logical weight is supplied by the cache's weigher; it
/// need not equal allocator bytes, but it must use one consistent unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryCacheLimits {
    max_entries: usize,
    max_weight: usize,
}

impl QueryCacheLimits {
    /// Construct entry-count and logical-weight bounds.
    pub const fn new(max_entries: usize, max_weight: usize) -> Self {
        Self {
            max_entries,
            max_weight,
        }
    }

    /// Maximum number of resident `(query, max_distance)` results.
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    /// Maximum aggregate logical weight of resident results.
    pub const fn max_weight(self) -> usize {
        self.max_weight
    }

    const fn admission_enabled(self) -> bool {
        self.max_entries != 0 && self.max_weight != 0
    }
}

impl Default for QueryCacheLimits {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_WEIGHT)
    }
}

/// Computes the logical residency weight of one complete cache entry.
///
/// Custom weighers can account for nested allocations (for example, the
/// capacities of result `String`s) or application-specific retention costs.
/// The returned weight is clamped to at least one so every entry consumes
/// finite capacity.
pub trait QueryCacheWeigher<V> {
    /// Return the logical weight of `results` under this key.
    fn weight(&self, query: &str, max_distance: usize, results: &[V]) -> usize;
}

impl<V, F> QueryCacheWeigher<V> for F
where
    F: Fn(&str, usize, &[V]) -> usize,
{
    fn weight(&self, query: &str, max_distance: usize, results: &[V]) -> usize {
        self(query, max_distance, results)
    }
}

/// Conservative shallow-size weigher used by [`VersionedQueryCache::new`].
///
/// It includes resident metadata, the query bytes, and the result slice's
/// inline elements. A custom weigher should be used when `V` owns substantial
/// heap data whose size matters to the application's memory budget.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultQueryCacheWeigher;

impl<V> QueryCacheWeigher<V> for DefaultQueryCacheWeigher {
    fn weight(&self, query: &str, _max_distance: usize, results: &[V]) -> usize {
        size_of::<Resident<V>>()
            .saturating_add(2 * size_of::<usize>())
            .saturating_add(query.len())
            .saturating_add(size_of_val(results))
            .max(1)
    }
}

/// Cumulative cache-policy counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueryCacheStats {
    requests: u64,
    hits: u64,
    misses: u64,
    admissions: u64,
    rejections: u64,
    evictions: u64,
}

impl QueryCacheStats {
    /// Total lookups, including hits and misses.
    pub const fn requests(self) -> u64 {
        self.requests
    }

    /// Lookups served by a resident result.
    pub const fn hits(self) -> u64 {
        self.hits
    }

    /// Lookups whose exact compute closure ran.
    pub const fn misses(self) -> u64 {
        self.misses
    }

    /// Miss results admitted to residency.
    pub const fn admissions(self) -> u64 {
        self.admissions
    }

    /// Computed miss results not admitted due to limits or admission policy.
    pub const fn rejections(self) -> u64 {
        self.rejections
    }

    /// Resident results displaced by accepted candidates.
    pub const fn evictions(self) -> u64 {
        self.evictions
    }
}

#[derive(Clone, Debug)]
struct FrequencySketch {
    /// Four row-major arrays, packed two 4-bit counters per byte.
    counters: Box<[u8]>,
    /// Two-probe doorkeeper. A first observation sets bits but not counters.
    doorkeeper: Box<[u64]>,
    width_mask: usize,
    accesses: usize,
    reset_at: usize,
}

impl FrequencySketch {
    fn new(max_entries: usize) -> Self {
        let desired = max_entries
            .saturating_mul(8)
            .clamp(MIN_SKETCH_WIDTH, MAX_SKETCH_WIDTH);
        let width = desired.next_power_of_two().min(MAX_SKETCH_WIDTH);
        let counter_count = width * SKETCH_ROWS;
        let reset_at = max_entries
            .saturating_mul(ACCESSES_PER_ENTRY_BEFORE_AGING)
            .max(MIN_SKETCH_WIDTH);
        Self {
            counters: vec![0; counter_count.div_ceil(2)].into_boxed_slice(),
            doorkeeper: vec![0; width.div_ceil(64)].into_boxed_slice(),
            width_mask: width - 1,
            accesses: 0,
            reset_at,
        }
    }

    #[inline]
    fn record(&mut self, hash: u64) {
        self.accesses = self.accesses.saturating_add(1);
        if self.doorkeeper_contains(hash) {
            for row in 0..SKETCH_ROWS {
                let index = self.counter_index(hash, row);
                self.increment_counter(index);
            }
        } else {
            self.set_doorkeeper(hash);
        }

        if self.accesses >= self.reset_at {
            self.age();
        }
    }

    #[inline]
    fn estimate(&self, hash: u64) -> u8 {
        let mut estimate = u8::MAX;
        for row in 0..SKETCH_ROWS {
            estimate = estimate.min(self.counter(self.counter_index(hash, row)));
        }
        estimate.saturating_add(u8::from(self.doorkeeper_contains(hash)))
    }

    fn clear(&mut self) {
        self.counters.fill(0);
        self.doorkeeper.fill(0);
        self.accesses = 0;
    }

    fn age(&mut self) {
        // Each nibble can be halved independently with one mask-and-shift.
        for byte in &mut self.counters {
            *byte = (*byte >> 1) & 0x77;
        }
        self.doorkeeper.fill(0);
        self.accesses = 0;
    }

    #[inline(always)]
    fn counter_index(&self, hash: u64, row: usize) -> usize {
        let mixed = mix64(hash ^ ROW_SEEDS[row]);
        row * (self.width_mask + 1) + ((mixed as usize) & self.width_mask)
    }

    #[inline(always)]
    fn counter(&self, index: usize) -> u8 {
        let byte = self.counters[index >> 1];
        if index & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        }
    }

    #[inline(always)]
    fn increment_counter(&mut self, index: usize) {
        let byte = &mut self.counters[index >> 1];
        let shift = (index & 1) * 4;
        let value = (*byte >> shift) & 0x0f;
        if value != 0x0f {
            *byte = (*byte & !(0x0f << shift)) | ((value + 1) << shift);
        }
    }

    #[inline(always)]
    fn doorkeeper_contains(&self, hash: u64) -> bool {
        self.doorkeeper_bit(hash) && self.doorkeeper_bit(hash.rotate_left(32) ^ ROW_SEEDS[1])
    }

    #[inline(always)]
    fn set_doorkeeper(&mut self, hash: u64) {
        self.set_doorkeeper_bit(hash);
        self.set_doorkeeper_bit(hash.rotate_left(32) ^ ROW_SEEDS[1]);
    }

    #[inline(always)]
    fn doorkeeper_bit(&self, hash: u64) -> bool {
        let bit = (mix64(hash) as usize) & self.width_mask;
        self.doorkeeper[bit >> 6] & (1u64 << (bit & 63)) != 0
    }

    #[inline(always)]
    fn set_doorkeeper_bit(&mut self, hash: u64) {
        let bit = (mix64(hash) as usize) & self.width_mask;
        self.doorkeeper[bit >> 6] |= 1u64 << (bit & 63);
    }
}

const ROW_SEEDS: [u64; SKETCH_ROWS] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
    0xd6e8_feb8_6659_fd93,
];

#[inline(always)]
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline]
fn cache_query_hash<S: BuildHasher>(hash_builder: &S, query: &[u8]) -> u64 {
    hash_builder.hash_one(query)
}

#[inline]
fn cache_frequency_hash(query_hash: u64, max_distance: usize) -> u64 {
    mix64(query_hash ^ mix64(max_distance as u64 ^ ROW_SEEDS[0]))
}

fn cache_query_hasher() -> AHashRandomState {
    #[cfg(feature = "benchmark-controls")]
    if std::env::var_os("LIBLEVENSHTEIN_CAUSAL_QUERY_CACHE_FIXED_HASH_SEED").is_some() {
        return AHashRandomState::with_seeds(
            0x43d2_7a91_b4e8_c56f,
            0x9f18_6c05_7bd3_a2e4,
            0x26b7_d94c_e150_8fa3,
            0xda84_31f6_2c79_b05e,
        );
    }
    AHashRandomState::new()
}

#[cfg(feature = "benchmark-controls")]
fn allocating_victim_plan_enabled() -> bool {
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_ALLOCATING_QUERY_CACHE_VICTIM_PLAN").is_some()
    })
}

#[derive(Clone, Debug)]
struct Resident<V> {
    query: Arc<[u8]>,
    query_hash: u64,
    max_distance: usize,
    results: Arc<[V]>,
    weight: usize,
    frequency_hash: u64,
    visited: bool,
}

type QueryVariants = SmallVec<[(usize, usize); 4]>;

#[derive(Clone, Debug)]
struct QueryRecord {
    query: Arc<[u8]>,
    variants: QueryVariants,
}

type QueryHashBucket = SmallVec<[QueryRecord; 1]>;

/// A bounded cache of fuzzy-query results scoped to one dictionary version.
///
/// Results are shared via `Arc<[V]>`, so a hit clones only an `Arc`. Query
/// lookup borrows the query bytes; it does not allocate an owned lookup key.
/// The string-oriented methods preserve the original ergonomic API, while the
/// binary-key methods let byte, token, and structured-query callers share the
/// same policy without lossy encoding. This type
/// is intentionally mutable, single-owner, and synchronization-free: callers
/// choose owner sharding or synchronization at the workload boundary instead
/// of paying for coordination on every local hit.
#[derive(Clone, Debug)]
pub struct VersionedQueryCache<V, W = DefaultQueryCacheWeigher> {
    version: u64,
    limits: QueryCacheLimits,
    weigher: W,
    query_hasher: AHashRandomState,
    index: FxHashMap<u64, QueryHashBucket>,
    slots: Vec<Option<Resident<V>>>,
    free_slots: Vec<usize>,
    victim_scratch: Vec<usize>,
    pending_clear_start: usize,
    pending_clear_len: usize,
    resident_weight: usize,
    hand: usize,
    frequency: Option<FrequencySketch>,
    stats: QueryCacheStats,
}

impl<V> Default for VersionedQueryCache<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> VersionedQueryCache<V> {
    /// Create an empty cache with 1,024-entry and 64 MiB logical-weight bounds.
    pub fn new() -> Self {
        Self::with_limits(QueryCacheLimits::default())
    }

    /// Create an empty cache with explicit hard bounds and the default weigher.
    pub fn with_limits(limits: QueryCacheLimits) -> Self {
        Self::with_limits_and_weigher(limits, DefaultQueryCacheWeigher)
    }

    /// Create a cache with explicit limits and a custom, monomorphized weigher.
    pub fn with_limits_and_weigher<W>(
        limits: QueryCacheLimits,
        weigher: W,
    ) -> VersionedQueryCache<V, W>
    where
        W: QueryCacheWeigher<V>,
    {
        VersionedQueryCache::from_parts(limits, weigher)
    }
}

impl<V, W> VersionedQueryCache<V, W>
where
    W: QueryCacheWeigher<V>,
{
    fn from_parts(limits: QueryCacheLimits, weigher: W) -> Self {
        Self {
            version: 0,
            limits,
            weigher,
            query_hasher: cache_query_hasher(),
            index: FxHashMap::default(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            victim_scratch: Vec::new(),
            pending_clear_start: 0,
            pending_clear_len: 0,
            resident_weight: 0,
            hand: 0,
            frequency: None,
            stats: QueryCacheStats::default(),
        }
    }

    /// Number of resident `(query, max_distance)` entries.
    pub fn len(&self) -> usize {
        self.slots.len() - self.free_slots.len()
    }

    /// Whether the cache holds no resident entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dictionary version to which resident results belong.
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Configured hard limits.
    pub const fn limits(&self) -> QueryCacheLimits {
        self.limits
    }

    /// Aggregate logical weight of resident entries.
    pub const fn resident_weight(&self) -> usize {
        self.resident_weight
    }

    /// Cumulative policy counters since construction or the last [`reset_stats`](Self::reset_stats).
    pub const fn stats(&self) -> QueryCacheStats {
        self.stats
    }

    /// Reset counters without changing residency, frequency, or dictionary version.
    pub fn reset_stats(&mut self) {
        self.stats = QueryCacheStats::default();
    }

    /// Test residency without affecting recency or frequency.
    pub fn contains(&self, query: &str, max_distance: usize) -> bool {
        self.contains_key(query.as_bytes(), max_distance)
    }

    /// Test a binary query key without affecting recency or frequency.
    ///
    /// Callers sharing a cache across query families must include an
    /// unambiguous family/domain discriminator in `query_key`. A cache owned by
    /// one fixed automaton and unit domain can use the query units directly.
    pub fn contains_key(&self, query_key: &[u8], max_distance: usize) -> bool {
        let query_hash = cache_query_hash(&self.query_hasher, query_key);
        self.slot_for(query_hash, query_key, max_distance).is_some()
    }

    /// Drop every entry while retaining the current dictionary version.
    pub fn clear(&mut self) {
        self.clear_residency_and_policy();
    }

    /// Look up `(query, max_distance)` at `dict_version` or compute it exactly.
    ///
    /// A version change clears stale residency before lookup. The closure runs
    /// exactly once on a miss. If it panics, no partially constructed resident
    /// is installed. A computed result can be returned without admission when
    /// it exceeds the weight bound or loses the approximate-frequency contest.
    pub fn get_or_compute<F>(
        &mut self,
        query: &str,
        max_distance: usize,
        dict_version: u64,
        compute: F,
    ) -> Arc<[V]>
    where
        F: FnOnce() -> Vec<V>,
    {
        match self.try_get_or_compute(query, max_distance, dict_version, || {
            Ok::<_, std::convert::Infallible>(compute())
        }) {
            Ok(results) => results,
            Err(error) => match error {},
        }
    }

    /// Fallible form of [`get_or_compute`](Self::get_or_compute).
    ///
    /// A failed miss increments request/miss counters but never admits a
    /// partial result. This is the appropriate entry point when exact
    /// computation crosses an I/O, callback, or foreign-provider boundary.
    pub fn try_get_or_compute<F, E>(
        &mut self,
        query: &str,
        max_distance: usize,
        dict_version: u64,
        compute: F,
    ) -> Result<Arc<[V]>, E>
    where
        F: FnOnce() -> Result<Vec<V>, E>,
    {
        self.try_get_or_compute_impl(
            query.as_bytes(),
            max_distance,
            dict_version,
            compute,
            |weigher, results| weigher.weight(query, max_distance, results),
        )
    }

    /// Look up an arbitrary binary query key or compute it exactly.
    ///
    /// `weigh` runs only after a successful miss and supplies the logical
    /// residency weight. The key is copied only when the result is admitted;
    /// hits and rejected misses use the caller's borrowed slice directly.
    /// Approximate admission affects residency only, never returned values.
    pub fn try_get_or_compute_key<F, E, M>(
        &mut self,
        query_key: &[u8],
        max_distance: usize,
        dict_version: u64,
        compute: F,
        weigh: M,
    ) -> Result<Arc<[V]>, E>
    where
        F: FnOnce() -> Result<Vec<V>, E>,
        M: FnOnce(&[V]) -> usize,
    {
        self.try_get_or_compute_impl(
            query_key,
            max_distance,
            dict_version,
            compute,
            |_default_weigher, results| weigh(results),
        )
    }

    fn try_get_or_compute_impl<F, E, M>(
        &mut self,
        query_key: &[u8],
        max_distance: usize,
        dict_version: u64,
        compute: F,
        weigh: M,
    ) -> Result<Arc<[V]>, E>
    where
        F: FnOnce() -> Result<Vec<V>, E>,
        M: FnOnce(&W, &[V]) -> usize,
    {
        self.reconcile_version(dict_version);
        self.stats.requests = self.stats.requests.saturating_add(1);

        if !self.limits.admission_enabled() {
            self.stats.misses = self.stats.misses.saturating_add(1);
            self.stats.rejections = self.stats.rejections.saturating_add(1);
            return compute().map(Into::into);
        }

        let query_hash = cache_query_hash(&self.query_hasher, query_key);
        let frequency_hash = cache_frequency_hash(query_hash, max_distance);
        self.frequency
            .get_or_insert_with(|| FrequencySketch::new(self.limits.max_entries))
            .record(frequency_hash);

        if let Some(slot) = self.slot_for(query_hash, query_key, max_distance) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            let resident = self.slots[slot].as_mut().expect("indexed resident slot");
            resident.visited = true;
            return Ok(Arc::clone(&resident.results));
        }

        self.stats.misses = self.stats.misses.saturating_add(1);
        let results: Arc<[V]> = compute()?.into();
        let weight = weigh(&self.weigher, results.as_ref()).max(1);
        if self.admit(
            query_hash,
            query_key,
            max_distance,
            &results,
            weight,
            frequency_hash,
        ) {
            self.stats.admissions = self.stats.admissions.saturating_add(1);
        } else {
            self.stats.rejections = self.stats.rejections.saturating_add(1);
        }
        Ok(results)
    }

    fn reconcile_version(&mut self, dict_version: u64) {
        if dict_version != self.version {
            self.clear_residency_and_policy();
            self.version = dict_version;
        }
    }

    fn clear_residency_and_policy(&mut self) {
        self.index.clear();
        self.slots.clear();
        self.free_slots.clear();
        self.victim_scratch.clear();
        self.pending_clear_start = 0;
        self.pending_clear_len = 0;
        self.resident_weight = 0;
        self.hand = 0;
        if let Some(frequency) = &mut self.frequency {
            frequency.clear();
        }
    }

    #[inline]
    fn slot_for(&self, query_hash: u64, query: &[u8], max_distance: usize) -> Option<usize> {
        self.index.get(&query_hash).and_then(|bucket| {
            bucket
                .iter()
                .find(|record| record.query.as_ref() == query)
                .and_then(|record| {
                    record
                        .variants
                        .binary_search_by_key(&max_distance, |&(distance, _)| distance)
                        .ok()
                        .map(|index| record.variants[index].1)
                })
        })
    }

    fn admit(
        &mut self,
        query_hash: u64,
        query: &[u8],
        max_distance: usize,
        results: &Arc<[V]>,
        weight: usize,
        frequency_hash: u64,
    ) -> bool {
        if weight > self.limits.max_weight {
            return false;
        }

        let over_entries = self.len() >= self.limits.max_entries;
        let over_weight = self.resident_weight > self.limits.max_weight - weight;
        if over_entries || over_weight {
            if !self.plan_victims(weight, frequency_hash) {
                return false;
            }
            self.apply_planned_victims();
        }

        self.insert_resident(
            query_hash,
            query,
            max_distance,
            results,
            weight,
            frequency_hash,
        );
        debug_assert!(self.len() <= self.limits.max_entries);
        debug_assert!(self.resident_weight <= self.limits.max_weight);
        true
    }

    fn plan_victims(&mut self, candidate_weight: usize, candidate_hash: u64) -> bool {
        #[cfg(feature = "benchmark-controls")]
        if allocating_victim_plan_enabled() {
            return self.plan_victims_allocating(candidate_weight, candidate_hash);
        }
        self.plan_victims_reused(candidate_weight, candidate_hash)
    }

    fn plan_victims_reused(&mut self, candidate_weight: usize, candidate_hash: u64) -> bool {
        if self.slots.is_empty() {
            return false;
        }

        let candidate_frequency = self
            .frequency
            .as_ref()
            .expect("admission-enabled requests initialize the frequency sketch")
            .estimate(candidate_hash);
        self.victim_scratch.clear();
        self.pending_clear_len = 0;
        let slot_count = self.slots.len();
        let start_hand = self.hand % slot_count;
        let mut hand = start_hand;
        let mut removed_weight = 0usize;
        let mut removed_entries = 0usize;
        let required_entries = usize::from(self.len() >= self.limits.max_entries);
        let admissible_resident_weight = self.limits.max_weight - candidate_weight;
        let required_weight = self
            .resident_weight
            .saturating_sub(admissible_resident_weight);
        // One circular pass considers every unreferenced resident and
        // logically clears every referenced resident. A second pass considers
        // exactly those formerly referenced residents. Therefore a third pass
        // cannot discover another legal victim: every resident has already
        // either been selected or rejected by TinyLFU admission.
        let max_steps = slot_count.saturating_mul(2).max(1);
        let mut steps = 0usize;

        while removed_entries < required_entries || removed_weight < required_weight {
            if steps >= max_steps {
                self.victim_scratch.clear();
                return false;
            }
            let slot = hand;
            hand = (hand + 1) % slot_count;
            steps += 1;
            let Some(victim) = self.slots[slot].as_ref() else {
                continue;
            };

            if steps <= slot_count {
                if victim.visited {
                    continue;
                }
            } else if !victim.visited {
                // Unreferenced residents were already considered during the
                // first pass. Referenced residents become eligible exactly
                // once during this virtual second pass.
                continue;
            }

            let victim_hash = victim.frequency_hash;
            let victim_weight = victim.weight;
            let victim_frequency = self
                .frequency
                .as_ref()
                .expect("admission-enabled requests initialize the frequency sketch")
                .estimate(victim_hash);
            // Compare frequency/weight ratios exactly with cross-products.
            // Ties retain the resident, preventing one-hit scans from cycling
            // an equally valuable cache at full capacity.
            let candidate_utility = u128::from(candidate_frequency) * victim_weight.max(1) as u128;
            let victim_utility = u128::from(victim_frequency) * candidate_weight.max(1) as u128;
            if candidate_utility <= victim_utility {
                self.victim_scratch.clear();
                return false;
            }

            self.victim_scratch.push(slot);
            removed_entries += 1;
            removed_weight = removed_weight
                .checked_add(victim_weight)
                .expect("selected resident weights stay within the configured bound");
        }

        self.hand = hand;
        self.pending_clear_start = start_hand;
        self.pending_clear_len = steps.min(slot_count);
        true
    }

    #[cfg(feature = "benchmark-controls")]
    fn plan_victims_allocating(&mut self, candidate_weight: usize, candidate_hash: u64) -> bool {
        if self.slots.is_empty() {
            return false;
        }

        let candidate_frequency = self
            .frequency
            .as_ref()
            .expect("admission-enabled requests initialize the frequency sketch")
            .estimate(candidate_hash);
        let mut visited = self
            .slots
            .iter()
            .map(|resident| resident.as_ref().is_some_and(|resident| resident.visited))
            .collect::<Vec<_>>();
        let mut selected = vec![false; self.slots.len()];
        let mut victims = Vec::new();
        let mut hand = self.hand % self.slots.len();
        let mut removed_weight = 0usize;
        let mut removed_entries = 0usize;
        let required_entries = usize::from(self.len() >= self.limits.max_entries);
        let admissible_resident_weight = self.limits.max_weight - candidate_weight;
        let required_weight = self
            .resident_weight
            .saturating_sub(admissible_resident_weight);
        let max_steps = self.slots.len().saturating_mul(3).max(1);
        let mut steps = 0usize;

        while removed_entries < required_entries || removed_weight < required_weight {
            if steps >= max_steps {
                return false;
            }
            let slot = hand;
            hand = (hand + 1) % self.slots.len();
            steps += 1;
            let Some(victim) = self.slots[slot].as_ref() else {
                continue;
            };
            if selected[slot] {
                continue;
            }
            if visited[slot] {
                visited[slot] = false;
                continue;
            }

            let victim_frequency = self
                .frequency
                .as_ref()
                .expect("admission-enabled requests initialize the frequency sketch")
                .estimate(victim.frequency_hash);
            let candidate_utility = u128::from(candidate_frequency) * victim.weight.max(1) as u128;
            let victim_utility = u128::from(victim_frequency) * candidate_weight.max(1) as u128;
            if candidate_utility <= victim_utility {
                return false;
            }

            selected[slot] = true;
            victims.push(slot);
            removed_entries += 1;
            removed_weight = removed_weight
                .checked_add(victim.weight)
                .expect("selected resident weights stay within the configured bound");
        }

        for (slot, visited) in self.slots.iter_mut().zip(visited) {
            if let Some(resident) = slot {
                resident.visited = visited;
            }
        }
        self.hand = hand;
        self.victim_scratch.clear();
        self.victim_scratch.extend(victims);
        true
    }

    fn apply_planned_victims(&mut self) {
        let slot_count = self.slots.len();
        for offset in 0..self.pending_clear_len {
            let slot = (self.pending_clear_start + offset) % slot_count;
            if let Some(resident) = &mut self.slots[slot] {
                resident.visited = false;
            }
        }
        self.pending_clear_len = 0;
        while let Some(victim) = self.victim_scratch.pop() {
            self.remove_resident(victim);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
    }

    fn insert_resident(
        &mut self,
        query_hash: u64,
        query: &[u8],
        max_distance: usize,
        results: &Arc<[V]>,
        weight: usize,
        frequency_hash: u64,
    ) {
        let query: Arc<[u8]> = self
            .index
            .get(&query_hash)
            .and_then(|bucket| bucket.iter().find(|record| record.query.as_ref() == query))
            .map(|record| Arc::clone(&record.query))
            .unwrap_or_else(|| Arc::from(query));
        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None => {
                self.ensure_metadata_capacity(self.slots.len() + 1);
                self.slots.push(None);
                self.slots.len() - 1
            }
        };
        self.slots[slot] = Some(Resident {
            query: Arc::clone(&query),
            query_hash,
            max_distance,
            results: Arc::clone(results),
            weight,
            frequency_hash,
            visited: false,
        });
        let bucket = self.index.entry(query_hash).or_default();
        let record_index = bucket
            .iter()
            .position(|record| record.query.as_ref() == query.as_ref())
            .unwrap_or_else(|| {
                bucket.push(QueryRecord {
                    query: Arc::clone(&query),
                    variants: QueryVariants::new(),
                });
                bucket.len() - 1
            });
        let variants = &mut bucket[record_index].variants;
        let insertion = variants
            .binary_search_by_key(&max_distance, |&(distance, _)| distance)
            .expect_err("a cache miss cannot insert a duplicate distance variant");
        variants.insert(insertion, (max_distance, slot));
        self.resident_weight = self
            .resident_weight
            .checked_add(weight)
            .expect("admission planning proves the aggregate weight cannot overflow");
    }

    fn remove_resident(&mut self, slot: usize) {
        let resident = self.slots[slot].take().expect("victim slot is resident");
        self.resident_weight = self
            .resident_weight
            .checked_sub(resident.weight)
            .expect("resident weight invariant");
        let remove_query = {
            let bucket = self
                .index
                .get_mut(&resident.query_hash)
                .expect("resident query hash is indexed");
            let record_index = bucket
                .iter()
                .position(|record| record.query.as_ref() == resident.query.as_ref())
                .expect("resident query is indexed in its collision bucket");
            let variants = &mut bucket[record_index].variants;
            let variant = variants
                .binary_search_by_key(&resident.max_distance, |&(distance, _)| distance)
                .expect("resident variant is indexed");
            assert_eq!(
                variants[variant].1, slot,
                "distance maps to its resident slot"
            );
            variants.remove(variant);
            if variants.is_empty() {
                bucket.swap_remove(record_index);
            }
            bucket.is_empty()
        };
        if remove_query {
            self.index.remove(&resident.query_hash);
        }
        self.free_slots.push(slot);
    }

    fn ensure_metadata_capacity(&mut self, target: usize) {
        if self.slots.capacity() < target {
            self.slots.reserve(target - self.slots.len());
        }
        if self.free_slots.capacity() < target {
            self.free_slots.reserve(target - self.free_slots.len());
        }
        if self.victim_scratch.capacity() < target {
            self.victim_scratch
                .reserve(target - self.victim_scratch.len());
        }
        if self.index.capacity() < target {
            self.index.reserve(target.saturating_sub(self.index.len()));
        }
    }

    #[cfg(test)]
    fn assert_invariants(&self) {
        assert!(self.len() <= self.limits.max_entries);
        assert!(self.resident_weight <= self.limits.max_weight);
        let actual_weight: usize = self
            .slots
            .iter()
            .filter_map(Option::as_ref)
            .map(|resident| resident.weight)
            .sum();
        assert_eq!(self.resident_weight, actual_weight);
        let indexed: usize = self
            .index
            .values()
            .flat_map(|bucket| bucket.iter())
            .map(|record| record.variants.len())
            .sum();
        assert_eq!(indexed, self.len());
        for (&query_hash, bucket) in &self.index {
            for record in bucket {
                assert!(record
                    .variants
                    .windows(2)
                    .all(|window| window[0].0 < window[1].0));
                assert_eq!(
                    cache_query_hash(&self.query_hasher, record.query.as_ref()),
                    query_hash
                );
                for &(distance, slot) in &record.variants {
                    let resident = self.slots[slot].as_ref().expect("index points to resident");
                    assert_eq!(distance, resident.max_distance);
                    assert_eq!(query_hash, resident.query_hash);
                    assert_eq!(record.query.as_ref(), resident.query.as_ref());
                }
            }
        }
        assert!(self.victim_scratch.is_empty());
        assert_eq!(self.pending_clear_len, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn unit_weigher(_query: &str, _distance: usize, _results: &[u32]) -> usize {
        1
    }

    fn unit_cache(
        max_entries: usize,
    ) -> VersionedQueryCache<u32, impl QueryCacheWeigher<u32> + Clone> {
        VersionedQueryCache::with_limits_and_weigher(
            QueryCacheLimits::new(max_entries, max_entries),
            unit_weigher,
        )
    }

    #[test]
    fn hit_does_not_recompute_and_returns_same_arc() {
        let mut cache: VersionedQueryCache<u32> = VersionedQueryCache::new();
        let calls = Cell::new(0);
        let a = cache.get_or_compute("q", 1, 0, || {
            calls.set(calls.get() + 1);
            vec![1, 2, 3]
        });
        let b = cache.get_or_compute("q", 1, 0, || {
            calls.set(calls.get() + 1);
            vec![9, 9]
        });
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(calls.get(), 1);
        assert_eq!(cache.stats().hits(), 1);
        cache.assert_invariants();
    }

    #[test]
    fn borrowed_query_lookup_handles_multiple_distances() {
        let mut cache = unit_cache(4);
        let _ = cache.get_or_compute("query", 1, 0, || vec![1]);
        let _ = cache.get_or_compute("query", 2, 0, || vec![2]);
        assert!(cache.contains("query", 1));
        assert!(cache.contains("query", 2));
        assert_eq!(&*cache.get_or_compute("query", 2, 0, Vec::new), &[2]);
        cache.assert_invariants();
    }

    #[test]
    fn binary_keys_are_exact_and_allocate_only_when_admitted() {
        let mut cache = unit_cache(4);
        let first = cache
            .try_get_or_compute_key(&[0, 0xff, 1], 2, 7, || Ok::<_, ()>(vec![11]), |_| 1)
            .unwrap();
        let second = cache
            .try_get_or_compute_key(&[0, 0xff, 2], 2, 7, || Ok::<_, ()>(vec![22]), |_| 1)
            .unwrap();
        let hit = cache
            .try_get_or_compute_key(&[0, 0xff, 1], 2, 7, || Ok::<_, ()>(vec![99]), |_| 1)
            .unwrap();

        assert_eq!(&*first, &[11]);
        assert_eq!(&*second, &[22]);
        assert!(Arc::ptr_eq(&first, &hit));
        assert!(cache.contains_key(&[0, 0xff, 1], 2));
        assert!(cache.contains_key(&[0, 0xff, 2], 2));
        cache.assert_invariants();
    }

    #[test]
    fn failed_computation_never_installs_a_partial_entry() {
        let mut cache = unit_cache(4);
        let error = cache
            .try_get_or_compute("query", 1, 9, || Err::<Vec<u32>, _>("provider fault"))
            .unwrap_err();

        assert_eq!(error, "provider fault");
        assert!(!cache.contains("query", 1));
        assert_eq!(cache.stats().requests(), 1);
        assert_eq!(cache.stats().misses(), 1);
        assert_eq!(cache.stats().admissions(), 0);
        assert_eq!(cache.stats().rejections(), 0);
        cache.assert_invariants();
    }

    #[test]
    fn entry_and_weight_limits_hold_after_every_operation() {
        let weigher = |_query: &str, _distance: usize, results: &[u32]| results.len();
        let mut cache =
            VersionedQueryCache::with_limits_and_weigher(QueryCacheLimits::new(5, 11), weigher);
        for index in 0..500 {
            let query = format!("q{}", index % 37);
            let result_len = index % 7 + 1;
            let _ = cache.get_or_compute(&query, index % 3, 0, || vec![index as u32; result_len]);
            cache.assert_invariants();
        }
    }

    #[test]
    fn near_usize_max_weights_never_bypass_the_hard_bound() {
        let weigher = |query: &str, _distance: usize, _results: &[u32]| match query {
            "almost-max" => usize::MAX - 1,
            "small" => 2,
            _ => 1,
        };
        let mut cache = VersionedQueryCache::with_limits_and_weigher(
            QueryCacheLimits::new(2, usize::MAX),
            weigher,
        );

        let _ = cache.get_or_compute("almost-max", 0, 0, || vec![1]);
        assert_eq!(cache.resident_weight(), usize::MAX - 1);
        let _ = cache.get_or_compute("small", 0, 0, || vec![2]);

        assert!(cache.contains("small", 0));
        assert!(!cache.contains("almost-max", 0));
        assert_eq!(cache.resident_weight(), 2);
        cache.assert_invariants();
    }

    #[test]
    fn one_heavy_hot_candidate_can_transactionally_displace_multiple_victims() {
        let weigher = |_query: &str, _distance: usize, results: &[u32]| results[0] as usize;
        let mut cache =
            VersionedQueryCache::with_limits_and_weigher(QueryCacheLimits::new(4, 9), weigher);
        for key in ["a", "b", "c"] {
            let _ = cache.get_or_compute(key, 0, 0, || vec![3]);
        }

        for _ in 0..16 {
            let _ = cache.get_or_compute("heavy", 0, 0, || vec![7]);
            if cache.contains("heavy", 0) {
                break;
            }
        }

        assert!(cache.contains("heavy", 0));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resident_weight(), 7);
        assert_eq!(cache.stats().evictions(), 3);
        cache.assert_invariants();
    }

    #[test]
    fn many_distance_variants_share_one_query_allocation_and_remain_exact() {
        let mut cache = unit_cache(64);
        for distance in (0..64).rev() {
            let _ = cache.get_or_compute("one-query", distance, 0, || vec![distance as u32]);
        }

        let query_hash = cache_query_hash(&cache.query_hasher, b"one-query");
        let variants = &cache
            .index
            .get(&query_hash)
            .expect("query hash is indexed")
            .iter()
            .find(|record| record.query.as_ref() == b"one-query")
            .expect("query is indexed")
            .variants;
        assert!(variants.windows(2).all(|window| window[0].0 < window[1].0));
        let variant_slots = variants.iter().map(|&(_, slot)| slot).collect::<Vec<_>>();
        let first_query = cache.slots[variant_slots[0]]
            .as_ref()
            .expect("first variant is resident")
            .query
            .clone();
        for distance in 0..64 {
            let slot = cache
                .slot_for(query_hash, b"one-query", distance)
                .expect("variant exists");
            let resident = cache.slots[slot].as_ref().expect("variant is resident");
            assert!(Arc::ptr_eq(&first_query, &resident.query));
            assert_eq!(
                &*cache.get_or_compute("one-query", distance, 0, Vec::new),
                &[distance as u32]
            );
        }
        cache.assert_invariants();
    }

    #[test]
    fn pressure_rejections_reuse_all_eviction_metadata_capacity() {
        const CAPACITY: usize = 16;
        let mut cache = unit_cache(CAPACITY);
        for key in 0..CAPACITY {
            let query = format!("hot-{key}");
            let _ = cache.get_or_compute(&query, 0, 0, || vec![key as u32]);
        }
        for _ in 0..8 {
            for key in 0..CAPACITY {
                let query = format!("hot-{key}");
                let _ = cache.get_or_compute(&query, 0, 0, Vec::new);
            }
        }
        let capacities = (
            cache.index.capacity(),
            cache.slots.capacity(),
            cache.free_slots.capacity(),
            cache.victim_scratch.capacity(),
        );

        for key in 0..(CAPACITY * 10) {
            let query = format!("scan-{key}");
            let _ = cache.get_or_compute(&query, 0, 0, || vec![key as u32]);
        }

        assert_eq!(
            capacities,
            (
                cache.index.capacity(),
                cache.slots.capacity(),
                cache.free_slots.capacity(),
                cache.victim_scratch.capacity(),
            )
        );
        cache.assert_invariants();
    }

    #[test]
    fn rejected_candidate_does_not_advance_or_age_sieve_residents() {
        let mut cache = unit_cache(2);
        for (query, value) in [("hot-a", 1), ("hot-b", 2)] {
            let _ = cache.get_or_compute(query, 0, 0, || vec![value]);
        }
        for _ in 0..8 {
            for query in ["hot-a", "hot-b"] {
                let _ = cache.get_or_compute(query, 0, 0, Vec::new);
            }
        }
        let hand_before = cache.hand;
        let visited_before = cache
            .slots
            .iter()
            .map(|slot| slot.as_ref().map(|resident| resident.visited))
            .collect::<Vec<_>>();

        let result = cache.get_or_compute("one-hit-cold", 0, 0, || vec![3]);

        assert_eq!(&*result, &[3]);
        assert!(!cache.contains("one-hit-cold", 0));
        assert_eq!(cache.hand, hand_before);
        assert_eq!(
            cache
                .slots
                .iter()
                .map(|slot| slot.as_ref().map(|resident| resident.visited))
                .collect::<Vec<_>>(),
            visited_before
        );
        cache.assert_invariants();
    }

    #[test]
    fn disabled_cache_defers_all_policy_storage() {
        let mut cache = unit_cache(0);
        assert!(cache.frequency.is_none());
        assert_eq!(cache.index.capacity(), 0);
        assert_eq!(cache.slots.capacity(), 0);

        assert_eq!(&*cache.get_or_compute("q", 1, 0, || vec![7]), &[7]);

        assert!(cache.frequency.is_none());
        assert_eq!(cache.index.capacity(), 0);
        assert_eq!(cache.slots.capacity(), 0);
        assert_eq!(cache.stats().rejections(), 1);
        cache.assert_invariants();
    }

    #[test]
    fn custom_weigher_can_enforce_deep_string_capacity() {
        let deep_string_weigher = |query: &str, _distance: usize, results: &[String]| {
            query
                .len()
                .checked_add(results.iter().map(String::capacity).sum::<usize>())
                .expect("test weight fits")
        };
        let mut cache = VersionedQueryCache::with_limits_and_weigher(
            QueryCacheLimits::new(4, 64),
            deep_string_weigher,
        );
        let mut retained = String::with_capacity(128);
        retained.push_str("result");
        let result = cache.get_or_compute("q", 0, 0, || vec![retained]);

        assert_eq!(&result[0], "result");
        assert!(cache.is_empty(), "deep capacity exceeds the logical bound");
        cache.assert_invariants();
    }

    #[test]
    fn oversized_and_disabled_entries_are_computed_but_never_admitted() {
        let weigher = |_query: &str, _distance: usize, results: &[u32]| results.len();
        let mut oversized =
            VersionedQueryCache::with_limits_and_weigher(QueryCacheLimits::new(4, 2), weigher);
        let calls = Cell::new(0);
        for _ in 0..2 {
            let _ = oversized.get_or_compute("large", 1, 0, || {
                calls.set(calls.get() + 1);
                vec![1, 2, 3]
            });
        }
        assert_eq!(calls.get(), 2);
        assert!(oversized.is_empty());

        let mut disabled = unit_cache(0);
        let value = disabled.get_or_compute("q", 0, 0, || vec![7]);
        assert_eq!(&*value, &[7]);
        assert!(disabled.is_empty());
    }

    #[test]
    fn hot_set_survives_a_ten_capacity_one_hit_scan() {
        const CAPACITY: usize = 20;
        let mut cache = unit_cache(CAPACITY);
        for key in 0..CAPACITY {
            let query = format!("hot-{key}");
            let _ = cache.get_or_compute(&query, 1, 0, || vec![key as u32]);
        }
        for _ in 0..12 {
            for key in 0..CAPACITY {
                let query = format!("hot-{key}");
                let _ = cache.get_or_compute(&query, 1, 0, Vec::new);
            }
        }

        for key in 0..(10 * CAPACITY) {
            let query = format!("scan-{key}");
            let _ = cache.get_or_compute(&query, 1, 0, || vec![key as u32]);
        }

        let retained = (0..CAPACITY)
            .filter(|key| cache.contains(&format!("hot-{key}"), 1))
            .count();
        assert!(
            retained * 100 >= CAPACITY * 95,
            "retained {retained}/{CAPACITY}"
        );
        cache.assert_invariants();
    }

    #[test]
    fn aging_adapts_to_a_disjoint_hot_phase() {
        const CAPACITY: usize = 8;
        let mut cache = unit_cache(CAPACITY);
        for phase in ["old", "new"] {
            for _ in 0..40 {
                for key in 0..CAPACITY {
                    let query = format!("{phase}-{key}");
                    let _ = cache.get_or_compute(&query, 1, 0, || vec![key as u32]);
                }
            }
        }
        let new_residents = (0..CAPACITY)
            .filter(|key| cache.contains(&format!("new-{key}"), 1))
            .count();
        assert_eq!(new_residents, CAPACITY);
        cache.assert_invariants();
    }

    #[test]
    fn version_change_clears_stale_residency_and_policy() {
        let mut cache = unit_cache(4);
        let _ = cache.get_or_compute("q", 1, 7, || vec![1]);
        let fresh = cache.get_or_compute("q", 1, 8, || vec![2]);
        assert_eq!(&*fresh, &[2]);
        assert_eq!(cache.version(), 8);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resident_weight(), 1);
        cache.assert_invariants();
    }

    #[test]
    fn panicking_compute_never_installs_partial_residency() {
        let mut cache = unit_cache(4);
        let _ = cache.get_or_compute("stable", 1, 0, || vec![1]);
        let before_len = cache.len();
        let before_weight = cache.resident_weight();
        let panic = catch_unwind(AssertUnwindSafe(|| {
            cache.get_or_compute("panic", 1, 0, || -> Vec<u32> { panic!("boom") });
        }));
        assert!(panic.is_err());
        assert_eq!(cache.len(), before_len);
        assert_eq!(cache.resident_weight(), before_weight);
        assert!(!cache.contains("panic", 1));
        cache.assert_invariants();
    }

    #[test]
    fn clone_has_independent_policy_state_and_shared_result_storage() {
        let mut cache = unit_cache(4);
        let original = cache.get_or_compute("q", 1, 0, || vec![1]);
        let mut cloned = cache.clone();
        let clone_hit = cloned.get_or_compute("q", 1, 0, Vec::new);
        assert!(Arc::ptr_eq(&original, &clone_hit));
        cloned.clear();
        assert!(cloned.is_empty());
        assert!(cache.contains("q", 1));
    }

    #[test]
    fn nibble_counters_saturate_and_age_independently() {
        let mut sketch = FrequencySketch::new(8);
        let query_hash = cache_query_hash(&AHashRandomState::new(), b"hot");
        let hash = cache_frequency_hash(query_hash, 1);
        for _ in 0..1000 {
            sketch.record(hash);
        }
        assert!(sketch.estimate(hash) <= 16);
        sketch.clear();
        assert_eq!(sketch.estimate(hash), 0);
    }
}
