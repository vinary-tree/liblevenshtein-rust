//! Shared snapshot-cursor traversal with an owned-node compatibility arena.

use libdictenstein::{
    CharUnit, DictionaryNode, DictionaryTraversalRoot, MappedDictionaryNode,
    SnapshotTraversalCursor, SnapshotTraversalGraph,
};
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::mem::size_of;
use std::sync::Arc;

use smallvec::SmallVec;

#[cfg(feature = "perf-instrumentation")]
use libdictenstein::SnapshotNodeIdentity;

/// Initial breadth-first traversal storage. Heap profiling on broad fuzzy
/// queries shows that 64 entries cover the common queue-growth staircase while
/// keeping each independently owned cursor's reservation bounded and small.
const TRAVERSAL_QUEUE_INITIAL_CAPACITY: usize = 64;

/// Initial compact parent-chain arena for lazy term reconstruction. The arena
/// stores metadata only; returned terms still own their public result buffers.
const PATH_ARENA_INITIAL_CAPACITY: usize = 512;

/// Maximum logical allocation capacity retained by one thread after queries
/// finish. Checked-out iterators own their buffers and are not counted here.
const MAX_RETAINED_TRAVERSAL_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Bound polymorphic workloads independently of the byte limit. Entries are
/// kept in most-recent-return order and the oldest entry is evicted first.
const MAX_RETAINED_TRAVERSAL_BUFFER_TYPES: usize = 4;

/// Bounded speculative fanout for lazy depth-first traversal. Eight covers the
/// common small-node case inline while limiting early-stop work on broad roots.
const DFS_EDGE_PAGE_CAPACITY: usize = 8;

struct ReusableTraversalBuffers<Q, P> {
    queue: VecDeque<Q>,
    path: Vec<P>,
}

struct ReusableTraversalQueue<Q> {
    queue: VecDeque<Q>,
}

struct ReusableBucketedTraversalBuffers<Q, P> {
    buckets: Vec<VecDeque<Q>>,
    path: Vec<P>,
}

struct ReusableBucketedTraversalQueues<Q> {
    buckets: Vec<VecDeque<Q>>,
}

struct ErasedTraversalBuffers {
    type_id: TypeId,
    logical_bytes: usize,
    buffers: Box<dyn Any>,
}

#[derive(Default)]
struct ThreadTraversalBufferCache {
    entries: VecDeque<ErasedTraversalBuffers>,
    logical_bytes: usize,
}

thread_local! {
    static TRAVERSAL_BUFFER_CACHE: RefCell<ThreadTraversalBufferCache> =
        RefCell::new(ThreadTraversalBufferCache::default());
}

/// Construct the common breadth-first queue with the profiled bounded
/// reservation. Monomorphization keeps this shared policy at zero abstraction
/// cost for every query surface and unit domain.
#[inline]
pub(crate) fn new_traversal_queue<T>() -> VecDeque<T> {
    VecDeque::with_capacity(TRAVERSAL_QUEUE_INITIAL_CAPACITY)
}

/// Construct the common parent-chain arena with the profiled bounded
/// reservation. The element type remains local to each specialized query.
#[inline]
pub(crate) fn new_path_arena<T>() -> Vec<T> {
    Vec::with_capacity(PATH_ARENA_INITIAL_CAPACITY)
}

/// Max-heap whose ordering function may borrow query-local context.
///
/// `std::collections::BinaryHeap` requires `T: Ord`, which forces ordering data
/// into every entry. Traversals whose ordering depends on a shared path arena
/// use this container so entries retain only compact path keys. The sift
/// operations are otherwise the conventional binary-heap algorithm.
pub(crate) struct ContextHeap<T> {
    entries: Vec<T>,
}

impl<T> ContextHeap<T> {
    #[inline]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push_by<F>(&mut self, value: T, mut compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        self.entries.push(value);
        let mut child = self.entries.len() - 1;
        while child > 0 {
            let parent = (child - 1) / 2;
            if compare(&self.entries[child], &self.entries[parent]) != std::cmp::Ordering::Greater {
                break;
            }
            self.entries.swap(child, parent);
            child = parent;
        }
    }

    pub(crate) fn pop_by<F>(&mut self, mut compare: F) -> Option<T>
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        let result = self.entries.pop()?;
        if self.entries.is_empty() {
            return Some(result);
        }

        let result = std::mem::replace(&mut self.entries[0], result);
        let mut parent = 0;
        loop {
            let left = parent * 2 + 1;
            if left >= self.entries.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.entries.len()
                && compare(&self.entries[right], &self.entries[left]) == std::cmp::Ordering::Greater
            {
                right
            } else {
                left
            };
            if compare(&self.entries[child], &self.entries[parent]) != std::cmp::Ordering::Greater {
                break;
            }
            self.entries.swap(parent, child);
            parent = child;
        }
        Some(result)
    }
}

/// Acquire exact typed queue/path owners from the current thread's bounded
/// warm cache, or construct the profiled initial capacities on a miss.
///
/// The returned containers have ordinary ownership and may move with a query
/// iterator to another thread. No borrow, lock, atomic, or TLS reference is
/// retained while traversal is active.
pub(crate) fn acquire_traversal_buffers<Q: 'static, P: 'static>() -> (VecDeque<Q>, Vec<P>) {
    acquire_traversal_buffers_with_capacity(
        TRAVERSAL_QUEUE_INITIAL_CAPACITY,
        PATH_ARENA_INITIAL_CAPACITY,
    )
}

/// Acquire queue/path owners while preserving a surface-specific cold-start
/// reservation. A retained exact-typed allocation may be larger than these
/// minima, but cache misses do not inflate naturally narrow traversals.
pub(crate) fn acquire_traversal_buffers_with_capacity<Q: 'static, P: 'static>(
    queue_capacity: usize,
    path_capacity: usize,
) -> (VecDeque<Q>, Vec<P>) {
    if traversal_buffer_reuse_disabled() {
        return (
            VecDeque::with_capacity(queue_capacity),
            Vec::with_capacity(path_capacity),
        );
    }

    match take_cached_traversal_storage::<ReusableTraversalBuffers<Q, P>>() {
        Some(buffers) => (buffers.queue, buffers.path),
        None => (
            VecDeque::with_capacity(queue_capacity),
            Vec::with_capacity(path_capacity),
        ),
    }
}

/// Return empty exact typed owners to the current thread's bounded warm cache.
///
/// Clearing before erasure drops every live element with its concrete type.
/// Full `TypeId` equality and `Any::downcast` prevent allocation reuse across
/// merely layout-compatible element types.
pub(crate) fn release_traversal_buffers<Q: 'static, P: 'static>(
    mut queue: VecDeque<Q>,
    mut path: Vec<P>,
) {
    queue.clear();
    path.clear();
    if traversal_buffer_reuse_disabled() {
        return;
    }

    let logical_bytes = queue
        .capacity()
        .saturating_mul(size_of::<Q>())
        .saturating_add(path.capacity().saturating_mul(size_of::<P>()));
    retain_traversal_storage(ReusableTraversalBuffers { queue, path }, logical_bytes);
}

/// Acquire an exact-typed queue for cursor-native traversals that need no
/// parent-path storage.
pub(crate) fn acquire_traversal_queue<Q: 'static>() -> VecDeque<Q> {
    if traversal_buffer_reuse_disabled() {
        return new_traversal_queue();
    }
    take_cached_traversal_storage::<ReusableTraversalQueue<Q>>()
        .map_or_else(new_traversal_queue, |buffers| buffers.queue)
}

/// Return an empty cursor-native traversal queue to the bounded thread cache.
pub(crate) fn release_traversal_queue<Q: 'static>(mut queue: VecDeque<Q>) {
    queue.clear();
    if traversal_buffer_reuse_disabled() {
        return;
    }
    let logical_bytes = queue.capacity().saturating_mul(size_of::<Q>());
    retain_traversal_storage(ReusableTraversalQueue { queue }, logical_bytes);
}

/// Acquire distance-bucket queues plus a shared parent-path arena.
pub(crate) fn acquire_bucketed_traversal_buffers<Q: 'static, P: 'static>(
    bucket_count: usize,
) -> (Vec<VecDeque<Q>>, Vec<P>) {
    if traversal_buffer_reuse_disabled() {
        return (
            (0..bucket_count).map(|_| new_traversal_queue()).collect(),
            new_path_arena(),
        );
    }

    let Some(mut buffers) =
        take_cached_traversal_storage::<ReusableBucketedTraversalBuffers<Q, P>>()
    else {
        return (
            (0..bucket_count).map(|_| new_traversal_queue()).collect(),
            new_path_arena(),
        );
    };
    buffers.buckets.truncate(bucket_count);
    buffers
        .buckets
        .resize_with(bucket_count, new_traversal_queue);
    (buffers.buckets, buffers.path)
}

/// Return empty distance buckets and their path owner to the bounded cache.
pub(crate) fn release_bucketed_traversal_buffers<Q: 'static, P: 'static>(
    mut buckets: Vec<VecDeque<Q>>,
    mut path: Vec<P>,
) {
    for bucket in &mut buckets {
        bucket.clear();
    }
    path.clear();
    if traversal_buffer_reuse_disabled() {
        return;
    }
    let logical_bytes = buckets
        .iter()
        .map(|bucket| bucket.capacity().saturating_mul(size_of::<Q>()))
        .fold(0usize, usize::saturating_add)
        .saturating_add(path.capacity().saturating_mul(size_of::<P>()));
    retain_traversal_storage(
        ReusableBucketedTraversalBuffers { buckets, path },
        logical_bytes,
    );
}

/// Acquire exact-typed distance buckets for a cursor-native traversal that
/// does not own parent-path storage.
pub(crate) fn acquire_bucketed_traversal_queues<Q: 'static>(
    bucket_count: usize,
) -> Vec<VecDeque<Q>> {
    if traversal_buffer_reuse_disabled() {
        return (0..bucket_count).map(|_| new_traversal_queue()).collect();
    }

    let Some(mut buffers) = take_cached_traversal_storage::<ReusableBucketedTraversalQueues<Q>>()
    else {
        return (0..bucket_count).map(|_| new_traversal_queue()).collect();
    };
    buffers.buckets.truncate(bucket_count);
    buffers
        .buckets
        .resize_with(bucket_count, new_traversal_queue);
    buffers.buckets
}

/// Return empty cursor-native distance buckets to the bounded thread cache.
pub(crate) fn release_bucketed_traversal_queues<Q: 'static>(mut buckets: Vec<VecDeque<Q>>) {
    for bucket in &mut buckets {
        bucket.clear();
    }
    if traversal_buffer_reuse_disabled() {
        return;
    }
    let logical_bytes = buckets
        .iter()
        .map(|bucket| bucket.capacity().saturating_mul(size_of::<Q>()))
        .fold(0usize, usize::saturating_add);
    retain_traversal_storage(ReusableBucketedTraversalQueues { buckets }, logical_bytes);
}

const NO_RESULT_PATH: u32 = u32::MAX;

/// Stable index into a query-local append-only parent-path arena.
///
/// The distinguished root key represents the empty path and therefore never
/// indexes the backing slice. Copying a key is constant-time; dictionary paths
/// remain logically distinct even when the physical dictionary is a DAG.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct ParentPathKey(u32);

impl ParentPathKey {
    pub(crate) const ROOT: Self = Self(NO_RESULT_PATH);

    #[inline(always)]
    fn index(self) -> Option<usize> {
        (self != Self::ROOT).then_some(self.0 as usize)
    }
}

/// One-word cursor whose active representation is selected once by the
/// containing [`TraversalSession`].
///
/// Owned arenas and compact graphs use a dense index; native backends retain
/// their exact associated cursor (including strict-provenance pointers). The
/// session mode is the discriminant, so no tag is repeated in every hot queue
/// entry.
#[repr(C)]
union TraversalCursorRepr<C: Copy> {
    dense: SnapshotTraversalCursor,
    native: C,
}

impl<C: Copy> Copy for TraversalCursorRepr<C> {}

impl<C: Copy> Clone for TraversalCursorRepr<C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[repr(transparent)]
pub(crate) struct TraversalCursor<C: Copy> {
    repr: TraversalCursorRepr<C>,
}

impl<C: Copy> Copy for TraversalCursor<C> {}

impl<C: Copy> Clone for TraversalCursor<C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: Copy> TraversalCursor<C> {
    #[inline(always)]
    fn dense(cursor: SnapshotTraversalCursor) -> Self {
        Self {
            repr: TraversalCursorRepr { dense: cursor },
        }
    }

    #[inline(always)]
    fn native(cursor: C) -> Self {
        Self {
            repr: TraversalCursorRepr { native: cursor },
        }
    }

    /// Read the dense member after the session mode established it as active.
    #[inline(always)]
    unsafe fn dense_value(self) -> SnapshotTraversalCursor {
        // SAFETY: delegated to the caller's session-mode invariant.
        unsafe { self.repr.dense }
    }

    /// Read the native member after the session mode established it as active.
    #[inline(always)]
    unsafe fn native_value(self) -> C {
        // SAFETY: delegated to the caller's session-mode invariant.
        unsafe { self.repr.native }
    }
}

/// Compact shared parent-chain node for backends that cannot derive a key
/// from their immutable traversal cursor.
pub(crate) struct ParentPathNode<U: CharUnit> {
    label: U,
    depth: u32,
    parent: ParentPathKey,
}

/// Append one logical edge to a shared query-local path arena.
#[inline]
pub(crate) fn push_parent_path<U: CharUnit>(
    arena: &mut Vec<ParentPathNode<U>>,
    parent: ParentPathKey,
    label: U,
) -> ParentPathKey {
    let depth = match parent.index() {
        Some(index) => arena[index]
            .depth
            .checked_add(1)
            .expect("a dictionary path depth fits in u32"),
        None => 1,
    };
    let key = ParentPathKey(
        u32::try_from(arena.len()).expect("a query-local parent-path arena index fits in u32"),
    );
    assert_ne!(
        key,
        ParentPathKey::ROOT,
        "a query-local parent-path arena exhausted its compact key space"
    );
    arena.push(ParentPathNode {
        label,
        depth,
        parent,
    });
    crate::causal_perf::record_path_arena_nodes_created(1);
    key
}

/// Materialize one complete logical path exactly once at a result boundary.
pub(crate) fn materialize_parent_path<U: CharUnit>(
    arena: &[ParentPathNode<U>],
    key: ParentPathKey,
) -> Vec<U> {
    let mut units = Vec::with_capacity(parent_path_depth(arena, key));
    let mut current = key;
    while let Some(index) = current.index() {
        let node = &arena[index];
        units.push(node.label);
        current = node.parent;
    }
    units.reverse();
    crate::causal_perf::record_term_units_materialized(units.len() as u64);
    units
}

/// Compare two canonically shared arena paths with exactly the same semantics
/// as slice lexicographic ordering, without materializing either path.
///
/// The traversal invariant is that a logical parent is expanded once and each
/// of its labels occurs once, so equal prefixes have equal keys.
pub(crate) fn compare_parent_paths<U: CharUnit>(
    arena: &[ParentPathNode<U>],
    left: ParentPathKey,
    right: ParentPathKey,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    if left == right {
        return Ordering::Equal;
    }

    let left_depth = parent_path_depth(arena, left);
    let right_depth = parent_path_depth(arena, right);
    let mut left_aligned = left;
    let mut right_aligned = right;
    let mut left_remaining = left_depth;
    let mut right_remaining = right_depth;

    while left_remaining > right_remaining {
        left_aligned = arena[left_aligned.index().expect("positive depth has a node")].parent;
        left_remaining -= 1;
    }
    while right_remaining > left_remaining {
        right_aligned = arena[right_aligned.index().expect("positive depth has a node")].parent;
        right_remaining -= 1;
    }

    if left_aligned == right_aligned {
        return left_depth.cmp(&right_depth);
    }

    loop {
        let left_index = left_aligned
            .index()
            .expect("distinct non-prefix paths have a left node");
        let right_index = right_aligned
            .index()
            .expect("distinct non-prefix paths have a right node");
        let left_parent = arena[left_index].parent;
        let right_parent = arena[right_index].parent;
        if left_parent == right_parent {
            debug_assert_ne!(arena[left_index].label, arena[right_index].label);
            return arena[left_index].label.cmp(&arena[right_index].label);
        }
        left_aligned = left_parent;
        right_aligned = right_parent;
    }
}

#[inline(always)]
fn parent_path_depth<U: CharUnit>(arena: &[ParentPathNode<U>], key: ParentPathKey) -> usize {
    key.index().map_or(0, |index| arena[index].depth as usize)
}

/// Queue payload used by the compatibility parent-path representation.
pub(crate) struct ParentPathTrace<U: CharUnit, C: Copy> {
    label: Option<U>,
    position: TraversalCursor<C>,
    parent: ParentPathKey,
}

/// Queue payload used when an immutable cursor uniquely identifies its exact
/// key relative to the captured traversal root.
pub(crate) struct CursorPathTrace<C: Copy> {
    position: TraversalCursor<C>,
}

/// Product-state queue entry composed from a backend-specific path trace and
/// a surface-specific automaton frontier.
pub(crate) struct PathFrontier<T, F> {
    pub(crate) trace: T,
    pub(crate) frontier: F,
}

type DfsEdge<N> = (
    <N as DictionaryNode>::Unit,
    TraversalCursor<<N as DictionaryNode>::SnapshotCursor>,
);

type DfsEdgePage<N> = SmallVec<[DfsEdge<N>; DFS_EDGE_PAGE_CAPACITY]>;

/// One DFS node's finality and lazily consumed outgoing edge source.
///
/// Nodes without bounded page addressing remain eager. Compact graphs,
/// capable native cursors, and capable owned-node providers buffer at most one
/// inline page.
pub(crate) struct DfsNodeEdges<N: DictionaryNode> {
    is_final: bool,
    source: DfsEdgeSource<N>,
}

enum DfsEdgeSource<N: DictionaryNode> {
    Eager(std::vec::IntoIter<DfsEdge<N>>),
    Paged {
        position: TraversalCursor<N::SnapshotCursor>,
        /// An owned parent is retained only for the graphless paging seam.
        /// Captured graph/native modes keep their revision owner in the
        /// traversal session and leave this empty.
        owned: Option<N>,
        next: usize,
        total: usize,
        page: smallvec::IntoIter<[DfsEdge<N>; DFS_EDGE_PAGE_CAPACITY]>,
    },
}

impl<N: DictionaryNode> DfsNodeEdges<N> {
    #[inline(always)]
    pub(crate) fn is_final(&self) -> bool {
        self.is_final
    }
}

impl<T, F> PathFrontier<T, F> {
    #[inline(always)]
    pub(crate) fn new(trace: T, frontier: F) -> Self {
        Self { trace, frontier }
    }
}

/// Statically selected result-path representation shared by query surfaces.
///
/// Each iterator selects one strategy when it captures the dictionary. The
/// iterator core is then monomorphized for that strategy, so path operations
/// introduce neither a vtable nor a representation branch in the dictionary
/// walk.
pub(crate) trait ResultPathStrategy<N>: 'static
where
    N: DictionaryNode,
{
    type Trace: 'static;
    type Storage: Default + 'static;
    type Expansion;

    fn acquire_queue<Q: 'static>() -> (VecDeque<Q>, Self::Storage);
    fn release_queue<Q: 'static>(pending: VecDeque<Q>, storage: Self::Storage);
    fn cold_queue<Q>() -> (VecDeque<Q>, Self::Storage);
    fn acquire_buckets<Q: 'static>(bucket_count: usize) -> (Vec<VecDeque<Q>>, Self::Storage);
    fn release_buckets<Q: 'static>(pending: Vec<VecDeque<Q>>, storage: Self::Storage);
    fn cold_buckets<Q>(bucket_count: usize) -> (Vec<VecDeque<Q>>, Self::Storage);

    fn root(position: TraversalCursor<N::SnapshotCursor>) -> Self::Trace;
    fn position(trace: &Self::Trace) -> TraversalCursor<N::SnapshotCursor>;
    fn begin_expansion(trace: &Self::Trace) -> Self::Expansion;
    fn child_trace(
        trace: &Self::Trace,
        expansion: &mut Self::Expansion,
        label: N::Unit,
        position: TraversalCursor<N::SnapshotCursor>,
        storage: &mut Self::Storage,
    ) -> Self::Trace;
    fn materialize_units(
        trace: &Self::Trace,
        traversal: &TraversalSession<N>,
        storage: &Self::Storage,
    ) -> Vec<N::Unit>;
}

pub(crate) struct ParentArenaPath;

impl<N> ResultPathStrategy<N> for ParentArenaPath
where
    N: DictionaryNode,
{
    type Trace = ParentPathTrace<N::Unit, N::SnapshotCursor>;
    type Storage = Vec<ParentPathNode<N::Unit>>;
    type Expansion = Option<ParentPathKey>;

    #[inline]
    fn acquire_queue<Q: 'static>() -> (VecDeque<Q>, Self::Storage) {
        crate::causal_perf::record_parent_path_queries(1);
        acquire_traversal_buffers()
    }

    #[inline]
    fn release_queue<Q: 'static>(pending: VecDeque<Q>, storage: Self::Storage) {
        release_traversal_buffers(pending, storage);
    }

    #[inline]
    fn cold_queue<Q>() -> (VecDeque<Q>, Self::Storage) {
        (new_traversal_queue(), new_path_arena())
    }

    #[inline]
    fn acquire_buckets<Q: 'static>(bucket_count: usize) -> (Vec<VecDeque<Q>>, Self::Storage) {
        crate::causal_perf::record_parent_path_queries(1);
        acquire_bucketed_traversal_buffers(bucket_count)
    }

    #[inline]
    fn release_buckets<Q: 'static>(pending: Vec<VecDeque<Q>>, storage: Self::Storage) {
        release_bucketed_traversal_buffers(pending, storage);
    }

    #[inline]
    fn cold_buckets<Q>(bucket_count: usize) -> (Vec<VecDeque<Q>>, Self::Storage) {
        (
            (0..bucket_count).map(|_| new_traversal_queue()).collect(),
            new_path_arena(),
        )
    }

    #[inline(always)]
    fn root(position: TraversalCursor<N::SnapshotCursor>) -> Self::Trace {
        ParentPathTrace {
            label: None,
            position,
            parent: ParentPathKey::ROOT,
        }
    }

    #[inline(always)]
    fn position(trace: &Self::Trace) -> TraversalCursor<N::SnapshotCursor> {
        trace.position
    }

    #[inline(always)]
    fn begin_expansion(_trace: &Self::Trace) -> Self::Expansion {
        None
    }

    #[inline(always)]
    fn child_trace(
        trace: &Self::Trace,
        expansion: &mut Self::Expansion,
        label: N::Unit,
        position: TraversalCursor<N::SnapshotCursor>,
        storage: &mut Self::Storage,
    ) -> Self::Trace {
        let parent_path = match *expansion {
            Some(path) => path,
            None => {
                let path = match trace.label {
                    Some(parent_label) => push_parent_path(storage, trace.parent, parent_label),
                    None => ParentPathKey::ROOT,
                };
                *expansion = Some(path);
                path
            }
        };
        ParentPathTrace {
            label: Some(label),
            position,
            parent: parent_path,
        }
    }

    #[inline]
    fn materialize_units(
        trace: &Self::Trace,
        _traversal: &TraversalSession<N>,
        storage: &Self::Storage,
    ) -> Vec<N::Unit> {
        match trace.label {
            Some(label) => {
                let mut units = materialize_parent_path(storage, trace.parent);
                units.push(label);
                crate::causal_perf::record_term_units_materialized(1);
                units
            }
            None => Vec::new(),
        }
    }
}

pub(crate) struct CursorNativePath;

impl<N> ResultPathStrategy<N> for CursorNativePath
where
    N: DictionaryNode,
{
    type Trace = CursorPathTrace<N::SnapshotCursor>;
    type Storage = ();
    type Expansion = ();

    #[inline]
    fn acquire_queue<Q: 'static>() -> (VecDeque<Q>, Self::Storage) {
        crate::causal_perf::record_cursor_key_queries(1);
        (acquire_traversal_queue(), ())
    }

    #[inline]
    fn release_queue<Q: 'static>(pending: VecDeque<Q>, _storage: Self::Storage) {
        release_traversal_queue(pending);
    }

    #[inline]
    fn cold_queue<Q>() -> (VecDeque<Q>, Self::Storage) {
        (new_traversal_queue(), ())
    }

    #[inline]
    fn acquire_buckets<Q: 'static>(bucket_count: usize) -> (Vec<VecDeque<Q>>, Self::Storage) {
        crate::causal_perf::record_cursor_key_queries(1);
        (acquire_bucketed_traversal_queues(bucket_count), ())
    }

    #[inline]
    fn release_buckets<Q: 'static>(pending: Vec<VecDeque<Q>>, _storage: Self::Storage) {
        release_bucketed_traversal_queues(pending);
    }

    #[inline]
    fn cold_buckets<Q>(bucket_count: usize) -> (Vec<VecDeque<Q>>, Self::Storage) {
        (
            (0..bucket_count).map(|_| new_traversal_queue()).collect(),
            (),
        )
    }

    #[inline(always)]
    fn root(position: TraversalCursor<N::SnapshotCursor>) -> Self::Trace {
        CursorPathTrace { position }
    }

    #[inline(always)]
    fn position(trace: &Self::Trace) -> TraversalCursor<N::SnapshotCursor> {
        trace.position
    }

    #[inline(always)]
    fn begin_expansion(_trace: &Self::Trace) -> Self::Expansion {}

    #[inline(always)]
    fn child_trace(
        _trace: &Self::Trace,
        _expansion: &mut Self::Expansion,
        _label: N::Unit,
        position: TraversalCursor<N::SnapshotCursor>,
        _storage: &mut Self::Storage,
    ) -> Self::Trace {
        CursorPathTrace { position }
    }

    #[inline]
    fn materialize_units(
        trace: &Self::Trace,
        traversal: &TraversalSession<N>,
        _storage: &Self::Storage,
    ) -> Vec<N::Unit> {
        let units = traversal
            .cursor_key_units(trace.position)
            .expect("cursor-native path strategy requires exact key reconstruction");
        crate::causal_perf::record_cursor_key_reconstructions(1);
        crate::causal_perf::record_cursor_key_reverse_steps(units.len() as u64);
        crate::causal_perf::record_term_units_materialized(units.len() as u64);
        units
    }
}

fn take_cached_traversal_storage<B: 'static>() -> Option<B> {
    let type_id = TypeId::of::<B>();
    TRAVERSAL_BUFFER_CACHE
        .try_with(|cache| {
            let mut cache = cache.try_borrow_mut().ok()?;
            let index = cache
                .entries
                .iter()
                .position(|entry| entry.type_id == type_id)?;
            let entry = cache
                .entries
                .remove(index)
                .expect("a located traversal-buffer entry exists");
            cache.logical_bytes = cache.logical_bytes.saturating_sub(entry.logical_bytes);
            entry.buffers.downcast::<B>().ok().map(|buffers| *buffers)
        })
        .ok()
        .flatten()
}

fn retain_traversal_storage<B: 'static>(buffers: B, logical_bytes: usize) {
    if logical_bytes > MAX_RETAINED_TRAVERSAL_BUFFER_BYTES {
        return;
    }
    let type_id = TypeId::of::<B>();
    let _ = TRAVERSAL_BUFFER_CACHE.try_with(|cache| {
        let Ok(mut cache) = cache.try_borrow_mut() else {
            return;
        };

        if let Some(index) = cache
            .entries
            .iter()
            .position(|entry| entry.type_id == type_id)
        {
            if cache.entries[index].logical_bytes >= logical_bytes {
                return;
            }
            let previous = cache
                .entries
                .remove(index)
                .expect("a located traversal-buffer entry exists");
            cache.logical_bytes = cache.logical_bytes.saturating_sub(previous.logical_bytes);
        }

        while cache.entries.len() >= MAX_RETAINED_TRAVERSAL_BUFFER_TYPES
            || cache.logical_bytes.saturating_add(logical_bytes)
                > MAX_RETAINED_TRAVERSAL_BUFFER_BYTES
        {
            let Some(evicted) = cache.entries.pop_front() else {
                break;
            };
            cache.logical_bytes = cache.logical_bytes.saturating_sub(evicted.logical_bytes);
        }

        cache.logical_bytes = cache.logical_bytes.saturating_add(logical_bytes);
        cache.entries.push_back(ErasedTraversalBuffers {
            type_id,
            logical_bytes,
            buffers: Box::new(buffers),
        });
    });
}

#[inline]
fn traversal_buffer_reuse_disabled() -> bool {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_TRAVERSAL_BUFFER_REUSE").is_some()
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        false
    }
}

struct OwnedTraversalArena<N> {
    nodes: Vec<Option<N>>,
    free: Vec<usize>,
}

impl<N> OwnedTraversalArena<N> {
    fn new(root: N) -> Self {
        Self {
            nodes: vec![Some(root)],
            free: Vec::new(),
        }
    }

    #[inline]
    fn take(&mut self, cursor: SnapshotTraversalCursor) -> N {
        let index = cursor.get() - 1;
        let node = self.nodes[index]
            .take()
            .expect("an owned traversal cursor is expanded exactly once");
        self.free.push(index);
        node
    }

    #[inline]
    fn insert(&mut self, node: N) -> SnapshotTraversalCursor {
        crate::causal_perf::record_owned_traversal_arena_insertions(1);
        let index = match self.free.pop() {
            Some(index) => {
                debug_assert!(self.nodes[index].is_none());
                self.nodes[index] = Some(node);
                index
            }
            None => {
                self.nodes.push(Some(node));
                self.nodes.len() - 1
            }
        };
        SnapshotTraversalCursor::new(index + 1).expect("an owned traversal arena index is non-zero")
    }
}

enum TraversalMode<N: DictionaryNode> {
    /// Compatibility arena. Each queued cursor owns exactly one node slot; an
    /// expanded node is taken from its slot and dropped after its children have
    /// been appended, so no clone is needed merely to satisfy borrow rules.
    Owned(OwnedTraversalArena<N>),
    /// Compact topology uses dense graph positions while values retain their
    /// backend-native graph handles behind the owner boundary.
    Graph {
        graph: Arc<SnapshotTraversalGraph<N::Unit, N::SnapshotGraphValueHandle>>,
        owner: N,
    },
    /// Backend-native zero-copy traversal (for example a retained DynamicDawg
    /// pointer cursor or a DAT dense state cursor).
    Native { owner: N },
}

/// Query-start dictionary revision whose queued locations are always one
/// untagged cursor selected once per session. Built-in backends keep this to
/// one machine word; external backends pay only for their native cursor width,
/// never for a repeated per-entry mode tag.
pub(crate) struct TraversalSession<N: DictionaryNode> {
    mode: TraversalMode<N>,
    /// Owned traversal consumes node handles as they are expanded. Retain one
    /// root policy handle only for semantic decorators whose terminal
    /// visibility depends on the complete root-relative key.
    owned_final_units_owner: Option<N>,
}

/// Deferred owned-node source produced only for a final dictionary node.
///
/// Owned fallbacks retain that one expanded node until the caller establishes
/// that its automaton state is in range. Captured backends retain only a copy
/// cursor and materialize an owned handle lazily through the session owner.
pub(crate) enum DeferredNodeSource<N: DictionaryNode> {
    Owned(N),
    Captured(TraversalCursor<N::SnapshotCursor>),
}

pub(crate) type MappedValueSource<N> = DeferredNodeSource<N>;

impl<N: DictionaryNode> TraversalSession<N> {
    /// Capture a traversal root and return its uniform root cursor.
    pub(crate) fn capture(
        root: DictionaryTraversalRoot<N>,
    ) -> (Self, TraversalCursor<N::SnapshotCursor>) {
        let (graph, owner) = root.into_parts().into_projection_and_root();
        if cursor_traversal_disabled() {
            return Self::owned(owner);
        }

        if let Some(graph) = graph {
            let root = graph.root_cursor();
            return (
                Self {
                    mode: TraversalMode::Graph { owner, graph },
                    owned_final_units_owner: None,
                },
                TraversalCursor::dense(root),
            );
        }

        if !owner.snapshot_cursor_requires_full_projection() {
            if let Some(root) = owner.snapshot_root_cursor() {
                return (
                    Self {
                        mode: TraversalMode::Native { owner },
                        owned_final_units_owner: None,
                    },
                    TraversalCursor::native(root),
                );
            }
        }

        Self::owned(owner)
    }

    fn owned(root: N) -> (Self, TraversalCursor<N::SnapshotCursor>) {
        let root_cursor = SnapshotTraversalCursor::new(1).expect("one is non-zero");
        let owned_final_units_owner = root.requires_final_units().then(|| root.clone());
        (
            Self {
                mode: TraversalMode::Owned(OwnedTraversalArena::new(root)),
                owned_final_units_owner,
            },
            TraversalCursor::dense(root_cursor),
        )
    }

    /// Capture a revision whose accepting cursors may need to escape later as
    /// owned node handles. Value-less flat graphs are intentionally ignored.
    pub(crate) fn capture_nodes(
        root: DictionaryTraversalRoot<N>,
    ) -> (Self, TraversalCursor<N::SnapshotCursor>) {
        let (_, owner) = root.into_parts().into_projection_and_root();
        if !cursor_traversal_disabled()
            && !owner.snapshot_cursor_requires_full_projection()
            && owner.supports_snapshot_cursor_nodes()
        {
            if let Some(cursor) = owner.snapshot_root_cursor() {
                return (
                    Self {
                        mode: TraversalMode::Native { owner },
                        owned_final_units_owner: None,
                    },
                    TraversalCursor::native(cursor),
                );
            }
        }
        Self::owned(owner)
    }

    /// Inspect finality and open an outgoing-edge source tailored to the
    /// captured backend. Paged modes do not fetch an edge until
    /// [`next_dfs_edge`](Self::next_dfs_edge) is called.
    pub(crate) fn open_dfs_node(
        &mut self,
        cursor: TraversalCursor<N::SnapshotCursor>,
    ) -> DfsNodeEdges<N> {
        if dfs_edge_paging_disabled() {
            return self.open_eager_dfs_node(cursor);
        }

        match &mut self.mode {
            TraversalMode::Owned(arena) => {
                // SAFETY: owned sessions construct only dense cursors. The
                // expanded node moves into the DFS frame when it can page, so
                // its captured revision and values outlive every page.
                let node = arena.take(unsafe { cursor.dense_value() });
                if node.supports_efficient_edge_paging() {
                    let (is_final, total) = node.visit_edge_page_and_finality(0, 0, |_, _| {
                        unreachable!("a zero-capacity page has no edge")
                    });
                    crate::causal_perf::record_dfs_nodes_paged(1);
                    DfsNodeEdges {
                        is_final,
                        source: DfsEdgeSource::Paged {
                            position: cursor,
                            owned: Some(node),
                            next: 0,
                            total,
                            page: DfsEdgePage::<N>::new().into_iter(),
                        },
                    }
                } else {
                    let mut buffered = Vec::new();
                    let is_final = node.filter_map_edges_and_finality(
                        |_| Some(()),
                        |label, child, ()| {
                            let child = arena.insert(child);
                            buffered.push((label, TraversalCursor::dense(child)));
                        },
                    );
                    Self::eager_dfs_node(is_final, buffered)
                }
            }
            TraversalMode::Graph { owner: _, graph } => {
                // SAFETY: graph-mode cursors always originate from this exact
                // immutable captured graph.
                let edges = unsafe { graph.edges_and_finality_unchecked(cursor.dense_value()) };
                crate::causal_perf::record_dfs_nodes_paged(1);
                DfsNodeEdges {
                    is_final: edges.is_final(),
                    source: DfsEdgeSource::Paged {
                        position: cursor,
                        owned: None,
                        next: 0,
                        total: edges.edges().len(),
                        page: DfsEdgePage::<N>::new().into_iter(),
                    },
                }
            }
            TraversalMode::Native { owner }
                if owner.supports_efficient_snapshot_cursor_edge_paging() =>
            {
                // A zero-capacity request obtains stable metadata without
                // speculatively constructing a child cursor.
                let (is_final, total) = unsafe {
                    owner
                        .visit_snapshot_cursor_edge_page(cursor.native_value(), 0, 0, |_, _| {
                            unreachable!("a zero-capacity page has no edge")
                        })
                        .expect("an advertised native pager supports cursor paging")
                };
                crate::causal_perf::record_dfs_nodes_paged(1);
                DfsNodeEdges {
                    is_final,
                    source: DfsEdgeSource::Paged {
                        position: cursor,
                        owned: None,
                        next: 0,
                        total,
                        page: DfsEdgePage::<N>::new().into_iter(),
                    },
                }
            }
            TraversalMode::Native { .. } => self.open_eager_dfs_node(cursor),
        }
    }

    #[inline]
    fn eager_dfs_node(is_final: bool, edges: Vec<DfsEdge<N>>) -> DfsNodeEdges<N> {
        crate::causal_perf::record_dfs_nodes_eager(1);
        crate::causal_perf::record_dfs_edges_fetched(edges.len() as u64);
        crate::causal_perf::record_dfs_edge_buffer_size(edges.len());
        DfsNodeEdges {
            is_final,
            source: DfsEdgeSource::Eager(edges.into_iter()),
        }
    }

    pub(crate) fn open_eager_dfs_node(
        &mut self,
        cursor: TraversalCursor<N::SnapshotCursor>,
    ) -> DfsNodeEdges<N> {
        let mut edges = Vec::new();
        let is_final = self.filter_map_edges_and_finality(
            cursor,
            |_| Some(()),
            |label, child, ()| edges.push((label, child)),
        );
        Self::eager_dfs_node(is_final, edges)
    }

    /// Consume the next outgoing edge, refilling one bounded inline page only
    /// when the previous page is exhausted.
    pub(crate) fn next_dfs_edge(&mut self, edges: &mut DfsNodeEdges<N>) -> Option<DfsEdge<N>> {
        match &mut edges.source {
            DfsEdgeSource::Eager(edges) => {
                let edge = edges.next();
                if edge.is_some() {
                    crate::causal_perf::record_dfs_edges_consumed(1);
                }
                edge
            }
            DfsEdgeSource::Paged {
                position,
                owned,
                next,
                total,
                page,
            } => {
                if let Some(edge) = page.next() {
                    crate::causal_perf::record_dfs_edges_consumed(1);
                    return Some(edge);
                }
                if *next >= *total {
                    return None;
                }

                let start = *next;
                let mut refill = DfsEdgePage::<N>::new();
                let page_capacity = dfs_edge_page_capacity();
                crate::causal_perf::record_dfs_edge_page_requests(1);
                match (owned.as_ref(), &mut self.mode) {
                    (Some(node), TraversalMode::Owned(arena)) => {
                        let confirmed_total =
                            node.visit_edge_page(start, page_capacity, |label, child| {
                                let child = arena.insert(child);
                                refill.push((label, TraversalCursor::dense(child)));
                            });
                        assert_eq!(confirmed_total, *total, "captured edge count changed");
                    }
                    (None, TraversalMode::Graph { owner: _, graph }) => {
                        // SAFETY: the frame retains a cursor from this exact
                        // immutable graph-mode session.
                        let node =
                            unsafe { graph.edges_and_finality_unchecked(position.dense_value()) };
                        debug_assert_eq!(node.is_final(), edges.is_final);
                        debug_assert_eq!(node.edges().len(), *total);
                        refill.extend(node.edges().iter().skip(start).take(page_capacity).map(
                            |edge| (edge.label(), TraversalCursor::dense(edge.target_cursor())),
                        ));
                    }
                    (None, TraversalMode::Native { owner }) => {
                        let (is_final, confirmed_total) = unsafe {
                            owner
                                .visit_snapshot_cursor_edge_page(
                                    position.native_value(),
                                    start,
                                    page_capacity,
                                    |label, child| {
                                        refill.push((label, TraversalCursor::native(child)))
                                    },
                                )
                                .expect("an opened native pager remains supported")
                        };
                        assert_eq!(is_final, edges.is_final, "captured finality changed");
                        assert_eq!(confirmed_total, *total, "captured edge count changed");
                    }
                    (Some(_), TraversalMode::Graph { .. } | TraversalMode::Native { .. })
                    | (None, TraversalMode::Owned(_)) => {
                        unreachable!("DFS page origin and traversal mode remain paired")
                    }
                }
                assert!(
                    !refill.is_empty(),
                    "a non-empty edge remainder yielded no page"
                );
                crate::causal_perf::record_dfs_edges_fetched(refill.len() as u64);
                crate::causal_perf::record_dfs_edge_buffer_size(refill.len());
                if refill.spilled() {
                    crate::causal_perf::record_dfs_edge_buffer_spills(1);
                }
                *next = next
                    .checked_add(refill.len())
                    .expect("a dictionary edge index fits in usize");
                *page = refill.into_iter();
                let edge = page.next();
                debug_assert!(edge.is_some());
                crate::causal_perf::record_dfs_edges_consumed(1);
                edge
            }
        }
    }

    /// Read finality and project outgoing edges without changing the queued
    /// cursor representation.
    #[inline(always)]
    pub(crate) fn filter_map_edges_and_finality<T, P, F>(
        &mut self,
        cursor: TraversalCursor<N::SnapshotCursor>,
        project: P,
        mut visitor: F,
    ) -> bool
    where
        P: FnMut(N::Unit) -> Option<T>,
        F: FnMut(N::Unit, TraversalCursor<N::SnapshotCursor>, T),
    {
        match &mut self.mode {
            TraversalMode::Owned(arena) => {
                // SAFETY: owned sessions construct only dense cursors.
                let node = arena.take(unsafe { cursor.dense_value() });
                let mut visitor = visitor;
                node.filter_map_edges_and_finality(project, |label, child, value| {
                    let child_cursor = arena.insert(child);
                    visitor(label, TraversalCursor::dense(child_cursor), value);
                })
            }
            TraversalMode::Graph { owner: _, graph } => {
                // SAFETY: captured cursors are the validated root or targets
                // read from this exact immutable graph.
                let edges = unsafe { graph.edges_and_finality_unchecked(cursor.dense_value()) };
                let is_final = edges.is_final();
                let mut project = project;
                let mut visitor = visitor;
                for edge in edges.edges() {
                    let label = edge.label();
                    if let Some(value) = project(label) {
                        visitor(label, TraversalCursor::dense(edge.target_cursor()), value);
                    }
                }
                is_final
            }
            TraversalMode::Native { owner } => {
                // SAFETY: `capture` obtains the root cursor from `owner`; all
                // descendants are produced by this same retained revision and
                // cursors never escape the session.
                unsafe {
                    owner
                        .filter_map_snapshot_cursor_edges_and_finality(
                            cursor.native_value(),
                            project,
                            |label, child, value| {
                                visitor(label, TraversalCursor::native(child), value)
                            },
                        )
                        .expect(
                            "a backend that returns a root cursor must support cursor traversal",
                        )
                }
            }
        }
    }

    /// Expand one node and retain its owned representation only when final.
    #[inline(always)]
    pub(crate) fn filter_map_edges_and_final_source<T, P, F>(
        &mut self,
        cursor: TraversalCursor<N::SnapshotCursor>,
        project: P,
        mut visitor: F,
    ) -> Option<DeferredNodeSource<N>>
    where
        P: FnMut(N::Unit) -> Option<T>,
        F: FnMut(N::Unit, TraversalCursor<N::SnapshotCursor>, T),
    {
        match &mut self.mode {
            TraversalMode::Owned(arena) => {
                // SAFETY: owned sessions construct only dense cursors.
                let node = arena.take(unsafe { cursor.dense_value() });
                let mut visitor = visitor;
                let is_final =
                    node.filter_map_edges_and_finality(project, |label, child, value| {
                        let child_cursor = arena.insert(child);
                        visitor(label, TraversalCursor::dense(child_cursor), value);
                    });
                is_final.then_some(DeferredNodeSource::Owned(node))
            }
            TraversalMode::Native { owner } => {
                // SAFETY: a captured session retains the exact immutable owner
                // from which this cursor and all descendants originated.
                let is_final = unsafe {
                    owner
                        .filter_map_snapshot_cursor_edges_and_finality(
                            cursor.native_value(),
                            project,
                            |label, child, value| {
                                visitor(label, TraversalCursor::native(child), value)
                            },
                        )
                        .expect("a native cursor backend supports cursor traversal")
                };
                is_final.then_some(DeferredNodeSource::Captured(cursor))
            }
            TraversalMode::Graph { owner: _, graph } => {
                // SAFETY: captured cursors are the validated root or targets
                // read from this exact immutable graph.
                let edges = unsafe { graph.edges_and_finality_unchecked(cursor.dense_value()) };
                let is_final = edges.is_final();
                let mut project = project;
                let mut visitor = visitor;
                for edge in edges.edges() {
                    let label = edge.label();
                    if let Some(value) = project(label) {
                        visitor(label, TraversalCursor::dense(edge.target_cursor()), value);
                    }
                }
                is_final.then_some(DeferredNodeSource::Captured(cursor))
            }
        }
    }

    /// Resolve a deferred accepting node into an owned handle.
    #[inline]
    pub(crate) fn resolve_node(&self, source: DeferredNodeSource<N>) -> N {
        match source {
            DeferredNodeSource::Owned(node) => node,
            DeferredNodeSource::Captured(cursor) => match &self.mode {
                TraversalMode::Native { owner } => {
                    // SAFETY: the source was created by this exact session and
                    // the retained owner still covers the cursor allocation.
                    unsafe { owner.snapshot_cursor_node(cursor.native_value()) }
                        .expect("capture_nodes validated cursor node materialization")
                }
                TraversalMode::Graph { .. } => {
                    unreachable!("flat graph captures cannot materialize owned nodes")
                }
                TraversalMode::Owned(_) => {
                    unreachable!("captured sources do not belong to owned sessions")
                }
            },
        }
    }

    /// Release one queued owned fallback cursor that will never be expanded.
    #[inline]
    pub(crate) fn discard_unexpanded(&mut self, cursor: TraversalCursor<N::SnapshotCursor>) {
        if let TraversalMode::Owned(arena) = &mut self.mode {
            // SAFETY: owned sessions construct only dense cursors.
            drop(arena.take(unsafe { cursor.dense_value() }));
        }
    }

    /// Whether this captured dictionary view needs the accepted root-to-node
    /// units to decide terminal visibility or mapped-value semantics.
    #[inline]
    pub(crate) fn requires_final_units(&self) -> bool {
        match &self.mode {
            TraversalMode::Graph { owner, .. } | TraversalMode::Native { owner } => {
                owner.requires_final_units()
            }
            TraversalMode::Owned(_) => self.owned_final_units_owner.is_some(),
        }
    }

    /// Apply a semantic decorator's root-relative terminal visibility policy.
    #[inline]
    pub(crate) fn accepts_final_units(&self, units: &[N::Unit]) -> bool {
        match &self.mode {
            TraversalMode::Graph { owner, .. } | TraversalMode::Native { owner } => {
                owner.accepts_final_units(units)
            }
            TraversalMode::Owned(_) => self
                .owned_final_units_owner
                .as_ref()
                .is_none_or(|owner| owner.accepts_final_units(units)),
        }
    }

    /// Whether this captured backend can derive exact root-relative key units
    /// from a cursor without a query-owned parent path.
    #[inline]
    pub(crate) fn supports_cursor_key_units(&self) -> bool {
        if cursor_key_reconstruction_disabled() {
            return false;
        }
        match &self.mode {
            TraversalMode::Native { owner } => owner.supports_snapshot_cursor_key_units(),
            TraversalMode::Owned(_) | TraversalMode::Graph { .. } => false,
        }
    }

    /// Materialize exact units from a cursor whose retained owner advertised
    /// [`DictionaryNode::supports_snapshot_cursor_key_units`].
    #[inline]
    pub(crate) fn cursor_key_units(
        &self,
        cursor: TraversalCursor<N::SnapshotCursor>,
    ) -> Option<Vec<N::Unit>> {
        match &self.mode {
            TraversalMode::Native { owner } if owner.supports_snapshot_cursor_key_units() => {
                // SAFETY: the session owns the exact immutable revision and
                // every queued cursor descends from its captured root.
                unsafe { owner.snapshot_cursor_key_units(cursor.native_value()) }
            }
            TraversalMode::Owned(_)
            | TraversalMode::Graph { .. }
            | TraversalMode::Native { .. } => None,
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn product_identity(
        &self,
        cursor: TraversalCursor<N::SnapshotCursor>,
    ) -> Option<TraversalProductIdentity> {
        match &self.mode {
            TraversalMode::Owned(arena) => arena.nodes[unsafe { cursor.dense_value() }.get() - 1]
                .as_ref()
                .and_then(DictionaryNode::snapshot_node_identity)
                .map(TraversalProductIdentity::Node),
            TraversalMode::Graph { .. } => Some(TraversalProductIdentity::Cursor(unsafe {
                cursor.dense_value()
            })),
            TraversalMode::Native { .. } => None,
        }
    }

    #[cfg(test)]
    fn is_flat(&self) -> bool {
        matches!(self.mode, TraversalMode::Graph { .. })
    }

    #[cfg(test)]
    fn owned_slot_count(&self) -> Option<usize> {
        match &self.mode {
            TraversalMode::Owned(arena) => Some(arena.nodes.len()),
            TraversalMode::Graph { .. } | TraversalMode::Native { .. } => None,
        }
    }

    #[cfg(test)]
    fn owned_live_slot_count(&self) -> Option<usize> {
        match &self.mode {
            TraversalMode::Owned(arena) => {
                Some(arena.nodes.iter().filter(|node| node.is_some()).count())
            }
            TraversalMode::Graph { .. } | TraversalMode::Native { .. } => None,
        }
    }
}

#[inline]
fn cursor_key_reconstruction_disabled() -> bool {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_CURSOR_KEY_RECONSTRUCTION").is_some()
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        false
    }
}

impl<N: MappedDictionaryNode> TraversalSession<N> {
    /// Capture a mapped revision, preferring its compact immutable graph when
    /// the retained owner can resolve graph-local value cursors.
    pub(crate) fn capture_mapped(
        root: DictionaryTraversalRoot<N>,
    ) -> (Self, TraversalCursor<N::SnapshotCursor>) {
        let (graph, owner) = root.into_parts().into_projection_and_root();
        if !cursor_traversal_disabled() && owner.supports_snapshot_graph_values() {
            if let Some(graph) = graph {
                let root = graph.root_cursor();
                return (
                    Self {
                        mode: TraversalMode::Graph { owner, graph },
                        owned_final_units_owner: None,
                    },
                    TraversalCursor::dense(root),
                );
            }
        }
        if !cursor_traversal_disabled()
            && !owner.snapshot_cursor_requires_full_projection()
            && owner.supports_snapshot_cursor_values()
        {
            if let Some(cursor) = owner.snapshot_root_cursor() {
                return (
                    Self {
                        mode: TraversalMode::Native { owner },
                        owned_final_units_owner: None,
                    },
                    TraversalCursor::native(cursor),
                );
            }
        }
        Self::owned(owner)
    }

    /// Resolve a final node's value after the automaton has accepted it.
    #[inline]
    pub(crate) fn resolve_final_value(
        &self,
        source: MappedValueSource<N>,
        units: Option<&[N::Unit]>,
    ) -> Option<N::Value> {
        debug_assert!(
            !self.requires_final_units() || units.is_some(),
            "semantic dictionary wrappers require root-relative terminal units"
        );
        match source {
            MappedValueSource::Owned(node) => match units {
                Some(units) => node.value_at_final_with_units(units),
                None => node.value_at_final(),
            },
            MappedValueSource::Captured(cursor) => match &self.mode {
                TraversalMode::Native { owner } => match units {
                    Some(units) => {
                        // SAFETY: the source and exact root-relative units were
                        // produced by this retained traversal session.
                        unsafe {
                            owner.snapshot_cursor_value_with_units(cursor.native_value(), units)
                        }
                        .expect("capture_mapped validated cursor value access")
                    }
                    None => {
                        // SAFETY: the source was created by this exact session and
                        // has not escaped the retained revision.
                        unsafe { owner.snapshot_cursor_value(cursor.native_value()) }
                            .expect("capture_mapped validated cursor value access")
                    }
                },
                TraversalMode::Graph { owner, graph } => match units {
                    Some(units) => {
                        // SAFETY: graph, cursor, owner, and exact root-relative
                        // units were produced by this retained session.
                        unsafe {
                            owner.snapshot_graph_cursor_value_with_units(
                                graph,
                                cursor.dense_value(),
                                units,
                            )
                        }
                        .expect("capture_mapped validated graph value access")
                    }
                    None => {
                        // SAFETY: the graph and owner were captured together and
                        // the source cursor was produced by this exact graph.
                        unsafe { owner.snapshot_graph_cursor_value(graph, cursor.dense_value()) }
                            .expect("capture_mapped validated graph value access")
                    }
                },
                _ => unreachable!("captured value sources belong to a native mapped session"),
            },
        }
    }
}

#[cfg(feature = "perf-instrumentation")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TraversalProductIdentity {
    Node(SnapshotNodeIdentity),
    Cursor(SnapshotTraversalCursor),
}

#[inline]
fn cursor_traversal_disabled() -> bool {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_SNAPSHOT_CURSORS").is_some()
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        false
    }
}

#[inline]
fn dfs_edge_paging_disabled() -> bool {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_DFS_EDGE_PAGING").is_some()
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        false
    }
}

#[inline]
fn dfs_edge_page_capacity() -> usize {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static CAPACITY: OnceLock<usize> = OnceLock::new();
        *CAPACITY.get_or_init(|| {
            let Some(value) = std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DFS_EDGE_PAGE_CAPACITY")
            else {
                return DFS_EDGE_PAGE_CAPACITY;
            };
            let capacity = value
                .to_str()
                .and_then(|value| value.parse().ok())
                .expect("DFS edge page capacity must be an integer");
            assert!(
                matches!(capacity, 1 | 4 | 8 | 16 | 32),
                "DFS edge page capacity must be 1, 4, 8, 16, or 32"
            );
            capacity
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        DFS_EDGE_PAGE_CAPACITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::dynamic_dawg::char::{DynamicDawgChar, DynamicDawgCharNode};
    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::{PathMapDictionary, PathMapDictionaryChar};
    use libdictenstein::suffix_automaton::{SuffixAutomaton, SuffixAutomatonChar};
    use libdictenstein::{CharUnit, Dictionary, MappedDictionaryNode};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parent_path_metadata_uses_compact_checked_handles() {
        assert_eq!(size_of::<ParentPathKey>(), size_of::<u32>());
        assert_eq!(size_of::<ParentPathNode<char>>(), 12);
        assert_eq!(size_of::<ParentPathNode<u8>>(), 12);
        assert_eq!(size_of::<ParentPathNode<u64>>(), 16);
        assert_eq!(
            size_of::<ParentPathTrace<char, SnapshotTraversalCursor>>(),
            16
        );
    }

    #[test]
    fn parent_path_comparison_matches_slice_lexicographic_order() {
        let paths: &[&[u8]] = &[
            b"",
            b"a",
            b"aa",
            b"ab",
            b"aba",
            b"abb",
            b"b",
            b"ba",
            b"long-shared-prefix-a",
            b"long-shared-prefix-b",
        ];
        let mut arena = Vec::new();
        let mut canonical = std::collections::HashMap::from([(Vec::new(), ParentPathKey::ROOT)]);
        let keys: Vec<_> = paths
            .iter()
            .map(|path| {
                let mut prefix = Vec::new();
                let mut parent = ParentPathKey::ROOT;
                for label in path.iter().copied() {
                    prefix.push(label);
                    parent = *canonical
                        .entry(prefix.clone())
                        .or_insert_with(|| push_parent_path(&mut arena, parent, label));
                }
                parent
            })
            .collect();

        for (left_index, left) in paths.iter().enumerate() {
            assert_eq!(materialize_parent_path(&arena, keys[left_index]), *left);
            for (right_index, right) in paths.iter().enumerate() {
                assert_eq!(
                    compare_parent_paths(&arena, keys[left_index], keys[right_index]),
                    left.cmp(right),
                    "left={left:?}, right={right:?}"
                );
            }
        }
    }

    #[test]
    fn context_heap_matches_standard_max_heap_order() {
        let priorities = [3_i32, -5, 8, 8, 0, 12, 1, 12, -20];
        let mut context_heap = ContextHeap::with_capacity(priorities.len());
        let mut standard_heap = std::collections::BinaryHeap::new();
        for (index, priority) in priorities.into_iter().enumerate() {
            context_heap.push_by(index, |left, right| {
                priorities[*left].cmp(&priorities[*right])
            });
            standard_heap.push(priority);
        }

        let mut actual = Vec::new();
        while let Some(index) =
            context_heap.pop_by(|left, right| priorities[*left].cmp(&priorities[*right]))
        {
            actual.push(priorities[index]);
        }
        assert_eq!(
            actual,
            standard_heap
                .into_sorted_vec()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn dat_dfs_paging_is_lazy_bounded_and_order_exact() {
        let terms: Vec<String> = (b'a'..=b'z')
            .map(|label| char::from(label).to_string())
            .collect();
        let dictionary = DoubleArrayTrie::from_terms(terms.iter().map(String::as_str));

        let (mut paged_session, paged_root) =
            TraversalSession::capture(dictionary.traversal_root());
        let mut paged = paged_session.open_dfs_node(paged_root);
        assert!(!paged.is_final());
        let DfsEdgeSource::Paged {
            next, total, page, ..
        } = &paged.source
        else {
            panic!("the byte DAT advertises native O(1) edge paging");
        };
        assert_eq!((*next, *total, page.len()), (0, 26, 0));

        let first = paged_session
            .next_dfs_edge(&mut paged)
            .expect("the root has one page of edges")
            .0;
        assert_eq!(first, b'a');
        let DfsEdgeSource::Paged { next, page, .. } = &paged.source else {
            unreachable!()
        };
        assert_eq!((*next, page.len()), (DFS_EDGE_PAGE_CAPACITY, 7));

        let mut paged_labels = vec![first];
        while let Some((label, _)) = paged_session.next_dfs_edge(&mut paged) {
            paged_labels.push(label);
        }

        let (mut eager_session, eager_root) =
            TraversalSession::capture(dictionary.traversal_root());
        let mut eager = eager_session.open_eager_dfs_node(eager_root);
        let mut eager_labels = Vec::new();
        while let Some((label, _)) = eager_session.next_dfs_edge(&mut eager) {
            eager_labels.push(label);
        }
        assert_eq!(paged_labels, eager_labels);
        assert_eq!(paged_labels, (b'a'..=b'z').collect::<Vec<_>>());
    }

    #[test]
    fn session_selected_native_cursor_remains_one_machine_word() {
        type Native = <DynamicDawgCharNode as DictionaryNode>::SnapshotCursor;
        assert_eq!(size_of::<Native>(), size_of::<usize>());
        assert_eq!(size_of::<TraversalCursor<Native>>(), size_of::<usize>());
        assert_eq!(
            size_of::<TraversalCursor<SnapshotTraversalCursor>>(),
            size_of::<usize>()
        );
    }

    #[derive(Clone)]
    struct FullProjectionNode;

    impl DictionaryNode for FullProjectionNode {
        type Unit = u8;
        type SnapshotCursor = SnapshotTraversalCursor;
        type SnapshotGraphValueHandle = SnapshotTraversalCursor;

        fn snapshot_root_cursor(&self) -> Option<Self::SnapshotCursor> {
            panic!("ordinary query capture must not construct a full projection")
        }

        fn snapshot_cursor_requires_full_projection(&self) -> bool {
            true
        }

        fn is_final(&self) -> bool {
            false
        }

        fn transition(&self, _label: u8) -> Option<Self> {
            None
        }

        fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
            Box::new(std::iter::empty())
        }
    }

    #[test]
    fn ordinary_query_capture_keeps_full_projection_backends_lazy() {
        let (session, _) =
            TraversalSession::capture(DictionaryTraversalRoot::owned(FullProjectionNode));
        assert!(matches!(session.mode, TraversalMode::Owned(_)));
    }

    #[derive(Clone)]
    struct OwnedChainNode {
        units: Arc<[u8]>,
        depth: usize,
    }

    impl DictionaryNode for OwnedChainNode {
        type Unit = u8;
        type SnapshotCursor = SnapshotTraversalCursor;
        type SnapshotGraphValueHandle = SnapshotTraversalCursor;

        fn is_final(&self) -> bool {
            self.depth == self.units.len()
        }

        fn transition(&self, label: u8) -> Option<Self> {
            (self.units.get(self.depth) == Some(&label)).then(|| Self {
                units: Arc::clone(&self.units),
                depth: self.depth + 1,
            })
        }

        fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
            match self.units.get(self.depth).copied() {
                Some(label) => Box::new(std::iter::once((
                    label,
                    Self {
                        units: Arc::clone(&self.units),
                        depth: self.depth + 1,
                    },
                ))),
                None => Box::new(std::iter::empty()),
            }
        }
    }

    #[derive(Clone)]
    struct OwnedFanoutNode {
        labels: Arc<[u8]>,
        is_leaf: bool,
    }

    #[derive(Default)]
    struct OwnedPageStats {
        calls: AtomicUsize,
        children_constructed: AtomicUsize,
        max_capacity: AtomicUsize,
    }

    #[derive(Clone)]
    struct OwnedPagedFanoutNode<U: CharUnit> {
        labels: Arc<[U]>,
        paging: bool,
        revision: u64,
        value: Option<u64>,
        stats: Arc<OwnedPageStats>,
    }

    impl<U: CharUnit> OwnedPagedFanoutNode<U> {
        fn child(&self, label: U) -> Self {
            Self {
                labels: Arc::from([]),
                paging: self.paging,
                revision: self.revision,
                value: Some(self.revision ^ label.hash_to_u64()),
                stats: Arc::clone(&self.stats),
            }
        }
    }

    impl<U: CharUnit> DictionaryNode for OwnedPagedFanoutNode<U> {
        type Unit = U;
        type SnapshotCursor = SnapshotTraversalCursor;
        type SnapshotGraphValueHandle = SnapshotTraversalCursor;

        fn is_final(&self) -> bool {
            self.value.is_some()
        }

        fn transition(&self, label: U) -> Option<Self> {
            self.labels
                .binary_search(&label)
                .is_ok()
                .then(|| self.child(label))
        }

        fn edges(&self) -> Box<dyn Iterator<Item = (U, Self)> + '_> {
            Box::new(
                self.labels
                    .iter()
                    .copied()
                    .map(|label| (label, self.child(label))),
            )
        }

        fn supports_efficient_edge_paging(&self) -> bool {
            self.paging
        }

        fn visit_edge_page_and_finality<F>(
            &self,
            start: usize,
            capacity: usize,
            mut visitor: F,
        ) -> (bool, usize)
        where
            F: FnMut(U, Self),
        {
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            self.stats
                .max_capacity
                .fetch_max(capacity, Ordering::Relaxed);
            let total = self.labels.len();
            let end = start.saturating_add(capacity).min(total);
            let page = self.labels.get(start.min(total)..end).unwrap_or_default();
            self.stats
                .children_constructed
                .fetch_add(page.len(), Ordering::Relaxed);
            for label in page.iter().copied() {
                visitor(label, self.child(label));
            }
            (self.is_final(), total)
        }
    }

    impl<U: CharUnit> MappedDictionaryNode for OwnedPagedFanoutNode<U> {
        type Value = u64;

        fn value(&self) -> Option<Self::Value> {
            self.value
        }
    }

    impl DictionaryNode for OwnedFanoutNode {
        type Unit = u8;
        type SnapshotCursor = SnapshotTraversalCursor;
        type SnapshotGraphValueHandle = SnapshotTraversalCursor;

        fn is_final(&self) -> bool {
            self.is_leaf
        }

        fn transition(&self, label: u8) -> Option<Self> {
            (!self.is_leaf && self.labels.contains(&label)).then(|| Self {
                labels: Arc::clone(&self.labels),
                is_leaf: true,
            })
        }

        fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
            if self.is_leaf {
                return Box::new(std::iter::empty());
            }
            Box::new(self.labels.iter().copied().map(|label| {
                (
                    label,
                    Self {
                        labels: Arc::clone(&self.labels),
                        is_leaf: true,
                    },
                )
            }))
        }
    }

    #[test]
    fn dynamic_dawg_uses_flat_capture_and_both_dats_use_native_cursors() {
        let dynamic = DynamicDawgChar::<()>::from_sorted_terms(["a", "b"]);
        let (session, _) = TraversalSession::capture(dynamic.traversal_root());
        assert!(session.is_flat());

        let dat = DoubleArrayTrie::<()>::from_terms(["a", "b"]);
        let (session, _) = TraversalSession::capture(dat.traversal_root());
        assert!(!session.is_flat());
        assert_eq!(session.owned_slot_count(), None);

        let dat = DoubleArrayTrieChar::<()>::from_terms(["a", "b"]);
        let (session, _) = TraversalSession::capture(dat.traversal_root());
        assert!(!session.is_flat());
        assert_eq!(session.owned_slot_count(), None);
    }

    #[test]
    fn captured_session_keeps_its_revision_alive_across_mutation() {
        let dynamic = DynamicDawgChar::<()>::from_sorted_terms(["cat", "dog"]);
        let (mut session, root) = TraversalSession::capture(dynamic.traversal_root());
        dynamic.insert("cow");

        let mut labels = Vec::new();
        let is_final = session.filter_map_edges_and_finality(
            root,
            |_label| Some(()),
            |label, _, ()| labels.push(label),
        );
        assert!(!is_final);
        assert_eq!(labels, ['c', 'd']);
    }

    #[test]
    fn owned_arena_reuses_consumed_slots() {
        let root = OwnedChainNode {
            units: Arc::from(&b"abc"[..]),
            depth: 0,
        };
        let (mut session, mut cursor) =
            TraversalSession::capture(DictionaryTraversalRoot::owned(root));

        for expected in *b"abc" {
            let mut child = None;
            session.filter_map_edges_and_finality(
                cursor,
                |_label| Some(()),
                |label, next, ()| {
                    assert_eq!(label, expected);
                    child = Some(next);
                },
            );
            cursor = child.expect("the linear dictionary has one child");
        }

        assert_eq!(session.owned_slot_count(), Some(1));
    }

    #[test]
    fn discarded_owned_cursors_release_pruned_siblings_immediately() {
        let node = OwnedFanoutNode {
            labels: Arc::from((b'a'..=b'z').collect::<Vec<_>>()),
            is_leaf: false,
        };
        let (mut session, root) = TraversalSession::capture(DictionaryTraversalRoot::owned(node));
        let mut children = Vec::new();
        session.filter_map_edges_and_finality(root, Some, |_label, child, _| children.push(child));
        assert_eq!(session.owned_live_slot_count(), Some(children.len()));

        for child in children {
            session.discard_unexpanded(child);
        }
        assert_eq!(session.owned_live_slot_count(), Some(0));
    }

    fn assert_owned_paging_domain<U: CharUnit + std::fmt::Debug>(labels: Vec<U>)
    where
        crate::transducer::Unrestricted: crate::transducer::SubstitutionPolicyFor<U>,
    {
        const REVISION: u64 = 0x52a1_9e37;
        assert!(labels.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(labels.len() > DFS_EDGE_PAGE_CAPACITY);

        let stats = Arc::new(OwnedPageStats::default());
        let paged_root = OwnedPagedFanoutNode {
            labels: Arc::from(labels.clone()),
            paging: true,
            revision: REVISION,
            value: None,
            stats: Arc::clone(&stats),
        };
        let (mut session, root) =
            TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(paged_root.clone()));
        let mut edges = session.open_dfs_node(root);
        assert!(!edges.is_final());
        let DfsEdgeSource::Paged {
            owned,
            next,
            total,
            page,
            ..
        } = &edges.source
        else {
            panic!("an efficient owned-node pager must not use eager storage")
        };
        assert!(owned.is_some());
        assert_eq!((*next, *total, page.len()), (0, labels.len(), 0));
        assert_eq!(stats.calls.load(Ordering::Relaxed), 1);
        assert_eq!(stats.children_constructed.load(Ordering::Relaxed), 0);
        assert_eq!(session.owned_live_slot_count(), Some(0));

        let (first_label, first_child) = session
            .next_dfs_edge(&mut edges)
            .expect("the first bounded page contains an edge");
        assert_eq!(first_label, labels[0]);
        assert!(session.owned_live_slot_count().unwrap() <= DFS_EDGE_PAGE_CAPACITY);
        let source = session
            .filter_map_edges_and_final_source(first_child, |_| None::<()>, |_, _, _| {})
            .expect("every fanout child is final");
        assert_eq!(
            session.resolve_final_value(source, None),
            Some(REVISION ^ first_label.hash_to_u64())
        );

        let mut observed = vec![first_label];
        while let Some((label, child)) = session.next_dfs_edge(&mut edges) {
            assert!(session.owned_live_slot_count().unwrap() <= DFS_EDGE_PAGE_CAPACITY);
            observed.push(label);
            session.discard_unexpanded(child);
        }
        assert_eq!(observed, labels);
        assert_eq!(session.owned_live_slot_count(), Some(0));
        assert_eq!(
            stats.children_constructed.load(Ordering::Relaxed),
            labels.len()
        );
        assert_eq!(
            stats.max_capacity.load(Ordering::Relaxed),
            DFS_EDGE_PAGE_CAPACITY
        );
        assert_eq!(
            stats.calls.load(Ordering::Relaxed),
            1 + labels.len().div_ceil(DFS_EDGE_PAGE_CAPACITY)
        );

        let eager_root = OwnedPagedFanoutNode {
            labels: Arc::from(labels.clone()),
            paging: false,
            revision: REVISION,
            value: None,
            stats: Arc::new(OwnedPageStats::default()),
        };
        let (mut eager_session, eager_root_cursor) =
            TraversalSession::capture(DictionaryTraversalRoot::owned(eager_root.clone()));
        let mut eager = eager_session.open_dfs_node(eager_root_cursor);
        assert!(matches!(&eager.source, DfsEdgeSource::Eager(_)));
        let mut eager_observed = Vec::new();
        while let Some((label, child)) = eager_session.next_dfs_edge(&mut eager) {
            eager_observed.push(label);
            eager_session.discard_unexpanded(child);
        }
        assert_eq!(eager_observed, labels);

        let early_stats = Arc::new(OwnedPageStats::default());
        let early_root = OwnedPagedFanoutNode {
            labels: Arc::from(labels.clone()),
            paging: true,
            revision: REVISION,
            value: None,
            stats: Arc::clone(&early_stats),
        };
        let mut early = crate::transducer::PrefixQueryIterator::new(
            early_root,
            Vec::new(),
            1,
            crate::transducer::Algorithm::Standard,
        );
        assert_eq!(early.next().unwrap().units, vec![labels[0]]);
        assert!(
            early_stats.children_constructed.load(Ordering::Relaxed) <= DFS_EDGE_PAGE_CAPACITY,
            "an early-stop query must construct at most one bounded edge page"
        );
        drop(early);

        let paged_results = crate::transducer::PrefixQueryIterator::new(
            paged_root,
            Vec::new(),
            1,
            crate::transducer::Algorithm::Standard,
        )
        .map(|result| result.units)
        .collect::<Vec<_>>();
        let eager_results = crate::transducer::PrefixQueryIterator::new(
            eager_root,
            Vec::new(),
            1,
            crate::transducer::Algorithm::Standard,
        )
        .map(|result| result.units)
        .collect::<Vec<_>>();
        let expected = labels
            .into_iter()
            .map(|label| vec![label])
            .collect::<Vec<_>>();
        assert_eq!(paged_results, eager_results);
        assert_eq!(paged_results, expected);
    }

    #[test]
    fn owned_paging_is_lazy_bounded_and_exact_for_every_unit_domain() {
        assert_owned_paging_domain((0_u8..96).collect());
        assert_owned_paging_domain(
            (0_u32..96)
                .map(|offset| char::from_u32(0x400 + offset).unwrap())
                .collect(),
        );
        assert_owned_paging_domain((0_u64..96).map(|offset| (offset << 40) | offset).collect());
    }

    #[test]
    fn slice_backed_suffix_automata_select_owned_paging() {
        let byte = SuffixAutomaton::<()>::from_texts(["alpha", "omega"]);
        let (mut session, root) = TraversalSession::capture(byte.traversal_root());
        let edges = session.open_dfs_node(root);
        assert!(matches!(
            edges.source,
            DfsEdgeSource::Paged { owned: Some(_), .. }
        ));

        let chars = SuffixAutomatonChar::<()>::from_texts(["café", "東京"]);
        let (mut session, root) = TraversalSession::capture(chars.traversal_root());
        let edges = session.open_dfs_node(root);
        assert!(matches!(
            edges.source,
            DfsEdgeSource::Paged { owned: Some(_), .. }
        ));
    }

    #[cfg(feature = "pathmap-backend")]
    #[test]
    fn byte_pathmap_pages_by_mask_rank_while_char_pathmap_stays_exact_eager() {
        let byte = PathMapDictionary::<()>::from_terms(["alpha", "omega"]);
        let (mut session, root) = TraversalSession::capture(byte.traversal_root());
        let edges = session.open_dfs_node(root);
        assert!(matches!(
            edges.source,
            DfsEdgeSource::Paged { owned: Some(_), .. }
        ));

        let chars = PathMapDictionaryChar::<()>::from_terms(["café", "東京"]);
        let (mut session, root) = TraversalSession::capture(chars.traversal_root());
        let edges = session.open_dfs_node(root);
        assert!(matches!(edges.source, DfsEdgeSource::Eager(_)));
    }

    #[test]
    fn mapped_native_cursor_keeps_old_revision_and_resolves_value_lazily() {
        let dynamic = DynamicDawgChar::from_sorted_terms_with_values([("cat", 7_u64), ("dog", 8)]);
        let root_node = dynamic.root();
        let (mut session, mut position) =
            TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(root_node));
        dynamic.insert_with_value("cow", 9);

        for expected in ['c', 'a', 't'] {
            let mut child = None;
            let final_source = session.filter_map_edges_and_final_source(
                position,
                |label| (label == expected).then_some(()),
                |_label, next, ()| child = Some(next),
            );
            assert!(final_source.is_none());
            position = child.expect("the captured old revision contains cat");
        }

        let final_source = session
            .filter_map_edges_and_final_source(position, |_| None::<()>, |_, _, _| {})
            .expect("cat is final in the captured revision");
        assert_eq!(session.resolve_final_value(final_source, None), Some(7));
    }

    #[test]
    fn exact_typed_traversal_buffers_retain_capacity_but_not_values() {
        #[derive(Clone)]
        struct DropToken(Arc<AtomicUsize>);

        impl Drop for DropToken {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        let (mut queue, mut path) = acquire_traversal_buffers::<DropToken, DropToken>();
        queue.reserve(2_048);
        path.reserve(4_096);
        let queue_capacity = queue.capacity();
        let path_capacity = path.capacity();
        queue.push_back(DropToken(Arc::clone(&dropped)));
        path.push(DropToken(Arc::clone(&dropped)));

        release_traversal_buffers(queue, path);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);

        let (queue, path) = acquire_traversal_buffers::<DropToken, DropToken>();
        assert!(queue.is_empty());
        assert!(path.is_empty());
        assert!(queue.capacity() >= queue_capacity);
        assert!(path.capacity() >= path_capacity);
        release_traversal_buffers(queue, path);
    }

    #[test]
    fn owned_buffers_can_cross_threads_and_return_to_the_destination_cache() {
        let handle = std::thread::spawn(|| {
            let (mut queue, mut path) = acquire_traversal_buffers::<[u8; 13], [u8; 17]>();
            queue.reserve(1_024);
            path.reserve(2_048);
            let capacities = (queue.capacity(), path.capacity());
            (capacities, queue, path)
        });
        let (capacities, queue, path) = handle.join().expect("buffer owner crosses threads");
        release_traversal_buffers(queue, path);

        let (queue, path) = acquire_traversal_buffers::<[u8; 13], [u8; 17]>();
        assert!(queue.capacity() >= capacities.0);
        assert!(path.capacity() >= capacities.1);
        release_traversal_buffers(queue, path);
    }

    #[test]
    fn bucketed_storage_adapts_bucket_count_and_drops_every_value() {
        struct BucketDropToken(Arc<AtomicUsize>);
        struct PathDropToken(Arc<AtomicUsize>);

        impl Drop for BucketDropToken {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        impl Drop for PathDropToken {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        let (mut buckets, mut path) =
            acquire_bucketed_traversal_buffers::<BucketDropToken, PathDropToken>(3);
        buckets[0].reserve(1_024);
        let first_capacity = buckets[0].capacity();
        buckets[0].push_back(BucketDropToken(Arc::clone(&dropped)));
        buckets[2].push_back(BucketDropToken(Arc::clone(&dropped)));
        path.push(PathDropToken(Arc::clone(&dropped)));
        release_bucketed_traversal_buffers(buckets, path);
        assert_eq!(dropped.load(Ordering::Relaxed), 3);

        let (buckets, path) =
            acquire_bucketed_traversal_buffers::<BucketDropToken, PathDropToken>(2);
        assert_eq!(buckets.len(), 2);
        assert!(buckets.iter().all(VecDeque::is_empty));
        assert!(buckets[0].capacity() >= first_capacity);
        assert!(path.is_empty());
        release_bucketed_traversal_buffers(buckets, path);

        let (buckets, path) =
            acquire_bucketed_traversal_buffers::<BucketDropToken, PathDropToken>(5);
        assert_eq!(buckets.len(), 5);
        assert!(buckets.iter().all(VecDeque::is_empty));
        assert!(path.is_empty());
        release_bucketed_traversal_buffers(buckets, path);
    }
}
