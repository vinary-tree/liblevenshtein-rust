//! Project-owned binding model for automata over foreign dictionary resources.
//!
//! liblevenshtein deliberately does not construct or mutate dictionaries here.
//! Concrete dictionary CRUD belongs to libdictenstein. This module consumes the
//! shared [`vinary_tree_interop`] dictionary-resource contract and exposes lazy,
//! snapshot-stable automaton cursors.

#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::nfa::NFAChar;
#[cfg(feature = "bindings-phonetic")]
use crate::transducer::language::{LanguageProduct, MappedLanguageQueryIterator};
use crate::transducer::{
    Algorithm, RankedValueQueryIterator, Suggestion, ValueYieldingQueryIterator,
};
use arc_swap::ArcSwapOption;
use libdictenstein::concurrent_slots::{AtomicOnceBox, AtomicTakeBox, HybridOnceBoxSlots};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{
    CharUnit, DictionaryNode, DictionaryTraversalRoot, MappedDictionaryNode,
    SnapshotTraversalCursor, SnapshotTraversalEdge, SnapshotTraversalGraph, SnapshotTraversalNode,
};
use rustc_hash::FxHashMap;
use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::num::{NonZeroU64, NonZeroUsize};
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use vinary_tree_interop::{
    dictionary_flags, VtDictionaryEdge, VtDictionaryGraphVTable, VtDictionaryGraphView,
    VtDictionaryVTable, VtDictionaryVisitVTable, VtOptionalU64, VtResource, VtResourceVTable,
    VtSnapshotIdentity, VtSnapshotIdentityVTable, VtStatus, VtUnitDomain, VtValueDomain,
    VT_ABI_VERSION, VT_DICTIONARY_GRAPH_INTERFACE_ID, VT_DICTIONARY_GRAPH_INTERFACE_VERSION,
    VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION, VT_DICTIONARY_VISIT_INTERFACE_ID,
    VT_DICTIONARY_VISIT_INTERFACE_VERSION, VT_RECOMMENDED_EDGE_BATCH,
    VT_SNAPSHOT_IDENTITY_INTERFACE_ID, VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION,
};
#[cfg(test)]
use vinary_tree_interop::{VtDictionaryGraphEdge, VtDictionaryGraphNode};

/// Default number of results transferred across a managed-language boundary.
pub const DEFAULT_MATCH_BATCH: usize = 256;

/// Error raised while validating or invoking a cross-project resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindingError {
    /// One or both resource words were null.
    NullResource,
    /// The base ABI version or vtable layout is incompatible.
    IncompatibleResourceAbi,
    /// The resource does not publish the dictionary interface.
    MissingDictionaryInterface,
    /// The dictionary interface is too old or incomplete.
    IncompatibleDictionaryInterface,
    /// A different key domain was supplied to a domain-specific operation.
    UnitDomainMismatch {
        /// Domain required by the operation.
        expected: VtUnitDomain,
        /// Domain published by the provider.
        actual: VtUnitDomain,
    },
    /// Byte-payload values require a future binding interface version.
    UnsupportedValueDomain(VtValueDomain),
    /// A provider callback returned an error status.
    Provider(VtStatus),
    /// A provider returned malformed output despite reporting success.
    InvalidProviderOutput(&'static str),
    /// Ordered streaming is not yet defined for this unit domain.
    UnsupportedOrdering(VtUnitDomain),
    /// A batch size of zero was requested.
    EmptyBatch,
}

impl fmt::Display for BindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullResource => formatter.write_str("resource context or vtable is null"),
            Self::IncompatibleResourceAbi => {
                formatter.write_str("resource uses an incompatible vinary-tree interop ABI")
            }
            Self::MissingDictionaryInterface => {
                formatter.write_str("resource does not implement the dictionary interface")
            }
            Self::IncompatibleDictionaryInterface => {
                formatter.write_str("dictionary interface is incomplete or incompatible")
            }
            Self::UnitDomainMismatch { expected, actual } => write!(
                formatter,
                "dictionary unit domain mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::UnsupportedValueDomain(domain) => {
                write!(
                    formatter,
                    "dictionary value domain {domain:?} is unsupported"
                )
            }
            Self::Provider(status) => write!(formatter, "dictionary provider returned {status:?}"),
            Self::InvalidProviderOutput(message) => {
                write!(
                    formatter,
                    "dictionary provider returned invalid output: {message}"
                )
            }
            Self::UnsupportedOrdering(domain) => {
                write!(formatter, "ordered streaming is unsupported for {domain:?}")
            }
            Self::EmptyBatch => formatter.write_str("batch size must be greater than zero"),
        }
    }
}

impl std::error::Error for BindingError {}

fn status(raw: u32) -> Result<(), BindingError> {
    // The wire carries a raw u32 (interop status rule): decode before any
    // enum-typed use, and treat an out-of-range discriminant as provider
    // misbehavior rather than undefined behavior (ledger LLEV-B6).
    let Some(decoded) = VtStatus::from_raw(raw) else {
        return Err(BindingError::InvalidProviderOutput(
            "provider returned an out-of-range status code",
        ));
    };
    if decoded.is_ok() {
        Ok(())
    } else {
        Err(BindingError::Provider(decoded))
    }
}

#[derive(Clone)]
enum CallGate {
    Parallel,
    Serial(Arc<Mutex<()>>),
}

/// One owned retain of a provider resource plus its discovered dictionary
/// interface. All callbacks for the default custom-provider mode are serialized
/// on the caller's thread. Providers opt into concurrent/reentrant calls by
/// publishing [`dictionary_flags::PARALLEL_REENTRANT`].
#[derive(Clone)]
enum ForeignCapturedGraph {
    Byte(Arc<SnapshotTraversalGraph<u8, ForeignGraphValueHandle>>),
    Unicode(Arc<SnapshotTraversalGraph<char, ForeignGraphValueHandle>>),
    U64(Arc<SnapshotTraversalGraph<u64, ForeignGraphValueHandle>>),
}

impl ForeignCapturedGraph {
    #[cfg(feature = "perf-instrumentation")]
    fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Byte(left), Self::Byte(right)) => Arc::ptr_eq(left, right),
            (Self::Unicode(left), Self::Unicode(right)) => Arc::ptr_eq(left, right),
            (Self::U64(left), Self::U64(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}

struct Provider {
    resource: VtResource,
    dictionary: *const VtDictionaryVTable,
    visit: Option<*const VtDictionaryVisitVTable>,
    graph_vtable: Option<*const VtDictionaryGraphVTable>,
    graph: Option<ForeignCapturedGraph>,
    gate: CallGate,
    fault: AtomicTakeBox<BindingError>,
    identity: Option<VtSnapshotIdentity>,
    node_cache: Option<Arc<NodeCache>>,
    graph_memo: Arc<CapturedGraphMemo>,
}

struct CachedForeignNode {
    node: u64,
    is_final: bool,
    edges: Box<[CachedForeignEdge]>,
}

/// Immutable edge descriptor plus a non-owning shortcut to its child cache.
/// The shortcut is populated after the child is first inspected and remains
/// valid until the query provider releases the append-only node cache.
struct CachedForeignEdge {
    label: u64,
    node: u64,
    cached_child: AtomicUsize,
}

/// Stable node identity and its lazily published immutable descriptor.
/// Foreign handles point directly at this append-only entry, so the traversal
/// queue does not carry a redundant numeric ID plus an optional cache hint.
struct CachedForeignEntry {
    node: u64,
    descriptor: AtomicOnceBox<CachedForeignNode>,
}

impl CachedForeignEntry {
    fn new(node: u64) -> Self {
        Self {
            node,
            descriptor: AtomicOnceBox::new(),
        }
    }
}

const FOREIGN_TARGET_READY_TAG: usize = 1;

/// One-word, non-owning target for a foreign snapshot cursor.
///
/// Pending targets name the stable cache entry whose descriptor may not have
/// been populated yet. Ready targets name the immutable descriptor directly.
/// Both allocations are at least two-byte aligned, so the low bit distinguishes
/// the two states without changing the size of a queued dictionary node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
struct ForeignTarget(NonZeroUsize);

/// Consumer-local pointer capability used only by native foreign traversal.
///
/// Keeping this opaque prevents a tagged cache pointer from being mistaken
/// for either a dense graph index or a provider-owned ABI handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
struct ForeignSnapshotCursor(ForeignTarget);

/// Opaque non-zero provider handle retained by a captured foreign graph.
///
/// Unlike a native foreign cursor this value crosses the provider ABI and is
/// passed back byte-for-byte to the graph value callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
struct ForeignGraphValueHandle(NonZeroU64);

impl ForeignGraphValueHandle {
    #[inline]
    const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    #[inline]
    const fn get(self) -> u64 {
        self.0.get()
    }
}

impl ForeignTarget {
    #[inline]
    fn pending(entry: NonNull<CachedForeignEntry>) -> Self {
        let address = entry.as_ptr().expose_provenance();
        debug_assert_eq!(address & FOREIGN_TARGET_READY_TAG, 0);
        Self(NonZeroUsize::new(address).expect("foreign cache entries are non-null"))
    }

    #[inline]
    fn ready(node: NonNull<CachedForeignNode>) -> Self {
        let address = node.as_ptr().expose_provenance();
        debug_assert_eq!(address & FOREIGN_TARGET_READY_TAG, 0);
        Self(
            NonZeroUsize::new(address | FOREIGN_TARGET_READY_TAG)
                .expect("foreign cached nodes are non-null"),
        )
    }

    #[inline]
    fn from_encoded(encoded: usize) -> Option<Self> {
        NonZeroUsize::new(encoded).map(Self)
    }

    #[inline]
    fn encoded(self) -> usize {
        self.0.get()
    }

    #[inline]
    fn is_ready(self) -> bool {
        self.encoded() & FOREIGN_TARGET_READY_TAG != 0
    }

    #[inline]
    fn pending_entry(self) -> Option<NonNull<CachedForeignEntry>> {
        if self.is_ready() {
            return None;
        }
        // SAFETY: pending targets are encoded only from live append-only cache
        // entries, and the owning Provider outlives every traversal cursor.
        Some(unsafe {
            NonNull::new_unchecked(std::ptr::with_exposed_provenance_mut(self.encoded()))
        })
    }

    #[inline]
    fn ready_node(self) -> Option<NonNull<CachedForeignNode>> {
        if !self.is_ready() {
            return None;
        }
        let address = self.encoded() & !FOREIGN_TARGET_READY_TAG;
        // SAFETY: ready targets are encoded only from published immutable
        // descriptors retained by the same append-only cache as their entry.
        Some(unsafe { NonNull::new_unchecked(std::ptr::with_exposed_provenance_mut(address)) })
    }

    #[inline]
    fn ready_if_published(self) -> Option<Self> {
        let entry = self.pending_entry()?;
        crate::causal_perf::record_foreign_pending_descriptor_loads(1);
        // SAFETY: target construction and Provider ownership keep the entry
        // alive. Acquire publication inside AtomicOnceBox makes the immutable
        // descriptor fully visible before its pointer is encoded as ready.
        unsafe { entry.as_ref() }
            .descriptor
            .get()
            .map(|node| Self::ready(NonNull::from(node)))
    }
}

const _: () = assert!(std::mem::align_of::<CachedForeignEntry>() >= 2);
const _: () = assert!(std::mem::align_of::<CachedForeignNode>() >= 2);

/// Promote an already-published child target without invoking a provider.
/// State is monotonic (`pending -> ready`), and both allocations remain live
/// until the snapshot cache is exclusively released, so no ABA is possible.
#[inline]
fn promote_published_foreign_target(slot: &AtomicUsize, target: ForeignTarget) -> ForeignTarget {
    if target.is_ready() {
        return target;
    }
    let Some(ready) = target.ready_if_published() else {
        return target;
    };
    match slot.compare_exchange(
        target.encoded(),
        ready.encoded(),
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {
            crate::causal_perf::record_foreign_edge_cursor_promotions(1);
            ready
        }
        Err(observed) => ForeignTarget::from_encoded(observed)
            .expect("foreign child target cannot return to the empty state"),
    }
}

#[inline(always)]
fn foreign_ready_cursors_enabled() -> bool {
    #[cfg(feature = "resource-profiling")]
    {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_FOREIGN_READY_CURSORS").is_none()
        })
    }
    #[cfg(not(feature = "resource-profiling"))]
    {
        true
    }
}

#[inline(always)]
#[cfg(feature = "resource-profiling")]
fn monolithic_foreign_inspection_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_MONOLITHIC_FOREIGN_INSPECT").is_some()
    })
}

#[inline(always)]
fn foreign_snapshot_graph_enabled() -> bool {
    #[cfg(feature = "resource-profiling")]
    {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_FOREIGN_SNAPSHOT_GRAPH").is_none()
        })
    }
    #[cfg(not(feature = "resource-profiling"))]
    {
        true
    }
}

const NODE_CACHE_CHUNK_SIZE: usize = 256;
const NODE_CACHE_DENSE_CHUNKS: usize = 512;
const NODE_CACHE_SHARDS: usize = 64;
type NodeCache = HybridOnceBoxSlots<
    CachedForeignEntry,
    NODE_CACHE_CHUNK_SIZE,
    NODE_CACHE_DENSE_CHUNKS,
    NODE_CACHE_SHARDS,
>;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct NodeCacheKey {
    identity: VtSnapshotIdentity,
    resource_vtable: usize,
    dictionary_vtable: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GraphCacheKey {
    identity: VtSnapshotIdentity,
    resource_vtable: usize,
    dictionary_vtable: usize,
    graph_vtable: usize,
    unit_domain: u32,
}

type NodeCacheRegistry = Mutex<FxHashMap<NodeCacheKey, Weak<NodeCache>>>;

fn shared_node_cache(key: Option<NodeCacheKey>) -> Arc<NodeCache> {
    let Some(key) = key else {
        return Arc::new(NodeCache::new());
    };
    static REGISTRY: OnceLock<NodeCacheRegistry> = OnceLock::new();
    let mut registry = REGISTRY
        .get_or_init(|| Mutex::new(FxHashMap::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cache) = registry.get(&key).and_then(Weak::upgrade) {
        return cache;
    }
    registry.retain(|_, cache| cache.strong_count() != 0);
    let cache = Arc::new(NodeCache::new());
    registry.insert(key, Arc::downgrade(&cache));
    cache
}

struct CapturedGraphEntry {
    key: GraphCacheKey,
    graph: OnceLock<Result<ForeignCapturedGraph, BindingError>>,
}

/// Revision-keyed publication shared by a live resource and all snapshots it
/// produces. Readers perform one atomic `Arc` load. The winning revision cell
/// uses `OnceLock` for single-flight graph validation/import, so concurrent
/// first queries neither serialize on a process-wide registry nor duplicate
/// the O(nodes + edges) decode.
struct CapturedGraphMemo {
    current: ArcSwapOption<CapturedGraphEntry>,
    #[cfg(feature = "resource-profiling")]
    legacy_control: Mutex<Option<(GraphCacheKey, ForeignCapturedGraph)>>,
}

impl CapturedGraphMemo {
    fn new() -> Self {
        Self {
            current: ArcSwapOption::empty(),
            #[cfg(feature = "resource-profiling")]
            legacy_control: Mutex::new(None),
        }
    }
}

#[cfg(feature = "resource-profiling")]
fn legacy_snapshot_locks_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_LEGACY_SNAPSHOT_LOCKS").is_some()
    })
}

fn shared_captured_graph(
    key: Option<GraphCacheKey>,
    memo: &CapturedGraphMemo,
    decode: impl FnOnce() -> Result<ForeignCapturedGraph, BindingError>,
) -> Result<ForeignCapturedGraph, BindingError> {
    let Some(key) = key else {
        return decode();
    };

    #[cfg(feature = "resource-profiling")]
    if legacy_snapshot_locks_enabled() {
        {
            let current = memo
                .legacy_control
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some((_, graph)) = current.as_ref().filter(|(cached, _)| *cached == key) {
                return Ok(graph.clone());
            }
        }
        let decoded = decode()?;
        let mut current = memo
            .legacy_control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, graph)) = current.as_ref().filter(|(cached, _)| *cached == key) {
            return Ok(graph.clone());
        }
        *current = Some((key, decoded.clone()));
        return Ok(decoded);
    }

    let mut decode = Some(decode);
    let candidate = Arc::new(CapturedGraphEntry {
        key,
        graph: OnceLock::new(),
    });

    loop {
        let current = memo.current.load_full();
        if let Some(entry) = current.as_ref().filter(|entry| entry.key == key) {
            return entry
                .graph
                .get_or_init(|| decode.take().expect("the graph decoder runs at most once")())
                .clone();
        }

        let previous = memo
            .current
            .compare_and_swap(&current, Some(Arc::clone(&candidate)));
        if previous.as_ref().map(Arc::as_ptr) == current.as_ref().map(Arc::as_ptr) {
            return candidate
                .graph
                .get_or_init(|| decode.take().expect("the graph decoder runs at most once")())
                .clone();
        }
    }
}

// Raw provider pointers are never dereferenced outside `call`, and `call`
// serializes every callback unless the provider explicitly advertises the
// stronger parallel/reentrant contract.
unsafe impl Send for Provider {}
unsafe impl Sync for Provider {}

impl fmt::Debug for Provider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Provider")
            .field("context", &self.resource.context)
            .field("dictionary", &self.dictionary)
            .field("unit_domain", &self.vtable().unit_domain)
            .field("value_domain", &self.vtable().value_domain)
            .finish_non_exhaustive()
    }
}

impl Drop for Provider {
    fn drop(&mut self) {
        let release = unsafe { (*self.resource.vtable).release };
        if let Some(release) = release {
            match &self.gate {
                CallGate::Parallel => unsafe { release(self.resource.context) },
                CallGate::Serial(lock) => {
                    let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    unsafe { release(self.resource.context) };
                }
            }
        }
    }
}

impl Provider {
    unsafe fn from_borrowed(resource: VtResource) -> Result<Arc<Self>, BindingError> {
        validate_base(resource)?;
        let base = &*resource.vtable;
        let retain = base.retain.ok_or(BindingError::IncompatibleResourceAbi)?;
        let release = base.release.ok_or(BindingError::IncompatibleResourceAbi)?;
        retain(resource.context);
        match Self::from_owned_with_graph_memo(resource, Arc::new(CapturedGraphMemo::new())) {
            Ok(provider) => Ok(provider),
            Err(error) => {
                release(resource.context);
                Err(error)
            }
        }
    }

    unsafe fn from_owned_with_graph_memo(
        resource: VtResource,
        graph_memo: Arc<CapturedGraphMemo>,
    ) -> Result<Arc<Self>, BindingError> {
        validate_base(resource)?;
        let base = &*resource.vtable;
        let query = base
            .query_interface
            .ok_or(BindingError::IncompatibleResourceAbi)?;
        let mut dictionary: *const c_void = std::ptr::null();
        let result = query(
            resource.context,
            &VT_DICTIONARY_INTERFACE_ID,
            VT_DICTIONARY_INTERFACE_VERSION,
            &mut dictionary,
        );
        if VtStatus::from_raw(result) == Some(VtStatus::Unsupported) {
            return Err(BindingError::MissingDictionaryInterface);
        }
        status(result)?;
        if dictionary.is_null() {
            return Err(BindingError::InvalidProviderOutput(
                "query_interface returned a null vtable",
            ));
        }
        let dictionary = dictionary.cast::<VtDictionaryVTable>();
        validate_dictionary(&*dictionary)?;
        let descriptor = &*dictionary;
        if descriptor.value_domain == VtValueDomain::Bytes {
            return Err(BindingError::UnsupportedValueDomain(
                descriptor.value_domain,
            ));
        }
        let gate = if descriptor.flags & dictionary_flags::PARALLEL_REENTRANT != 0 {
            CallGate::Parallel
        } else {
            CallGate::Serial(Arc::new(Mutex::new(())))
        };
        let mut visit: *const c_void = std::ptr::null();
        let visit_status = query(
            resource.context,
            &VT_DICTIONARY_VISIT_INTERFACE_ID,
            VT_DICTIONARY_VISIT_INTERFACE_VERSION,
            &mut visit,
        );
        let visit = if VtStatus::from_raw(visit_status) == Some(VtStatus::Unsupported) {
            None
        } else {
            status(visit_status)?;
            if visit.is_null() {
                return Err(BindingError::InvalidProviderOutput(
                    "node-visit query returned a null vtable",
                ));
            }
            let visit = visit.cast::<VtDictionaryVisitVTable>();
            validate_dictionary_visit(&*visit)?;
            Some(visit)
        };
        let identity = if descriptor.flags & dictionary_flags::IMMUTABLE != 0 {
            let mut identity_vtable: *const c_void = std::ptr::null();
            let identity_status = query(
                resource.context,
                &VT_SNAPSHOT_IDENTITY_INTERFACE_ID,
                VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION,
                &mut identity_vtable,
            );
            if VtStatus::from_raw(identity_status) == Some(VtStatus::Unsupported) {
                None
            } else {
                status(identity_status)?;
                if identity_vtable.is_null() {
                    return Err(BindingError::InvalidProviderOutput(
                        "snapshot-identity query returned a null vtable",
                    ));
                }
                let identity_vtable = &*identity_vtable.cast::<VtSnapshotIdentityVTable>();
                validate_snapshot_identity(identity_vtable)?;
                let callback = identity_vtable
                    .identity
                    .ok_or(BindingError::IncompatibleDictionaryInterface)?;
                let mut identity = VtSnapshotIdentity::default();
                let raw = match &gate {
                    CallGate::Parallel => callback(resource.context, &mut identity),
                    CallGate::Serial(lock) => {
                        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        callback(resource.context, &mut identity)
                    }
                };
                status(raw)?;
                Some(identity)
            }
        } else {
            None
        };
        let mut graph_vtable: *const c_void = std::ptr::null();
        let graph_status = if foreign_snapshot_graph_enabled() {
            query(
                resource.context,
                &VT_DICTIONARY_GRAPH_INTERFACE_ID,
                VT_DICTIONARY_GRAPH_INTERFACE_VERSION,
                &mut graph_vtable,
            )
        } else {
            VtStatus::Unsupported.to_raw()
        };
        let (graph_vtable, graph) = if VtStatus::from_raw(graph_status)
            == Some(VtStatus::Unsupported)
        {
            (None, None)
        } else {
            status(graph_status)?;
            if descriptor.flags & dictionary_flags::IMMUTABLE == 0 {
                return Err(BindingError::InvalidProviderOutput(
                    "mutable resource exposed an immutable snapshot graph",
                ));
            }
            if graph_vtable.is_null() {
                return Err(BindingError::InvalidProviderOutput(
                    "snapshot-graph query returned a null vtable",
                ));
            }
            let graph_vtable = graph_vtable.cast::<VtDictionaryGraphVTable>();
            validate_dictionary_graph(&*graph_vtable, descriptor.value_domain)?;
            let graph_key = identity.map(|identity| GraphCacheKey {
                identity,
                resource_vtable: resource.vtable as usize,
                dictionary_vtable: dictionary as usize,
                graph_vtable: graph_vtable as usize,
                unit_domain: descriptor.unit_domain as u32,
            });
            let graph = shared_captured_graph(graph_key, &graph_memo, || {
                let callback = (*graph_vtable)
                    .graph
                    .ok_or(BindingError::IncompatibleDictionaryInterface)?;
                let mut view = VtDictionaryGraphView::default();
                let raw = match &gate {
                    CallGate::Parallel => callback(resource.context, &mut view),
                    CallGate::Serial(lock) => {
                        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        callback(resource.context, &mut view)
                    }
                };
                status(raw)?;
                decode_captured_graph(descriptor.unit_domain, view)
            })?;
            (Some(graph_vtable), Some(graph))
        };
        // Snapshot producer/revision tokens are scoped to the provider
        // implementation that minted them. Including both ABI vtable
        // addresses prevents independent providers or separately loaded DSO
        // copies with equal numeric tokens from sharing incompatible nodes.
        let node_cache = graph.is_none().then(|| {
            shared_node_cache(identity.map(|identity| NodeCacheKey {
                identity,
                resource_vtable: resource.vtable as usize,
                dictionary_vtable: dictionary as usize,
            }))
        });
        Ok(Arc::new(Self {
            resource,
            dictionary,
            visit,
            graph_vtable,
            graph,
            gate,
            fault: AtomicTakeBox::new(),
            identity,
            node_cache,
            graph_memo,
        }))
    }

    fn vtable(&self) -> &VtDictionaryVTable {
        // Validated for the lifetime of the retained resource at construction.
        unsafe { &*self.dictionary }
    }

    fn visit_vtable(&self) -> Option<&VtDictionaryVisitVTable> {
        self.visit.map(|visit| unsafe { &*visit })
    }

    fn call<T>(&self, operation: impl FnOnce() -> T) -> T {
        match &self.gate {
            CallGate::Parallel => operation(),
            CallGate::Serial(lock) => {
                let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                operation()
            }
        }
    }

    fn record_fault(&self, error: BindingError) {
        self.fault.publish_if_empty(error);
    }

    fn take_fault(&self) -> Option<BindingError> {
        self.fault.take()
    }

    /// Retain one query-local provider owner while sharing only immutable
    /// snapshot state, the callback gate, and the append-only node cache.
    ///
    /// Keeping the fault mailbox query-local prevents concurrent cursors over
    /// the same snapshot from consuming one another's provider errors. The
    /// retained allocation also becomes the stable owner behind copy-only
    /// [`ProviderRef`] node keys for the entire cursor lifetime.
    fn fork_query_owner(&self) -> Arc<Self> {
        let retain = unsafe {
            (*self.resource.vtable)
                .retain
                .expect("validated providers always publish retain")
        };
        self.call(|| unsafe { retain(self.resource.context) });
        Arc::new(Self {
            resource: self.resource,
            dictionary: self.dictionary,
            visit: self.visit,
            graph_vtable: self.graph_vtable,
            graph: self.graph.clone(),
            gate: self.gate.clone(),
            fault: AtomicTakeBox::new(),
            identity: self.identity,
            node_cache: self.node_cache.clone(),
            graph_memo: Arc::clone(&self.graph_memo),
        })
    }

    fn node_entry(&self, node: u64) -> &CachedForeignEntry {
        let node_cache = self
            .node_cache
            .as_ref()
            .expect("compact graph traversal does not enter the legacy node cache");
        if let Some(entry) = node_cache.get(node) {
            return entry;
        }
        node_cache.install_if_absent(node, CachedForeignEntry::new(node))
    }

    fn cache_node<'a>(
        &self,
        entry: &'a CachedForeignEntry,
        value: CachedForeignNode,
    ) -> &'a CachedForeignNode {
        entry.descriptor.publish_if_absent(value)
    }

    fn snapshot(self: &Arc<Self>) -> Result<Arc<Self>, BindingError> {
        if self.vtable().flags & dictionary_flags::IMMUTABLE != 0 {
            return Ok(Arc::clone(self));
        }
        let callback = self
            .vtable()
            .snapshot
            .ok_or(BindingError::IncompatibleDictionaryInterface)?;
        let mut snapshot = VtResource::NULL;
        self.call(|| status(unsafe { callback(self.resource.context, &mut snapshot) }))?;
        if snapshot.is_null() {
            return Err(BindingError::InvalidProviderOutput(
                "snapshot returned a null resource",
            ));
        }
        let snapshot = match unsafe {
            Provider::from_owned_with_graph_memo(snapshot, Arc::clone(&self.graph_memo))
        } {
            Ok(provider) => provider,
            Err(error) => {
                // The provider transferred ONE owned retain with the snapshot
                // resource (VtResource contract). Validation failed before a
                // `Provider` took ownership of that retain, so release it
                // here; otherwise every failed snapshot decode leaks the
                // snapshot context (wave-W3 fault-injection finding, pinned
                // by tests/ffi_provider_fault_injection.rs ledger balance).
                // The object never entered service, so no gate guards it —
                // mirroring the `from_borrowed` error path's raw release.
                let release = unsafe { (*snapshot.vtable).release };
                if let Some(release) = release {
                    unsafe { release(snapshot.context) };
                }
                return Err(error);
            }
        };
        if snapshot.vtable().unit_domain != self.vtable().unit_domain {
            return Err(BindingError::InvalidProviderOutput(
                "snapshot changed the unit domain",
            ));
        }
        if snapshot.vtable().value_domain != self.vtable().value_domain {
            return Err(BindingError::InvalidProviderOutput(
                "snapshot changed the value domain",
            ));
        }
        Ok(snapshot)
    }

    fn root(self: &Arc<Self>) -> Result<u64, BindingError> {
        let callback = self
            .vtable()
            .root
            .ok_or(BindingError::IncompatibleDictionaryInterface)?;
        let mut root = 0;
        self.call(|| status(unsafe { callback(self.resource.context, &mut root) }))?;
        Ok(root)
    }
}

unsafe fn validate_base(resource: VtResource) -> Result<(), BindingError> {
    if resource.is_null() {
        return Err(BindingError::NullResource);
    }
    let base = &*resource.vtable;
    if base.struct_size < std::mem::size_of::<VtResourceVTable>()
        || base.abi_version != VT_ABI_VERSION
        || base.reserved != 0
        || base.retain.is_none()
        || base.release.is_none()
        || base.query_interface.is_none()
    {
        return Err(BindingError::IncompatibleResourceAbi);
    }
    Ok(())
}

fn validate_dictionary(vtable: &VtDictionaryVTable) -> Result<(), BindingError> {
    if vtable.struct_size < std::mem::size_of::<VtDictionaryVTable>()
        || vtable.interface_version < VT_DICTIONARY_INTERFACE_VERSION
        || vtable.snapshot.is_none()
        || vtable.root.is_none()
        || vtable.node_is_final.is_none()
        || vtable.node_edges.is_none()
    {
        return Err(BindingError::IncompatibleDictionaryInterface);
    }
    if vtable.value_domain == VtValueDomain::OptionalU64 && vtable.node_value_u64.is_none() {
        return Err(BindingError::IncompatibleDictionaryInterface);
    }
    Ok(())
}

fn validate_dictionary_visit(vtable: &VtDictionaryVisitVTable) -> Result<(), BindingError> {
    if vtable.struct_size < std::mem::size_of::<VtDictionaryVisitVTable>()
        || vtable.interface_version < VT_DICTIONARY_VISIT_INTERFACE_VERSION
        || vtable.reserved != 0
        || vtable.node_visit.is_none()
    {
        return Err(BindingError::IncompatibleDictionaryInterface);
    }
    Ok(())
}

fn validate_dictionary_graph(
    vtable: &VtDictionaryGraphVTable,
    value_domain: VtValueDomain,
) -> Result<(), BindingError> {
    if vtable.struct_size < std::mem::size_of::<VtDictionaryGraphVTable>()
        || vtable.interface_version < VT_DICTIONARY_GRAPH_INTERFACE_VERSION
        || vtable.reserved != 0
        || vtable.graph.is_none()
        || (value_domain == VtValueDomain::OptionalU64 && vtable.node_value_u64.is_none())
    {
        return Err(BindingError::IncompatibleDictionaryInterface);
    }
    Ok(())
}

fn validate_snapshot_identity(vtable: &VtSnapshotIdentityVTable) -> Result<(), BindingError> {
    if vtable.struct_size < std::mem::size_of::<VtSnapshotIdentityVTable>()
        || vtable.interface_version < VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION
        || vtable.reserved != 0
        || vtable.identity.is_none()
    {
        return Err(BindingError::IncompatibleDictionaryInterface);
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
struct BindingValue {
    id: Option<u64>,
}

impl DictionaryValue for BindingValue {}

trait InteropUnit: CharUnit {
    const DOMAIN: VtUnitDomain;

    fn from_abi(label: u64) -> Option<Self>;
    /// Decode an edge that was already validated before entering the immutable
    /// foreign-node cache.
    fn from_validated_abi(label: u64) -> Self;
    fn to_abi(self) -> u64;
    fn captured_graph(
        provider: &Provider,
    ) -> Option<Arc<SnapshotTraversalGraph<Self, ForeignGraphValueHandle>>>;
}

impl InteropUnit for u8 {
    const DOMAIN: VtUnitDomain = VtUnitDomain::Byte;

    fn from_abi(label: u64) -> Option<Self> {
        u8::try_from(label).ok()
    }

    #[inline(always)]
    fn from_validated_abi(label: u64) -> Self {
        label as u8
    }

    fn to_abi(self) -> u64 {
        u64::from(self)
    }

    fn captured_graph(
        provider: &Provider,
    ) -> Option<Arc<SnapshotTraversalGraph<Self, ForeignGraphValueHandle>>> {
        match provider.graph.as_ref()? {
            ForeignCapturedGraph::Byte(graph) => Some(Arc::clone(graph)),
            ForeignCapturedGraph::Unicode(_) | ForeignCapturedGraph::U64(_) => None,
        }
    }
}

impl InteropUnit for char {
    const DOMAIN: VtUnitDomain = VtUnitDomain::UnicodeScalar;

    fn from_abi(label: u64) -> Option<Self> {
        u32::try_from(label).ok().and_then(char::from_u32)
    }

    #[inline(always)]
    fn from_validated_abi(label: u64) -> Self {
        // SAFETY: `try_inspect_node` validates every raw label with
        // `from_abi` before publishing its immutable cache entry. Providers
        // and unit domains are fixed for the lifetime of that cache.
        unsafe { char::from_u32_unchecked(label as u32) }
    }

    fn to_abi(self) -> u64 {
        u64::from(u32::from(self))
    }

    fn captured_graph(
        provider: &Provider,
    ) -> Option<Arc<SnapshotTraversalGraph<Self, ForeignGraphValueHandle>>> {
        match provider.graph.as_ref()? {
            ForeignCapturedGraph::Unicode(graph) => Some(Arc::clone(graph)),
            ForeignCapturedGraph::Byte(_) | ForeignCapturedGraph::U64(_) => None,
        }
    }
}

impl InteropUnit for u64 {
    const DOMAIN: VtUnitDomain = VtUnitDomain::U64;

    fn from_abi(label: u64) -> Option<Self> {
        Some(label)
    }

    #[inline(always)]
    fn from_validated_abi(label: u64) -> Self {
        label
    }

    fn to_abi(self) -> u64 {
        self
    }

    fn captured_graph(
        provider: &Provider,
    ) -> Option<Arc<SnapshotTraversalGraph<Self, ForeignGraphValueHandle>>> {
        match provider.graph.as_ref()? {
            ForeignCapturedGraph::U64(graph) => Some(Arc::clone(graph)),
            ForeignCapturedGraph::Byte(_) | ForeignCapturedGraph::Unicode(_) => None,
        }
    }
}

fn validate_graph_slice<T>(pointer: *const T, len: usize) -> Result<(), BindingError> {
    if len == 0 {
        return Ok(());
    }
    if pointer.is_null() || !pointer.is_aligned() {
        return Err(BindingError::InvalidProviderOutput(
            "snapshot graph slice pointer was null or misaligned",
        ));
    }
    let bytes =
        len.checked_mul(std::mem::size_of::<T>())
            .ok_or(BindingError::InvalidProviderOutput(
                "snapshot graph slice length overflowed",
            ))?;
    if bytes > isize::MAX as usize {
        return Err(BindingError::InvalidProviderOutput(
            "snapshot graph slice exceeded the addressable range",
        ));
    }
    Ok(())
}

fn decode_graph<U: InteropUnit>(
    view: VtDictionaryGraphView,
) -> Result<Arc<SnapshotTraversalGraph<U, ForeignGraphValueHandle>>, BindingError> {
    crate::causal_perf::record_foreign_graph_decodes(1);
    if view.reserved != 0 || view.node_count == 0 {
        return Err(BindingError::InvalidProviderOutput(
            "snapshot graph header was invalid",
        ));
    }
    validate_graph_slice(view.nodes, view.node_count)?;
    validate_graph_slice(view.edges, view.edge_count)?;
    let root = u32::try_from(view.root)
        .map_err(|_| BindingError::InvalidProviderOutput("snapshot graph root exceeded u32"))?;
    // SAFETY: the retained provider owns both immutable slices. Their
    // pointers, alignment, and byte lengths were validated above, and the
    // provider's successful callback guarantees readable initialized POD.
    let raw_nodes = unsafe { std::slice::from_raw_parts(view.nodes, view.node_count) };
    // SAFETY: same retained-view argument as for `raw_nodes`.
    let raw_edges = unsafe { std::slice::from_raw_parts(view.edges, view.edge_count) };
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for node in raw_nodes {
        if node.reserved != [0; 7] || node.is_final > 1 {
            return Err(BindingError::InvalidProviderOutput(
                "snapshot graph node flags were invalid",
            ));
        }
        let edge_start = u32::try_from(node.edge_start).map_err(|_| {
            BindingError::InvalidProviderOutput("snapshot graph edge start exceeded u32")
        })?;
        let edge_len = u32::try_from(node.edge_len).map_err(|_| {
            BindingError::InvalidProviderOutput("snapshot graph edge length exceeded u32")
        })?;
        let value_cursor = ForeignGraphValueHandle::new(node.value_cursor).ok_or(
            BindingError::InvalidProviderOutput("snapshot graph value cursor was invalid"),
        )?;
        nodes.push(SnapshotTraversalNode::new(
            edge_start,
            edge_len,
            node.is_final == 1,
            value_cursor,
        ));
    }
    let mut edges = Vec::with_capacity(raw_edges.len());
    for edge in raw_edges {
        let label = U::from_abi(edge.label).ok_or(BindingError::InvalidProviderOutput(
            "snapshot graph label was outside its unit domain",
        ))?;
        let target = u32::try_from(edge.target).map_err(|_| {
            BindingError::InvalidProviderOutput("snapshot graph target exceeded u32")
        })?;
        edges.push(SnapshotTraversalEdge::new(label, target));
    }
    SnapshotTraversalGraph::new(nodes, edges, root)
        .map(Arc::new)
        .ok_or(BindingError::InvalidProviderOutput(
            "snapshot graph ranges, targets, or label order were invalid",
        ))
}

fn decode_captured_graph(
    domain: VtUnitDomain,
    view: VtDictionaryGraphView,
) -> Result<ForeignCapturedGraph, BindingError> {
    match domain {
        VtUnitDomain::Byte => decode_graph(view).map(ForeignCapturedGraph::Byte),
        VtUnitDomain::UnicodeScalar => decode_graph(view).map(ForeignCapturedGraph::Unicode),
        VtUnitDomain::U64 => decode_graph(view).map(ForeignCapturedGraph::U64),
    }
}

/// Non-owning pointer to the provider allocation retained by a [`QueryCursor`].
///
/// This is the owner/key split for foreign traversal: queued intersections
/// copy only this pointer and a node identifier instead of incrementing and
/// decrementing an `Arc` for every accepted dictionary edge.
#[derive(Clone, Copy)]
struct ProviderRef(NonNull<Provider>);

impl ProviderRef {
    fn new(owner: &Arc<Provider>) -> Self {
        Self(NonNull::from(owner.as_ref()))
    }
}

impl Deref for ProviderRef {
    type Target = Provider;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: `ProviderRef` is constructed only from the query-local
        // `Arc<Provider>` stored in `QueryCursor`. `QueryCursor` declares its
        // traversal field before that owner, so all node keys are destroyed
        // before the final retain is released. Arc allocations do not move.
        unsafe { self.0.as_ref() }
    }
}

// SAFETY: the pointee is `Send + Sync`, the pointer is immutable, and the
// enclosing cursor retains its allocation until every copied key is dropped.
unsafe impl Send for ProviderRef {}
unsafe impl Sync for ProviderRef {}

#[derive(Clone, Copy)]
struct ForeignNode<U: InteropUnit> {
    provider: ProviderRef,
    /// `None` is the compact-graph owner sentinel. It is never expanded as an
    /// ABI node: the captured graph supplies all traversal cursors, while this
    /// value retains the provider and resolves graph-local values.
    target: Option<ForeignTarget>,
    _unit: PhantomData<fn() -> U>,
}

// SAFETY: both non-owning pointers refer into append-only allocations retained
// by the cursor's `Arc<Provider>`. The entry's descriptor publication cell
// supplies its own synchronization and published descriptors are immutable.
unsafe impl<U: InteropUnit> Send for ForeignNode<U> {}
unsafe impl<U: InteropUnit> Sync for ForeignNode<U> {}

impl<U: InteropUnit> fmt::Debug for ForeignNode<U> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ForeignNode")
            .field("id", &self.target.map(|_| self.id()))
            .field("domain", &U::DOMAIN)
            .finish()
    }
}

impl<U: InteropUnit> ForeignNode<U> {
    fn new(provider: ProviderRef, id: u64) -> Self {
        let entry = NonNull::from(provider.node_entry(id));
        Self::from_entry(provider, entry)
    }

    fn traversal_root(
        owner: &Arc<Provider>,
    ) -> Result<DictionaryTraversalRoot<Self>, BindingError> {
        let provider = ProviderRef::new(owner);
        if let Some(graph) = U::captured_graph(owner) {
            return Ok(DictionaryTraversalRoot::captured(
                Self {
                    provider,
                    target: None,
                    _unit: PhantomData,
                },
                graph,
            ));
        }
        Ok(DictionaryTraversalRoot::owned(Self::new(
            provider,
            owner.root()?,
        )))
    }

    fn from_entry(provider: ProviderRef, entry: NonNull<CachedForeignEntry>) -> Self {
        let pending = ForeignTarget::pending(entry);
        let target = if foreign_ready_cursors_enabled() {
            pending.ready_if_published().unwrap_or(pending)
        } else {
            pending
        };
        Self::from_target(provider, target)
    }

    #[inline]
    fn from_target(provider: ProviderRef, target: ForeignTarget) -> Self {
        Self {
            provider,
            target: Some(target),
            _unit: PhantomData,
        }
    }

    #[inline]
    fn traversal_cursor(self) -> ForeignSnapshotCursor {
        let target = self
            .target
            .expect("compact-graph owners are never converted to native cursors");
        if target.is_ready() {
            crate::causal_perf::record_foreign_ready_cursors_emitted(1);
        } else {
            crate::causal_perf::record_foreign_pending_cursors_emitted(1);
        }
        ForeignSnapshotCursor(target)
    }

    /// Reconstitute a lightweight node view while the query-local provider
    /// owner retains the append-only cache entry.
    ///
    /// # Safety
    ///
    /// `cursor` must have originated from this provider and query snapshot.
    #[inline]
    unsafe fn from_traversal_cursor(provider: ProviderRef, cursor: ForeignSnapshotCursor) -> Self {
        Self::from_target(provider, cursor.0)
    }

    #[inline(always)]
    fn pending_entry(&self) -> Option<&CachedForeignEntry> {
        let entry = self.target?.pending_entry()?;
        // SAFETY: entries live in the append-only cache owned by `provider` and
        // the query cursor drops all node handles before releasing that owner.
        Some(unsafe { entry.as_ref() })
    }

    #[inline(always)]
    fn ready_node(&self) -> Option<&CachedForeignNode> {
        let node = self.target?.ready_node()?;
        // SAFETY: descriptors are immutable after publication and retained by
        // the append-only cache owned by `provider`.
        Some(unsafe { node.as_ref() })
    }

    #[inline(always)]
    fn id(&self) -> u64 {
        if let Some(node) = self.ready_node() {
            node.node
        } else {
            self.pending_entry()
                .expect("foreign target is pending or ready")
                .node
        }
    }

    #[inline]
    fn child_from_edge(&self, edge: &CachedForeignEdge) -> Self {
        let ready_cursors = foreign_ready_cursors_enabled();
        let mut encoded = edge.cached_child.load(Ordering::Acquire);
        if encoded == 0 {
            crate::causal_perf::record_foreign_child_directory_lookups(1);
            let canonical = NonNull::from(self.provider.node_entry(edge.node));
            let pending = ForeignTarget::pending(canonical);
            let candidate = if ready_cursors {
                pending.ready_if_published().unwrap_or(pending)
            } else {
                pending
            };
            match edge.cached_child.compare_exchange(
                0,
                candidate.encoded(),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                // This thread just checked descriptor publication while
                // choosing `candidate`; a second acquire load cannot improve
                // this visit and would double the cold-edge lookup work.
                Ok(_) => return Self::from_target(self.provider, candidate),
                Err(existing) => encoded = existing,
            }
        }
        let mut target = ForeignTarget::from_encoded(encoded).expect("published child target");
        if ready_cursors {
            target = promote_published_foreign_target(&edge.cached_child, target);
        }
        Self::from_target(self.provider, target)
    }

    fn callback_failed<T>(&self, error: BindingError, fallback: T) -> T {
        self.provider.record_fault(error);
        fallback
    }

    #[inline]
    fn try_filter_map_expanded_edges<T, P, F>(
        &self,
        mut project: P,
        mut visitor: F,
    ) -> Result<(), BindingError>
    where
        P: FnMut(U) -> Option<T>,
        F: FnMut(U, Self, T),
    {
        let callback = self
            .provider
            .vtable()
            .node_edges
            .ok_or(BindingError::IncompatibleDictionaryInterface)?;
        let mut start = 0usize;
        loop {
            // The ABI's recommended page is a fixed upper bound. Keeping that
            // page on the stack avoids one allocation for every visited
            // dictionary node without changing pagination or acceptance.
            let mut page = [VtDictionaryEdge::default(); VT_RECOMMENDED_EDGE_BATCH];
            let capacity = page.len();
            let mut written = 0usize;
            let mut total = 0usize;
            crate::causal_perf::record_foreign_edge_pages(1);
            crate::causal_perf::record_foreign_edge_callbacks(1);
            let callback_status = self.provider.call(|| unsafe {
                callback(
                    self.provider.resource.context,
                    self.id(),
                    start,
                    page.as_mut_ptr(),
                    capacity,
                    &mut written,
                    &mut total,
                )
            });
            status(callback_status)?;
            if written > capacity
                || written > total.saturating_sub(start)
                || (written == 0 && start < total)
            {
                return Err(BindingError::InvalidProviderOutput(
                    "invalid edge page lengths",
                ));
            }
            crate::causal_perf::record_foreign_edge_descriptors(written as u64);
            for edge in page.into_iter().take(written) {
                let label = U::from_abi(edge.label).ok_or(BindingError::InvalidProviderOutput(
                    "edge label is outside its domain",
                ))?;
                if let Some(projected) = project(label) {
                    visitor(label, Self::new(self.provider, edge.node), projected);
                }
            }
            start = start.saturating_add(written);
            if start >= total {
                return Ok(());
            }
        }
    }

    #[inline]
    fn try_for_each_expanded_edge<F>(&self, mut visitor: F) -> Result<(), BindingError>
    where
        F: FnMut(U, Self),
    {
        self.try_filter_map_expanded_edges(|_| Some(()), |label, child, ()| visitor(label, child))
    }

    /// Inspect a node without bringing the provider's large pagination frame
    /// into the overwhelmingly common ready/published-descriptor path.
    #[inline(always)]
    fn try_inspect_node(&self) -> Result<&CachedForeignNode, BindingError> {
        #[cfg(feature = "resource-profiling")]
        if monolithic_foreign_inspection_enabled() {
            return self.try_inspect_node_slow_or_monolithic(None);
        }

        if let Some(cached) = self.ready_node() {
            crate::causal_perf::record_foreign_ready_descriptor_reads(1);
            crate::causal_perf::record_foreign_node_cache_hits(1);
            return Ok(cached);
        }
        let entry = self
            .pending_entry()
            .expect("foreign target is pending or ready");
        crate::causal_perf::record_foreign_pending_descriptor_loads(1);
        if let Some(cached) = entry.descriptor.get() {
            crate::causal_perf::record_foreign_node_cache_hits(1);
            return Ok(cached);
        }
        self.try_inspect_node_slow_or_monolithic(Some(entry))
    }

    /// Populate an unpublished descriptor. In profiling builds, `None` also
    /// preserves the former monolithic ready path as a same-binary causal
    /// control. In production this function is cold and reached only for a
    /// genuine cache miss, so its 256-edge stack page and provider machinery
    /// never impose stack probes on warm traversal.
    #[cfg_attr(not(feature = "resource-profiling"), cold)]
    #[inline(never)]
    fn try_inspect_node_slow_or_monolithic<'a>(
        &'a self,
        known_miss: Option<&'a CachedForeignEntry>,
    ) -> Result<&'a CachedForeignNode, BindingError> {
        let entry = match known_miss {
            Some(entry) => entry,
            None => {
                if let Some(cached) = self.ready_node() {
                    crate::causal_perf::record_foreign_ready_descriptor_reads(1);
                    crate::causal_perf::record_foreign_node_cache_hits(1);
                    return Ok(cached);
                }
                let entry = self
                    .pending_entry()
                    .expect("foreign target is pending or ready");
                crate::causal_perf::record_foreign_pending_descriptor_loads(1);
                if let Some(cached) = entry.descriptor.get() {
                    crate::causal_perf::record_foreign_node_cache_hits(1);
                    return Ok(cached);
                }
                entry
            }
        };
        crate::causal_perf::record_foreign_node_cache_misses(1);
        let callback = self
            .provider
            .visit_vtable()
            .and_then(|vtable| vtable.node_visit)
            .ok_or(BindingError::IncompatibleDictionaryInterface)?;
        let mut start = 0usize;
        let mut observed_finality = None;
        let mut edges = Vec::new();
        loop {
            let mut page = [VtDictionaryEdge::default(); VT_RECOMMENDED_EDGE_BATCH];
            let capacity = page.len();
            let mut is_final = 0u8;
            let mut written = 0usize;
            let mut total = 0usize;
            crate::causal_perf::record_foreign_edge_pages(1);
            crate::causal_perf::record_foreign_edge_callbacks(1);
            let callback_status = self.provider.call(|| unsafe {
                callback(
                    self.provider.resource.context,
                    self.id(),
                    start,
                    &mut is_final,
                    page.as_mut_ptr(),
                    capacity,
                    &mut written,
                    &mut total,
                )
            });
            status(callback_status)?;
            let is_final = match is_final {
                0 => false,
                1 => true,
                _ => {
                    return Err(BindingError::InvalidProviderOutput(
                        "fused is_final was not zero or one",
                    ))
                }
            };
            if observed_finality
                .replace(is_final)
                .is_some_and(|old| old != is_final)
            {
                return Err(BindingError::InvalidProviderOutput(
                    "fused is_final changed between edge pages",
                ));
            }
            if written > capacity
                || written > total.saturating_sub(start)
                || (written == 0 && start < total)
            {
                return Err(BindingError::InvalidProviderOutput(
                    "invalid fused edge page lengths",
                ));
            }
            crate::causal_perf::record_foreign_edge_descriptors(written as u64);
            for edge in page.into_iter().take(written) {
                U::from_abi(edge.label).ok_or(BindingError::InvalidProviderOutput(
                    "edge label is outside its domain",
                ))?;
                edges.push(CachedForeignEdge {
                    label: edge.label,
                    node: edge.node,
                    cached_child: AtomicUsize::new(0),
                });
            }
            start = start.saturating_add(written);
            if start >= total {
                let cached = self.provider.cache_node(
                    entry,
                    CachedForeignNode {
                        node: entry.node,
                        is_final,
                        edges: edges.into_boxed_slice(),
                    },
                );
                return Ok(cached);
            }
        }
    }

    #[inline]
    fn try_filter_map_inspected_edges_and_finality<T, P, F>(
        &self,
        mut project: P,
        mut visitor: F,
    ) -> Result<bool, BindingError>
    where
        P: FnMut(U) -> Option<T>,
        F: FnMut(U, Self, T),
    {
        let cached = self.try_inspect_node()?;
        if cached.edges.is_empty() {
            crate::causal_perf::record_foreign_leaf_expansions(1);
        }
        for edge in &cached.edges {
            let label = U::from_validated_abi(edge.label);
            if let Some(projected) = project(label) {
                visitor(label, self.child_from_edge(edge), projected);
            }
        }
        Ok(cached.is_final)
    }

    #[inline]
    fn try_visit_edges_and_finality<F>(&self, mut visitor: F) -> Result<bool, BindingError>
    where
        F: FnMut(U, Self),
    {
        self.try_filter_map_inspected_edges_and_finality(
            |_| Some(()),
            |label, child, ()| visitor(label, child),
        )
    }

    fn expanded_edges(&self) -> Vec<(U, Self)> {
        // Every page requests the same recommended capacity. The result Vec
        // grows only from edges actually delivered — never from the
        // provider-claimed `total`, whose inflation would otherwise drive a
        // preallocation abort (finding LLEV-B8). The per-page acceptance
        // check below is exactly the predicate proved in
        // docs/verification/abi/theories/ConsumerAcceptance.v `accepts_dec`.
        let mut result = Vec::with_capacity(VT_RECOMMENDED_EDGE_BATCH);
        let visit = |label, child| result.push((label, child));
        let expanded = if self.provider.visit_vtable().is_some() {
            self.try_visit_edges_and_finality(visit).map(|_| ())
        } else {
            self.try_for_each_expanded_edge(visit)
        };
        if let Err(error) = expanded {
            return self.callback_failed(error, vec![]);
        }
        result
    }
}

impl<U: InteropUnit> DictionaryNode for ForeignNode<U> {
    type Unit = U;
    type SnapshotCursor = ForeignSnapshotCursor;
    type SnapshotGraphValueHandle = ForeignGraphValueHandle;

    #[inline]
    fn snapshot_root_cursor(&self) -> Option<Self::SnapshotCursor> {
        self.provider
            .visit_vtable()
            .is_some()
            .then(|| (*self).traversal_cursor())
    }

    #[inline]
    fn supports_snapshot_cursor_nodes(&self) -> bool {
        self.provider.visit_vtable().is_some()
    }

    #[inline]
    unsafe fn snapshot_cursor_node(&self, cursor: Self::SnapshotCursor) -> Option<Self> {
        self.provider.visit_vtable()?;
        // SAFETY: inherited from the DictionaryNode cursor contract.
        Some(unsafe { Self::from_traversal_cursor(self.provider, cursor) })
    }

    #[inline]
    unsafe fn filter_map_snapshot_cursor_edges_and_finality<T, P, F>(
        &self,
        cursor: Self::SnapshotCursor,
        project: P,
        mut visitor: F,
    ) -> Option<bool>
    where
        P: FnMut(Self::Unit) -> Option<T>,
        F: FnMut(Self::Unit, Self::SnapshotCursor, T),
    {
        self.provider.visit_vtable()?;
        // SAFETY: inherited from the DictionaryNode cursor contract.
        let node = unsafe { Self::from_traversal_cursor(self.provider, cursor) };
        match node
            .try_filter_map_inspected_edges_and_finality(project, |label, child, projected| {
                visitor(label, child.traversal_cursor(), projected)
            }) {
            Ok(is_final) => Some(is_final),
            Err(error) => Some(node.callback_failed(error, false)),
        }
    }

    fn is_final(&self) -> bool {
        if self.provider.visit_vtable().is_some() {
            return match self.try_inspect_node() {
                Ok(cached) => cached.is_final,
                Err(error) => self.callback_failed(error, false),
            };
        }
        let callback = match self.provider.vtable().node_is_final {
            Some(callback) => callback,
            None => {
                return self.callback_failed(BindingError::IncompatibleDictionaryInterface, false)
            }
        };
        crate::causal_perf::record_foreign_is_final_callbacks(1);
        let mut final_node = 0u8;
        let callback_status = self.provider.call(|| unsafe {
            callback(self.provider.resource.context, self.id(), &mut final_node)
        });
        if let Err(error) = status(callback_status) {
            return self.callback_failed(error, false);
        }
        match final_node {
            0 => false,
            1 => true,
            _ => self.callback_failed(
                BindingError::InvalidProviderOutput("is_final was not zero or one"),
                false,
            ),
        }
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        let callback = match self.provider.vtable().node_transition {
            Some(callback) => callback,
            None => {
                return self
                    .expanded_edges()
                    .into_iter()
                    .find_map(|(edge, node)| (edge == label).then_some(node))
            }
        };
        let mut child = 0u64;
        let mut found = 0u8;
        let callback_status = self.provider.call(|| unsafe {
            callback(
                self.provider.resource.context,
                self.id(),
                label.to_abi(),
                &mut child,
                &mut found,
            )
        });
        if let Err(error) = status(callback_status) {
            return self.callback_failed(error, None);
        }
        match found {
            0 => None,
            1 => Some(Self::new(self.provider, child)),
            _ => self.callback_failed(
                BindingError::InvalidProviderOutput("transition found was not zero or one"),
                None,
            ),
        }
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        Box::new(self.expanded_edges().into_iter())
    }

    #[inline]
    fn for_each_edge<F>(&self, visitor: F)
    where
        F: FnMut(Self::Unit, Self),
    {
        let result = if self.provider.visit_vtable().is_some() {
            self.try_visit_edges_and_finality(visitor).map(|_| ())
        } else {
            self.try_for_each_expanded_edge(visitor)
        };
        if let Err(error) = result {
            self.provider.record_fault(error);
        }
    }

    #[inline]
    fn filter_map_edges<T, P, F>(&self, project: P, visitor: F)
    where
        P: FnMut(Self::Unit) -> Option<T>,
        F: FnMut(Self::Unit, Self, T),
    {
        let result = if self.provider.visit_vtable().is_some() {
            self.try_filter_map_inspected_edges_and_finality(project, visitor)
                .map(|_| ())
        } else {
            self.try_filter_map_expanded_edges(project, visitor)
        };
        if let Err(error) = result {
            self.provider.record_fault(error);
        }
    }

    #[inline]
    fn visit_edges_and_finality<F>(&self, visitor: F) -> bool
    where
        F: FnMut(Self::Unit, Self),
    {
        if self.provider.visit_vtable().is_none() {
            let is_final = self.is_final();
            self.for_each_edge(visitor);
            return is_final;
        }
        match self.try_visit_edges_and_finality(visitor) {
            Ok(is_final) => is_final,
            Err(error) => self.callback_failed(error, false),
        }
    }

    #[inline]
    fn filter_map_edges_and_finality<T, P, F>(&self, project: P, visitor: F) -> bool
    where
        P: FnMut(Self::Unit) -> Option<T>,
        F: FnMut(Self::Unit, Self, T),
    {
        if self.provider.visit_vtable().is_none() {
            let is_final = self.is_final();
            self.filter_map_edges(project, visitor);
            return is_final;
        }
        match self.try_filter_map_inspected_edges_and_finality(project, visitor) {
            Ok(is_final) => is_final,
            Err(error) => self.callback_failed(error, false),
        }
    }

    fn edge_count(&self) -> Option<usize> {
        None
    }
}

impl<U: InteropUnit> MappedDictionaryNode for ForeignNode<U> {
    type Value = BindingValue;

    fn value(&self) -> Option<Self::Value> {
        if !self.is_final() {
            return None;
        }
        self.value_at_final()
    }

    fn value_at_final(&self) -> Option<Self::Value> {
        if self.provider.vtable().value_domain == VtValueDomain::Unit {
            return Some(BindingValue { id: None });
        }
        let callback = match self.provider.vtable().node_value_u64 {
            Some(callback) => callback,
            None => {
                return self.callback_failed(BindingError::IncompatibleDictionaryInterface, None)
            }
        };
        let mut value = VtOptionalU64::default();
        let callback_status = self
            .provider
            .call(|| unsafe { callback(self.provider.resource.context, self.id(), &mut value) });
        if let Err(error) = status(callback_status) {
            return self.callback_failed(error, None);
        }
        // VT-ABI-5: reserved bytes are part of the ABI contract and must be
        // zero — a provider writing garbage there would be silently
        // reinterpreted by a future interface revision (ledger LLEV-B7).
        if value.reserved != [0; 7] {
            return self.callback_failed(
                BindingError::InvalidProviderOutput("reserved bytes were not zero"),
                None,
            );
        }
        let id = match value.has_value {
            0 => None,
            1 => Some(value.value),
            _ => {
                return self.callback_failed(
                    BindingError::InvalidProviderOutput("has_value was not zero or one"),
                    None,
                )
            }
        };
        Some(BindingValue { id })
    }

    #[inline]
    fn supports_snapshot_cursor_values(&self) -> bool {
        self.provider.visit_vtable().is_some()
    }

    #[inline]
    fn supports_snapshot_graph_values(&self) -> bool {
        U::captured_graph(&self.provider).is_some() && self.provider.graph_vtable.is_some()
    }

    #[inline]
    unsafe fn snapshot_cursor_value(
        &self,
        cursor: Self::SnapshotCursor,
    ) -> Option<Option<Self::Value>> {
        self.provider.visit_vtable()?;
        // SAFETY: inherited from the MappedDictionaryNode cursor contract.
        let node = unsafe { Self::from_traversal_cursor(self.provider, cursor) };
        let is_final = match node.try_inspect_node() {
            Ok(cached) => cached.is_final,
            Err(error) => return Some(node.callback_failed(error, None)),
        };
        Some(is_final.then(|| node.value_at_final()).flatten())
    }

    #[inline]
    unsafe fn snapshot_graph_cursor_value(
        &self,
        graph: &SnapshotTraversalGraph<U, Self::SnapshotGraphValueHandle>,
        cursor: SnapshotTraversalCursor,
    ) -> Option<Option<Self::Value>> {
        let graph_vtable = self.provider.graph_vtable?;
        if self.provider.vtable().value_domain == VtValueDomain::Unit {
            return Some(Some(BindingValue { id: None }));
        }
        // SAFETY: the interface was validated while constructing the retained
        // provider and remains immutable for that provider's lifetime.
        let callback = unsafe { (*graph_vtable).node_value_u64 }?;
        let value_cursor = graph.value_handle(cursor).get();
        let mut value = VtOptionalU64::default();
        let callback_status = self
            .provider
            .call(|| unsafe { callback(self.provider.resource.context, value_cursor, &mut value) });
        if let Err(error) = status(callback_status) {
            return Some(self.callback_failed(error, None));
        }
        if value.reserved != [0; 7] {
            return Some(self.callback_failed(
                BindingError::InvalidProviderOutput("graph value reserved bytes were not zero"),
                None,
            ));
        }
        let id = match value.has_value {
            0 => None,
            1 => Some(value.value),
            _ => {
                return Some(self.callback_failed(
                    BindingError::InvalidProviderOutput("graph value presence was not zero or one"),
                    None,
                ))
            }
        };
        Some(Some(BindingValue { id }))
    }
}

#[derive(Clone, Debug)]
enum ForeignDictionary {
    Byte(Arc<Provider>),
    Unicode(Arc<Provider>),
    U64(Arc<Provider>),
}

impl ForeignDictionary {
    unsafe fn from_resource(resource: VtResource) -> Result<Self, BindingError> {
        let provider = Provider::from_borrowed(resource)?;
        match provider.vtable().unit_domain {
            VtUnitDomain::Byte => Ok(Self::Byte(provider)),
            VtUnitDomain::UnicodeScalar => Ok(Self::Unicode(provider)),
            VtUnitDomain::U64 => Ok(Self::U64(provider)),
        }
    }

    fn unit_domain(&self) -> VtUnitDomain {
        match self {
            Self::Byte(_) => VtUnitDomain::Byte,
            Self::Unicode(_) => VtUnitDomain::UnicodeScalar,
            Self::U64(_) => VtUnitDomain::U64,
        }
    }

    fn snapshot(&self) -> Result<Self, BindingError> {
        fn immutable_or_snapshot(provider: &Arc<Provider>) -> Result<Arc<Provider>, BindingError> {
            if provider.vtable().flags & dictionary_flags::IMMUTABLE != 0 {
                Ok(Arc::clone(provider))
            } else {
                provider.snapshot()
            }
        }

        match self {
            Self::Byte(provider) => Ok(Self::Byte(immutable_or_snapshot(provider)?)),
            Self::Unicode(provider) => Ok(Self::Unicode(immutable_or_snapshot(provider)?)),
            Self::U64(provider) => Ok(Self::U64(immutable_or_snapshot(provider)?)),
        }
    }
}

/// Result ordering for a lazy query cursor.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueryOrder {
    /// Dictionary traversal order with bounded reusable batch storage.
    #[default]
    Traversal = 0,
    /// Increasing distance and then term, buffering at most one distance layer.
    DistanceThenTerm = 1,
}

/// Immutable options captured when a cursor is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryOptions {
    /// Maximum accepted edit distance.
    pub max_distance: usize,
    /// Edit-distance algorithm.
    pub algorithm: Algorithm,
    /// Result order.
    pub order: QueryOrder,
}

impl QueryOptions {
    /// Construct traversal-order options.
    pub const fn new(max_distance: usize, algorithm: Algorithm) -> Self {
        Self {
            max_distance,
            algorithm,
            order: QueryOrder::Traversal,
        }
    }

    /// Select the result order.
    pub const fn with_order(mut self, order: QueryOrder) -> Self {
        self.order = order;
        self
    }
}

/// Term representation returned by a resource query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatchTerm {
    /// Valid UTF-8 from a Unicode-scalar provider.
    Utf8(String),
    /// Raw bytes from a byte provider.
    Bytes(Vec<u8>),
    /// Raw token identifiers from a u64 provider.
    U64(Vec<u64>),
}

/// One fuzzy match. Cursors produce these lazily; this is not a result-set API.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Match {
    /// Matched term in its native unit domain.
    pub term: MatchTerm,
    /// Exact edit distance.
    pub distance: usize,
    /// Optional provider value.
    pub id: Option<u64>,
}

/// Reusable safe-Rust batch buffer.
#[derive(Debug, Default)]
pub struct MatchBatch {
    matches: Vec<Match>,
}

impl MatchBatch {
    /// Borrow the matches written by the last cursor advance.
    pub fn as_slice(&self) -> &[Match] {
        &self.matches
    }

    /// Number of matches in this batch.
    pub fn len(&self) -> usize {
        self.matches.len()
    }

    /// Return whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    fn clear(&mut self) {
        self.matches.clear();
    }
}

type CharTraversal = ValueYieldingQueryIterator<ForeignNode<char>>;
type ByteTraversal = ValueYieldingQueryIterator<ForeignNode<u8>, Vec<u8>>;
type U64Traversal = ValueYieldingQueryIterator<ForeignNode<u64>, Vec<u64>>;
type CharScorer = fn(&str, usize, &BindingValue) -> f64;
type CharOrdered = RankedValueQueryIterator<ForeignNode<char>, CharScorer>;
#[cfg(feature = "bindings-phonetic")]
type CharLanguage = MappedLanguageQueryIterator<ForeignNode<char>, NFAChar>;

fn term_only_score(_term: &str, _distance: usize, _value: &BindingValue) -> f64 {
    0.0
}

// Query cursors are already uniquely owned and frequently created. Keeping
// every variant inline avoids one heap allocation per query; the larger enum
// is an intentional hot-path space/time tradeoff.
#[allow(clippy::large_enum_variant)]
enum CursorInner {
    CharTraversal(CharTraversal),
    ByteTraversal(ByteTraversal),
    U64Traversal(U64Traversal),
    CharOrdered(CharOrdered),
    #[cfg(feature = "bindings-phonetic")]
    CharLanguage(CharLanguage),
}

/// Lazy query cursor retaining the exact provider snapshot captured at query
/// start. It may outlive the transducer and source dictionary resources.
pub struct QueryCursor {
    // Rust drops fields in declaration order. The traversal (and every
    // non-owning ProviderRef it contains) must therefore precede `provider`.
    inner: CursorInner,
    provider: Arc<Provider>,
}

impl fmt::Debug for QueryCursor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueryCursor")
            .field("unit_domain", &self.provider.vtable().unit_domain)
            .finish_non_exhaustive()
    }
}

impl QueryCursor {
    /// Process-local producer/revision identity when the provider advertises it.
    pub fn snapshot_identity(&self) -> Option<(u64, u64)> {
        self.provider
            .identity
            .map(|identity| (identity.producer, identity.revision))
    }

    /// Whether two cursors share the same identity-keyed traversal storage.
    ///
    /// Compact-graph providers share their validated immutable graph and do
    /// not allocate a legacy foreign-node cache. Graphless providers retain
    /// the identity-keyed node-cache behavior represented by this method's
    /// historical name.
    #[cfg(feature = "perf-instrumentation")]
    pub fn shares_node_cache_with(&self, other: &Self) -> bool {
        match (
            &self.provider.node_cache,
            &other.provider.node_cache,
            &self.provider.graph,
            &other.provider.graph,
        ) {
            (Some(left), Some(right), _, _) => Arc::ptr_eq(left, right),
            (None, None, Some(left), Some(right)) => left.ptr_eq(right),
            _ => false,
        }
    }

    fn next_match(&mut self) -> Result<Option<Match>, BindingError> {
        if let Some(error) = self.provider.take_fault() {
            return Err(error);
        }
        let next = match &mut self.inner {
            CursorInner::CharTraversal(cursor) => {
                cursor.next().map(|(term, distance, value)| Match {
                    term: MatchTerm::Utf8(term),
                    distance,
                    id: value.id,
                })
            }
            CursorInner::ByteTraversal(cursor) => {
                cursor.next().map(|(term, distance, value)| Match {
                    term: MatchTerm::Bytes(term),
                    distance,
                    id: value.id,
                })
            }
            CursorInner::U64Traversal(cursor) => {
                cursor.next().map(|(term, distance, value)| Match {
                    term: MatchTerm::U64(term),
                    distance,
                    id: value.id,
                })
            }
            CursorInner::CharOrdered(cursor) => cursor.next().map(
                |Suggestion {
                     term,
                     distance,
                     value,
                     ..
                 }| Match {
                    term: MatchTerm::Utf8(term),
                    distance,
                    id: value.id,
                },
            ),
            #[cfg(feature = "bindings-phonetic")]
            CursorInner::CharLanguage(cursor) => cursor.next().map(|item| Match {
                term: MatchTerm::Utf8(item.units.into_iter().collect()),
                distance: usize::from(item.distance),
                id: item.value.and_then(|value| value.id),
            }),
        };
        if let Some(error) = self.provider.take_fault() {
            return Err(error);
        }
        Ok(next)
    }

    /// Fill reusable storage with at most `max_matches` lazy results.
    pub fn next_batch(
        &mut self,
        batch: &mut MatchBatch,
        max_matches: usize,
    ) -> Result<usize, BindingError> {
        if max_matches == 0 {
            return Err(BindingError::EmptyBatch);
        }
        batch.clear();
        while batch.matches.len() < max_matches {
            match self.next_match()? {
                Some(item) => batch.matches.push(item),
                None => break,
            }
        }
        Ok(batch.len())
    }
}

#[cfg(test)]
mod compact_foreign_handle_tests {
    use super::*;
    #[cfg(feature = "resource-profiling")]
    use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};

    #[test]
    fn foreign_node_is_exactly_an_owner_pointer_and_entry_pointer() {
        assert_eq!(
            std::mem::size_of::<ForeignNode<char>>(),
            2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn foreign_targets_round_trip_pending_and_ready_pointers() {
        let entry = Box::new(CachedForeignEntry::new(42));
        let node = Box::new(CachedForeignNode {
            node: 42,
            is_final: true,
            edges: Box::new([]),
        });

        let pending = ForeignTarget::pending(NonNull::from(entry.as_ref()));
        assert!(!pending.is_ready());
        assert_eq!(
            unsafe { pending.pending_entry().unwrap().as_ref() }.node,
            42
        );
        assert!(pending.ready_node().is_none());
        assert_eq!(
            ForeignTarget::from_encoded(pending.encoded()),
            Some(pending)
        );

        let ready = ForeignTarget::ready(NonNull::from(node.as_ref()));
        assert!(ready.is_ready());
        assert!(ready.pending_entry().is_none());
        assert_eq!(unsafe { ready.ready_node().unwrap().as_ref() }.node, 42);
        assert_eq!(ForeignTarget::from_encoded(ready.encoded()), Some(ready));
    }

    #[test]
    fn published_foreign_target_promotion_is_monotonic() {
        let entry = Box::new(CachedForeignEntry::new(7));
        let pending = ForeignTarget::pending(NonNull::from(entry.as_ref()));
        let published = entry.descriptor.publish_if_absent(CachedForeignNode {
            node: 7,
            is_final: false,
            edges: Box::new([]),
        });
        let slot = AtomicUsize::new(pending.encoded());

        let ready = promote_published_foreign_target(&slot, pending);
        assert!(ready.is_ready());
        assert_eq!(
            unsafe { ready.ready_node().unwrap().as_ref() } as *const CachedForeignNode,
            published as *const CachedForeignNode
        );
        assert_eq!(slot.load(Ordering::Acquire), ready.encoded());
        assert_eq!(promote_published_foreign_target(&slot, ready), ready);
    }

    #[test]
    fn concurrent_foreign_target_promoters_converge_on_one_ready_descriptor() {
        let entry = Box::new(CachedForeignEntry::new(99));
        let pending = ForeignTarget::pending(NonNull::from(entry.as_ref()));
        let published = entry.descriptor.publish_if_absent(CachedForeignNode {
            node: 99,
            is_final: true,
            edges: Box::new([]),
        });
        let slot = AtomicUsize::new(pending.encoded());

        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        let observed = ForeignTarget::from_encoded(slot.load(Ordering::Acquire))
                            .expect("slot is initialized");
                        promote_published_foreign_target(&slot, observed)
                    })
                })
                .collect();
            for handle in handles {
                let target = handle.join().expect("promoter thread");
                assert!(target.is_ready());
                assert_eq!(
                    unsafe { target.ready_node().unwrap().as_ref() } as *const CachedForeignNode,
                    published as *const CachedForeignNode
                );
            }
        });

        let final_target = ForeignTarget::from_encoded(slot.load(Ordering::Acquire)).unwrap();
        assert!(final_target.is_ready());
    }

    #[cfg(feature = "resource-profiling")]
    #[test]
    fn forked_query_owners_have_independent_fault_mailboxes() {
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
        let resource = dictionary.resource();
        let transducer = unsafe {
            ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard)
                .expect("libdictenstein resource")
        };
        let ForeignDictionary::Unicode(provider) = &transducer.dictionary else {
            panic!("Unicode dictionary must retain a Unicode provider");
        };
        let first = provider.fork_query_owner();
        let second = provider.fork_query_owner();

        first.record_fault(BindingError::Provider(VtStatus::Closed));
        second.record_fault(BindingError::Provider(VtStatus::IoError));

        assert_eq!(
            first.take_fault(),
            Some(BindingError::Provider(VtStatus::Closed))
        );
        assert_eq!(
            second.take_fault(),
            Some(BindingError::Provider(VtStatus::IoError))
        );
        assert_eq!(first.take_fault(), None);
        assert_eq!(second.take_fault(), None);
    }
}

#[cfg(test)]
mod snapshot_graph_decode_tests {
    use super::*;

    fn node(
        edge_start: u64,
        edge_len: u64,
        value_cursor: u64,
        is_final: u8,
    ) -> VtDictionaryGraphNode {
        VtDictionaryGraphNode {
            edge_start,
            edge_len,
            value_cursor,
            is_final,
            reserved: [0; 7],
        }
    }

    fn edge(label: u64, target: u64) -> VtDictionaryGraphEdge {
        VtDictionaryGraphEdge { label, target }
    }

    fn view(
        nodes: &[VtDictionaryGraphNode],
        edges: &[VtDictionaryGraphEdge],
        root: u64,
    ) -> VtDictionaryGraphView {
        VtDictionaryGraphView {
            nodes: nodes.as_ptr(),
            node_count: nodes.len(),
            edges: edges.as_ptr(),
            edge_count: edges.len(),
            root,
            reserved: 0,
        }
    }

    fn fixture() -> ([VtDictionaryGraphNode; 3], [VtDictionaryGraphEdge; 2]) {
        (
            [node(0, 2, 11, 0), node(2, 0, 12, 1), node(2, 0, 13, 1)],
            [edge(u64::from(b'a'), 1), edge(u64::from(b'z'), 2)],
        )
    }

    fn assert_invalid<U: InteropUnit>(view: VtDictionaryGraphView) {
        assert!(matches!(
            decode_graph::<U>(view),
            Err(BindingError::InvalidProviderOutput(_))
        ));
    }

    #[test]
    fn valid_views_decode_every_unit_domain() {
        let (mut nodes, byte_edges) = fixture();
        nodes[2].value_cursor = u64::MAX;
        let byte = decode_graph::<u8>(view(&nodes, &byte_edges, 0)).expect("byte graph");
        assert_eq!(byte.node_count(), 3);
        assert_eq!(byte.root_cursor().get(), 1);
        let last = SnapshotTraversalCursor::new(3).expect("three is non-zero");
        assert_eq!(byte.value_handle(last).get(), u64::MAX);

        let char_edges = [edge(u64::from('a'), 1), edge(u64::from('\u{10ffff}'), 2)];
        let unicode = decode_graph::<char>(view(&nodes, &char_edges, 0)).expect("Unicode graph");
        assert_eq!(unicode.node_count(), 3);

        let token_edges = [edge(1u64 << 40, 1), edge(u64::MAX, 2)];
        let tokens = decode_graph::<u64>(view(&nodes, &token_edges, 0)).expect("u64 graph");
        assert_eq!(tokens.node_count(), 3);
    }

    #[test]
    fn graph_header_and_slice_bounds_are_validated_before_dereference() {
        let (nodes, edges) = fixture();

        let mut malformed = view(&nodes, &edges, 0);
        malformed.reserved = 1;
        assert_invalid::<u8>(malformed);

        malformed = view(&nodes, &edges, 0);
        malformed.node_count = 0;
        assert_invalid::<u8>(malformed);

        malformed = view(&nodes, &edges, 0);
        malformed.nodes = std::ptr::null();
        assert_invalid::<u8>(malformed);

        malformed = view(&nodes, &edges, 0);
        malformed.nodes = std::ptr::without_provenance(1);
        assert_invalid::<u8>(malformed);

        malformed = view(&nodes, &edges, 0);
        malformed.node_count = usize::MAX;
        assert_invalid::<u8>(malformed);

        malformed = view(&nodes, &edges, 0);
        malformed.edge_count = usize::MAX;
        assert_invalid::<u8>(malformed);

        malformed = view(&nodes, &edges, u64::MAX);
        assert_invalid::<u8>(malformed);
    }

    #[test]
    fn node_flags_ranges_value_cursors_and_root_are_validated() {
        let (mut nodes, edges) = fixture();
        nodes[0].reserved[3] = 1;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (mut nodes, edges) = fixture();
        nodes[1].is_final = 2;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (mut nodes, edges) = fixture();
        nodes[2].value_cursor = 0;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (mut nodes, edges) = fixture();
        nodes[0].edge_start = u64::MAX;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (mut nodes, edges) = fixture();
        nodes[0].edge_len = u64::MAX;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (nodes, edges) = fixture();
        assert_invalid::<u8>(view(&nodes, &edges, nodes.len() as u64));
    }

    #[test]
    fn edge_targets_labels_and_per_node_order_are_validated() {
        let (nodes, mut edges) = fixture();
        edges[0].target = u64::MAX;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (nodes, mut edges) = fixture();
        edges[0].target = nodes.len() as u64;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (nodes, mut edges) = fixture();
        edges[0].label = 256;
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (nodes, _) = fixture();
        let surrogate_edges = [edge(0xd800, 1), edge(0xe000, 2)];
        assert_invalid::<char>(view(&nodes, &surrogate_edges, 0));

        let (nodes, mut edges) = fixture();
        edges.swap(0, 1);
        assert_invalid::<u8>(view(&nodes, &edges, 0));

        let (nodes, mut edges) = fixture();
        edges[1].label = edges[0].label;
        assert_invalid::<u8>(view(&nodes, &edges, 0));
    }
}

/// A Levenshtein automaton configuration retaining a live dictionary resource.
///
/// Constructing this object is O(1): the resource is retained, not serialized.
/// Each query invokes the provider's O(1) `snapshot` callback before any node is
/// read, which is the binding-level query-start snapshot boundary. The first
/// compact-graph negotiation for one immutable revision may validate and import
/// that graph in O(nodes + edges); later queries share the memoized graph.
#[derive(Clone, Debug)]
pub struct ResourceTransducer {
    dictionary: ForeignDictionary,
    algorithm: Algorithm,
}

impl ResourceTransducer {
    /// Retain a dictionary resource supplied by libdictenstein or a custom host
    /// provider.
    ///
    /// # Safety
    ///
    /// `resource` must obey the vinary-tree interop retain/release contract. Its
    /// vtables and callbacks must remain valid until the final release.
    pub unsafe fn from_resource(
        resource: VtResource,
        algorithm: Algorithm,
    ) -> Result<Self, BindingError> {
        Ok(Self {
            dictionary: ForeignDictionary::from_resource(resource)?,
            algorithm,
        })
    }

    /// Unit domain required by this transducer's query entry point.
    pub fn unit_domain(&self) -> VtUnitDomain {
        self.dictionary.unit_domain()
    }

    /// Capture one immutable dictionary revision for reuse across queries.
    ///
    /// The returned transducer deliberately does not observe later mutations
    /// to the source resource. Its preferred compact traversal graph, or its
    /// fallback lazy node arena and validated edge pages, is shared by every
    /// cursor. The provider revision capture is O(1). A cold optional graph is
    /// validated and imported once in O(nodes + edges), outside provider locks;
    /// calling this on an already imported immutable revision is O(1).
    pub fn snapshot(&self) -> Result<Self, BindingError> {
        Ok(Self {
            dictionary: self.dictionary.snapshot()?,
            algorithm: self.algorithm,
        })
    }

    /// Start a lazy Unicode query over the revision visible now.
    pub fn query_utf8(
        &self,
        query: &str,
        max_distance: usize,
        order: QueryOrder,
    ) -> Result<QueryCursor, BindingError> {
        let ForeignDictionary::Unicode(provider) = &self.dictionary else {
            return Err(BindingError::UnitDomainMismatch {
                expected: VtUnitDomain::UnicodeScalar,
                actual: self.unit_domain(),
            });
        };
        let snapshot = provider.snapshot()?;
        let owner = snapshot.fork_query_owner();
        let inner = match order {
            QueryOrder::Traversal => {
                CursorInner::CharTraversal(ValueYieldingQueryIterator::with_traversal_root(
                    ForeignNode::<char>::traversal_root(&owner)?,
                    query.to_owned(),
                    max_distance,
                    self.algorithm,
                ))
            }
            QueryOrder::DistanceThenTerm => {
                CursorInner::CharOrdered(RankedValueQueryIterator::with_traversal_root(
                    ForeignNode::<char>::traversal_root(&owner)?,
                    query.to_owned(),
                    max_distance,
                    self.algorithm,
                    term_only_score as CharScorer,
                ))
            }
        };
        Ok(QueryCursor {
            inner,
            provider: owner,
        })
    }

    /// Start a lazy raw-byte query over the revision visible now.
    pub fn query_bytes(
        &self,
        query: &[u8],
        max_distance: usize,
        order: QueryOrder,
    ) -> Result<QueryCursor, BindingError> {
        let ForeignDictionary::Byte(provider) = &self.dictionary else {
            return Err(BindingError::UnitDomainMismatch {
                expected: VtUnitDomain::Byte,
                actual: self.unit_domain(),
            });
        };
        if order != QueryOrder::Traversal {
            return Err(BindingError::UnsupportedOrdering(VtUnitDomain::Byte));
        }
        let snapshot = provider.snapshot()?;
        let owner = snapshot.fork_query_owner();
        Ok(QueryCursor {
            inner: CursorInner::ByteTraversal(
                ValueYieldingQueryIterator::with_unit_query_traversal_root(
                    ForeignNode::<u8>::traversal_root(&owner)?,
                    query.to_vec(),
                    max_distance,
                    self.algorithm,
                ),
            ),
            provider: owner,
        })
    }

    /// Start a lazy u64-token query over the revision visible now.
    pub fn query_u64(
        &self,
        query: &[u64],
        max_distance: usize,
        order: QueryOrder,
    ) -> Result<QueryCursor, BindingError> {
        let ForeignDictionary::U64(provider) = &self.dictionary else {
            return Err(BindingError::UnitDomainMismatch {
                expected: VtUnitDomain::U64,
                actual: self.unit_domain(),
            });
        };
        if order != QueryOrder::Traversal {
            return Err(BindingError::UnsupportedOrdering(VtUnitDomain::U64));
        }
        let snapshot = provider.snapshot()?;
        let owner = snapshot.fork_query_owner();
        Ok(QueryCursor {
            inner: CursorInner::U64Traversal(
                ValueYieldingQueryIterator::with_unit_query_traversal_root(
                    ForeignNode::<u64>::traversal_root(&owner)?,
                    query.to_vec(),
                    max_distance,
                    self.algorithm,
                ),
            ),
            provider: owner,
        })
    }

    /// Start a lazy Unicode dictionary × phonetic-language product query.
    #[cfg(feature = "bindings-phonetic")]
    pub fn query_pattern(
        &self,
        pattern: &PhoneticPattern,
        max_distance: u8,
    ) -> Result<QueryCursor, BindingError> {
        let ForeignDictionary::Unicode(provider) = &self.dictionary else {
            return Err(BindingError::UnitDomainMismatch {
                expected: VtUnitDomain::UnicodeScalar,
                actual: self.unit_domain(),
            });
        };
        let snapshot = provider.snapshot()?;
        let owner = snapshot.fork_query_owner();
        Ok(QueryCursor {
            inner: CursorInner::CharLanguage(MappedLanguageQueryIterator::from_traversal_root(
                ForeignNode::<char>::traversal_root(&owner)?,
                LanguageProduct::new(pattern.nfa.clone(), max_distance),
            )),
            provider: owner,
        })
    }
}

/// Parse or compile failure from the phonetic binding surface.
#[cfg(feature = "bindings-phonetic")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhoneticBindingError {
    message: String,
}

#[cfg(feature = "bindings-phonetic")]
impl fmt::Display for PhoneticBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[cfg(feature = "bindings-phonetic")]
impl std::error::Error for PhoneticBindingError {}

#[cfg(feature = "bindings-phonetic")]
fn phonetic_error(error: impl fmt::Display) -> PhoneticBindingError {
    PhoneticBindingError {
        message: error.to_string(),
    }
}

/// Reusable Unicode phonetic regular-language automaton.
#[cfg(feature = "bindings-phonetic")]
#[derive(Clone, Debug)]
pub struct PhoneticPattern {
    nfa: NFAChar,
}

#[cfg(feature = "bindings-phonetic")]
impl PhoneticPattern {
    /// Compile a phonetic regular expression under the public state ceiling.
    pub fn from_regex(pattern: &str) -> Result<Self, PhoneticBindingError> {
        use crate::transducer::language::LANGUAGE_PRODUCT_MAX_STATES;

        let source_bound = pattern.chars().count().saturating_mul(2);
        if source_bound > LANGUAGE_PRODUCT_MAX_STATES {
            return Err(PhoneticBindingError {
                message: format!(
                    "pattern requires at least {source_bound} states; maximum is {LANGUAGE_PRODUCT_MAX_STATES}"
                ),
            });
        }
        let regex = crate::phonetic::regex::parse(pattern).map_err(phonetic_error)?;
        let estimated =
            crate::phonetic::nfa::estimate_thompson_states(&regex).map_err(phonetic_error)?;
        if estimated > LANGUAGE_PRODUCT_MAX_STATES {
            return Err(PhoneticBindingError {
                message: format!(
                    "pattern requires {estimated} states; maximum is {LANGUAGE_PRODUCT_MAX_STATES}"
                ),
            });
        }
        let nfa = crate::phonetic::nfa::compile(&regex).map_err(phonetic_error)?;
        Ok(Self { nfa })
    }

    /// Parse and compile an import-free `.llre` document.
    pub fn from_llre(source: &str) -> Result<Self, PhoneticBindingError> {
        use crate::phonetic::llre::{compile_with_options, parse_str, CompileOptions};
        use crate::transducer::language::LANGUAGE_PRODUCT_MAX_STATES;

        let file = parse_str(source).map_err(phonetic_error)?;
        if !file.imports.is_empty() {
            return Err(PhoneticBindingError {
                message: "binding LLRE compilation does not resolve imports".into(),
            });
        }
        let compiled = compile_with_options(
            &file,
            &CompileOptions {
                max_states: Some(LANGUAGE_PRODUCT_MAX_STATES),
                use_trampolining: true,
                optimize: true,
            },
        )
        .map_err(phonetic_error)?;
        Ok(Self { nfa: compiled.nfa })
    }

    /// Test complete-string acceptance.
    pub fn matches(&self, input: &str) -> bool {
        self.nfa.accepts(input)
    }

    /// Number of NFA states.
    pub fn state_count(&self) -> usize {
        self.nfa.num_states()
    }

    /// Number of NFA transitions.
    pub fn transition_count(&self) -> usize {
        self.nfa.num_transitions()
    }
}

/// Reusable Unicode `.llev` rewrite-rule set.
#[cfg(feature = "bindings-phonetic")]
#[derive(Clone, Debug)]
pub struct PhoneticRuleSet {
    inner: crate::phonetic::llev::RuleSetChar,
}

#[cfg(feature = "bindings-phonetic")]
impl PhoneticRuleSet {
    /// Parse an import-free rewrite-rule document.
    pub fn parse(source: &str) -> Result<Self, PhoneticBindingError> {
        let file = crate::phonetic::llev::parse_str(source).map_err(phonetic_error)?;
        if file.has_includes() {
            return Err(PhoneticBindingError {
                message: "binding LLEV parsing does not resolve includes".into(),
            });
        }
        crate::phonetic::llev::RuleSetChar::from_llev(&file)
            .map(|inner| Self { inner })
            .map_err(phonetic_error)
    }

    /// Built-in English orthography normalization rules.
    pub fn english_orthography() -> Self {
        Self {
            inner: crate::phonetic::llev::RuleSetChar {
                rules: crate::phonetic::orthography_rules_char(),
                name: Some("English orthography".into()),
                version: None,
            },
        }
    }

    /// Built-in English phonetic transformation rules.
    pub fn english_phonetic() -> Self {
        Self {
            inner: crate::phonetic::llev::RuleSetChar {
                rules: crate::phonetic::phonetic_rules_char(),
                name: Some("English phonetic".into()),
                version: None,
            },
        }
    }

    /// Apply rules to a fixed point using the native fuel bound.
    pub fn apply(&self, input: &str) -> String {
        self.inner.apply(input)
    }

    /// Number of enabled rules.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return whether no rules are enabled.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
