#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use vinary_tree_interop::{
    dictionary_flags, VtDictionaryEdge, VtDictionaryVTable, VtInterfaceId, VtOptionalU64,
    VtResource, VtResourceVTable, VtStatus, VtUnitDomain, VtValueDomain, VT_ABI_VERSION,
    VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION,
};

#[derive(Clone, Default)]
struct Node {
    edges: BTreeMap<char, u64>,
    // Outer option distinguishes a final term without an ID from a non-final node.
    value: Option<Option<u64>>,
}

#[derive(Clone)]
struct Revision {
    nodes: Vec<Node>,
    len: usize,
}

impl Default for Revision {
    fn default() -> Self {
        Self {
            nodes: vec![Node::default()],
            len: 0,
        }
    }
}

impl Revision {
    fn insert(&mut self, term: &str, value: Option<u64>) {
        let mut current = 0usize;
        for unit in term.chars() {
            let next = match self.nodes[current].edges.get(&unit).copied() {
                Some(next) => next as usize,
                None => {
                    let next = self.nodes.len();
                    self.nodes.push(Node::default());
                    self.nodes[current].edges.insert(unit, next as u64);
                    next
                }
            };
            current = next;
        }
        if self.nodes[current].value.is_none() {
            self.len += 1;
        }
        self.nodes[current].value = Some(value);
    }

    fn remove(&mut self, term: &str) {
        let mut current = 0usize;
        for unit in term.chars() {
            let Some(next) = self.nodes[current].edges.get(&unit).copied() else {
                return;
            };
            current = next as usize;
        }
        if self.nodes[current].value.take().is_some() {
            self.len -= 1;
        }
    }
}

#[derive(Default)]
struct Metrics {
    snapshots: AtomicUsize,
    edge_batches: AtomicUsize,
    // Lifecycle ledger (VT-LIFE invariants): every vtable retain/release call
    // and every Context destruction, observable without touching the Arcs.
    retains: AtomicUsize,
    releases: AtomicUsize,
    context_drops: AtomicUsize,
    // Callback concurrency observed inside node_edges, aggregated across
    // every context (base + snapshots). Distinct captured objects carry
    // distinct consumer gates (VT-GATE-1), so this aggregate may legally
    // exceed one even for a serialized provider — per-OBJECT serialization
    // is what VT-GATE-2 constrains, observed via the snapshot() counters
    // below (all snapshot() calls in these tests hit the one base object).
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    // Query-start concurrency observed inside snapshot() on the base object —
    // the shared-object contention point the consumer gate actually guards.
    snapshot_in_flight: AtomicUsize,
    max_snapshot_in_flight: AtomicUsize,
    // Test-controlled holds: while true, the corresponding callback blocks
    // after registering itself in-flight, letting tests park a thread
    // provably inside a callback without sleeps.
    edge_hold: Mutex<bool>,
    edge_hold_cv: Condvar,
    snapshot_hold: Mutex<bool>,
    snapshot_hold_cv: Condvar,
}

/// RAII registration of one in-flight callback, honoring a test hold.
struct CallbackGuard<'a> {
    live: &'a AtomicUsize,
}

impl<'a> CallbackGuard<'a> {
    fn enter(
        live: &'a AtomicUsize,
        peak: &'a AtomicUsize,
        hold: &'a Mutex<bool>,
        hold_cv: &'a Condvar,
    ) -> Self {
        let now_live = live.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now_live, Ordering::SeqCst);
        let mut held = hold.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while *held {
            held = hold_cv
                .wait(held)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        Self { live }
    }
}

impl Drop for CallbackGuard<'_> {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Store {
    current: RwLock<Arc<Revision>>,
    metrics: Arc<Metrics>,
}

enum ContextKind {
    Mutable(Arc<Store>),
    Snapshot {
        revision: Arc<Revision>,
        metrics: Arc<Metrics>,
    },
}

struct Context {
    kind: ContextKind,
    // Which capability set query_interface hands out: false = serialized
    // (CallGate::Serial on the consumer side), true = PARALLEL_REENTRANT.
    reentrant: bool,
    // Raw presence byte returned by the optional length callback. Tests use
    // zero for an unknown count and values above one for hostile output.
    len_presence: u8,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.metrics().context_drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Context {
    fn revision(&self) -> Arc<Revision> {
        match &self.kind {
            ContextKind::Mutable(store) => store
                .current
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
            ContextKind::Snapshot { revision, .. } => Arc::clone(revision),
        }
    }

    fn metrics(&self) -> &Arc<Metrics> {
        match &self.kind {
            ContextKind::Mutable(store) => &store.metrics,
            ContextKind::Snapshot { metrics, .. } => metrics,
        }
    }

    fn immutable(&self) -> bool {
        matches!(self.kind, ContextKind::Snapshot { .. })
    }
}

fn resource(context: Context) -> VtResource {
    VtResource {
        context: Arc::into_raw(Arc::new(context)).cast_mut().cast(),
        vtable: &RESOURCE_VTABLE,
    }
}

unsafe fn context<'a>(raw: *mut c_void) -> &'a Context {
    &*raw.cast::<Context>()
}

unsafe extern "C" fn retain(raw: *mut c_void) {
    context(raw)
        .metrics()
        .retains
        .fetch_add(1, Ordering::SeqCst);
    Arc::increment_strong_count(raw.cast::<Context>());
}

unsafe extern "C" fn release(raw: *mut c_void) {
    // Clone the metrics handle first: this release may destroy the Context.
    let metrics = Arc::clone(context(raw).metrics());
    metrics.releases.fetch_add(1, Ordering::SeqCst);
    drop(Arc::from_raw(raw.cast::<Context>()));
}

unsafe extern "C" fn query_interface(
    raw: *mut c_void,
    interface_id: *const VtInterfaceId,
    minimum_version: u32,
    out_vtable: *mut *const c_void,
) -> u32 {
    query_interface_status(raw, interface_id, minimum_version, out_vtable).to_raw()
}

unsafe fn query_interface_status(
    raw: *mut c_void,
    interface_id: *const VtInterfaceId,
    minimum_version: u32,
    out_vtable: *mut *const c_void,
) -> VtStatus {
    if interface_id.is_null() || out_vtable.is_null() {
        return VtStatus::NullPointer;
    }
    if (*interface_id).bytes != VT_DICTIONARY_INTERFACE_ID.bytes
        || minimum_version > VT_DICTIONARY_INTERFACE_VERSION
    {
        return VtStatus::Unsupported;
    }
    let source = context(raw);
    let dictionary = match (source.immutable(), source.reentrant) {
        (true, false) => &SNAPSHOT_DICTIONARY_VTABLE,
        (false, false) => &MUTABLE_DICTIONARY_VTABLE,
        (true, true) => &SNAPSHOT_REENTRANT_VTABLE,
        (false, true) => &MUTABLE_REENTRANT_VTABLE,
    };
    out_vtable.write((dictionary as *const VtDictionaryVTable).cast());
    VtStatus::Ok
}

unsafe extern "C" fn snapshot(raw: *mut c_void, out: *mut VtResource) -> u32 {
    snapshot_status(raw, out).to_raw()
}

unsafe fn snapshot_status(raw: *mut c_void, out: *mut VtResource) -> VtStatus {
    if out.is_null() {
        return VtStatus::NullPointer;
    }
    let source = context(raw);
    let metrics = source.metrics();
    metrics.snapshots.fetch_add(1, Ordering::Relaxed);
    // Registers this query-start callback in-flight and honors the test
    // hold; the guard un-registers on every return path.
    let _in_flight = CallbackGuard::enter(
        &metrics.snapshot_in_flight,
        &metrics.max_snapshot_in_flight,
        &metrics.snapshot_hold,
        &metrics.snapshot_hold_cv,
    );
    out.write(resource(Context {
        kind: ContextKind::Snapshot {
            revision: source.revision(),
            metrics: Arc::clone(source.metrics()),
        },
        reentrant: source.reentrant,
        len_presence: source.len_presence,
    }));
    VtStatus::Ok
}

unsafe extern "C" fn root(_raw: *mut c_void, out_node: *mut u64) -> u32 {
    root_status(_raw, out_node).to_raw()
}

unsafe fn root_status(_raw: *mut c_void, out_node: *mut u64) -> VtStatus {
    if out_node.is_null() {
        return VtStatus::NullPointer;
    }
    out_node.write(0);
    VtStatus::Ok
}

unsafe extern "C" fn len(raw: *mut c_void, out_len: *mut usize, out_known: *mut u8) -> u32 {
    len_status(raw, out_len, out_known).to_raw()
}

unsafe fn len_status(raw: *mut c_void, out_len: *mut usize, out_known: *mut u8) -> VtStatus {
    if out_len.is_null() || out_known.is_null() {
        return VtStatus::NullPointer;
    }
    let source = context(raw);
    out_len.write(source.revision().len);
    out_known.write(source.len_presence);
    VtStatus::Ok
}

unsafe extern "C" fn node_is_final(raw: *mut c_void, node: u64, out_is_final: *mut u8) -> u32 {
    node_is_final_status(raw, node, out_is_final).to_raw()
}

unsafe fn node_is_final_status(raw: *mut c_void, node: u64, out_is_final: *mut u8) -> VtStatus {
    if out_is_final.is_null() {
        return VtStatus::NullPointer;
    }
    let revision = context(raw).revision();
    let Some(node) = revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument;
    };
    out_is_final.write(u8::from(node.value.is_some()));
    VtStatus::Ok
}

unsafe extern "C" fn node_value_u64(
    raw: *mut c_void,
    node: u64,
    out_value: *mut VtOptionalU64,
) -> u32 {
    node_value_u64_status(raw, node, out_value).to_raw()
}

unsafe fn node_value_u64_status(
    raw: *mut c_void,
    node: u64,
    out_value: *mut VtOptionalU64,
) -> VtStatus {
    if out_value.is_null() {
        return VtStatus::NullPointer;
    }
    let revision = context(raw).revision();
    let Some(node) = revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument;
    };
    let Some(value) = node.value else {
        return VtStatus::InvalidArgument;
    };
    out_value.write(VtOptionalU64 {
        value: value.unwrap_or_default(),
        has_value: u8::from(value.is_some()),
        reserved: [0; 7],
    });
    VtStatus::Ok
}

unsafe extern "C" fn node_transition(
    raw: *mut c_void,
    node: u64,
    label: u64,
    out_child: *mut u64,
    out_found: *mut u8,
) -> u32 {
    node_transition_status(raw, node, label, out_child, out_found).to_raw()
}

unsafe fn node_transition_status(
    raw: *mut c_void,
    node: u64,
    label: u64,
    out_child: *mut u64,
    out_found: *mut u8,
) -> VtStatus {
    if out_child.is_null() || out_found.is_null() {
        return VtStatus::NullPointer;
    }
    let Some(label) = u32::try_from(label).ok().and_then(char::from_u32) else {
        return VtStatus::InvalidArgument;
    };
    let revision = context(raw).revision();
    let Some(node) = revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument;
    };
    match node.edges.get(&label) {
        Some(child) => {
            out_child.write(*child);
            out_found.write(1);
        }
        None => {
            out_child.write(0);
            out_found.write(0);
        }
    }
    VtStatus::Ok
}

unsafe extern "C" fn node_edges(
    raw: *mut c_void,
    node: u64,
    start: usize,
    out_edges: *mut VtDictionaryEdge,
    capacity: usize,
    out_written: *mut usize,
    out_total: *mut usize,
) -> u32 {
    node_edges_status(
        raw,
        node,
        start,
        out_edges,
        capacity,
        out_written,
        out_total,
    )
    .to_raw()
}

unsafe fn node_edges_status(
    raw: *mut c_void,
    node: u64,
    start: usize,
    out_edges: *mut VtDictionaryEdge,
    capacity: usize,
    out_written: *mut usize,
    out_total: *mut usize,
) -> VtStatus {
    if (capacity != 0 && out_edges.is_null()) || out_written.is_null() || out_total.is_null() {
        return VtStatus::NullPointer;
    }
    let source = context(raw);
    source
        .metrics()
        .edge_batches
        .fetch_add(1, Ordering::Relaxed);
    // Registers this call in-flight (max tracked) and honors a test-imposed
    // hold; the guard un-registers on every return path.
    let edge_metrics = source.metrics();
    let _in_flight = CallbackGuard::enter(
        &edge_metrics.in_flight,
        &edge_metrics.max_in_flight,
        &edge_metrics.edge_hold,
        &edge_metrics.edge_hold_cv,
    );
    let revision = source.revision();
    let Some(node) = revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument;
    };
    let total = node.edges.len();
    let mut written = 0;
    for (label, child) in node.edges.iter().skip(start).take(capacity) {
        out_edges.add(written).write(VtDictionaryEdge {
            label: u64::from(u32::from(*label)),
            node: *child,
        });
        written += 1;
    }
    out_written.write(written);
    out_total.write(total);
    VtStatus::Ok
}

static RESOURCE_VTABLE: VtResourceVTable = VtResourceVTable {
    struct_size: std::mem::size_of::<VtResourceVTable>(),
    abi_version: VT_ABI_VERSION,
    reserved: 0,
    retain: Some(retain),
    release: Some(release),
    query_interface: Some(query_interface),
};

const fn dictionary_vtable(flags: u64) -> VtDictionaryVTable {
    VtDictionaryVTable {
        struct_size: std::mem::size_of::<VtDictionaryVTable>(),
        interface_version: VT_DICTIONARY_INTERFACE_VERSION,
        unit_domain: VtUnitDomain::UnicodeScalar,
        value_domain: VtValueDomain::OptionalU64,
        flags,
        snapshot: Some(snapshot),
        root: Some(root),
        len: Some(len),
        node_is_final: Some(node_is_final),
        node_value_u64: Some(node_value_u64),
        node_transition: Some(node_transition),
        node_edges: Some(node_edges),
    }
}

static MUTABLE_DICTIONARY_VTABLE: VtDictionaryVTable = dictionary_vtable(0);
static SNAPSHOT_DICTIONARY_VTABLE: VtDictionaryVTable =
    dictionary_vtable(dictionary_flags::IMMUTABLE);
static MUTABLE_REENTRANT_VTABLE: VtDictionaryVTable =
    dictionary_vtable(dictionary_flags::PARALLEL_REENTRANT);
static SNAPSHOT_REENTRANT_VTABLE: VtDictionaryVTable =
    dictionary_vtable(dictionary_flags::IMMUTABLE | dictionary_flags::PARALLEL_REENTRANT);

pub struct TestDictionary {
    store: Arc<Store>,
    resource: VtResource,
}

impl TestDictionary {
    pub fn new(entries: impl IntoIterator<Item = (String, Option<u64>)>) -> Self {
        Self::with_reentrancy(entries, false)
    }

    /// Build a provider that advertises `PARALLEL_REENTRANT` when `reentrant`
    /// is true; the default provider stays serialized so consumers exercise
    /// `CallGate::Serial`.
    pub fn with_reentrancy(
        entries: impl IntoIterator<Item = (String, Option<u64>)>,
        reentrant: bool,
    ) -> Self {
        Self::with_options(entries, reentrant, 1)
    }

    pub fn with_len_presence(
        entries: impl IntoIterator<Item = (String, Option<u64>)>,
        len_presence: u8,
    ) -> Self {
        Self::with_options(entries, false, len_presence)
    }

    fn with_options(
        entries: impl IntoIterator<Item = (String, Option<u64>)>,
        reentrant: bool,
        len_presence: u8,
    ) -> Self {
        let mut revision = Revision::default();
        for (term, value) in entries {
            revision.insert(&term, value);
        }
        let store = Arc::new(Store {
            current: RwLock::new(Arc::new(revision)),
            metrics: Arc::new(Metrics::default()),
        });
        let resource = resource(Context {
            kind: ContextKind::Mutable(Arc::clone(&store)),
            reentrant,
            len_presence,
        });
        Self { store, resource }
    }

    pub fn resource(&self) -> VtResource {
        self.resource
    }

    fn mutate(&self, operation: impl FnOnce(&mut Revision)) {
        let mut current = self
            .store
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut next = (**current).clone();
        operation(&mut next);
        *current = Arc::new(next);
    }

    pub fn insert(&self, term: &str, value: Option<u64>) {
        self.mutate(|revision| revision.insert(term, value));
    }

    pub fn update(&self, term: &str, value: Option<u64>) {
        self.insert(term, value);
    }

    pub fn remove(&self, term: &str) {
        self.mutate(|revision| revision.remove(term));
    }

    pub fn clear(&self) {
        self.mutate(|revision| *revision = Revision::default());
    }

    pub fn compact(&self) {
        self.mutate(|_revision| {});
    }

    pub fn checkpoint(&self) {
        self.mutate(|_revision| {});
    }

    pub fn snapshot_calls(&self) -> usize {
        self.store.metrics.snapshots.load(Ordering::Relaxed)
    }

    pub fn edge_batch_calls(&self) -> usize {
        self.store.metrics.edge_batches.load(Ordering::Relaxed)
    }

    /// Vtable `retain` calls observed across this provider and every
    /// snapshot resource it produced.
    pub fn retain_calls(&self) -> usize {
        self.store.metrics.retains.load(Ordering::SeqCst)
    }

    /// Vtable `release` calls observed across this provider and every
    /// snapshot resource it produced.
    pub fn release_calls(&self) -> usize {
        self.store.metrics.releases.load(Ordering::SeqCst)
    }

    /// Number of resource contexts (the root plus every snapshot) whose
    /// final release destroyed them — the executable face of VT-LIFE-5/6.
    pub fn context_drops(&self) -> usize {
        self.store.metrics.context_drops.load(Ordering::SeqCst)
    }

    /// Callbacks currently inside `node_edges`.
    pub fn edges_in_flight(&self) -> usize {
        self.store.metrics.in_flight.load(Ordering::SeqCst)
    }

    /// High-water mark of concurrent `node_edges` callbacks — the executable
    /// face of VT-GATE-2 (== 1 under a serialized consumer gate).
    pub fn max_edges_in_flight(&self) -> usize {
        self.store.metrics.max_in_flight.load(Ordering::SeqCst)
    }

    /// Park every subsequent `node_edges` call after it registers in-flight,
    /// until `resume_edges`. Lets tests hold a thread provably inside a
    /// callback without timing assumptions.
    pub fn pause_edges(&self) {
        let mut held = self
            .store
            .metrics
            .edge_hold
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *held = true;
    }

    /// Query-start callbacks currently inside `snapshot()`.
    pub fn snapshots_in_flight(&self) -> usize {
        self.store.metrics.snapshot_in_flight.load(Ordering::SeqCst)
    }

    /// High-water mark of concurrent `snapshot()` callbacks on this provider
    /// — the executable face of VT-GATE-2 (== 1 under a serialized consumer
    /// gate, since every query start crosses the one base object).
    pub fn max_snapshots_in_flight(&self) -> usize {
        self.store
            .metrics
            .max_snapshot_in_flight
            .load(Ordering::SeqCst)
    }

    /// Park every subsequent `snapshot()` call after it registers in-flight,
    /// until `resume_snapshots`.
    pub fn pause_snapshots(&self) {
        let mut held = self
            .store
            .metrics
            .snapshot_hold
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *held = true;
    }

    /// Release callbacks parked by `pause_snapshots`.
    pub fn resume_snapshots(&self) {
        let mut held = self
            .store
            .metrics
            .snapshot_hold
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *held = false;
        drop(held);
        self.store.metrics.snapshot_hold_cv.notify_all();
    }

    /// A metrics handle that outlives this provider, so teardown-complete
    /// ledgers (release of the root's own retain included) stay observable
    /// after `drop(TestDictionary)`.
    pub fn metrics_probe(&self) -> MetricsProbe {
        MetricsProbe(Arc::clone(&self.store.metrics))
    }

    /// Release callbacks parked by `pause_edges`.
    pub fn resume_edges(&self) {
        let mut held = self
            .store
            .metrics
            .edge_hold
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *held = false;
        drop(held);
        self.store.metrics.edge_hold_cv.notify_all();
    }
}

impl Drop for TestDictionary {
    fn drop(&mut self) {
        unsafe { release(self.resource.context) };
    }
}

/// Provider-independent view of the lifecycle ledger (valid after the
/// provider itself is dropped).
pub struct MetricsProbe(Arc<Metrics>);

impl MetricsProbe {
    pub fn snapshot_calls(&self) -> usize {
        self.0.snapshots.load(Ordering::SeqCst)
    }

    pub fn retain_calls(&self) -> usize {
        self.0.retains.load(Ordering::SeqCst)
    }

    pub fn release_calls(&self) -> usize {
        self.0.releases.load(Ordering::SeqCst)
    }

    pub fn context_drops(&self) -> usize {
        self.0.context_drops.load(Ordering::SeqCst)
    }
}
