//! Programmable misbehaving dictionary provider for adversarial ABI tests.
//!
//! This provider implements the full vinary-tree interop dictionary vtable
//! (Unicode-scalar units, optional-u64 values, SERIALIZED — no
//! `PARALLEL_REENTRANT` flag, so consumers route every callback through
//! `CallGate::Serial`) and lets a test program two independent classes of
//! provider misbehavior:
//!
//! 1. **Status injection** — any callback can be armed to return an arbitrary
//!    RAW `u32` status instead of executing, including values with no
//!    [`vinary_tree_interop::VtStatus`] discriminant (e.g. `9`, `42`,
//!    `u32::MAX`). The raw-u32 status wire makes such garbage a VALUE the
//!    consumer must decode and reject, never undefined behavior (ledger
//!    LLEV-B6). Injections can be sticky, skip a number of calls first, or
//!    fire exactly once.
//! 2. **Structural misbehavior** — the callback returns `Ok` but writes
//!    malformed data: an over-capacity `out_written` claim, a paging total
//!    that shrinks mid-pagination, an inflated total that yields an empty
//!    page with work remaining, garbage node identifiers or edge labels,
//!    `has_value = 7`, nonzero `VtOptionalU64::reserved` bytes (ledger
//!    LLEV-B7), `is_final = 7`, and a null snapshot resource.
//!
//! Callbacks NEVER panic (an unwind across `extern "C"` aborts the process):
//! every lock uses poison recovery, every lookup is bounds-checked, and
//! misbehavior writes stay strictly inside caller-provided capacity.
//!
//! The provider also keeps the lifecycle ledger used across the ABI tests:
//! counted vtable `retain`/`release` calls, context births and destructions,
//! and per-callback call counters, observable after the provider itself is
//! dropped through [`FaultMetricsProbe`].

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use vinary_tree_interop::{
    VtDictionaryEdge, VtDictionaryVTable, VtInterfaceId, VtOptionalU64, VtResource,
    VtResourceVTable, VtStatus, VtUnitDomain, VtValueDomain, VT_ABI_VERSION,
    VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION,
};

/// Provider callback selected for fault programming.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FaultOp {
    /// Base-resource interface negotiation.
    QueryInterface,
    /// Query-start revision capture.
    Snapshot,
    /// Root-node lookup.
    Root,
    /// Term-count report.
    Len,
    /// Node finality test.
    NodeIsFinal,
    /// Final-node value read.
    NodeValueU64,
    /// Single-edge follow.
    NodeTransition,
    /// Batched edge expansion.
    NodeEdges,
}

impl FaultOp {
    const COUNT: usize = 8;

    fn index(self) -> usize {
        match self {
            Self::QueryInterface => 0,
            Self::Snapshot => 1,
            Self::Root => 2,
            Self::Len => 3,
            Self::NodeIsFinal => 4,
            Self::NodeValueU64 => 5,
            Self::NodeTransition => 6,
            Self::NodeEdges => 7,
        }
    }
}

/// One armed status injection.
struct ArmedFault {
    /// Raw wire status to return; intentionally NOT a `VtStatus` so tests can
    /// inject out-of-range discriminants.
    raw: u32,
    /// Number of calls that still execute normally before the fault fires.
    skip: usize,
    /// Whether the fault disarms after firing once.
    one_shot: bool,
}

#[derive(Default)]
struct FaultPlan {
    overrides: Mutex<BTreeMap<FaultOp, ArmedFault>>,
    // Structural misbehavior toggles. Tests arm at most one at a time; when
    // several are armed the precedence is documented at each callback.
    edges_overshoot_written: AtomicBool,
    edges_deflate_total: AtomicBool,
    edges_inflate_total: AtomicBool,
    edges_garbage_nodes: AtomicBool,
    edges_garbage_labels: AtomicBool,
    value_has_value_seven: AtomicBool,
    value_nonzero_reserved: AtomicBool,
    is_final_seven: AtomicBool,
    transition_garbage: AtomicBool,
    snapshot_null: AtomicBool,
}

impl FaultPlan {
    /// Consult (and advance) the injection state for one callback invocation.
    /// Never panics: the poisoned-lock branch recovers the inner state.
    fn fire(&self, op: FaultOp) -> Option<u32> {
        let mut overrides = self
            .overrides
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let armed = overrides.get_mut(&op)?;
        if armed.skip > 0 {
            armed.skip -= 1;
            return None;
        }
        let raw = armed.raw;
        if armed.one_shot {
            overrides.remove(&op);
        }
        Some(raw)
    }
}

#[derive(Default)]
struct Metrics {
    // Lifecycle ledger: releases - retains == contexts_created ==
    // context_drops at full teardown (constructor births are uncounted
    // retains, mirroring tests/support/interop_dictionary.rs).
    retains: AtomicUsize,
    releases: AtomicUsize,
    contexts_created: AtomicUsize,
    context_drops: AtomicUsize,
    snapshots: AtomicUsize,
    op_calls: [AtomicUsize; FaultOp::COUNT],
}

#[derive(Clone, Default)]
struct Node {
    edges: BTreeMap<char, u64>,
    // Outer option distinguishes a final term without an ID from a non-final
    // node.
    value: Option<Option<u64>>,
}

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
}

/// State shared by the base context and every snapshot context: the immutable
/// revision, the fault plan (faults keep applying on snapshot objects, where
/// most traversal callbacks actually run), and the ledger.
struct FaultCore {
    revision: Revision,
    plan: FaultPlan,
    metrics: Arc<Metrics>,
}

struct Context {
    core: Arc<FaultCore>,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.core
            .metrics
            .context_drops
            .fetch_add(1, Ordering::SeqCst);
    }
}

fn resource(core: Arc<FaultCore>) -> VtResource {
    core.metrics.contexts_created.fetch_add(1, Ordering::SeqCst);
    VtResource {
        context: Arc::into_raw(Arc::new(Context { core })).cast_mut().cast(),
        vtable: &RESOURCE_VTABLE,
    }
}

unsafe fn context<'a>(raw: *mut c_void) -> &'a Context {
    &*raw.cast::<Context>()
}

/// Count one invocation and consult the injection plan.
unsafe fn enter(raw: *mut c_void, op: FaultOp) -> Option<u32> {
    let core = &context(raw).core;
    core.metrics.op_calls[op.index()].fetch_add(1, Ordering::SeqCst);
    core.plan.fire(op)
}

unsafe extern "C" fn retain(raw: *mut c_void) {
    context(raw)
        .core
        .metrics
        .retains
        .fetch_add(1, Ordering::SeqCst);
    Arc::increment_strong_count(raw.cast::<Context>());
}

unsafe extern "C" fn release(raw: *mut c_void) {
    // Clone the metrics handle first: this release may destroy the Context.
    let metrics = Arc::clone(&context(raw).core.metrics);
    metrics.releases.fetch_add(1, Ordering::SeqCst);
    drop(Arc::from_raw(raw.cast::<Context>()));
}

unsafe extern "C" fn query_interface(
    raw: *mut c_void,
    interface_id: *const VtInterfaceId,
    minimum_version: u32,
    out_vtable: *mut *const c_void,
) -> u32 {
    if let Some(injected) = enter(raw, FaultOp::QueryInterface) {
        return injected;
    }
    if interface_id.is_null() || out_vtable.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    if (*interface_id).bytes != VT_DICTIONARY_INTERFACE_ID.bytes
        || minimum_version > VT_DICTIONARY_INTERFACE_VERSION
    {
        return VtStatus::Unsupported.to_raw();
    }
    out_vtable.write((&DICTIONARY_VTABLE as *const VtDictionaryVTable).cast());
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn snapshot(raw: *mut c_void, out: *mut VtResource) -> u32 {
    let source = context(raw);
    if let Some(injected) = enter(raw, FaultOp::Snapshot) {
        return injected;
    }
    if out.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    source.core.metrics.snapshots.fetch_add(1, Ordering::SeqCst);
    if source.core.plan.snapshot_null.load(Ordering::SeqCst) {
        // Misbehavior: claim success while transferring no resource at all.
        out.write(VtResource::NULL);
        return VtStatus::Ok.to_raw();
    }
    out.write(resource(Arc::clone(&source.core)));
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn root(raw: *mut c_void, out_node: *mut u64) -> u32 {
    if let Some(injected) = enter(raw, FaultOp::Root) {
        return injected;
    }
    if out_node.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    out_node.write(0);
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn len(raw: *mut c_void, out_len: *mut usize, out_known: *mut u8) -> u32 {
    if let Some(injected) = enter(raw, FaultOp::Len) {
        return injected;
    }
    if out_len.is_null() || out_known.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    out_len.write(context(raw).core.revision.len);
    out_known.write(1);
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn node_is_final(raw: *mut c_void, node: u64, out_is_final: *mut u8) -> u32 {
    let source = context(raw);
    if let Some(injected) = enter(raw, FaultOp::NodeIsFinal) {
        return injected;
    }
    if out_is_final.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    if source.core.plan.is_final_seven.load(Ordering::SeqCst) {
        // Misbehavior: an out-of-domain boolean.
        out_is_final.write(7);
        return VtStatus::Ok.to_raw();
    }
    let Some(node) = source.core.revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    out_is_final.write(u8::from(node.value.is_some()));
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn node_value_u64(
    raw: *mut c_void,
    node: u64,
    out_value: *mut VtOptionalU64,
) -> u32 {
    let source = context(raw);
    if let Some(injected) = enter(raw, FaultOp::NodeValueU64) {
        return injected;
    }
    if out_value.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    // Misbehavior precedence: has_value=7 beats nonzero-reserved; both beat
    // the normal read so they fire deterministically on any final node.
    if source
        .core
        .plan
        .value_has_value_seven
        .load(Ordering::SeqCst)
    {
        out_value.write(VtOptionalU64 {
            value: 0,
            has_value: 7,
            reserved: [0; 7],
        });
        return VtStatus::Ok.to_raw();
    }
    if source
        .core
        .plan
        .value_nonzero_reserved
        .load(Ordering::SeqCst)
    {
        out_value.write(VtOptionalU64 {
            value: 41,
            has_value: 1,
            reserved: [0xA5, 0, 0, 0, 0, 0, 0],
        });
        return VtStatus::Ok.to_raw();
    }
    let Some(node) = source.core.revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    let Some(value) = node.value else {
        return VtStatus::InvalidArgument.to_raw();
    };
    out_value.write(VtOptionalU64 {
        value: value.unwrap_or_default(),
        has_value: u8::from(value.is_some()),
        reserved: [0; 7],
    });
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn node_transition(
    raw: *mut c_void,
    node: u64,
    label: u64,
    out_child: *mut u64,
    out_found: *mut u8,
) -> u32 {
    let source = context(raw);
    if let Some(injected) = enter(raw, FaultOp::NodeTransition) {
        return injected;
    }
    if out_child.is_null() || out_found.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    if source.core.plan.transition_garbage.load(Ordering::SeqCst) {
        // Misbehavior: claim a hit on a node identifier that cannot exist.
        out_child.write(u64::MAX);
        out_found.write(1);
        return VtStatus::Ok.to_raw();
    }
    let Some(label) = u32::try_from(label).ok().and_then(char::from_u32) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    let Some(node) = source.core.revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
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
    VtStatus::Ok.to_raw()
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
    let source = context(raw);
    if let Some(injected) = enter(raw, FaultOp::NodeEdges) {
        return injected;
    }
    if (capacity != 0 && out_edges.is_null()) || out_written.is_null() || out_total.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    let Some(node) = source.core.revision.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    let plan = &source.core.plan;
    let garbage_nodes = plan.edges_garbage_nodes.load(Ordering::SeqCst);
    let garbage_labels = plan.edges_garbage_labels.load(Ordering::SeqCst);
    let total = node.edges.len();
    // The real page is written first and stays strictly within `capacity`
    // no matter which lie is told about it below: misbehavior corrupts the
    // REPORT, never the caller's memory.
    let mut written = 0usize;
    for (label, child) in node.edges.iter().skip(start).take(capacity) {
        out_edges.add(written).write(VtDictionaryEdge {
            label: if garbage_labels {
                u64::MAX
            } else {
                u64::from(u32::from(*label))
            },
            node: if garbage_nodes { u64::MAX } else { *child },
        });
        written += 1;
    }
    let mut reported_written = written;
    let mut reported_total = total;
    // Report-corruption precedence: overshoot > deflate > inflate.
    if plan.edges_overshoot_written.load(Ordering::SeqCst) {
        // Claim one more entry than the caller has room for.
        reported_written = capacity.saturating_add(1);
    } else if plan.edges_deflate_total.load(Ordering::SeqCst) && start > 0 {
        // Second and later pages shrink the total below what was already
        // consumed (an unstable total across pages).
        reported_total = start;
    } else if plan.edges_inflate_total.load(Ordering::SeqCst) {
        // Promise more edges than exist, producing an eventual empty page
        // with `start < total` (paging that makes no progress).
        reported_total = total.saturating_add(3);
    }
    out_written.write(reported_written);
    out_total.write(reported_total);
    VtStatus::Ok.to_raw()
}

static RESOURCE_VTABLE: VtResourceVTable = VtResourceVTable {
    struct_size: std::mem::size_of::<VtResourceVTable>(),
    abi_version: VT_ABI_VERSION,
    reserved: 0,
    retain: Some(retain),
    release: Some(release),
    query_interface: Some(query_interface),
};

static DICTIONARY_VTABLE: VtDictionaryVTable = VtDictionaryVTable {
    struct_size: std::mem::size_of::<VtDictionaryVTable>(),
    interface_version: VT_DICTIONARY_INTERFACE_VERSION,
    unit_domain: VtUnitDomain::UnicodeScalar,
    value_domain: VtValueDomain::OptionalU64,
    // Deliberately serialized: no PARALLEL_REENTRANT, so the consumer's
    // CallGate::Serial wraps every callback (the VT-GATE-3 error-path tests
    // depend on this).
    flags: 0,
    snapshot: Some(snapshot),
    root: Some(root),
    len: Some(len),
    node_is_final: Some(node_is_final),
    node_value_u64: Some(node_value_u64),
    node_transition: Some(node_transition),
    node_edges: Some(node_edges),
};

/// Owning handle for one programmable misbehaving provider.
pub struct FaultDictionary {
    core: Arc<FaultCore>,
    resource: VtResource,
}

impl FaultDictionary {
    /// Build a provider over an immutable term set.
    pub fn new(entries: impl IntoIterator<Item = (String, Option<u64>)>) -> Self {
        let mut revision = Revision::default();
        for (term, value) in entries {
            revision.insert(&term, value);
        }
        let core = Arc::new(FaultCore {
            revision,
            plan: FaultPlan::default(),
            metrics: Arc::new(Metrics::default()),
        });
        let resource = resource(Arc::clone(&core));
        Self { core, resource }
    }

    /// The two-word resource handed to consumers (borrow semantics: the
    /// consumer must retain before storing an owned copy).
    pub fn resource(&self) -> VtResource {
        self.resource
    }

    /// Arm `op` to return `raw` on every subsequent call.
    pub fn fail_op(&self, op: FaultOp, raw: u32) {
        self.fail_op_after(op, raw, 0);
    }

    /// Arm `op` to execute normally `skip` more times, then return `raw` on
    /// every later call.
    pub fn fail_op_after(&self, op: FaultOp, raw: u32, skip: usize) {
        self.arm(
            op,
            ArmedFault {
                raw,
                skip,
                one_shot: false,
            },
        );
    }

    /// Arm `op` to execute normally `skip` more times, return `raw` exactly
    /// once, and then heal.
    pub fn fail_op_once_after(&self, op: FaultOp, raw: u32, skip: usize) {
        self.arm(
            op,
            ArmedFault {
                raw,
                skip,
                one_shot: true,
            },
        );
    }

    fn arm(&self, op: FaultOp, fault: ArmedFault) {
        let mut overrides = self
            .core
            .plan
            .overrides
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        overrides.insert(op, fault);
    }

    /// Disarm one status injection.
    pub fn clear_fault(&self, op: FaultOp) {
        let mut overrides = self
            .core
            .plan
            .overrides
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        overrides.remove(&op);
    }

    /// Disarm every status injection and structural misbehavior.
    pub fn clear_faults(&self) {
        {
            let mut overrides = self
                .core
                .plan
                .overrides
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            overrides.clear();
        }
        let plan = &self.core.plan;
        for toggle in [
            &plan.edges_overshoot_written,
            &plan.edges_deflate_total,
            &plan.edges_inflate_total,
            &plan.edges_garbage_nodes,
            &plan.edges_garbage_labels,
            &plan.value_has_value_seven,
            &plan.value_nonzero_reserved,
            &plan.is_final_seven,
            &plan.transition_garbage,
            &plan.snapshot_null,
        ] {
            toggle.store(false, Ordering::SeqCst);
        }
    }

    /// `node_edges` claims `out_written = capacity + 1` (writes stay bounded).
    pub fn set_edges_overshoot_written(&self, active: bool) {
        self.core
            .plan
            .edges_overshoot_written
            .store(active, Ordering::SeqCst);
    }

    /// `node_edges` reports a total that shrinks once paging has started.
    pub fn set_edges_deflate_total(&self, active: bool) {
        self.core
            .plan
            .edges_deflate_total
            .store(active, Ordering::SeqCst);
    }

    /// `node_edges` inflates the total, eventually producing an empty page
    /// while `start < total`.
    pub fn set_edges_inflate_total(&self, active: bool) {
        self.core
            .plan
            .edges_inflate_total
            .store(active, Ordering::SeqCst);
    }

    /// Every reported edge targets node `u64::MAX`.
    pub fn set_edges_garbage_nodes(&self, active: bool) {
        self.core
            .plan
            .edges_garbage_nodes
            .store(active, Ordering::SeqCst);
    }

    /// Every reported edge label is `u64::MAX` (outside every unit domain).
    pub fn set_edges_garbage_labels(&self, active: bool) {
        self.core
            .plan
            .edges_garbage_labels
            .store(active, Ordering::SeqCst);
    }

    /// `node_value_u64` writes `has_value = 7`.
    pub fn set_value_has_value_seven(&self, active: bool) {
        self.core
            .plan
            .value_has_value_seven
            .store(active, Ordering::SeqCst);
    }

    /// `node_value_u64` writes nonzero reserved bytes (ledger LLEV-B7).
    pub fn set_value_nonzero_reserved(&self, active: bool) {
        self.core
            .plan
            .value_nonzero_reserved
            .store(active, Ordering::SeqCst);
    }

    /// `node_is_final` writes 7 instead of zero or one.
    pub fn set_is_final_seven(&self, active: bool) {
        self.core
            .plan
            .is_final_seven
            .store(active, Ordering::SeqCst);
    }

    /// `node_transition` claims `found = 1` with child `u64::MAX`.
    pub fn set_transition_garbage(&self, active: bool) {
        self.core
            .plan
            .transition_garbage
            .store(active, Ordering::SeqCst);
    }

    /// `snapshot` writes a NULL resource and returns `Ok`.
    pub fn set_snapshot_null(&self, active: bool) {
        self.core.plan.snapshot_null.store(active, Ordering::SeqCst);
    }

    /// Counted vtable retains across the base and every snapshot context.
    pub fn retain_calls(&self) -> usize {
        self.core.metrics.retains.load(Ordering::SeqCst)
    }

    /// Counted vtable releases across the base and every snapshot context.
    pub fn release_calls(&self) -> usize {
        self.core.metrics.releases.load(Ordering::SeqCst)
    }

    /// Contexts born (base plus every non-null snapshot).
    pub fn contexts_created(&self) -> usize {
        self.core.metrics.contexts_created.load(Ordering::SeqCst)
    }

    /// Contexts whose final release destroyed them.
    pub fn context_drops(&self) -> usize {
        self.core.metrics.context_drops.load(Ordering::SeqCst)
    }

    /// Successful passes through the snapshot callback.
    pub fn snapshot_calls(&self) -> usize {
        self.core.metrics.snapshots.load(Ordering::SeqCst)
    }

    /// Invocations of one callback (counted even when a fault fires).
    pub fn calls(&self, op: FaultOp) -> usize {
        self.core.metrics.op_calls[op.index()].load(Ordering::SeqCst)
    }

    /// A ledger handle that outlives this provider, for teardown-complete
    /// balance checks after `drop(FaultDictionary)`.
    pub fn probe(&self) -> FaultMetricsProbe {
        FaultMetricsProbe(Arc::clone(&self.core.metrics))
    }
}

impl Drop for FaultDictionary {
    fn drop(&mut self) {
        unsafe { release(self.resource.context) };
    }
}

/// Provider-independent view of the lifecycle ledger.
pub struct FaultMetricsProbe(Arc<Metrics>);

impl FaultMetricsProbe {
    pub fn retain_calls(&self) -> usize {
        self.0.retains.load(Ordering::SeqCst)
    }

    pub fn release_calls(&self) -> usize {
        self.0.releases.load(Ordering::SeqCst)
    }

    pub fn contexts_created(&self) -> usize {
        self.0.contexts_created.load(Ordering::SeqCst)
    }

    pub fn context_drops(&self) -> usize {
        self.0.context_drops.load(Ordering::SeqCst)
    }

    /// The full-teardown lifecycle law: every context born was destroyed, and
    /// destruction happened exactly through the counted release calls that
    /// outnumber counted retains by the uncounted constructor births.
    pub fn assert_balanced(&self) {
        let retains = self.retain_calls();
        let releases = self.release_calls();
        let created = self.contexts_created();
        let drops = self.context_drops();
        assert_eq!(
            drops, created,
            "every context born must be destroyed exactly once \
             (created {created}, dropped {drops})"
        );
        assert_eq!(
            releases,
            retains + created,
            "releases must cover every counted retain plus every uncounted \
             constructor birth (retains {retains}, releases {releases}, \
             births {created})"
        );
    }
}
