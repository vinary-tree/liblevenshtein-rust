#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
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
    Arc::increment_strong_count(raw.cast::<Context>());
}

unsafe extern "C" fn release(raw: *mut c_void) {
    drop(Arc::from_raw(raw.cast::<Context>()));
}

unsafe extern "C" fn query_interface(
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
    let dictionary = if context(raw).immutable() {
        &SNAPSHOT_DICTIONARY_VTABLE
    } else {
        &MUTABLE_DICTIONARY_VTABLE
    };
    out_vtable.write((dictionary as *const VtDictionaryVTable).cast());
    VtStatus::Ok
}

unsafe extern "C" fn snapshot(raw: *mut c_void, out: *mut VtResource) -> VtStatus {
    if out.is_null() {
        return VtStatus::NullPointer;
    }
    let source = context(raw);
    source.metrics().snapshots.fetch_add(1, Ordering::Relaxed);
    out.write(resource(Context {
        kind: ContextKind::Snapshot {
            revision: source.revision(),
            metrics: Arc::clone(source.metrics()),
        },
    }));
    VtStatus::Ok
}

unsafe extern "C" fn root(_raw: *mut c_void, out_node: *mut u64) -> VtStatus {
    if out_node.is_null() {
        return VtStatus::NullPointer;
    }
    out_node.write(0);
    VtStatus::Ok
}

unsafe extern "C" fn len(raw: *mut c_void, out_len: *mut usize, out_known: *mut u8) -> VtStatus {
    if out_len.is_null() || out_known.is_null() {
        return VtStatus::NullPointer;
    }
    out_len.write(context(raw).revision().len);
    out_known.write(1);
    VtStatus::Ok
}

unsafe extern "C" fn node_is_final(raw: *mut c_void, node: u64, out_is_final: *mut u8) -> VtStatus {
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
) -> VtStatus {
    if (capacity != 0 && out_edges.is_null()) || out_written.is_null() || out_total.is_null() {
        return VtStatus::NullPointer;
    }
    let source = context(raw);
    source
        .metrics()
        .edge_batches
        .fetch_add(1, Ordering::Relaxed);
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

pub struct TestDictionary {
    store: Arc<Store>,
    resource: VtResource,
}

impl TestDictionary {
    pub fn new(entries: impl IntoIterator<Item = (String, Option<u64>)>) -> Self {
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
}

impl Drop for TestDictionary {
    fn drop(&mut self) {
        unsafe { release(self.resource.context) };
    }
}
