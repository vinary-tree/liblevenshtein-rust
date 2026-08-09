//! A custom `vt.dictionary.v1` provider for paging and arena pressure, plus
//! programmable paging misbehavior.
//!
//! Two independent stress axes share one provider:
//!   - high OUT-DEGREE at a single node (root with `degree` edges) drives the
//!     consumer's `node_edges` paging loop across the recommended batch many
//!     times — the fixture for the paging-correspondence laws;
//!   - long TERMS (paths up to a few kilobytes) drive the FFI cursor's byte
//!     arena through multiple reallocations within one batch fill — the
//!     fixture for arena-stability (used by the W3 T2 suite).
//!
//! A `Misbehavior` mode makes the root's `node_edges` emit exactly the
//! adversarial replies the `ConsumerAcceptance` rejection lemmas name
//! (overfill, past-end, stalled progress, inflated total), so a consumer can
//! be shown to reject each without aborting.
//!
//! The honest path reuses the same firstn/skipn paging shape proved in
//! `docs/verification/abi/theories/PagingLaws.v`.

#![allow(dead_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use vinary_tree_interop::{
    dictionary_flags, VtDictionaryEdge, VtDictionaryVTable, VtInterfaceId, VtOptionalU64,
    VtResource, VtResourceVTable, VtStatus, VtUnitDomain, VtValueDomain, VT_ABI_VERSION,
    VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION,
};

/// A paging misbehavior the root node injects into its `node_edges` reply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Misbehavior {
    /// Report `written` greater than the requested capacity.
    Overfill,
    /// Report `written` greater than `total - start`.
    PastEnd,
    /// Report `written == 0` while items still remain (no progress).
    StalledProgress,
    /// Report an enormous `out_total` (the preallocation-abort vector).
    InflatedTotal,
}

/// One immutable node: its optional value and its outgoing edges as
/// (label, child) in ascending label order.
#[derive(Clone, Default)]
struct Node {
    value: Option<Option<u64>>,
    edges: Vec<(u64, u64)>,
}

struct Model {
    nodes: Vec<Node>,
    len: usize,
    misbehavior: Option<Misbehavior>,
    /// High-water mark of the largest capacity any node_edges call requested.
    max_capacity: AtomicUsize,
}

struct Context {
    model: Arc<Model>,
}

fn resource(model: Arc<Model>) -> VtResource {
    VtResource {
        context: Arc::into_raw(Arc::new(Context { model })).cast_mut().cast(),
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
) -> u32 {
    if interface_id.is_null() || out_vtable.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    if (*interface_id).bytes != VT_DICTIONARY_INTERFACE_ID.bytes
        || minimum_version > VT_DICTIONARY_INTERFACE_VERSION
    {
        return VtStatus::Unsupported.to_raw();
    }
    let _ = context(raw);
    out_vtable.write((&DICTIONARY_VTABLE as *const VtDictionaryVTable).cast());
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn snapshot(raw: *mut c_void, out: *mut VtResource) -> u32 {
    if out.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    // The model is already immutable; a snapshot retains the same context.
    let source = context(raw);
    out.write(resource(Arc::clone(&source.model)));
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn root(_raw: *mut c_void, out_node: *mut u64) -> u32 {
    if out_node.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    out_node.write(0);
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn len(raw: *mut c_void, out_len: *mut usize, out_known: *mut u8) -> u32 {
    if out_len.is_null() || out_known.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    out_len.write(context(raw).model.len);
    out_known.write(1);
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn node_is_final(raw: *mut c_void, node: u64, out: *mut u8) -> u32 {
    if out.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    let Some(node) = context(raw).model.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    out.write(u8::from(node.value.is_some()));
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn node_value_u64(raw: *mut c_void, node: u64, out: *mut VtOptionalU64) -> u32 {
    if out.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    let Some(node) = context(raw).model.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    let value = match node.value {
        Some(Some(id)) => VtOptionalU64 {
            value: id,
            has_value: 1,
            reserved: [0; 7],
        },
        _ => VtOptionalU64::default(),
    };
    out.write(value);
    VtStatus::Ok.to_raw()
}

unsafe extern "C" fn node_transition(
    raw: *mut c_void,
    node: u64,
    label: u64,
    out_child: *mut u64,
    out_found: *mut u8,
) -> u32 {
    if out_child.is_null() || out_found.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    let Some(node) = context(raw).model.nodes.get(node as usize) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    match node
        .edges
        .iter()
        .find(|(edge_label, _)| *edge_label == label)
    {
        Some((_, child)) => {
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
    if (capacity != 0 && out_edges.is_null()) || out_written.is_null() || out_total.is_null() {
        return VtStatus::NullPointer.to_raw();
    }
    let model = &context(raw).model;
    model.max_capacity.fetch_max(capacity, Ordering::Relaxed);
    let node_index = node as usize;
    let Some(node) = model.nodes.get(node_index) else {
        return VtStatus::InvalidArgument.to_raw();
    };
    let total = node.edges.len();

    // Misbehavior is injected only at the root (node 0), so every other node's
    // paging — and the honest path everywhere when no mode is set — stays
    // faithful.
    if node_index == 0 {
        if let Some(misbehavior) = model.misbehavior {
            let honest = total.saturating_sub(start).min(capacity);
            let (written, reported_total) = match misbehavior {
                Misbehavior::Overfill => (capacity + 1, total),
                Misbehavior::PastEnd => (total.saturating_sub(start) + 1, total),
                Misbehavior::StalledProgress => (0, total.max(1)),
                Misbehavior::InflatedTotal => (honest, usize::MAX),
            };
            // Fill only addressable slots, but REPORT the adversarial counts;
            // the consumer must reject on the counts, not the buffer contents.
            for offset in 0..honest {
                let (label, child) = node.edges[start + offset];
                out_edges
                    .add(offset)
                    .write(VtDictionaryEdge { label, node: child });
            }
            out_written.write(written);
            out_total.write(reported_total);
            return VtStatus::Ok.to_raw();
        }
    }

    let mut written = 0usize;
    for (label, child) in node.edges.iter().skip(start).take(capacity) {
        out_edges.add(written).write(VtDictionaryEdge {
            label: *label,
            node: *child,
        });
        written += 1;
    }
    out_written.write(written);
    out_total.write(total);
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
    flags: dictionary_flags::IMMUTABLE | dictionary_flags::PARALLEL_REENTRANT,
    snapshot: Some(snapshot),
    root: Some(root),
    len: Some(len),
    node_is_final: Some(node_is_final),
    node_value_u64: Some(node_value_u64),
    node_transition: Some(node_transition),
    node_edges: Some(node_edges),
};

/// Owning handle: the resource is released when this drops.
pub struct HighDegreeDictionary {
    model: Arc<Model>,
    resource: VtResource,
    query: String,
}

impl HighDegreeDictionary {
    fn from_model(model: Model, query: String) -> Self {
        let model = Arc::new(model);
        let resource = resource(Arc::clone(&model));
        Self {
            model,
            resource,
            query,
        }
    }

    /// One root node carrying `degree` outgoing edges to `degree` distinct
    /// final children (labels 0..degree as Unicode scalars past ASCII, so
    /// the labels never collide with a query character). Query is a single
    /// scalar that matches the length-1 leaves within distance 1.
    pub fn single_node(degree: usize) -> Self {
        // Labels start at 0x100 so they are valid, distinct scalars that never
        // collide with a query character.
        Self::from_model(Self::single_node_model(degree, None), "\u{100}".to_string())
    }

    /// Build a single-root model of `degree` edges, optionally injecting a
    /// paging misbehavior at the root.
    fn single_node_model(degree: usize, misbehavior: Option<Misbehavior>) -> Model {
        let mut nodes = vec![Node::default()];
        for index in 0..degree {
            let child = nodes.len() as u64;
            let label = 0x100u64 + index as u64;
            nodes[0].edges.push((label, child));
            nodes.push(Node {
                value: Some(Some(index as u64)),
                edges: vec![],
            });
        }
        Model {
            len: degree,
            nodes,
            misbehavior,
            max_capacity: AtomicUsize::new(0),
        }
    }

    /// `n` length-2 terms sharing a common first scalar, so the root has one
    /// edge and the second level has `n` edges — a star whose leaves all lie
    /// within distance 1 of the shared two-scalar query.
    pub fn star(n: usize) -> Self {
        let mut nodes = vec![Node::default()];
        let hub = nodes.len() as u64;
        nodes[0].edges.push((u64::from('z' as u32), hub));
        nodes.push(Node::default());
        for index in 0..n {
            let leaf = nodes.len() as u64;
            let label = 0x100u64 + index as u64;
            nodes[hub as usize].edges.push((label, leaf));
            nodes.push(Node {
                value: Some(Some(index as u64)),
                edges: vec![],
            });
        }
        Self::from_model(
            Model {
                len: n,
                nodes,
                misbehavior: None,
                max_capacity: AtomicUsize::new(0),
            },
            "z\u{100}".to_string(),
        )
    }

    /// A root of `degree` edges whose `node_edges` injects `misbehavior`.
    pub fn misbehaving(degree: usize, misbehavior: Misbehavior) -> Self {
        Self::from_model(
            Self::single_node_model(degree, Some(misbehavior)),
            "\u{100}".to_string(),
        )
    }

    /// `count` distinct terms each `length` ASCII scalars long, built as a
    /// proper trie. Matching them forces the FFI cursor's byte arena through
    /// many reallocations within a single batch fill — the fixture for
    /// arena-stability. Returns the handle and the exact term set (sorted) so
    /// a test can assert the recovered terms.
    ///
    /// The returned `query()` is empty. Do NOT query it at
    /// `max_distance >= length`: over a deep shared-prefix trie the automaton's
    /// cost grows with the distance band, and a kilobyte-scale `length` runs for
    /// minutes. To drive the arena, query one of the returned terms at a small
    /// fixed distance (e.g. 3) with an in-test banded oracle — that still forces
    /// the full realloc pressure while keeping the automaton cheap.
    pub fn long_terms(count: usize, length: usize) -> (Self, Vec<String>) {
        assert!(length >= 4, "length must leave room for a per-term suffix");
        let mut terms = Vec::with_capacity(count);
        for index in 0..count {
            // A shared 'a'-run prefix then a unique zero-padded suffix, so the
            // terms share deep structure yet stay distinct.
            let suffix = format!("{index:04}");
            let prefix_len = length - suffix.len();
            let mut term = "a".repeat(prefix_len);
            term.push_str(&suffix);
            terms.push(term);
        }
        terms.sort();

        // Build the trie.
        let mut nodes = vec![Node::default()];
        for (value, term) in terms.iter().enumerate() {
            let mut current = 0usize;
            for scalar in term.chars() {
                let label = u64::from(scalar as u32);
                let existing = nodes[current]
                    .edges
                    .iter()
                    .find(|(edge_label, _)| *edge_label == label)
                    .map(|(_, child)| *child as usize);
                current = match existing {
                    Some(child) => child,
                    None => {
                        let child = nodes.len();
                        nodes[current].edges.push((label, child as u64));
                        nodes.push(Node::default());
                        child
                    }
                };
            }
            nodes[current].value = Some(Some(value as u64));
        }
        // Edges must be enumerated in ascending label order (the paging laws
        // and the honest node_edges both assume it).
        for node in &mut nodes {
            node.edges.sort_by_key(|(label, _)| *label);
        }
        let handle = Self::from_model(
            Model {
                len: count,
                nodes,
                misbehavior: None,
                max_capacity: AtomicUsize::new(0),
            },
            String::new(),
        );
        (handle, terms)
    }

    pub fn resource(&self) -> VtResource {
        self.resource
    }

    /// The query string these terms are built to match.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Directly invoke the root's `node_edges` and return
    /// `(written, total, page_len)` — the raw reply the acceptance predicate
    /// consumes. `page_len` is the number of edges actually written into the
    /// buffer.
    pub fn raw_node_edges(&self, start: usize, capacity: usize) -> (usize, usize, usize) {
        let mut buffer = vec![VtDictionaryEdge::default(); capacity];
        let mut written = 0usize;
        let mut total = 0usize;
        let status = unsafe {
            node_edges(
                self.resource.context,
                0,
                start,
                buffer.as_mut_ptr(),
                capacity,
                &mut written,
                &mut total,
            )
        };
        assert_eq!(status, VtStatus::Ok.to_raw());
        let page_len = written.min(capacity);
        (written, total, page_len)
    }

    /// Page the root from 0 by the recommended batch and collect every label
    /// in delivery order — the executable lossless-decomposition walk.
    pub fn walk_all_labels(&self) -> Vec<u32> {
        let capacity = vinary_tree_interop::VT_RECOMMENDED_EDGE_BATCH;
        let mut labels = Vec::new();
        let mut start = 0usize;
        loop {
            let mut buffer = vec![VtDictionaryEdge::default(); capacity];
            let mut written = 0usize;
            let mut total = 0usize;
            let status = unsafe {
                node_edges(
                    self.resource.context,
                    0,
                    start,
                    buffer.as_mut_ptr(),
                    capacity,
                    &mut written,
                    &mut total,
                )
            };
            assert_eq!(status, VtStatus::Ok.to_raw());
            for edge in buffer.into_iter().take(written) {
                // Recover the 0-based edge index from the 0x100-based label.
                labels.push((edge.label - 0x100) as u32);
            }
            start += written;
            if start >= total || written == 0 {
                break;
            }
        }
        labels
    }

    /// The largest capacity any `node_edges` call has requested so far.
    pub fn max_edge_page_capacity(&self) -> usize {
        self.model.max_capacity.load(Ordering::Relaxed)
    }
}

impl Drop for HighDegreeDictionary {
    fn drop(&mut self) {
        unsafe { release(self.resource.context) };
    }
}
