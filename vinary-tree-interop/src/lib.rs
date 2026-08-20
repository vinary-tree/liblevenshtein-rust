//! Stable resource ABI shared by modular vinary-tree language bindings.
//!
//! This crate is intentionally dependency-free. It defines layouts and
//! constants only; project crates own the safe wrappers and concrete resource
//! implementations.
//!
//! Status wire rule: every callback returns the RAW `u32` status on the Rust
//! side. [`VtStatus`] is the vocabulary type — producers encode with
//! [`VtStatus::to_raw`], consumers decode with [`VtStatus::from_raw`] and
//! treat out-of-range values as provider errors. This keeps a misbehaving
//! foreign provider's garbage discriminant a VALUE, never undefined
//! behavior.

#![no_std]
#![warn(missing_docs)]

use core::ffi::c_void;

/// Current base resource ABI version.
pub const VT_ABI_VERSION: u32 = 1;

/// Version of [`VtDictionaryVTable`].
pub const VT_DICTIONARY_INTERFACE_VERSION: u32 = 1;

/// Version of [`VtDictionaryVisitVTable`].
pub const VT_DICTIONARY_VISIT_INTERFACE_VERSION: u32 = 1;

/// Version of [`VtDictionaryGraphVTable`].
pub const VT_DICTIONARY_GRAPH_INTERFACE_VERSION: u32 = 1;

/// Version of [`VtDictionaryEntriesVTable`].
pub const VT_DICTIONARY_ENTRIES_INTERFACE_VERSION: u32 = 1;

/// Version of [`VtSnapshotIdentityVTable`].
pub const VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION: u32 = 1;

/// Version of [`VtWfstVTable`].
pub const VT_WFST_INTERFACE_VERSION: u32 = 1;

/// Recommended number of provider edges requested in one call.
pub const VT_RECOMMENDED_EDGE_BATCH: usize = 256;

/// Recommended number of WFST arcs requested in one call.
pub const VT_RECOMMENDED_ARC_BATCH: usize = 256;

/// Stable 128-bit identifier for the dictionary provider interface.
pub const VT_DICTIONARY_INTERFACE_ID: VtInterfaceId = VtInterfaceId {
    bytes: *b"vt.dictionary.v1",
};

/// Stable identifier for fused dictionary-node inspection.
///
/// This is a separate optional interface so providers compiled against the
/// original [`VtDictionaryVTable`] remain binary compatible.
pub const VT_DICTIONARY_VISIT_INTERFACE_ID: VtInterfaceId = VtInterfaceId {
    bytes: *b"vt.dict.visit.v1",
};

/// Stable identifier for a compact immutable dictionary-snapshot graph.
///
/// This optional interface lets a consumer validate and capture one complete
/// graph at snapshot acquisition instead of rediscovering nodes through
/// callbacks while matching. The returned slices remain owned by the retained
/// immutable snapshot resource.
pub const VT_DICTIONARY_GRAPH_INTERFACE_ID: VtInterfaceId = VtInterfaceId {
    bytes: *b"vt.dict.graph.v1",
};

/// Stable identifier for finite lexicographic dictionary-entry streaming.
///
/// This is a separate optional interface so the original dictionary vtable
/// and providers compiled against it remain binary compatible.
pub const VT_DICTIONARY_ENTRIES_INTERFACE_ID: VtInterfaceId = VtInterfaceId {
    bytes: *b"vt.dict.entry.v1",
};

/// Stable identifier for immutable snapshot identity metadata.
///
/// This optional interface lets consumers share caches across independently
/// retained resource handles that name the same producer revision.
pub const VT_SNAPSHOT_IDENTITY_INTERFACE_ID: VtInterfaceId = VtInterfaceId {
    bytes: *b"vt.snapshot.id.1",
};

/// Stable 128-bit identifier for the scalar WFST provider interface.
pub const VT_WFST_INTERFACE_ID: VtInterfaceId = VtInterfaceId {
    bytes: *b"vt.scalar-wfst.1",
};

/// Result status used by every interop callback.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VtStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// Entries are exhausted, or an entries reducer requests early stop.
    End = 1,
    /// An argument was invalid.
    InvalidArgument = 2,
    /// A required pointer was null.
    NullPointer = 3,
    /// The requested interface or operation is unsupported.
    Unsupported = 4,
    /// The provider reported an I/O error.
    IoError = 5,
    /// A resource was already closed.
    Closed = 6,
    /// A configured resource limit was exceeded.
    LimitExceeded = 7,
    /// A provider failed without a more specific portable status.
    ProviderError = 8,
    /// The cursor already has a live batch lease.
    BatchInUse = 9,
}

impl VtStatus {
    /// Return whether this status represents success.
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Encode this status as its raw wire value.
    pub const fn to_raw(self) -> u32 {
        self as u32
    }

    /// Decode a raw wire value, rejecting anything outside the published
    /// range.
    ///
    /// Every interop callback RETURNS `u32` on the Rust side: a status
    /// arriving from a foreign provider is untrusted input, and reading an
    /// out-of-range discriminant into this enum would be undefined behavior
    /// before any check could run. Consumers must decode with `from_raw`
    /// and treat `None` as a provider error; producers encode with
    /// [`VtStatus::to_raw`]. The C header keeps its `VtStatus` returns —
    /// C enums are integer-typed, so the ABI is identical and only the
    /// Rust-side type carries the validation obligation.
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Ok),
            1 => Some(Self::End),
            2 => Some(Self::InvalidArgument),
            3 => Some(Self::NullPointer),
            4 => Some(Self::Unsupported),
            5 => Some(Self::IoError),
            6 => Some(Self::Closed),
            7 => Some(Self::LimitExceeded),
            8 => Some(Self::ProviderError),
            9 => Some(Self::BatchInUse),
            _ => None,
        }
    }
}

/// Stable interface identifier.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VtInterfaceId {
    /// Identifier bytes. These are compared byte-for-byte.
    pub bytes: [u8; 16],
}

/// Two-word retained resource passed between independently built libraries.
///
/// A non-null resource owns one retain. Copying the two words does not retain
/// automatically: the receiver must call `retain` before storing another owned
/// copy and must eventually call `release` once for every owned retain.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VtResource {
    /// Provider-defined resource context.
    pub context: *mut c_void,
    /// Base resource vtable.
    pub vtable: *const VtResourceVTable,
}

impl VtResource {
    /// Null resource value used for output initialization.
    pub const NULL: Self = Self {
        context: core::ptr::null_mut(),
        vtable: core::ptr::null(),
    };

    /// Return whether either required word is null.
    pub fn is_null(self) -> bool {
        self.context.is_null() || self.vtable.is_null()
    }
}

/// Base resource vtable.
#[repr(C)]
pub struct VtResourceVTable {
    /// Size of this struct in bytes, for additive ABI evolution.
    pub struct_size: usize,
    /// Must equal [`VT_ABI_VERSION`].
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Add one owned retain for `context`.
    pub retain: Option<unsafe extern "C" fn(context: *mut c_void)>,
    /// Release one owned retain for `context`.
    pub release: Option<unsafe extern "C" fn(context: *mut c_void)>,
    /// Discover a versioned project-neutral interface vtable.
    ///
    /// On success, writes a provider-owned immutable vtable pointer to
    /// `out_vtable`. The pointer remains valid while the resource is retained.
    pub query_interface: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            interface_id: *const VtInterfaceId,
            minimum_version: u32,
            out_vtable: *mut *const c_void,
        ) -> u32,
    >,
}

/// Dictionary edge-label domain.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VtUnitDomain {
    /// Raw byte labels.
    Byte = 1,
    /// Unicode scalar labels encoded as their u32 scalar value.
    UnicodeScalar = 2,
    /// Unsigned 64-bit token labels.
    U64 = 3,
}

/// Dictionary value domain.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VtValueDomain {
    /// Set semantics with no attached value.
    Unit = 0,
    /// Optional unsigned 64-bit value.
    OptionalU64 = 1,
    /// Opaque byte payload. Reserved for a later interface version.
    Bytes = 2,
}

/// Provider callback/threading flags.
pub mod dictionary_flags {
    /// Calls may run concurrently and re-enter the provider.
    pub const PARALLEL_REENTRANT: u64 = 1 << 0;
    /// The dictionary uses suffix/substring distance semantics.
    pub const SUFFIX_BASED: u64 = 1 << 1;
    /// `snapshot` is already immutable and may retain itself directly.
    pub const IMMUTABLE: u64 = 1 << 2;
}

/// Optional u64 value returned for a final dictionary node.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtOptionalU64 {
    /// Value payload; meaningful only when `has_value` is one.
    pub value: u64,
    /// Zero or one.
    pub has_value: u8,
    /// Reserved; providers must write zero.
    pub reserved: [u8; 7],
}

/// One edge in a batched node expansion.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtDictionaryEdge {
    /// Label represented in the provider's [`VtUnitDomain`].
    pub label: u64,
    /// Provider-defined node identifier within the captured snapshot.
    pub node: u64,
}

/// One node in a compact immutable dictionary-snapshot graph.
///
/// `edge_start..edge_start + edge_len` indexes the graph view's edge slice.
/// `value_cursor` is an opaque provider token passed only to the graph
/// interface's value callback; it is deliberately independent of the dense
/// graph index and the base dictionary interface's node identifiers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtDictionaryGraphNode {
    /// First outgoing edge in [`VtDictionaryGraphView::edges`].
    pub edge_start: u64,
    /// Number of outgoing edges.
    pub edge_len: u64,
    /// Provider-defined value cursor valid for this retained snapshot.
    pub value_cursor: u64,
    /// Zero or one.
    pub is_final: u8,
    /// Reserved; providers must write zero.
    pub reserved: [u8; 7],
}

/// One edge in a compact immutable dictionary-snapshot graph.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtDictionaryGraphEdge {
    /// Label represented in the provider's [`VtUnitDomain`].
    pub label: u64,
    /// Zero-based target index in [`VtDictionaryGraphView::nodes`].
    pub target: u64,
}

/// Borrowed immutable graph slices owned by a retained snapshot resource.
///
/// A successful callback may use null pointers only for empty slices. The
/// consumer validates all counts, ranges, targets, labels, flags, reserved
/// bytes, and the root before publishing a safe traversal graph.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VtDictionaryGraphView {
    /// Contiguous graph nodes.
    pub nodes: *const VtDictionaryGraphNode,
    /// Number of graph nodes.
    pub node_count: usize,
    /// Contiguous graph edges.
    pub edges: *const VtDictionaryGraphEdge,
    /// Number of graph edges.
    pub edge_count: usize,
    /// Zero-based root node index.
    pub root: u64,
    /// Reserved; providers must write zero.
    pub reserved: u64,
}

impl Default for VtDictionaryGraphView {
    fn default() -> Self {
        Self {
            nodes: core::ptr::null(),
            node_count: 0,
            edges: core::ptr::null(),
            edge_count: 0,
            root: 0,
            reserved: 0,
        }
    }
}

/// Stable identity of one immutable producer revision.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VtSnapshotIdentity {
    /// Stable identifier of the shared producer for its process lifetime.
    pub producer: u64,
    /// Monotone semantic revision within `producer`.
    pub revision: u64,
}

/// Ordering guaranteed by a dictionary-entries cursor.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VtDictionaryEntryOrder {
    /// Unsigned unit-wise lexicographic order, with a prefix before extensions.
    Lexicographic = 1,
}

/// Flags describing optional fields in [`VtDictionaryEntriesInfo`].
pub mod dictionary_entries_info_flags {
    /// [`crate::VtDictionaryEntriesInfo::exact_len`] is present and exact.
    pub const EXACT_LEN: u64 = 1 << 0;
    /// [`crate::VtDictionaryEntriesInfo::identity`] is present and immutable.
    pub const SNAPSHOT_IDENTITY: u64 = 1 << 1;
}

/// One dictionary-entry descriptor into parallel unit and value arenas.
///
/// Offsets and lengths count arena elements rather than bytes. For the unit
/// value domain `value_len` is zero. For optional-u64 values it is zero for a
/// present entry without a mapped value and one for a mapped value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtDictionaryEntry {
    /// First unit of the key in the batch unit arena.
    pub unit_offset: usize,
    /// Number of units in the key.
    pub unit_len: usize,
    /// First value in the compact u64 value arena.
    pub value_offset: usize,
    /// Zero or one in version 1.
    pub value_len: usize,
    /// Reserved; providers must write zero.
    pub reserved: u64,
}

/// Hard upper bounds for one dictionary-entry batch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtDictionaryEntryBatchLimits {
    /// Maximum descriptors; must be nonzero.
    pub max_entries: usize,
    /// Maximum unit-arena elements.
    pub max_units: usize,
    /// Maximum value-arena elements.
    pub max_values: usize,
    /// Reserved; consumers must write zero.
    pub reserved: u64,
}

/// Cursor-owned borrowed batch returned by the entries interface.
///
/// `units` points to `u8`, `u32`, or `u64` elements according to the captured
/// unit-domain discriminant. All pointers remain valid only for the live batch
/// lease identified by `generation`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VtDictionaryEntryBatchView {
    /// Contiguous entry descriptors, or null when `entry_count` is zero.
    pub entries: *const VtDictionaryEntry,
    /// Number of entry descriptors.
    pub entry_count: usize,
    /// Contiguous unit arena, or null when `unit_count` is zero.
    pub units: *const c_void,
    /// Number of unit-domain elements.
    pub unit_count: usize,
    /// Compact u64 value arena, or null when `value_count` is zero.
    pub values: *const u64,
    /// Number of u64 values.
    pub value_count: usize,
    /// Nonzero, strictly increasing lease generation.
    pub generation: u64,
    /// Reserved; providers must write zero.
    pub reserved: u64,
}

impl Default for VtDictionaryEntryBatchView {
    fn default() -> Self {
        Self {
            entries: core::ptr::null(),
            entry_count: 0,
            units: core::ptr::null(),
            unit_count: 0,
            values: core::ptr::null(),
            value_count: 0,
            generation: 0,
            reserved: 0,
        }
    }
}

/// Immutable metadata captured when a dictionary-entry cursor is opened.
///
/// Domain and order fields intentionally remain raw integers: a consumer must
/// validate foreign discriminants before constructing the corresponding Rust
/// enums. Optional fields are meaningful only when their flag is set.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VtDictionaryEntriesInfo {
    /// Raw [`VtUnitDomain`] discriminant.
    pub unit_domain: u32,
    /// Raw [`VtValueDomain`] discriminant.
    pub value_domain: u32,
    /// Raw [`VtDictionaryEntryOrder`] discriminant.
    pub order: u32,
    /// Reserved; providers must write zero.
    pub reserved0: u32,
    /// Bitset from [`dictionary_entries_info_flags`].
    pub flags: u64,
    /// Exact entry count when `EXACT_LEN` is set; otherwise zero.
    pub exact_len: usize,
    /// Snapshot identity when `SNAPSHOT_IDENTITY` is set; otherwise zero.
    pub identity: VtSnapshotIdentity,
    /// Reserved; providers must write zero.
    pub reserved: [u64; 2],
}

/// Move-only two-word handle for one captured dictionary-entry stream.
///
/// The ABI does not make this type `Copy`: copying its words does not create a
/// second owned cursor. Pass ownership explicitly and call the cursor vtable's
/// `close` operation exactly once.
#[repr(C)]
#[derive(Debug)]
pub struct VtDictionaryEntriesCursor {
    /// Provider-defined cursor context.
    pub context: *mut c_void,
    /// Cursor vtable, valid until the cursor is closed.
    pub vtable: *const VtDictionaryEntriesVTable,
}

impl VtDictionaryEntriesCursor {
    /// Null cursor value used for output initialization.
    pub const NULL: Self = Self {
        context: core::ptr::null_mut(),
        vtable: core::ptr::null(),
    };

    /// Return whether either required word is null.
    pub fn is_null(&self) -> bool {
        self.context.is_null() || self.vtable.is_null()
    }
}

/// Callback used by [`VtDictionaryEntriesVTable::reduce`].
///
/// Return raw status values. `Ok` continues, `End` stops successfully, known
/// errors abort and propagate, and unknown values are treated as invalid input.
pub type VtDictionaryEntryReducer = unsafe extern "C" fn(
    reducer_context: *mut c_void,
    batch: *const VtDictionaryEntryBatchView,
) -> u32;

/// Versioned immutable dictionary-snapshot interface.
///
/// Node identifiers are valid only while the snapshot resource is retained.
/// Implementations must make `snapshot` O(1) with structural sharing (or an
/// equivalent immutable revision); copying the whole dictionary or holding a
/// long-lived read lock violates the interface contract.
#[repr(C)]
pub struct VtDictionaryVTable {
    /// Size of this struct in bytes, for additive interface evolution.
    pub struct_size: usize,
    /// Must be at least [`VT_DICTIONARY_INTERFACE_VERSION`].
    pub interface_version: u32,
    /// One of [`VtUnitDomain`].
    pub unit_domain: VtUnitDomain,
    /// One of [`VtValueDomain`].
    pub value_domain: VtValueDomain,
    /// Bitset from [`dictionary_flags`].
    pub flags: u64,
    /// Capture the revision visible at operation start.
    pub snapshot:
        Option<unsafe extern "C" fn(context: *mut c_void, out_snapshot: *mut VtResource) -> u32>,
    /// Return the root node identifier for this immutable resource.
    pub root: Option<unsafe extern "C" fn(context: *mut c_void, out_node: *mut u64) -> u32>,
    /// Return the number of stored terms when cheaply available.
    pub len: Option<
        unsafe extern "C" fn(context: *mut c_void, out_len: *mut usize, out_known: *mut u8) -> u32,
    >,
    /// Test whether a node terminates a stored term.
    pub node_is_final:
        Option<unsafe extern "C" fn(context: *mut c_void, node: u64, out_is_final: *mut u8) -> u32>,
    /// Read the optional u64 value of a final node.
    pub node_value_u64: Option<
        unsafe extern "C" fn(context: *mut c_void, node: u64, out_value: *mut VtOptionalU64) -> u32,
    >,
    /// Follow one edge without enumerating siblings.
    pub node_transition: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            node: u64,
            label: u64,
            out_child: *mut u64,
            out_found: *mut u8,
        ) -> u32,
    >,
    /// Copy a page of outgoing edges into caller-owned contiguous storage.
    ///
    /// `start` is an edge offset. `out_total` receives the complete edge count,
    /// while `out_written` receives at most `capacity`. A consumer can therefore
    /// expand common nodes in one ABI call and page unusually high-degree nodes.
    pub node_edges: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            node: u64,
            start: usize,
            out_edges: *mut VtDictionaryEdge,
            capacity: usize,
            out_written: *mut usize,
            out_total: *mut usize,
        ) -> u32,
    >,
}

/// Optional fused dictionary-node inspection interface.
///
/// Consumers that need both finality and outgoing edges can discover this
/// interface through [`VtResourceVTable::query_interface`]. A provider can
/// then validate and lock a node once instead of once per property. Providers
/// that do not expose it remain usable through [`VtDictionaryVTable`].
#[repr(C)]
pub struct VtDictionaryVisitVTable {
    /// Size of this struct in bytes, for additive interface evolution.
    pub struct_size: usize,
    /// Must be at least [`VT_DICTIONARY_VISIT_INTERFACE_VERSION`].
    pub interface_version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Read finality and copy one page of outgoing edges in one call.
    pub node_visit: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            node: u64,
            start: usize,
            out_is_final: *mut u8,
            out_edges: *mut VtDictionaryEdge,
            capacity: usize,
            out_written: *mut usize,
            out_total: *mut usize,
        ) -> u32,
    >,
}

/// Optional compact immutable dictionary-snapshot graph interface.
///
/// Providers expose this interface only on immutable snapshot resources. The
/// graph callback returns borrowed slices whose lifetime is exactly the
/// lifetime of the retained resource. Consumers copy or otherwise validate
/// the full view before traversal, so no provider callback, lock, or atomic
/// cache publication is required while enumerating edges.
#[repr(C)]
pub struct VtDictionaryGraphVTable {
    /// Size of this struct in bytes, for additive interface evolution.
    pub struct_size: usize,
    /// Must be at least [`VT_DICTIONARY_GRAPH_INTERFACE_VERSION`].
    pub interface_version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Borrow the complete compact graph owned by this immutable snapshot.
    pub graph: Option<
        unsafe extern "C" fn(context: *mut c_void, out_graph: *mut VtDictionaryGraphView) -> u32,
    >,
    /// Read the optional u64 value for a final graph node's opaque cursor.
    ///
    /// The provider must validate that the cursor belongs to this retained
    /// graph before translating it to or dereferencing backend state.
    pub node_value_u64: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            value_cursor: u64,
            out_value: *mut VtOptionalU64,
        ) -> u32,
    >,
}

/// Optional immutable snapshot-identity interface.
///
/// Providers return this vtable only for immutable resources. Consumers must
/// treat identity as an optimization hint scoped to the current process; it is
/// not a persistent or globally unique identifier.
#[repr(C)]
pub struct VtSnapshotIdentityVTable {
    /// Size of this struct in bytes, for additive interface evolution.
    pub struct_size: usize,
    /// Must be at least [`VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION`].
    pub interface_version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Read the immutable resource's producer/revision identity.
    pub identity: Option<
        unsafe extern "C" fn(context: *mut c_void, out_identity: *mut VtSnapshotIdentity) -> u32,
    >,
}

/// Optional finite lexicographic dictionary-entry streaming interface.
///
/// `open` captures one immutable revision and returns an independent owned
/// cursor. Each cursor permits at most one live borrowed batch lease. Full
/// lifetime, limit, cancellation, reducer, and validation rules are specified
/// in the authoritative ABI reference.
#[repr(C)]
pub struct VtDictionaryEntriesVTable {
    /// Size of this struct in bytes, for additive interface evolution.
    pub struct_size: usize,
    /// Must be at least [`VT_DICTIONARY_ENTRIES_INTERFACE_VERSION`].
    pub interface_version: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Capture one immutable revision and create an owned cursor.
    pub open: Option<
        unsafe extern "C" fn(
            resource_context: *mut c_void,
            out_cursor: *mut VtDictionaryEntriesCursor,
            out_info: *mut VtDictionaryEntriesInfo,
        ) -> u32,
    >,
    /// Borrow the next nonempty batch, or return `End` after exhaustion.
    pub next_batch: Option<
        unsafe extern "C" fn(
            cursor: *mut VtDictionaryEntriesCursor,
            limits: *const VtDictionaryEntryBatchLimits,
            out_batch: *mut VtDictionaryEntryBatchView,
        ) -> u32,
    >,
    /// Settle the live lease with its exact generation.
    pub release_batch: Option<
        unsafe extern "C" fn(cursor: *mut VtDictionaryEntriesCursor, generation: u64) -> u32,
    >,
    /// Stream leased batches through one callback at a time.
    pub reduce: Option<
        unsafe extern "C" fn(
            cursor: *mut VtDictionaryEntriesCursor,
            limits: *const VtDictionaryEntryBatchLimits,
            reducer: Option<VtDictionaryEntryReducer>,
            reducer_context: *mut c_void,
            out_count: *mut usize,
        ) -> u32,
    >,
    /// Idempotently request sticky exhaustion without invalidating a lease.
    pub cancel: Option<unsafe extern "C" fn(cursor: *mut VtDictionaryEntriesCursor) -> u32>,
    /// Free a lease-free cursor and zero both handle words.
    pub close: Option<unsafe extern "C" fn(cursor: *mut VtDictionaryEntriesCursor) -> u32>,
}

/// Portable scalar semiring used by a [`VtWfstVTable`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VtWeightDomain {
    /// Min-plus costs, where positive infinity is zero and 0 is one.
    TropicalF64 = 1,
    /// Log-semiring scalar representation.
    LogF64 = 2,
    /// Direct probability representation.
    ProbabilityF64 = 3,
    /// Max-plus scores.
    ArcticF64 = 4,
    /// Signed tropical representation.
    SignedTropicalF64 = 5,
    /// Path-count representation carried in an f64 slot.
    CountF64 = 6,
    /// Boolean values represented as exactly 0 or 1.
    BooleanF64 = 7,
}

/// Scalar-WFST provider callback/threading flags.
pub mod wfst_flags {
    /// Calls may run concurrently and re-enter the provider.
    pub const PARALLEL_REENTRANT: u64 = 1 << 0;
    /// `snapshot` may retain the resource directly.
    pub const IMMUTABLE: u64 = 1 << 1;
    /// States or arcs may be expanded on demand.
    pub const LAZY: u64 = 1 << 2;
    /// The provider guarantees that the reachable graph is acyclic.
    pub const ACYCLIC: u64 = 1 << 3;
}

/// One scalar weighted transition in a caller-owned page.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VtWfstArc {
    /// Input label value; meaningful only when `has_input` is one.
    pub input_label: u64,
    /// Output label value; meaningful only when `has_output` is one.
    pub output_label: u64,
    /// Provider-defined target state within the retained snapshot.
    pub target_state: u64,
    /// Scalar semiring value interpreted according to `weight_domain`.
    pub weight: f64,
    /// Zero for input epsilon, one for a present input label.
    pub has_input: u8,
    /// Zero for output epsilon, one for a present output label.
    pub has_output: u8,
    /// Reserved; providers must write zero.
    pub reserved: [u8; 6],
}

/// Versioned immutable scalar weighted-finite-state-transducer interface.
///
/// State identifiers are valid only while the captured resource is retained.
/// A lazy implementation may expand a state during `state_info` or
/// `state_arcs`. Project-owned providers advertising parallel/reentrant access
/// must make those expansions safe for independent concurrent callers.
#[repr(C)]
pub struct VtWfstVTable {
    /// Size of this struct in bytes, for additive interface evolution.
    pub struct_size: usize,
    /// Must be at least [`VT_WFST_INTERFACE_VERSION`].
    pub interface_version: u32,
    /// Input and output label representation.
    pub unit_domain: VtUnitDomain,
    /// Scalar semiring representation.
    pub weight_domain: VtWeightDomain,
    /// Reserved; must be zero.
    pub reserved: u32,
    /// Bitset from [`wfst_flags`].
    pub flags: u64,
    /// Capture the revision visible at operation start.
    pub snapshot:
        Option<unsafe extern "C" fn(context: *mut c_void, out_snapshot: *mut VtResource) -> u32>,
    /// Return the initial state.
    pub start: Option<unsafe extern "C" fn(context: *mut c_void, out_state: *mut u64) -> u32>,
    /// Return the number of states when cheaply known.
    pub num_states: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            out_count: *mut usize,
            out_known: *mut u8,
        ) -> u32,
    >,
    /// Validate a state and read its finality/final weight.
    pub state_info: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            state: u64,
            out_valid: *mut u8,
            out_is_final: *mut u8,
            out_final_weight: *mut f64,
        ) -> u32,
    >,
    /// Copy a page of outgoing arcs into caller-owned contiguous storage.
    pub state_arcs: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            state: u64,
            start: usize,
            out_arcs: *mut VtWfstArc,
            capacity: usize,
            out_written: *mut usize,
            out_total: *mut usize,
        ) -> u32,
    >,
}

const _: () = {
    assert!(core::mem::size_of::<VtResource>() == 2 * core::mem::size_of::<usize>());
    assert!(core::mem::align_of::<VtResource>() == core::mem::align_of::<usize>());
    // The POD payload types are built solely from u64/f64/u8 fields, so their
    // layout must be identical on every supported target; any drift here is an
    // ABI break for independently built producers and consumers.
    assert!(core::mem::size_of::<VtInterfaceId>() == 16);
    assert!(core::mem::align_of::<VtInterfaceId>() == 1);
    assert!(core::mem::size_of::<VtOptionalU64>() == 16);
    assert!(core::mem::size_of::<VtDictionaryEdge>() == 16);
    assert!(core::mem::size_of::<VtDictionaryGraphNode>() == 32);
    assert!(core::mem::size_of::<VtDictionaryGraphEdge>() == 16);
    assert!(core::mem::size_of::<VtDictionaryEntriesCursor>() == 2 * core::mem::size_of::<usize>());
    assert!(core::mem::size_of::<VtSnapshotIdentity>() == 16);
    assert!(core::mem::size_of::<VtWfstArc>() == 40);
    // Absent vtable operations are NULL on the C side: the Option-of-function
    // null-pointer optimization is load-bearing for the whole vtable ABI.
    assert!(
        core::mem::size_of::<Option<unsafe extern "C" fn(*mut core::ffi::c_void)>>()
            == core::mem::size_of::<*const core::ffi::c_void>()
    );
};
