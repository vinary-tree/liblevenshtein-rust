//! Stable resource ABI shared by modular vinary-tree language bindings.
//!
//! This crate is intentionally dependency-free. It defines layouts and
//! constants only; project crates own the safe wrappers and concrete resource
//! implementations.

#![no_std]
#![warn(missing_docs)]

use core::ffi::c_void;

/// Current base resource ABI version.
pub const VT_ABI_VERSION: u32 = 1;

/// Version of [`VtDictionaryVTable`].
pub const VT_DICTIONARY_INTERFACE_VERSION: u32 = 1;

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
    /// An iterator or paginated operation is exhausted.
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
}

impl VtStatus {
    /// Return whether this status represents success.
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
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
        ) -> VtStatus,
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
    pub snapshot: Option<
        unsafe extern "C" fn(context: *mut c_void, out_snapshot: *mut VtResource) -> VtStatus,
    >,
    /// Return the root node identifier for this immutable resource.
    pub root: Option<unsafe extern "C" fn(context: *mut c_void, out_node: *mut u64) -> VtStatus>,
    /// Return the number of stored terms when cheaply available.
    pub len: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            out_len: *mut usize,
            out_known: *mut u8,
        ) -> VtStatus,
    >,
    /// Test whether a node terminates a stored term.
    pub node_is_final: Option<
        unsafe extern "C" fn(context: *mut c_void, node: u64, out_is_final: *mut u8) -> VtStatus,
    >,
    /// Read the optional u64 value of a final node.
    pub node_value_u64: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            node: u64,
            out_value: *mut VtOptionalU64,
        ) -> VtStatus,
    >,
    /// Follow one edge without enumerating siblings.
    pub node_transition: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            node: u64,
            label: u64,
            out_child: *mut u64,
            out_found: *mut u8,
        ) -> VtStatus,
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
        ) -> VtStatus,
    >,
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
    pub snapshot: Option<
        unsafe extern "C" fn(context: *mut c_void, out_snapshot: *mut VtResource) -> VtStatus,
    >,
    /// Return the initial state.
    pub start: Option<unsafe extern "C" fn(context: *mut c_void, out_state: *mut u64) -> VtStatus>,
    /// Return the number of states when cheaply known.
    pub num_states: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            out_count: *mut usize,
            out_known: *mut u8,
        ) -> VtStatus,
    >,
    /// Validate a state and read its finality/final weight.
    pub state_info: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            state: u64,
            out_valid: *mut u8,
            out_is_final: *mut u8,
            out_final_weight: *mut f64,
        ) -> VtStatus,
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
        ) -> VtStatus,
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
    assert!(core::mem::size_of::<VtWfstArc>() == 40);
    // Absent vtable operations are NULL on the C side: the Option-of-function
    // null-pointer optimization is load-bearing for the whole vtable ABI.
    assert!(
        core::mem::size_of::<Option<unsafe extern "C" fn(*mut core::ffi::c_void)>>()
            == core::mem::size_of::<*const core::ffi::c_void>()
    );
};
