//! `query_interface` handshake and additive-evolution contract.
//!
//! A hand-rolled provider (static vtables, atomic refcount, no dependencies)
//! exercises the negotiation rules the ABI reference publishes:
//!
//! - wrong interface identifier → `Unsupported`, output untouched;
//! - `minimum_version` above the provider's support → `Unsupported`;
//! - null identifier or output pointer → `NullPointer`;
//! - success writes a provider-owned vtable pointer and leaves the refcount
//!   unchanged (discovery is not retention);
//! - a FUTURE, larger interface vtable (extra trailing operations) remains
//!   consumable through the `v1` prefix, discovered via `struct_size` — the
//!   additive-evolution law.
//!
//! INVARIANT-HOOK: VT-QI-1 — interface identifiers are compared exactly.
//! INVARIANT-HOOK: VT-QI-2 — the `minimum_version` handshake gates discovery.
//! INVARIANT-HOOK: VT-QI-3 — null in, `NullPointer` out; no partial writes.
//!
//! Formal home: `docs/verification/tla/AbiResourceLifecycle.tla` (wave W1)
//! owns the retain/release protocol; this file pins the negotiation surface
//! the model abstracts.

use core::ffi::c_void;
use core::mem::size_of;
use core::sync::atomic::{AtomicUsize, Ordering};

use vinary_tree_interop::{
    VtDictionaryVTable, VtInterfaceId, VtResource, VtResourceVTable, VtStatus, VtUnitDomain,
    VtValueDomain, VT_ABI_VERSION, VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION,
    VT_WFST_INTERFACE_ID,
};

/// Provider context: one atomic ledger of live retains.
struct CountingContext {
    retains: AtomicUsize,
    releases: AtomicUsize,
}

unsafe extern "C" fn counting_retain(context: *mut c_void) {
    let counting = unsafe { &*(context as *const CountingContext) };
    counting.retains.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn counting_release(context: *mut c_void) {
    let counting = unsafe { &*(context as *const CountingContext) };
    counting.releases.fetch_add(1, Ordering::SeqCst);
}

/// A future dictionary vtable: the published `v1` prefix plus one trailing
/// operation, advertised through `struct_size` exactly as the evolution
/// policy prescribes.
#[repr(C)]
struct ExtendedDictionaryVTable {
    base: VtDictionaryVTable,
    node_degree_hint: Option<unsafe extern "C" fn(context: *mut c_void, node: u64) -> u64>,
}

unsafe extern "C" fn degree_hint(_context: *mut c_void, node: u64) -> u64 {
    node.wrapping_mul(2)
}

fn extended_dictionary_vtable() -> &'static ExtendedDictionaryVTable {
    static EXTENDED: ExtendedDictionaryVTable = ExtendedDictionaryVTable {
        base: VtDictionaryVTable {
            struct_size: size_of::<ExtendedDictionaryVTable>(),
            interface_version: VT_DICTIONARY_INTERFACE_VERSION,
            unit_domain: VtUnitDomain::Byte,
            value_domain: VtValueDomain::Unit,
            flags: 0,
            snapshot: None,
            root: None,
            len: None,
            node_is_final: None,
            node_value_u64: None,
            node_transition: None,
            node_edges: None,
        },
        node_degree_hint: Some(degree_hint),
    };
    &EXTENDED
}

unsafe extern "C" fn counting_query_interface(
    _context: *mut c_void,
    interface_id: *const VtInterfaceId,
    minimum_version: u32,
    out_vtable: *mut *const c_void,
) -> VtStatus {
    if interface_id.is_null() || out_vtable.is_null() {
        return VtStatus::NullPointer;
    }
    let requested = unsafe { *interface_id };
    if requested != VT_DICTIONARY_INTERFACE_ID {
        return VtStatus::Unsupported;
    }
    if minimum_version > VT_DICTIONARY_INTERFACE_VERSION {
        return VtStatus::Unsupported;
    }
    unsafe {
        *out_vtable = (extended_dictionary_vtable() as *const ExtendedDictionaryVTable).cast();
    }
    VtStatus::Ok
}

static RESOURCE_VTABLE: VtResourceVTable = VtResourceVTable {
    struct_size: size_of::<VtResourceVTable>(),
    abi_version: VT_ABI_VERSION,
    reserved: 0,
    retain: Some(counting_retain),
    release: Some(counting_release),
    query_interface: Some(counting_query_interface),
};

struct Provider {
    context: Box<CountingContext>,
}

impl Provider {
    fn new() -> Self {
        Self {
            context: Box::new(CountingContext {
                retains: AtomicUsize::new(0),
                releases: AtomicUsize::new(0),
            }),
        }
    }

    fn resource(&self) -> VtResource {
        VtResource {
            context: (&*self.context as *const CountingContext as *mut CountingContext).cast(),
            vtable: &RESOURCE_VTABLE,
        }
    }

    fn retains(&self) -> usize {
        self.context.retains.load(Ordering::SeqCst)
    }

    fn releases(&self) -> usize {
        self.context.releases.load(Ordering::SeqCst)
    }
}

fn query(
    resource: VtResource,
    id: *const VtInterfaceId,
    minimum_version: u32,
    out: *mut *const c_void,
) -> VtStatus {
    let vtable = unsafe { &*resource.vtable };
    let query_interface = vtable
        .query_interface
        .expect("test provider always publishes query_interface");
    unsafe { query_interface(resource.context, id, minimum_version, out) }
}

const POISON: *const c_void = 0xDEAD_BEEFusize as *const c_void;

#[test]
fn base_vtable_advertises_its_own_size_and_version() {
    let resource = Provider::new();
    let vtable = unsafe { &*resource.resource().vtable };
    assert_eq!(vtable.struct_size, size_of::<VtResourceVTable>());
    assert_eq!(vtable.abi_version, VT_ABI_VERSION);
    assert_eq!(vtable.reserved, 0);
}

#[test]
fn wrong_interface_id_is_unsupported_and_writes_nothing() {
    let provider = Provider::new();
    let mut out = POISON;
    let status = query(
        provider.resource(),
        &VT_WFST_INTERFACE_ID,
        1,
        &mut out,
    );
    assert_eq!(status, VtStatus::Unsupported);
    assert_eq!(out, POISON, "a failed negotiation must not touch the output");
    assert_eq!(provider.retains(), 0, "discovery must not retain");
}

#[test]
fn unknown_interface_id_is_unsupported() {
    let provider = Provider::new();
    let unknown = VtInterfaceId {
        bytes: *b"vt.not-a-thing.0",
    };
    let mut out = POISON;
    assert_eq!(
        query(provider.resource(), &unknown, 1, &mut out),
        VtStatus::Unsupported
    );
    assert_eq!(out, POISON);
}

#[test]
fn minimum_version_above_support_is_unsupported() {
    let provider = Provider::new();
    let mut out = POISON;
    let status = query(
        provider.resource(),
        &VT_DICTIONARY_INTERFACE_ID,
        VT_DICTIONARY_INTERFACE_VERSION + 1,
        &mut out,
    );
    assert_eq!(status, VtStatus::Unsupported);
    assert_eq!(out, POISON);
}

#[test]
fn null_arguments_are_null_pointer_errors() {
    let provider = Provider::new();
    let mut out = POISON;
    assert_eq!(
        query(provider.resource(), core::ptr::null(), 1, &mut out),
        VtStatus::NullPointer
    );
    assert_eq!(out, POISON);
    assert_eq!(
        query(
            provider.resource(),
            &VT_DICTIONARY_INTERFACE_ID,
            1,
            core::ptr::null_mut()
        ),
        VtStatus::NullPointer
    );
}

#[test]
fn successful_negotiation_yields_a_prefix_consumable_future_vtable() {
    let provider = Provider::new();
    let mut out: *const c_void = core::ptr::null();
    let status = query(provider.resource(), &VT_DICTIONARY_INTERFACE_ID, 1, &mut out);
    assert_eq!(status, VtStatus::Ok);
    assert!(!out.is_null());

    // The consumer knows only the published v1 layout: read through the
    // prefix, discover the extension by size.
    let prefix = unsafe { &*(out as *const VtDictionaryVTable) };
    assert_eq!(prefix.interface_version, VT_DICTIONARY_INTERFACE_VERSION);
    assert_eq!(prefix.unit_domain, VtUnitDomain::Byte);
    assert_eq!(prefix.value_domain, VtValueDomain::Unit);
    assert!(
        prefix.struct_size > size_of::<VtDictionaryVTable>(),
        "the provider advertises a future vtable strictly larger than v1"
    );

    // A v1 consumer stops here. A consumer that KNOWS the extension may use
    // struct_size as the gate — the additive-evolution law end-to-end:
    assert!(prefix.struct_size >= size_of::<ExtendedDictionaryVTable>());
    let extended = unsafe { &*(out as *const ExtendedDictionaryVTable) };
    let hint = extended
        .node_degree_hint
        .expect("extension operation present in the extended table");
    assert_eq!(unsafe { hint(provider.resource().context, 21) }, 42);
}

#[test]
fn retain_release_ledger_balances() {
    let provider = Provider::new();
    let resource = provider.resource();
    let vtable = unsafe { &*resource.vtable };
    let retain = vtable.retain.expect("retain published");
    let release = vtable.release.expect("release published");

    // "Copying the two words does not retain": the copy below is free, and
    // the ledger moves only through explicit retain/release pairs.
    let copied = resource;
    assert_eq!(provider.retains(), 0);
    for _ in 0..3 {
        unsafe { retain(copied.context) };
    }
    assert_eq!(provider.retains(), 3);
    for _ in 0..3 {
        unsafe { release(copied.context) };
    }
    assert_eq!(provider.releases(), 3);
    assert_eq!(
        provider.retains(),
        provider.releases(),
        "every owned retain must be released exactly once"
    );
}
