//! Compact linear-memory ABI used by the Node WASI facade.

use libdictenstein::bindings::{
    BindingEntries, BindingTerm, BindingUnitDomain, DynamicDawgBinding, OwnedDictionaryResource,
    PersistentARTrieBinding,
};
use liblevenshtein::bindings::{
    MatchBatch, MatchTerm, QueryCursor, QueryOrder, ResourceTransducer,
};
use liblevenshtein::transducer::Algorithm;
use lling_llang::bindings::OwnedWfstResource;
use lling_llang::prelude::{MutableWfst, TropicalWeight, VectorWfst, Wfst};
use std::collections::HashMap;
use std::ffi::c_void;
use std::slice;
use std::str;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use vinary_tree_interop::{
    VtResource, VtStatus, VtWfstArc, VtWfstVTable, VT_WFST_INTERFACE_ID, VT_WFST_INTERFACE_VERSION,
};

const FAILURE: u32 = u32::MAX;
const MATCH_RECORD_SIZE: usize = 32;
const ENTRY_RECORD_HEADER_SIZE: usize = 24;

#[derive(Clone)]
enum Dictionary {
    Dynamic(DynamicDawgBinding),
    Persistent(PersistentARTrieBinding),
}

impl Dictionary {
    fn resource(&self) -> OwnedDictionaryResource {
        match self {
            Self::Dynamic(value) => value.resource(),
            Self::Persistent(value) => value.resource(),
        }
    }

    fn insert_text(&self, term: &[u8], value: Option<u64>) -> Result<bool, String> {
        match self {
            Self::Dynamic(dictionary) => dictionary.insert_text(term, value),
            Self::Persistent(dictionary) => dictionary.insert_text(term, value),
        }
        .map_err(|error| error.to_string())
    }

    fn remove_text(&self, term: &[u8]) -> Result<bool, String> {
        match self {
            Self::Dynamic(dictionary) => dictionary.remove_text(term),
            Self::Persistent(dictionary) => dictionary.remove_text(term),
        }
        .map_err(|error| error.to_string())
    }

    fn value_text(&self, term: &[u8]) -> Result<Option<Option<u64>>, String> {
        match self {
            Self::Dynamic(dictionary) => dictionary.value_text(term),
            Self::Persistent(dictionary) => dictionary.value_text(term),
        }
        .map_err(|error| error.to_string())
    }

    fn insert_u64(&self, term: &[u64], value: Option<u64>) -> Result<bool, String> {
        match self {
            Self::Dynamic(dictionary) => dictionary.insert_u64(term, value),
            Self::Persistent(dictionary) => dictionary.insert_u64(term, value),
        }
        .map_err(|error| error.to_string())
    }

    fn remove_u64(&self, term: &[u64]) -> Result<bool, String> {
        match self {
            Self::Dynamic(dictionary) => dictionary.remove_u64(term),
            Self::Persistent(dictionary) => dictionary.remove_u64(term),
        }
        .map_err(|error| error.to_string())
    }

    fn value_u64(&self, term: &[u64]) -> Result<Option<Option<u64>>, String> {
        match self {
            Self::Dynamic(dictionary) => dictionary.value_u64(term),
            Self::Persistent(dictionary) => dictionary.value_u64(term),
        }
        .map_err(|error| error.to_string())
    }

    fn len(&self) -> usize {
        match self {
            Self::Dynamic(dictionary) => dictionary.len(),
            Self::Persistent(dictionary) => dictionary.len(),
        }
    }
}

struct Cursor {
    inner: QueryCursor,
    batch: MatchBatch,
    encoded: Vec<u8>,
}

struct EntryCursor {
    inner: BindingEntries,
    exact_len: usize,
    encoded: Vec<u8>,
}

struct WasiWfst {
    resource: OwnedWfstResource,
    encoded: Vec<u8>,
}

enum Handle {
    Dictionary(Dictionary),
    Transducer(ResourceTransducer),
    Cursor(Cursor),
    EntryCursor(EntryCursor),
    WfstBuilder(VectorWfst<char, TropicalWeight>),
    Wfst(WasiWfst),
}

struct Registry {
    next: u32,
    handles: HashMap<u32, Handle>,
    error: Vec<u8>,
}

impl Registry {
    fn new() -> Self {
        Self {
            next: 1,
            handles: HashMap::new(),
            error: Vec::new(),
        }
    }

    fn insert(&mut self, handle: Handle) -> u32 {
        let result = self.next;
        self.next = self.next.checked_add(1).unwrap_or(1);
        self.handles.insert(result, handle);
        result
    }

    fn fail(&mut self, error: impl std::fmt::Display) -> u32 {
        self.error.clear();
        self.error.extend_from_slice(error.to_string().as_bytes());
        FAILURE
    }
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::new()))
}

/// Lock the global registry, recovering from mutex poisoning.
///
/// The registry is a plain handle table (a `u32` counter, a handle map, and
/// an error buffer) with no invariant spanning a critical section: every
/// operation leaves the table structurally valid even if it unwinds
/// mid-call, so observing the state after a poisoned lock is always sound.
/// Recovering instead of unwrapping keeps this boundary panic-free — a
/// panic here would trap and kill the whole WASM instance instead of
/// reporting a status the caller can handle.
fn locked_registry() -> MutexGuard<'static, Registry> {
    registry().lock().unwrap_or_else(PoisonError::into_inner)
}

/// Decode a raw interop status, mapping anything but `Ok` to an error.
///
/// Interop callbacks return raw `u32` statuses on the Rust side; anything
/// outside the published range is untrusted provider output and must be
/// rejected rather than read into [`VtStatus`].
fn require_ok(raw: u32, operation: &str) -> Result<(), String> {
    match VtStatus::from_raw(raw) {
        Some(VtStatus::Ok) => Ok(()),
        Some(status) => Err(format!("{operation} failed: {status:?}")),
        None => Err(format!("{operation} returned an out-of-range status {raw}")),
    }
}

fn selected_domain(value: u32) -> Result<BindingUnitDomain, &'static str> {
    match value {
        0 => Ok(BindingUnitDomain::Byte),
        1 => Ok(BindingUnitDomain::UnicodeScalar),
        2 => Ok(BindingUnitDomain::U64),
        _ => Err("unknown unit domain"),
    }
}

fn selected_algorithm(value: u32) -> Result<Algorithm, &'static str> {
    match value {
        0 => Ok(Algorithm::Standard),
        1 => Ok(Algorithm::Transposition),
        2 => Ok(Algorithm::MergeAndSplit),
        3 => Ok(Algorithm::DamerauLevenshtein),
        _ => Err("unknown algorithm"),
    }
}

fn selected_duallity_kind(value: u32) -> Result<duallity::bindings::WfstKind, &'static str> {
    use duallity::bindings::WfstKind;
    match value {
        0 => Ok(WfstKind::Levenshtein),
        1 => Ok(WfstKind::UniversalStandard),
        2 => Ok(WfstKind::UniversalTransposition),
        3 => Ok(WfstKind::UniversalMergeAndSplit),
        4 => Ok(WfstKind::GeneralizedStandard),
        5 => Ok(WfstKind::GeneralizedTransposition),
        6 => Ok(WfstKind::GeneralizedMergeAndSplit),
        7 => Ok(WfstKind::GeneralizedPhonetic),
        8 => Ok(WfstKind::Fzf),
        _ => Err("unknown duallity WFST kind"),
    }
}

unsafe fn bytes<'a>(pointer: u32, length: u32) -> &'a [u8] {
    if length == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(pointer as *const u8, length as usize) }
    }
}

unsafe fn tokens(pointer: u32, length: u32) -> Result<Vec<u64>, &'static str> {
    let byte_length = length.checked_mul(8).ok_or("token byte length overflow")?;
    let raw = unsafe { bytes(pointer, byte_length) };
    Ok(raw
        .chunks_exact(8)
        .map(|chunk| {
            let mut word = [0; 8];
            word.copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect())
}

fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f64(buffer: &mut [u8], offset: usize, value: f64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

unsafe fn wfst_table(resource: VtResource) -> Result<*const VtWfstVTable, String> {
    if resource.is_null() {
        return Err("WFST resource is null".into());
    }
    let mut interface: *const c_void = std::ptr::null();
    let query = (*resource.vtable)
        .query_interface
        .ok_or("resource has no query_interface")?;
    let status = query(
        resource.context,
        &VT_WFST_INTERFACE_ID,
        VT_WFST_INTERFACE_VERSION,
        &mut interface,
    );
    if VtStatus::from_raw(status) != Some(VtStatus::Ok) || interface.is_null() {
        return Err("resource has no compatible scalar WFST interface".into());
    }
    Ok(interface.cast())
}

/// Allocate a caller-owned linear-memory byte buffer.
#[no_mangle]
pub extern "C" fn vt_alloc(length: u32) -> u32 {
    let value = vec![0_u8; length as usize].into_boxed_slice();
    Box::into_raw(value) as *mut u8 as u32
}

/// Release a buffer previously returned by `vt_alloc`.
#[no_mangle]
pub unsafe extern "C" fn vt_dealloc(pointer: u32, length: u32) {
    if pointer != 0 {
        let raw = std::ptr::slice_from_raw_parts_mut(pointer as *mut u8, length as usize);
        drop(unsafe { Box::from_raw(raw) });
    }
}

/// Pointer to the most recent error message.
#[no_mangle]
pub extern "C" fn vt_error_pointer() -> u32 {
    locked_registry().error.as_ptr() as u32
}

/// Byte length of the most recent error message.
#[no_mangle]
pub extern "C" fn vt_error_length() -> u32 {
    locked_registry().error.len() as u32
}

/// Create an in-memory DynamicDAWG and return its handle.
#[no_mangle]
pub extern "C" fn vt_dynamic_dawg_new(domain: u32) -> u32 {
    let Ok(domain) = selected_domain(domain) else {
        return locked_registry().fail("unknown unit domain");
    };
    locked_registry().insert(Handle::Dictionary(Dictionary::Dynamic(
        DynamicDawgBinding::new(domain),
    )))
}

/// Create a filesystem-backed persistent ARTrie at a preopened WASI path.
#[no_mangle]
pub unsafe extern "C" fn vt_persistent_artrie_create(
    path_pointer: u32,
    path_length: u32,
    domain: u32,
) -> u32 {
    let result = (|| {
        let path = str::from_utf8(unsafe { bytes(path_pointer, path_length) })
            .map_err(|_| "path is not UTF-8")?;
        let domain = selected_domain(domain)?;
        PersistentARTrieBinding::create(path, domain).map_err(|error| error.to_string())
    })();
    let mut registry = locked_registry();
    match result {
        Ok(dictionary) => registry.insert(Handle::Dictionary(Dictionary::Persistent(dictionary))),
        Err(error) => registry.fail(error),
    }
}

/// Open a filesystem-backed persistent ARTrie at a preopened WASI path.
#[no_mangle]
pub unsafe extern "C" fn vt_persistent_artrie_open(
    path_pointer: u32,
    path_length: u32,
    domain: u32,
) -> u32 {
    let result = (|| {
        let path = str::from_utf8(unsafe { bytes(path_pointer, path_length) })
            .map_err(|_| "path is not UTF-8")?;
        let domain = selected_domain(domain)?;
        PersistentARTrieBinding::open(path, domain).map_err(|error| error.to_string())
    })();
    let mut registry = locked_registry();
    match result {
        Ok(dictionary) => registry.insert(Handle::Dictionary(Dictionary::Persistent(dictionary))),
        Err(error) => registry.fail(error),
    }
}

/// Return the current number of visible dictionary terms.
#[no_mangle]
pub extern "C" fn vt_dictionary_len(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => dictionary.len() as u32,
        _ => registry.fail("invalid dictionary handle"),
    }
}

/// Insert or update a text/byte term; returns one for a new term and zero for an update.
#[no_mangle]
pub unsafe extern "C" fn vt_dictionary_put_text(
    handle: u32,
    term_pointer: u32,
    term_length: u32,
    value_present: u32,
    value: u64,
) -> u32 {
    let term = unsafe { bytes(term_pointer, term_length) };
    let mut registry = locked_registry();
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => {
            dictionary.insert_text(term, (value_present != 0).then_some(value))
        }
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(value) => u32::from(value),
        Err(error) => registry.fail(error),
    }
}

/// Remove a text/byte term; returns one when the term existed.
#[no_mangle]
pub unsafe extern "C" fn vt_dictionary_remove_text(
    handle: u32,
    term_pointer: u32,
    term_length: u32,
) -> u32 {
    let term = unsafe { bytes(term_pointer, term_length) };
    let mut registry = locked_registry();
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => dictionary.remove_text(term),
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(value) => u32::from(value),
        Err(error) => registry.fail(error),
    }
}

/// Write found/value-present/value to a caller-owned 16-byte lookup record.
#[no_mangle]
pub unsafe extern "C" fn vt_dictionary_get_text(
    handle: u32,
    term_pointer: u32,
    term_length: u32,
    output_pointer: u32,
) -> u32 {
    let term = unsafe { bytes(term_pointer, term_length) };
    let mut registry = locked_registry();
    if output_pointer == 0 {
        return registry.fail("output pointer is null");
    }
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => dictionary.value_text(term),
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(value) => {
            let output = unsafe { slice::from_raw_parts_mut(output_pointer as *mut u8, 16) };
            put_u32(output, 0, u32::from(value.is_some()));
            put_u32(output, 4, u32::from(value.flatten().is_some()));
            put_u64(output, 8, value.flatten().unwrap_or_default());
            0
        }
        Err(error) => registry.fail(error),
    }
}

/// Insert or update a u64-token term.
#[no_mangle]
pub unsafe extern "C" fn vt_dictionary_put_u64(
    handle: u32,
    term_pointer: u32,
    term_length: u32,
    value_present: u32,
    value: u64,
) -> u32 {
    let term = match unsafe { tokens(term_pointer, term_length) } {
        Ok(term) => term,
        Err(error) => return locked_registry().fail(error),
    };
    let mut registry = locked_registry();
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => {
            dictionary.insert_u64(&term, (value_present != 0).then_some(value))
        }
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(value) => u32::from(value),
        Err(error) => registry.fail(error),
    }
}

/// Remove a u64-token term.
#[no_mangle]
pub unsafe extern "C" fn vt_dictionary_remove_u64(
    handle: u32,
    term_pointer: u32,
    term_length: u32,
) -> u32 {
    let term = match unsafe { tokens(term_pointer, term_length) } {
        Ok(term) => term,
        Err(error) => return locked_registry().fail(error),
    };
    let mut registry = locked_registry();
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => dictionary.remove_u64(&term),
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(value) => u32::from(value),
        Err(error) => registry.fail(error),
    }
}

/// Write a u64-token three-state lookup to a 16-byte record.
#[no_mangle]
pub unsafe extern "C" fn vt_dictionary_get_u64(
    handle: u32,
    term_pointer: u32,
    term_length: u32,
    output_pointer: u32,
) -> u32 {
    let term = match unsafe { tokens(term_pointer, term_length) } {
        Ok(term) => term,
        Err(error) => return locked_registry().fail(error),
    };
    let mut registry = locked_registry();
    if output_pointer == 0 {
        return registry.fail("output pointer is null");
    }
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(dictionary)) => dictionary.value_u64(&term),
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(value) => {
            let output = unsafe { slice::from_raw_parts_mut(output_pointer as *mut u8, 16) };
            put_u32(output, 0, u32::from(value.is_some()));
            put_u32(output, 4, u32::from(value.flatten().is_some()));
            put_u64(output, 8, value.flatten().unwrap_or_default());
            0
        }
        Err(error) => registry.fail(error),
    }
}

/// Capture a snapshot-owning entry cursor and return its handle.
#[no_mangle]
pub extern "C" fn vt_dictionary_entries_open(handle: u32) -> u32 {
    let result = {
        let registry = locked_registry();
        match registry.handles.get(&handle) {
            Some(Handle::Dictionary(dictionary)) => Ok(dictionary.resource().entries()),
            _ => Err("invalid dictionary handle"),
        }
    };
    let mut registry = locked_registry();
    match result {
        Ok(inner) => {
            let exact_len = inner.size_hint().1.unwrap_or_default();
            registry.insert(Handle::EntryCursor(EntryCursor {
                inner,
                exact_len,
                encoded: Vec::new(),
            }))
        }
        Err(error) => registry.fail(error),
    }
}

/// Exact number of records captured by an entry cursor.
#[no_mangle]
pub extern "C" fn vt_entry_cursor_len(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::EntryCursor(cursor)) => match u32::try_from(cursor.exact_len) {
            Ok(length) => length,
            Err(_) => registry.fail("dictionary snapshot length exceeds WASI u32 range"),
        },
        _ => registry.fail("invalid dictionary entry cursor handle"),
    }
}

/// Advance an entry cursor and encode copied records into linear memory.
#[no_mangle]
pub extern "C" fn vt_entry_cursor_next_batch(handle: u32, maximum: u32) -> u32 {
    if maximum == 0 {
        return locked_registry().fail("batch size must be positive");
    }
    let mut registry = locked_registry();
    let result = match registry.handles.get_mut(&handle) {
        Some(Handle::EntryCursor(cursor)) => (|| {
            cursor.encoded.clear();
            let mut count = 0u32;
            while count < maximum {
                let Some(entry) = cursor.inner.next() else {
                    break;
                };
                let entry = entry
                    .map_err(|status| format!("dictionary entry traversal failed: {status:?}"))?;
                let (domain, payload) = match entry.term {
                    BindingTerm::Bytes(bytes) => (0u32, bytes),
                    BindingTerm::Unicode(text) => (1u32, text.into_bytes()),
                    BindingTerm::U64(tokens) => {
                        let mut bytes = vec![0; tokens.len() * 8];
                        for (index, token) in tokens.into_iter().enumerate() {
                            put_u64(&mut bytes, index * 8, token);
                        }
                        (2u32, bytes)
                    }
                };
                let record = cursor.encoded.len();
                cursor
                    .encoded
                    .resize(record + ENTRY_RECORD_HEADER_SIZE + payload.len(), 0);
                put_u32(&mut cursor.encoded, record, payload.len() as u32);
                put_u32(&mut cursor.encoded, record + 4, domain);
                put_u32(
                    &mut cursor.encoded,
                    record + 8,
                    u32::from(entry.value.is_some()),
                );
                put_u64(
                    &mut cursor.encoded,
                    record + 16,
                    entry.value.unwrap_or_default(),
                );
                cursor.encoded[record + ENTRY_RECORD_HEADER_SIZE..][..payload.len()]
                    .copy_from_slice(&payload);
                count += 1;
            }
            Ok::<_, String>(count)
        })(),
        _ => Err("invalid dictionary entry cursor handle".into()),
    };
    match result {
        Ok(count) => count,
        Err(error) => registry.fail(error),
    }
}

/// Pointer to the current copied entry batch.
#[no_mangle]
pub extern "C" fn vt_entry_cursor_batch_pointer(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::EntryCursor(cursor)) => cursor.encoded.as_ptr() as u32,
        _ => registry.fail("invalid dictionary entry cursor handle"),
    }
}

/// Clear an in-memory DynamicDAWG.
#[no_mangle]
pub extern "C" fn vt_dictionary_clear(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::Dictionary(Dictionary::Dynamic(dictionary))) => {
            dictionary.clear();
            0
        }
        Some(Handle::Dictionary(Dictionary::Persistent(_))) => {
            registry.fail("persistent dictionaries cannot be cleared")
        }
        _ => registry.fail("invalid dictionary handle"),
    }
}

/// Compact an in-memory DynamicDAWG and return the reclaimed-node count.
#[no_mangle]
pub extern "C" fn vt_dictionary_compact(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::Dictionary(Dictionary::Dynamic(dictionary))) => dictionary.compact() as u32,
        Some(Handle::Dictionary(Dictionary::Persistent(_))) => {
            registry.fail("persistent dictionaries do not expose compact")
        }
        _ => registry.fail("invalid dictionary handle"),
    }
}

/// Durably checkpoint a persistent ARTrie.
#[no_mangle]
pub extern "C" fn vt_dictionary_checkpoint(handle: u32) -> u32 {
    let mut registry = locked_registry();
    let result = match registry.handles.get(&handle) {
        Some(Handle::Dictionary(Dictionary::Persistent(dictionary))) => {
            dictionary.checkpoint().map_err(|error| error.to_string())
        }
        Some(Handle::Dictionary(Dictionary::Dynamic(_))) => {
            Err("in-memory dictionaries do not checkpoint".into())
        }
        _ => Err("invalid dictionary handle".into()),
    };
    match result {
        Ok(()) => 0,
        Err(error) => registry.fail(error),
    }
}

/// Retain a dictionary as an liblevenshtein automaton configuration in O(1).
#[no_mangle]
pub extern "C" fn vt_transducer_new(dictionary_handle: u32, algorithm: u32) -> u32 {
    let result = (|| {
        let algorithm = selected_algorithm(algorithm)?;
        let resource = {
            let registry = locked_registry();
            let Some(Handle::Dictionary(dictionary)) = registry.handles.get(&dictionary_handle)
            else {
                return Err("invalid dictionary handle".into());
            };
            dictionary.resource()
        };
        unsafe { ResourceTransducer::from_resource(resource.as_raw(), algorithm) }
            .map_err(|error| error.to_string())
    })();
    let mut registry = locked_registry();
    match result {
        Ok(transducer) => registry.insert(Handle::Transducer(transducer)),
        Err(error) => registry.fail(error),
    }
}

/// Allocate an empty Unicode/tropical lling-llang VectorWfst builder.
#[no_mangle]
pub extern "C" fn vt_wfst_builder_new() -> u32 {
    locked_registry().insert(Handle::WfstBuilder(VectorWfst::new()))
}

/// Add one state and return its compact identifier.
#[no_mangle]
pub extern "C" fn vt_wfst_builder_add_state(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get_mut(&handle) {
        Some(Handle::WfstBuilder(graph)) => graph.add_state(),
        _ => registry.fail("invalid WFST builder handle"),
    }
}

/// Set the initial state.
#[no_mangle]
pub extern "C" fn vt_wfst_builder_set_start(handle: u32, state: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get_mut(&handle) {
        Some(Handle::WfstBuilder(graph)) => {
            if graph.try_set_start(state) {
                0
            } else {
                registry.fail("unknown WFST start state")
            }
        }
        _ => registry.fail("invalid WFST builder handle"),
    }
}

/// Set a final state and tropical cost.
#[no_mangle]
pub extern "C" fn vt_wfst_builder_set_final(handle: u32, state: u32, weight: f64) -> u32 {
    let mut registry = locked_registry();
    if weight.is_nan() {
        return registry.fail("WFST weight must not be NaN");
    }
    match registry.handles.get_mut(&handle) {
        Some(Handle::WfstBuilder(graph)) => match graph.state_mut(state) {
            Some(state) => {
                state.is_final = true;
                state.final_weight = TropicalWeight::new(weight);
                0
            }
            None => registry.fail("unknown WFST final state"),
        },
        _ => registry.fail("invalid WFST builder handle"),
    }
}

/// Add one Unicode arc; zero presence denotes epsilon.
#[no_mangle]
pub extern "C" fn vt_wfst_builder_add_arc(
    handle: u32,
    from: u32,
    input: u32,
    has_input: u32,
    output: u32,
    has_output: u32,
    to: u32,
    weight: f64,
) -> u32 {
    let mut registry = locked_registry();
    let decode = |label, present| match present {
        0 => Ok(None),
        1 => char::from_u32(label)
            .map(Some)
            .ok_or("arc label is not a Unicode scalar"),
        _ => Err("arc label presence must be zero or one"),
    };
    if weight.is_nan() {
        return registry.fail("WFST weight must not be NaN");
    }
    let input = match decode(input, has_input) {
        Ok(value) => value,
        Err(error) => return registry.fail(error),
    };
    let output = match decode(output, has_output) {
        Ok(value) => value,
        Err(error) => return registry.fail(error),
    };
    match registry.handles.get_mut(&handle) {
        Some(Handle::WfstBuilder(graph))
            if graph.is_valid_state(from) && graph.is_valid_state(to) =>
        {
            graph.add_arc(from, input, output, to, TropicalWeight::new(weight));
            0
        }
        Some(Handle::WfstBuilder(_)) => registry.fail("unknown WFST arc source or target"),
        _ => registry.fail("invalid WFST builder handle"),
    }
}

/// Consume a builder and freeze its graph into an immutable WFST handle.
#[no_mangle]
pub extern "C" fn vt_wfst_builder_build(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.remove(&handle) {
        Some(Handle::WfstBuilder(graph)) if graph.start() != u32::MAX => {
            registry.insert(Handle::Wfst(WasiWfst {
                resource: OwnedWfstResource::from_wfst(graph),
                encoded: Vec::new(),
            }))
        }
        Some(Handle::WfstBuilder(graph)) => {
            registry.handles.insert(handle, Handle::WfstBuilder(graph));
            registry.fail("WFST has no start state")
        }
        Some(other) => {
            registry.handles.insert(handle, other);
            registry.fail("invalid WFST builder handle")
        }
        None => registry.fail("invalid WFST builder handle"),
    }
}

/// Capture a dictionary revision and construct a lazy duallity WFST.
#[no_mangle]
pub unsafe extern "C" fn vt_duallity_wfst_new(
    dictionary_handle: u32,
    query_pointer: u32,
    query_length: u32,
    maximum_distance: u32,
    algorithm: u32,
    kind: u32,
) -> u32 {
    let result = (|| {
        let query = str::from_utf8(unsafe { bytes(query_pointer, query_length) })
            .map_err(|_| "query is not UTF-8")?;
        let algorithm = selected_algorithm(algorithm)?;
        let kind = selected_duallity_kind(kind)?;
        let dictionary = {
            let registry = locked_registry();
            let Some(Handle::Dictionary(dictionary)) = registry.handles.get(&dictionary_handle)
            else {
                return Err("invalid dictionary handle".into());
            };
            dictionary.resource()
        };
        unsafe {
            duallity::bindings::create_wfst(
                dictionary.as_raw(),
                query,
                maximum_distance as usize,
                algorithm,
                kind,
            )
        }
        .map_err(|error| error.to_string())
    })();
    let mut registry = locked_registry();
    match result {
        Ok(resource) => registry.insert(Handle::Wfst(WasiWfst {
            resource,
            encoded: Vec::new(),
        })),
        Err(error) => registry.fail(error),
    }
}

/// Lazily compose two tropical scalar WFST handles.
#[no_mangle]
pub extern "C" fn vt_wfst_compose(first: u32, second: u32) -> u32 {
    let result = (|| -> Result<OwnedWfstResource, String> {
        let registry = locked_registry();
        let Some(Handle::Wfst(first)) = registry.handles.get(&first) else {
            return Err("invalid first WFST handle".into());
        };
        let Some(Handle::Wfst(second)) = registry.handles.get(&second) else {
            return Err("invalid second WFST handle".into());
        };
        OwnedWfstResource::compose(first.resource.as_raw(), second.resource.as_raw())
            .map_err(|error| error.to_string())
    })();
    let mut registry = locked_registry();
    match result {
        Ok(resource) => registry.insert(Handle::Wfst(WasiWfst {
            resource,
            encoded: Vec::new(),
        })),
        Err(error) => registry.fail(error),
    }
}

/// Write the initial u64 state to an eight-byte caller buffer.
#[no_mangle]
pub extern "C" fn vt_wfst_start(handle: u32, output_pointer: u32) -> u32 {
    if output_pointer == 0 {
        return locked_registry().fail("output pointer is null");
    }
    let result = (|| -> Result<u64, String> {
        let registry = locked_registry();
        let Some(Handle::Wfst(wfst)) = registry.handles.get(&handle) else {
            return Err("invalid WFST handle".into());
        };
        unsafe {
            let table = &*wfst_table(wfst.resource.as_raw())?;
            let start = table.start.ok_or("WFST vtable has no start")?;
            let mut state = 0u64;
            require_ok(
                start(wfst.resource.as_raw().context, &mut state),
                "WFST start",
            )?;
            Ok(state)
        }
    })();
    match result {
        Ok(state) => {
            unsafe { slice::from_raw_parts_mut(output_pointer as *mut u8, 8) }
                .copy_from_slice(&state.to_le_bytes());
            0
        }
        Err(error) => locked_registry().fail(error),
    }
}

/// Return the numeric VtWeightDomain for a WFST handle.
#[no_mangle]
pub extern "C" fn vt_wfst_weight_domain(handle: u32) -> u32 {
    let mut registry = locked_registry();
    let Some(Handle::Wfst(wfst)) = registry.handles.get(&handle) else {
        return registry.fail("invalid WFST handle");
    };
    match unsafe { wfst_table(wfst.resource.as_raw()) } {
        Ok(table) => unsafe { (*table).weight_domain as u32 },
        Err(error) => registry.fail(error),
    }
}

/// Expand one state into a contiguous header plus 40-byte arc records.
#[no_mangle]
pub extern "C" fn vt_wfst_state(handle: u32, state: u64) -> u32 {
    let mut registry = locked_registry();
    let resource = match registry.handles.get(&handle) {
        Some(Handle::Wfst(wfst)) => wfst.resource.as_raw(),
        _ => return registry.fail("invalid WFST handle"),
    };
    let result = (|| unsafe {
        let table = wfst_table(resource)?;
        let table = &*table;
        let state_info = table.state_info.ok_or("WFST vtable has no state_info")?;
        let mut valid = 0;
        let mut final_state = 0;
        let mut final_weight = 0.0;
        require_ok(
            state_info(
                resource.context,
                state,
                &mut valid,
                &mut final_state,
                &mut final_weight,
            ),
            "WFST state_info",
        )?;
        let mut arcs = Vec::new();
        if valid == 1 {
            let state_arcs = table.state_arcs.ok_or("WFST vtable has no state_arcs")?;
            let mut offset = 0;
            loop {
                let mut page = vec![VtWfstArc::default(); 256];
                let mut written = 0;
                let mut total = 0;
                require_ok(
                    state_arcs(
                        resource.context,
                        state,
                        offset,
                        page.as_mut_ptr(),
                        page.len(),
                        &mut written,
                        &mut total,
                    ),
                    "WFST state_arcs",
                )?;
                if written > page.len()
                    || offset + written > total
                    || (written == 0 && offset < total)
                {
                    return Err("invalid WFST arc paging".into());
                }
                arcs.extend(page.into_iter().take(written));
                offset += written;
                if offset == total {
                    break;
                }
            }
        }
        Ok::<_, String>((valid, final_state, final_weight, arcs))
    })();
    match result {
        Ok((valid, final_state, final_weight, arcs)) => {
            let Some(Handle::Wfst(wfst)) = registry.handles.get_mut(&handle) else {
                return registry.fail("invalid WFST handle");
            };
            wfst.encoded.clear();
            wfst.encoded.resize(16 + arcs.len() * 40, 0);
            put_u32(&mut wfst.encoded, 0, u32::from(valid));
            put_u32(&mut wfst.encoded, 4, u32::from(final_state));
            put_f64(&mut wfst.encoded, 8, final_weight);
            for (index, arc) in arcs.iter().enumerate() {
                let record = 16 + index * 40;
                put_u64(&mut wfst.encoded, record, arc.input_label);
                put_u64(&mut wfst.encoded, record + 8, arc.output_label);
                put_u64(&mut wfst.encoded, record + 16, arc.target_state);
                put_f64(&mut wfst.encoded, record + 24, arc.weight);
                put_u32(&mut wfst.encoded, record + 32, u32::from(arc.has_input));
                put_u32(&mut wfst.encoded, record + 36, u32::from(arc.has_output));
            }
            arcs.len() as u32
        }
        Err(error) => registry.fail(error),
    }
}

/// Pointer to the current encoded WFST state.
#[no_mangle]
pub extern "C" fn vt_wfst_state_pointer(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::Wfst(wfst)) => wfst.encoded.as_ptr() as u32,
        _ => registry.fail("invalid WFST handle"),
    }
}

/// Start a lazy text query that captures its dictionary snapshot now.
#[no_mangle]
pub unsafe extern "C" fn vt_query_text(
    transducer_handle: u32,
    query_pointer: u32,
    query_length: u32,
    maximum_distance: u32,
    order: u32,
) -> u32 {
    let result = (|| {
        let query = str::from_utf8(unsafe { bytes(query_pointer, query_length) })
            .map_err(|_| "query is not UTF-8")?;
        let order = match order {
            0 => QueryOrder::Traversal,
            1 => QueryOrder::DistanceThenTerm,
            _ => return Err("unknown query order".into()),
        };
        let registry = locked_registry();
        let Some(Handle::Transducer(transducer)) = registry.handles.get(&transducer_handle) else {
            return Err("invalid transducer handle".into());
        };
        transducer
            .query_utf8(query, maximum_distance as usize, order)
            .map_err(|error| error.to_string())
    })();
    let mut registry = locked_registry();
    match result {
        Ok(cursor) => registry.insert(Handle::Cursor(Cursor {
            inner: cursor,
            batch: MatchBatch::default(),
            encoded: Vec::new(),
        })),
        Err(error) => registry.fail(error),
    }
}

/// Advance a lazy cursor by at most `maximum` and encode one contiguous batch.
#[no_mangle]
pub extern "C" fn vt_cursor_next_batch(handle: u32, maximum: u32) -> u32 {
    if maximum == 0 {
        return locked_registry().fail("batch size must be positive");
    }
    let mut registry = locked_registry();
    let result = match registry.handles.get_mut(&handle) {
        Some(Handle::Cursor(cursor)) => cursor
            .inner
            .next_batch(&mut cursor.batch, maximum as usize)
            .map_err(|error| error.to_string())
            .map(|count| {
                let data_length: usize = cursor
                    .batch
                    .as_slice()
                    .iter()
                    .map(|value| match &value.term {
                        MatchTerm::Utf8(term) => term.len(),
                        MatchTerm::Bytes(term) => term.len(),
                        MatchTerm::U64(term) => term.len() * 8,
                    })
                    .sum();
                cursor.encoded.clear();
                cursor
                    .encoded
                    .resize(count * MATCH_RECORD_SIZE + data_length, 0);
                let base = cursor.encoded.as_ptr() as u32;
                let mut data_offset = count * MATCH_RECORD_SIZE;
                for (index, value) in cursor.batch.as_slice().iter().enumerate() {
                    let record = index * MATCH_RECORD_SIZE;
                    let (domain, term_length) = match &value.term {
                        MatchTerm::Utf8(term) => {
                            cursor.encoded[data_offset..data_offset + term.len()]
                                .copy_from_slice(term.as_bytes());
                            (1, term.len())
                        }
                        MatchTerm::Bytes(term) => {
                            cursor.encoded[data_offset..data_offset + term.len()]
                                .copy_from_slice(term);
                            (0, term.len())
                        }
                        MatchTerm::U64(term) => {
                            for (offset, token) in term.iter().enumerate() {
                                put_u64(&mut cursor.encoded, data_offset + offset * 8, *token);
                            }
                            (2, term.len())
                        }
                    };
                    put_u32(&mut cursor.encoded, record, base + data_offset as u32);
                    put_u32(&mut cursor.encoded, record + 4, term_length as u32);
                    put_u32(&mut cursor.encoded, record + 8, domain);
                    put_u32(&mut cursor.encoded, record + 12, value.distance as u32);
                    put_u32(
                        &mut cursor.encoded,
                        record + 16,
                        u32::from(value.id.is_some()),
                    );
                    put_u64(
                        &mut cursor.encoded,
                        record + 24,
                        value.id.unwrap_or_default(),
                    );
                    data_offset += match &value.term {
                        MatchTerm::U64(term) => term.len() * 8,
                        MatchTerm::Utf8(term) => term.len(),
                        MatchTerm::Bytes(term) => term.len(),
                    };
                }
                count as u32
            }),
        _ => Err("invalid cursor handle".into()),
    };
    match result {
        Ok(count) => count,
        Err(error) => registry.fail(error),
    }
}

/// Pointer to the current contiguous cursor batch.
#[no_mangle]
pub extern "C" fn vt_cursor_batch_pointer(handle: u32) -> u32 {
    let mut registry = locked_registry();
    match registry.handles.get(&handle) {
        Some(Handle::Cursor(cursor)) => cursor.encoded.as_ptr() as u32,
        _ => registry.fail("invalid cursor handle"),
    }
}

/// Release any dictionary, transducer, or cursor handle.
#[no_mangle]
pub extern "C" fn vt_handle_close(handle: u32) -> u32 {
    u32::from(locked_registry().handles.remove(&handle).is_none())
}
