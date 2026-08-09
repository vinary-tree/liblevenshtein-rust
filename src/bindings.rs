//! Project-owned binding model for automata over foreign dictionary resources.
//!
//! liblevenshtein deliberately does not construct or mutate dictionaries here.
//! Concrete dictionary CRUD belongs to libdictenstein. This module consumes the
//! shared [`vinary_tree_interop`] dictionary-resource contract and exposes lazy,
//! snapshot-stable automaton cursors.

#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::nfa::NFAChar;
#[cfg(feature = "bindings-phonetic")]
use crate::transducer::language::{LanguageProduct, LanguageQueryIterator};
use crate::transducer::{
    Algorithm, RankedValueQueryIterator, Suggestion, ValueYieldingQueryIterator,
};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{CharUnit, DictionaryNode, MappedDictionaryNode};
use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use vinary_tree_interop::{
    dictionary_flags, VtDictionaryEdge, VtDictionaryVTable, VtOptionalU64, VtResource,
    VtResourceVTable, VtStatus, VtUnitDomain, VtValueDomain, VT_ABI_VERSION,
    VT_DICTIONARY_INTERFACE_ID, VT_DICTIONARY_INTERFACE_VERSION, VT_RECOMMENDED_EDGE_BATCH,
};

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

enum CallGate {
    Parallel,
    Serial(Mutex<()>),
}

/// One owned retain of a provider resource plus its discovered dictionary
/// interface. All callbacks for the default custom-provider mode are serialized
/// on the caller's thread. Providers opt into concurrent/reentrant calls by
/// publishing [`dictionary_flags::PARALLEL_REENTRANT`].
struct Provider {
    resource: VtResource,
    dictionary: *const VtDictionaryVTable,
    gate: CallGate,
    fault: Arc<Mutex<Option<BindingError>>>,
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
        match Self::from_owned(resource) {
            Ok(provider) => Ok(provider),
            Err(error) => {
                release(resource.context);
                Err(error)
            }
        }
    }

    unsafe fn from_owned(resource: VtResource) -> Result<Arc<Self>, BindingError> {
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
            CallGate::Serial(Mutex::new(()))
        };
        Ok(Arc::new(Self {
            resource,
            dictionary,
            gate,
            fault: Arc::new(Mutex::new(None)),
        }))
    }

    fn vtable(&self) -> &VtDictionaryVTable {
        // Validated for the lifetime of the retained resource at construction.
        unsafe { &*self.dictionary }
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
        let mut fault = self
            .fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if fault.is_none() {
            *fault = Some(error);
        }
    }

    fn take_fault(&self) -> Option<BindingError> {
        self.fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    fn snapshot(self: &Arc<Self>) -> Result<Arc<Self>, BindingError> {
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
        let snapshot = match unsafe { Provider::from_owned(snapshot) } {
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
    fn to_abi(self) -> u64;
}

impl InteropUnit for u8 {
    const DOMAIN: VtUnitDomain = VtUnitDomain::Byte;

    fn from_abi(label: u64) -> Option<Self> {
        u8::try_from(label).ok()
    }

    fn to_abi(self) -> u64 {
        u64::from(self)
    }
}

impl InteropUnit for char {
    const DOMAIN: VtUnitDomain = VtUnitDomain::UnicodeScalar;

    fn from_abi(label: u64) -> Option<Self> {
        u32::try_from(label).ok().and_then(char::from_u32)
    }

    fn to_abi(self) -> u64 {
        u64::from(u32::from(self))
    }
}

impl InteropUnit for u64 {
    const DOMAIN: VtUnitDomain = VtUnitDomain::U64;

    fn from_abi(label: u64) -> Option<Self> {
        Some(label)
    }

    fn to_abi(self) -> u64 {
        self
    }
}

#[derive(Clone)]
struct ForeignNode<U: InteropUnit> {
    provider: Arc<Provider>,
    id: u64,
    _unit: PhantomData<fn() -> U>,
}

impl<U: InteropUnit> fmt::Debug for ForeignNode<U> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ForeignNode")
            .field("id", &self.id)
            .field("domain", &U::DOMAIN)
            .finish()
    }
}

impl<U: InteropUnit> ForeignNode<U> {
    fn new(provider: Arc<Provider>, id: u64) -> Self {
        Self {
            provider,
            id,
            _unit: PhantomData,
        }
    }

    fn callback_failed<T>(&self, error: BindingError, fallback: T) -> T {
        self.provider.record_fault(error);
        fallback
    }

    fn expanded_edges(&self) -> Vec<(U, Self)> {
        let callback = match self.provider.vtable().node_edges {
            Some(callback) => callback,
            None => {
                return self.callback_failed(BindingError::IncompatibleDictionaryInterface, vec![])
            }
        };
        // Every page requests the same recommended capacity. The result Vec
        // grows only from edges actually delivered — never from the
        // provider-claimed `total`, whose inflation would otherwise drive a
        // preallocation abort (finding LLEV-B8). The per-page acceptance
        // check below is exactly the predicate proved in
        // docs/verification/abi/theories/ConsumerAcceptance.v `accepts_dec`.
        let mut result = Vec::with_capacity(VT_RECOMMENDED_EDGE_BATCH);
        let mut start = 0usize;
        loop {
            let capacity = VT_RECOMMENDED_EDGE_BATCH;
            let mut page = vec![VtDictionaryEdge::default(); capacity];
            let mut written = 0usize;
            let mut total = 0usize;
            let callback_status = self.provider.call(|| unsafe {
                callback(
                    self.provider.resource.context,
                    self.id,
                    start,
                    page.as_mut_ptr(),
                    capacity,
                    &mut written,
                    &mut total,
                )
            });
            if let Err(error) = status(callback_status) {
                return self.callback_failed(error, vec![]);
            }
            // ConsumerAcceptance.accepts_dec, in order: written<=capacity;
            // written<=total-start; and empty-page-only-when-exhausted
            // (progress). The claimed-total ceiling of the model is realized
            // structurally — no allocation is ever sized from `total` — so a
            // fabricated total cannot force an abort; it can only fail the
            // written<=total-start bound above.
            if written > capacity
                || written > total.saturating_sub(start)
                || (written == 0 && start < total)
            {
                return self.callback_failed(
                    BindingError::InvalidProviderOutput("invalid edge page lengths"),
                    vec![],
                );
            }
            for edge in page.into_iter().take(written) {
                let Some(label) = U::from_abi(edge.label) else {
                    return self.callback_failed(
                        BindingError::InvalidProviderOutput("edge label is outside its domain"),
                        vec![],
                    );
                };
                result.push((label, Self::new(Arc::clone(&self.provider), edge.node)));
            }
            start = start.saturating_add(written);
            if start >= total {
                break;
            }
        }
        result
    }
}

impl<U: InteropUnit> DictionaryNode for ForeignNode<U> {
    type Unit = U;

    fn is_final(&self) -> bool {
        let callback = match self.provider.vtable().node_is_final {
            Some(callback) => callback,
            None => {
                return self.callback_failed(BindingError::IncompatibleDictionaryInterface, false)
            }
        };
        let mut final_node = 0u8;
        let callback_status = self
            .provider
            .call(|| unsafe { callback(self.provider.resource.context, self.id, &mut final_node) });
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
                self.id,
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
            1 => Some(Self::new(Arc::clone(&self.provider), child)),
            _ => self.callback_failed(
                BindingError::InvalidProviderOutput("transition found was not zero or one"),
                None,
            ),
        }
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        Box::new(self.expanded_edges().into_iter())
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
            .call(|| unsafe { callback(self.provider.resource.context, self.id, &mut value) });
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
type CharLanguage = LanguageQueryIterator<ForeignNode<char>, NFAChar>;

fn term_only_score(_term: &str, _distance: usize, _value: &BindingValue) -> f64 {
    0.0
}

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
                id: item.node.value().and_then(|value| value.id),
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

/// A Levenshtein automaton configuration retaining a live dictionary resource.
///
/// Constructing this object is O(1): the resource is retained, not serialized.
/// Each query invokes the provider's O(1) `snapshot` callback before any node is
/// read, which is the binding-level query-start snapshot boundary.
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
        let root = ForeignNode::<char>::new(Arc::clone(&snapshot), snapshot.root()?);
        let inner = match order {
            QueryOrder::Traversal => CursorInner::CharTraversal(ValueYieldingQueryIterator::new(
                root,
                query.to_owned(),
                max_distance,
                self.algorithm,
            )),
            QueryOrder::DistanceThenTerm => {
                CursorInner::CharOrdered(RankedValueQueryIterator::new(
                    root,
                    query.to_owned(),
                    max_distance,
                    self.algorithm,
                    term_only_score as CharScorer,
                ))
            }
        };
        Ok(QueryCursor {
            inner,
            provider: snapshot,
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
        let root = ForeignNode::<u8>::new(Arc::clone(&snapshot), snapshot.root()?);
        Ok(QueryCursor {
            inner: CursorInner::ByteTraversal(ValueYieldingQueryIterator::with_unit_query(
                root,
                query.to_vec(),
                max_distance,
                self.algorithm,
            )),
            provider: snapshot,
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
        let root = ForeignNode::<u64>::new(Arc::clone(&snapshot), snapshot.root()?);
        Ok(QueryCursor {
            inner: CursorInner::U64Traversal(ValueYieldingQueryIterator::with_unit_query(
                root,
                query.to_vec(),
                max_distance,
                self.algorithm,
            )),
            provider: snapshot,
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
        let root = ForeignNode::<char>::new(Arc::clone(&snapshot), snapshot.root()?);
        Ok(QueryCursor {
            inner: CursorInner::CharLanguage(LanguageQueryIterator::new(
                root,
                LanguageProduct::new(pattern.nfa.clone(), max_distance),
            )),
            provider: snapshot,
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
