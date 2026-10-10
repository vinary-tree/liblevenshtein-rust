//! Bounded native Bincode persistence of value-bearing dictionaries.

#[cfg(feature = "serialization")]
use super::dictionary_binary::{invalid, read_u64, BoundedWriter};
use super::{index::boundary, LlevOwnedBytes, LlevStatus};
#[cfg(feature = "serialization")]
use libdictenstein::{
    double_array_trie::{DoubleArrayTrie, DoubleArrayTrieChar},
    serialization::{extract_terms_with_values, extract_terms_with_values_char, BincodeSerializer},
};

const VALUE_U64: u32 = 1;
const VALUE_BYTES: u32 = 2;
#[cfg(feature = "serialization")]
const UNIT_BYTE: u32 = 1;
#[cfg(feature = "serialization")]
const UNIT_UNICODE: u32 = 2;

/// One borrowed term and typed value for native Bincode persistence.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevValueEntryInput {
    /// Borrowed UTF-8 term data.
    pub term_data: *const u8,
    /// Term byte length.
    pub term_len: usize,
    /// Borrowed raw value bytes when value kind is two.
    pub value_data: *const u8,
    /// Value byte length when value kind is two.
    pub value_len: usize,
    /// Unsigned value when value kind is one.
    pub value_u64: u64,
}

/// Caller-selected resource ceilings for value-preserving persistence.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevValueLimits {
    /// Maximum number of input or output entries.
    pub max_entries: usize,
    /// Maximum UTF-8 bytes in one term.
    pub max_term_bytes: usize,
    /// Maximum aggregate term bytes.
    pub max_total_term_bytes: usize,
    /// Maximum bytes in one value; a u64 requires eight.
    pub max_value_bytes: usize,
    /// Maximum aggregate value bytes.
    pub max_total_value_bytes: usize,
    /// Maximum encoded Bincode payload bytes.
    pub max_payload_bytes: usize,
}

/// Borrowed entry view, valid until the decoded snapshot is freed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevValueEntryView {
    /// Borrowed UTF-8 term data.
    pub term_data: *const u8,
    /// Term byte length.
    pub term_len: usize,
    /// Borrowed value bytes for kind two.
    pub value_data: *const u8,
    /// Value byte length for kind two.
    pub value_len: usize,
    /// Unsigned value for kind one.
    pub value_u64: u64,
    /// One for u64, two for raw bytes.
    pub value_kind: u32,
    /// Must be ignored; fixed to zero.
    pub reserved: u32,
}

#[cfg_attr(not(feature = "serialization"), allow(dead_code))]
enum ValueEntries {
    U64(Vec<(String, u64)>),
    Bytes(Vec<(String, Vec<u8>)>),
}

/// Owned snapshot of decoded value-bearing dictionary entries.
pub struct LlevDecodedValueEntries {
    entries: ValueEntries,
}

#[cfg(feature = "serialization")]
fn ceilings(raw: *const LlevValueLimits) -> Result<LlevValueLimits, (LlevStatus, String)> {
    let limits =
        unsafe { raw.as_ref() }.ok_or((LlevStatus::NullPointer, "value limits are null".into()))?;
    if limits.max_payload_bytes < 8
        || limits.max_term_bytes > limits.max_total_term_bytes
        || limits.max_value_bytes > limits.max_total_value_bytes
    {
        return Err(invalid("invalid valued-dictionary ceilings"));
    }
    Ok(*limits)
}

#[cfg(feature = "serialization")]
fn format(unit_domain: u32, value_kind: u32) -> Result<(), (LlevStatus, String)> {
    if !matches!(unit_domain, UNIT_BYTE | UNIT_UNICODE) {
        return Err(invalid("unknown valued-dictionary unit domain"));
    }
    if !matches!(value_kind, VALUE_U64 | VALUE_BYTES) {
        return Err(invalid("unknown valued-dictionary value kind"));
    }
    Ok(())
}

#[cfg(feature = "serialization")]
fn validate_entries(
    entries: &ValueEntries,
    limits: LlevValueLimits,
) -> Result<(), (LlevStatus, String)> {
    let mut total_terms = 0usize;
    let mut total_values = 0usize;
    match entries {
        ValueEntries::U64(values) => {
            if values.len() > limits.max_entries || limits.max_value_bytes < 8 {
                return Err((LlevStatus::LimitExceeded, "u64 value ceiling".into()));
            }
            total_values = values.len().checked_mul(8).ok_or((
                LlevStatus::LimitExceeded,
                "aggregate u64 value bytes overflow".into(),
            ))?;
            for (term, _) in values {
                total_terms = total_terms.checked_add(term.len()).ok_or((
                    LlevStatus::LimitExceeded,
                    "aggregate term bytes overflow".into(),
                ))?;
                if term.len() > limits.max_term_bytes {
                    return Err((LlevStatus::LimitExceeded, "term byte ceiling".into()));
                }
            }
        }
        ValueEntries::Bytes(values) => {
            if values.len() > limits.max_entries {
                return Err((LlevStatus::LimitExceeded, "entry count ceiling".into()));
            }
            for (term, value) in values {
                total_terms = total_terms.checked_add(term.len()).ok_or((
                    LlevStatus::LimitExceeded,
                    "aggregate term bytes overflow".into(),
                ))?;
                total_values = total_values.checked_add(value.len()).ok_or((
                    LlevStatus::LimitExceeded,
                    "aggregate value bytes overflow".into(),
                ))?;
                if term.len() > limits.max_term_bytes || value.len() > limits.max_value_bytes {
                    return Err((LlevStatus::LimitExceeded, "entry byte ceiling".into()));
                }
            }
        }
    }
    if total_terms > limits.max_total_term_bytes || total_values > limits.max_total_value_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "aggregate entry byte ceiling".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "serialization")]
fn expected_len(entries: &ValueEntries) -> Result<usize, (LlevStatus, String)> {
    let mut size = 8usize;
    match entries {
        ValueEntries::U64(values) => {
            for (term, _) in values {
                size = size
                    .checked_add(8)
                    .and_then(|n| n.checked_add(term.len()))
                    .and_then(|n| n.checked_add(8))
                    .ok_or((
                        LlevStatus::LimitExceeded,
                        "valued Bincode size overflow".into(),
                    ))?;
            }
        }
        ValueEntries::Bytes(values) => {
            for (term, value) in values {
                size = size
                    .checked_add(8)
                    .and_then(|n| n.checked_add(term.len()))
                    .and_then(|n| n.checked_add(8))
                    .and_then(|n| n.checked_add(value.len()))
                    .ok_or((
                        LlevStatus::LimitExceeded,
                        "valued Bincode size overflow".into(),
                    ))?;
            }
        }
    }
    Ok(size)
}

#[cfg(feature = "serialization")]
fn preflight(
    input: &[u8],
    value_kind: u32,
    limits: LlevValueLimits,
) -> Result<(), (LlevStatus, String)> {
    if input.len() > limits.max_payload_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "valued payload byte ceiling".into(),
        ));
    }
    let mut offset = 0usize;
    let count = usize::try_from(read_u64(input, &mut offset)?)
        .map_err(|_| (LlevStatus::LimitExceeded, "entry count overflow".into()))?;
    if count > limits.max_entries {
        return Err((LlevStatus::LimitExceeded, "entry count ceiling".into()));
    }
    let mut total_terms = 0usize;
    let mut total_values = 0usize;
    for _ in 0..count {
        let term_len = usize::try_from(read_u64(input, &mut offset)?)
            .map_err(|_| (LlevStatus::LimitExceeded, "term length overflow".into()))?;
        total_terms = total_terms
            .checked_add(term_len)
            .ok_or((LlevStatus::LimitExceeded, "term bytes overflow".into()))?;
        if term_len > limits.max_term_bytes || total_terms > limits.max_total_term_bytes {
            return Err((LlevStatus::LimitExceeded, "term byte ceiling".into()));
        }
        let end = offset
            .checked_add(term_len)
            .ok_or_else(|| invalid("valued term offset overflow"))?;
        let term = input
            .get(offset..end)
            .ok_or_else(|| invalid("truncated valued term"))?;
        std::str::from_utf8(term).map_err(|_| invalid("valued term is not UTF-8"))?;
        offset = end;

        let value_len = if value_kind == VALUE_U64 {
            8
        } else {
            usize::try_from(read_u64(input, &mut offset)?)
                .map_err(|_| (LlevStatus::LimitExceeded, "value length overflow".into()))?
        };
        total_values = total_values
            .checked_add(value_len)
            .ok_or((LlevStatus::LimitExceeded, "value bytes overflow".into()))?;
        if value_len > limits.max_value_bytes || total_values > limits.max_total_value_bytes {
            return Err((LlevStatus::LimitExceeded, "value byte ceiling".into()));
        }
        offset = offset
            .checked_add(value_len)
            .ok_or_else(|| invalid("valued data offset overflow"))?;
        if offset > input.len() {
            return Err(invalid("truncated valued data"));
        }
    }
    if offset != input.len() {
        return Err(invalid("valued Bincode payload has trailing bytes"));
    }
    Ok(())
}

#[cfg(feature = "serialization")]
unsafe fn input_entries(
    raw: *const LlevValueEntryInput,
    count: usize,
    value_kind: u32,
    limits: LlevValueLimits,
) -> Result<ValueEntries, (LlevStatus, String)> {
    if count > limits.max_entries {
        return Err((LlevStatus::LimitExceeded, "entry count ceiling".into()));
    }
    let source = if count == 0 {
        &[][..]
    } else {
        if raw.is_null() {
            return Err((LlevStatus::NullPointer, "entry array is null".into()));
        }
        std::slice::from_raw_parts(raw, count)
    };
    let mut terms = Vec::new();
    terms
        .try_reserve(count)
        .map_err(|_| (LlevStatus::LimitExceeded, "entry allocation failed".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve(count)
        .map_err(|_| (LlevStatus::LimitExceeded, "value allocation failed".into()))?;
    let mut total_terms = 0usize;
    let mut total_values = 0usize;
    for entry in source {
        total_terms = total_terms
            .checked_add(entry.term_len)
            .ok_or((LlevStatus::LimitExceeded, "term bytes overflow".into()))?;
        let value_len = if value_kind == VALUE_U64 {
            8
        } else {
            entry.value_len
        };
        total_values = total_values
            .checked_add(value_len)
            .ok_or((LlevStatus::LimitExceeded, "value bytes overflow".into()))?;
        if entry.term_len > limits.max_term_bytes
            || total_terms > limits.max_total_term_bytes
            || value_len > limits.max_value_bytes
            || total_values > limits.max_total_value_bytes
        {
            return Err((LlevStatus::LimitExceeded, "entry byte ceiling".into()));
        }
        let term = if entry.term_len == 0 {
            &[][..]
        } else {
            if entry.term_data.is_null() {
                return Err((LlevStatus::NullPointer, "term data is null".into()));
            }
            std::slice::from_raw_parts(entry.term_data, entry.term_len)
        };
        terms.push(
            std::str::from_utf8(term)
                .map_err(|_| invalid("valued term is not UTF-8"))?
                .to_owned(),
        );
        if value_kind == VALUE_BYTES {
            let value = if entry.value_len == 0 {
                &[][..]
            } else {
                if entry.value_data.is_null() {
                    return Err((LlevStatus::NullPointer, "value data is null".into()));
                }
                std::slice::from_raw_parts(entry.value_data, entry.value_len)
            };
            bytes.push(value.to_vec());
        }
    }
    if value_kind == VALUE_U64 {
        Ok(ValueEntries::U64(
            terms
                .into_iter()
                .zip(source.iter().map(|entry| entry.value_u64))
                .collect(),
        ))
    } else {
        Ok(ValueEntries::Bytes(terms.into_iter().zip(bytes).collect()))
    }
}

#[cfg(feature = "serialization")]
fn native_serialize(
    entries: ValueEntries,
    unit_domain: u32,
    limits: LlevValueLimits,
) -> Result<Vec<u8>, (LlevStatus, String)> {
    let value_kind = if matches!(&entries, ValueEntries::U64(_)) {
        VALUE_U64
    } else {
        VALUE_BYTES
    };
    let mut writer = BoundedWriter {
        bytes: Vec::new(),
        max_bytes: limits.max_payload_bytes,
    };
    let expected = match entries {
        ValueEntries::U64(values) if unit_domain == UNIT_BYTE => {
            let dictionary = DoubleArrayTrie::<u64>::from_terms_with_values(values);
            let canonical = ValueEntries::U64(extract_terms_with_values(&dictionary));
            validate_entries(&canonical, limits)?;
            let expected = expected_len(&canonical)?;
            if expected > limits.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "valued payload byte ceiling".into(),
                ));
            }
            BincodeSerializer::serialize_with_values(&dictionary, &mut writer)
                .map_err(|error| invalid(format!("native valued encode failed: {error}")))?;
            expected
        }
        ValueEntries::U64(values) => {
            let dictionary = DoubleArrayTrieChar::<u64>::from_terms_with_values(values);
            let canonical = ValueEntries::U64(extract_terms_with_values_char(&dictionary));
            validate_entries(&canonical, limits)?;
            let expected = expected_len(&canonical)?;
            if expected > limits.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "valued payload byte ceiling".into(),
                ));
            }
            BincodeSerializer::serialize_with_values_char(&dictionary, &mut writer)
                .map_err(|error| invalid(format!("native valued encode failed: {error}")))?;
            expected
        }
        ValueEntries::Bytes(values) if unit_domain == UNIT_BYTE => {
            let dictionary = DoubleArrayTrie::<Vec<u8>>::from_terms_with_values(values);
            let canonical = ValueEntries::Bytes(extract_terms_with_values(&dictionary));
            validate_entries(&canonical, limits)?;
            let expected = expected_len(&canonical)?;
            if expected > limits.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "valued payload byte ceiling".into(),
                ));
            }
            BincodeSerializer::serialize_with_values(&dictionary, &mut writer)
                .map_err(|error| invalid(format!("native valued encode failed: {error}")))?;
            expected
        }
        ValueEntries::Bytes(values) => {
            let dictionary = DoubleArrayTrieChar::<Vec<u8>>::from_terms_with_values(values);
            let canonical = ValueEntries::Bytes(extract_terms_with_values_char(&dictionary));
            validate_entries(&canonical, limits)?;
            let expected = expected_len(&canonical)?;
            if expected > limits.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "valued payload byte ceiling".into(),
                ));
            }
            BincodeSerializer::serialize_with_values_char(&dictionary, &mut writer)
                .map_err(|error| invalid(format!("native valued encode failed: {error}")))?;
            expected
        }
    };
    if expected > limits.max_payload_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "valued payload byte ceiling".into(),
        ));
    }
    if writer.bytes.len() != expected {
        return Err(invalid("native valued encoder returned incomplete payload"));
    }
    preflight(&writer.bytes, value_kind, limits)?;
    Ok(writer.bytes)
}

#[cfg(feature = "serialization")]
fn native_deserialize(
    input: &[u8],
    unit_domain: u32,
    value_kind: u32,
) -> Result<ValueEntries, (LlevStatus, String)> {
    match (unit_domain, value_kind) {
        (UNIT_BYTE, VALUE_U64) => {
            let dictionary: DoubleArrayTrie<u64> =
                BincodeSerializer::deserialize_with_values(input)
                    .map_err(|error| invalid(format!("native valued decode failed: {error}")))?;
            Ok(ValueEntries::U64(extract_terms_with_values(&dictionary)))
        }
        (UNIT_UNICODE, VALUE_U64) => {
            let dictionary: DoubleArrayTrieChar<u64> =
                BincodeSerializer::deserialize_with_values(input)
                    .map_err(|error| invalid(format!("native valued decode failed: {error}")))?;
            Ok(ValueEntries::U64(extract_terms_with_values_char(
                &dictionary,
            )))
        }
        (UNIT_BYTE, VALUE_BYTES) => {
            let dictionary: DoubleArrayTrie<Vec<u8>> =
                BincodeSerializer::deserialize_with_values(input)
                    .map_err(|error| invalid(format!("native valued decode failed: {error}")))?;
            Ok(ValueEntries::Bytes(extract_terms_with_values(&dictionary)))
        }
        (UNIT_UNICODE, VALUE_BYTES) => {
            let dictionary: DoubleArrayTrieChar<Vec<u8>> =
                BincodeSerializer::deserialize_with_values(input)
                    .map_err(|error| invalid(format!("native valued decode failed: {error}")))?;
            Ok(ValueEntries::Bytes(extract_terms_with_values_char(
                &dictionary,
            )))
        }
        _ => Err(invalid("unsupported valued dictionary format")),
    }
}

/// Serialize UTF-8 term/value pairs using native value-preserving Bincode.
///
/// # Safety
/// Nonempty arrays and nested slices, limits, and output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_valued_dictionary_serialize(
    unit_domain: u32,
    value_kind: u32,
    entries: *const LlevValueEntryInput,
    entry_count: usize,
    raw_limits: *const LlevValueLimits,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    boundary(|| {
        let output = out_bytes
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "valued output is null".into()))?;
        *output = LlevOwnedBytes::default();
        #[cfg(feature = "serialization")]
        {
            format(unit_domain, value_kind)?;
            let limits = ceilings(raw_limits)?;
            let source = input_entries(entries, entry_count, value_kind, limits)?;
            let bytes = native_serialize(source, unit_domain, limits)?;
            let boxed = bytes.into_boxed_slice();
            output.len = boxed.len();
            output.data = Box::into_raw(boxed) as *mut u8;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (
                unit_domain,
                value_kind,
                entries,
                entry_count,
                raw_limits,
                output,
            );
            Err((
                LlevStatus::Unsupported,
                "valued Bincode requires serialization".into(),
            ))
        }
    })
}

/// Decode complete value-preserving native Bincode into an owned snapshot.
///
/// # Safety
/// Nonempty input, limits, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_valued_dictionary_deserialize(
    unit_domain: u32,
    value_kind: u32,
    data: *const u8,
    len: usize,
    raw_limits: *const LlevValueLimits,
    out_entries: *mut *mut LlevDecodedValueEntries,
) -> LlevStatus {
    boundary(|| {
        let output = out_entries.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded entries output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        #[cfg(feature = "serialization")]
        {
            format(unit_domain, value_kind)?;
            let limits = ceilings(raw_limits)?;
            if len > limits.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "valued payload byte ceiling".into(),
                ));
            }
            let input = if len == 0 {
                &[][..]
            } else {
                if data.is_null() {
                    return Err((LlevStatus::NullPointer, "valued input is null".into()));
                }
                std::slice::from_raw_parts(data, len)
            };
            preflight(input, value_kind, limits)?;
            let entries = native_deserialize(input, unit_domain, value_kind)?;
            validate_entries(&entries, limits)?;
            *output = Box::into_raw(Box::new(LlevDecodedValueEntries { entries }));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (unit_domain, value_kind, data, len, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "valued Bincode requires serialization".into(),
            ))
        }
    })
}

/// Return the number of entries in a decoded value snapshot.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_value_entries_len(
    entries: *const LlevDecodedValueEntries,
    out_len: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let source = entries
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "decoded entries are null".into()))?;
        let output = out_len
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "entry count output is null".into()))?;
        *output = match &source.entries {
            ValueEntries::U64(values) => values.len(),
            ValueEntries::Bytes(values) => values.len(),
        };
        Ok(LlevStatus::Ok)
    })
}

/// Borrow one entry until the decoded snapshot is freed.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_value_entry_at(
    entries: *const LlevDecodedValueEntries,
    index: usize,
    out_view: *mut LlevValueEntryView,
) -> LlevStatus {
    boundary(|| {
        let source = entries
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "decoded entries are null".into()))?;
        let output = out_view
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "entry view output is null".into()))?;
        *output = match &source.entries {
            ValueEntries::U64(values) => {
                let (term, value) = values.get(index).ok_or((
                    LlevStatus::InvalidArgument,
                    "entry index is out of bounds".into(),
                ))?;
                LlevValueEntryView {
                    term_data: term.as_ptr(),
                    term_len: term.len(),
                    value_u64: *value,
                    value_kind: VALUE_U64,
                    ..LlevValueEntryView::default()
                }
            }
            ValueEntries::Bytes(values) => {
                let (term, value) = values.get(index).ok_or((
                    LlevStatus::InvalidArgument,
                    "entry index is out of bounds".into(),
                ))?;
                LlevValueEntryView {
                    term_data: term.as_ptr(),
                    term_len: term.len(),
                    value_data: value.as_ptr(),
                    value_len: value.len(),
                    value_kind: VALUE_BYTES,
                    ..LlevValueEntryView::default()
                }
            }
        };
        Ok(LlevStatus::Ok)
    })
}

/// Free a decoded value-bearing dictionary snapshot.
///
/// # Safety
/// Handle must be null or returned by `llev_valued_dictionary_deserialize` and freed once.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_value_entries_free(entries: *mut LlevDecodedValueEntries) {
    if !entries.is_null() {
        drop(Box::from_raw(entries));
    }
}
