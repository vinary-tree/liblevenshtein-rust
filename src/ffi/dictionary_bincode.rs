//! Bounded C bridge to the native dictionary bincode wire format.

use super::{index::boundary, LlevOwnedBytes, LlevStatus};
#[cfg(feature = "serialization")]
use libdictenstein::{
    double_array_trie::DoubleArrayTrie,
    serialization::{extract_terms, BincodeSerializer, DictionarySerializer},
};
#[cfg(feature = "serialization")]
use std::io::{self, Write};

/// Borrowed UTF-8 term valid for the duration of a call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevUtf8Slice {
    /// Borrowed term bytes; may be null when length is zero.
    pub data: *const u8,
    /// Number of bytes in the UTF-8 term.
    pub len: usize,
}

/// Caller-selected input and output ceilings for dictionary bincode.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevDictionaryBincodeLimits {
    /// Maximum number of terms accepted or returned.
    pub max_terms: usize,
    /// Maximum UTF-8 bytes in one term.
    pub max_term_bytes: usize,
    /// Maximum UTF-8 bytes across all terms.
    pub max_total_term_bytes: usize,
    /// Maximum binary payload bytes read or written.
    pub max_payload_bytes: usize,
}

/// Immutable decoded term snapshot; borrowed term views last until free.
pub struct LlevDecodedBincodeTerms {
    terms: Vec<String>,
}

#[cfg(feature = "serialization")]
const FORMAT_VERSION: u32 = 1;

#[cfg(feature = "serialization")]
fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

#[cfg(feature = "serialization")]
fn limits(
    raw: *const LlevDictionaryBincodeLimits,
) -> Result<LlevDictionaryBincodeLimits, (LlevStatus, String)> {
    let value = unsafe { raw.as_ref() }
        .ok_or((LlevStatus::NullPointer, "bincode limits are null".into()))?;
    if value.max_payload_bytes < 8 || value.max_term_bytes > value.max_total_term_bytes {
        return Err(invalid("invalid bincode resource ceilings"));
    }
    Ok(*value)
}

#[cfg(feature = "serialization")]
struct BoundedWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
}

#[cfg(feature = "serialization")]
impl Write for BoundedWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let requested = self
            .bytes
            .len()
            .checked_add(input.len())
            .ok_or_else(|| io::Error::other("bincode output length overflow"))?;
        if requested > self.max_bytes {
            return Err(io::Error::other("bincode output byte ceiling"));
        }
        self.bytes
            .try_reserve(input.len())
            .map_err(|_| io::Error::other("bincode output allocation failed"))?;
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "serialization")]
fn read_u64(input: &[u8], offset: &mut usize) -> Result<u64, (LlevStatus, String)> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| invalid("bincode offset overflow"))?;
    let bytes: [u8; 8] = input
        .get(*offset..end)
        .ok_or_else(|| invalid("truncated bincode length"))?
        .try_into()
        .map_err(|_| invalid("truncated bincode length"))?;
    *offset = end;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(feature = "serialization")]
fn preflight(
    input: &[u8],
    ceilings: LlevDictionaryBincodeLimits,
) -> Result<(), (LlevStatus, String)> {
    let mut offset = 0;
    let count = usize::try_from(read_u64(input, &mut offset)?).map_err(|_| {
        (
            LlevStatus::LimitExceeded,
            "bincode term count overflow".into(),
        )
    })?;
    if count > ceilings.max_terms {
        return Err((
            LlevStatus::LimitExceeded,
            "bincode term count ceiling".into(),
        ));
    }
    let mut total = 0usize;
    for _ in 0..count {
        let len = usize::try_from(read_u64(input, &mut offset)?).map_err(|_| {
            (
                LlevStatus::LimitExceeded,
                "bincode term length overflow".into(),
            )
        })?;
        total = total.checked_add(len).ok_or((
            LlevStatus::LimitExceeded,
            "bincode term bytes overflow".into(),
        ))?;
        if len > ceilings.max_term_bytes || total > ceilings.max_total_term_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "bincode term byte ceiling".into(),
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| invalid("bincode offset overflow"))?;
        let bytes = input
            .get(offset..end)
            .ok_or_else(|| invalid("truncated bincode term"))?;
        std::str::from_utf8(bytes).map_err(|_| invalid("bincode term is not UTF-8"))?;
        offset = end;
    }
    if offset != input.len() {
        return Err(invalid("bincode payload has trailing bytes"));
    }
    Ok(())
}

/// Serialize accepted UTF-8 terms with the native fixed-int little-endian
/// bincode format. Version 1 is the current and only supported wire revision.
///
/// # Safety
/// Nonempty input slices and the output pointer must be valid. The output is
/// released with `llev_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn llev_dictionary_bincode_serialize(
    format_version: u32,
    terms: *const LlevUtf8Slice,
    term_count: usize,
    raw_limits: *const LlevDictionaryBincodeLimits,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    boundary(|| {
        let output = out_bytes
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "bincode output is null".into()))?;
        *output = LlevOwnedBytes::default();
        #[cfg(feature = "serialization")]
        {
            if format_version != FORMAT_VERSION {
                return Err(invalid("unsupported bincode format version"));
            }
            let ceilings = limits(raw_limits)?;
            if term_count > ceilings.max_terms {
                return Err((
                    LlevStatus::LimitExceeded,
                    "bincode term count ceiling".into(),
                ));
            }
            let source = if term_count == 0 {
                &[][..]
            } else {
                if terms.is_null() {
                    return Err((LlevStatus::NullPointer, "bincode terms are null".into()));
                }
                std::slice::from_raw_parts(terms, term_count)
            };
            let mut owned = Vec::new();
            owned.try_reserve(term_count).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "bincode term allocation failed".into(),
                )
            })?;
            let mut total = 0usize;
            for term in source {
                total = total.checked_add(term.len).ok_or((
                    LlevStatus::LimitExceeded,
                    "bincode term bytes overflow".into(),
                ))?;
                if term.len > ceilings.max_term_bytes || total > ceilings.max_total_term_bytes {
                    return Err((
                        LlevStatus::LimitExceeded,
                        "bincode term byte ceiling".into(),
                    ));
                }
                let bytes = if term.len == 0 {
                    &[][..]
                } else {
                    if term.data.is_null() {
                        return Err((LlevStatus::NullPointer, "bincode term is null".into()));
                    }
                    std::slice::from_raw_parts(term.data, term.len)
                };
                let value =
                    std::str::from_utf8(bytes).map_err(|_| invalid("bincode term is not UTF-8"))?;
                owned.push(value.to_owned());
            }
            let dictionary = DoubleArrayTrie::from_terms(owned);
            // This format is a legacy fixed-width Vec<String>: count and each
            // term length are u64 LE. Preflight the exact native term set before
            // calling the writer; bincode's writer adapter may report success
            // after a rejected write, so its return value alone is insufficient.
            let native_terms = extract_terms(&dictionary);
            let mut expected = 8usize;
            for term in &native_terms {
                expected = expected
                    .checked_add(8)
                    .and_then(|size| size.checked_add(term.len()))
                    .ok_or((
                        LlevStatus::LimitExceeded,
                        "bincode payload length overflow".into(),
                    ))?;
            }
            if expected > ceilings.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "bincode payload byte ceiling".into(),
                ));
            }
            let mut writer = BoundedWriter {
                bytes: Vec::new(),
                max_bytes: ceilings.max_payload_bytes,
            };
            BincodeSerializer::serialize(&dictionary, &mut writer).map_err(|error| {
                (
                    LlevStatus::LimitExceeded,
                    format!("bounded bincode serialization failed: {error}"),
                )
            })?;
            if writer.bytes.len() != expected {
                return Err(invalid(
                    "native bincode encoder returned an incomplete payload",
                ));
            }
            preflight(&writer.bytes, ceilings)?;
            let boxed = writer.bytes.into_boxed_slice();
            output.len = boxed.len();
            output.data = Box::into_raw(boxed) as *mut u8;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (format_version, terms, term_count, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "dictionary bincode requires serialization feature".into(),
            ))
        }
    })
}

/// Decode native bincode bytes into an independently owned term snapshot.
///
/// # Safety
/// Nonempty input and output pointer must be valid; free the handle once.
#[no_mangle]
pub unsafe extern "C" fn llev_dictionary_bincode_deserialize(
    format_version: u32,
    data: *const u8,
    len: usize,
    raw_limits: *const LlevDictionaryBincodeLimits,
    out_terms: *mut *mut LlevDecodedBincodeTerms,
) -> LlevStatus {
    boundary(|| {
        let output = out_terms.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded terms output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        #[cfg(feature = "serialization")]
        {
            if format_version != FORMAT_VERSION {
                return Err(invalid("unsupported bincode format version"));
            }
            let ceilings = limits(raw_limits)?;
            if len > ceilings.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "bincode payload byte ceiling".into(),
                ));
            }
            let input = if len == 0 {
                &[][..]
            } else {
                if data.is_null() {
                    return Err((LlevStatus::NullPointer, "bincode data is null".into()));
                }
                std::slice::from_raw_parts(data, len)
            };
            preflight(input, ceilings)?;
            let dictionary: DoubleArrayTrie = BincodeSerializer::deserialize(input)
                .map_err(|error| invalid(format!("native bincode decode failed: {error}")))?;
            let terms = extract_terms(&dictionary);
            *output = Box::into_raw(Box::new(LlevDecodedBincodeTerms { terms }));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (format_version, data, len, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "dictionary bincode requires serialization feature".into(),
            ))
        }
    })
}

/// Return the number of terms in the decoded immutable snapshot.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_bincode_terms_len(
    terms: *const LlevDecodedBincodeTerms,
    out_len: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let source = terms
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "decoded terms are null".into()))?;
        let output = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded length output is null".into(),
        ))?;
        *output = source.terms.len();
        Ok(LlevStatus::Ok)
    })
}

/// Return a borrowed UTF-8 term view valid until the snapshot is freed.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_bincode_term_at(
    terms: *const LlevDecodedBincodeTerms,
    index: usize,
    out_term: *mut LlevUtf8Slice,
) -> LlevStatus {
    boundary(|| {
        let source = terms
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "decoded terms are null".into()))?;
        let output = out_term.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded term output is null".into(),
        ))?;
        let value = source.terms.get(index).ok_or((
            LlevStatus::InvalidArgument,
            "decoded term index is out of range".into(),
        ))?;
        *output = LlevUtf8Slice {
            data: value.as_ptr(),
            len: value.len(),
        };
        Ok(LlevStatus::Ok)
    })
}

/// Free a decoded immutable term snapshot; null is a no-op.
///
/// # Safety
/// Pointer must be a live handle from this library and freed only once.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_bincode_terms_free(terms: *mut LlevDecodedBincodeTerms) {
    if !terms.is_null() {
        drop(Box::from_raw(terms));
    }
}
