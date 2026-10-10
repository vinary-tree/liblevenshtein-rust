//! Bounded native persistence of suffix-automaton source texts.

#[cfg(feature = "protobuf")]
use super::dictionary_binary::protobuf_wire;
#[cfg(feature = "serialization")]
use super::dictionary_binary::{input_strings, invalid, limits, preflight, BoundedWriter};
use super::dictionary_binary::{LlevDecodedDictionaryTerms, LlevDictionaryLimits, LlevUtf8Slice};
use super::{index::boundary, LlevOwnedBytes, LlevStatus};
#[cfg(feature = "protobuf")]
use libdictenstein::serialization::SuffixAutomatonProtobufSerializer;
#[cfg(feature = "serialization")]
use libdictenstein::{serialization::BincodeSerializer, suffix_automaton::SuffixAutomaton};
#[cfg(feature = "protobuf")]
use prost::Message;

#[cfg(feature = "serialization")]
fn require_format(format_id: u32) -> Result<(), (LlevStatus, String)> {
    match format_id {
        1 => Ok(()),
        2 if !cfg!(feature = "protobuf") => Err((
            LlevStatus::Unsupported,
            "suffix protobuf requires protobuf feature".into(),
        )),
        2 => Ok(()),
        _ => Err(invalid("unsupported suffix source format ID")),
    }
}

#[cfg(feature = "serialization")]
fn preflight_suffix(
    input: &[u8],
    format_id: u32,
    ceilings: LlevDictionaryLimits,
) -> Result<(), (LlevStatus, String)> {
    require_format(format_id)?;
    if input.len() > ceilings.max_payload_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "suffix payload byte ceiling".into(),
        ));
    }
    match format_id {
        1 => preflight(input, ceilings),
        #[cfg(feature = "protobuf")]
        2 => {
            let message = protobuf_wire::SuffixAutomaton::decode(input)
                .map_err(|error| invalid(format!("malformed suffix protobuf: {error}")))?;
            if usize::try_from(message.string_count).ok() != Some(message.source_texts.len()) {
                return Err(invalid("suffix source text count mismatch"));
            }
            if message.source_texts.len() > ceilings.max_terms {
                return Err((
                    LlevStatus::LimitExceeded,
                    "suffix source count ceiling".into(),
                ));
            }
            let mut total = 0usize;
            for text in &message.source_texts {
                total = total.checked_add(text.len()).ok_or((
                    LlevStatus::LimitExceeded,
                    "suffix source byte overflow".into(),
                ))?;
                if text.len() > ceilings.max_term_bytes || total > ceilings.max_total_term_bytes {
                    return Err((
                        LlevStatus::LimitExceeded,
                        "suffix source byte ceiling".into(),
                    ));
                }
            }
            Ok(())
        }
        _ => Err(invalid("unsupported suffix source format ID")),
    }
}

/// Serialize source texts using native suffix-automaton bincode V1 or protobuf V1.
///
/// # Safety
/// Nonempty input slices and output pointer must be valid. Free output bytes once.
#[no_mangle]
pub unsafe extern "C" fn llev_suffix_source_serialize(
    format_id: u32,
    texts: *const LlevUtf8Slice,
    text_count: usize,
    raw_limits: *const LlevDictionaryLimits,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    boundary(|| {
        let output = out_bytes
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "suffix output is null".into()))?;
        *output = LlevOwnedBytes::default();
        #[cfg(feature = "serialization")]
        {
            require_format(format_id)?;
            let ceilings = limits(raw_limits)?;
            let source = input_strings(texts, text_count, ceilings)?;
            let automaton = SuffixAutomaton::from_texts(source);
            let expected = if format_id == 1 {
                let mut size = 8usize;
                for text in automaton.source_texts() {
                    size = size
                        .checked_add(8)
                        .and_then(|n| n.checked_add(text.len()))
                        .ok_or((
                            LlevStatus::LimitExceeded,
                            "suffix bincode size overflow".into(),
                        ))?;
                }
                if size > ceilings.max_payload_bytes {
                    return Err((
                        LlevStatus::LimitExceeded,
                        "suffix payload byte ceiling".into(),
                    ));
                }
                Some(size)
            } else {
                None
            };
            let mut writer = BoundedWriter {
                bytes: Vec::new(),
                max_bytes: ceilings.max_payload_bytes,
            };
            let result = match format_id {
                1 => BincodeSerializer::serialize_suffix_automaton(&automaton, &mut writer),
                #[cfg(feature = "protobuf")]
                2 => SuffixAutomatonProtobufSerializer::serialize_suffix_automaton(
                    &automaton,
                    &mut writer,
                ),
                _ => return Err(invalid("unsupported suffix source format ID")),
            };
            result.map_err(|error| {
                (
                    LlevStatus::LimitExceeded,
                    format!("bounded suffix serialization failed: {error}"),
                )
            })?;
            if expected.is_some_and(|size| writer.bytes.len() != size) {
                return Err(invalid(
                    "native suffix bincode encoder returned an incomplete payload",
                ));
            }
            preflight_suffix(&writer.bytes, format_id, ceilings)?;
            let boxed = writer.bytes.into_boxed_slice();
            output.len = boxed.len();
            output.data = Box::into_raw(boxed) as *mut u8;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (format_id, texts, text_count, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "suffix persistence requires serialization".into(),
            ))
        }
    })
}

/// Decode suffix source texts into an immutable owned string snapshot.
///
/// # Safety
/// Nonempty input and output pointer must be valid; free the snapshot once.
#[no_mangle]
pub unsafe extern "C" fn llev_suffix_source_deserialize(
    format_id: u32,
    data: *const u8,
    len: usize,
    raw_limits: *const LlevDictionaryLimits,
    out_texts: *mut *mut LlevDecodedDictionaryTerms,
) -> LlevStatus {
    boundary(|| {
        let output = out_texts.as_mut().ok_or((
            LlevStatus::NullPointer,
            "suffix source output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        #[cfg(feature = "serialization")]
        {
            require_format(format_id)?;
            let ceilings = limits(raw_limits)?;
            if len > ceilings.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "suffix payload byte ceiling".into(),
                ));
            }
            let input = if len == 0 {
                &[][..]
            } else {
                if data.is_null() {
                    return Err((LlevStatus::NullPointer, "suffix data is null".into()));
                }
                std::slice::from_raw_parts(data, len)
            };
            preflight_suffix(input, format_id, ceilings)?;
            let automaton: SuffixAutomaton = match format_id {
                1 => BincodeSerializer::deserialize_suffix_automaton(input),
                #[cfg(feature = "protobuf")]
                2 => SuffixAutomatonProtobufSerializer::deserialize_suffix_automaton(input),
                _ => return Err(invalid("unsupported suffix source format ID")),
            }
            .map_err(|error| invalid(format!("native suffix decode failed: {error}")))?;
            let texts = automaton.source_texts();
            let total = texts
                .iter()
                .try_fold(0usize, |sum, text| sum.checked_add(text.len()))
                .ok_or((
                    LlevStatus::LimitExceeded,
                    "decoded suffix bytes overflow".into(),
                ))?;
            if texts.len() > ceilings.max_terms
                || total > ceilings.max_total_term_bytes
                || texts
                    .iter()
                    .any(|text| text.len() > ceilings.max_term_bytes)
            {
                return Err((
                    LlevStatus::LimitExceeded,
                    "decoded suffix source ceiling".into(),
                ));
            }
            *output = Box::into_raw(Box::new(LlevDecodedDictionaryTerms { terms: texts }));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (format_id, data, len, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "suffix persistence requires serialization".into(),
            ))
        }
    })
}
