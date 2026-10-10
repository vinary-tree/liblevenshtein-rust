//! Native, versioned persistence for generalized operation sets.

#[cfg(feature = "serialization")]
use super::automata::parse_operation_set;
use super::automata::LlevGeneralizedOperation;
use super::index::boundary;
use super::{LlevOwnedBytes, LlevStatus};
#[cfg(feature = "serialization")]
use crate::transducer::OperationSetBinaryLimits;
use crate::transducer::{OperationApplicability, OperationSet, SubstitutionPair};

#[cfg(feature = "serialization")]
const BINARY_V1: u32 = 1;
#[cfg(feature = "serialization")]
const PROTOBUF_V1: u32 = 2;
#[cfg(feature = "serialization")]
const GZIP_BINARY_V1: u32 = 3;
#[cfg(feature = "serialization")]
const GZIP_PROTOBUF_V1: u32 = 4;

/// Caller-selected resource policy for one operation-set codec call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevOperationSetLimits {
    /// Maximum encoded and inflated payload bytes.
    pub max_payload_bytes: usize,
    /// Maximum operation count.
    pub max_operations: usize,
    /// Maximum UTF-8 bytes in one operation name.
    pub max_operation_name_bytes: usize,
    /// Maximum listed pairs on one operation.
    pub max_restriction_pairs_per_operation: usize,
    /// Maximum listed pairs in the set.
    pub max_total_restriction_pairs: usize,
    /// Maximum UTF-8 bytes across all listed text pairs.
    pub max_restriction_text_bytes: usize,
}

#[cfg(feature = "serialization")]
impl From<LlevOperationSetLimits> for OperationSetBinaryLimits {
    fn from(value: LlevOperationSetLimits) -> Self {
        let maximum = OperationSetBinaryLimits::default();
        Self {
            max_payload_bytes: value.max_payload_bytes.min(maximum.max_payload_bytes),
            max_operations: value.max_operations.min(maximum.max_operations),
            max_operation_name_bytes: value
                .max_operation_name_bytes
                .min(maximum.max_operation_name_bytes),
            max_restriction_pairs_per_operation: value
                .max_restriction_pairs_per_operation
                .min(maximum.max_restriction_pairs_per_operation),
            max_total_restriction_pairs: value
                .max_total_restriction_pairs
                .min(maximum.max_total_restriction_pairs),
            max_restriction_text_bytes: value
                .max_restriction_text_bytes
                .min(maximum.max_restriction_text_bytes),
        }
    }
}

/// Borrowed metadata for one operation in a decoded snapshot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevSerializedOperationView {
    /// Source scalars consumed by this operation.
    pub consume_source: usize,
    /// Target scalars consumed by this operation.
    pub consume_target: usize,
    /// Nonnegative finite edit cost.
    pub weight: f64,
    /// Borrowed UTF-8 operation name.
    pub name_data: *const u8,
    /// Byte length of the operation name.
    pub name_len: usize,
    /// Any 0, equal 1, adjacent transpose 2, or listed 3.
    pub applicability: u32,
    /// Number of canonical restriction pairs.
    pub restriction_count: usize,
}

/// Borrowed view of either a raw byte pair (kind 1) or UTF-8 pair (kind 2).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevSerializedRestrictionView {
    /// One for a byte pair, two for a UTF-8 string pair.
    pub kind: u32,
    /// Raw source byte when kind is one.
    pub source_byte: u8,
    /// Raw target byte when kind is one.
    pub target_byte: u8,
    /// Fixed zero padding.
    pub reserved: [u8; 2],
    /// Borrowed UTF-8 source when kind is two.
    pub source_data: *const u8,
    /// Source text byte length.
    pub source_len: usize,
    /// Borrowed UTF-8 target when kind is two.
    pub target_data: *const u8,
    /// Target text byte length.
    pub target_len: usize,
}

/// Owns the native operation set and stable canonical restriction views.
pub struct LlevDecodedOperationSet {
    operations: OperationSet,
    restrictions: Vec<Vec<SubstitutionPair>>,
}

#[cfg(feature = "serialization")]
impl LlevDecodedOperationSet {
    fn new(operations: OperationSet) -> Self {
        let restrictions = operations
            .operations()
            .iter()
            .map(|operation| match operation.applicability() {
                OperationApplicability::Listed(pairs) => pairs.pairs(),
                _ => Vec::new(),
            })
            .collect();
        Self {
            operations,
            restrictions,
        }
    }
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

#[cfg(feature = "serialization")]
fn limited(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::LimitExceeded, message.into())
}

#[cfg(feature = "serialization")]
fn require_format(format_id: u32) -> Result<(), (LlevStatus, String)> {
    match format_id {
        BINARY_V1 if cfg!(feature = "serialization") => Ok(()),
        PROTOBUF_V1 if cfg!(feature = "protobuf") => Ok(()),
        GZIP_BINARY_V1 if cfg!(feature = "compression") => Ok(()),
        GZIP_PROTOBUF_V1 if cfg!(all(feature = "compression", feature = "protobuf")) => Ok(()),
        BINARY_V1 | PROTOBUF_V1 | GZIP_BINARY_V1 | GZIP_PROTOBUF_V1 => Err((
            LlevStatus::Unsupported,
            "operation-set format is not built into this library".into(),
        )),
        _ => Err(invalid("unsupported operation-set format ID")),
    }
}

#[cfg(feature = "serialization")]
fn limits(
    raw: *const LlevOperationSetLimits,
) -> Result<OperationSetBinaryLimits, (LlevStatus, String)> {
    let raw = unsafe { raw.as_ref() }.ok_or((
        LlevStatus::NullPointer,
        "operation-set limits are null".into(),
    ))?;
    Ok((*raw).into())
}

#[cfg(feature = "serialization")]
fn serialize_native(
    set: &OperationSet,
    format_id: u32,
    policy: OperationSetBinaryLimits,
) -> Result<Vec<u8>, (LlevStatus, String)> {
    require_format(format_id)?;
    let bytes = match format_id {
        #[cfg(feature = "serialization")]
        BINARY_V1 => set
            .to_binary()
            .map_err(|error| invalid(error.to_string()))?,
        #[cfg(feature = "protobuf")]
        PROTOBUF_V1 => set
            .to_protobuf()
            .map_err(|error| invalid(error.to_string()))?,
        #[cfg(feature = "compression")]
        GZIP_BINARY_V1 => set
            .to_binary_gzip()
            .map_err(|error| invalid(error.to_string()))?,
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V1 => set
            .to_protobuf_gzip()
            .map_err(|error| invalid(error.to_string()))?,
        _ => return Err(invalid("unsupported operation-set format ID")),
    };
    if bytes.len() > policy.max_payload_bytes {
        return Err(limited("operation-set encoded payload byte ceiling"));
    }
    // Apply the same policy to the inflated representation and every field.
    let _ = deserialize_native(&bytes, format_id, policy)?;
    Ok(bytes)
}

#[cfg(feature = "serialization")]
fn deserialize_native(
    bytes: &[u8],
    format_id: u32,
    policy: OperationSetBinaryLimits,
) -> Result<OperationSet, (LlevStatus, String)> {
    require_format(format_id)?;
    if bytes.len() > policy.max_payload_bytes {
        return Err(limited("operation-set input payload byte ceiling"));
    }
    let result = match format_id {
        #[cfg(feature = "serialization")]
        BINARY_V1 => OperationSet::from_binary_with_limits(bytes, policy)
            .map_err(|error| invalid(error.to_string())),
        #[cfg(feature = "protobuf")]
        PROTOBUF_V1 => OperationSet::from_protobuf_with_limits(bytes, policy)
            .map_err(|error| invalid(error.to_string())),
        #[cfg(feature = "compression")]
        GZIP_BINARY_V1 => OperationSet::from_binary_gzip_with_limits(bytes, policy)
            .map_err(|error| invalid(error.to_string())),
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V1 => OperationSet::from_protobuf_gzip_with_limits(bytes, policy)
            .map_err(|error| invalid(error.to_string())),
        _ => Err(invalid("unsupported operation-set format ID")),
    }?;
    Ok(result)
}

#[cfg(feature = "serialization")]
fn publish(
    bytes: Vec<u8>,
    out_bytes: *mut LlevOwnedBytes,
) -> Result<LlevStatus, (LlevStatus, String)> {
    let output = unsafe { out_bytes.as_mut() }.ok_or((
        LlevStatus::NullPointer,
        "operation-set output is null".into(),
    ))?;
    let boxed = bytes.into_boxed_slice();
    output.len = boxed.len();
    output.data = Box::into_raw(boxed) as *mut u8;
    Ok(LlevStatus::Ok)
}

/// Encode Julia/C operation descriptors with a native operation-set codec.
///
/// # Safety
/// Descriptors, nested borrowed slices, limits, and output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_operation_set_serialize(
    format_id: u32,
    operations: *const LlevGeneralizedOperation,
    operation_count: usize,
    raw_limits: *const LlevOperationSetLimits,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    #[cfg(feature = "serialization")]
    return boundary(|| {
        let output = out_bytes.as_mut().ok_or((
            LlevStatus::NullPointer,
            "operation-set output is null".into(),
        ))?;
        *output = LlevOwnedBytes::default();
        require_format(format_id)?;
        let policy = limits(raw_limits)?;
        if operation_count > policy.max_operations {
            return Err(limited("operation count ceiling"));
        }
        let set = parse_operation_set(operations, operation_count)?;
        publish(serialize_native(&set, format_id, policy)?, out_bytes)
    });
    #[cfg(not(feature = "serialization"))]
    boundary(|| {
        let output = out_bytes.as_mut().ok_or((
            LlevStatus::NullPointer,
            "operation-set output is null".into(),
        ))?;
        *output = LlevOwnedBytes::default();
        let _ = (format_id, operations, operation_count, raw_limits);
        Err((
            LlevStatus::Unsupported,
            "operation-set persistence requires serialization".into(),
        ))
    })
}

/// Decode one complete native operation-set payload into an owned snapshot.
///
/// # Safety
/// Nonempty input and output pointer must be valid; free the handle once.
#[no_mangle]
pub unsafe extern "C" fn llev_operation_set_deserialize(
    format_id: u32,
    data: *const u8,
    len: usize,
    raw_limits: *const LlevOperationSetLimits,
    out_set: *mut *mut LlevDecodedOperationSet,
) -> LlevStatus {
    #[cfg(feature = "serialization")]
    return boundary(|| {
        let output = out_set.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded operation-set output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        require_format(format_id)?;
        let policy = limits(raw_limits)?;
        if len > policy.max_payload_bytes {
            return Err(limited("operation-set input payload byte ceiling"));
        }
        let input = if len == 0 {
            &[][..]
        } else {
            if data.is_null() {
                return Err((LlevStatus::NullPointer, "operation-set data is null".into()));
            }
            std::slice::from_raw_parts(data, len)
        };
        let decoded = deserialize_native(input, format_id, policy)?;
        *output = Box::into_raw(Box::new(LlevDecodedOperationSet::new(decoded)));
        Ok(LlevStatus::Ok)
    });
    #[cfg(not(feature = "serialization"))]
    boundary(|| {
        let output = out_set.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded operation-set output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let _ = (format_id, data, len, raw_limits);
        Err((
            LlevStatus::Unsupported,
            "operation-set persistence requires serialization".into(),
        ))
    })
}

/// Re-encode a decoded set, preserving raw-byte restrictions exactly.
///
/// # Safety
/// Handle, limits, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_operation_set_serialize(
    set: *const LlevDecodedOperationSet,
    format_id: u32,
    raw_limits: *const LlevOperationSetLimits,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    #[cfg(feature = "serialization")]
    return boundary(|| {
        let output = out_bytes.as_mut().ok_or((
            LlevStatus::NullPointer,
            "operation-set output is null".into(),
        ))?;
        *output = LlevOwnedBytes::default();
        let source = set.as_ref().ok_or((
            LlevStatus::NullPointer,
            "decoded operation set is null".into(),
        ))?;
        let policy = limits(raw_limits)?;
        publish(
            serialize_native(&source.operations, format_id, policy)?,
            out_bytes,
        )
    });
    #[cfg(not(feature = "serialization"))]
    boundary(|| {
        let output = out_bytes.as_mut().ok_or((
            LlevStatus::NullPointer,
            "operation-set output is null".into(),
        ))?;
        *output = LlevOwnedBytes::default();
        let _ = (set, format_id, raw_limits);
        Err((
            LlevStatus::Unsupported,
            "operation-set persistence requires serialization".into(),
        ))
    })
}

/// Count operations in a decoded snapshot.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_operation_set_len(
    set: *const LlevDecodedOperationSet,
    out_len: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let source = set.as_ref().ok_or((
            LlevStatus::NullPointer,
            "decoded operation set is null".into(),
        ))?;
        *out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "operation count output is null".into(),
        ))? = source.operations.operations().len();
        Ok(LlevStatus::Ok)
    })
}

/// Borrow operation metadata until the decoded snapshot is freed.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_operation_set_operation_at(
    set: *const LlevDecodedOperationSet,
    index: usize,
    out_view: *mut LlevSerializedOperationView,
) -> LlevStatus {
    boundary(|| {
        let source = set.as_ref().ok_or((
            LlevStatus::NullPointer,
            "decoded operation set is null".into(),
        ))?;
        let operation = source
            .operations
            .operations()
            .get(index)
            .ok_or_else(|| invalid("operation index is out of bounds"))?;
        let applicability = match operation.applicability() {
            OperationApplicability::Any => 0,
            OperationApplicability::Equal => 1,
            OperationApplicability::AdjacentTranspose => 2,
            OperationApplicability::Listed(_) => 3,
        };
        *out_view.as_mut().ok_or((
            LlevStatus::NullPointer,
            "operation view output is null".into(),
        ))? = LlevSerializedOperationView {
            consume_source: operation.consume_x(),
            consume_target: operation.consume_y(),
            weight: operation.weight(),
            name_data: operation.name().as_ptr(),
            name_len: operation.name().len(),
            applicability,
            restriction_count: source.restrictions[index].len(),
        };
        Ok(LlevStatus::Ok)
    })
}

/// Borrow one canonical restriction view until the snapshot is freed.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_operation_set_restriction_at(
    set: *const LlevDecodedOperationSet,
    operation_index: usize,
    restriction_index: usize,
    out_view: *mut LlevSerializedRestrictionView,
) -> LlevStatus {
    boundary(|| {
        let source = set.as_ref().ok_or((
            LlevStatus::NullPointer,
            "decoded operation set is null".into(),
        ))?;
        let restriction = source
            .restrictions
            .get(operation_index)
            .and_then(|pairs| pairs.get(restriction_index))
            .ok_or_else(|| invalid("restriction index is out of bounds"))?;
        let view = match restriction {
            SubstitutionPair::Bytes { source, target } => LlevSerializedRestrictionView {
                kind: 1,
                source_byte: *source,
                target_byte: *target,
                ..LlevSerializedRestrictionView::default()
            },
            SubstitutionPair::Strings { source, target } => LlevSerializedRestrictionView {
                kind: 2,
                source_data: source.as_ptr(),
                source_len: source.len(),
                target_data: target.as_ptr(),
                target_len: target.len(),
                ..LlevSerializedRestrictionView::default()
            },
        };
        *out_view.as_mut().ok_or((
            LlevStatus::NullPointer,
            "restriction view output is null".into(),
        ))? = view;
        Ok(LlevStatus::Ok)
    })
}

/// Free one decoded operation-set snapshot.
///
/// # Safety
/// Handle must be null or returned by `llev_operation_set_deserialize` and freed once.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_operation_set_free(set: *mut LlevDecodedOperationSet) {
    if !set.is_null() {
        drop(Box::from_raw(set));
    }
}
