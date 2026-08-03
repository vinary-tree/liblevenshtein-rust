//! Portable protobuf persistence for generalized operation sets.
//!
//! The schema contains no maps, so canonical restriction ordering and retained
//! operation ordering make bytes emitted by this implementation deterministic.
//! A non-allocating wire preflight counts every known repeated/string resource
//! before `prost` constructs vectors or strings from untrusted input.

use super::operation_set_binary::{
    OperationSetLimitViolation, MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES,
};
use super::{
    OperationApplicability, OperationSet, OperationSetBinaryLimits, OperationSetValidationError,
    OperationType, SubstitutionPair, SubstitutionSet,
};
use prost::Message;
use std::fmt;

mod wire {
    include!(concat!(env!("OUT_DIR"), "/liblevenshtein.operations.rs"));
}

/// Failure while encoding or decoding an [`OperationSet`] protobuf message.
#[derive(Debug)]
pub enum OperationSetProtobufError {
    /// The operation set violates its semantic invariants.
    Validation(OperationSetValidationError),
    /// The encoded or supplied message exceeds the payload ceiling.
    PayloadTooLarge {
        /// Observed bytes.
        observed: usize,
        /// Accepted bytes.
        limit: usize,
    },
    /// The protobuf wire stream is malformed.
    MalformedWire(String),
    /// Prost could not encode the validated message.
    Encode(String),
    /// Prost could not decode the preflighted message.
    Decode(String),
    /// The container does not carry a supported schema version.
    UnsupportedFormat,
    /// A field cannot be represented by the Rust semantic model.
    InvalidField {
        /// Zero-based operation index, when the field belongs to an operation.
        operation: Option<usize>,
        /// Stable field identifier.
        field: &'static str,
        /// Reason for rejection.
        reason: &'static str,
    },
    /// A decoded collection exceeds a caller-selected resource policy.
    ResourceLimit {
        /// Stable resource identifier.
        resource: &'static str,
        /// Observed count or byte length.
        observed: usize,
        /// Accepted maximum.
        limit: usize,
    },
}

impl fmt::Display for OperationSetProtobufError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => write!(formatter, "invalid operation set: {error}"),
            Self::PayloadTooLarge { observed, limit } => write!(
                formatter,
                "operation-set protobuf contains {observed} bytes (limit {limit})"
            ),
            Self::MalformedWire(error) => {
                write!(formatter, "malformed operation-set protobuf: {error}")
            }
            Self::Encode(error) => write!(formatter, "cannot encode operation set: {error}"),
            Self::Decode(error) => write!(formatter, "cannot decode operation set: {error}"),
            Self::UnsupportedFormat => {
                formatter.write_str("unsupported or missing operation-set protobuf format")
            }
            Self::InvalidField {
                operation,
                field,
                reason,
            } => {
                if let Some(index) = operation {
                    write!(
                        formatter,
                        "operation {index} protobuf field {field} is invalid: {reason}"
                    )
                } else {
                    write!(formatter, "protobuf field {field} is invalid: {reason}")
                }
            }
            Self::ResourceLimit {
                resource,
                observed,
                limit,
            } => write!(
                formatter,
                "operation-set {resource} is {observed} (limit {limit})"
            ),
        }
    }
}

impl std::error::Error for OperationSetProtobufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<OperationSetValidationError> for OperationSetProtobufError {
    fn from(error: OperationSetValidationError) -> Self {
        Self::Validation(error)
    }
}

impl From<OperationSetLimitViolation> for OperationSetProtobufError {
    fn from(violation: OperationSetLimitViolation) -> Self {
        Self::ResourceLimit {
            resource: violation.resource,
            observed: violation.observed,
            limit: violation.limit,
        }
    }
}

impl OperationSet {
    /// Encode this operation set with the portable versioned protobuf schema.
    ///
    /// Operation order and exact IEEE-754 weight bits are retained. Listed
    /// substitution pairs are emitted in canonical order, making output from
    /// this implementation deterministic without relying on hash iteration.
    pub fn to_protobuf(&self) -> Result<Vec<u8>, OperationSetProtobufError> {
        let limits = OperationSetBinaryLimits::default();
        self.validate()?;
        self.validate_persistence_limits(limits)?;

        let message = wire::OperationSetContainer {
            format: Some(wire::operation_set_container::Format::V1(
                operation_set_to_wire(self)?,
            )),
        };
        let encoded_len = message.encoded_len();
        let payload_limit = limits
            .max_payload_bytes
            .min(MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES);
        if encoded_len > payload_limit {
            return Err(OperationSetProtobufError::PayloadTooLarge {
                observed: encoded_len,
                limit: payload_limit,
            });
        }
        let mut encoded = Vec::with_capacity(encoded_len);
        message
            .encode(&mut encoded)
            .map_err(|error| OperationSetProtobufError::Encode(error.to_string()))?;
        Ok(encoded)
    }

    /// Decode and validate a portable operation-set protobuf message.
    pub fn from_protobuf(bytes: &[u8]) -> Result<Self, OperationSetProtobufError> {
        Self::from_protobuf_with_limits(bytes, OperationSetBinaryLimits::default())
    }

    /// Decode protobuf with a caller-selected persistence resource policy.
    ///
    /// A wire-level preflight enforces operation, pair, name, and string-byte
    /// limits before `prost` allocates their decoded representations. Unknown
    /// protobuf fields remain forward-compatible and are skipped without being
    /// retained; a missing or unknown container version is rejected.
    pub fn from_protobuf_with_limits(
        bytes: &[u8],
        limits: OperationSetBinaryLimits,
    ) -> Result<Self, OperationSetProtobufError> {
        let payload_limit = limits
            .max_payload_bytes
            .min(MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES);
        if bytes.len() > payload_limit {
            return Err(OperationSetProtobufError::PayloadTooLarge {
                observed: bytes.len(),
                limit: payload_limit,
            });
        }
        preflight_wire(bytes, limits)?;
        let container = wire::OperationSetContainer::decode(bytes)
            .map_err(|error| OperationSetProtobufError::Decode(error.to_string()))?;
        let wire::operation_set_container::Format::V1(message) = container
            .format
            .ok_or(OperationSetProtobufError::UnsupportedFormat)?;
        let operation_set = operation_set_from_wire(message)?;
        operation_set.validate()?;
        operation_set.validate_persistence_limits(limits)?;
        Ok(operation_set)
    }
}

fn operation_set_to_wire(
    operation_set: &OperationSet,
) -> Result<wire::OperationSetV1, OperationSetProtobufError> {
    let operations = operation_set
        .operations()
        .iter()
        .map(operation_to_wire)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(wire::OperationSetV1 { operations })
}

fn operation_to_wire(
    operation: &OperationType,
) -> Result<wire::OperationTypeV1, OperationSetProtobufError> {
    let consume_x = u64::try_from(operation.consume_x()).map_err(|_| {
        OperationSetProtobufError::InvalidField {
            operation: None,
            field: "consume_x",
            reason: "value does not fit uint64",
        }
    })?;
    let consume_y = u64::try_from(operation.consume_y()).map_err(|_| {
        OperationSetProtobufError::InvalidField {
            operation: None,
            field: "consume_y",
            reason: "value does not fit uint64",
        }
    })?;
    let (applicability, restriction) = match operation.applicability() {
        OperationApplicability::Any => {
            (wire::OperationApplicabilityV1::ApplicabilityAny, Vec::new())
        }
        OperationApplicability::Equal => (
            wire::OperationApplicabilityV1::ApplicabilityEqual,
            Vec::new(),
        ),
        OperationApplicability::AdjacentTranspose => (
            wire::OperationApplicabilityV1::ApplicabilityAdjacentTranspose,
            Vec::new(),
        ),
        OperationApplicability::Listed(set) => (
            wire::OperationApplicabilityV1::ApplicabilityListed,
            set.pairs().into_iter().map(pair_to_wire).collect(),
        ),
    };
    Ok(wire::OperationTypeV1 {
        consume_x,
        consume_y,
        weight_bits: operation.weight().to_bits(),
        applicability: applicability.into(),
        restriction,
        name: operation.name().to_owned(),
    })
}

fn pair_to_wire(pair: SubstitutionPair) -> wire::SubstitutionPairV1 {
    use wire::substitution_pair_v1::Pair;

    let pair = match pair {
        SubstitutionPair::Bytes { source, target } => Pair::Bytes(wire::ByteSubstitutionV1 {
            source: u32::from(source),
            target: u32::from(target),
        }),
        SubstitutionPair::Strings { source, target } => Pair::Strings(wire::StringSubstitutionV1 {
            source: source.into(),
            target: target.into(),
        }),
    };
    wire::SubstitutionPairV1 { pair: Some(pair) }
}

fn operation_set_from_wire(
    message: wire::OperationSetV1,
) -> Result<OperationSet, OperationSetProtobufError> {
    let mut operation_set = OperationSet::with_capacity(message.operations.len());
    for (index, operation) in message.operations.into_iter().enumerate() {
        operation_set.add(operation_from_wire(index, operation)?);
    }
    Ok(operation_set)
}

fn operation_from_wire(
    index: usize,
    operation: wire::OperationTypeV1,
) -> Result<OperationType, OperationSetProtobufError> {
    let consume_x = usize::try_from(operation.consume_x)
        .map_err(|_| invalid_operation_field(index, "consume_x", "value does not fit usize"))?;
    let consume_y = usize::try_from(operation.consume_y)
        .map_err(|_| invalid_operation_field(index, "consume_y", "value does not fit usize"))?;
    let weight = f64::from_bits(operation.weight_bits);
    if !weight.is_finite() || weight < 0.0 {
        return Err(invalid_operation_field(
            index,
            "weight_bits",
            "weight must be finite and non-negative",
        ));
    }
    if weight == 0.0 && consume_x != consume_y {
        return Err(invalid_operation_field(
            index,
            "weight_bits",
            "zero-weight operations must preserve length",
        ));
    }

    let applicability = wire::OperationApplicabilityV1::try_from(operation.applicability)
        .map_err(|_| invalid_operation_field(index, "applicability", "unknown discriminator"))?;
    let semantic_applicability = match applicability {
        wire::OperationApplicabilityV1::ApplicabilityUnspecified => {
            return Err(invalid_operation_field(
                index,
                "applicability",
                "unspecified discriminator",
            ));
        }
        wire::OperationApplicabilityV1::ApplicabilityAny => {
            require_empty_restriction(index, &operation.restriction)?;
            OperationApplicability::Any
        }
        wire::OperationApplicabilityV1::ApplicabilityEqual => {
            require_empty_restriction(index, &operation.restriction)?;
            OperationApplicability::Equal
        }
        wire::OperationApplicabilityV1::ApplicabilityAdjacentTranspose => {
            require_empty_restriction(index, &operation.restriction)?;
            OperationApplicability::AdjacentTranspose
        }
        wire::OperationApplicabilityV1::ApplicabilityListed => {
            OperationApplicability::Listed(restriction_from_wire(index, operation.restriction)?)
        }
    };

    Ok(OperationType::with_owned_applicability(
        consume_x,
        consume_y,
        weight,
        semantic_applicability,
        operation.name,
    ))
}

fn require_empty_restriction(
    index: usize,
    restriction: &[wire::SubstitutionPairV1],
) -> Result<(), OperationSetProtobufError> {
    if restriction.is_empty() {
        Ok(())
    } else {
        Err(invalid_operation_field(
            index,
            "restriction",
            "pairs require listed applicability",
        ))
    }
}

fn restriction_from_wire(
    operation_index: usize,
    pairs: Vec<wire::SubstitutionPairV1>,
) -> Result<SubstitutionSet, OperationSetProtobufError> {
    use wire::substitution_pair_v1::Pair;

    let mut restriction = SubstitutionSet::with_capacity(pairs.len());
    for pair in pairs {
        match pair.pair.ok_or_else(|| {
            invalid_operation_field(operation_index, "restriction.pair", "missing oneof value")
        })? {
            Pair::Bytes(bytes) => {
                let source = u8::try_from(bytes.source).map_err(|_| {
                    invalid_operation_field(
                        operation_index,
                        "restriction.bytes.source",
                        "value exceeds 255",
                    )
                })?;
                let target = u8::try_from(bytes.target).map_err(|_| {
                    invalid_operation_field(
                        operation_index,
                        "restriction.bytes.target",
                        "value exceeds 255",
                    )
                })?;
                restriction.allow_byte(source, target);
            }
            Pair::Strings(strings) => {
                if strings.source.is_empty() || strings.target.is_empty() {
                    return Err(invalid_operation_field(
                        operation_index,
                        "restriction.strings",
                        "source and target must be non-empty",
                    ));
                }
                restriction.allow_str(&strings.source, &strings.target);
            }
        }
    }
    Ok(restriction)
}

fn invalid_operation_field(
    operation: usize,
    field: &'static str,
    reason: &'static str,
) -> OperationSetProtobufError {
    OperationSetProtobufError::InvalidField {
        operation: Some(operation),
        field,
        reason,
    }
}

#[derive(Default)]
struct PreflightCounts {
    operations: usize,
    total_pairs: usize,
    text_bytes: usize,
}

fn preflight_wire(
    bytes: &[u8],
    limits: OperationSetBinaryLimits,
) -> Result<(), OperationSetProtobufError> {
    let mut counts = PreflightCounts::default();
    scan_container(WireCursor::new(bytes, 0), limits, &mut counts)
}

fn scan_container(
    mut cursor: WireCursor<'_>,
    limits: OperationSetBinaryLimits,
    counts: &mut PreflightCounts,
) -> Result<(), OperationSetProtobufError> {
    while !cursor.is_empty() {
        let (tag, wire_type) = cursor.read_key()?;
        if tag == 1 {
            require_wire_type(tag, wire_type, 2, cursor.offset)?;
            let (message, offset) = cursor.read_length_delimited()?;
            scan_operation_set(WireCursor::new(message, offset), limits, counts)?;
        } else {
            cursor.skip_value(wire_type)?;
        }
    }
    Ok(())
}

fn scan_operation_set(
    mut cursor: WireCursor<'_>,
    limits: OperationSetBinaryLimits,
    counts: &mut PreflightCounts,
) -> Result<(), OperationSetProtobufError> {
    while !cursor.is_empty() {
        let (tag, wire_type) = cursor.read_key()?;
        if tag == 1 {
            require_wire_type(tag, wire_type, 2, cursor.offset)?;
            counts.operations = counts.operations.checked_add(1).ok_or(
                OperationSetProtobufError::ResourceLimit {
                    resource: "operation count",
                    observed: usize::MAX,
                    limit: limits.max_operations,
                },
            )?;
            check_limit("operation count", counts.operations, limits.max_operations)?;
            let (operation, offset) = cursor.read_length_delimited()?;
            scan_operation(WireCursor::new(operation, offset), limits, counts)?;
        } else {
            cursor.skip_value(wire_type)?;
        }
    }
    Ok(())
}

fn scan_operation(
    mut cursor: WireCursor<'_>,
    limits: OperationSetBinaryLimits,
    counts: &mut PreflightCounts,
) -> Result<(), OperationSetProtobufError> {
    let mut operation_pairs = 0usize;
    while !cursor.is_empty() {
        let (tag, wire_type) = cursor.read_key()?;
        match tag {
            5 => {
                require_wire_type(tag, wire_type, 2, cursor.offset)?;
                operation_pairs = operation_pairs.checked_add(1).ok_or(
                    OperationSetProtobufError::ResourceLimit {
                        resource: "restriction pairs per operation",
                        observed: usize::MAX,
                        limit: limits.max_restriction_pairs_per_operation,
                    },
                )?;
                check_limit(
                    "restriction pairs per operation",
                    operation_pairs,
                    limits.max_restriction_pairs_per_operation,
                )?;
                counts.total_pairs = counts.total_pairs.checked_add(1).ok_or(
                    OperationSetProtobufError::ResourceLimit {
                        resource: "total restriction pairs",
                        observed: usize::MAX,
                        limit: limits.max_total_restriction_pairs,
                    },
                )?;
                check_limit(
                    "total restriction pairs",
                    counts.total_pairs,
                    limits.max_total_restriction_pairs,
                )?;
                let (pair, offset) = cursor.read_length_delimited()?;
                scan_pair(WireCursor::new(pair, offset), limits, counts)?;
            }
            6 => {
                require_wire_type(tag, wire_type, 2, cursor.offset)?;
                let (name, _) = cursor.read_length_delimited()?;
                check_limit(
                    "operation-name bytes",
                    name.len(),
                    limits.max_operation_name_bytes,
                )?;
            }
            _ => cursor.skip_value(wire_type)?,
        }
    }
    Ok(())
}

fn scan_pair(
    mut cursor: WireCursor<'_>,
    limits: OperationSetBinaryLimits,
    counts: &mut PreflightCounts,
) -> Result<(), OperationSetProtobufError> {
    while !cursor.is_empty() {
        let (tag, wire_type) = cursor.read_key()?;
        if tag == 2 {
            require_wire_type(tag, wire_type, 2, cursor.offset)?;
            let (strings, offset) = cursor.read_length_delimited()?;
            scan_string_pair(WireCursor::new(strings, offset), limits, counts)?;
        } else {
            cursor.skip_value(wire_type)?;
        }
    }
    Ok(())
}

fn scan_string_pair(
    mut cursor: WireCursor<'_>,
    limits: OperationSetBinaryLimits,
    counts: &mut PreflightCounts,
) -> Result<(), OperationSetProtobufError> {
    while !cursor.is_empty() {
        let (tag, wire_type) = cursor.read_key()?;
        if tag == 1 || tag == 2 {
            require_wire_type(tag, wire_type, 2, cursor.offset)?;
            let (text, _) = cursor.read_length_delimited()?;
            counts.text_bytes = counts.text_bytes.checked_add(text.len()).ok_or(
                OperationSetProtobufError::ResourceLimit {
                    resource: "restriction text bytes",
                    observed: usize::MAX,
                    limit: limits.max_restriction_text_bytes,
                },
            )?;
            check_limit(
                "restriction text bytes",
                counts.text_bytes,
                limits.max_restriction_text_bytes,
            )?;
        } else {
            cursor.skip_value(wire_type)?;
        }
    }
    Ok(())
}

fn check_limit(
    resource: &'static str,
    observed: usize,
    limit: usize,
) -> Result<(), OperationSetProtobufError> {
    if observed > limit {
        Err(OperationSetProtobufError::ResourceLimit {
            resource,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}

fn require_wire_type(
    tag: u32,
    observed: u8,
    expected: u8,
    offset: usize,
) -> Result<(), OperationSetProtobufError> {
    if observed == expected {
        Ok(())
    } else {
        Err(OperationSetProtobufError::MalformedWire(format!(
            "field {tag} at byte {offset} has wire type {observed}, expected {expected}"
        )))
    }
}

struct WireCursor<'a> {
    remaining: &'a [u8],
    offset: usize,
}

impl<'a> WireCursor<'a> {
    fn new(remaining: &'a [u8], offset: usize) -> Self {
        Self { remaining, offset }
    }

    fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }

    fn read_key(&mut self) -> Result<(u32, u8), OperationSetProtobufError> {
        let key_offset = self.offset;
        let key = self.read_varint()?;
        let tag = key >> 3;
        if tag == 0 || tag > 0x1fff_ffff {
            return Err(OperationSetProtobufError::MalformedWire(format!(
                "invalid field tag {tag} at byte {key_offset}"
            )));
        }
        Ok((tag as u32, (key & 0x07) as u8))
    }

    fn read_varint(&mut self) -> Result<u64, OperationSetProtobufError> {
        let start = self.offset;
        let mut value = 0u64;
        for index in 0..10 {
            let byte = self.take_exact(1)?[0];
            if index == 9 && byte > 1 {
                return Err(OperationSetProtobufError::MalformedWire(format!(
                    "varint overflows uint64 at byte {start}"
                )));
            }
            value |= u64::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(OperationSetProtobufError::MalformedWire(format!(
            "unterminated varint at byte {start}"
        )))
    }

    fn read_length_delimited(&mut self) -> Result<(&'a [u8], usize), OperationSetProtobufError> {
        let declared = self.read_varint()?;
        let length = usize::try_from(declared).map_err(|_| {
            OperationSetProtobufError::MalformedWire(format!(
                "length {declared} at byte {} does not fit usize",
                self.offset
            ))
        })?;
        let offset = self.offset;
        Ok((self.take_exact(length)?, offset))
    }

    fn skip_value(&mut self, wire_type: u8) -> Result<(), OperationSetProtobufError> {
        match wire_type {
            0 => {
                self.read_varint()?;
            }
            1 => {
                self.take_exact(8)?;
            }
            2 => {
                self.read_length_delimited()?;
            }
            5 => {
                self.take_exact(4)?;
            }
            _ => {
                return Err(OperationSetProtobufError::MalformedWire(format!(
                    "unsupported wire type {wire_type} at byte {}",
                    self.offset
                )));
            }
        }
        Ok(())
    }

    fn take_exact(&mut self, length: usize) -> Result<&'a [u8], OperationSetProtobufError> {
        if self.remaining.len() < length {
            return Err(OperationSetProtobufError::MalformedWire(format!(
                "field at byte {} declares {length} bytes but only {} remain",
                self.offset,
                self.remaining.len()
            )));
        }
        let (taken, remaining) = self.remaining.split_at(length);
        self.remaining = remaining;
        self.offset = self.offset.checked_add(length).ok_or_else(|| {
            OperationSetProtobufError::MalformedWire("wire offset overflowed usize".to_owned())
        })?;
        Ok(taken)
    }
}
