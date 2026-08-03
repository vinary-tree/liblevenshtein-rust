//! Versioned compact binary persistence for generalized operation sets.
//!
//! The public operation-model types deliberately do not implement generic
//! Serde traits. This module keeps Serde on private wire types and exposes only
//! a small fixed envelope around a deterministic bincode payload. Consequently
//! callers cannot select JSON, TOML, or another text encoding for operation
//! persistence.

use super::operation_type::MAX_OPERATION_NAME_BYTES;
use super::{
    OperationApplicability, OperationSet, OperationSetValidationError, OperationType,
    SubstitutionPair, SubstitutionSet, MAX_OPERATION_SET_TOTAL_CONSUMPTION, MAX_SUBSTITUTION_PAIRS,
    MAX_SUBSTITUTION_TEXT_BYTES,
};
use serde::{
    de::{SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::fmt;

/// Magic prefix for an `OperationSet` binary envelope.
pub const OPERATION_SET_BINARY_MAGIC: [u8; 8] = *b"LLEVOPS\0";

/// Current binary-envelope version.
pub const OPERATION_SET_BINARY_VERSION: u16 = 1;

/// Maximum accepted payload size (64 MiB).
pub const MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

pub(super) const HEADER_BYTES: usize = 20;
const SUPPORTED_FLAGS: u16 = 0;

/// Private bincode representation of a complete operation set.
///
/// Newtype and field ordering intentionally match the version-1 representation
/// that predated the removal of public generic Serde implementations.
#[derive(Serialize)]
struct BinaryOperationSet(Vec<BinaryOperationType>);

#[derive(Serialize, Deserialize)]
struct BinaryOperationType {
    consume_x: u64,
    consume_y: u64,
    weight: f64,
    applicability: BinaryApplicability,
    name: String,
}

#[derive(Serialize, Deserialize)]
enum BinaryApplicability {
    Any,
    Equal,
    AdjacentTranspose,
    Listed(BinarySubstitutionSet),
}

#[derive(Serialize)]
struct BinarySubstitutionSet(Vec<BinarySubstitutionPair>);

#[derive(Serialize, Deserialize)]
enum BinarySubstitutionPair {
    Bytes { source: u8, target: u8 },
    Strings { source: Box<str>, target: Box<str> },
}

impl<'de> Deserialize<'de> for BinaryOperationSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OperationSetVisitor;

        impl<'de> Visitor<'de> for OperationSetVisitor {
            type Value = BinaryOperationSet;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a bounded sequence of generalized operations")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let hinted = sequence.size_hint().unwrap_or(0);
                if hinted > MAX_OPERATION_SET_TOTAL_CONSUMPTION {
                    return Err(serde::de::Error::custom(format_args!(
                        "operation sequence declares {hinted} entries (limit {MAX_OPERATION_SET_TOTAL_CONSUMPTION})"
                    )));
                }
                let mut operations = Vec::with_capacity(hinted);
                while let Some(operation) = sequence.next_element()? {
                    if operations.len() == MAX_OPERATION_SET_TOTAL_CONSUMPTION {
                        return Err(serde::de::Error::custom(format_args!(
                            "operation sequence exceeds {MAX_OPERATION_SET_TOTAL_CONSUMPTION} entries"
                        )));
                    }
                    operations.push(operation);
                }
                Ok(BinaryOperationSet(operations))
            }
        }

        deserializer.deserialize_seq(OperationSetVisitor)
    }
}

impl<'de> Deserialize<'de> for BinarySubstitutionSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SubstitutionSetVisitor;

        impl<'de> Visitor<'de> for SubstitutionSetVisitor {
            type Value = BinarySubstitutionSet;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a bounded sequence of substitution pairs")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let hinted = sequence.size_hint().unwrap_or(0);
                if hinted > MAX_SUBSTITUTION_PAIRS {
                    return Err(serde::de::Error::custom(format_args!(
                        "substitution sequence declares {hinted} pairs (limit {MAX_SUBSTITUTION_PAIRS})"
                    )));
                }
                let mut pairs = Vec::with_capacity(hinted);
                let mut text_bytes = 0usize;
                while let Some(pair) = sequence.next_element::<BinarySubstitutionPair>()? {
                    if pairs.len() == MAX_SUBSTITUTION_PAIRS {
                        return Err(serde::de::Error::custom(format_args!(
                            "substitution sequence exceeds {MAX_SUBSTITUTION_PAIRS} pairs"
                        )));
                    }
                    if let BinarySubstitutionPair::Strings { source, target } = &pair {
                        if source.is_empty() || target.is_empty() {
                            return Err(serde::de::Error::custom(
                                "serialized substitution strings must be non-empty",
                            ));
                        }
                        text_bytes = text_bytes
                            .checked_add(source.len())
                            .and_then(|value| value.checked_add(target.len()))
                            .ok_or_else(|| {
                                serde::de::Error::custom(
                                    "serialized substitution text length overflowed usize",
                                )
                            })?;
                        if text_bytes > MAX_SUBSTITUTION_TEXT_BYTES {
                            return Err(serde::de::Error::custom(format_args!(
                                "substitution strings contain {text_bytes} UTF-8 bytes (limit {MAX_SUBSTITUTION_TEXT_BYTES})"
                            )));
                        }
                    }
                    pairs.push(pair);
                }
                Ok(BinarySubstitutionSet(pairs))
            }
        }

        deserializer.deserialize_seq(SubstitutionSetVisitor)
    }
}

/// Resource policy applied while decoding an operation-set persistence format.
///
/// The default limits are independent of dictionary size: an operation set
/// describes the edit grammar and remains small even for very large
/// dictionaries. The same policy is used by the bincode envelope and the
/// protobuf schema. Callers may impose stricter limits at trust boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationSetBinaryLimits {
    /// Maximum declared payload bytes.
    pub max_payload_bytes: usize,
    /// Maximum number of operations.
    pub max_operations: usize,
    /// Maximum UTF-8 bytes in one diagnostic operation name.
    pub max_operation_name_bytes: usize,
    /// Maximum restriction pairs attached to one operation.
    pub max_restriction_pairs_per_operation: usize,
    /// Maximum restriction pairs across the entire operation set.
    pub max_total_restriction_pairs: usize,
    /// Maximum aggregate UTF-8 bytes in string restrictions.
    pub max_restriction_text_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct OperationSetLimitViolation {
    pub(super) resource: &'static str,
    pub(super) observed: usize,
    pub(super) limit: usize,
}

impl Default for OperationSetBinaryLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES,
            max_operations: MAX_OPERATION_SET_TOTAL_CONSUMPTION,
            max_operation_name_bytes: MAX_OPERATION_NAME_BYTES,
            max_restriction_pairs_per_operation: MAX_SUBSTITUTION_PAIRS,
            max_total_restriction_pairs: MAX_SUBSTITUTION_PAIRS,
            max_restriction_text_bytes: MAX_SUBSTITUTION_TEXT_BYTES,
        }
    }
}

/// Failure while encoding or decoding an [`OperationSet`] binary envelope.
#[derive(Debug)]
pub enum OperationSetBinaryError {
    /// The operation set violates its semantic or resource invariants.
    Validation(OperationSetValidationError),
    /// Bincode could not encode the validated payload.
    Encode(String),
    /// Input is shorter than the fixed envelope header.
    TruncatedHeader {
        /// Bytes present.
        observed: usize,
        /// Bytes required.
        required: usize,
    },
    /// The envelope magic does not identify an operation set.
    InvalidMagic,
    /// The envelope uses a version this library does not understand.
    UnsupportedVersion(u16),
    /// Reserved feature flags are non-zero.
    UnsupportedFlags(u16),
    /// The declared payload exceeds the allocation/work limit.
    PayloadTooLarge {
        /// Declared bytes.
        observed: u64,
        /// Accepted bytes.
        limit: usize,
    },
    /// The byte slice does not contain exactly the declared payload.
    LengthMismatch {
        /// Total bytes implied by the header.
        expected: usize,
        /// Total bytes supplied.
        observed: usize,
    },
    /// Bincode could not decode the payload.
    Decode(String),
    /// Bincode decoded a prefix rather than the complete payload.
    TrailingPayload {
        /// Bytes consumed by the decoder.
        consumed: usize,
        /// Bytes in the declared payload.
        payload_len: usize,
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

impl fmt::Display for OperationSetBinaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => write!(formatter, "invalid operation set: {error}"),
            Self::Encode(error) => write!(formatter, "cannot encode operation set: {error}"),
            Self::TruncatedHeader { observed, required } => write!(
                formatter,
                "operation-set binary header has {observed} bytes (requires {required})"
            ),
            Self::InvalidMagic => formatter.write_str("invalid operation-set binary magic"),
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported operation-set binary version {version}"
                )
            }
            Self::UnsupportedFlags(flags) => {
                write!(
                    formatter,
                    "unsupported operation-set binary flags 0x{flags:04x}"
                )
            }
            Self::PayloadTooLarge { observed, limit } => write!(
                formatter,
                "operation-set payload declares {observed} bytes (limit {limit})"
            ),
            Self::LengthMismatch { expected, observed } => write!(
                formatter,
                "operation-set envelope contains {observed} bytes (header declares {expected})"
            ),
            Self::Decode(error) => write!(formatter, "cannot decode operation set: {error}"),
            Self::TrailingPayload {
                consumed,
                payload_len,
            } => write!(
                formatter,
                "operation-set decoder consumed {consumed} of {payload_len} payload bytes"
            ),
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

impl std::error::Error for OperationSetBinaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<OperationSetValidationError> for OperationSetBinaryError {
    fn from(error: OperationSetValidationError) -> Self {
        Self::Validation(error)
    }
}

impl BinaryOperationSet {
    fn from_operation_set(operation_set: &OperationSet) -> Result<Self, OperationSetBinaryError> {
        let operations = operation_set
            .operations()
            .iter()
            .map(BinaryOperationType::from_operation)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(operations))
    }

    fn into_operation_set(self) -> Result<OperationSet, String> {
        let mut operation_set = OperationSet::with_capacity(self.0.len());
        for (index, operation) in self.0.into_iter().enumerate() {
            operation_set.add(operation.into_operation(index)?);
        }
        operation_set
            .validate()
            .map_err(|error| error.to_string())?;
        Ok(operation_set)
    }
}

impl BinaryOperationType {
    fn from_operation(operation: &OperationType) -> Result<Self, OperationSetBinaryError> {
        let consume_x = u64::try_from(operation.consume_x()).map_err(|_| {
            OperationSetBinaryError::Encode(
                "operation source consumption does not fit uint64".to_owned(),
            )
        })?;
        let consume_y = u64::try_from(operation.consume_y()).map_err(|_| {
            OperationSetBinaryError::Encode(
                "operation target consumption does not fit uint64".to_owned(),
            )
        })?;
        let applicability = match operation.applicability() {
            OperationApplicability::Any => BinaryApplicability::Any,
            OperationApplicability::Equal => BinaryApplicability::Equal,
            OperationApplicability::AdjacentTranspose => BinaryApplicability::AdjacentTranspose,
            OperationApplicability::Listed(restriction) => {
                BinaryApplicability::Listed(BinarySubstitutionSet(
                    restriction
                        .pairs()
                        .into_iter()
                        .map(BinarySubstitutionPair::from)
                        .collect(),
                ))
            }
        };
        Ok(Self {
            consume_x,
            consume_y,
            weight: operation.weight(),
            applicability,
            name: operation.name().to_owned(),
        })
    }

    fn into_operation(self, index: usize) -> Result<OperationType, String> {
        let consume_x = usize::try_from(self.consume_x)
            .map_err(|_| decode_field_error(index, "consume_x", "value does not fit usize"))?;
        let consume_y = usize::try_from(self.consume_y)
            .map_err(|_| decode_field_error(index, "consume_y", "value does not fit usize"))?;
        if self.name.is_empty() {
            return Err(decode_field_error(index, "name", "must be non-empty"));
        }
        if self.name.len() > MAX_OPERATION_NAME_BYTES {
            return Err(decode_field_error(
                index,
                "name",
                &format!(
                    "contains {} UTF-8 bytes (limit {MAX_OPERATION_NAME_BYTES})",
                    self.name.len()
                ),
            ));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(decode_field_error(
                index,
                "weight",
                "must be finite and non-negative",
            ));
        }
        if consume_x == 0 && consume_y == 0 {
            return Err(decode_field_error(
                index,
                "consumption",
                "must advance at least one input",
            ));
        }
        if self.weight == 0.0 && consume_x != consume_y {
            return Err(decode_field_error(
                index,
                "weight",
                "zero-weight operations must preserve length",
            ));
        }

        let applicability = match self.applicability {
            BinaryApplicability::Any => OperationApplicability::Any,
            BinaryApplicability::Equal => {
                if consume_x != consume_y {
                    return Err(decode_field_error(
                        index,
                        "applicability",
                        "equality requires equal source and target consumption",
                    ));
                }
                OperationApplicability::Equal
            }
            BinaryApplicability::AdjacentTranspose => {
                if consume_x != 2 || consume_y != 2 {
                    return Err(decode_field_error(
                        index,
                        "applicability",
                        "adjacent transposition requires consumption (2, 2)",
                    ));
                }
                OperationApplicability::AdjacentTranspose
            }
            BinaryApplicability::Listed(restriction) => OperationApplicability::Listed(
                restriction.into_substitution_set(index, consume_x, consume_y)?,
            ),
        };

        Ok(OperationType::with_owned_applicability(
            consume_x,
            consume_y,
            self.weight,
            applicability,
            self.name,
        ))
    }
}

impl From<SubstitutionPair> for BinarySubstitutionPair {
    fn from(pair: SubstitutionPair) -> Self {
        match pair {
            SubstitutionPair::Bytes { source, target } => Self::Bytes { source, target },
            SubstitutionPair::Strings { source, target } => Self::Strings { source, target },
        }
    }
}

impl BinarySubstitutionSet {
    fn into_substitution_set(
        self,
        operation_index: usize,
        consume_x: usize,
        consume_y: usize,
    ) -> Result<SubstitutionSet, String> {
        let mut restriction = SubstitutionSet::with_capacity(self.0.len());
        for (pair_index, pair) in self.0.into_iter().enumerate() {
            match pair {
                BinarySubstitutionPair::Bytes { source, target } => {
                    if consume_x != 1 || consume_y != 1 {
                        return Err(decode_restriction_arity_error(
                            operation_index,
                            pair_index,
                            1,
                            1,
                            consume_x,
                            consume_y,
                        ));
                    }
                    restriction.allow_byte(source, target);
                }
                BinarySubstitutionPair::Strings { source, target } => {
                    let source_units = source.chars().count();
                    let target_units = target.chars().count();
                    if source_units != consume_x || target_units != consume_y {
                        return Err(decode_restriction_arity_error(
                            operation_index,
                            pair_index,
                            source_units,
                            target_units,
                            consume_x,
                            consume_y,
                        ));
                    }
                    restriction.allow_str(&source, &target);
                }
            }
        }
        Ok(restriction)
    }
}

fn decode_field_error(index: usize, field: &str, reason: &str) -> String {
    format!("operation {index} field {field}: {reason}")
}

fn decode_restriction_arity_error(
    operation_index: usize,
    pair_index: usize,
    source_units: usize,
    target_units: usize,
    consume_x: usize,
    consume_y: usize,
) -> String {
    format!(
        "operation {operation_index} restriction pair {pair_index} consumes ({source_units}, {target_units}), not declared ({consume_x}, {consume_y})"
    )
}

impl OperationSet {
    /// Encode this operation set in the supported compact binary envelope.
    ///
    /// Operation order is preserved. Restriction entries are serialized in
    /// canonical order, so equal operation sets produce byte-identical output
    /// regardless of their hash-table iteration order.
    ///
    /// The public model deliberately has no generic Serde implementation. This
    /// makes unsupported text persistence a compile-time error:
    ///
    /// ```compile_fail
    /// use liblevenshtein::transducer::OperationSet;
    ///
    /// fn requires_generic_serde<T: serde::Serialize>() {}
    /// requires_generic_serde::<OperationSet>();
    /// ```
    pub fn to_binary(&self) -> Result<Vec<u8>, OperationSetBinaryError> {
        self.validate()?;
        let wire = BinaryOperationSet::from_operation_set(self)?;
        let payload = bincode::serde::encode_to_vec(wire, bincode::config::legacy())
            .map_err(|error| OperationSetBinaryError::Encode(error.to_string()))?;
        if payload.len() > MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES {
            return Err(OperationSetBinaryError::PayloadTooLarge {
                observed: payload.len() as u64,
                limit: MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES,
            });
        }
        let total_len = HEADER_BYTES.checked_add(payload.len()).ok_or(
            OperationSetBinaryError::PayloadTooLarge {
                observed: payload.len() as u64,
                limit: MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES,
            },
        )?;
        let mut encoded = Vec::with_capacity(total_len);
        encoded.extend_from_slice(&OPERATION_SET_BINARY_MAGIC);
        encoded.extend_from_slice(&OPERATION_SET_BINARY_VERSION.to_le_bytes());
        encoded.extend_from_slice(&SUPPORTED_FLAGS.to_le_bytes());
        encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&payload);
        Ok(encoded)
    }

    /// Decode and validate a compact binary operation-set envelope.
    ///
    /// Decoding rejects unknown versions or flags, oversized declarations,
    /// truncation, appended bytes, non-canonical semantic values, and every
    /// invariant enforced by [`OperationSet::validate`].
    pub fn from_binary(bytes: &[u8]) -> Result<Self, OperationSetBinaryError> {
        Self::from_binary_with_limits(bytes, OperationSetBinaryLimits::default())
    }

    /// Decode with a caller-selected resource policy.
    ///
    /// The fixed format ceiling remains authoritative even when a caller
    /// supplies a larger value. Smaller values provide an application-specific
    /// trust boundary without changing the wire format.
    pub fn from_binary_with_limits(
        bytes: &[u8],
        limits: OperationSetBinaryLimits,
    ) -> Result<Self, OperationSetBinaryError> {
        if bytes.len() < HEADER_BYTES {
            return Err(OperationSetBinaryError::TruncatedHeader {
                observed: bytes.len(),
                required: HEADER_BYTES,
            });
        }
        if bytes[..8] != OPERATION_SET_BINARY_MAGIC {
            return Err(OperationSetBinaryError::InvalidMagic);
        }

        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != OPERATION_SET_BINARY_VERSION {
            return Err(OperationSetBinaryError::UnsupportedVersion(version));
        }
        let flags = u16::from_le_bytes([bytes[10], bytes[11]]);
        if flags != SUPPORTED_FLAGS {
            return Err(OperationSetBinaryError::UnsupportedFlags(flags));
        }
        let declared = u64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .expect("fixed-length header slice has eight bytes"),
        );
        let payload_limit = limits
            .max_payload_bytes
            .min(MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES);
        if declared > payload_limit as u64 {
            return Err(OperationSetBinaryError::PayloadTooLarge {
                observed: declared,
                limit: payload_limit,
            });
        }
        let payload_len =
            usize::try_from(declared).map_err(|_| OperationSetBinaryError::PayloadTooLarge {
                observed: declared,
                limit: payload_limit,
            })?;
        let expected = HEADER_BYTES.checked_add(payload_len).ok_or(
            OperationSetBinaryError::PayloadTooLarge {
                observed: declared,
                limit: payload_limit,
            },
        )?;
        if bytes.len() != expected {
            return Err(OperationSetBinaryError::LengthMismatch {
                expected,
                observed: bytes.len(),
            });
        }

        let payload = &bytes[HEADER_BYTES..];
        let (wire, consumed): (BinaryOperationSet, usize) = bincode::serde::decode_from_slice(
            payload,
            bincode::config::legacy().with_limit::<MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES>(),
        )
        .map_err(|error| OperationSetBinaryError::Decode(error.to_string()))?;
        if consumed != payload_len {
            return Err(OperationSetBinaryError::TrailingPayload {
                consumed,
                payload_len,
            });
        }
        let operation_set = wire
            .into_operation_set()
            .map_err(OperationSetBinaryError::Decode)?;
        operation_set
            .validate_persistence_limits(limits)
            .map_err(|violation| OperationSetBinaryError::ResourceLimit {
                resource: violation.resource,
                observed: violation.observed,
                limit: violation.limit,
            })?;
        Ok(operation_set)
    }

    pub(super) fn validate_persistence_limits(
        &self,
        limits: OperationSetBinaryLimits,
    ) -> Result<(), OperationSetLimitViolation> {
        if self.len() > limits.max_operations {
            return Err(OperationSetLimitViolation {
                resource: "operation count",
                observed: self.len(),
                limit: limits.max_operations,
            });
        }
        let mut total_pairs = 0usize;
        let mut text_bytes = 0usize;
        for operation in self.operations() {
            if operation.name().len() > limits.max_operation_name_bytes {
                return Err(OperationSetLimitViolation {
                    resource: "operation-name bytes",
                    observed: operation.name().len(),
                    limit: limits.max_operation_name_bytes,
                });
            }
            let OperationApplicability::Listed(restriction) = operation.applicability() else {
                continue;
            };
            let pairs = restriction.pairs();
            if pairs.len() > limits.max_restriction_pairs_per_operation {
                return Err(OperationSetLimitViolation {
                    resource: "restriction pairs per operation",
                    observed: pairs.len(),
                    limit: limits.max_restriction_pairs_per_operation,
                });
            }
            total_pairs =
                total_pairs
                    .checked_add(pairs.len())
                    .ok_or(OperationSetLimitViolation {
                        resource: "total restriction pairs",
                        observed: usize::MAX,
                        limit: limits.max_total_restriction_pairs,
                    })?;
            if total_pairs > limits.max_total_restriction_pairs {
                return Err(OperationSetLimitViolation {
                    resource: "total restriction pairs",
                    observed: total_pairs,
                    limit: limits.max_total_restriction_pairs,
                });
            }
            for pair in pairs {
                if let SubstitutionPair::Strings { source, target } = pair {
                    text_bytes = text_bytes
                        .checked_add(source.len())
                        .and_then(|value| value.checked_add(target.len()))
                        .ok_or(OperationSetLimitViolation {
                            resource: "restriction text bytes",
                            observed: usize::MAX,
                            limit: limits.max_restriction_text_bytes,
                        })?;
                    if text_bytes > limits.max_restriction_text_bytes {
                        return Err(OperationSetLimitViolation {
                            resource: "restriction text bytes",
                            observed: text_bytes,
                            limit: limits.max_restriction_text_bytes,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
