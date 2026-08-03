//! Optional gzip transport wrapper for operation-set binary persistence.
//!
//! Gzip does not define a third semantic format: decompression must yield one
//! complete bincode envelope or protobuf message, which is then validated by
//! the corresponding decoder. Both compressed and decompressed byte counts are
//! bounded before the semantic object is admitted.

use super::operation_set_binary::{HEADER_BYTES, MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES};
#[cfg(feature = "protobuf")]
use super::OperationSetProtobufError;
use super::{OperationSet, OperationSetBinaryError, OperationSetBinaryLimits};
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fmt;
use std::io::{BufReader, Cursor, Read, Write};

/// Maximum accepted gzip input size (64 MiB).
pub const MAX_OPERATION_SET_GZIP_INPUT_BYTES: usize = 64 * 1024 * 1024;

/// Failure while applying gzip to an operation-set binary representation.
#[derive(Debug)]
pub enum OperationSetGzipError {
    /// The supplied or produced gzip stream exceeds its fixed input ceiling.
    CompressedPayloadTooLarge {
        /// Observed bytes.
        observed: usize,
        /// Accepted bytes.
        limit: usize,
    },
    /// Inflation produced more bytes than the selected inner-format policy.
    DecompressedPayloadTooLarge {
        /// Observed lower bound (the decoder stops at limit plus one).
        observed: usize,
        /// Accepted bytes.
        limit: usize,
    },
    /// The gzip stream is malformed or failed its checksum.
    Gzip(String),
    /// Bytes follow the single complete gzip member.
    TrailingCompressedData {
        /// Bytes consumed by the gzip member.
        consumed: usize,
        /// Bytes supplied by the caller.
        observed: usize,
    },
    /// The decompressed bincode envelope is invalid.
    Binary(OperationSetBinaryError),
    /// The decompressed protobuf message is invalid.
    #[cfg(feature = "protobuf")]
    Protobuf(OperationSetProtobufError),
}

impl fmt::Display for OperationSetGzipError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompressedPayloadTooLarge { observed, limit } => write!(
                formatter,
                "operation-set gzip contains {observed} bytes (limit {limit})"
            ),
            Self::DecompressedPayloadTooLarge { observed, limit } => write!(
                formatter,
                "operation-set gzip expands to more than {limit} bytes (observed at least {observed})"
            ),
            Self::Gzip(error) => write!(formatter, "invalid operation-set gzip stream: {error}"),
            Self::TrailingCompressedData { consumed, observed } => write!(
                formatter,
                "operation-set gzip member consumes {consumed} of {observed} supplied bytes"
            ),
            Self::Binary(error) => write!(formatter, "invalid compressed bincode data: {error}"),
            #[cfg(feature = "protobuf")]
            Self::Protobuf(error) => write!(formatter, "invalid compressed protobuf data: {error}"),
        }
    }
}

impl std::error::Error for OperationSetGzipError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Binary(error) => Some(error),
            #[cfg(feature = "protobuf")]
            Self::Protobuf(error) => Some(error),
            _ => None,
        }
    }
}

impl From<OperationSetBinaryError> for OperationSetGzipError {
    fn from(error: OperationSetBinaryError) -> Self {
        Self::Binary(error)
    }
}

#[cfg(feature = "protobuf")]
impl From<OperationSetProtobufError> for OperationSetGzipError {
    fn from(error: OperationSetProtobufError) -> Self {
        Self::Protobuf(error)
    }
}

impl OperationSet {
    /// Encode the versioned bincode envelope, then wrap it in one gzip member.
    pub fn to_binary_gzip(&self) -> Result<Vec<u8>, OperationSetGzipError> {
        compress_gzip(&self.to_binary()?)
    }

    /// Inflate one gzip member and decode its complete bincode envelope.
    pub fn from_binary_gzip(bytes: &[u8]) -> Result<Self, OperationSetGzipError> {
        Self::from_binary_gzip_with_limits(bytes, OperationSetBinaryLimits::default())
    }

    /// Decode a gzip-wrapped bincode envelope with caller-selected inner limits.
    pub fn from_binary_gzip_with_limits(
        bytes: &[u8],
        limits: OperationSetBinaryLimits,
    ) -> Result<Self, OperationSetGzipError> {
        let payload_limit = limits
            .max_payload_bytes
            .min(MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES);
        let output_limit = HEADER_BYTES
            .checked_add(payload_limit)
            .expect("fixed payload ceiling plus header fits usize");
        let decompressed = decompress_gzip(bytes, output_limit)?;
        Ok(Self::from_binary_with_limits(&decompressed, limits)?)
    }

    /// Encode protobuf, then wrap the message in one gzip member.
    #[cfg(feature = "protobuf")]
    pub fn to_protobuf_gzip(&self) -> Result<Vec<u8>, OperationSetGzipError> {
        compress_gzip(&self.to_protobuf()?)
    }

    /// Inflate one gzip member and decode its complete protobuf message.
    #[cfg(feature = "protobuf")]
    pub fn from_protobuf_gzip(bytes: &[u8]) -> Result<Self, OperationSetGzipError> {
        Self::from_protobuf_gzip_with_limits(bytes, OperationSetBinaryLimits::default())
    }

    /// Decode gzip-wrapped protobuf with caller-selected inner limits.
    #[cfg(feature = "protobuf")]
    pub fn from_protobuf_gzip_with_limits(
        bytes: &[u8],
        limits: OperationSetBinaryLimits,
    ) -> Result<Self, OperationSetGzipError> {
        let output_limit = limits
            .max_payload_bytes
            .min(MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES);
        let decompressed = decompress_gzip(bytes, output_limit)?;
        Ok(Self::from_protobuf_with_limits(&decompressed, limits)?)
    }
}

fn compress_gzip(bytes: &[u8]) -> Result<Vec<u8>, OperationSetGzipError> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(bytes)
        .map_err(|error| OperationSetGzipError::Gzip(error.to_string()))?;
    let compressed = encoder
        .finish()
        .map_err(|error| OperationSetGzipError::Gzip(error.to_string()))?;
    if compressed.len() > MAX_OPERATION_SET_GZIP_INPUT_BYTES {
        return Err(OperationSetGzipError::CompressedPayloadTooLarge {
            observed: compressed.len(),
            limit: MAX_OPERATION_SET_GZIP_INPUT_BYTES,
        });
    }
    Ok(compressed)
}

fn decompress_gzip(bytes: &[u8], output_limit: usize) -> Result<Vec<u8>, OperationSetGzipError> {
    if bytes.len() > MAX_OPERATION_SET_GZIP_INPUT_BYTES {
        return Err(OperationSetGzipError::CompressedPayloadTooLarge {
            observed: bytes.len(),
            limit: MAX_OPERATION_SET_GZIP_INPUT_BYTES,
        });
    }

    let reader = BufReader::new(Cursor::new(bytes));
    let decoder = GzDecoder::new(reader);
    let read_limit = u64::try_from(output_limit)
        .expect("fixed operation-set payload ceiling fits u64")
        .saturating_add(1);
    let mut limited = decoder.take(read_limit);
    let mut decompressed = Vec::with_capacity(output_limit.min(64 * 1024));
    limited
        .read_to_end(&mut decompressed)
        .map_err(|error| OperationSetGzipError::Gzip(error.to_string()))?;
    let decoder = limited.into_inner();

    if decompressed.len() > output_limit {
        return Err(OperationSetGzipError::DecompressedPayloadTooLarge {
            observed: decompressed.len(),
            limit: output_limit,
        });
    }

    let reader = decoder.into_inner();
    let physical_position = usize::try_from(reader.get_ref().position())
        .expect("slice-backed cursor position fits usize");
    let consumed = physical_position
        .checked_sub(reader.buffer().len())
        .expect("buffered bytes cannot exceed cursor position");
    if consumed != bytes.len() {
        return Err(OperationSetGzipError::TrailingCompressedData {
            consumed,
            observed: bytes.len(),
        });
    }
    Ok(decompressed)
}
