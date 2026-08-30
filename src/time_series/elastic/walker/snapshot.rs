//! Portable, checksummed snapshots for exact elastic indexes.
//!
//! A complete snapshot binds the quantized dictionary language to every
//! full-precision collision member and to all configuration that gives those
//! bytes meaning. Loading is fail-closed: the checksum, schema, package,
//! kernel, quantizer, fold identity, scaling, weighting, and every reconstructed
//! key must agree before an index is returned.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use super::ElasticTransducer;
use crate::time_series::elastic::ElasticKernel;
use crate::time_series::encoding::QuantizationConfig;
use crate::time_series::kernels::{
    DtwConfig, ErpConfig, FrechetConfig, MetricTwedConfig, TwedConfig,
};
use crate::time_series::msm::{MetricMsmConfig, MsmConfig};
use crate::time_series::msm_kernel::{MetricMsmKernel, MsmKernel};

const MAGIC: &[u8; 8] = b"LLEVESNP";
const SCHEMA_VERSION: u32 = 1;
const CHECKSUM_LEN: usize = 32;

/// Kernel configuration that can be bound into a stable snapshot schema.
///
/// Implementations must use a globally unique tag and encode every value that
/// can affect transition, pruning, or exact-scoring semantics. Floating-point
/// configuration uses exact IEEE-754 bits; approximate equality is forbidden.
pub trait ElasticSnapshotKernel: ElasticKernel + Sized {
    /// Stable kernel identifier stored in the snapshot.
    const SNAPSHOT_TAG: &'static str;

    /// Exact, deterministic configuration words.
    fn snapshot_config_words(&self) -> Vec<u64>;

    /// Reconstruct and validate this kernel from exact configuration words.
    fn from_snapshot_config_words(words: &[u64]) -> Option<Self>;
}

impl ElasticSnapshotKernel for ErpConfig {
    const SNAPSHOT_TAG: &'static str = "erp-l1-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        vec![self.gap_value().to_bits()]
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [gap] = words else { return None };
        let gap = f64::from_bits(*gap);
        gap.is_finite().then(|| Self::new(gap))
    }
}

impl ElasticSnapshotKernel for MsmKernel {
    const SNAPSHOT_TAG: &'static str = "msm-raw-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        vec![self.config().split_merge_cost().to_bits()]
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [cost] = words else { return None };
        let cost = f64::from_bits(*cost);
        MsmConfig::try_new(cost).ok().map(Self::new)
    }
}

impl ElasticSnapshotKernel for MetricMsmKernel {
    const SNAPSHOT_TAG: &'static str = "msm-metric-nonempty-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        vec![self.config().split_merge_cost().to_bits()]
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [cost] = words else { return None };
        let cost = f64::from_bits(*cost);
        MetricMsmConfig::try_new(cost).ok().map(Self::new)
    }
}

impl ElasticSnapshotKernel for TwedConfig {
    const SNAPSHOT_TAG: &'static str = "twed-unit-grid-raw-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        vec![self.stiffness().to_bits(), self.gap_penalty().to_bits()]
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [stiffness, gap] = words else {
            return None;
        };
        let stiffness = f64::from_bits(*stiffness);
        let gap = f64::from_bits(*gap);
        (stiffness.is_finite() && stiffness >= 0.0 && gap.is_finite() && gap >= 0.0)
            .then(|| Self::new(stiffness, gap))
    }
}

impl ElasticSnapshotKernel for MetricTwedConfig {
    const SNAPSHOT_TAG: &'static str = "twed-unit-grid-metric-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        vec![self.stiffness().to_bits(), self.gap_penalty().to_bits()]
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [stiffness, gap] = words else {
            return None;
        };
        Self::try_new(f64::from_bits(*stiffness), f64::from_bits(*gap)).ok()
    }
}

impl ElasticSnapshotKernel for FrechetConfig {
    const SNAPSHOT_TAG: &'static str = "frechet-discrete-scalar-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        Vec::new()
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        words.is_empty().then(Self::new)
    }
}

impl ElasticSnapshotKernel for DtwConfig {
    const SNAPSHOT_TAG: &'static str = "dtw-banded-diagnostic-v1";

    fn snapshot_config_words(&self) -> Vec<u64> {
        vec![u64::try_from(self.band).unwrap_or(u64::MAX)]
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [band] = words else { return None };
        usize::try_from(*band).ok().map(Self::new)
    }
}

/// Experiment identity and fold-local transformations bound to a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct ElasticSnapshotMetadata {
    /// Stable training-fold identity; held-out data must never affect it.
    pub training_fold_id: String,
    /// Exact build provenance, normally a commit plus lockfile/toolchain digest.
    pub build_provenance: String,
    /// Finite, strictly positive fold-local channel scales in fixed order.
    pub channel_scales: Vec<f64>,
    /// Finite, strictly positive fixed channel weights in the same order.
    pub channel_weights: Vec<f64>,
}

impl ElasticSnapshotMetadata {
    /// Validate metadata whose channel ordering is fixed by the caller's schema.
    pub fn try_new(
        training_fold_id: impl Into<String>,
        build_provenance: impl Into<String>,
        channel_scales: Vec<f64>,
        channel_weights: Vec<f64>,
    ) -> Result<Self, ElasticSnapshotError> {
        let metadata = Self {
            training_fold_id: training_fold_id.into(),
            build_provenance: build_provenance.into(),
            channel_scales,
            channel_weights,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<(), ElasticSnapshotError> {
        if self.training_fold_id.is_empty()
            || self.build_provenance.is_empty()
            || self.channel_scales.len() != self.channel_weights.len()
            || self.channel_scales.is_empty()
            || self
                .channel_scales
                .iter()
                .chain(&self.channel_weights)
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ElasticSnapshotError::InvalidMetadata);
        }
        Ok(())
    }
}

/// SHA-256 identity of the complete canonical snapshot payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElasticSnapshotIdentity(pub [u8; CHECKSUM_LEN]);

impl fmt::Display for ElasticSnapshotIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A loaded exact index together with the identity and metadata it verified.
#[derive(Debug)]
pub struct ElasticSnapshot<K: ElasticKernel> {
    /// Reconstructed exact index, including all collision originals.
    pub index: ElasticTransducer<K, u64>,
    /// Content identity covering every byte before the checksum footer.
    pub identity: ElasticSnapshotIdentity,
    /// Verified experiment metadata.
    pub metadata: ElasticSnapshotMetadata,
}

/// Fail-closed snapshot write/load error.
#[derive(Debug)]
pub enum ElasticSnapshotError {
    /// File-system operation failed.
    Io(io::Error),
    /// Checked size arithmetic overflowed or the configured byte ceiling was crossed.
    ResourceLimit {
        /// Configured maximum complete snapshot size.
        limit: u64,
        /// Observed or required byte count.
        requested: u64,
    },
    /// Bytes do not form the canonical schema.
    InvalidFormat,
    /// Metadata is empty, mismatched, nonfinite, nonpositive, or dimensionally inconsistent.
    InvalidMetadata,
    /// The payload SHA-256 does not match its footer.
    ChecksumMismatch,
    /// Package, kernel, quantizer, or expected experiment configuration differs.
    ConfigurationMismatch,
    /// A stored quantized key does not match its full-precision original.
    OriginalKeyMismatch {
        /// Stable identifier whose original does not reproduce its stored key.
        stable_id: u64,
    },
    /// Stable identifiers are duplicated or not strictly sorted.
    InvalidStableIdOrder,
}

impl fmt::Display for ElasticSnapshotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "snapshot I/O failed: {error}"),
            Self::ResourceLimit { limit, requested } => write!(
                formatter,
                "snapshot requires {requested} bytes, exceeding limit {limit}"
            ),
            Self::InvalidFormat => formatter.write_str("invalid elastic snapshot format"),
            Self::InvalidMetadata => formatter.write_str("invalid elastic snapshot metadata"),
            Self::ChecksumMismatch => formatter.write_str("elastic snapshot checksum mismatch"),
            Self::ConfigurationMismatch => {
                formatter.write_str("elastic snapshot configuration mismatch")
            }
            Self::OriginalKeyMismatch { stable_id } => write!(
                formatter,
                "snapshot key does not match original series for stable id {stable_id}"
            ),
            Self::InvalidStableIdOrder => {
                formatter.write_str("snapshot stable ids are not unique and strictly increasing")
            }
        }
    }
}

impl std::error::Error for ElasticSnapshotError {}

impl From<io::Error> for ElasticSnapshotError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

struct Encoder {
    bytes: Vec<u8>,
    limit: u64,
}

impl Encoder {
    fn new(limit: u64) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }

    fn append(&mut self, bytes: &[u8]) -> Result<(), ElasticSnapshotError> {
        let requested = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested: u64::MAX,
            })?;
        if requested > self.limit {
            return Err(ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested,
            });
        }
        self.bytes.try_reserve_exact(bytes.len()).map_err(|_| {
            ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested,
            }
        })?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), ElasticSnapshotError> {
        self.append(&[value])
    }

    fn u32(&mut self, value: u32) -> Result<(), ElasticSnapshotError> {
        self.append(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), ElasticSnapshotError> {
        self.append(&value.to_le_bytes())
    }

    fn f64(&mut self, value: f64) -> Result<(), ElasticSnapshotError> {
        self.u64(value.to_bits())
    }

    fn len(&mut self, value: usize) -> Result<(), ElasticSnapshotError> {
        self.u64(
            u64::try_from(value).map_err(|_| ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested: u64::MAX,
            })?,
        )
    }

    fn string(&mut self, value: &str) -> Result<(), ElasticSnapshotError> {
        self.len(value.len())?;
        self.append(value.as_bytes())
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ElasticSnapshotError> {
        let end = self
            .offset
            .checked_add(len)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(ElasticSnapshotError::InvalidFormat)?;
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, ElasticSnapshotError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ElasticSnapshotError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ElasticSnapshotError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, ElasticSnapshotError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn len(&mut self) -> Result<usize, ElasticSnapshotError> {
        usize::try_from(self.u64()?).map_err(|_| ElasticSnapshotError::InvalidFormat)
    }

    fn string(&mut self) -> Result<String, ElasticSnapshotError> {
        let len = self.len()?;
        let source = std::str::from_utf8(self.take(len)?)
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        let mut value = String::new();
        value
            .try_reserve_exact(source.len())
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        value.push_str(source);
        Ok(value)
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn quantizers_match(left: &QuantizationConfig, right: &QuantizationConfig) -> bool {
    left.min_value.to_bits() == right.min_value.to_bits()
        && left.max_value.to_bits() == right.max_value.to_bits()
        && left.num_bins == right.num_bins
        && left.clamp_outliers == right.clamp_outliers
}

fn append_metadata(
    encoder: &mut Encoder,
    metadata: &ElasticSnapshotMetadata,
) -> Result<(), ElasticSnapshotError> {
    encoder.string(&metadata.training_fold_id)?;
    encoder.string(&metadata.build_provenance)?;
    encoder.len(metadata.channel_scales.len())?;
    for scale in &metadata.channel_scales {
        encoder.f64(*scale)?;
    }
    encoder.len(metadata.channel_weights.len())?;
    for weight in &metadata.channel_weights {
        encoder.f64(*weight)?;
    }
    Ok(())
}

fn read_f64_vector(decoder: &mut Decoder<'_>) -> Result<Vec<f64>, ElasticSnapshotError> {
    let len = decoder.len()?;
    let required = len
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    if required > decoder.bytes.len().saturating_sub(decoder.offset) {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
    for _ in 0..len {
        values.push(decoder.f64()?);
    }
    Ok(values)
}

fn read_metadata(
    decoder: &mut Decoder<'_>,
) -> Result<ElasticSnapshotMetadata, ElasticSnapshotError> {
    ElasticSnapshotMetadata::try_new(
        decoder.string()?,
        decoder.string()?,
        read_f64_vector(decoder)?,
        read_f64_vector(decoder)?,
    )
}

fn append_quantizer(
    encoder: &mut Encoder,
    quantizer: &QuantizationConfig,
) -> Result<(), ElasticSnapshotError> {
    encoder.f64(quantizer.min_value)?;
    encoder.f64(quantizer.max_value)?;
    encoder.u32(quantizer.num_bins)?;
    encoder.u8(u8::from(quantizer.clamp_outliers))
}

fn read_quantizer(decoder: &mut Decoder<'_>) -> Result<QuantizationConfig, ElasticSnapshotError> {
    let minimum = decoder.f64()?;
    let maximum = decoder.f64()?;
    let bins = decoder.u32()?;
    let clamp = match decoder.u8()? {
        0 => false,
        1 => true,
        _ => return Err(ElasticSnapshotError::InvalidFormat),
    };
    let mut quantizer = QuantizationConfig::try_uniform(minimum, maximum, bins)
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    if bins > 256 {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    quantizer.clamp_outliers = clamp;
    Ok(quantizer)
}

fn append_kernel<K: ElasticSnapshotKernel>(
    encoder: &mut Encoder,
    kernel: &K,
) -> Result<(), ElasticSnapshotError> {
    encoder.string(K::SNAPSHOT_TAG)?;
    let words = kernel.snapshot_config_words();
    encoder.len(words.len())?;
    for word in words {
        encoder.u64(word)?;
    }
    Ok(())
}

fn read_kernel<K: ElasticSnapshotKernel>(
    decoder: &mut Decoder<'_>,
) -> Result<K, ElasticSnapshotError> {
    if decoder.string()? != K::SNAPSHOT_TAG {
        return Err(ElasticSnapshotError::ConfigurationMismatch);
    }
    let count = decoder.len()?;
    let required = count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    if required > decoder.bytes.len().saturating_sub(decoder.offset) {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    let mut words = Vec::new();
    words
        .try_reserve_exact(count)
        .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
    for _ in 0..count {
        words.push(decoder.u64()?);
    }
    K::from_snapshot_config_words(&words).ok_or(ElasticSnapshotError::InvalidFormat)
}

impl<K> ElasticTransducer<K, u64>
where
    K: ElasticSnapshotKernel,
{
    /// Atomically write a complete, canonical exact-index snapshot.
    ///
    /// `max_snapshot_bytes` is a hard ceiling covering the payload and checksum.
    /// The temporary file is created beside `path`, synchronized, renamed, and
    /// followed by a parent-directory sync on Unix. Stable IDs are serialized
    /// in ascending order, so insertion order and hash seeding do not affect
    /// the identity.
    pub fn write_complete_snapshot(
        &self,
        path: impl AsRef<Path>,
        metadata: &ElasticSnapshotMetadata,
        max_snapshot_bytes: u64,
    ) -> Result<ElasticSnapshotIdentity, ElasticSnapshotError> {
        metadata.validate()?;
        let mut encoder = Encoder::new(max_snapshot_bytes);
        encoder.append(MAGIC)?;
        encoder.u32(SCHEMA_VERSION)?;
        encoder.string(env!("CARGO_PKG_VERSION"))?;
        append_kernel(&mut encoder, &self.kernel)?;
        append_quantizer(&mut encoder, &self.quant)?;
        append_metadata(&mut encoder, metadata)?;

        let mut stable_ids = Vec::new();
        stable_ids
            .try_reserve_exact(self.originals.len())
            .map_err(|_| ElasticSnapshotError::ResourceLimit {
                limit: max_snapshot_bytes,
                requested: max_snapshot_bytes.saturating_add(1),
            })?;
        stable_ids.extend(self.originals.keys().copied());
        stable_ids.sort_unstable();
        encoder.len(stable_ids.len())?;

        for stable_id in stable_ids {
            let stored = self
                .originals
                .get(&stable_id)
                .ok_or(ElasticSnapshotError::InvalidFormat)?;
            if stored.series.iter().any(|value| !value.is_finite()) {
                return Err(ElasticSnapshotError::InvalidFormat);
            }
            encoder.u64(stable_id)?;
            encoder.len(stored.series.len())?;
            for value in &stored.series {
                encoder.u8(self.quant.quantize_u8(*value))?;
            }
            encoder.len(stored.series.len())?;
            for value in &stored.series {
                encoder.f64(*value)?;
            }
        }

        let identity = ElasticSnapshotIdentity(sha256(&encoder.bytes)?);
        encoder.append(&identity.0)?;
        atomic_write(path.as_ref(), &encoder.bytes)?;
        Ok(identity)
    }

    /// Load a complete snapshot only if every expected binding agrees exactly.
    ///
    /// No index escapes on corruption or mismatch. Every stored byte key is
    /// replayed against its full-precision original before insertion, and the
    /// decoder rejects trailing bytes, duplicate IDs, and noncanonical order.
    pub fn load_complete_snapshot(
        path: impl AsRef<Path>,
        expected_quantizer: &QuantizationConfig,
        expected_kernel: &K,
        expected_metadata: &ElasticSnapshotMetadata,
        max_snapshot_bytes: u64,
    ) -> Result<ElasticSnapshot<K>, ElasticSnapshotError> {
        expected_metadata.validate()?;
        let bytes = read_bounded(path.as_ref(), max_snapshot_bytes)?;
        let payload_len = bytes
            .len()
            .checked_sub(CHECKSUM_LEN)
            .ok_or(ElasticSnapshotError::InvalidFormat)?;
        let (payload, checksum) = bytes.split_at(payload_len);
        let actual = sha256(payload)?;
        if checksum != actual {
            return Err(ElasticSnapshotError::ChecksumMismatch);
        }
        let identity = ElasticSnapshotIdentity(actual);
        let mut decoder = Decoder::new(payload);
        if decoder.take(MAGIC.len())? != MAGIC
            || decoder.u32()? != SCHEMA_VERSION
            || decoder.string()? != env!("CARGO_PKG_VERSION")
        {
            return Err(ElasticSnapshotError::ConfigurationMismatch);
        }

        let kernel = read_kernel::<K>(&mut decoder)?;
        if kernel.snapshot_config_words() != expected_kernel.snapshot_config_words() {
            return Err(ElasticSnapshotError::ConfigurationMismatch);
        }
        let quantizer = read_quantizer(&mut decoder)?;
        if !quantizers_match(&quantizer, expected_quantizer) {
            return Err(ElasticSnapshotError::ConfigurationMismatch);
        }
        let metadata = read_metadata(&mut decoder)?;
        if &metadata != expected_metadata {
            return Err(ElasticSnapshotError::ConfigurationMismatch);
        }

        let entry_count = decoder.len()?;
        let minimum_entry_bytes = std::mem::size_of::<u64>() * 3;
        if entry_count > decoder.bytes.len().saturating_sub(decoder.offset) / minimum_entry_bytes {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        let mut index = Self::new(quantizer, kernel);
        index.originals.try_reserve(entry_count).map_err(|_| {
            ElasticSnapshotError::ResourceLimit {
                limit: max_snapshot_bytes,
                requested: max_snapshot_bytes.saturating_add(1),
            }
        })?;
        index.buckets.try_reserve(entry_count).map_err(|_| {
            ElasticSnapshotError::ResourceLimit {
                limit: max_snapshot_bytes,
                requested: max_snapshot_bytes.saturating_add(1),
            }
        })?;

        let mut previous_id = None;
        for _ in 0..entry_count {
            let stable_id = decoder.u64()?;
            if previous_id.is_some_and(|previous| stable_id <= previous) {
                return Err(ElasticSnapshotError::InvalidStableIdOrder);
            }
            previous_id = Some(stable_id);

            let key_len = decoder.len()?;
            let key = decoder.take(key_len)?;
            let series_len = decoder.len()?;
            if key_len != series_len {
                return Err(ElasticSnapshotError::OriginalKeyMismatch { stable_id });
            }
            let required = series_len
                .checked_mul(std::mem::size_of::<f64>())
                .ok_or(ElasticSnapshotError::InvalidFormat)?;
            if required > decoder.bytes.len().saturating_sub(decoder.offset) {
                return Err(ElasticSnapshotError::InvalidFormat);
            }
            let mut series = Vec::new();
            series.try_reserve_exact(series_len).map_err(|_| {
                ElasticSnapshotError::ResourceLimit {
                    limit: max_snapshot_bytes,
                    requested: max_snapshot_bytes.saturating_add(1),
                }
            })?;
            for expected_bin in key {
                let value = decoder.f64()?;
                if !value.is_finite() || index.quant.quantize_u8(value) != *expected_bin {
                    return Err(ElasticSnapshotError::OriginalKeyMismatch { stable_id });
                }
                series.push(value);
            }
            if !index.insert(stable_id, &series) {
                return Err(ElasticSnapshotError::InvalidStableIdOrder);
            }
        }
        if !decoder.finished() {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        Ok(ElasticSnapshot {
            index,
            identity,
            metadata,
        })
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, ElasticSnapshotError> {
    let mut file = File::open(path)?;
    let requested = file.metadata()?.len();
    if requested > limit {
        return Err(ElasticSnapshotError::ResourceLimit { limit, requested });
    }
    let len = usize::try_from(requested)
        .map_err(|_| ElasticSnapshotError::ResourceLimit { limit, requested })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| ElasticSnapshotError::ResourceLimit { limit, requested })?;
    bytes.resize(len, 0);
    file.read_exact(&mut bytes)?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(ElasticSnapshotError::ResourceLimit {
            limit,
            requested: limit.saturating_add(1),
        });
    }
    Ok(bytes)
}

fn partial_path(path: &Path, nonce: u8) -> Result<PathBuf, ElasticSnapshotError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or(ElasticSnapshotError::InvalidFormat)?
        .to_string_lossy();
    Ok(parent.join(format!(".{name}.partial-{}-{nonce}", std::process::id())))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ElasticSnapshotError> {
    let mut temporary = None;
    let mut file = None;
    for nonce in 0..=u8::MAX {
        let candidate = partial_path(path, nonce)?;
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(created) => {
                temporary = Some(candidate);
                file = Some(created);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    let temporary = temporary.ok_or_else(|| {
        ElasticSnapshotError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "all snapshot partial-file names are occupied",
        ))
    })?;
    let write_result = (|| -> Result<(), ElasticSnapshotError> {
        let mut file = file.ok_or(ElasticSnapshotError::InvalidFormat)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        File::open(path.parent().unwrap_or_else(|| Path::new(".")))?.sync_all()?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

fn sha256(message: &[u8]) -> Result<[u8; CHECKSUM_LEN], ElasticSnapshotError> {
    const INITIAL: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let bit_len = u64::try_from(message.len())
        .ok()
        .and_then(|length| length.checked_mul(8))
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    let mut state = INITIAL;
    let mut chunks = message.chunks_exact(64);
    for chunk in &mut chunks {
        let block: &[u8; 64] = chunk
            .try_into()
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        sha256_compress(&mut state, block);
    }

    let remainder = chunks.remainder();
    let mut final_blocks = [[0_u8; 64]; 2];
    final_blocks[0][..remainder.len()].copy_from_slice(remainder);
    final_blocks[0][remainder.len()] = 0x80;
    let block_count = if remainder.len() <= 55 { 1 } else { 2 };
    final_blocks[block_count - 1][56..].copy_from_slice(&bit_len.to_be_bytes());
    for block in &final_blocks[..block_count] {
        sha256_compress(&mut state, block);
    }

    let mut digest = [0_u8; CHECKSUM_LEN];
    for (output, word) in digest.chunks_exact_mut(4).zip(state) {
        output.copy_from_slice(&word.to_be_bytes());
    }
    Ok(digest)
}

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    const ROUND: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut schedule = [0_u32; 64];
    for (index, word) in block.chunks_exact(4).enumerate() {
        let word: [u8; 4] = word.try_into().unwrap_or([0; 4]);
        schedule[index] = u32::from_be_bytes(word);
    }
    for index in 16..64 {
        let small0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let small1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(small0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(small1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ ((!e) & g);
        let temporary1 = h
            .wrapping_add(big1)
            .wrapping_add(choose)
            .wrapping_add(ROUND[index])
            .wrapping_add(schedule[index]);
        let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temporary2 = big0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temporary1);
        d = c;
        c = b;
        b = a;
        a = temporary1.wrapping_add(temporary2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

#[cfg(test)]
mod tests {
    use super::sha256;

    fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn sha256_matches_fips_vectors() {
        assert_eq!(
            hex(sha256(b"").expect("empty FIPS vector must hash")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(sha256(b"abc").expect("abc FIPS vector must hash")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
