//! Portable, checksummed snapshots for exact elastic indexes.
//!
//! A complete snapshot binds the quantized dictionary language to every
//! full-precision collision member and to all configuration that gives those
//! bytes meaning. Loading is fail-closed: the checksum, schema, package,
//! kernel, quantizer, fold identity, scaling, weighting, and every reconstructed
//! key must agree before an index is returned.

use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::{ElasticDictionaryBackend, ElasticSnapshotIdentity, ElasticTransducer};
use crate::time_series::elastic::ElasticKernel;
use crate::time_series::encoding::QuantizationConfig;
use crate::time_series::kernels::{
    DtwConfig, ErpConfig, FrechetConfig, MetricTwedConfig, TwedConfig,
};
use crate::time_series::msm::{MetricMsmConfig, MsmConfig};
use crate::time_series::msm_kernel::{MetricMsmKernel, MsmKernel};

const MAGIC: &[u8; 8] = b"LLEVESNP";
const SCHEMA_VERSION: u32 = 3;
const CHECKSUM_LEN: usize = 32;
const MAX_KERNEL_CONFIG_WORDS: usize = 8;
const MAX_METADATA_STRING_BYTES: usize = 64 * 1024;
const MAX_CHANNELS: usize = 65_536;
const MAX_SCHEMA_STRING_BYTES: usize = 256;
const HASH_BUFFER_BYTES: usize = 16 * 1024;
const IMPLEMENTATION_VERSION: &str = "elastic-exact-snapshot-v3";
const DICTIONARY_SCHEMA: &str = "libdictenstein-persistent-artrie-byte-v1";
const BUNDLE_MAGIC: &[u8; 8] = b"LLEVEBND";
const BUNDLE_SCHEMA_VERSION: u32 = 1;
const GENERATION_MANIFEST: &str = "manifest.snapshot";
const GENERATION_DICTIONARY: &str = "dictionary.part";
const GENERATION_SEAL: &str = "bundle.seal";
const BUNDLE_SEAL_PAYLOAD_LEN: usize = 8 + 4 + 32 + 8 + 32 + 8 + 32;
const BUNDLE_SEAL_LEN: usize = BUNDLE_SEAL_PAYLOAD_LEN + CHECKSUM_LEN;

use libdictenstein::persistent_artrie::{PersistentARTrie, PersistentARTrieNode, BLOCK_SIZE};
use libdictenstein::Dictionary;

const DEFAULT_BACKEND_MEMORY_BYTES: usize = 8 * 1024 * 1024;

/// Independent resource ceilings for complete snapshot publication and loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElasticSnapshotLimits {
    /// Maximum canonical manifest bytes, including its checksum footer.
    pub max_manifest_bytes: u64,
    /// Maximum bytes across the manifest, dictionary, WAL, and fixed seal.
    pub max_bundle_bytes: u64,
    /// Maximum number of indexed originals.
    pub max_entries: usize,
    /// Maximum samples in one original series.
    pub max_series_len: usize,
    /// Maximum samples retained across all originals.
    pub max_total_samples: usize,
    /// Maximum resident bytes assigned to the persistent trie page cache.
    pub max_backend_memory_bytes: usize,
}

impl ElasticSnapshotLimits {
    /// Derive conservative compatibility limits from the historical byte ceiling.
    pub fn from_byte_ceiling(max_snapshot_bytes: u64) -> Self {
        let addressable = usize::try_from(max_snapshot_bytes).unwrap_or(usize::MAX);
        let sample_bound = addressable / (std::mem::size_of::<f64>() + 1);
        Self {
            max_manifest_bytes: max_snapshot_bytes,
            max_bundle_bytes: max_snapshot_bytes,
            max_entries: addressable / (3 * std::mem::size_of::<u64>()),
            max_series_len: sample_bound,
            max_total_samples: sample_bound,
            max_backend_memory_bytes: addressable.min(DEFAULT_BACKEND_MEMORY_BYTES),
        }
    }

    fn validate(self) -> Result<Self, ElasticSnapshotError> {
        if self.max_manifest_bytes < CHECKSUM_LEN as u64
            || self.max_bundle_bytes < self.max_manifest_bytes
            || self.max_entries == 0
            || self.max_series_len == 0
            || self.max_total_samples == 0
            || self.max_backend_memory_bytes < BLOCK_SIZE
        {
            return Err(ElasticSnapshotError::ResourceLimit {
                limit: self.max_bundle_bytes,
                requested: self.max_manifest_bytes.max(CHECKSUM_LEN as u64),
            });
        }
        Ok(self)
    }

    fn backend_pool_pages(self) -> usize {
        (self.max_backend_memory_bytes / BLOCK_SIZE).max(1)
    }
}

/// Disk-backed search-only owner for one verified, immutable generation.
///
/// The content-addressed generation itself is never opened for mutation. A
/// load copies its sealed trie into a same-filesystem private directory because
/// the storage engine requires a writable open. This type deliberately does
/// not implement [`super::ElasticMutableDictionaryBackend`], so insertion is
/// absent from the loaded index's public semantic boundary. The owner closes
/// and removes the private copy iteratively on drop.
pub struct SnapshotPersistentDictionary {
    trie: Option<PersistentARTrie<usize>>,
    working_directory: PathBuf,
}

impl fmt::Debug for SnapshotPersistentDictionary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SnapshotPersistentDictionary")
            .field("working_directory", &self.working_directory)
            .finish_non_exhaustive()
    }
}

impl ElasticDictionaryBackend for SnapshotPersistentDictionary {
    type Label = u8;
    type Node = PersistentARTrieNode<usize>;

    fn elastic_root(&self) -> Self::Node {
        self.trie.as_ref().expect("snapshot trie is live").root()
    }

    fn elastic_bucket(&self, key: &[u8]) -> Option<usize> {
        self.trie
            .as_ref()
            .expect("snapshot trie is live")
            .get_value_bytes(key)
    }

    fn elastic_len(&self) -> Option<usize> {
        self.trie.as_ref().expect("snapshot trie is live").len()
    }
}

impl Drop for SnapshotPersistentDictionary {
    fn drop(&mut self) {
        if let Some(trie) = self.trie.take() {
            trie.close();
            drop(trie);
        }
        let _ = fs::remove_dir_all(&self.working_directory);
    }
}

/// Kernel configuration that can be bound into a stable snapshot schema.
///
/// Implementations must use a globally unique tag and encode every value that
/// can affect transition, pruning, or exact-scoring semantics. Floating-point
/// configuration uses exact IEEE-754 bits; approximate equality is forbidden.
pub trait ElasticSnapshotKernel: ElasticKernel + Sized {
    /// Stable kernel identifier stored in the snapshot.
    const SNAPSHOT_TAG: &'static str;

    /// Exact number of deterministic configuration words.
    const SNAPSHOT_CONFIG_WORD_COUNT: usize;

    /// Exact deterministic word at `index`, without allocating.
    fn snapshot_config_word(&self, index: usize) -> Option<u64>;

    /// Reconstruct and validate this kernel from exact configuration words.
    fn from_snapshot_config_words(words: &[u64]) -> Option<Self>;
}

impl ElasticSnapshotKernel for ErpConfig {
    const SNAPSHOT_TAG: &'static str = "erp-l1-v1";
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 1;

    fn snapshot_config_word(&self, index: usize) -> Option<u64> {
        (index == 0).then(|| self.gap_value().to_bits())
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [gap] = words else { return None };
        let gap = f64::from_bits(*gap);
        gap.is_finite().then(|| Self::new(gap))
    }
}

impl ElasticSnapshotKernel for MsmKernel {
    const SNAPSHOT_TAG: &'static str = "msm-raw-v1";
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 1;

    fn snapshot_config_word(&self, index: usize) -> Option<u64> {
        (index == 0).then(|| self.config().split_merge_cost().to_bits())
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [cost] = words else { return None };
        let cost = f64::from_bits(*cost);
        MsmConfig::try_new(cost).ok().map(Self::new)
    }
}

impl ElasticSnapshotKernel for MetricMsmKernel {
    const SNAPSHOT_TAG: &'static str = "msm-metric-nonempty-v1";
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 1;

    fn snapshot_config_word(&self, index: usize) -> Option<u64> {
        (index == 0).then(|| self.config().split_merge_cost().to_bits())
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        let [cost] = words else { return None };
        let cost = f64::from_bits(*cost);
        MetricMsmConfig::try_new(cost).ok().map(Self::new)
    }
}

impl ElasticSnapshotKernel for TwedConfig {
    const SNAPSHOT_TAG: &'static str = "twed-unit-grid-raw-v1";
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 2;

    fn snapshot_config_word(&self, index: usize) -> Option<u64> {
        [self.stiffness().to_bits(), self.gap_penalty().to_bits()]
            .get(index)
            .copied()
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
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 2;

    fn snapshot_config_word(&self, index: usize) -> Option<u64> {
        [self.stiffness().to_bits(), self.gap_penalty().to_bits()]
            .get(index)
            .copied()
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
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 0;

    fn snapshot_config_word(&self, _index: usize) -> Option<u64> {
        None
    }

    fn from_snapshot_config_words(words: &[u64]) -> Option<Self> {
        words.is_empty().then(Self::new)
    }
}

impl ElasticSnapshotKernel for DtwConfig {
    const SNAPSHOT_TAG: &'static str = "dtw-banded-diagnostic-v1";
    const SNAPSHOT_CONFIG_WORD_COUNT: usize = 1;

    fn snapshot_config_word(&self, index: usize) -> Option<u64> {
        (index == 0).then(|| u64::try_from(self.band).unwrap_or(u64::MAX))
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
        training_fold_id: impl AsRef<str>,
        build_provenance: impl AsRef<str>,
        channel_scales: Vec<f64>,
        channel_weights: Vec<f64>,
    ) -> Result<Self, ElasticSnapshotError> {
        let training_fold_id = bounded_string(training_fold_id.as_ref())?;
        let build_provenance = bounded_string(build_provenance.as_ref())?;
        let metadata = Self {
            training_fold_id,
            build_provenance,
            channel_scales,
            channel_weights,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<(), ElasticSnapshotError> {
        if self.training_fold_id.is_empty()
            || self.build_provenance.is_empty()
            || self.training_fold_id.len() > MAX_METADATA_STRING_BYTES
            || self.build_provenance.len() > MAX_METADATA_STRING_BYTES
            || self.channel_scales.len() != self.channel_weights.len()
            || self.channel_scales.is_empty()
            || self.channel_scales.len() > MAX_CHANNELS
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

fn bounded_string(value: &str) -> Result<String, ElasticSnapshotError> {
    if value.len() > MAX_METADATA_STRING_BYTES {
        return Err(ElasticSnapshotError::InvalidMetadata);
    }
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_| ElasticSnapshotError::AllocationFailed {
            requested: value.len(),
        })?;
    owned.push_str(value);
    Ok(owned)
}

/// A loaded exact index together with the identity and metadata it verified.
#[derive(Debug)]
pub struct ElasticSnapshot<
    K: ElasticKernel,
    D: ElasticDictionaryBackend = SnapshotPersistentDictionary,
> {
    /// Reconstructed exact index, including all collision originals.
    pub index: ElasticTransducer<K, u64, D>,
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
    /// A bounded in-memory allocation failed before any partial index escaped.
    AllocationFailed {
        /// Requested allocation size in bytes when known.
        requested: usize,
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
            Self::AllocationFailed { requested } => {
                write!(formatter, "snapshot allocation of {requested} bytes failed")
            }
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

#[derive(Clone)]
struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    byte_len: u64,
}

impl Sha256 {
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

    fn new() -> Self {
        Self {
            state: Self::INITIAL,
            block: [0; 64],
            block_len: 0,
            byte_len: 0,
        }
    }

    fn update(&mut self, mut bytes: &[u8]) -> Result<(), ElasticSnapshotError> {
        let incoming =
            u64::try_from(bytes.len()).map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        self.byte_len = self
            .byte_len
            .checked_add(incoming)
            .ok_or(ElasticSnapshotError::InvalidFormat)?;

        if self.block_len != 0 {
            let copied = bytes.len().min(64 - self.block_len);
            self.block[self.block_len..self.block_len + copied].copy_from_slice(&bytes[..copied]);
            self.block_len += copied;
            bytes = &bytes[copied..];
            if self.block_len != 64 {
                return Ok(());
            }
            sha256_compress(&mut self.state, &self.block);
            self.block_len = 0;
        }

        let mut blocks = bytes.chunks_exact(64);
        for block in &mut blocks {
            let block: &[u8; 64] = block
                .try_into()
                .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
            sha256_compress(&mut self.state, block);
        }
        let remainder = blocks.remainder();
        self.block[..remainder.len()].copy_from_slice(remainder);
        self.block_len = remainder.len();
        Ok(())
    }

    fn finalize(mut self) -> Result<[u8; CHECKSUM_LEN], ElasticSnapshotError> {
        let bit_len = self
            .byte_len
            .checked_mul(8)
            .ok_or(ElasticSnapshotError::InvalidFormat)?;
        let mut final_blocks = [[0_u8; 64]; 2];
        final_blocks[0][..self.block_len].copy_from_slice(&self.block[..self.block_len]);
        final_blocks[0][self.block_len] = 0x80;
        let count = if self.block_len <= 55 { 1 } else { 2 };
        final_blocks[count - 1][56..].copy_from_slice(&bit_len.to_be_bytes());
        for block in &final_blocks[..count] {
            sha256_compress(&mut self.state, block);
        }

        let mut digest = [0_u8; CHECKSUM_LEN];
        for (output, word) in digest.chunks_exact_mut(4).zip(self.state) {
            output.copy_from_slice(&word.to_be_bytes());
        }
        Ok(digest)
    }
}

fn verify_manifest_before_parse(
    file: &mut File,
    file_len: u64,
    limit: u64,
) -> Result<(u64, [u8; CHECKSUM_LEN]), ElasticSnapshotError> {
    if file_len > limit {
        return Err(ElasticSnapshotError::ResourceLimit {
            limit,
            requested: file_len,
        });
    }
    let payload_len = file_len
        .checked_sub(CHECKSUM_LEN as u64)
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut remaining = payload_len;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    while remaining != 0 {
        let chunk = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        file.read_exact(&mut buffer[..chunk])?;
        hasher.update(&buffer[..chunk])?;
        remaining -= chunk as u64;
    }
    let digest = hasher.finalize()?;
    let mut footer = [0_u8; CHECKSUM_LEN];
    file.read_exact(&mut footer)?;
    let mut extra = [0_u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    if footer != digest {
        return Err(ElasticSnapshotError::ChecksumMismatch);
    }
    file.seek(SeekFrom::Start(0))?;
    Ok((payload_len, digest))
}

fn hash_open_prefix(file: &mut File, len: u64) -> Result<[u8; CHECKSUM_LEN], ElasticSnapshotError> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut remaining = len;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    while remaining != 0 {
        let chunk = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        file.read_exact(&mut buffer[..chunk])?;
        hasher.update(&buffer[..chunk])?;
        remaining -= chunk as u64;
    }
    hasher.finalize()
}

fn hash_regular_file(
    path: &Path,
    limit: u64,
) -> Result<(u64, [u8; CHECKSUM_LEN]), ElasticSnapshotError> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    let len = metadata.len();
    if len > limit {
        return Err(ElasticSnapshotError::ResourceLimit {
            limit,
            requested: len,
        });
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut remaining = len;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    while remaining != 0 {
        let chunk = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
            .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        file.read_exact(&mut buffer[..chunk])?;
        hasher.update(&buffer[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok((len, hasher.finalize()?))
}

#[derive(Clone, Copy)]
struct BundleSeal {
    manifest_identity: [u8; CHECKSUM_LEN],
    dictionary_len: u64,
    dictionary_digest: [u8; CHECKSUM_LEN],
    wal_len: u64,
    wal_digest: [u8; CHECKSUM_LEN],
}

impl BundleSeal {
    fn encode(self) -> Result<[u8; BUNDLE_SEAL_LEN], ElasticSnapshotError> {
        let mut bytes = [0_u8; BUNDLE_SEAL_LEN];
        let mut offset = 0;
        append_fixed(&mut bytes, &mut offset, BUNDLE_MAGIC)?;
        append_fixed(
            &mut bytes,
            &mut offset,
            &BUNDLE_SCHEMA_VERSION.to_le_bytes(),
        )?;
        append_fixed(&mut bytes, &mut offset, &self.manifest_identity)?;
        append_fixed(&mut bytes, &mut offset, &self.dictionary_len.to_le_bytes())?;
        append_fixed(&mut bytes, &mut offset, &self.dictionary_digest)?;
        append_fixed(&mut bytes, &mut offset, &self.wal_len.to_le_bytes())?;
        append_fixed(&mut bytes, &mut offset, &self.wal_digest)?;
        if offset != BUNDLE_SEAL_PAYLOAD_LEN {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        let digest = sha256(&bytes[..BUNDLE_SEAL_PAYLOAD_LEN])?;
        bytes[BUNDLE_SEAL_PAYLOAD_LEN..].copy_from_slice(&digest);
        Ok(bytes)
    }

    fn decode(path: &Path) -> Result<Self, ElasticSnapshotError> {
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file() || metadata.len() != BUNDLE_SEAL_LEN as u64 {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        let mut bytes = [0_u8; BUNDLE_SEAL_LEN];
        let mut file = File::open(path)?;
        file.read_exact(&mut bytes)?;
        let digest = sha256(&bytes[..BUNDLE_SEAL_PAYLOAD_LEN])?;
        if bytes[BUNDLE_SEAL_PAYLOAD_LEN..] != digest {
            return Err(ElasticSnapshotError::ChecksumMismatch);
        }
        let mut offset = 0;
        if take_fixed::<8>(&bytes, &mut offset)? != *BUNDLE_MAGIC
            || u32::from_le_bytes(take_fixed::<4>(&bytes, &mut offset)?) != BUNDLE_SCHEMA_VERSION
        {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        let manifest_identity = take_fixed::<CHECKSUM_LEN>(&bytes, &mut offset)?;
        let dictionary_len = u64::from_le_bytes(take_fixed::<8>(&bytes, &mut offset)?);
        let dictionary_digest = take_fixed::<CHECKSUM_LEN>(&bytes, &mut offset)?;
        let wal_len = u64::from_le_bytes(take_fixed::<8>(&bytes, &mut offset)?);
        let wal_digest = take_fixed::<CHECKSUM_LEN>(&bytes, &mut offset)?;
        if offset != BUNDLE_SEAL_PAYLOAD_LEN {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        Ok(Self {
            manifest_identity,
            dictionary_len,
            dictionary_digest,
            wal_len,
            wal_digest,
        })
    }
}

fn append_fixed(
    destination: &mut [u8],
    offset: &mut usize,
    source: &[u8],
) -> Result<(), ElasticSnapshotError> {
    let end = offset
        .checked_add(source.len())
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    let slot = destination
        .get_mut(*offset..end)
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    slot.copy_from_slice(source);
    *offset = end;
    Ok(())
}

fn take_fixed<const N: usize>(
    source: &[u8],
    offset: &mut usize,
) -> Result<[u8; N], ElasticSnapshotError> {
    let end = offset
        .checked_add(N)
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    let value = source
        .get(*offset..end)
        .ok_or(ElasticSnapshotError::InvalidFormat)?
        .try_into()
        .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
    *offset = end;
    Ok(value)
}

struct StreamingEncoder<'a, W: Write> {
    writer: &'a mut W,
    payload_bytes: u64,
    limit: u64,
}

impl<'a, W: Write> StreamingEncoder<'a, W> {
    fn new(writer: &'a mut W, limit: u64) -> Result<Self, ElasticSnapshotError> {
        if limit < CHECKSUM_LEN as u64 {
            return Err(ElasticSnapshotError::ResourceLimit {
                limit,
                requested: CHECKSUM_LEN as u64,
            });
        }
        Ok(Self {
            writer,
            payload_bytes: 0,
            limit,
        })
    }

    fn append(&mut self, bytes: &[u8]) -> Result<(), ElasticSnapshotError> {
        let payload = self
            .payload_bytes
            .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                ElasticSnapshotError::ResourceLimit {
                    limit: self.limit,
                    requested: u64::MAX,
                }
            })?)
            .ok_or(ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested: u64::MAX,
            })?;
        let requested = payload.checked_add(CHECKSUM_LEN as u64).ok_or(
            ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested: u64::MAX,
            },
        )?;
        if requested > self.limit {
            return Err(ElasticSnapshotError::ResourceLimit {
                limit: self.limit,
                requested,
            });
        }
        self.writer.write_all(bytes)?;
        self.payload_bytes = payload;
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

    fn finish(self) -> u64 {
        self.payload_bytes
    }
}

struct BoundedDecoder<'a, R: Read> {
    reader: &'a mut R,
    remaining: u64,
}

impl<'a, R: Read> BoundedDecoder<'a, R> {
    fn new(reader: &'a mut R, payload_len: u64) -> Self {
        Self {
            reader,
            remaining: payload_len,
        }
    }

    fn read_into(&mut self, output: &mut [u8]) -> Result<(), ElasticSnapshotError> {
        let len = u64::try_from(output.len()).map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        if len > self.remaining {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        self.reader.read_exact(output)?;
        self.remaining -= len;
        Ok(())
    }

    fn bytes(&mut self, len: usize, field_limit: usize) -> Result<Vec<u8>, ElasticSnapshotError> {
        let requested = u64::try_from(len).map_err(|_| ElasticSnapshotError::InvalidFormat)?;
        if len > field_limit {
            return Err(ElasticSnapshotError::ResourceLimit {
                limit: u64::try_from(field_limit).unwrap_or(u64::MAX),
                requested,
            });
        }
        if requested > self.remaining {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| ElasticSnapshotError::AllocationFailed { requested: len })?;
        bytes.resize(len, 0);
        self.read_into(&mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ElasticSnapshotError> {
        let mut bytes = [0; 1];
        self.read_into(&mut bytes)?;
        Ok(bytes[0])
    }

    fn u32(&mut self) -> Result<u32, ElasticSnapshotError> {
        let mut bytes = [0; 4];
        self.read_into(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ElasticSnapshotError> {
        let mut bytes = [0; 8];
        self.read_into(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, ElasticSnapshotError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn len(&mut self) -> Result<usize, ElasticSnapshotError> {
        usize::try_from(self.u64()?).map_err(|_| ElasticSnapshotError::InvalidFormat)
    }

    fn string(&mut self, field_limit: usize) -> Result<String, ElasticSnapshotError> {
        let len = self.len()?;
        let bytes = self.bytes(len, field_limit)?;
        String::from_utf8(bytes).map_err(|_| ElasticSnapshotError::InvalidFormat)
    }

    fn finish(self) -> Result<(), ElasticSnapshotError> {
        if self.remaining != 0 {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        Ok(())
    }
}

fn quantizers_match(left: &QuantizationConfig, right: &QuantizationConfig) -> bool {
    left.min_value.to_bits() == right.min_value.to_bits()
        && left.max_value.to_bits() == right.max_value.to_bits()
        && left.num_bins == right.num_bins
        && left.clamp_outliers == right.clamp_outliers
}

fn append_metadata<W: Write>(
    encoder: &mut StreamingEncoder<'_, W>,
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

fn read_f64_vector<R: Read>(
    decoder: &mut BoundedDecoder<'_, R>,
) -> Result<Vec<f64>, ElasticSnapshotError> {
    let len = decoder.len()?;
    if len > MAX_CHANNELS {
        return Err(ElasticSnapshotError::ResourceLimit {
            limit: MAX_CHANNELS as u64,
            requested: u64::try_from(len).unwrap_or(u64::MAX),
        });
    }
    let required = len
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    if u64::try_from(required).map_or(true, |required| required > decoder.remaining) {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| ElasticSnapshotError::AllocationFailed {
            requested: required,
        })?;
    for _ in 0..len {
        values.push(decoder.f64()?);
    }
    Ok(values)
}

fn read_metadata<R: Read>(
    decoder: &mut BoundedDecoder<'_, R>,
) -> Result<ElasticSnapshotMetadata, ElasticSnapshotError> {
    ElasticSnapshotMetadata::try_new(
        decoder.string(MAX_METADATA_STRING_BYTES)?,
        decoder.string(MAX_METADATA_STRING_BYTES)?,
        read_f64_vector(decoder)?,
        read_f64_vector(decoder)?,
    )
}

fn append_quantizer<W: Write>(
    encoder: &mut StreamingEncoder<'_, W>,
    quantizer: &QuantizationConfig,
) -> Result<(), ElasticSnapshotError> {
    encoder.f64(quantizer.min_value)?;
    encoder.f64(quantizer.max_value)?;
    encoder.u32(quantizer.num_bins)?;
    encoder.u8(u8::from(quantizer.clamp_outliers))
}

fn read_quantizer<R: Read>(
    decoder: &mut BoundedDecoder<'_, R>,
) -> Result<QuantizationConfig, ElasticSnapshotError> {
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

fn append_kernel<K: ElasticSnapshotKernel, W: Write>(
    encoder: &mut StreamingEncoder<'_, W>,
    kernel: &K,
) -> Result<(), ElasticSnapshotError> {
    encoder.string(K::SNAPSHOT_TAG)?;
    encoder.len(K::SNAPSHOT_CONFIG_WORD_COUNT)?;
    for index in 0..K::SNAPSHOT_CONFIG_WORD_COUNT {
        encoder.u64(
            kernel
                .snapshot_config_word(index)
                .ok_or(ElasticSnapshotError::InvalidFormat)?,
        )?;
    }
    Ok(())
}

fn read_kernel<K: ElasticSnapshotKernel, R: Read>(
    decoder: &mut BoundedDecoder<'_, R>,
) -> Result<K, ElasticSnapshotError> {
    if decoder.string(MAX_SCHEMA_STRING_BYTES)? != K::SNAPSHOT_TAG {
        return Err(ElasticSnapshotError::ConfigurationMismatch);
    }
    let count = decoder.len()?;
    if count != K::SNAPSHOT_CONFIG_WORD_COUNT || count > MAX_KERNEL_CONFIG_WORDS {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    let required = count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(ElasticSnapshotError::InvalidFormat)?;
    if u64::try_from(required).map_or(true, |required| required > decoder.remaining) {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    let mut words = [0_u64; MAX_KERNEL_CONFIG_WORDS];
    for word in &mut words[..count] {
        *word = decoder.u64()?;
    }
    K::from_snapshot_config_words(&words[..count]).ok_or(ElasticSnapshotError::InvalidFormat)
}

fn kernel_configs_match<K: ElasticSnapshotKernel>(left: &K, right: &K) -> bool {
    (0..K::SNAPSHOT_CONFIG_WORD_COUNT)
        .all(|index| left.snapshot_config_word(index) == right.snapshot_config_word(index))
}

fn validate_source_bijection<K, D>(
    index: &ElasticTransducer<K, u64, D>,
    limits: ElasticSnapshotLimits,
) -> Result<(), ElasticSnapshotError>
where
    K: ElasticSnapshotKernel,
    D: ElasticDictionaryBackend<Label = u8>,
{
    if index.originals.len() > limits.max_entries {
        return Err(ElasticSnapshotError::ResourceLimit {
            limit: limits.max_entries as u64,
            requested: u64::try_from(index.originals.len()).unwrap_or(u64::MAX),
        });
    }
    let mut active_buckets = 0_usize;
    let mut member_count = 0_usize;
    let mut sample_count = 0_usize;
    for (bucket_id, bucket) in index.buckets.iter().enumerate() {
        let Some(first_id) = bucket.first() else {
            continue;
        };
        active_buckets = active_buckets
            .checked_add(1)
            .ok_or(ElasticSnapshotError::InvalidFormat)?;
        let first = index
            .originals
            .get(first_id)
            .ok_or(ElasticSnapshotError::InvalidFormat)?;
        if first.series.len() > limits.max_series_len {
            return Err(ElasticSnapshotError::ResourceLimit {
                limit: limits.max_series_len as u64,
                requested: u64::try_from(first.series.len()).unwrap_or(u64::MAX),
            });
        }
        let mut key = Vec::new();
        key.try_reserve_exact(first.series.len()).map_err(|_| {
            ElasticSnapshotError::AllocationFailed {
                requested: first.series.len(),
            }
        })?;
        for value in &first.series {
            if !value.is_finite() {
                return Err(ElasticSnapshotError::InvalidFormat);
            }
            key.push(index.quant.quantize_u8(*value));
        }
        if index.dawg.elastic_bucket(&key) != Some(bucket_id) {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        for (slot, stable_id) in bucket.iter().enumerate() {
            let stored = index
                .originals
                .get(stable_id)
                .ok_or(ElasticSnapshotError::InvalidFormat)?;
            if stored.bucket_location != (bucket_id, slot) {
                return Err(ElasticSnapshotError::InvalidFormat);
            }
            if stored.series.len() != key.len()
                || stored.series.iter().zip(&key).any(|(value, expected)| {
                    !value.is_finite() || index.quant.quantize_u8(*value) != *expected
                })
            {
                return Err(ElasticSnapshotError::OriginalKeyMismatch {
                    stable_id: *stable_id,
                });
            }
            member_count = member_count
                .checked_add(1)
                .ok_or(ElasticSnapshotError::InvalidFormat)?;
            sample_count = sample_count.checked_add(stored.series.len()).ok_or(
                ElasticSnapshotError::ResourceLimit {
                    limit: limits.max_total_samples as u64,
                    requested: u64::MAX,
                },
            )?;
            if sample_count > limits.max_total_samples {
                return Err(ElasticSnapshotError::ResourceLimit {
                    limit: limits.max_total_samples as u64,
                    requested: u64::try_from(sample_count).unwrap_or(u64::MAX),
                });
            }
        }
    }
    if member_count != index.originals.len() || index.dawg.elastic_len() != Some(active_buckets) {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    Ok(())
}

fn persistent_error(error: impl fmt::Display) -> ElasticSnapshotError {
    ElasticSnapshotError::Io(io::Error::other(error.to_string()))
}

fn identity_hex(identity: &[u8; CHECKSUM_LEN]) -> Result<String, ElasticSnapshotError> {
    let mut result = String::new();
    result.try_reserve_exact(CHECKSUM_LEN * 2).map_err(|_| {
        ElasticSnapshotError::AllocationFailed {
            requested: CHECKSUM_LEN * 2,
        }
    })?;
    for byte in identity {
        write!(&mut result, "{byte:02x}").map_err(|_| ElasticSnapshotError::InvalidFormat)?;
    }
    Ok(result)
}

fn generation_root(path: &Path) -> Result<PathBuf, ElasticSnapshotError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or(ElasticSnapshotError::InvalidFormat)?
        .to_string_lossy();
    Ok(parent.join(format!(".{name}.elastic-generations")))
}

fn generation_path(
    path: &Path,
    identity: &[u8; CHECKSUM_LEN],
) -> Result<PathBuf, ElasticSnapshotError> {
    Ok(generation_root(path)?.join(identity_hex(identity)?))
}

fn create_unique_directory(parent: &Path, kind: &str) -> Result<PathBuf, ElasticSnapshotError> {
    fs::create_dir_all(parent)?;
    for nonce in 0..=u16::MAX {
        let candidate = parent.join(format!(".{kind}-{}-{nonce}", std::process::id()));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(ElasticSnapshotError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "all elastic snapshot staging names are occupied",
    )))
}

fn checked_bundle_size(
    manifest_len: u64,
    dictionary_len: u64,
    wal_len: u64,
    limits: ElasticSnapshotLimits,
) -> Result<u64, ElasticSnapshotError> {
    let requested = manifest_len
        .checked_add(dictionary_len)
        .and_then(|value| value.checked_add(wal_len))
        .and_then(|value| value.checked_add(BUNDLE_SEAL_LEN as u64))
        .ok_or(ElasticSnapshotError::ResourceLimit {
            limit: limits.max_bundle_bytes,
            requested: u64::MAX,
        })?;
    if requested > limits.max_bundle_bytes {
        return Err(ElasticSnapshotError::ResourceLimit {
            limit: limits.max_bundle_bytes,
            requested,
        });
    }
    Ok(requested)
}

fn verify_generation(
    generation: &Path,
    identity: &[u8; CHECKSUM_LEN],
    limits: ElasticSnapshotLimits,
) -> Result<BundleSeal, ElasticSnapshotError> {
    let mut manifest = File::open(generation.join(GENERATION_MANIFEST))?;
    let manifest_len = manifest.metadata()?.len();
    let (_, manifest_identity) =
        verify_manifest_before_parse(&mut manifest, manifest_len, limits.max_manifest_bytes)?;
    if &manifest_identity != identity {
        return Err(ElasticSnapshotError::ChecksumMismatch);
    }
    let seal = BundleSeal::decode(&generation.join(GENERATION_SEAL))?;
    if &seal.manifest_identity != identity {
        return Err(ElasticSnapshotError::ChecksumMismatch);
    }
    let remaining = limits.max_bundle_bytes.saturating_sub(manifest_len);
    let (dictionary_len, dictionary_digest) =
        hash_regular_file(&generation.join(GENERATION_DICTIONARY), remaining)?;
    let remaining = remaining.saturating_sub(dictionary_len);
    let (wal_len, wal_digest) = hash_regular_file(&generation.join("dictionary.wal"), remaining)?;
    checked_bundle_size(manifest_len, dictionary_len, wal_len, limits)?;
    if dictionary_len != seal.dictionary_len
        || dictionary_digest != seal.dictionary_digest
        || wal_len != seal.wal_len
        || wal_digest != seal.wal_digest
    {
        return Err(ElasticSnapshotError::ChecksumMismatch);
    }
    let mut component_count = 0_usize;
    for entry in fs::read_dir(generation)? {
        let entry = entry?;
        let name = entry.file_name();
        if name != GENERATION_MANIFEST
            && name != GENERATION_DICTIONARY
            && name != "dictionary.wal"
            && name != GENERATION_SEAL
        {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        component_count += 1;
    }
    if component_count != 4 {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    Ok(seal)
}

fn copy_regular_file(source: &Path, destination: &Path) -> Result<(), ElasticSnapshotError> {
    if !fs::symlink_metadata(source)?.file_type().is_file() {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    fs::copy(source, destination)?;
    File::open(destination)?.sync_all()?;
    Ok(())
}

fn quantized_key(
    quantizer: &QuantizationConfig,
    series: &[f64],
) -> Result<Vec<u8>, ElasticSnapshotError> {
    let mut key = Vec::new();
    key.try_reserve_exact(series.len())
        .map_err(|_| ElasticSnapshotError::AllocationFailed {
            requested: series.len(),
        })?;
    for value in series {
        if !value.is_finite() {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        key.push(quantizer.quantize_u8(*value));
    }
    Ok(key)
}

fn validate_durable_dictionary<K, D>(
    dictionary: &PersistentARTrie<usize>,
    source: &ElasticTransducer<K, u64, D>,
    stable_ids: &[u64],
) -> Result<(), ElasticSnapshotError>
where
    K: ElasticSnapshotKernel,
    D: ElasticDictionaryBackend<Label = u8>,
{
    let mut next_bucket = 0_usize;
    for stable_id in stable_ids {
        let stored = source
            .originals
            .get(stable_id)
            .ok_or(ElasticSnapshotError::InvalidFormat)?;
        let key = quantized_key(&source.quant, &stored.series)?;
        match dictionary.get_value_bytes(&key) {
            Some(bucket) if bucket < next_bucket => {}
            Some(bucket) if bucket == next_bucket => {
                next_bucket = next_bucket
                    .checked_add(1)
                    .ok_or(ElasticSnapshotError::InvalidFormat)?;
            }
            _ => return Err(ElasticSnapshotError::InvalidFormat),
        }
    }
    if dictionary.len() != Some(next_bucket) {
        return Err(ElasticSnapshotError::InvalidFormat);
    }
    Ok(())
}

fn publish_generation<K, D>(
    source: &ElasticTransducer<K, u64, D>,
    published_path: &Path,
    manifest_partial: &Path,
    manifest_identity: [u8; CHECKSUM_LEN],
    stable_ids: &[u64],
    limits: ElasticSnapshotLimits,
) -> Result<PathBuf, ElasticSnapshotError>
where
    K: ElasticSnapshotKernel,
    D: ElasticDictionaryBackend<Label = u8>,
{
    let root = generation_root(published_path)?;
    let generation = generation_path(published_path, &manifest_identity)?;
    if generation.exists() {
        verify_generation(&generation, &manifest_identity, limits)?;
        return Ok(generation);
    }

    let staging = create_unique_directory(&root, "generation-stage")?;
    let result = (|| {
        let staged_manifest = staging.join(GENERATION_MANIFEST);
        copy_regular_file(manifest_partial, &staged_manifest)?;
        let dictionary_path = staging.join(GENERATION_DICTIONARY);
        let dictionary = PersistentARTrie::<usize>::create_with_buffer_pool_size(
            &dictionary_path,
            limits.backend_pool_pages(),
        )
        .map_err(persistent_error)?;
        let mut next_bucket = 0_usize;
        for stable_id in stable_ids {
            let stored = source
                .originals
                .get(stable_id)
                .ok_or(ElasticSnapshotError::InvalidFormat)?;
            let key = quantized_key(&source.quant, &stored.series)?;
            if dictionary.get_value_bytes(&key).is_none() {
                if !dictionary
                    .try_insert_with_value_bytes(&key, next_bucket)
                    .map_err(persistent_error)?
                {
                    return Err(ElasticSnapshotError::InvalidFormat);
                }
                next_bucket = next_bucket
                    .checked_add(1)
                    .ok_or(ElasticSnapshotError::InvalidFormat)?;
            }
        }
        if dictionary.len() != Some(next_bucket) {
            return Err(ElasticSnapshotError::InvalidFormat);
        }
        dictionary.checkpoint().map_err(persistent_error)?;
        dictionary.sync().map_err(persistent_error)?;
        dictionary.close();
        drop(dictionary);

        let reopened = PersistentARTrie::<usize>::open_with_buffer_pool_size(
            &dictionary_path,
            limits.backend_pool_pages(),
        )
        .map_err(persistent_error)?;
        validate_durable_dictionary(&reopened, source, stable_ids)?;
        reopened.close();
        drop(reopened);

        let pending = staging.join("wal_pending");
        if pending.exists() {
            fs::remove_dir(&pending)?;
        }
        let write_lock = staging.join("dictionary.part.wlock");
        if write_lock.exists() {
            fs::remove_file(&write_lock)?;
        }
        let manifest_len = fs::metadata(&staged_manifest)?.len();
        let remaining = limits.max_bundle_bytes.saturating_sub(manifest_len);
        let (dictionary_len, dictionary_digest) = hash_regular_file(&dictionary_path, remaining)?;
        let remaining = remaining.saturating_sub(dictionary_len);
        let wal_path = staging.join("dictionary.wal");
        let (wal_len, wal_digest) = hash_regular_file(&wal_path, remaining)?;
        checked_bundle_size(manifest_len, dictionary_len, wal_len, limits)?;
        let seal = BundleSeal {
            manifest_identity,
            dictionary_len,
            dictionary_digest,
            wal_len,
            wal_digest,
        }
        .encode()?;
        let seal_path = staging.join(GENERATION_SEAL);
        let mut seal_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&seal_path)?;
        seal_file.write_all(&seal)?;
        seal_file.sync_all()?;
        drop(seal_file);
        #[cfg(unix)]
        File::open(&staging)?.sync_all()?;

        match fs::rename(&staging, &generation) {
            Ok(()) => {}
            // POSIX platforms do not agree on whether renaming a directory
            // over a concurrently published nonempty directory reports
            // AlreadyExists or DirectoryNotEmpty. Existence is only a reason
            // to converge after the winner's complete seal is reverified.
            Err(_) if generation.exists() => {
                verify_generation(&generation, &manifest_identity, limits)?;
                fs::remove_dir_all(&staging)?;
            }
            Err(error) => return Err(error.into()),
        }
        #[cfg(unix)]
        File::open(&root)?.sync_all()?;
        verify_generation(&generation, &manifest_identity, limits)?;
        Ok(generation.clone())
    })();
    if result.is_err() && staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn open_working_dictionary(
    published_path: &Path,
    generation: &Path,
    limits: ElasticSnapshotLimits,
) -> Result<SnapshotPersistentDictionary, ElasticSnapshotError> {
    let root = generation_root(published_path)?;
    let working_directory = create_unique_directory(&root, "load")?;
    let result = (|| {
        let dictionary_path = working_directory.join(GENERATION_DICTIONARY);
        copy_regular_file(&generation.join(GENERATION_DICTIONARY), &dictionary_path)?;
        copy_regular_file(
            &generation.join("dictionary.wal"),
            &working_directory.join("dictionary.wal"),
        )?;
        #[cfg(unix)]
        File::open(&working_directory)?.sync_all()?;
        let trie = PersistentARTrie::<usize>::open_with_buffer_pool_size(
            &dictionary_path,
            limits.backend_pool_pages(),
        )
        .map_err(persistent_error)?;
        Ok(SnapshotPersistentDictionary {
            trie: Some(trie),
            working_directory: working_directory.clone(),
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&working_directory);
    }
    result
}

#[cfg(feature = "persistent-artrie")]
impl<K, D> ElasticTransducer<K, u64, D>
where
    K: ElasticSnapshotKernel,
    D: ElasticDictionaryBackend<Label = u8>,
{
    /// Stream a complete canonical snapshot to a same-directory partial file,
    /// sync it, atomically rename it, and then sync the containing directory.
    pub fn write_complete_snapshot(
        &self,
        path: impl AsRef<Path>,
        metadata: &ElasticSnapshotMetadata,
        max_snapshot_bytes: u64,
    ) -> Result<ElasticSnapshotIdentity, ElasticSnapshotError> {
        self.write_complete_snapshot_with_limits(
            path,
            metadata,
            ElasticSnapshotLimits::from_byte_ceiling(max_snapshot_bytes),
        )
    }

    /// Publish a complete content-addressed generation under explicit resource ceilings.
    pub fn write_complete_snapshot_with_limits(
        &self,
        path: impl AsRef<Path>,
        metadata: &ElasticSnapshotMetadata,
        limits: ElasticSnapshotLimits,
    ) -> Result<ElasticSnapshotIdentity, ElasticSnapshotError> {
        let limits = limits.validate()?;
        metadata.validate()?;
        validate_source_bijection(self, limits)?;
        let path = path.as_ref();
        let (temporary, mut file) = create_partial_file(path)?;
        let result = (|| {
            let mut stable_ids = Vec::new();
            stable_ids
                .try_reserve_exact(self.originals.len())
                .map_err(|_| ElasticSnapshotError::AllocationFailed {
                    requested: self
                        .originals
                        .len()
                        .saturating_mul(std::mem::size_of::<u64>()),
                })?;
            stable_ids.extend(self.originals.keys().copied());
            stable_ids.sort_unstable();
            let payload_len = {
                let mut encoder = StreamingEncoder::new(&mut file, limits.max_manifest_bytes)?;
                encoder.append(MAGIC)?;
                encoder.u32(SCHEMA_VERSION)?;
                encoder.string(env!("CARGO_PKG_NAME"))?;
                encoder.string(env!("CARGO_PKG_VERSION"))?;
                encoder.string(IMPLEMENTATION_VERSION)?;
                encoder.string(DICTIONARY_SCHEMA)?;
                append_kernel(&mut encoder, &self.kernel)?;
                append_quantizer(&mut encoder, &self.quant)?;
                append_metadata(&mut encoder, metadata)?;

                encoder.len(stable_ids.len())?;

                for stable_id in &stable_ids {
                    let stored = self
                        .originals
                        .get(stable_id)
                        .ok_or(ElasticSnapshotError::InvalidFormat)?;
                    let key = quantized_key(&self.quant, &stored.series)?;

                    encoder.u64(*stable_id)?;
                    encoder.len(key.len())?;
                    encoder.append(&key)?;
                    encoder.len(stored.series.len())?;
                    for value in &stored.series {
                        encoder.f64(*value)?;
                    }
                }
                encoder.finish()
            };

            file.flush()?;
            let digest = hash_open_prefix(&mut file, payload_len)?;
            file.seek(SeekFrom::Start(payload_len))?;
            file.write_all(&digest)?;
            file.sync_all()?;
            drop(file);
            publish_generation(self, path, &temporary, digest, &stable_ids, limits)?;
            fs::rename(&temporary, path)?;
            #[cfg(unix)]
            File::open(path.parent().unwrap_or_else(|| Path::new(".")))?.sync_all()?;
            Ok(ElasticSnapshotIdentity(digest))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
}

#[cfg(feature = "persistent-artrie")]
impl<K> ElasticTransducer<K, u64>
where
    K: ElasticSnapshotKernel,
{
    /// Stream and verify a complete snapshot into a persistent byte-key trie.
    /// No index escapes until the payload checksum, all exact configuration
    /// bindings, canonical stable-id order, and every original/key relation
    /// have been checked.
    pub fn load_complete_snapshot(
        path: impl AsRef<Path>,
        expected_quantizer: &QuantizationConfig,
        expected_kernel: &K,
        expected_metadata: &ElasticSnapshotMetadata,
        max_snapshot_bytes: u64,
    ) -> Result<ElasticSnapshot<K>, ElasticSnapshotError> {
        Self::load_complete_snapshot_with_limits(
            path,
            expected_quantizer,
            expected_kernel,
            expected_metadata,
            ElasticSnapshotLimits::from_byte_ceiling(max_snapshot_bytes),
        )
    }

    /// Verify and load a content-addressed persistent generation under explicit limits.
    pub fn load_complete_snapshot_with_limits(
        path: impl AsRef<Path>,
        expected_quantizer: &QuantizationConfig,
        expected_kernel: &K,
        expected_metadata: &ElasticSnapshotMetadata,
        limits: ElasticSnapshotLimits,
    ) -> Result<ElasticSnapshot<K>, ElasticSnapshotError> {
        let limits = limits.validate()?;
        expected_metadata.validate()?;
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let (payload_len, digest) =
            verify_manifest_before_parse(&mut file, file_len, limits.max_manifest_bytes)?;
        let generation = generation_path(path, &digest)?;
        verify_generation(&generation, &digest, limits)?;

        let (mut index, metadata) = {
            let mut decoder = BoundedDecoder::new(&mut file, payload_len);
            if decoder.bytes(MAGIC.len(), MAGIC.len())? != MAGIC
                || decoder.u32()? != SCHEMA_VERSION
                || decoder.string(MAX_SCHEMA_STRING_BYTES)? != env!("CARGO_PKG_NAME")
                || decoder.string(MAX_SCHEMA_STRING_BYTES)? != env!("CARGO_PKG_VERSION")
                || decoder.string(MAX_SCHEMA_STRING_BYTES)? != IMPLEMENTATION_VERSION
                || decoder.string(MAX_SCHEMA_STRING_BYTES)? != DICTIONARY_SCHEMA
            {
                return Err(ElasticSnapshotError::ConfigurationMismatch);
            }

            let kernel = read_kernel::<K, _>(&mut decoder)?;
            if !kernel_configs_match(&kernel, expected_kernel) {
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
            if entry_count > limits.max_entries
                || entry_count
                    > usize::try_from(decoder.remaining)
                        .unwrap_or(usize::MAX)
                        .checked_div(minimum_entry_bytes)
                        .unwrap_or(0)
            {
                return Err(ElasticSnapshotError::ResourceLimit {
                    limit: limits.max_entries as u64,
                    requested: u64::try_from(entry_count).unwrap_or(u64::MAX),
                });
            }

            let dictionary = open_working_dictionary(path, &generation, limits)?;
            let bin_count = usize::try_from(quantizer.num_bins)
                .map_err(|_| ElasticSnapshotError::InvalidFormat)?;
            let mut bin_bounds = Vec::new();
            bin_bounds.try_reserve_exact(bin_count).map_err(|_| {
                ElasticSnapshotError::AllocationFailed {
                    requested: bin_count.saturating_mul(std::mem::size_of::<(f64, f64)>()),
                }
            })?;
            bin_bounds.extend((0..quantizer.num_bins).map(|bin| quantizer.bin_bounds(bin)));
            let mut index = ElasticTransducer::<K, u64, SnapshotPersistentDictionary> {
                dawg: dictionary,
                quant: quantizer,
                kernel,
                bin_bounds,
                buckets: Vec::new(),
                originals: std::collections::HashMap::new(),
                snapshot_identity: None,
            };
            index.originals.try_reserve(entry_count).map_err(|_| {
                ElasticSnapshotError::AllocationFailed {
                    requested: entry_count
                        .saturating_mul(std::mem::size_of::<(u64, super::StoredSeries)>()),
                }
            })?;
            index.buckets.try_reserve(entry_count).map_err(|_| {
                ElasticSnapshotError::AllocationFailed {
                    requested: entry_count.saturating_mul(std::mem::size_of::<Vec<u64>>()),
                }
            })?;

            let mut previous_id = None;
            let mut total_samples = 0_usize;
            for _ in 0..entry_count {
                let stable_id = decoder.u64()?;
                if previous_id.is_some_and(|previous| stable_id <= previous) {
                    return Err(ElasticSnapshotError::InvalidStableIdOrder);
                }
                previous_id = Some(stable_id);

                let key_len = decoder.len()?;
                let key = decoder.bytes(key_len, limits.max_series_len)?;
                let series_len = decoder.len()?;
                if key_len != series_len || series_len > limits.max_series_len {
                    return Err(ElasticSnapshotError::OriginalKeyMismatch { stable_id });
                }
                total_samples = total_samples.checked_add(series_len).ok_or(
                    ElasticSnapshotError::ResourceLimit {
                        limit: limits.max_total_samples as u64,
                        requested: u64::MAX,
                    },
                )?;
                if total_samples > limits.max_total_samples {
                    return Err(ElasticSnapshotError::ResourceLimit {
                        limit: limits.max_total_samples as u64,
                        requested: u64::try_from(total_samples).unwrap_or(u64::MAX),
                    });
                }
                let required = series_len
                    .checked_mul(std::mem::size_of::<f64>())
                    .ok_or(ElasticSnapshotError::InvalidFormat)?;
                if u64::try_from(required).map_or(true, |required| required > decoder.remaining) {
                    return Err(ElasticSnapshotError::InvalidFormat);
                }
                let mut series = Vec::new();
                series.try_reserve_exact(series_len).map_err(|_| {
                    ElasticSnapshotError::AllocationFailed {
                        requested: required,
                    }
                })?;
                for expected_bin in &key {
                    let value = decoder.f64()?;
                    if !value.is_finite() || index.quant.quantize_u8(value) != *expected_bin {
                        return Err(ElasticSnapshotError::OriginalKeyMismatch { stable_id });
                    }
                    series.push(value);
                }

                let bucket_id = index
                    .dawg
                    .elastic_bucket(&key)
                    .ok_or(ElasticSnapshotError::InvalidFormat)?;
                if bucket_id > index.buckets.len() {
                    return Err(ElasticSnapshotError::InvalidFormat);
                }
                if bucket_id == index.buckets.len() {
                    index.buckets.push(Vec::new());
                } else {
                    let first_id = *index.buckets[bucket_id]
                        .first()
                        .ok_or(ElasticSnapshotError::InvalidFormat)?;
                    let first = index
                        .originals
                        .get(&first_id)
                        .ok_or(ElasticSnapshotError::InvalidFormat)?;
                    if first.series.len() != key.len()
                        || first
                            .series
                            .iter()
                            .zip(&key)
                            .any(|(value, bin)| index.quant.quantize_u8(*value) != *bin)
                    {
                        return Err(ElasticSnapshotError::InvalidFormat);
                    }
                }
                let bucket = &mut index.buckets[bucket_id];
                bucket.try_reserve_exact(1).map_err(|_| {
                    ElasticSnapshotError::AllocationFailed {
                        requested: bucket
                            .len()
                            .saturating_add(1)
                            .saturating_mul(std::mem::size_of::<u64>()),
                    }
                })?;
                let slot = bucket.len();
                bucket.push(stable_id);
                if index
                    .originals
                    .insert(
                        stable_id,
                        super::StoredSeries {
                            series,
                            bucket_location: (bucket_id, slot),
                        },
                    )
                    .is_some()
                {
                    return Err(ElasticSnapshotError::InvalidStableIdOrder);
                }
            }
            decoder.finish()?;
            if index.dawg.elastic_len() != Some(index.buckets.len())
                || index.buckets.iter().any(Vec::is_empty)
            {
                return Err(ElasticSnapshotError::InvalidFormat);
            }
            (index, metadata)
        };
        index.snapshot_identity = Some(ElasticSnapshotIdentity(digest));

        Ok(ElasticSnapshot {
            index,
            identity: ElasticSnapshotIdentity(digest),
            metadata,
        })
    }
}

fn create_partial_file(path: &Path) -> Result<(PathBuf, File), ElasticSnapshotError> {
    for nonce in 0..=u8::MAX {
        let candidate = partial_path(path, nonce)?;
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(ElasticSnapshotError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "all snapshot partial-file names are occupied",
    )))
}

fn partial_path(path: &Path, nonce: u8) -> Result<PathBuf, ElasticSnapshotError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or(ElasticSnapshotError::InvalidFormat)?
        .to_string_lossy();
    Ok(parent.join(format!(".{name}.partial-{}-{nonce}", std::process::id())))
}

fn sha256(message: &[u8]) -> Result<[u8; CHECKSUM_LEN], ElasticSnapshotError> {
    let mut hasher = Sha256::new();
    hasher.update(message)?;
    hasher.finalize()
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

#[cfg(all(test, feature = "persistent-artrie"))]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::thread;

    use super::{
        partial_path, sha256, BoundedDecoder, ElasticSnapshotError, ElasticSnapshotMetadata,
        Sha256, SnapshotPersistentDictionary,
    };
    use crate::time_series::elastic::walker::{ElasticMutableDictionaryBackend, ElasticTransducer};
    use crate::time_series::encoding::QuantizationConfig;
    use crate::time_series::kernels::ErpConfig;

    fn scratch(prefix: &str) -> tempfile::TempDir {
        fs::create_dir_all("target/test-tmp").expect("create disk-backed test scratch root");
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir_in("target/test-tmp")
            .expect("create disk-backed snapshot scratch directory")
    }

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

        let message = vec![0x5a; 257];
        let expected = sha256(&message).expect("one-shot digest");
        for chunk_size in 1..=80 {
            let mut streaming = Sha256::new();
            for chunk in message.chunks(chunk_size) {
                streaming.update(chunk).expect("streaming digest update");
            }
            assert_eq!(streaming.finalize().expect("streaming digest"), expected);
        }
    }

    fn fixture_metadata() -> ElasticSnapshotMetadata {
        ElasticSnapshotMetadata::try_new(
            "training-fold-3",
            "commit=abc;lock=def;toolchain=1.95",
            vec![2.0],
            vec![0.5],
        )
        .expect("valid metadata")
    }

    fn fixture_quantizer() -> QuantizationConfig {
        QuantizationConfig::try_uniform(0.0, 100.0, 4).expect("valid quantizer")
    }

    fn fixture_index(reverse: bool, include_third: bool) -> ElasticTransducer<ErpConfig, u64> {
        let mut index = ElasticTransducer::new(fixture_quantizer(), ErpConfig::new(0.0));
        let mut entries = vec![(3, vec![10.1, 20.1]), (7, vec![10.2, 20.2])];
        if include_third {
            entries.push((11, vec![80.0]));
        }
        if reverse {
            entries.reverse();
        }
        for (id, series) in entries {
            assert!(index.insert(id, &series));
        }
        index
    }

    #[test]
    fn malicious_lengths_and_metadata_fail_before_allocation() {
        let encoded = u64::MAX.to_le_bytes();
        let mut input = &encoded[..];
        let mut decoder = BoundedDecoder::new(&mut input, encoded.len() as u64);
        assert!(matches!(
            decoder.string(256),
            Err(ElasticSnapshotError::ResourceLimit {
                limit: 256,
                requested: u64::MAX,
            })
        ));

        let oversized = "x".repeat(super::MAX_METADATA_STRING_BYTES + 1);
        assert!(matches!(
            ElasticSnapshotMetadata::try_new(oversized, "build", vec![1.0], vec![1.0]),
            Err(ElasticSnapshotError::InvalidMetadata)
        ));
    }

    #[test]
    fn source_bucket_and_dictionary_bijection_is_mandatory() {
        let directory = scratch("snapshot-bijection-");
        let metadata = fixture_metadata();

        let mut duplicate = fixture_index(false, false);
        let bucket = duplicate
            .originals
            .get(&3)
            .expect("stored original")
            .bucket_location
            .0;
        duplicate.buckets[bucket].push(3);
        assert!(matches!(
            duplicate.write_complete_snapshot(
                directory.path().join("duplicate.snapshot"),
                &metadata,
                1 << 20,
            ),
            Err(ElasticSnapshotError::InvalidFormat)
        ));

        let mut extra_terminal = fixture_index(false, false);
        assert!(extra_terminal
            .dawg
            .elastic_try_insert_bucket(b"extra", 99)
            .expect("in-memory dictionary mutation"));
        assert!(matches!(
            extra_terminal.write_complete_snapshot(
                directory.path().join("extra-terminal.snapshot"),
                &metadata,
                1 << 20,
            ),
            Err(ElasticSnapshotError::InvalidFormat)
        ));
    }

    #[test]
    fn snapshot_identity_is_insertion_order_independent_and_retains_collisions() {
        let directory = scratch("snapshot-identity-");
        let first_path = directory.path().join("first.snapshot");
        let second_path = directory.path().join("second.snapshot");
        let metadata = fixture_metadata();
        let first = fixture_index(false, false);
        let second = fixture_index(true, false);

        let first_identity = first
            .write_complete_snapshot(&first_path, &metadata, 1 << 20)
            .expect("stream first snapshot");
        let second_identity = second
            .write_complete_snapshot(&second_path, &metadata, 1 << 20)
            .expect("stream second snapshot");
        assert_eq!(first_identity, second_identity);
        assert_eq!(
            fs::read(&first_path).expect("read first"),
            fs::read(&second_path).expect("read second")
        );

        let loaded = ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
            &first_path,
            &fixture_quantizer(),
            &ErpConfig::new(0.0),
            &metadata,
            1 << 20,
        )
        .expect("load persistent snapshot");
        fn assert_persistent_backend(
            _: &ElasticTransducer<ErpConfig, u64, SnapshotPersistentDictionary>,
        ) {
        }
        assert_persistent_backend(&loaded.index);
        assert_eq!(loaded.identity, first_identity);
        assert_eq!(loaded.index.len(), 2);
        assert_eq!(loaded.index.get_original(&3), Some(&[10.1, 20.1][..]));
        assert_eq!(loaded.index.get_original(&7), Some(&[10.2, 20.2][..]));
        let mut exact = loaded.index.search_range(&[10.1, 20.1], 1.0);
        exact.sort_unstable_by_key(|(id, _)| *id);
        assert_eq!(
            exact.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![3, 7]
        );
    }

    #[test]
    fn corruption_and_configuration_mismatch_fail_closed() {
        let directory = scratch("snapshot-rejection-");
        let path = directory.path().join("index.snapshot");
        let metadata = fixture_metadata();
        fixture_index(false, false)
            .write_complete_snapshot(&path, &metadata, 1 << 20)
            .expect("write snapshot");

        let mismatch = ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
            &path,
            &fixture_quantizer(),
            &ErpConfig::new(1.0),
            &metadata,
            1 << 20,
        );
        assert!(matches!(
            mismatch,
            Err(ElasticSnapshotError::ConfigurationMismatch)
        ));

        let mut bytes = fs::read(&path).expect("read snapshot");
        let payload_byte = bytes.len() / 2;
        bytes[payload_byte] ^= 0x01;
        fs::write(&path, bytes).expect("write corrupt snapshot");
        assert!(ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
            &path,
            &fixture_quantizer(),
            &ErpConfig::new(0.0),
            &metadata,
            1 << 20,
        )
        .is_err());
    }

    #[test]
    fn failed_or_crashed_publication_never_replaces_the_last_complete_snapshot() {
        let directory = scratch("snapshot-publication-");
        let path = directory.path().join("index.snapshot");
        let metadata = fixture_metadata();
        let index = fixture_index(false, false);
        let identity = index
            .write_complete_snapshot(&path, &metadata, 1 << 20)
            .expect("write initial snapshot");
        let complete = fs::read(&path).expect("read initial snapshot");

        let failed = fixture_index(false, true).write_complete_snapshot(&path, &metadata, 64);
        assert!(matches!(
            failed,
            Err(ElasticSnapshotError::ResourceLimit { .. })
        ));
        assert_eq!(fs::read(&path).expect("read preserved snapshot"), complete);

        let stale = partial_path(&path, 250).expect("partial path");
        fs::write(&stale, b"simulated crash before rename").expect("write stale partial");
        let loaded = ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
            &path,
            &fixture_quantizer(),
            &ErpConfig::new(0.0),
            &metadata,
            1 << 20,
        )
        .expect("stale partial must be invisible");
        assert_eq!(loaded.identity, identity);
    }

    #[test]
    fn concurrent_readers_observe_only_complete_old_or_new_generations() {
        let directory = scratch("snapshot-concurrent-");
        let path = Arc::new(directory.path().join("index.snapshot"));
        let metadata = Arc::new(fixture_metadata());
        fixture_index(false, false)
            .write_complete_snapshot(path.as_ref(), metadata.as_ref(), 1 << 20)
            .expect("write initial generation");

        let writer_path = Arc::clone(&path);
        let writer_metadata = Arc::clone(&metadata);
        let writer = thread::spawn(move || {
            let old = fixture_index(false, false);
            let new = fixture_index(false, true);
            for generation in 0..20 {
                let index = if generation % 2 == 0 { &new } else { &old };
                index
                    .write_complete_snapshot(
                        writer_path.as_ref(),
                        writer_metadata.as_ref(),
                        1 << 20,
                    )
                    .expect("publish complete generation");
            }
        });

        for _ in 0..40 {
            let loaded = ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
                path.as_ref(),
                &fixture_quantizer(),
                &ErpConfig::new(0.0),
                metadata.as_ref(),
                1 << 20,
            )
            .expect("reader must observe a complete generation");
            assert!(matches!(loaded.index.len(), 2 | 3));
            assert_eq!(loaded.index.get_original(&3), Some(&[10.1, 20.1][..]));
            assert_eq!(loaded.index.get_original(&7), Some(&[10.2, 20.2][..]));
        }
        writer.join().expect("writer thread");
    }
}
