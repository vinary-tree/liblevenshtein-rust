//! Compatibility name for exact MSM retrieval through the elastic walker.

use super::elastic::ElasticTransducer;
use super::msm_kernel::MsmKernel;

/// Exact Move-Split-Merge similarity index.
///
/// This public name remains source-compatible while the implementation is the
/// generic [`ElasticTransducer`] instantiated with [`MsmKernel`].
pub type MsmTransducer<V = usize> = ElasticTransducer<MsmKernel, V>;
