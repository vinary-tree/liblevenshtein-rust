//! Elastic time-series kernels built on the shared trie walker.

mod dtw;
mod erp;
mod frechet;
mod keogh;
mod soft_dtw;
mod twed;

pub use dtw::{DtwConfig, DtwKernel, DtwTransducer};
pub use erp::{erp_gap_mass_lower_bound, ErpConfig, ErpKernel, ErpTransducer};
pub use frechet::{
    frechet_candidate_lower_bound, frechet_endpoint_lower_bound,
    frechet_one_sided_hausdorff_lower_bound, FrechetConfig, FrechetKernel, FrechetTransducer,
};
pub use keogh::{keogh_envelopes, lb_keogh, lb_keogh_squared, KeoghPlan};
pub use soft_dtw::{SoftDtwAnalysis, SoftDtwConfig, SoftDtwConfigError};
pub use twed::{
    twed_length_lower_bound, MetricTwedConfig, MetricTwedConfigError, MetricTwedKernel,
    MetricTwedTransducer, MetricUnitGridTwedConfig, MetricUnitGridTwedKernel,
    MetricUnitGridTwedTransducer, TwedConfig, TwedKernel, TwedTransducer, UnitGridTwedConfig,
    UnitGridTwedKernel, UnitGridTwedTransducer,
};
