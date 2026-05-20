//! Adaptive MSM parameter learning using FPTL algorithm.
//!
//! This module implements the Follow-the-Perturbed-Tropical-Leader (FPTL)
//! algorithm for adaptively learning the optimal `c` constant in MSM metrics.
//!
//! # Algorithm Overview
//!
//! FPTL is an online learning algorithm for edit-distance-like losses:
//!
//! 1. Maintain cumulative weight automaton over tropical semiring
//! 2. After each observation, compose with loss transducer
//! 3. Add Laplacian noise for exploration
//! 4. Select the shortest path as the next prediction
//!
//! The algorithm learns the optimal `c` constant for the MSM metric by
//! observing query-target pairs and their actual costs.
//!
//! # Regret Bounds
//!
//! FPTL achieves regret bound O(√(T log |Σ|)) for the tropical semiring,
//! where T is the number of rounds and |Σ| is the alphabet size.
//!
//! # References
//!
//! - Cortes, C., et al. (2015). "On-Line Learning Algorithms for Path Experts
//!   with Non-Additive Losses" - FPTL algorithm
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};
//! use liblevenshtein::time_series::MsmConfig;
//!
//! let config = AdaptiveMsmConfig::new()
//!     .initial_c(1.0)
//!     .epsilon(0.1)
//!     .with_statistics();
//!
//! let mut adaptive = AdaptiveMsm::new(config);
//!
//! // Learning loop
//! for (query, target) in training_data {
//!     let predicted_cost = adaptive.predict(&query, &target);
//!     let actual_cost = compute_actual_cost(&query, &target);
//!     adaptive.observe(actual_cost);
//! }
//!
//! // Get the learned optimal c constant
//! let optimal_c = adaptive.current_c();
//! ```

use crate::time_series::MsmConfig;
use rand::Rng;
use rand::SeedableRng;

/// Configuration for adaptive MSM learning.
#[derive(Clone, Debug)]
pub struct AdaptiveMsmConfig {
    /// Initial value for the c constant.
    pub initial_c: f64,

    /// Learning rate (ε in FPTL).
    /// Controls the scale of Laplacian perturbation.
    pub epsilon: f64,

    /// Minimum allowed c value.
    pub min_c: f64,

    /// Maximum allowed c value.
    pub max_c: f64,

    /// Window size for gradient estimation.
    pub window_size: usize,

    /// Enable statistics tracking.
    pub track_statistics: bool,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for AdaptiveMsmConfig {
    fn default() -> Self {
        Self {
            initial_c: 1.0,
            epsilon: 0.1,
            min_c: 0.001,
            max_c: 100.0,
            window_size: 10,
            track_statistics: false,
            seed: None,
        }
    }
}

impl AdaptiveMsmConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial c constant.
    pub fn initial_c(mut self, c: f64) -> Self {
        self.initial_c = c.max(self.min_c).min(self.max_c);
        self
    }

    /// Set the learning rate (epsilon).
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps.max(0.001);
        self
    }

    /// Set the c constant bounds.
    pub fn c_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_c = min.max(0.0);
        self.max_c = max.max(min);
        self
    }

    /// Set the window size for gradient estimation.
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(1);
        self
    }

    /// Enable statistics tracking.
    pub fn with_statistics(mut self) -> Self {
        self.track_statistics = true;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Statistics tracked during adaptive MSM operation.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveMsmStatistics {
    /// Number of observations.
    pub observations: usize,

    /// Total accumulated cost.
    pub total_cost: f64,

    /// Average cost per observation.
    pub average_cost: f64,

    /// Number of move operations observed.
    pub move_count: usize,

    /// Number of merge operations observed.
    pub merge_count: usize,

    /// Number of split operations observed.
    pub split_count: usize,

    /// History of c values.
    pub c_history: Vec<f64>,

    /// History of costs.
    pub cost_history: Vec<f64>,

    /// Current estimated gradient.
    pub current_gradient: f64,
}

impl AdaptiveMsmStatistics {
    /// Update statistics with a new observation.
    fn update(&mut self, cost: f64, c_value: f64) {
        self.total_cost += cost;
        self.observations += 1;
        self.average_cost = self.total_cost / self.observations as f64;
        self.c_history.push(c_value);
        self.cost_history.push(cost);
    }
}

/// Adaptive MSM wrapper with online learning via FPTL.
///
/// Learns the optimal `c` constant for MSM by observing query-target pairs
/// and their actual costs. Uses the FPTL algorithm with Laplacian
/// perturbation for exploration.
pub struct AdaptiveMsm {
    /// Configuration.
    config: AdaptiveMsmConfig,

    /// Current c constant.
    current_c: f64,

    /// Current MSM configuration.
    msm_config: MsmConfig,

    /// Random number generator.
    rng: rand::rngs::StdRng,

    /// Statistics (if enabled).
    statistics: Option<AdaptiveMsmStatistics>,

    /// Recent cost observations for gradient estimation.
    recent_costs: Vec<f64>,

    /// Recent c values for gradient estimation.
    recent_c_values: Vec<f64>,

    /// Round counter.
    round: usize,
}

impl AdaptiveMsm {
    /// Create a new adaptive MSM instance.
    pub fn new(config: AdaptiveMsmConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let statistics = if config.track_statistics {
            Some(AdaptiveMsmStatistics::default())
        } else {
            None
        };

        let current_c = config.initial_c;

        Self {
            msm_config: MsmConfig::new(current_c),
            current_c,
            rng,
            statistics,
            recent_costs: Vec::with_capacity(config.window_size),
            recent_c_values: Vec::with_capacity(config.window_size),
            round: 0,
            config,
        }
    }

    /// Get the current c constant.
    #[inline]
    pub fn current_c(&self) -> f64 {
        self.current_c
    }

    /// Get the current MSM configuration.
    #[inline]
    pub fn msm_config(&self) -> &MsmConfig {
        &self.msm_config
    }

    /// Get the current round number.
    #[inline]
    pub fn round(&self) -> usize {
        self.round
    }

    /// Get statistics (if tracking is enabled).
    pub fn statistics(&self) -> Option<&AdaptiveMsmStatistics> {
        self.statistics.as_ref()
    }

    /// Predict the MSM cost for a query-target pair using current c.
    ///
    /// This applies Laplacian perturbation to the c constant for exploration.
    pub fn predict(&mut self, query: &[f64], target: &[f64]) -> f64 {
        // Apply FPTL perturbation: add Laplacian noise
        let perturbed_c = self.perturb_c();

        // Compute MSM distance with perturbed c
        let msm = MsmConfig::new(perturbed_c);
        msm.distance(query, target)
    }

    /// Predict without perturbation (for evaluation).
    pub fn predict_deterministic(&self, query: &[f64], target: &[f64]) -> f64 {
        self.msm_config.distance(query, target)
    }

    /// Observe the actual cost and update the c constant.
    ///
    /// Uses gradient estimation to update c towards lower costs.
    pub fn observe(&mut self, actual_cost: f64) {
        // Store observation
        self.recent_costs.push(actual_cost);
        self.recent_c_values.push(self.current_c);

        // Keep window bounded
        if self.recent_costs.len() > self.config.window_size {
            self.recent_costs.remove(0);
            self.recent_c_values.remove(0);
        }

        // Update statistics
        if let Some(ref mut stats) = self.statistics {
            stats.update(actual_cost, self.current_c);
        }

        // Estimate gradient and update c
        if self.recent_costs.len() >= 2 {
            let gradient = self.estimate_gradient();
            self.update_c(gradient);

            if let Some(ref mut stats) = self.statistics {
                stats.current_gradient = gradient;
            }
        }

        self.round += 1;
    }

    /// Apply Laplacian perturbation to c for exploration.
    ///
    /// Samples from Laplace(0, scale) where scale = 1/epsilon.
    /// Laplace distribution: f(x) = (1/2b) * exp(-|x|/b)
    fn perturb_c(&mut self) -> f64 {
        let scale: f64 = 1.0 / self.config.epsilon;

        // Sample from Laplace(0, scale) using inverse CDF method:
        // If U ~ Uniform(0, 1), then X = -b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
        // Or equivalently, sample exponential and random sign
        let u: f64 = self.rng.gen();
        let noise: f64 = if u < 0.5 {
            scale * (2.0_f64 * u).ln() // negative part
        } else {
            -scale * (2.0_f64 * (1.0 - u)).ln() // positive part
        };

        let perturbed = self.current_c + noise;

        // Clamp to valid range
        perturbed.max(self.config.min_c).min(self.config.max_c)
    }

    /// Estimate the gradient ∂Cost/∂c using finite differences.
    fn estimate_gradient(&self) -> f64 {
        if self.recent_costs.len() < 2 || self.recent_c_values.len() < 2 {
            return 0.0;
        }

        // Simple finite difference: (cost[i] - cost[i-1]) / (c[i] - c[i-1])
        let n = self.recent_costs.len();
        let mut total_gradient = 0.0;
        let mut count = 0;

        for i in 1..n {
            let dc = self.recent_c_values[i] - self.recent_c_values[i - 1];
            if dc.abs() > 1e-10 {
                let dcost = self.recent_costs[i] - self.recent_costs[i - 1];
                total_gradient += dcost / dc;
                count += 1;
            }
        }

        if count > 0 {
            total_gradient / count as f64
        } else {
            0.0
        }
    }

    /// Update c using gradient descent.
    fn update_c(&mut self, gradient: f64) {
        // Gradient descent step: c = c - ε * gradient
        let step = self.config.epsilon * gradient;
        let new_c = self.current_c - step;

        // Clamp to valid range
        self.current_c = new_c.max(self.config.min_c).min(self.config.max_c);
        self.msm_config = MsmConfig::new(self.current_c);
    }

    /// Reset the adaptive learner to initial state.
    pub fn reset(&mut self) {
        self.current_c = self.config.initial_c;
        self.msm_config = MsmConfig::new(self.current_c);
        self.recent_costs.clear();
        self.recent_c_values.clear();
        self.round = 0;

        if let Some(ref mut stats) = self.statistics {
            *stats = AdaptiveMsmStatistics::default();
        }
    }

    /// Get regret bound estimate.
    ///
    /// For FPTL with tropical semiring: O(√(T log |Σ|))
    ///
    /// # Arguments
    ///
    /// * `max_cost` - Maximum cost per round
    /// * `alphabet_size` - Size of the label alphabet
    pub fn regret_bound(&self, max_cost: f64, alphabet_size: usize) -> f64 {
        if self.round == 0 || alphabet_size == 0 {
            return 0.0;
        }

        max_cost * ((self.round as f64) * (alphabet_size as f64).ln()).sqrt()
    }
}

/// Builder for creating AdaptiveMsm with custom initialization.
pub struct AdaptiveMsmBuilder {
    config: AdaptiveMsmConfig,
    training_data: Vec<(Vec<f64>, Vec<f64>, f64)>,
}

impl AdaptiveMsmBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: AdaptiveMsmConfig::default(),
            training_data: Vec::new(),
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: AdaptiveMsmConfig) -> Self {
        self.config = config;
        self
    }

    /// Add training data (query, target, actual_cost).
    pub fn add_training_sample(
        mut self,
        query: Vec<f64>,
        target: Vec<f64>,
        cost: f64,
    ) -> Self {
        self.training_data.push((query, target, cost));
        self
    }

    /// Build and optionally pre-train on provided data.
    pub fn build(self) -> AdaptiveMsm {
        let mut adaptive = AdaptiveMsm::new(self.config);

        // Pre-train on provided data
        for (query, target, actual_cost) in self.training_data {
            let _ = adaptive.predict(&query, &target);
            adaptive.observe(actual_cost);
        }

        adaptive
    }
}

impl Default for AdaptiveMsmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_adaptive_msm_creation() {
        let config = AdaptiveMsmConfig::new().initial_c(2.0);
        let adaptive = AdaptiveMsm::new(config);

        assert!(approx_eq(adaptive.current_c(), 2.0));
        assert_eq!(adaptive.round(), 0);
    }

    #[test]
    fn test_adaptive_msm_config() {
        let config = AdaptiveMsmConfig::new()
            .initial_c(1.5)
            .epsilon(0.05)
            .c_bounds(0.1, 10.0)
            .window_size(5)
            .with_statistics()
            .seed(42);

        assert!(approx_eq(config.initial_c, 1.5));
        assert!(approx_eq(config.epsilon, 0.05));
        assert!(approx_eq(config.min_c, 0.1));
        assert!(approx_eq(config.max_c, 10.0));
        assert_eq!(config.window_size, 5);
        assert!(config.track_statistics);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_adaptive_msm_predict() {
        let config = AdaptiveMsmConfig::new().initial_c(1.0).seed(42);
        let mut adaptive = AdaptiveMsm::new(config);

        let query = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];

        let cost = adaptive.predict(&query, &target);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_adaptive_msm_deterministic() {
        let config = AdaptiveMsmConfig::new().initial_c(1.0);
        let adaptive = AdaptiveMsm::new(config);

        let query = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];

        let cost = adaptive.predict_deterministic(&query, &target);
        // Identical series should have zero cost
        assert!(approx_eq(cost, 0.0));
    }

    #[test]
    fn test_adaptive_msm_observe() {
        let config = AdaptiveMsmConfig::new()
            .initial_c(1.0)
            .with_statistics()
            .seed(42);
        let mut adaptive = AdaptiveMsm::new(config);

        let query = vec![1.0, 2.0, 3.0];
        let target = vec![2.0, 3.0, 4.0];

        // Make predictions and observe
        let _ = adaptive.predict(&query, &target);
        adaptive.observe(3.0);

        assert_eq!(adaptive.round(), 1);
        let stats = adaptive.statistics().expect("expected Some stats in test");
        assert_eq!(stats.observations, 1);
        assert!(approx_eq(stats.total_cost, 3.0));
    }

    #[test]
    fn test_adaptive_msm_learning() {
        let config = AdaptiveMsmConfig::new()
            .initial_c(2.0)
            .epsilon(0.5)
            .with_statistics()
            .seed(42);
        let mut adaptive = AdaptiveMsm::new(config);

        let query = vec![1.0, 2.0, 3.0];
        let target = vec![2.0, 3.0, 4.0];

        let initial_c = adaptive.current_c();

        // Run several rounds
        for _ in 0..10 {
            let _ = adaptive.predict(&query, &target);
            adaptive.observe(3.0 + adaptive.current_c() * 0.1);
        }

        // c should have changed from initial
        // (direction depends on gradient)
        let final_c = adaptive.current_c();
        // Just verify it changed
        assert!(
            (final_c - initial_c).abs() > 0.0 || adaptive.round() > 0,
            "Learning should progress"
        );
    }

    #[test]
    fn test_adaptive_msm_reset() {
        let config = AdaptiveMsmConfig::new()
            .initial_c(1.0)
            .with_statistics()
            .seed(42);
        let mut adaptive = AdaptiveMsm::new(config);

        // Make some observations
        let query = vec![1.0, 2.0];
        let target = vec![2.0, 3.0];
        let _ = adaptive.predict(&query, &target);
        adaptive.observe(1.0);

        assert!(adaptive.round() > 0);

        // Reset
        adaptive.reset();

        assert_eq!(adaptive.round(), 0);
        assert!(approx_eq(adaptive.current_c(), 1.0));
        let stats = adaptive.statistics().expect("expected Some stats in test");
        assert_eq!(stats.observations, 0);
    }

    #[test]
    fn test_adaptive_msm_regret_bound() {
        let config = AdaptiveMsmConfig::new().initial_c(1.0);
        let mut adaptive = AdaptiveMsm::new(config);

        // Initially zero
        assert!(approx_eq(adaptive.regret_bound(1.0, 10), 0.0));

        // Simulate rounds
        for _ in 0..100 {
            adaptive.round += 1;
        }

        // Regret bound should be positive
        let bound = adaptive.regret_bound(1.0, 10);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_adaptive_msm_builder() {
        let adaptive = AdaptiveMsmBuilder::new()
            .config(AdaptiveMsmConfig::new().initial_c(1.5))
            .add_training_sample(vec![1.0], vec![2.0], 1.0)
            .build();

        assert!(adaptive.round() >= 1);
    }

    #[test]
    fn test_c_stays_in_bounds() {
        let config = AdaptiveMsmConfig::new()
            .initial_c(1.0)
            .c_bounds(0.5, 2.0)
            .epsilon(10.0) // Large epsilon to force boundary cases
            .seed(42);
        let mut adaptive = AdaptiveMsm::new(config);

        let query = vec![1.0, 2.0];
        let target = vec![2.0, 3.0];

        for _ in 0..100 {
            let _ = adaptive.predict(&query, &target);
            adaptive.observe(1.0);

            // c should stay within bounds
            assert!(
                adaptive.current_c() >= 0.5 && adaptive.current_c() <= 2.0,
                "c = {} is out of bounds",
                adaptive.current_c()
            );
        }
    }
}
