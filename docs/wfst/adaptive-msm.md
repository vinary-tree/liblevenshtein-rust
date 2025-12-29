# Adaptive MSM

The Adaptive MSM module implements the Follow-the-Perturbed-Tropical-Leader (FPTL) algorithm for adaptively learning the optimal `c` constant in Move-Split-Merge (MSM) time series metrics.

## Concepts

### The MSM Metric

The Move-Split-Merge (MSM) metric measures similarity between time series using three operations:

| Operation | Cost | Description |
|-----------|------|-------------|
| **Move** | \|a - b\| | Match a point in one series to a point in the other |
| **Split** | c | Split one point into two consecutive points |
| **Merge** | c | Merge two consecutive points into one |

The `c` constant controls the trade-off between matching points (move) versus inserting/deleting points (split/merge).

```
Series A:  [1.0, 2.0, 3.0, 4.0]
Series B:  [1.5, 2.5, 4.0]

Possible alignment (c = 1.0):
  A: 1.0 ─move─ 1.5   cost: 0.5
  A: 2.0 ─move─ 2.5   cost: 0.5
  A: 3.0 ─merge      cost: 1.0 (c)
  A: 4.0 ─move─ 4.0   cost: 0.0
                      ─────────
                Total: 2.0
```

### The c Constant Challenge

The optimal `c` depends on the data characteristics:

| c Value | Behavior | Best for |
|---------|----------|----------|
| Small c | Prefers split/merge | Series with insertions/deletions |
| Large c | Prefers move | Series with value differences |

**Problem**: The optimal `c` is unknown beforehand and may vary across domains.

### FPTL Algorithm

The Follow-the-Perturbed-Tropical-Leader (FPTL) algorithm learns the optimal `c` online:

```
FPTL for MSM:
    c ← initial_c

    for each (query, target) pair:
        1. Perturb: c_perturbed ← c + Laplacian_noise(ε)
        2. Predict: cost ← MSM(query, target, c_perturbed)
        3. Observe: actual_cost ← ground_truth
        4. Update: c ← c - ε × gradient_estimate

    return learned_c
```

The key innovation is **Laplacian perturbation**: adding random noise from the Laplace distribution enables exploration while maintaining theoretical regret bounds.

### Laplacian Perturbation

The Laplace distribution is used (not Gaussian) because it provides:

1. **Exploration**: Occasionally large perturbations discover better regions
2. **Concentration**: Most perturbations are small, maintaining stability
3. **Optimal regret**: Matches the theoretical lower bound for online learning

```
Laplace(0, 1/ε):

       ╱╲
      ╱  ╲
     ╱    ╲
    ╱      ╲
───╱────────╲───
         ↑
    Scale = 1/ε

Larger ε → Narrower peak → Less exploration
Smaller ε → Wider peak → More exploration
```

### Regret Bound

FPTL achieves regret bound:

```
O(√(T log |Σ|))
```

Where:
- **T**: Number of rounds
- **|Σ|**: Alphabet size (discretization of c space)

This is optimal for online learning with tropical (min-plus) losses.

## Core API

### Configuration

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsmConfig, AdaptiveMsm};

let config = AdaptiveMsmConfig::new()
    .initial_c(1.0)         // Starting c value
    .epsilon(0.1)           // Learning rate / perturbation scale
    .c_bounds(0.001, 100.0) // Min and max allowed c
    .window_size(10)        // Window for gradient estimation
    .with_statistics()      // Enable statistics tracking
    .seed(42);              // Random seed for reproducibility

let adaptive = AdaptiveMsm::new(config);
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_c` | 1.0 | Starting value for c |
| `epsilon` | 0.1 | Learning rate (also controls perturbation scale) |
| `min_c` | 0.001 | Minimum allowed c value |
| `max_c` | 100.0 | Maximum allowed c value |
| `window_size` | 10 | History window for gradient estimation |
| `track_statistics` | false | Enable detailed statistics |
| `seed` | None | Random seed for Laplacian noise |

### Main Methods

| Method | Description |
|--------|-------------|
| `predict(&query, &target)` | Compute MSM with perturbed c |
| `predict_deterministic(&query, &target)` | Compute MSM with current c (no perturbation) |
| `observe(actual_cost)` | Update c based on observed cost |
| `current_c()` | Get current c value |
| `msm_config()` | Get current MSM configuration |
| `reset()` | Reset to initial state |
| `round()` | Current round number |
| `statistics()` | Get learning statistics (if enabled) |
| `regret_bound(max_cost, alphabet_size)` | Theoretical regret bound |

### Statistics

```rust
if let Some(stats) = adaptive.statistics() {
    println!("Observations: {}", stats.observations);
    println!("Total cost: {}", stats.total_cost);
    println!("Average cost: {}", stats.average_cost);
    println!("Current gradient: {}", stats.current_gradient);
    println!("c history: {:?}", stats.c_history);
    println!("Cost history: {:?}", stats.cost_history);
}
```

## Examples

### Basic Adaptive Learning

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

// Configure adaptive MSM
let config = AdaptiveMsmConfig::new()
    .initial_c(1.0)
    .epsilon(0.1)
    .with_statistics()
    .seed(42);

let mut adaptive = AdaptiveMsm::new(config);

// Training data: (query, target, ground_truth_cost)
let training_data = vec![
    (vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5], 1.5),
    (vec![1.0, 3.0], vec![1.0, 2.0, 3.0], 0.8),
    (vec![2.0, 4.0, 6.0], vec![2.0, 5.0, 6.0], 1.0),
];

// Online learning loop
for (query, target, actual_cost) in &training_data {
    // Predict with current (perturbed) c
    let predicted_cost = adaptive.predict(query, target);

    // Observe actual cost and update
    adaptive.observe(*actual_cost);

    println!("Round {}: predicted={:.3}, actual={:.3}, c={:.4}",
        adaptive.round(), predicted_cost, actual_cost, adaptive.current_c());
}

println!("Learned c: {:.4}", adaptive.current_c());
```

### Deterministic Evaluation

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

let config = AdaptiveMsmConfig::new().initial_c(1.0);
let adaptive = AdaptiveMsm::new(config);

let query = vec![1.0, 2.0, 3.0];
let target = vec![1.0, 2.0, 3.0];

// Deterministic prediction (no perturbation)
let cost = adaptive.predict_deterministic(&query, &target);
println!("MSM distance: {}", cost);  // Should be 0.0 for identical series
```

### Learning c from Data

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

// Start with a suboptimal c
let config = AdaptiveMsmConfig::new()
    .initial_c(5.0)     // Too high - will be learned
    .epsilon(0.5)       // Larger epsilon for faster learning
    .c_bounds(0.1, 10.0)
    .with_statistics()
    .seed(42);

let mut adaptive = AdaptiveMsm::new(config);

println!("Initial c: {:.2}", adaptive.current_c());

// Simulate learning from data where optimal c ≈ 1.0
for i in 0..50 {
    let query = vec![1.0 + i as f64 * 0.1, 2.0, 3.0];
    let target = vec![1.0, 2.0, 3.0 + i as f64 * 0.1];

    // Ground truth uses c = 1.0
    let ground_truth = compute_msm_with_c(&query, &target, 1.0);

    let _ = adaptive.predict(&query, &target);
    adaptive.observe(ground_truth);
}

println!("Learned c: {:.2}", adaptive.current_c());

fn compute_msm_with_c(query: &[f64], target: &[f64], c: f64) -> f64 {
    // Simplified MSM computation
    use liblevenshtein::time_series::MsmConfig;
    MsmConfig::new(c).distance(query, target)
}
```

### Monitoring Convergence

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

let config = AdaptiveMsmConfig::new()
    .initial_c(1.0)
    .epsilon(0.1)
    .with_statistics();

let mut adaptive = AdaptiveMsm::new(config);

// Run learning...
for _ in 0..100 {
    let query = vec![1.0, 2.0];
    let target = vec![1.5, 2.5];
    let _ = adaptive.predict(&query, &target);
    adaptive.observe(1.0);
}

// Analyze convergence
if let Some(stats) = adaptive.statistics() {
    println!("c values over time:");
    for (i, c) in stats.c_history.iter().enumerate().step_by(10) {
        println!("  Round {}: c = {:.4}", i, c);
    }

    println!("\nGradient: {:.6}", stats.current_gradient);
    println!("Average cost: {:.4}", stats.average_cost);
}
```

### Controlling c Bounds

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

// Constrain c to a narrow range
let config = AdaptiveMsmConfig::new()
    .initial_c(1.0)
    .c_bounds(0.5, 2.0)   // c stays in [0.5, 2.0]
    .epsilon(10.0)        // Very large perturbation
    .seed(42);

let mut adaptive = AdaptiveMsm::new(config);

// Even with extreme perturbations, c stays bounded
for _ in 0..100 {
    let query = vec![1.0, 2.0];
    let target = vec![2.0, 3.0];
    let _ = adaptive.predict(&query, &target);
    adaptive.observe(1.0);

    let c = adaptive.current_c();
    assert!(c >= 0.5 && c <= 2.0, "c out of bounds: {}", c);
}

println!("c remained within bounds");
```

### Computing Regret Bound

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

let mut adaptive = AdaptiveMsm::new(AdaptiveMsmConfig::default());

// Simulate 100 rounds
for _ in 0..100 {
    adaptive.round += 1;  // (normally incremented by observe())
}

let max_cost = 10.0;      // Maximum MSM cost per round
let alphabet_size = 100;  // Discretization of c space

let bound = adaptive.regret_bound(max_cost, alphabet_size);
println!("Regret bound after {} rounds: {:.2}", adaptive.round(), bound);

// Bound is: max_cost × √(T × ln(|Σ|))
// = 10.0 × √(100 × ln(100)) ≈ 48.0
```

### Using the Builder

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsmBuilder, AdaptiveMsmConfig};

// Build with pre-training data
let adaptive = AdaptiveMsmBuilder::new()
    .config(
        AdaptiveMsmConfig::new()
            .initial_c(1.0)
            .epsilon(0.1)
    )
    .add_training_sample(vec![1.0, 2.0], vec![1.5, 2.5], 1.0)
    .add_training_sample(vec![2.0, 3.0], vec![2.0, 4.0], 1.5)
    .add_training_sample(vec![1.0, 3.0], vec![1.0, 2.0, 3.0], 0.8)
    .build();

println!("Pre-trained c: {:.4}", adaptive.current_c());
println!("Rounds completed: {}", adaptive.round());
```

### Resetting the Learner

```rust
use liblevenshtein::wfst::msm::{AdaptiveMsm, AdaptiveMsmConfig};

let config = AdaptiveMsmConfig::new()
    .initial_c(1.0)
    .with_statistics();

let mut adaptive = AdaptiveMsm::new(config);

// Run some rounds...
for _ in 0..50 {
    let _ = adaptive.predict(&[1.0, 2.0], &[2.0, 3.0]);
    adaptive.observe(1.0);
}

println!("Before reset: c={:.4}, rounds={}", adaptive.current_c(), adaptive.round());

// Reset to initial state
adaptive.reset();

println!("After reset: c={:.4}, rounds={}", adaptive.current_c(), adaptive.round());
// c is back to 1.0, rounds is 0
```

## Algorithm Details

### Gradient Estimation

The gradient ∂Cost/∂c is estimated using finite differences over the observation window:

```
gradient ≈ (cost[i] - cost[i-1]) / (c[i] - c[i-1])
```

This is averaged over the window to reduce noise.

### Update Rule

After gradient estimation, c is updated via gradient descent:

```
c_new = c_old - ε × gradient
c_new = clamp(c_new, min_c, max_c)
```

### Laplacian Sampling

Laplacian noise is sampled using the inverse CDF method:

```rust
// Sample U ~ Uniform(0, 1)
let u: f64 = rng.random();

// Transform to Laplace(0, scale)
let noise = if u < 0.5 {
    scale * (2.0 * u).ln()         // Negative part
} else {
    -scale * (2.0 * (1.0 - u)).ln() // Positive part
};
```

## When to Use Adaptive MSM

**Choose AdaptiveMsm when:**

| Scenario | Why? |
|----------|------|
| Unknown optimal c | Learn from data |
| Varying data characteristics | Adapt to distribution shifts |
| Online time series matching | Incremental updates |
| Multiple domains | Learn domain-specific c |

**Use fixed MsmConfig when:**

| Scenario | Why? |
|----------|------|
| Optimal c is known | No need for learning overhead |
| Very few samples | Insufficient data to learn |
| Batch processing | Can grid-search c offline |
| Reproducibility critical | Perturbation adds randomness |

## Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| `predict()` | O(nm) for MSM on n×m series |
| `observe()` | O(window_size) for gradient |
| Space | O(window_size) for history |

## References

- Cortes, C., Kuznetsov, V., Mohri, M., & Warmuth, M. K. (2015). "On-Line Learning Algorithms for Path Experts with Non-Additive Losses". JMLR 16, 2015. (Appendix B defines FPTL)
- Stefan, A., Athitsos, V., & Das, G. (2013). "The Move-Split-Merge Metric for Time Series". IEEE TKDE.

## Related Documentation

- [MSM Metric](../time_series/msm.md) - The underlying MSM distance computation
- [Time Series WFST](architecture.md) - WFST representation of time series metrics
- [MSM-WFST Integration](../integration/msm-wfst.md) - Using MSM with WFST pipelines
