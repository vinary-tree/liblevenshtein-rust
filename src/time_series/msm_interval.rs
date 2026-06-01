//! Interval-relaxed Move-Split-Merge transitions for exact MSM-over-trie search.
//!
//! # Why this module exists
//!
//! [`crate::time_series::MsmTransducer`] walks a trie of *quantized* reference
//! series, computing the MSM dynamic-programming column for the query against
//! the (shared) trie path **once per prefix**. Along a trie branch the target
//! sequence is fixed, but each target element is only known up to its
//! quantization *bin* — an interval `[lo, hi]` rather than a scalar. To prune
//! soundly we therefore evaluate the MSM recurrence with each per-element cost
//! replaced by an **admissible lower bound**: the minimum the true cost could
//! take for *any* concrete value(s) inside the bin interval(s).
//!
//! The MSM recurrence (Stefan et al. 2012; see [`crate::time_series::MsmConfig`])
//! with query `x` (rows `i`) and target `y` (columns `j`) is
//!
//! ```text
//! cost[i][j] = min{
//!   cost[i-1][j-1] + |x_i - y_j|,            // Move
//!   cost[i-1][j]   + C(x_i, x_{i-1}, y_j),   // Merge-like
//!   cost[i][j-1]   + C(y_j, x_i, y_{j-1}),   // Split-like
//! }
//! C(a,b,c) = c_const                       if b ≤ a ≤ c or b ≥ a ≥ c
//!          = c_const + min(|a-b|, |a-c|)    otherwise
//! ```
//!
//! Here `x` is the (full-precision) query and `y` is the trie path, so the
//! *free* (interval-valued) arguments are exactly the target values `y_*`. We
//! derive, in closed form, the minimum of each cost term over those intervals.
//!
//! # The three admissible lower bounds
//!
//! Let `interval_dist(v, lo, hi) = max(0, lo − v, v − hi)` be the distance from
//! a scalar `v` to the interval `[lo, hi]` (0 iff `v ∈ [lo, hi]`).
//!
//! * **Move** `|x_i − y_j|`, `y_j ∈ [lo, hi]`: minimized at `y_j = clamp(x_i,
//!   lo, hi)`, giving `interval_dist(x_i, lo, hi)`.
//!
//! * **Merge** `C(x_i, x_{i-1}, y_j)` with scalars `a = x_i`, `b = x_{i-1}` and
//!   free `c = y_j ∈ [lo, hi]` ([`c_func_merge_lb`]). `C` is `c_const` plus a
//!   non-negative penalty that is `0` whenever `a` lies between `b` and `c`.
//!   Such a `c` exists in `[lo, hi]` iff `(a ≥ b ∧ hi ≥ a)` or `(a ≤ b ∧ lo ≤
//!   a)`; then the bound is `c_const`. Otherwise the penalty is
//!   `min(|a-b|, interval_dist(a, lo, hi))` (the `|a-c|` term is minimized by
//!   `c = clamp(a, lo, hi)`).
//!
//! * **Split** `C(y_j, x_i, y_{j-1})` with free `a = y_j ∈ [a_lo, a_hi]`, scalar
//!   `b = x_i`, free `c = y_{j-1} ∈ [c_lo, c_hi]` ([`c_func_split_lb`]). The
//!   union over `c ∈ [c_lo, c_hi]` of the "between" intervals `[min(b,c),
//!   max(b,c)]` is the contiguous `[min(b, c_lo), max(b, c_hi)]` (each contains
//!   `b`). A penalty of `0` is achievable iff `[a_lo, a_hi]` intersects that
//!   union. Otherwise — `a` is forced strictly above or strictly below both `b`
//!   and every `c` — the minimum penalty is the (exact) gap to the union:
//!   `a_lo − max(b, c_hi)` when `a` is above, or `min(b, c_lo) − a_hi` when
//!   below.
//!
//! Each bound is the *exact* minimum over its interval box, hence the tightest
//! admissible lower bound; summing/min-combining them through the DP yields a
//! column that lower-bounds the true column for every concrete reference whose
//! quantization matches the trie path. See [`step_interval_column`] and the
//! admissibility property tests below.
//!
//! Infinities are handled deliberately: the extreme quantization bins extend to
//! ±∞ (see [`crate::time_series::QuantizationConfig::bin_bounds`]). Every branch
//! that performs subtraction is reachable only when the relevant endpoints are
//! finite (an ±∞ endpoint always routes to the penalty-`0` / overlap branch), so
//! no `∞ − ∞` is ever evaluated.

/// Epsilon for float comparisons (mirrors `msm_transition::COST_EPSILON`).
pub const COST_EPSILON: f64 = 1e-9;

/// Distance from a scalar `v` to the closed interval `[lo, hi]`.
///
/// Returns `0.0` when `v ∈ [lo, hi]`, otherwise the distance to the nearer
/// endpoint. This is the admissible lower bound on the MSM **Move** cost
/// `|v − y|` for any `y ∈ [lo, hi]`. Safe with infinite endpoints.
#[inline]
pub fn interval_dist(v: f64, lo: f64, hi: f64) -> f64 {
    (lo - v).max(0.0).max(v - hi)
}

/// Admissible lower bound on the MSM `C(a, b, c)` **Merge** cost when `a` and
/// `b` are scalars and `c` ranges over the bin interval `[lo, hi]`.
///
/// Equals `min_{c ∈ [lo, hi]} C(a, b, c)` exactly. See the module docs for the
/// derivation. `c_const` is `MsmConfig::c`.
#[inline]
pub fn c_func_merge_lb(a: f64, b: f64, lo: f64, hi: f64, c_const: f64) -> f64 {
    // A penalty of 0 is achievable iff some c ∈ [lo, hi] places `a` between
    // `b` and `c`: either a ≥ b with c ≥ a (needs hi ≥ a), or a ≤ b with c ≤ a
    // (needs lo ≤ a).
    let penalty = if (a >= b && hi >= a) || (a <= b && lo <= a) {
        0.0
    } else {
        // |a - b| is fixed; |a - c| is minimized at c = clamp(a, lo, hi).
        (a - b).abs().min(interval_dist(a, lo, hi))
    };
    c_const + penalty
}

/// Admissible lower bound on the MSM `C(a, b, c)` **Split** cost when `b` is a
/// scalar (`= x_i`), `a` ranges over the current target bin `[a_lo, a_hi]`, and
/// `c` ranges over the previous target bin `[c_lo, c_hi]`.
///
/// Equals `min_{a ∈ [a_lo,a_hi], c ∈ [c_lo,c_hi]} C(a, b, c)` exactly. See the
/// module docs for the derivation.
#[inline]
pub fn c_func_split_lb(a_lo: f64, a_hi: f64, b: f64, c_lo: f64, c_hi: f64, c_const: f64) -> f64 {
    // Union over c ∈ [c_lo, c_hi] of the "a between b and c" intervals.
    let union_lo = b.min(c_lo);
    let union_hi = b.max(c_hi);
    let penalty = if a_lo <= union_hi && a_hi >= union_lo {
        // [a_lo, a_hi] meets the union ⇒ some (a, c) gives penalty 0.
        0.0
    } else if a_lo > union_hi {
        // `a` is forced strictly above both b and every c.
        a_lo - union_hi
    } else {
        // a_hi < union_lo: `a` is forced strictly below both b and every c.
        union_lo - a_hi
    };
    c_const + penalty
}

/// Compute the next interval-relaxed MSM DP column.
///
/// Given the column `prev_col` for the query against the first `depth − 1`
/// target elements, extend it by consuming one more target element whose value
/// lies in `curr = [lo, hi]`. `prev` is the interval of the *preceding* target
/// element (`y_{j-2}`), or `None` when this is the first target element
/// (`depth == 1`), in which case `prev_col` is ignored and the MSM base /
/// first-column rules apply.
///
/// The returned column has length `query.len() + 1`; entry `i` (1-indexed,
/// entry `0` unused and held at `+∞`) is an admissible lower bound on
/// `cost[i][depth]` for every concrete reference consistent with the bins seen
/// so far. `+∞` marks an unreachable cell.
///
/// # Panics
///
/// Panics if `query` is empty (the transducer guards this before calling).
pub fn step_interval_column(
    prev_col: &[f64],
    query: &[f64],
    curr: (f64, f64),
    prev: Option<(f64, f64)>,
    c_const: f64,
) -> Vec<f64> {
    let m = query.len();
    assert!(m > 0, "step_interval_column requires a non-empty query");
    let (clo, chi) = curr;
    let mut col = vec![f64::INFINITY; m + 1];

    match prev {
        // depth == 1: the first target element. Base case + first column.
        None => {
            // cost[1][1] = |x_0 - y_0|  →  interval_dist over the current bin.
            col[1] = interval_dist(query[0], clo, chi);
            // cost[i][1] = cost[i-1][1] + C(x_{i-1}, x_{i-2}, y_0)   (Merge over curr)
            for i in 2..=m {
                if col[i - 1].is_finite() {
                    col[i] =
                        col[i - 1] + c_func_merge_lb(query[i - 1], query[i - 2], clo, chi, c_const);
                }
            }
        }
        // depth >= 2: general column transition from prev_col.
        Some((plo, phi)) => {
            // First row: cost[1][j] = cost[1][j-1] + C(y_{j-1}, x_0, y_{j-2})  (Split)
            if prev_col[1].is_finite() {
                col[1] = prev_col[1] + c_func_split_lb(clo, chi, query[0], plo, phi, c_const);
            }
            for i in 2..=m {
                // Move: cost[i-1][j-1] + |x_{i-1} - y_{j-1}|
                let mv = if prev_col[i - 1].is_finite() {
                    prev_col[i - 1] + interval_dist(query[i - 1], clo, chi)
                } else {
                    f64::INFINITY
                };
                // Merge: cost[i-1][j] + C(x_{i-1}, x_{i-2}, y_{j-1})   (c = curr)
                let mg = if col[i - 1].is_finite() {
                    col[i - 1] + c_func_merge_lb(query[i - 1], query[i - 2], clo, chi, c_const)
                } else {
                    f64::INFINITY
                };
                // Split: cost[i][j-1] + C(y_{j-1}, x_{i-1}, y_{j-2})   (a = curr, c = prev)
                let sp = if prev_col[i].is_finite() {
                    prev_col[i] + c_func_split_lb(clo, chi, query[i - 1], plo, phi, c_const)
                } else {
                    f64::INFINITY
                };
                col[i] = mv.min(mg).min(sp);
            }
        }
    }

    col
}

/// The admissible subtree lower bound carried by a trie node holding `col`:
/// the minimum live cell. Every final reachable below this node (at any deeper
/// depth) has true MSM distance ≥ this value, because any DP path to a deeper
/// final must cross the current column at some row `i`, and all subsequent
/// MSM operation costs are non-negative. Hence pruning a subtree whose bound
/// exceeds the threshold can never drop a true match.
#[inline]
pub fn column_lower_bound(col: &[f64]) -> f64 {
    // Skip the unused index-0 sentinel.
    col[1..]
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, v| acc.min(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::MsmConfig;

    // ---- closed-form bounds vs the scalar C() function -------------------

    #[test]
    fn interval_dist_basic() {
        assert_eq!(interval_dist(5.0, 0.0, 10.0), 0.0); // inside
        assert_eq!(interval_dist(-2.0, 0.0, 10.0), 2.0); // below
        assert_eq!(interval_dist(13.0, 0.0, 10.0), 3.0); // above
        assert_eq!(interval_dist(5.0, f64::NEG_INFINITY, 10.0), 0.0);
        assert_eq!(interval_dist(20.0, 0.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn merge_lb_is_min_over_c() {
        let config = MsmConfig::new(1.3);
        // Brute-force the minimum of C(a,b,c) over a grid of c in [lo,hi] and
        // confirm the closed form matches (to grid resolution).
        for &(a, b, lo, hi) in &[
            (2.0_f64, 1.0_f64, 0.0_f64, 5.0_f64),
            (2.0, 5.0, 0.0, 5.0),
            (-3.0, 1.0, 0.0, 5.0),
            (9.0, 1.0, 0.0, 5.0),
            (3.0, 3.0, 1.0, 2.0),
            (0.5, 0.0, 1.0, 4.0),
        ] {
            let lb = c_func_merge_lb(a, b, lo, hi, config.c);
            let mut brute = f64::INFINITY;
            let steps = 4000;
            for s in 0..=steps {
                let c = lo + (hi - lo) * (s as f64 / steps as f64);
                brute = brute.min(config.c_func(a, b, c));
            }
            assert!(
                lb <= brute + 1e-6 && lb >= brute - 1e-2,
                "merge a={a} b={b} [{lo},{hi}]: lb={lb} brute={brute}"
            );
        }
    }

    #[test]
    fn split_lb_is_min_over_box() {
        let config = MsmConfig::new(0.7);
        for &(alo, ahi, b, clo, chi) in &[
            (0.0_f64, 5.0_f64, 2.0_f64, 0.0_f64, 5.0_f64),
            (6.0, 9.0, 2.0, 0.0, 4.0),   // a forced above
            (-5.0, -2.0, 2.0, 0.0, 4.0), // a forced below
            (1.0, 1.5, 3.0, 4.0, 6.0),
            (10.0, 12.0, 1.0, 2.0, 3.0),
        ] {
            let lb = c_func_split_lb(alo, ahi, b, clo, chi, config.c);
            let mut brute = f64::INFINITY;
            let steps = 200;
            for sa in 0..=steps {
                let a = alo + (ahi - alo) * (sa as f64 / steps as f64);
                for sc in 0..=steps {
                    let c = clo + (chi - clo) * (sc as f64 / steps as f64);
                    brute = brute.min(config.c_func(a, b, c));
                }
            }
            assert!(
                lb <= brute + 1e-6 && lb >= brute - 1e-2,
                "split [{alo},{ahi}] b={b} [{clo},{chi}]: lb={lb} brute={brute}"
            );
        }
    }

    // ---- the column is an admissible lower bound on the true DP ----------

    /// For a quantization-free setting (degenerate bins `[v, v]`), the
    /// interval column must reproduce the exact scalar MSM DP column.
    #[test]
    fn degenerate_bins_reproduce_scalar_dp() {
        let config = MsmConfig::new(1.0);
        let query = vec![1.0, 3.0, 2.0, 5.0];
        let target = vec![1.0, 2.5, 4.0];

        // Build the column incrementally with point-intervals [y, y].
        let m = query.len();
        let mut col = vec![f64::INFINITY; m + 1];
        let mut prev_interval: Option<(f64, f64)> = None;
        for (j, &y) in target.iter().enumerate() {
            col = step_interval_column(&col, &query, (y, y), prev_interval, config.c);
            prev_interval = Some((y, y));
            let _ = j;
        }
        // cost[m][n] from the reference DP.
        let exact = config.distance(&query, &target);
        assert!(
            (col[m] - exact).abs() < 1e-9,
            "interval col[m]={} != exact DP {}",
            col[m],
            exact
        );
    }

    /// With genuine (non-degenerate) bins the column must *lower-bound* the
    /// true distance to any concrete series whose values lie in those bins.
    #[test]
    fn interval_column_lower_bounds_concrete() {
        let config = MsmConfig::new(1.0);
        let query = vec![0.4, 2.1, 3.9, 1.2];
        // Bins (intervals) the reference is known to lie within.
        let bins = [(0.0, 1.0), (2.0, 3.0), (3.0, 4.0)];
        // Concrete realizations inside those bins.
        let concretes = [
            vec![0.0, 2.0, 3.0],
            vec![1.0, 3.0, 4.0],
            vec![0.5, 2.5, 3.5],
            vec![0.9, 2.1, 3.9],
        ];

        let m = query.len();
        let mut col = vec![f64::INFINITY; m + 1];
        let mut prev_interval: Option<(f64, f64)> = None;
        for &(lo, hi) in &bins {
            col = step_interval_column(&col, &query, (lo, hi), prev_interval, config.c);
            prev_interval = Some((lo, hi));
        }
        let lb_final = col[m];
        let lb_subtree = column_lower_bound(&col);
        for c in &concretes {
            let exact = config.distance(&query, c);
            assert!(
                lb_final <= exact + 1e-9,
                "col[m] LB {lb_final} exceeded exact {exact} for {c:?}"
            );
            assert!(
                lb_subtree <= exact + 1e-9,
                "subtree LB {lb_subtree} exceeded exact {exact} for {c:?}"
            );
        }
    }
}
