//! The ordered cost-monoid contract.

use std::cmp::Ordering;
use std::fmt::Debug;

/// A totally ordered, top-absorbing monoid for bounded dynamic programming.
///
/// A *cost* is an accumulated path value. `combine(a, w)` appends step cost
/// `w` to accumulated cost `a`; `compare` supplies the total order used by
/// pruning and priority queues; `within` tests a closed budget; and `select`
/// is fixed to the minimum under that same order.
///
/// # Laws
///
/// For lawful costs `a`, `b`, and `c`, and lawful non-negative step `w`:
///
/// 1. **L1 — monoid:** `combine` is associative and `ZERO` is a two-sided identity.
/// 2. **L2 — monotonicity:** `combine` is monotone in each argument.
/// 3. **L3 — totality:** `compare` is a total order.
/// 4. **L4 — positive order:** `compare(w, ZERO)` is not [`Ordering::Less`].
/// 5. **L5 — coherent choice:** `select(a, b)` is exactly the lesser operand.
/// 6. **L6 — downward closure:** if `a <= b` and `within(b, k)`, then
///    `within(a, k)`.
/// 7. **L7 — absorption:** combining either operand with `TOP` yields `TOP`.
///
/// L4 is a kernel obligation because a carrier cannot prevent a caller from
/// supplying a negative floating-point step. The lawful floating-point domain
/// excludes NaN and negative values. [`crate::cost::WeightedCost`] also carries
/// an explicit IEEE-754 trust boundary: L1 is exact for the real-number model
/// and exactly representable test domains, while arbitrary finite `f64`
/// evaluations are required to remain within the documented rounding envelope.
///
/// # Architectural boundary
///
/// This trait must not acquire a configurable choice operator, `star`,
/// `divide`, or `left_divide`. A computation requiring those operations is a
/// weighted finite-state-transducer problem rather than a bounded distance DP.
pub trait CostMonoid: 'static {
    /// Concrete cost carrier.
    type Cost: Copy + Debug + Send + Sync;

    /// Identity cost: no accumulated work.
    const ZERO: Self::Cost;

    /// Unreachable/infinite cost, absorbing under [`Self::combine`].
    const TOP: Self::Cost;

    /// Carrier-specific numerical proof/test envelope.
    ///
    /// Integer costs use zero. Floating-point carriers expose the rounding
    /// envelope used by analytical laws and differential assertions. This
    /// value must never expand an inclusive budget, establish canonical state
    /// equality, or justify product-state subsumption; [`Self::within`] remains
    /// exact under [`Self::compare`].
    const EPSILON: Self::Cost;

    /// Append `step` to `accumulated`.
    fn combine(accumulated: Self::Cost, step: Self::Cost) -> Self::Cost;

    /// Compare two costs using a total order.
    fn compare(a: Self::Cost, b: Self::Cost) -> Ordering;

    /// Return whether `cost` is within the inclusive `threshold`.
    fn within(cost: Self::Cost, threshold: Self::Cost) -> bool;

    /// Exact canonical key for a cost retained in a lazy product state.
    ///
    /// Returning `Some` opts the carrier into collision-checked state
    /// interning and observed-transition caching. Equal keys must imply equal
    /// future arithmetic for every lawful use of the cost. Returning `None`
    /// preserves exact execution but disables those two optimizations.
    /// Floating carriers must canonicalize signed zero and must never use an
    /// approximate or rounded key.
    #[doc(hidden)]
    #[inline]
    fn canonical_state_key(_cost: Self::Cost) -> Option<u64> {
        None
    }

    /// Choose the smaller cost under [`Self::compare`].
    ///
    /// Implementations must not override this method. Keeping the body here
    /// makes the choice/order relationship visible and testable in one place.
    #[inline(always)]
    fn select(a: Self::Cost, b: Self::Cost) -> Self::Cost {
        if Self::compare(a, b) == Ordering::Greater {
            b
        } else {
            a
        }
    }
}
