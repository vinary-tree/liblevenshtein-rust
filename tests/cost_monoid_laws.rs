//! Executable laws for the ordered cost monoids and exact cost scale.

use liblevenshtein::cost::{BottleneckCost, CostMonoid, CostScale, UnitCost, WeightedCost};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn regression_config(cases: u32) -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::WithSource(
        "proptest-regressions",
    ));
    config.cases = cases;
    config
}
use std::cmp::Ordering;

fn finite_cost() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(|bits| f64::from_bits(bits & i64::MAX as u64))
        .prop_filter("lawful non-negative finite WeightedCost", |value| {
            value.is_finite()
        })
}

fn dyadic_cost() -> impl Strategy<Value = f64> {
    (0_u32..=1_000_000).prop_map(|value| f64::from(value) / 1_024.0)
}

proptest! {
    #![proptest_config(regression_config(2_000))]

    #[test]
    fn unit_cost_l1_associativity_and_identity(a in any::<usize>(), b in any::<usize>(), c in any::<usize>()) {
        let left = UnitCost::combine(UnitCost::combine(a, b), c);
        let right = UnitCost::combine(a, UnitCost::combine(b, c));
        prop_assert_eq!(left, right);
        prop_assert_eq!(UnitCost::combine(UnitCost::ZERO, a), a);
        prop_assert_eq!(UnitCost::combine(a, UnitCost::ZERO), a);
    }

    #[test]
    fn unit_cost_l2_through_l7(a in 0_usize..1_000_000, b in 0_usize..1_000_000, step in 0_usize..1_000_000, threshold in 0_usize..2_000_000) {
        let (smaller, larger) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(UnitCost::compare(UnitCost::combine(smaller, step), UnitCost::combine(larger, step)) != Ordering::Greater);
        prop_assert!(UnitCost::compare(UnitCost::combine(step, smaller), UnitCost::combine(step, larger)) != Ordering::Greater);
        prop_assert_eq!(UnitCost::compare(a, b), a.cmp(&b));
        prop_assert!(UnitCost::compare(step, UnitCost::ZERO) != Ordering::Less);
        prop_assert_eq!(UnitCost::select(a, b), a.min(b));
        if UnitCost::within(larger, threshold) {
            prop_assert!(UnitCost::within(smaller, threshold));
        }
        prop_assert_eq!(UnitCost::combine(a, UnitCost::TOP), UnitCost::TOP);
        prop_assert_eq!(UnitCost::combine(UnitCost::TOP, a), UnitCost::TOP);
    }

    #[test]
    fn weighted_cost_l1_is_exact_on_dyadic_inputs(a in dyadic_cost(), b in dyadic_cost(), c in dyadic_cost()) {
        let left = WeightedCost::combine(WeightedCost::combine(a, b), c);
        let right = WeightedCost::combine(a, WeightedCost::combine(b, c));
        prop_assert_eq!(left.to_bits(), right.to_bits());
        prop_assert_eq!(WeightedCost::combine(WeightedCost::ZERO, a).to_bits(), a.to_bits());
        prop_assert_eq!(WeightedCost::combine(a, WeightedCost::ZERO).to_bits(), a.to_bits());
    }

    #[test]
    fn weighted_cost_l1_general_inputs_obey_rounding_envelope(a in finite_cost(), b in finite_cost(), c in finite_cost()) {
        let left = WeightedCost::combine(WeightedCost::combine(a, b), c);
        let right = WeightedCost::combine(a, WeightedCost::combine(b, c));
        prop_assume!(left.is_finite() && right.is_finite());
        let envelope = 4.0 * f64::EPSILON * (a + b + c).max(1.0);
        prop_assert!((left - right).abs() <= envelope, "left={left:?}, right={right:?}, envelope={envelope:?}");
    }

    #[test]
    fn weighted_cost_l2_through_l7(a in finite_cost(), b in finite_cost(), step in finite_cost(), threshold in finite_cost()) {
        let (smaller, larger) = if a.total_cmp(&b) != Ordering::Greater { (a, b) } else { (b, a) };
        prop_assert!(WeightedCost::compare(WeightedCost::combine(smaller, step), WeightedCost::combine(larger, step)) != Ordering::Greater);
        prop_assert!(WeightedCost::compare(WeightedCost::combine(step, smaller), WeightedCost::combine(step, larger)) != Ordering::Greater);
        prop_assert_eq!(WeightedCost::compare(a, b), a.total_cmp(&b));
        prop_assert!(WeightedCost::compare(step, WeightedCost::ZERO) != Ordering::Less);
        prop_assert_eq!(WeightedCost::select(a, b).to_bits(), smaller.to_bits());
        if WeightedCost::within(larger, threshold) {
            prop_assert!(WeightedCost::within(smaller, threshold));
        }
        prop_assert_eq!(WeightedCost::combine(a, WeightedCost::TOP), WeightedCost::TOP);
        prop_assert_eq!(WeightedCost::combine(WeightedCost::TOP, a), WeightedCost::TOP);
    }

    #[test]
    fn bottleneck_cost_l1_through_l7(a in finite_cost(), b in finite_cost(), c in finite_cost(), threshold in finite_cost()) {
        let left = BottleneckCost::combine(BottleneckCost::combine(a, b), c);
        let right = BottleneckCost::combine(a, BottleneckCost::combine(b, c));
        prop_assert_eq!(left.to_bits(), right.to_bits());
        prop_assert_eq!(BottleneckCost::combine(BottleneckCost::ZERO, a).to_bits(), a.to_bits());
        prop_assert_eq!(BottleneckCost::combine(a, BottleneckCost::ZERO).to_bits(), a.to_bits());

        let (smaller, larger) = if a.total_cmp(&b) != Ordering::Greater { (a, b) } else { (b, a) };
        prop_assert!(BottleneckCost::compare(BottleneckCost::combine(smaller, c), BottleneckCost::combine(larger, c)) != Ordering::Greater);
        prop_assert!(BottleneckCost::compare(BottleneckCost::combine(c, smaller), BottleneckCost::combine(c, larger)) != Ordering::Greater);
        prop_assert_eq!(BottleneckCost::compare(a, b), a.total_cmp(&b));
        prop_assert!(BottleneckCost::compare(c, BottleneckCost::ZERO) != Ordering::Less);
        prop_assert_eq!(BottleneckCost::select(a, b).to_bits(), smaller.to_bits());
        if BottleneckCost::within(larger, threshold) {
            prop_assert!(BottleneckCost::within(smaller, threshold));
        }
        prop_assert_eq!(BottleneckCost::combine(a, BottleneckCost::TOP), BottleneckCost::TOP);
        prop_assert_eq!(BottleneckCost::combine(BottleneckCost::TOP, a), BottleneckCost::TOP);
    }

    #[test]
    fn decimal_cost_scale_round_trips_thousandths(numerator in 0_usize..=1_000_000) {
        let scale = CostScale::default();
        let weight = numerator as f64 / 1_000.0;
        prop_assert_eq!(scale.to_scaled(weight), Ok(numerator));
        prop_assert!((scale.from_scaled(numerator) - weight).abs() <= f64::EPSILON * weight.max(1.0));
    }
}

#[test]
fn prelude_reexports_cost_surface() {
    use liblevenshtein::prelude::*;

    assert_eq!(UnitCost::combine(2, 3), 5);
    assert_eq!(WeightedCost::select(0.25, 0.5), 0.25);
    assert_eq!(BottleneckCost::combine(0.25, 0.5), 0.5);
    assert_eq!(CostScale::default().to_scaled(0.15), Ok(150));
}

#[test]
fn floating_domains_reject_nan_at_budget_boundary_and_absorb_top() {
    assert!(!WeightedCost::within(f64::NAN, 1.0));
    assert!(!WeightedCost::within(1.0, f64::NAN));
    assert!(!BottleneckCost::within(f64::NAN, 1.0));
    assert_eq!(
        WeightedCost::combine(f64::INFINITY, f64::NEG_INFINITY),
        WeightedCost::TOP
    );
    assert_eq!(
        BottleneckCost::combine(f64::INFINITY, f64::NAN),
        BottleneckCost::TOP
    );
}

#[test]
fn floating_cost_membership_is_exactly_inclusive() {
    let cutoff = 1.0_f64;
    let immediately_above = cutoff.next_up();

    assert!(WeightedCost::within(cutoff, cutoff));
    assert!(!WeightedCost::within(immediately_above, cutoff));
    assert!(BottleneckCost::within(cutoff, cutoff));
    assert!(!BottleneckCost::within(immediately_above, cutoff));
}
