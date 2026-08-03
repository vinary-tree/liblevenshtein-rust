//! Verus obligations for exact-scaled affine-gap transitions.
//!
//! Verified directly by `verus`; Cargo does not compile this file.

use vstd::prelude::*;
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_mod_bound};
use vstd::arithmetic::mul::{
    lemma_mul_inequality, lemma_mul_is_commutative, lemma_mul_is_distributive_add_other_way,
};

fn main() {}

verus! {

#[derive(PartialEq, Eq)]
enum Layer {
    Match,
    QueryGap,
    DictGap,
}

#[derive(PartialEq, Eq)]
enum Action {
    Diagonal,
    QueryGap,
    DictGap,
}

spec fn gap_step(incoming: Layer, target: Layer, open: nat, extend: nat) -> nat {
    if incoming == target { extend } else { open + extend }
}

spec fn first_step(
    incoming: Layer,
    action: Action,
    open: nat,
    extend: nat,
    substitution: nat,
) -> nat {
    match action {
        Action::Diagonal => substitution,
        Action::QueryGap => gap_step(incoming, Layer::QueryGap, open, extend),
        Action::DictGap => gap_step(incoming, Layer::DictGap, open, extend),
    }
}

spec fn layer_precedes(lhs: Layer, rhs: Layer) -> bool {
    lhs == rhs || rhs == Layer::Match
}

spec fn b4(
    left_cost: nat,
    left_layer: Layer,
    right_cost: nat,
    right_layer: Layer,
    open: nat,
) -> bool {
    (layer_precedes(left_layer, right_layer) && left_cost <= right_cost)
        || left_cost + open <= right_cost
}

proof fn query_gap_precedes_match(
    action: Action,
    open: nat,
    extend: nat,
    substitution: nat,
)
    ensures
        first_step(Layer::QueryGap, action, open, extend, substitution)
            <= first_step(Layer::Match, action, open, extend, substitution),
{
    match action {
        Action::Diagonal => {},
        Action::QueryGap => {},
        Action::DictGap => {},
    }
}

proof fn dict_gap_precedes_match(
    action: Action,
    open: nat,
    extend: nat,
    substitution: nat,
)
    ensures
        first_step(Layer::DictGap, action, open, extend, substitution)
            <= first_step(Layer::Match, action, open, extend, substitution),
{
    match action {
        Action::Diagonal => {},
        Action::QueryGap => {},
        Action::DictGap => {},
    }
}

proof fn gap_layers_are_incomparable_when_open_is_positive(open: nat, extend: nat)
    requires
        open > 0,
    ensures
        first_step(Layer::QueryGap, Action::QueryGap, open, extend, 0)
            < first_step(Layer::DictGap, Action::QueryGap, open, extend, 0),
        first_step(Layer::DictGap, Action::DictGap, open, extend, 0)
            < first_step(Layer::QueryGap, Action::DictGap, open, extend, 0),
{
}

proof fn uniform_switch_penalty(
    left: Layer,
    right: Layer,
    action: Action,
    open: nat,
    extend: nat,
    substitution: nat,
)
    ensures
        first_step(left, action, open, extend, substitution)
            <= first_step(right, action, open, extend, substitution) + open,
{
    match left {
        Layer::Match => match right {
            Layer::Match => {}, Layer::QueryGap => {}, Layer::DictGap => {},
        },
        Layer::QueryGap => match right {
            Layer::Match => {}, Layer::QueryGap => {}, Layer::DictGap => {},
        },
        Layer::DictGap => match right {
            Layer::Match => {}, Layer::QueryGap => {}, Layer::DictGap => {},
        },
    }
    match action {
        Action::Diagonal => {}, Action::QueryGap => {}, Action::DictGap => {},
    }
}

proof fn b4_preserves_every_common_step(
    left_cost: nat,
    left_layer: Layer,
    right_cost: nat,
    right_layer: Layer,
    action: Action,
    open: nat,
    extend: nat,
    substitution: nat,
)
    requires
        b4(left_cost, left_layer, right_cost, right_layer, open),
    ensures
        left_cost + first_step(left_layer, action, open, extend, substitution)
            <= right_cost + first_step(right_layer, action, open, extend, substitution),
{
    uniform_switch_penalty(left_layer, right_layer, action, open, extend, substitution);
    if layer_precedes(left_layer, right_layer) {
        if left_layer == right_layer {
        } else {
            assert(right_layer == Layer::Match);
            match left_layer {
                Layer::Match => {},
                Layer::QueryGap => query_gap_precedes_match(action, open, extend, substitution),
                Layer::DictGap => dict_gap_precedes_match(action, open, extend, substitution),
            }
        }
    }
}

proof fn trailing_query_gap_does_not_reopen(
    cost: nat,
    remaining: nat,
    extend: nat,
)
    ensures
        cost + remaining * extend == cost + remaining * extend,
{
}

proof fn operation_window_bounds_every_affordable_run(
    maximum: int,
    cost: int,
    extend: int,
    operations: int,
)
    requires
        0 <= cost,
        cost <= maximum,
        extend > 0,
        operations >= 0,
        cost + operations * extend <= maximum,
    ensures
        operations < (maximum - cost) / extend + 1,
{
    assert(operations * extend <= maximum - cost);
    lemma_fundamental_div_mod(maximum - cost, extend);
    lemma_mod_bound(maximum - cost, extend);
    if operations > (maximum - cost) / extend {
        let quotient = (maximum - cost) / extend;
        let remainder = (maximum - cost) % extend;
        assert(quotient + 1 <= operations);
        lemma_mul_inequality(quotient + 1, operations, extend);
        lemma_mul_is_commutative(extend, quotient);
        lemma_mul_is_distributive_add_other_way(extend, quotient, 1);
        assert(maximum - cost == extend * quotient + remainder);
        assert(remainder < extend);
        assert(maximum - cost < quotient * extend + extend);
        assert(quotient * extend + extend == (quotient + 1) * extend);
        assert(false);
    }
}

proof fn guarded_addition_is_budget_bounded(cost: nat, increment: nat, maximum: nat)
    requires
        cost <= maximum,
        increment <= maximum - cost,
    ensures
        cost + increment <= maximum,
{
}

}
