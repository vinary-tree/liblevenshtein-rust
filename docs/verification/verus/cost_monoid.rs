//! Verus model of exact CostMonoid and CostScale arithmetic.
//!
//! This file is verified directly with `verus`; it is not compiled by Cargo.

use vstd::prelude::*;
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;

fn main() {}

verus! {

pub open spec fn additive(left: nat, right: nat) -> nat {
    left + right
}

pub open spec fn bottleneck(left: nat, right: nat) -> nat {
    if left <= right { right } else { left }
}

proof fn additive_associative(a: nat, b: nat, c: nat)
    ensures
        additive(additive(a, b), c) == additive(a, additive(b, c)),
{
}

proof fn additive_two_sided_monotone(a: nat, b: nat, step: nat)
    requires
        a <= b,
    ensures
        additive(a, step) <= additive(b, step),
        additive(step, a) <= additive(step, b),
{
}

proof fn bottleneck_associative(a: nat, b: nat, c: nat)
    ensures
        bottleneck(bottleneck(a, b), c) == bottleneck(a, bottleneck(b, c)),
{
}

proof fn bottleneck_inflates(accumulated: nat, step: nat)
    ensures
        accumulated <= bottleneck(accumulated, step),
{
}

proof fn exact_scale(
    numerator: int,
    required_denominator: int,
    scale_denominator: int,
)
    requires
        numerator >= 0,
        required_denominator > 0,
        scale_denominator >= 0,
        scale_denominator % required_denominator == 0,
    ensures
        (numerator * (scale_denominator / required_denominator))
            * required_denominator
            == numerator * scale_denominator,
{
    let quotient = scale_denominator / required_denominator;
    lemma_fundamental_div_mod(scale_denominator, required_denominator);
    assert(scale_denominator
        == required_denominator * quotient
            + scale_denominator % required_denominator);
    assert(scale_denominator == required_denominator * quotient);
    assert(quotient * required_denominator
        == required_denominator * quotient) by (nonlinear_arith);
    assert(quotient * required_denominator == scale_denominator);
    assert((numerator * quotient) * required_denominator
        == numerator * (quotient * required_denominator)) by (nonlinear_arith);
    assert(numerator * (quotient * required_denominator)
        == numerator * scale_denominator);
}

}
