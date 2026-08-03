//! Exact decimal-to-integer scaling for configured operation costs.

use crate::transducer::OperationSet;
use std::fmt;

/// Default fixed-point denominator used when no operation-specific scale is requested.
pub const DEFAULT_COST_DENOMINATOR: u32 = 1_000;

/// Failure while deriving or applying an exact [`CostScale`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScaleError {
    /// A scale denominator must be strictly positive.
    ZeroDenominator,
    /// The supplied weight was NaN or infinite.
    NonFiniteWeight,
    /// Cost domains in this crate reject negative operation weights.
    NegativeWeight,
    /// The decimal representation requires a denominator larger than [`u32`].
    DenominatorOverflow,
    /// The selected scale cannot represent the weight exactly.
    InexactWeight {
        /// Configured denominator.
        scale_denominator: u32,
        /// Minimum reduced denominator needed by the weight.
        required_denominator: u64,
    },
    /// Scaling or least-common-multiple arithmetic exceeded the target integer type.
    CostOverflow,
}

impl fmt::Display for ScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => f.write_str("cost-scale denominator must be non-zero"),
            Self::NonFiniteWeight => f.write_str("cost weight must be finite"),
            Self::NegativeWeight => f.write_str("cost weight must be non-negative"),
            Self::DenominatorOverflow => {
                f.write_str("cost weight requires a denominator larger than u32")
            }
            Self::InexactWeight {
                scale_denominator,
                required_denominator,
            } => write!(
                f,
                "scale denominator {scale_denominator} cannot exactly represent denominator {required_denominator}"
            ),
            Self::CostOverflow => f.write_str("scaled cost arithmetic overflowed"),
        }
    }
}

impl std::error::Error for ScaleError {}

/// Exact fixed-point scale for decimal operation weights.
///
/// A value `c` represents the real cost `c / denominator`. Conversion parses
/// the shortest round-tripping decimal emitted by
/// [`ToString::to_string`], reduces
/// it as a rational, and uses checked integer arithmetic. It never silently
/// truncates or rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CostScale {
    denominator: u32,
}

impl CostScale {
    /// Construct a scale with an explicit positive denominator.
    pub const fn new(denominator: u32) -> Result<Self, ScaleError> {
        if denominator == 0 {
            Err(ScaleError::ZeroDenominator)
        } else {
            Ok(Self { denominator })
        }
    }

    /// Derive the least common denominator of all operation weights.
    pub fn for_operations(operations: &OperationSet) -> Result<Self, ScaleError> {
        Self::for_weights(operations.iter().map(|operation| operation.weight()))
    }

    /// Derive the least common denominator of arbitrary decimal weights.
    pub fn for_weights(weights: impl IntoIterator<Item = f64>) -> Result<Self, ScaleError> {
        let mut denominator = 1_u64;
        for weight in weights {
            let (_, required) = decimal_ratio(weight)?;
            denominator = checked_lcm(denominator, required)?;
            if denominator > u64::from(u32::MAX) {
                return Err(ScaleError::DenominatorOverflow);
            }
        }
        Self::new(denominator as u32)
    }

    /// Return the fixed-point denominator.
    pub const fn denominator(self) -> u32 {
        self.denominator
    }

    /// Return the least common exact scale containing both input scales.
    pub fn common(self, other: Self) -> Result<Self, ScaleError> {
        let denominator = checked_lcm(u64::from(self.denominator), u64::from(other.denominator))?;
        if denominator > u64::from(u32::MAX) {
            return Err(ScaleError::DenominatorOverflow);
        }
        Self::new(denominator as u32)
    }

    /// Convert an already-scaled cost into an exact compatible target scale.
    pub fn rescale(self, cost: usize, target: Self) -> Result<usize, ScaleError> {
        let source = self.denominator as usize;
        let target_denominator = target.denominator as usize;
        if !target_denominator.is_multiple_of(source) {
            return Err(ScaleError::InexactWeight {
                scale_denominator: target.denominator,
                required_denominator: u64::from(self.denominator),
            });
        }
        cost.checked_mul(target_denominator / source)
            .ok_or(ScaleError::CostOverflow)
    }

    /// Convert a decimal weight to an exact scaled integer.
    pub fn to_scaled(self, weight: f64) -> Result<usize, ScaleError> {
        let (numerator, required) = decimal_ratio(weight)?;
        let denominator = u128::from(self.denominator);
        let required_u128 = u128::from(required);
        if denominator % required_u128 != 0 {
            return Err(ScaleError::InexactWeight {
                scale_denominator: self.denominator,
                required_denominator: required,
            });
        }
        let multiplier = denominator / required_u128;
        let scaled = numerator
            .checked_mul(multiplier)
            .ok_or(ScaleError::CostOverflow)?;
        usize::try_from(scaled).map_err(|_| ScaleError::CostOverflow)
    }

    /// Convert a unit edit budget to the same exact scaled domain.
    pub fn scale_budget(self, budget: u8) -> Result<usize, ScaleError> {
        usize::from(budget)
            .checked_mul(self.denominator as usize)
            .ok_or(ScaleError::CostOverflow)
    }

    /// Convert a scaled integer back to a floating-point presentation value.
    pub fn from_scaled(self, cost: usize) -> f64 {
        cost as f64 / f64::from(self.denominator)
    }
}

impl Default for CostScale {
    fn default() -> Self {
        Self {
            denominator: DEFAULT_COST_DENOMINATOR,
        }
    }
}

fn checked_lcm(left: u64, right: u64) -> Result<u64, ScaleError> {
    if left == 0 || right == 0 {
        return Err(ScaleError::ZeroDenominator);
    }
    left.checked_div(gcd(left, right))
        .and_then(|reduced| reduced.checked_mul(right))
        .ok_or(ScaleError::CostOverflow)
}

const fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn decimal_ratio(weight: f64) -> Result<(u128, u64), ScaleError> {
    if !weight.is_finite() {
        return Err(ScaleError::NonFiniteWeight);
    }
    if weight < 0.0 {
        return Err(ScaleError::NegativeWeight);
    }
    if weight == 0.0 {
        return Ok((0, 1));
    }

    let text = weight.to_string();
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => {
            let exponent = exponent
                .parse::<i32>()
                .map_err(|_| ScaleError::CostOverflow)?;
            (mantissa, exponent)
        }
        None => (text.as_str(), 0),
    };
    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let mut numerator = digits
        .parse::<u128>()
        .map_err(|_| ScaleError::CostOverflow)?;
    let net_decimal_places = i32::try_from(fractional_digits)
        .map_err(|_| ScaleError::CostOverflow)?
        .checked_sub(exponent)
        .ok_or(ScaleError::CostOverflow)?;
    let mut denominator = 1_u128;
    if net_decimal_places > 0 {
        denominator = checked_pow10(net_decimal_places as u32)?;
    } else if net_decimal_places < 0 {
        numerator = numerator
            .checked_mul(checked_pow10(net_decimal_places.unsigned_abs())?)
            .ok_or(ScaleError::CostOverflow)?;
    }

    let divisor = gcd_u128(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;
    let denominator = u64::try_from(denominator).map_err(|_| ScaleError::DenominatorOverflow)?;
    Ok((numerator, denominator))
}

fn checked_pow10(exponent: u32) -> Result<u128, ScaleError> {
    10_u128
        .checked_pow(exponent)
        .ok_or(ScaleError::DenominatorOverflow)
}

const fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::{OperationSetBuilder, OperationType};

    #[test]
    fn derives_reduced_lcm_and_round_trips() {
        let operations = OperationSetBuilder::new()
            .with_match()
            .with_operation(OperationType::new(1, 1, 0.15, "cheap"))
            .with_operation(OperationType::new(1, 1, 0.125, "dyadic"))
            .build();
        let scale = CostScale::for_operations(&operations).expect("finite decimal weights");

        assert_eq!(scale.denominator(), 40);
        assert_eq!(scale.to_scaled(0.15), Ok(6));
        assert_eq!(scale.to_scaled(0.125), Ok(5));
        assert_eq!(scale.from_scaled(6), 0.15);
    }

    #[test]
    fn rejects_inexact_nonfinite_negative_and_overflowing_values() {
        let scale = CostScale::new(10).expect("non-zero");
        assert!(matches!(
            scale.to_scaled(0.15),
            Err(ScaleError::InexactWeight { .. })
        ));
        assert_eq!(
            scale.to_scaled(f64::INFINITY),
            Err(ScaleError::NonFiniteWeight)
        );
        assert_eq!(scale.to_scaled(-0.5), Err(ScaleError::NegativeWeight));
        assert_eq!(
            CostScale::for_weights([1.0e-100]),
            Err(ScaleError::DenominatorOverflow)
        );
        assert_eq!(scale.to_scaled(1.0e20), Err(ScaleError::CostOverflow));
        assert_eq!(
            CostScale::new(1)
                .expect("unit scale")
                .rescale(usize::MAX, scale),
            Err(ScaleError::CostOverflow)
        );
    }

    #[test]
    fn default_scale_is_documented_thousandths() {
        let scale = CostScale::default();
        assert_eq!(scale.denominator(), DEFAULT_COST_DENOMINATOR);
        assert_eq!(scale.to_scaled(0.15), Ok(150));
        assert_eq!(scale.scale_budget(3), Ok(3_000));
    }

    #[test]
    fn common_scale_and_rescale_are_exact() {
        let tenths = CostScale::new(10).expect("non-zero");
        let quarters = CostScale::new(4).expect("non-zero");
        let twentieths = tenths.common(quarters).expect("bounded lcm");

        assert_eq!(twentieths.denominator(), 20);
        assert_eq!(tenths.rescale(3, twentieths), Ok(6));
        assert!(matches!(
            twentieths.rescale(6, tenths),
            Err(ScaleError::InexactWeight { .. })
        ));
    }
}
