//! Explicitly bounded reverse-phonetic expansion ABI.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::utf8;
use super::{LlevOwnedString, LlevPhoneticRuleSet, LlevStatus};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::{
    expand_phonetic_alternatives_char_bounded, expand_with_costs, ExpansionLimitError,
};
use std::ffi::c_char;

/// Explicit work/output ceilings for reverse-phonetic expansion.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevPhoneticExpansionLimits {
    /// Maximum Unicode scalars in the input.
    pub max_input_scalars: usize,
    /// Maximum rewrite-rule count.
    pub max_rules: usize,
    /// Maximum sum of pattern and replacement phone units.
    pub max_rule_units: usize,
    /// Maximum intermediate parse-prefix nodes (exhaustive mode).
    pub max_nodes: usize,
    /// Maximum result and conservative intermediate UTF-8 bytes.
    pub max_output_bytes: usize,
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled".into(),
    )
}

#[cfg(feature = "bindings-phonetic")]
unsafe fn prepare<'a>(
    input: *const c_char,
    input_len: usize,
    rules: *const LlevPhoneticRuleSet,
    limits: *const LlevPhoneticExpansionLimits,
) -> Result<
    (
        &'a str,
        &'a [crate::phonetic::RewriteRuleChar],
        LlevPhoneticExpansionLimits,
    ),
    (LlevStatus, String),
> {
    let limits = *limits
        .as_ref()
        .ok_or((LlevStatus::NullPointer, "limits is null".into()))?;
    if limits.max_input_scalars == 0
        || limits.max_rules == 0
        || limits.max_rule_units == 0
        || limits.max_nodes == 0
        || limits.max_output_bytes == 0
    {
        return Err((
            LlevStatus::InvalidArgument,
            "all expansion ceilings must be positive".into(),
        ));
    }
    let input = utf8(input, input_len)?;
    if input.chars().count() > limits.max_input_scalars {
        return Err((
            LlevStatus::LimitExceeded,
            "expansion input exceeds max_input_scalars".into(),
        ));
    }
    let rules = rules
        .as_ref()
        .map(|value| value.inner.rules())
        .unwrap_or(&[]);
    if rules.len() > limits.max_rules {
        return Err((
            LlevStatus::LimitExceeded,
            "expansion rules exceed max_rules".into(),
        ));
    }
    let units = rules
        .iter()
        .try_fold(0usize, |total, rule| {
            total
                .checked_add(rule.pattern.len())?
                .checked_add(rule.replacement.len())
        })
        .ok_or((
            LlevStatus::LimitExceeded,
            "expansion rule units overflow".into(),
        ))?;
    if units > limits.max_rule_units {
        return Err((
            LlevStatus::LimitExceeded,
            "expansion rules exceed max_rule_units".into(),
        ));
    }
    Ok((input, rules, limits))
}

/// Expand every reverse-phonetic segmentation into a regex pattern.
///
/// Success is byte-for-byte identical to native Rust exhaustive expansion;
/// a limit error returns no partial pattern. Null rules mean identity.
///
/// # Safety
///
/// Input/rule/limit pointers and output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_expand(
    input: *const c_char,
    input_len: usize,
    rules: *const LlevPhoneticRuleSet,
    limits: *const LlevPhoneticExpansionLimits,
    out_pattern: *mut LlevOwnedString,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_pattern.is_null() {
                return Err((LlevStatus::NullPointer, "out_pattern is null".into()));
            }
            let (input, rules, limits) = prepare(input, input_len, rules, limits)?;
            let pattern = expand_phonetic_alternatives_char_bounded(
                input,
                rules,
                limits.max_nodes,
                limits.max_output_bytes,
            )
            .map_err(|error| match error {
                ExpansionLimitError::InvalidLimit => (
                    LlevStatus::InvalidArgument,
                    "invalid expansion ceiling".into(),
                ),
                ExpansionLimitError::NodeLimit => (
                    LlevStatus::LimitExceeded,
                    "expansion exceeds max_nodes".into(),
                ),
                ExpansionLimitError::OutputLimit => (
                    LlevStatus::LimitExceeded,
                    "expansion exceeds max_output_bytes".into(),
                ),
            })?;
            out_pattern.write(super::phonetic_dictionary::owned(pattern));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (input, input_len, rules, limits, out_pattern);
            Err(unavailable())
        }
    })
}

/// Expand with native greedy rule-cost accounting. This is distinct from the
/// exhaustive pattern: it selects one segmentation and reports its max rule
/// cost sum. Both outputs are changed only on success.
///
/// # Safety
///
/// Input/rule/limit pointers and outputs must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_expand_with_costs(
    input: *const c_char,
    input_len: usize,
    rules: *const LlevPhoneticRuleSet,
    limits: *const LlevPhoneticExpansionLimits,
    out_pattern: *mut LlevOwnedString,
    out_cost: *mut f64,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_pattern.is_null() || out_cost.is_null() {
                return Err((LlevStatus::NullPointer, "expansion output is null".into()));
            }
            let (input, rules, limits) = prepare(input, input_len, rules, limits)?;
            let (pattern, cost) = expand_with_costs(input, rules);
            if pattern.len() > limits.max_output_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "expansion exceeds max_output_bytes".into(),
                ));
            }
            out_pattern.write(super::phonetic_dictionary::owned(pattern));
            out_cost.write(cost);
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (input, input_len, rules, limits, out_pattern, out_cost);
            Err(unavailable())
        }
    })
}
