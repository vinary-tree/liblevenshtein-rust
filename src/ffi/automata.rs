//! Stable C ABI for standalone generalized and universal edit automata.
//!
//! The handles in this module own only automaton configuration or one online
//! prefix state. They do not own dictionaries and do not duplicate any edit
//! algorithm: every operation delegates to the corresponding public Rust
//! automaton.

use super::LlevStatus;
use super::index::{boundary, utf8};
use crate::cost::ScaleError;
use crate::transducer::generalized::{
    GeneralizedAutomaton, GeneralizedAutomatonError, GeneralizedOnlineAutomaton,
    GeneralizedOnlineLimits, GeneralizedOnlineObservation,
};
use crate::transducer::operation_type::MAX_OPERATION_NAME_BYTES;
use crate::transducer::universal::{
    MergeAndSplit, Standard, Transposition, UniversalAutomaton, UniversalOnlineAutomaton,
};
use crate::transducer::{
    MAX_OPERATION_SET_TOTAL_CONSUMPTION, MAX_SUBSTITUTION_PAIRS, MAX_SUBSTITUTION_TEXT_BYTES,
    OperationApplicability, OperationSet, OperationSetValidationError, OperationType,
    OwnedRestricted, OwnedRestrictedChar, SubstitutionPolicy, SubstitutionPolicyFor,
    SubstitutionSet, SubstitutionSetChar, Unrestricted,
};
use std::ffi::{c_char, c_void};
use std::mem::align_of;
use std::slice;
use std::sync::Arc;
use vinary_tree_interop::VtUnitDomain;

/// Maximum source units accepted by the default standalone-automaton policy.
pub const LLEV_DEFAULT_AUTOMATON_MAX_SOURCE_UNITS: usize = 1_000_000;
/// Maximum target units accepted by the default standalone-automaton policy.
pub const LLEV_DEFAULT_AUTOMATON_MAX_TARGET_UNITS: usize = 1_000_000;

const APPLICABILITY_ANY: u32 = 0;
const APPLICABILITY_EQUAL: u32 = 1;
const APPLICABILITY_ADJACENT_TRANSPOSE: u32 = 2;
const APPLICABILITY_LISTED: u32 = 3;

const UNIVERSAL_STANDARD: u32 = 0;
const UNIVERSAL_TRANSPOSITION: u32 = 1;
const UNIVERSAL_MERGE_AND_SPLIT: u32 = 2;
const UNIVERSAL_POLICY_UNRESTRICTED: u32 = 0;

/// Hard limits applied before and during standalone automaton evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlevAutomatonLimits {
    /// Maximum source units accepted by construction or complete matching.
    pub max_source_units: usize,
    /// Maximum target units accepted by complete matching or online advance.
    pub max_target_units: usize,
    /// Generalized row-ring and scratch-cell ceiling.
    pub max_retained_cells: usize,
    /// Generalized operation/cell relaxations permitted per target unit.
    pub max_step_work_units: usize,
}

impl Default for LlevAutomatonLimits {
    fn default() -> Self {
        let generalized = GeneralizedOnlineLimits::default();
        Self {
            max_source_units: LLEV_DEFAULT_AUTOMATON_MAX_SOURCE_UNITS,
            max_target_units: LLEV_DEFAULT_AUTOMATON_MAX_TARGET_UNITS,
            max_retained_cells: generalized.max_retained_cells,
            max_step_work_units: generalized.max_step_work_units,
        }
    }
}

impl LlevAutomatonLimits {
    fn generalized(self) -> GeneralizedOnlineLimits {
        GeneralizedOnlineLimits {
            max_retained_cells: self.max_retained_cells,
            max_step_work_units: self.max_step_work_units,
        }
    }
}

/// One borrowed source/target pair for a listed generalized operation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevGeneralizedRestriction {
    /// Borrowed UTF-8 source bytes.
    pub source_data: *const c_char,
    /// Source byte length.
    pub source_len: usize,
    /// Borrowed UTF-8 target bytes.
    pub target_data: *const c_char,
    /// Target byte length.
    pub target_len: usize,
}

/// One borrowed runtime operation declaration.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevGeneralizedOperation {
    /// Unicode source scalars consumed.
    pub consume_source: usize,
    /// Unicode target scalars consumed.
    pub consume_target: usize,
    /// Non-negative finite decimal cost.
    pub weight: f64,
    /// Borrowed non-empty UTF-8 diagnostic name.
    pub name_data: *const c_char,
    /// Name byte length.
    pub name_len: usize,
    /// One `LLEV_OPERATION_APPLICABILITY_*` value.
    pub applicability: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Borrowed contiguous restrictions; used only by `LISTED`.
    pub restrictions: *const LlevGeneralizedRestriction,
    /// Number of restriction descriptors.
    pub restriction_count: usize,
}

/// One directional zero-cost universal substitution pair.
///
/// Values are interpreted according to the unit domain supplied to the
/// universal constructor: byte values must fit `u8`, Unicode values must be
/// valid scalar values, and u64 values are accepted without narrowing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlevUniversalEquivalence {
    /// Source/dictionary unit.
    pub source: u64,
    /// Target/query unit.
    pub target: u64,
}

/// Exact observation of a generalized target prefix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LlevGeneralizedObservation {
    /// Number of committed target Unicode scalars.
    pub consumed_target_len: usize,
    /// Number of in-budget source positions in the current generation.
    pub active_positions: usize,
    /// Exact fixed-point numerator when `has_distance` is one.
    pub scaled_distance: usize,
    /// Fixed-point denominator shared by all configured operation costs.
    pub scale_denominator: u32,
    /// One when this exact target generation has an in-budget source cell.
    ///
    /// A zero value is not a permanent-death signal: a multi-target-unit
    /// operation can still connect an older retained generation to a later one.
    pub current_row_nonempty: u8,
    /// One when the complete source is reachable within budget.
    pub accepting: u8,
    /// One when `scaled_distance` is present.
    pub has_distance: u8,
    /// Must be ignored; fixed to zero.
    pub reserved: u8,
}

/// Exact observation of a universal target prefix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LlevUniversalObservation {
    /// Number of committed target units.
    pub consumed_target_len: usize,
    /// Number of source units retained by the bound online machine.
    pub source_len: usize,
    /// One while the universal frontier is non-empty.
    pub alive: u8,
    /// One when the committed prefix is accepted as a complete target.
    pub accepting: u8,
    /// Must be ignored; fixed to zero.
    pub reserved: [u8; 6],
}

/// Opaque immutable generalized-automaton configuration.
pub struct LlevGeneralizedAutomaton {
    inner: GeneralizedAutomaton,
    scale_denominator: u32,
}

/// Opaque exclusive generalized online state.
pub struct LlevGeneralizedOnlineAutomaton {
    inner: GeneralizedOnlineAutomaton,
    scale_denominator: u32,
    max_target_units: usize,
}

#[derive(Clone, Debug)]
struct RestrictedU64 {
    pairs: Arc<[(u64, u64)]>,
}

impl SubstitutionPolicy for RestrictedU64 {
    #[inline(always)]
    fn is_allowed(&self, dict_char: u8, query_char: u8) -> bool {
        dict_char == query_char
    }
}

impl SubstitutionPolicyFor<u64> for RestrictedU64 {
    #[inline(always)]
    fn is_allowed_for(&self, source: u64, target: u64) -> bool {
        source == target || self.pairs.binary_search(&(source, target)).is_ok()
    }
}

enum UniversalPolicy {
    Unrestricted,
    Bytes(OwnedRestricted),
    Text(OwnedRestrictedChar),
    U64(RestrictedU64),
}

/// Opaque immutable universal-automaton configuration.
pub struct LlevUniversalAutomaton {
    max_distance: u8,
    variant: u32,
    policy: UniversalPolicy,
}

enum UniversalOnline {
    StandardText(UniversalOnlineAutomaton<Standard>),
    StandardBytes(UniversalOnlineAutomaton<Standard, Unrestricted, u8>),
    StandardU64(UniversalOnlineAutomaton<Standard, Unrestricted, u64>),
    StandardRestrictedText(UniversalOnlineAutomaton<Standard, OwnedRestrictedChar>),
    StandardRestrictedBytes(UniversalOnlineAutomaton<Standard, OwnedRestricted, u8>),
    StandardRestrictedU64(UniversalOnlineAutomaton<Standard, RestrictedU64, u64>),
    TranspositionText(UniversalOnlineAutomaton<Transposition>),
    TranspositionBytes(UniversalOnlineAutomaton<Transposition, Unrestricted, u8>),
    TranspositionU64(UniversalOnlineAutomaton<Transposition, Unrestricted, u64>),
    TranspositionRestrictedText(UniversalOnlineAutomaton<Transposition, OwnedRestrictedChar>),
    TranspositionRestrictedBytes(UniversalOnlineAutomaton<Transposition, OwnedRestricted, u8>),
    TranspositionRestrictedU64(UniversalOnlineAutomaton<Transposition, RestrictedU64, u64>),
    MergeAndSplitText(UniversalOnlineAutomaton<MergeAndSplit>),
    MergeAndSplitBytes(UniversalOnlineAutomaton<MergeAndSplit, Unrestricted, u8>),
    MergeAndSplitU64(UniversalOnlineAutomaton<MergeAndSplit, Unrestricted, u64>),
    MergeAndSplitRestrictedText(UniversalOnlineAutomaton<MergeAndSplit, OwnedRestrictedChar>),
    MergeAndSplitRestrictedBytes(UniversalOnlineAutomaton<MergeAndSplit, OwnedRestricted, u8>),
    MergeAndSplitRestrictedU64(UniversalOnlineAutomaton<MergeAndSplit, RestrictedU64, u64>),
}

/// Opaque exclusive universal online state.
pub struct LlevUniversalOnlineAutomaton {
    inner: UniversalOnline,
    max_target_units: usize,
}

fn failure(status: LlevStatus, message: impl Into<String>) -> (LlevStatus, String) {
    (status, message.into())
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    failure(LlevStatus::InvalidArgument, message)
}

fn limited(message: impl Into<String>) -> (LlevStatus, String) {
    failure(LlevStatus::LimitExceeded, message)
}

fn contextual(
    context: impl AsRef<str>,
    (status, message): (LlevStatus, String),
) -> (LlevStatus, String) {
    (status, format!("{}: {message}", context.as_ref()))
}

unsafe fn borrowed_slice<'a, T>(
    data: *const T,
    len: usize,
    name: &str,
) -> Result<&'a [T], (LlevStatus, String)> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(failure(LlevStatus::NullPointer, format!("{name} is null")));
    }
    if !(data as usize).is_multiple_of(align_of::<T>()) {
        return Err(invalid(format!("{name} is not correctly aligned")));
    }
    Ok(slice::from_raw_parts(data, len))
}

unsafe fn limits(value: *const LlevAutomatonLimits) -> LlevAutomatonLimits {
    value.as_ref().copied().unwrap_or_default()
}

fn map_operation_validation(error: OperationSetValidationError) -> (LlevStatus, String) {
    let status = match error {
        OperationSetValidationError::ConsumptionOverflow { .. }
        | OperationSetValidationError::ConsumptionLimit { .. } => LlevStatus::LimitExceeded,
        _ => LlevStatus::InvalidArgument,
    };
    failure(status, error.to_string())
}

fn map_scale(error: ScaleError) -> (LlevStatus, String) {
    let status = match error {
        ScaleError::DenominatorOverflow | ScaleError::CostOverflow => LlevStatus::LimitExceeded,
        ScaleError::ZeroDenominator
        | ScaleError::NonFiniteWeight
        | ScaleError::NegativeWeight
        | ScaleError::InexactWeight { .. } => LlevStatus::InvalidArgument,
    };
    failure(status, error.to_string())
}

fn map_generalized(error: GeneralizedAutomatonError) -> (LlevStatus, String) {
    match error {
        GeneralizedAutomatonError::OperationSet(error) => map_operation_validation(error),
        GeneralizedAutomatonError::Scale(error) => map_scale(error),
        GeneralizedAutomatonError::ArithmeticOverflow
        | GeneralizedAutomatonError::ResourceLimit { .. } => limited(error.to_string()),
    }
}

unsafe fn parse_operation_set(
    operations: *const LlevGeneralizedOperation,
    operation_count: usize,
) -> Result<OperationSet, (LlevStatus, String)> {
    if operation_count > MAX_OPERATION_SET_TOTAL_CONSUMPTION {
        return Err(limited(format!(
            "operation count {operation_count} exceeds {MAX_OPERATION_SET_TOTAL_CONSUMPTION}"
        )));
    }
    let operations = borrowed_slice(operations, operation_count, "operations")?;
    let mut output = OperationSet::with_capacity(operation_count);
    let mut total_restrictions = 0usize;
    let mut total_restriction_bytes = 0usize;

    for (index, operation) in operations.iter().enumerate() {
        if operation.reserved != 0 {
            return Err(invalid(format!(
                "operation {index} reserved field is nonzero"
            )));
        }
        if !operation.weight.is_finite() || operation.weight < 0.0 {
            return Err(invalid(format!(
                "operation {index} weight must be finite and nonnegative"
            )));
        }
        if operation.consume_source == 0 && operation.consume_target == 0 {
            return Err(invalid(format!("operation {index} consumes no input")));
        }
        if operation.weight == 0.0 && operation.consume_source != operation.consume_target {
            return Err(invalid(format!(
                "operation {index} changes length at zero cost"
            )));
        }
        if operation.name_len == 0 || operation.name_len > MAX_OPERATION_NAME_BYTES {
            return Err(invalid(format!(
                "operation {index} name length {} is outside 1..={MAX_OPERATION_NAME_BYTES}",
                operation.name_len
            )));
        }
        let name = utf8(operation.name_data, operation.name_len)
            .map_err(|error| contextual(format!("operation {index} name"), error))?;

        total_restrictions = total_restrictions
            .checked_add(operation.restriction_count)
            .ok_or_else(|| limited("restriction count overflowed usize"))?;
        if total_restrictions > MAX_SUBSTITUTION_PAIRS {
            return Err(limited(format!(
                "restriction count {total_restrictions} exceeds {MAX_SUBSTITUTION_PAIRS}"
            )));
        }
        let restrictions = borrowed_slice(
            operation.restrictions,
            operation.restriction_count,
            "operation restrictions",
        )?;

        let applicability = match operation.applicability {
            APPLICABILITY_ANY if restrictions.is_empty() => OperationApplicability::Any,
            APPLICABILITY_EQUAL if restrictions.is_empty() => OperationApplicability::Equal,
            APPLICABILITY_ADJACENT_TRANSPOSE if restrictions.is_empty() => {
                OperationApplicability::AdjacentTranspose
            }
            APPLICABILITY_LISTED => {
                let mut set = SubstitutionSet::with_capacity(restrictions.len());
                for (pair_index, restriction) in restrictions.iter().enumerate() {
                    let source =
                        utf8(restriction.source_data, restriction.source_len).map_err(|error| {
                            contextual(
                                format!("operation {index} restriction {pair_index} source"),
                                error,
                            )
                        })?;
                    let target =
                        utf8(restriction.target_data, restriction.target_len).map_err(|error| {
                            contextual(
                                format!("operation {index} restriction {pair_index} target"),
                                error,
                            )
                        })?;
                    if source.is_empty() || target.is_empty() {
                        return Err(invalid(format!(
                            "operation {index} restriction {pair_index} contains an empty side"
                        )));
                    }
                    if source.chars().count() != operation.consume_source
                        || target.chars().count() != operation.consume_target
                    {
                        return Err(invalid(format!(
                            "operation {index} restriction {pair_index} arity disagrees with its operation"
                        )));
                    }
                    total_restriction_bytes = total_restriction_bytes
                        .checked_add(source.len())
                        .and_then(|value| value.checked_add(target.len()))
                        .ok_or_else(|| limited("restriction byte count overflowed usize"))?;
                    if total_restriction_bytes > MAX_SUBSTITUTION_TEXT_BYTES {
                        return Err(limited(format!(
                            "restriction text contains {total_restriction_bytes} bytes (limit {MAX_SUBSTITUTION_TEXT_BYTES})"
                        )));
                    }
                    set.allow_str(source, target);
                }
                OperationApplicability::Listed(set)
            }
            APPLICABILITY_ANY | APPLICABILITY_EQUAL | APPLICABILITY_ADJACENT_TRANSPOSE => {
                return Err(invalid(format!(
                    "operation {index} has restrictions but is not LISTED"
                )));
            }
            value => {
                return Err(invalid(format!(
                    "operation {index} has unknown applicability value {value}"
                )));
            }
        };
        output.add(OperationType::with_owned_applicability(
            operation.consume_source,
            operation.consume_target,
            operation.weight,
            applicability,
            name,
        ));
    }
    output.validate().map_err(map_operation_validation)?;
    Ok(output)
}

fn generalized_observation(
    observation: GeneralizedOnlineObservation,
    denominator: u32,
) -> LlevGeneralizedObservation {
    let distance = observation.distance_within_budget;
    LlevGeneralizedObservation {
        consumed_target_len: observation.consumed_target_len,
        active_positions: observation.active_positions,
        scaled_distance: distance.unwrap_or(0),
        scale_denominator: denominator,
        current_row_nonempty: u8::from(observation.active_positions != 0),
        accepting: u8::from(distance.is_some()),
        has_distance: u8::from(distance.is_some()),
        reserved: 0,
    }
}

fn check_limit(label: &str, observed: usize, limit: usize) -> Result<(), (LlevStatus, String)> {
    if observed > limit {
        Err(limited(format!(
            "{label} contains {observed} units (limit {limit})"
        )))
    } else {
        Ok(())
    }
}

enum Input<'a> {
    Text(&'a str),
    Bytes(&'a [u8]),
    U64(&'a [u64]),
}

impl Input<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Text(value) => value.chars().count(),
            Self::Bytes(value) => value.len(),
            Self::U64(value) => value.len(),
        }
    }
}

unsafe fn input<'a>(
    domain: u32,
    data: *const c_void,
    len: usize,
    label: &str,
) -> Result<Input<'a>, (LlevStatus, String)> {
    match domain {
        value if value == VtUnitDomain::UnicodeScalar as u32 => {
            utf8(data.cast::<c_char>(), len).map(Input::Text)
        }
        value if value == VtUnitDomain::Byte as u32 => {
            borrowed_slice(data.cast::<u8>(), len, label).map(Input::Bytes)
        }
        value if value == VtUnitDomain::U64 as u32 => {
            borrowed_slice(data.cast::<u64>(), len, label).map(Input::U64)
        }
        value => Err(invalid(format!("unknown unit domain {value}"))),
    }
}

unsafe fn universal_policy(
    unit_domain: u32,
    equivalences: *const LlevUniversalEquivalence,
    equivalence_count: usize,
) -> Result<UniversalPolicy, (LlevStatus, String)> {
    if equivalence_count == 0 {
        if unit_domain != UNIVERSAL_POLICY_UNRESTRICTED {
            return Err(invalid(
                "an empty universal substitution policy must use domain zero",
            ));
        }
        return Ok(UniversalPolicy::Unrestricted);
    }
    if equivalence_count > MAX_SUBSTITUTION_PAIRS {
        return Err(limited(format!(
            "universal substitution policy contains {equivalence_count} pairs (limit {MAX_SUBSTITUTION_PAIRS})"
        )));
    }
    let equivalences = borrowed_slice(equivalences, equivalence_count, "equivalences")?;
    match unit_domain {
        value if value == VtUnitDomain::Byte as u32 => {
            let mut set = SubstitutionSet::with_capacity(equivalence_count);
            for equivalence in equivalences {
                let source = u8::try_from(equivalence.source)
                    .map_err(|_| invalid("universal byte-policy source exceeds 255"))?;
                let target = u8::try_from(equivalence.target)
                    .map_err(|_| invalid("universal byte-policy target exceeds 255"))?;
                set.allow_byte(source, target);
            }
            Ok(UniversalPolicy::Bytes(OwnedRestricted::new(set)))
        }
        value if value == VtUnitDomain::UnicodeScalar as u32 => {
            let mut set = SubstitutionSetChar::with_capacity(equivalence_count);
            for equivalence in equivalences {
                let source = char::from_u32(
                    u32::try_from(equivalence.source)
                        .map_err(|_| invalid("invalid Unicode policy source scalar"))?,
                )
                .ok_or_else(|| invalid("invalid Unicode policy source scalar"))?;
                let target = char::from_u32(
                    u32::try_from(equivalence.target)
                        .map_err(|_| invalid("invalid Unicode policy target scalar"))?,
                )
                .ok_or_else(|| invalid("invalid Unicode policy target scalar"))?;
                set.allow(source, target);
            }
            Ok(UniversalPolicy::Text(OwnedRestrictedChar::new(set)))
        }
        value if value == VtUnitDomain::U64 as u32 => {
            let mut pairs = Vec::new();
            pairs
                .try_reserve_exact(equivalence_count)
                .map_err(|_| limited("unable to allocate universal u64 substitution policy"))?;
            pairs.extend(
                equivalences
                    .iter()
                    .map(|equivalence| (equivalence.source, equivalence.target)),
            );
            pairs.sort_unstable();
            pairs.dedup();
            Ok(UniversalPolicy::U64(RestrictedU64 {
                pairs: Arc::from(pairs),
            }))
        }
        value => Err(invalid(format!(
            "unknown universal substitution-policy unit domain {value}"
        ))),
    }
}

fn universal_online(
    automaton: &LlevUniversalAutomaton,
    source: Input<'_>,
) -> Result<UniversalOnline, (LlevStatus, String)> {
    let distance = automaton.max_distance;
    let domain_mismatch = || {
        failure(
            LlevStatus::DomainMismatch,
            "universal substitution policy and input use different unit domains",
        )
    };
    match (automaton.variant, &automaton.policy, source) {
        (UNIVERSAL_STANDARD, UniversalPolicy::Unrestricted, Input::Text(source)) => {
            Ok(UniversalOnline::StandardText(
                UniversalAutomaton::<Standard>::new(distance).online(source),
            ))
        }
        (UNIVERSAL_STANDARD, UniversalPolicy::Unrestricted, Input::Bytes(source)) => {
            Ok(UniversalOnline::StandardBytes(
                UniversalAutomaton::<Standard>::new(distance).online_bytes(source),
            ))
        }
        (UNIVERSAL_STANDARD, UniversalPolicy::Unrestricted, Input::U64(source)) => {
            Ok(UniversalOnline::StandardU64(
                UniversalAutomaton::<Standard>::new(distance).online_units(source),
            ))
        }
        (UNIVERSAL_TRANSPOSITION, UniversalPolicy::Unrestricted, Input::Text(source)) => {
            Ok(UniversalOnline::TranspositionText(
                UniversalAutomaton::<Transposition>::new(distance).online(source),
            ))
        }
        (UNIVERSAL_TRANSPOSITION, UniversalPolicy::Unrestricted, Input::Bytes(source)) => {
            Ok(UniversalOnline::TranspositionBytes(
                UniversalAutomaton::<Transposition>::new(distance).online_bytes(source),
            ))
        }
        (UNIVERSAL_TRANSPOSITION, UniversalPolicy::Unrestricted, Input::U64(source)) => {
            Ok(UniversalOnline::TranspositionU64(
                UniversalAutomaton::<Transposition>::new(distance).online_units(source),
            ))
        }
        (UNIVERSAL_MERGE_AND_SPLIT, UniversalPolicy::Unrestricted, Input::Text(source)) => {
            Ok(UniversalOnline::MergeAndSplitText(
                UniversalAutomaton::<MergeAndSplit>::new(distance).online(source),
            ))
        }
        (UNIVERSAL_MERGE_AND_SPLIT, UniversalPolicy::Unrestricted, Input::Bytes(source)) => {
            Ok(UniversalOnline::MergeAndSplitBytes(
                UniversalAutomaton::<MergeAndSplit>::new(distance).online_bytes(source),
            ))
        }
        (UNIVERSAL_MERGE_AND_SPLIT, UniversalPolicy::Unrestricted, Input::U64(source)) => {
            Ok(UniversalOnline::MergeAndSplitU64(
                UniversalAutomaton::<MergeAndSplit>::new(distance).online_units(source),
            ))
        }
        (UNIVERSAL_STANDARD, UniversalPolicy::Text(policy), Input::Text(source)) => {
            Ok(UniversalOnline::StandardRestrictedText(
                UniversalAutomaton::<Standard, _>::with_policy(distance, policy.clone())
                    .online(source),
            ))
        }
        (UNIVERSAL_STANDARD, UniversalPolicy::Bytes(policy), Input::Bytes(source)) => {
            Ok(UniversalOnline::StandardRestrictedBytes(
                UniversalAutomaton::<Standard, _>::with_policy(distance, policy.clone())
                    .online_bytes(source),
            ))
        }
        (UNIVERSAL_STANDARD, UniversalPolicy::U64(policy), Input::U64(source)) => {
            Ok(UniversalOnline::StandardRestrictedU64(
                UniversalAutomaton::<Standard, _>::with_policy(distance, policy.clone())
                    .online_units(source),
            ))
        }
        (UNIVERSAL_TRANSPOSITION, UniversalPolicy::Text(policy), Input::Text(source)) => {
            Ok(UniversalOnline::TranspositionRestrictedText(
                UniversalAutomaton::<Transposition, _>::with_policy(distance, policy.clone())
                    .online(source),
            ))
        }
        (UNIVERSAL_TRANSPOSITION, UniversalPolicy::Bytes(policy), Input::Bytes(source)) => {
            Ok(UniversalOnline::TranspositionRestrictedBytes(
                UniversalAutomaton::<Transposition, _>::with_policy(distance, policy.clone())
                    .online_bytes(source),
            ))
        }
        (UNIVERSAL_TRANSPOSITION, UniversalPolicy::U64(policy), Input::U64(source)) => {
            Ok(UniversalOnline::TranspositionRestrictedU64(
                UniversalAutomaton::<Transposition, _>::with_policy(distance, policy.clone())
                    .online_units(source),
            ))
        }
        (UNIVERSAL_MERGE_AND_SPLIT, UniversalPolicy::Text(policy), Input::Text(source)) => {
            Ok(UniversalOnline::MergeAndSplitRestrictedText(
                UniversalAutomaton::<MergeAndSplit, _>::with_policy(distance, policy.clone())
                    .online(source),
            ))
        }
        (UNIVERSAL_MERGE_AND_SPLIT, UniversalPolicy::Bytes(policy), Input::Bytes(source)) => {
            Ok(UniversalOnline::MergeAndSplitRestrictedBytes(
                UniversalAutomaton::<MergeAndSplit, _>::with_policy(distance, policy.clone())
                    .online_bytes(source),
            ))
        }
        (UNIVERSAL_MERGE_AND_SPLIT, UniversalPolicy::U64(policy), Input::U64(source)) => {
            Ok(UniversalOnline::MergeAndSplitRestrictedU64(
                UniversalAutomaton::<MergeAndSplit, _>::with_policy(distance, policy.clone())
                    .online_units(source),
            ))
        }
        _ => Err(domain_mismatch()),
    }
}

macro_rules! universal_observe {
    ($value:expr) => {
        match $value {
            UniversalOnline::StandardText(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::StandardBytes(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::StandardU64(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::StandardRestrictedText(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::StandardRestrictedBytes(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::StandardRestrictedU64(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::TranspositionText(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::TranspositionBytes(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::TranspositionU64(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::TranspositionRestrictedText(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::TranspositionRestrictedBytes(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::TranspositionRestrictedU64(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::MergeAndSplitText(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::MergeAndSplitBytes(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::MergeAndSplitU64(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::MergeAndSplitRestrictedText(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::MergeAndSplitRestrictedBytes(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
            UniversalOnline::MergeAndSplitRestrictedU64(value) => (
                value.input_length(),
                value.word_length(),
                value.state().is_some(),
                value.is_accepting(),
            ),
        }
    };
}

fn universal_observation(inner: &UniversalOnline) -> LlevUniversalObservation {
    let (consumed_target_len, source_len, alive, accepting) = universal_observe!(inner);
    LlevUniversalObservation {
        consumed_target_len,
        source_len,
        alive: u8::from(alive),
        accepting: u8::from(accepting),
        reserved: [0; 6],
    }
}

fn universal_advance(inner: &mut UniversalOnline, unit: u64) -> Result<(), (LlevStatus, String)> {
    match inner {
        UniversalOnline::StandardText(value) => value.advance(
            char::from_u32(u32::try_from(unit).map_err(|_| invalid("invalid Unicode scalar"))?)
                .ok_or_else(|| invalid("invalid Unicode scalar"))?,
        ),
        UniversalOnline::TranspositionText(value) => value.advance(
            char::from_u32(u32::try_from(unit).map_err(|_| invalid("invalid Unicode scalar"))?)
                .ok_or_else(|| invalid("invalid Unicode scalar"))?,
        ),
        UniversalOnline::MergeAndSplitText(value) => value.advance(
            char::from_u32(u32::try_from(unit).map_err(|_| invalid("invalid Unicode scalar"))?)
                .ok_or_else(|| invalid("invalid Unicode scalar"))?,
        ),
        UniversalOnline::StandardBytes(value) => {
            value.advance(u8::try_from(unit).map_err(|_| invalid("byte unit exceeds 255"))?)
        }
        UniversalOnline::StandardRestrictedText(value) => value.advance(
            char::from_u32(u32::try_from(unit).map_err(|_| invalid("invalid Unicode scalar"))?)
                .ok_or_else(|| invalid("invalid Unicode scalar"))?,
        ),
        UniversalOnline::StandardRestrictedBytes(value) => {
            value.advance(u8::try_from(unit).map_err(|_| invalid("byte unit exceeds 255"))?)
        }
        UniversalOnline::StandardRestrictedU64(value) => value.advance(unit),
        UniversalOnline::TranspositionBytes(value) => {
            value.advance(u8::try_from(unit).map_err(|_| invalid("byte unit exceeds 255"))?)
        }
        UniversalOnline::TranspositionRestrictedText(value) => value.advance(
            char::from_u32(u32::try_from(unit).map_err(|_| invalid("invalid Unicode scalar"))?)
                .ok_or_else(|| invalid("invalid Unicode scalar"))?,
        ),
        UniversalOnline::TranspositionRestrictedBytes(value) => {
            value.advance(u8::try_from(unit).map_err(|_| invalid("byte unit exceeds 255"))?)
        }
        UniversalOnline::TranspositionRestrictedU64(value) => value.advance(unit),
        UniversalOnline::MergeAndSplitBytes(value) => {
            value.advance(u8::try_from(unit).map_err(|_| invalid("byte unit exceeds 255"))?)
        }
        UniversalOnline::MergeAndSplitRestrictedText(value) => value.advance(
            char::from_u32(u32::try_from(unit).map_err(|_| invalid("invalid Unicode scalar"))?)
                .ok_or_else(|| invalid("invalid Unicode scalar"))?,
        ),
        UniversalOnline::MergeAndSplitRestrictedBytes(value) => {
            value.advance(u8::try_from(unit).map_err(|_| invalid("byte unit exceeds 255"))?)
        }
        UniversalOnline::MergeAndSplitRestrictedU64(value) => value.advance(unit),
        UniversalOnline::StandardU64(value) => value.advance(unit),
        UniversalOnline::TranspositionU64(value) => value.advance(unit),
        UniversalOnline::MergeAndSplitU64(value) => value.advance(unit),
    };
    Ok(())
}

/// Construct one immutable runtime-configured generalized automaton.
///
/// # Safety
///
/// `operations` must address `operation_count` live descriptors and every
/// nested pointer must satisfy its descriptor. `out_automaton` must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_automaton_new(
    max_distance: u8,
    operations: *const LlevGeneralizedOperation,
    operation_count: usize,
    out_automaton: *mut *mut LlevGeneralizedAutomaton,
) -> LlevStatus {
    boundary(|| {
        if out_automaton.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_automaton is null"));
        }
        out_automaton.write(std::ptr::null_mut());
        let operations = parse_operation_set(operations, operation_count)?;
        let inner = GeneralizedAutomaton::try_with_operations(max_distance, operations)
            .map_err(map_generalized)?;
        let scale_denominator = inner.cost_scale().map_err(map_generalized)?.denominator();
        out_automaton.write(Box::into_raw(Box::new(LlevGeneralizedAutomaton {
            inner,
            scale_denominator,
        })));
        Ok(LlevStatus::Ok)
    })
}

/// Release a generalized automaton handle.
///
/// # Safety
///
/// A non-null pointer must be a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_automaton_free(automaton: *mut LlevGeneralizedAutomaton) {
    if !automaton.is_null() {
        drop(Box::from_raw(automaton));
    }
}

/// Evaluate a complete Unicode target using the configured generalized grammar.
///
/// # Safety
///
/// The automaton and output must be live. Non-empty buffers must be readable.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_automaton_evaluate_utf8(
    automaton: *const LlevGeneralizedAutomaton,
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    configured_limits: *const LlevAutomatonLimits,
    out_observation: *mut LlevGeneralizedObservation,
) -> LlevStatus {
    boundary(|| {
        let automaton = automaton
            .as_ref()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "automaton is null"))?;
        if out_observation.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_observation is null"));
        }
        out_observation.write(LlevGeneralizedObservation::default());
        let source = utf8(source, source_len)?;
        let target = utf8(target, target_len)?;
        let configured_limits = limits(configured_limits);
        check_limit(
            "generalized source",
            source.chars().count(),
            configured_limits.max_source_units,
        )?;
        check_limit(
            "generalized target",
            target.chars().count(),
            configured_limits.max_target_units,
        )?;
        let mut online = automaton
            .inner
            .online_with_limits(source, configured_limits.generalized())
            .map_err(map_generalized)?;
        for unit in target.chars() {
            online.advance(unit).map_err(map_generalized)?;
        }
        out_observation.write(generalized_observation(
            online.observation(),
            automaton.scale_denominator,
        ));
        Ok(LlevStatus::Ok)
    })
}

/// Bind a generalized automaton to one Unicode source for online traversal.
///
/// # Safety
///
/// The automaton and output must be live; a non-empty source must be readable.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_online_new_utf8(
    automaton: *const LlevGeneralizedAutomaton,
    source: *const c_char,
    source_len: usize,
    configured_limits: *const LlevAutomatonLimits,
    out_online: *mut *mut LlevGeneralizedOnlineAutomaton,
) -> LlevStatus {
    boundary(|| {
        let automaton = automaton
            .as_ref()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "automaton is null"))?;
        if out_online.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_online is null"));
        }
        out_online.write(std::ptr::null_mut());
        let source = utf8(source, source_len)?;
        let configured_limits = limits(configured_limits);
        check_limit(
            "generalized source",
            source.chars().count(),
            configured_limits.max_source_units,
        )?;
        let inner = automaton
            .inner
            .online_with_limits(source, configured_limits.generalized())
            .map_err(map_generalized)?;
        out_online.write(Box::into_raw(Box::new(LlevGeneralizedOnlineAutomaton {
            inner,
            scale_denominator: automaton.scale_denominator,
            max_target_units: configured_limits.max_target_units,
        })));
        Ok(LlevStatus::Ok)
    })
}

/// Advance one Unicode scalar transactionally and return the committed state.
///
/// # Safety
///
/// The online handle and output must be live and exclusively borrowed.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_online_advance(
    online: *mut LlevGeneralizedOnlineAutomaton,
    scalar: u32,
    out_observation: *mut LlevGeneralizedObservation,
) -> LlevStatus {
    boundary(|| {
        let online = online
            .as_mut()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "online is null"))?;
        if out_observation.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_observation is null"));
        }
        out_observation.write(LlevGeneralizedObservation::default());
        let scalar = char::from_u32(scalar).ok_or_else(|| invalid("invalid Unicode scalar"))?;
        let next = online
            .inner
            .observation()
            .consumed_target_len
            .checked_add(1)
            .ok_or_else(|| limited("generalized target length overflowed usize"))?;
        check_limit("generalized target", next, online.max_target_units)?;
        let observation = online.inner.advance(scalar).map_err(map_generalized)?;
        out_observation.write(generalized_observation(
            observation,
            online.scale_denominator,
        ));
        Ok(LlevStatus::Ok)
    })
}

/// Observe the committed generalized prefix without advancing it.
///
/// # Safety
///
/// The online handle and output must be live.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_online_observation(
    online: *const LlevGeneralizedOnlineAutomaton,
    out_observation: *mut LlevGeneralizedObservation,
) -> LlevStatus {
    boundary(|| {
        let online = online
            .as_ref()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "online is null"))?;
        if out_observation.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_observation is null"));
        }
        out_observation.write(generalized_observation(
            online.inner.observation(),
            online.scale_denominator,
        ));
        Ok(LlevStatus::Ok)
    })
}

/// Release a generalized online handle.
///
/// # Safety
///
/// A non-null pointer must be a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn llev_generalized_online_free(online: *mut LlevGeneralizedOnlineAutomaton) {
    if !online.is_null() {
        drop(Box::from_raw(online));
    }
}

/// Construct one immutable universal automaton variant and substitution policy.
///
/// Domain zero with no pairs selects the allocation-free unrestricted policy.
/// A non-empty policy uses one explicit [`VtUnitDomain`] and interprets each
/// pair directionally from dictionary/source unit to query/target unit.
///
/// # Safety
///
/// `out_automaton` must be writable. A non-empty `equivalences` buffer must
/// address `equivalence_count` aligned descriptors.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_automaton_new(
    max_distance: u8,
    variant: u32,
    policy_unit_domain: u32,
    equivalences: *const LlevUniversalEquivalence,
    equivalence_count: usize,
    out_automaton: *mut *mut LlevUniversalAutomaton,
) -> LlevStatus {
    boundary(|| {
        if out_automaton.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_automaton is null"));
        }
        out_automaton.write(std::ptr::null_mut());
        match variant {
            UNIVERSAL_STANDARD | UNIVERSAL_TRANSPOSITION | UNIVERSAL_MERGE_AND_SPLIT => {}
            value => return Err(invalid(format!("unknown universal variant {value}"))),
        }
        let policy = universal_policy(policy_unit_domain, equivalences, equivalence_count)?;
        out_automaton.write(Box::into_raw(Box::new(LlevUniversalAutomaton {
            max_distance,
            variant,
            policy,
        })));
        Ok(LlevStatus::Ok)
    })
}

/// Release a universal automaton handle.
///
/// # Safety
///
/// A non-null pointer must be a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_automaton_free(automaton: *mut LlevUniversalAutomaton) {
    if !automaton.is_null() {
        drop(Box::from_raw(automaton));
    }
}

/// Evaluate one complete source/target pair in an explicit unit domain.
///
/// # Safety
///
/// The automaton and output must be live. Buffers must follow the selected
/// domain's byte-length or aligned-u64-length contract.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_automaton_evaluate(
    automaton: *const LlevUniversalAutomaton,
    unit_domain: u32,
    source_data: *const c_void,
    source_len: usize,
    target_data: *const c_void,
    target_len: usize,
    configured_limits: *const LlevAutomatonLimits,
    out_observation: *mut LlevUniversalObservation,
) -> LlevStatus {
    boundary(|| {
        let automaton = automaton
            .as_ref()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "automaton is null"))?;
        if out_observation.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_observation is null"));
        }
        out_observation.write(LlevUniversalObservation::default());
        let source = input(unit_domain, source_data, source_len, "source")?;
        let target = input(unit_domain, target_data, target_len, "target")?;
        let configured_limits = limits(configured_limits);
        check_limit(
            "universal source",
            source.len(),
            configured_limits.max_source_units,
        )?;
        check_limit(
            "universal target",
            target.len(),
            configured_limits.max_target_units,
        )?;
        let mut online = universal_online(automaton, source)?;
        match target {
            Input::Text(value) => {
                for unit in value.chars() {
                    universal_advance(&mut online, u64::from(u32::from(unit)))?;
                }
            }
            Input::Bytes(value) => {
                for &unit in value {
                    universal_advance(&mut online, u64::from(unit))?;
                }
            }
            Input::U64(value) => {
                for &unit in value {
                    universal_advance(&mut online, unit)?;
                }
            }
        }
        out_observation.write(universal_observation(&online));
        Ok(LlevStatus::Ok)
    })
}

/// Bind a universal automaton to one source in an explicit unit domain.
///
/// # Safety
///
/// The automaton and output must be live. The source must follow the selected
/// domain's buffer contract.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_online_new(
    automaton: *const LlevUniversalAutomaton,
    unit_domain: u32,
    source_data: *const c_void,
    source_len: usize,
    configured_limits: *const LlevAutomatonLimits,
    out_online: *mut *mut LlevUniversalOnlineAutomaton,
) -> LlevStatus {
    boundary(|| {
        let automaton = automaton
            .as_ref()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "automaton is null"))?;
        if out_online.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_online is null"));
        }
        out_online.write(std::ptr::null_mut());
        let source = input(unit_domain, source_data, source_len, "source")?;
        let configured_limits = limits(configured_limits);
        check_limit(
            "universal source",
            source.len(),
            configured_limits.max_source_units,
        )?;
        let inner = universal_online(automaton, source)?;
        out_online.write(Box::into_raw(Box::new(LlevUniversalOnlineAutomaton {
            inner,
            max_target_units: configured_limits.max_target_units,
        })));
        Ok(LlevStatus::Ok)
    })
}

/// Advance one domain-native unit and return the committed universal state.
///
/// # Safety
///
/// The online handle and output must be live and exclusively borrowed.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_online_advance(
    online: *mut LlevUniversalOnlineAutomaton,
    unit: u64,
    out_observation: *mut LlevUniversalObservation,
) -> LlevStatus {
    boundary(|| {
        let online = online
            .as_mut()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "online is null"))?;
        if out_observation.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_observation is null"));
        }
        out_observation.write(LlevUniversalObservation::default());
        let (consumed, _, _, _) = universal_observe!(&online.inner);
        let next = consumed
            .checked_add(1)
            .ok_or_else(|| limited("universal target length overflowed usize"))?;
        check_limit("universal target", next, online.max_target_units)?;
        universal_advance(&mut online.inner, unit)?;
        out_observation.write(universal_observation(&online.inner));
        Ok(LlevStatus::Ok)
    })
}

/// Observe the committed universal prefix without advancing it.
///
/// # Safety
///
/// The online handle and output must be live.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_online_observation(
    online: *const LlevUniversalOnlineAutomaton,
    out_observation: *mut LlevUniversalObservation,
) -> LlevStatus {
    boundary(|| {
        let online = online
            .as_ref()
            .ok_or_else(|| failure(LlevStatus::NullPointer, "online is null"))?;
        if out_observation.is_null() {
            return Err(failure(LlevStatus::NullPointer, "out_observation is null"));
        }
        out_observation.write(universal_observation(&online.inner));
        Ok(LlevStatus::Ok)
    })
}

/// Release a universal online handle.
///
/// # Safety
///
/// A non-null pointer must be a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn llev_universal_online_free(online: *mut LlevUniversalOnlineAutomaton) {
    if !online.is_null() {
        drop(Box::from_raw(online));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_parts(value: &str) -> (*const c_char, usize) {
        (value.as_ptr().cast(), value.len())
    }

    fn operation(
        consume_source: usize,
        consume_target: usize,
        weight: f64,
        name: &str,
        applicability: u32,
    ) -> LlevGeneralizedOperation {
        LlevGeneralizedOperation {
            consume_source,
            consume_target,
            weight,
            name_data: name.as_ptr().cast(),
            name_len: name.len(),
            applicability,
            reserved: 0,
            restrictions: std::ptr::null(),
            restriction_count: 0,
        }
    }

    #[test]
    fn generalized_ffi_matches_native_exact_cost_and_online_state() {
        let names = ["match", "substitute", "insert", "delete", "transpose"];
        let operations = [
            operation(1, 1, 0.0, names[0], APPLICABILITY_EQUAL),
            operation(1, 1, 1.0, names[1], APPLICABILITY_ANY),
            operation(0, 1, 1.0, names[2], APPLICABILITY_ANY),
            operation(1, 0, 1.0, names[3], APPLICABILITY_ANY),
            operation(2, 2, 0.25, names[4], APPLICABILITY_ADJACENT_TRANSPOSE),
        ];
        let mut automaton = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_generalized_automaton_new(
                    1,
                    operations.as_ptr(),
                    operations.len(),
                    &mut automaton,
                )
            },
            LlevStatus::Ok
        );
        let (source, source_len) = utf8_parts("ab");
        let (target, target_len) = utf8_parts("ba");
        let mut result = LlevGeneralizedObservation::default();
        assert_eq!(
            unsafe {
                llev_generalized_automaton_evaluate_utf8(
                    automaton,
                    source,
                    source_len,
                    target,
                    target_len,
                    std::ptr::null(),
                    &mut result,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(result.scale_denominator, 4);
        assert_eq!(result.scaled_distance, 1);
        assert_eq!(result.accepting, 1);

        let mut online = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_generalized_online_new_utf8(
                    automaton,
                    source,
                    source_len,
                    std::ptr::null(),
                    &mut online,
                )
            },
            LlevStatus::Ok
        );
        for unit in "ba".chars() {
            assert_eq!(
                unsafe { llev_generalized_online_advance(online, u32::from(unit), &mut result) },
                LlevStatus::Ok
            );
        }
        assert_eq!(result.scaled_distance, 1);
        assert_eq!(result.scale_denominator, 4);
        unsafe {
            llev_generalized_online_free(online);
            llev_generalized_automaton_free(automaton);
        }
    }

    #[test]
    fn generalized_ffi_enforces_limits_without_committing_a_step() {
        let names = ["match", "substitute", "insert", "delete"];
        let operations = [
            operation(1, 1, 0.0, names[0], APPLICABILITY_EQUAL),
            operation(1, 1, 1.0, names[1], APPLICABILITY_ANY),
            operation(0, 1, 1.0, names[2], APPLICABILITY_ANY),
            operation(1, 0, 1.0, names[3], APPLICABILITY_ANY),
        ];
        let mut automaton = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_generalized_automaton_new(
                    1,
                    operations.as_ptr(),
                    operations.len(),
                    &mut automaton,
                )
            },
            LlevStatus::Ok
        );
        let (source, source_len) = utf8_parts("a");
        let limits = LlevAutomatonLimits {
            max_target_units: 1,
            ..LlevAutomatonLimits::default()
        };
        let mut online = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_generalized_online_new_utf8(
                    automaton,
                    source,
                    source_len,
                    &limits,
                    &mut online,
                )
            },
            LlevStatus::Ok
        );
        let mut result = LlevGeneralizedObservation::default();
        assert_eq!(
            unsafe { llev_generalized_online_advance(online, u32::from('a'), &mut result) },
            LlevStatus::Ok
        );
        assert_eq!(
            unsafe { llev_generalized_online_advance(online, u32::from('b'), &mut result) },
            LlevStatus::LimitExceeded
        );
        assert_eq!(
            unsafe { llev_generalized_online_observation(online, &mut result) },
            LlevStatus::Ok
        );
        assert_eq!(result.consumed_target_len, 1);
        unsafe {
            llev_generalized_online_free(online);
            llev_generalized_automaton_free(automaton);
        }
    }

    #[test]
    fn generalized_empty_current_row_does_not_claim_permanent_death() {
        let name = "equal-pair";
        let operation = operation(2, 2, 0.0, name, APPLICABILITY_EQUAL);
        let mut automaton = std::ptr::null_mut();
        assert_eq!(
            unsafe { llev_generalized_automaton_new(0, &operation, 1, &mut automaton) },
            LlevStatus::Ok
        );
        let (source, source_len) = utf8_parts("ab");
        let mut online = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_generalized_online_new_utf8(
                    automaton,
                    source,
                    source_len,
                    std::ptr::null(),
                    &mut online,
                )
            },
            LlevStatus::Ok
        );

        let mut result = LlevGeneralizedObservation::default();
        assert_eq!(
            unsafe { llev_generalized_online_advance(online, u32::from('a'), &mut result) },
            LlevStatus::Ok
        );
        assert_eq!(result.current_row_nonempty, 0);
        assert_eq!(result.accepting, 0);
        assert_eq!(
            unsafe { llev_generalized_online_advance(online, u32::from('b'), &mut result) },
            LlevStatus::Ok
        );
        assert_eq!(result.current_row_nonempty, 1);
        assert_eq!(result.accepting, 1);

        unsafe {
            llev_generalized_online_free(online);
            llev_generalized_automaton_free(automaton);
        }
    }

    #[test]
    fn generalized_listed_operations_are_owned_directional_and_exact() {
        let name = "listed-equivalence";
        let source_restriction = "é";
        let target_restriction = "e";
        let restriction = LlevGeneralizedRestriction {
            source_data: source_restriction.as_ptr().cast(),
            source_len: source_restriction.len(),
            target_data: target_restriction.as_ptr().cast(),
            target_len: target_restriction.len(),
        };
        let mut operation = operation(1, 1, 0.25, name, APPLICABILITY_LISTED);
        operation.restrictions = &restriction;
        operation.restriction_count = 1;
        let mut automaton = std::ptr::null_mut();
        assert_eq!(
            unsafe { llev_generalized_automaton_new(1, &operation, 1, &mut automaton) },
            LlevStatus::Ok
        );

        // The constructor owns the parsed strings. Evaluation does not borrow
        // the descriptor array or its nested buffers.
        let mut observation = LlevGeneralizedObservation::default();
        assert_eq!(
            unsafe {
                llev_generalized_automaton_evaluate_utf8(
                    automaton,
                    source_restriction.as_ptr().cast(),
                    source_restriction.len(),
                    target_restriction.as_ptr().cast(),
                    target_restriction.len(),
                    std::ptr::null(),
                    &mut observation,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(observation.scale_denominator, 4);
        assert_eq!(observation.scaled_distance, 1);
        assert_eq!(observation.accepting, 1);
        assert_eq!(
            unsafe {
                llev_generalized_automaton_evaluate_utf8(
                    automaton,
                    target_restriction.as_ptr().cast(),
                    target_restriction.len(),
                    source_restriction.as_ptr().cast(),
                    source_restriction.len(),
                    std::ptr::null(),
                    &mut observation,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(observation.accepting, 0);
        unsafe { llev_generalized_automaton_free(automaton) };
    }

    #[test]
    fn generalized_constructor_rejects_structurally_unsafe_operations() {
        for invalid_operation in [
            operation(0, 0, 1.0, "no-progress", APPLICABILITY_ANY),
            operation(0, 1, 0.0, "free-growth", APPLICABILITY_ANY),
        ] {
            let mut automaton = std::ptr::dangling_mut::<LlevGeneralizedAutomaton>();
            assert_eq!(
                unsafe { llev_generalized_automaton_new(1, &invalid_operation, 1, &mut automaton) },
                LlevStatus::InvalidArgument
            );
            assert!(automaton.is_null());
        }

        let invalid_utf8 = [0xff_u8];
        let mut invalid_name = operation(1, 1, 1.0, "x", APPLICABILITY_ANY);
        invalid_name.name_data = invalid_utf8.as_ptr().cast();
        invalid_name.name_len = invalid_utf8.len();
        let mut automaton = std::ptr::dangling_mut::<LlevGeneralizedAutomaton>();
        assert_eq!(
            unsafe { llev_generalized_automaton_new(1, &invalid_name, 1, &mut automaton) },
            LlevStatus::InvalidUtf8
        );
        assert!(automaton.is_null());

        let mut null_name = operation(1, 1, 1.0, "x", APPLICABILITY_ANY);
        null_name.name_data = std::ptr::null();
        let mut automaton = std::ptr::dangling_mut::<LlevGeneralizedAutomaton>();
        assert_eq!(
            unsafe { llev_generalized_automaton_new(1, &null_name, 1, &mut automaton) },
            LlevStatus::NullPointer
        );
        assert!(automaton.is_null());

        let explicit = LlevAutomatonLimits {
            max_retained_cells: LLEV_DEFAULT_AUTOMATON_MAX_SOURCE_UNITS + 17,
            ..LlevAutomatonLimits::default()
        };
        assert_eq!(
            explicit.generalized().max_retained_cells,
            LLEV_DEFAULT_AUTOMATON_MAX_SOURCE_UNITS + 17
        );
    }

    #[test]
    fn universal_ffi_is_domain_complete_and_variant_exact() {
        for variant in [
            UNIVERSAL_STANDARD,
            UNIVERSAL_TRANSPOSITION,
            UNIVERSAL_MERGE_AND_SPLIT,
        ] {
            let mut automaton = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    llev_universal_automaton_new(
                        1,
                        variant,
                        UNIVERSAL_POLICY_UNRESTRICTED,
                        std::ptr::null(),
                        0,
                        &mut automaton,
                    )
                },
                LlevStatus::Ok
            );

            let text_source = "ab";
            let text_target = "ba";
            let mut observation = LlevUniversalObservation::default();
            assert_eq!(
                unsafe {
                    llev_universal_automaton_evaluate(
                        automaton,
                        VtUnitDomain::UnicodeScalar as u32,
                        text_source.as_ptr().cast(),
                        text_source.len(),
                        text_target.as_ptr().cast(),
                        text_target.len(),
                        std::ptr::null(),
                        &mut observation,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(
                observation.accepting != 0,
                variant == UNIVERSAL_TRANSPOSITION
            );

            let source = [1_u8, 2];
            let target = [2_u8, 1];
            assert_eq!(
                unsafe {
                    llev_universal_automaton_evaluate(
                        automaton,
                        VtUnitDomain::Byte as u32,
                        source.as_ptr().cast(),
                        source.len(),
                        target.as_ptr().cast(),
                        target.len(),
                        std::ptr::null(),
                        &mut observation,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(
                observation.accepting != 0,
                variant == UNIVERSAL_TRANSPOSITION
            );

            let source = [10_u64, 20];
            let target = [20_u64, 10];
            assert_eq!(
                unsafe {
                    llev_universal_automaton_evaluate(
                        automaton,
                        VtUnitDomain::U64 as u32,
                        source.as_ptr().cast(),
                        source.len(),
                        target.as_ptr().cast(),
                        target.len(),
                        std::ptr::null(),
                        &mut observation,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(
                observation.accepting != 0,
                variant == UNIVERSAL_TRANSPOSITION
            );
            unsafe { llev_universal_automaton_free(automaton) };
        }
    }

    #[test]
    fn universal_restricted_policies_are_directional_and_domain_exact() {
        unsafe fn evaluate(
            automaton: *const LlevUniversalAutomaton,
            domain: VtUnitDomain,
            source_data: *const c_void,
            source_len: usize,
            target_data: *const c_void,
            target_len: usize,
        ) -> Result<LlevUniversalObservation, LlevStatus> {
            let mut observation = LlevUniversalObservation::default();
            let status = llev_universal_automaton_evaluate(
                automaton,
                domain as u32,
                source_data,
                source_len,
                target_data,
                target_len,
                std::ptr::null(),
                &mut observation,
            );
            if status == LlevStatus::Ok {
                Ok(observation)
            } else {
                Err(status)
            }
        }

        let byte_source = [1_u8];
        let byte_target = [2_u8];
        let text_source = "é";
        let text_target = "e";
        let u64_source = [u64::MAX - 1];
        let u64_target = [u64::MAX];
        let cases = [
            (
                VtUnitDomain::Byte,
                LlevUniversalEquivalence {
                    source: u64::from(byte_source[0]),
                    target: u64::from(byte_target[0]),
                },
                byte_source.as_ptr().cast(),
                byte_source.len(),
                byte_target.as_ptr().cast(),
                byte_target.len(),
            ),
            (
                VtUnitDomain::UnicodeScalar,
                LlevUniversalEquivalence {
                    source: u64::from('é' as u32),
                    target: u64::from('e' as u32),
                },
                text_source.as_ptr().cast(),
                text_source.len(),
                text_target.as_ptr().cast(),
                text_target.len(),
            ),
            (
                VtUnitDomain::U64,
                LlevUniversalEquivalence {
                    source: u64_source[0],
                    target: u64_target[0],
                },
                u64_source.as_ptr().cast(),
                u64_source.len(),
                u64_target.as_ptr().cast(),
                u64_target.len(),
            ),
        ];
        for (domain, equivalence, source_data, source_len, target_data, target_len) in cases {
            let mut automaton = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    llev_universal_automaton_new(
                        0,
                        UNIVERSAL_STANDARD,
                        domain as u32,
                        &equivalence,
                        1,
                        &mut automaton,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(
                unsafe {
                    evaluate(
                        automaton,
                        domain,
                        source_data,
                        source_len,
                        target_data,
                        target_len,
                    )
                }
                .unwrap()
                .accepting,
                1
            );
            assert_eq!(
                unsafe {
                    evaluate(
                        automaton,
                        domain,
                        target_data,
                        target_len,
                        source_data,
                        source_len,
                    )
                }
                .unwrap()
                .accepting,
                0
            );

            let mismatch = unsafe {
                evaluate(
                    automaton,
                    VtUnitDomain::U64,
                    u64_source.as_ptr().cast(),
                    u64_source.len(),
                    u64_target.as_ptr().cast(),
                    u64_target.len(),
                )
            };
            if domain == VtUnitDomain::U64 {
                assert!(mismatch.is_ok());
            } else {
                assert_eq!(mismatch, Err(LlevStatus::DomainMismatch));
            }
            unsafe { llev_universal_automaton_free(automaton) };
        }
    }

    #[test]
    fn universal_constructor_rejects_invalid_policy_descriptors() {
        let invalid_byte = LlevUniversalEquivalence {
            source: 256,
            target: 1,
        };
        let invalid_scalar = LlevUniversalEquivalence {
            source: 0xd800,
            target: u64::from('a' as u32),
        };
        for (domain, equivalence) in [
            (VtUnitDomain::Byte, invalid_byte),
            (VtUnitDomain::UnicodeScalar, invalid_scalar),
        ] {
            let mut automaton = std::ptr::dangling_mut::<LlevUniversalAutomaton>();
            assert_eq!(
                unsafe {
                    llev_universal_automaton_new(
                        0,
                        UNIVERSAL_STANDARD,
                        domain as u32,
                        &equivalence,
                        1,
                        &mut automaton,
                    )
                },
                LlevStatus::InvalidArgument
            );
            assert!(automaton.is_null());
        }

        let mut automaton = std::ptr::dangling_mut::<LlevUniversalAutomaton>();
        assert_eq!(
            unsafe {
                llev_universal_automaton_new(
                    0,
                    UNIVERSAL_STANDARD,
                    VtUnitDomain::Byte as u32,
                    std::ptr::null(),
                    0,
                    &mut automaton,
                )
            },
            LlevStatus::InvalidArgument
        );
        assert!(automaton.is_null());
    }

    #[test]
    fn universal_online_validation_failures_do_not_commit() {
        let mut automaton = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_universal_automaton_new(
                    1,
                    UNIVERSAL_STANDARD,
                    UNIVERSAL_POLICY_UNRESTRICTED,
                    std::ptr::null(),
                    0,
                    &mut automaton,
                )
            },
            LlevStatus::Ok
        );
        let source = [b'a'];
        let limits = LlevAutomatonLimits {
            max_target_units: 1,
            ..LlevAutomatonLimits::default()
        };
        let mut online = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_universal_online_new(
                    automaton,
                    VtUnitDomain::Byte as u32,
                    source.as_ptr().cast(),
                    source.len(),
                    &limits,
                    &mut online,
                )
            },
            LlevStatus::Ok
        );
        let mut observation = LlevUniversalObservation::default();
        assert_eq!(
            unsafe { llev_universal_online_advance(online, 256, &mut observation) },
            LlevStatus::InvalidArgument
        );
        assert_eq!(
            unsafe { llev_universal_online_observation(online, &mut observation) },
            LlevStatus::Ok
        );
        assert_eq!(observation.consumed_target_len, 0);
        assert_eq!(
            unsafe { llev_universal_online_advance(online, u64::from(b'a'), &mut observation) },
            LlevStatus::Ok
        );
        assert_eq!(observation.consumed_target_len, 1);
        assert_eq!(
            unsafe { llev_universal_online_advance(online, u64::from(b'b'), &mut observation) },
            LlevStatus::LimitExceeded
        );
        assert_eq!(
            unsafe { llev_universal_online_observation(online, &mut observation) },
            LlevStatus::Ok
        );
        assert_eq!(observation.consumed_target_len, 1);
        unsafe {
            llev_universal_online_free(online);
            llev_universal_automaton_free(automaton);
        }
    }
}
