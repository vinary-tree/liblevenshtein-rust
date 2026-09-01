//! Collection of operation types for generalized Levenshtein distance.
//!
//! This module provides [`OperationSet`] and [`OperationSetBuilder`] for composing
//! multiple [`OperationType`]s into a complete edit distance metric.
//!
//! ## Overview
//!
//! An **operation set** is a collection of operation types that defines a complete
//! generalized Levenshtein distance metric. It replaces the hardcoded `Algorithm` enum
//! with a flexible, composable system.
//!
//! ## Examples
//!
//! ### Standard Levenshtein
//!
//! ```rust
//! # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
//! let standard = OperationSetBuilder::new()
//!     .with_match()           // ⟨1, 1, 0.0⟩
//!     .with_substitution()    // ⟨1, 1, 1.0⟩
//!     .with_insertion()       // ⟨0, 1, 1.0⟩
//!     .with_deletion()        // ⟨1, 0, 1.0⟩
//!     .build();
//!
//! assert_eq!(standard.operations().len(), 4);
//! ```
//!
//! ### With Transposition
//!
//! ```rust
//! # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
//! let transposition = OperationSetBuilder::new()
//!     .with_standard_ops()    // match, subst, insert, delete
//!     .with_transposition()   // ⟨2, 2, 1.0⟩
//!     .build();
//!
//! assert_eq!(transposition.operations().len(), 5);
//! ```
//!
//! ### Phonetic Corrections
//!
//! ```rust
//! # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder, OperationType, SubstitutionSet};
//! let mut phonetic = SubstitutionSet::new();
//! phonetic.allow_str("ph", "f");
//! phonetic.allow_str("ch", "k");
//!
//! let ops = OperationSetBuilder::new()
//!     .with_match()
//!     .with_operation(OperationType::with_restriction(
//!         2, 1, 0.15,
//!         phonetic,
//!         "consonant_digraphs"
//!     ))
//!     .with_standard_ops()
//!     .build();
//! ```

use crate::cost::{CostScale, ScaleError};
use crate::transducer::operation_type::MAX_OPERATION_NAME_BYTES;
use crate::transducer::{OperationApplicability, OperationType, SubstitutionPair};
use std::cmp::Ordering;
use std::fmt;

/// Maximum aggregate source-plus-target consumption declared by one validated
/// operation set.
///
/// Validation is an explicit boundary for untrusted or generated operation
/// sets. The common presets consume at most eight units in aggregate; 4,096
/// leaves ample room for contextual rule collections while bounding the work
/// performed by every alignment-cell expansion.
pub const MAX_OPERATION_SET_TOTAL_CONSUMPTION: usize = 4_096;

/// Structural failure reported by [`OperationSet::validate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationSetValidationError {
    /// An operation name is empty or exceeds the persistence resource limit.
    InvalidName {
        /// Zero-based operation index.
        index: usize,
        /// Observed UTF-8 byte length.
        observed: usize,
        /// Maximum accepted UTF-8 byte length.
        limit: usize,
    },
    /// One operation consumes neither input, so it cannot advance the acyclic
    /// alignment grid.
    NoProgress {
        /// Zero-based operation index.
        index: usize,
        /// Operation name.
        name: Box<str>,
    },
    /// The operation weight is negative or non-finite.
    InvalidWeight {
        /// Zero-based operation index.
        index: usize,
        /// Operation name.
        name: Box<str>,
    },
    /// A free operation changes length, violating the bounded-diagonal
    /// condition.
    ZeroWeightLengthChange {
        /// Zero-based operation index.
        index: usize,
        /// Operation name.
        name: Box<str>,
        /// Source units consumed.
        consume_x: usize,
        /// Target units consumed.
        consume_y: usize,
    },
    /// The explicit applicability tag is incompatible with declared arity.
    ApplicabilityArity {
        /// Zero-based operation index.
        index: usize,
        /// Operation name.
        name: Box<str>,
        /// Applicability tag.
        applicability: &'static str,
        /// Source units consumed.
        consume_x: usize,
        /// Target units consumed.
        consume_y: usize,
    },
    /// A listed restriction pair has a different arity from its operation.
    RestrictionArity {
        /// Zero-based operation index.
        index: usize,
        /// Operation name.
        name: Box<str>,
        /// Zero-based pair index in canonical order.
        pair_index: usize,
        /// Source scalar units in the pair.
        source_units: usize,
        /// Target scalar units in the pair.
        target_units: usize,
        /// Declared source consumption.
        consume_x: usize,
        /// Declared target consumption.
        consume_y: usize,
    },
    /// Checked aggregation of the declared consumptions overflowed `usize`.
    ConsumptionOverflow {
        /// Zero-based operation index at which aggregation overflowed.
        index: usize,
    },
    /// Aggregate declared consumption exceeds the public resource ceiling.
    ConsumptionLimit {
        /// Observed aggregate consumption.
        observed: usize,
        /// Configured hard ceiling.
        limit: usize,
    },
}

impl fmt::Display for OperationSetValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName {
                index,
                observed,
                limit,
            } => write!(
                formatter,
                "operation {index} name contains {observed} UTF-8 bytes (accepted range 1..={limit})"
            ),
            Self::NoProgress { index, name } => write!(
                formatter,
                "operation {index} ({name}) consumes neither input"
            ),
            Self::InvalidWeight { index, name } => write!(
                formatter,
                "operation {index} ({name}) has a negative or non-finite weight"
            ),
            Self::ZeroWeightLengthChange {
                index,
                name,
                consume_x,
                consume_y,
            } => write!(
                formatter,
                "zero-weight operation {index} ({name}) changes length ({consume_x} != {consume_y})"
            ),
            Self::ApplicabilityArity {
                index,
                name,
                applicability,
                consume_x,
                consume_y,
            } => write!(
                formatter,
                "operation {index} ({name}) uses {applicability} applicability with incompatible arity ({consume_x}, {consume_y})"
            ),
            Self::RestrictionArity {
                index,
                name,
                pair_index,
                source_units,
                target_units,
                consume_x,
                consume_y,
            } => write!(
                formatter,
                "restriction pair {pair_index} of operation {index} ({name}) consumes ({source_units}, {target_units}), not declared ({consume_x}, {consume_y})"
            ),
            Self::ConsumptionOverflow { index } => write!(
                formatter,
                "operation-set consumption overflowed at operation {index}"
            ),
            Self::ConsumptionLimit { observed, limit } => write!(
                formatter,
                "operation set declares aggregate consumption {observed} (limit {limit})"
            ),
        }
    }
}

impl std::error::Error for OperationSetValidationError {}

/// Exact minimum cost per Unicode scalar for a pure empty-side operation.
///
/// A finite value is a reduced rational `numerator / denominator`. `Infinite`
/// means that the operation set contains no rule capable of consuming the
/// requested side while leaving the other side empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmptySideRate {
    /// No qualifying operation exists.
    Infinite,
    /// Reduced, non-negative rational cost per consumed scalar.
    Finite {
        /// Reduced cost numerator.
        numerator: usize,
        /// Positive reduced denominator.
        denominator: usize,
    },
}

impl EmptySideRate {
    fn finite(scaled_cost: usize, consumed: usize, scale: CostScale) -> Result<Self, ScaleError> {
        let numerator = scaled_cost as u128;
        let denominator = (scale.denominator() as u128)
            .checked_mul(consumed as u128)
            .ok_or(ScaleError::CostOverflow)?;
        if denominator == 0 {
            return Err(ScaleError::ZeroDenominator);
        }
        let divisor = gcd_u128(numerator, denominator);
        let numerator =
            usize::try_from(numerator / divisor).map_err(|_| ScaleError::CostOverflow)?;
        let denominator =
            usize::try_from(denominator / divisor).map_err(|_| ScaleError::CostOverflow)?;
        Ok(Self::Finite {
            numerator,
            denominator,
        })
    }

    /// Return whether no qualifying empty-side operation exists.
    pub const fn is_infinite(self) -> bool {
        matches!(self, Self::Infinite)
    }

    /// Return the reduced finite ratio, or `None` for infinity.
    pub const fn ratio(self) -> Option<(usize, usize)> {
        match self {
            Self::Infinite => None,
            Self::Finite {
                numerator,
                denominator,
            } => Some((numerator, denominator)),
        }
    }

    /// Test whether consuming `scalar_count` scalars at this rate fits an
    /// integer edit budget.
    ///
    /// Infinity fits only the zero-length case.
    pub fn fits_budget(self, scalar_count: usize, budget: u8) -> bool {
        match self {
            Self::Infinite => scalar_count == 0,
            Self::Finite {
                numerator,
                denominator,
            } => {
                (numerator as u128) * (scalar_count as u128)
                    <= (u128::from(budget)) * (denominator as u128)
            }
        }
    }

    /// Return the greatest number of scalars this rate can consume within an
    /// integer edit budget. Infinity permits none.
    pub fn max_consumable(self, budget: u8) -> usize {
        match self {
            Self::Infinite => 0,
            Self::Finite {
                numerator,
                denominator,
            } => {
                if numerator == 0 {
                    usize::MAX
                } else {
                    let value = (u128::from(budget) * denominator as u128) / numerator as u128;
                    usize::try_from(value).unwrap_or(usize::MAX)
                }
            }
        }
    }
}

impl Ord for EmptySideRate {
    fn cmp(&self, other: &Self) -> Ordering {
        match (*self, *other) {
            (Self::Infinite, Self::Infinite) => Ordering::Equal,
            (Self::Infinite, Self::Finite { .. }) => Ordering::Greater,
            (Self::Finite { .. }, Self::Infinite) => Ordering::Less,
            (
                Self::Finite {
                    numerator: left_numerator,
                    denominator: left_denominator,
                },
                Self::Finite {
                    numerator: right_numerator,
                    denominator: right_denominator,
                },
            ) => ((left_numerator as u128) * (right_denominator as u128))
                .cmp(&((right_numerator as u128) * (left_denominator as u128))),
        }
    }
}

impl PartialOrd for EmptySideRate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// A collection of operation types defining a generalized Levenshtein distance metric.
///
/// # Examples
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
/// // Create using builder
/// let ops = OperationSetBuilder::new()
///     .with_standard_ops()
///     .build();
///
/// // Access operations
/// for op in ops.operations() {
///     println!("{}", op);
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct OperationSet {
    /// Vector of operation types.
    /// Order matters for iteration, but typically not for correctness.
    operations: Vec<OperationType>,
}

impl OperationSet {
    /// Validate the complete operation set as one bounded alignment grammar.
    ///
    /// This complements the assertions in [`OperationType::new`]. It checks
    /// every member again, rejects non-progressing rules, and uses checked
    /// arithmetic to enforce
    /// $`\sum_{t\in\mathcal O}(t^x+t^y)\le 4096`$ across the whole set.
    /// The zero-weight condition $`t^w=0\Rightarrow t^x=t^y`$ is the
    /// bounded-diagonal obligation from Mitankin, Mihov, and Schulz.
    pub fn validate(&self) -> Result<(), OperationSetValidationError> {
        let mut total = 0usize;
        for (index, operation) in self.operations.iter().enumerate() {
            if operation.name().is_empty() || operation.name().len() > MAX_OPERATION_NAME_BYTES {
                return Err(OperationSetValidationError::InvalidName {
                    index,
                    observed: operation.name().len(),
                    limit: MAX_OPERATION_NAME_BYTES,
                });
            }
            let weight = operation.weight();
            if !weight.is_finite() || weight < 0.0 {
                return Err(OperationSetValidationError::InvalidWeight {
                    index,
                    name: operation.name().into(),
                });
            }
            if operation.consume_x() == 0 && operation.consume_y() == 0 {
                return Err(OperationSetValidationError::NoProgress {
                    index,
                    name: operation.name().into(),
                });
            }
            if weight == 0.0 && operation.consume_x() != operation.consume_y() {
                return Err(OperationSetValidationError::ZeroWeightLengthChange {
                    index,
                    name: operation.name().into(),
                    consume_x: operation.consume_x(),
                    consume_y: operation.consume_y(),
                });
            }

            match operation.applicability() {
                OperationApplicability::Equal if operation.consume_x() != operation.consume_y() => {
                    return Err(OperationSetValidationError::ApplicabilityArity {
                        index,
                        name: operation.name().into(),
                        applicability: "equal",
                        consume_x: operation.consume_x(),
                        consume_y: operation.consume_y(),
                    });
                }
                OperationApplicability::AdjacentTranspose
                    if operation.consume_x() != 2 || operation.consume_y() != 2 =>
                {
                    return Err(OperationSetValidationError::ApplicabilityArity {
                        index,
                        name: operation.name().into(),
                        applicability: "adjacent-transpose",
                        consume_x: operation.consume_x(),
                        consume_y: operation.consume_y(),
                    });
                }
                OperationApplicability::Listed(restriction) => {
                    for (pair_index, pair) in restriction.pairs().into_iter().enumerate() {
                        let (source_units, target_units) = match pair {
                            SubstitutionPair::Bytes { .. } => (1, 1),
                            SubstitutionPair::Strings { source, target } => {
                                (source.chars().count(), target.chars().count())
                            }
                        };
                        if source_units != operation.consume_x()
                            || target_units != operation.consume_y()
                        {
                            return Err(OperationSetValidationError::RestrictionArity {
                                index,
                                name: operation.name().into(),
                                pair_index,
                                source_units,
                                target_units,
                                consume_x: operation.consume_x(),
                                consume_y: operation.consume_y(),
                            });
                        }
                    }
                }
                OperationApplicability::Any
                | OperationApplicability::Equal
                | OperationApplicability::AdjacentTranspose => {}
            }

            let consumption = operation
                .consume_x()
                .checked_add(operation.consume_y())
                .ok_or(OperationSetValidationError::ConsumptionOverflow { index })?;
            total = total
                .checked_add(consumption)
                .ok_or(OperationSetValidationError::ConsumptionOverflow { index })?;
            if total > MAX_OPERATION_SET_TOTAL_CONSUMPTION {
                return Err(OperationSetValidationError::ConsumptionLimit {
                    observed: total,
                    limit: MAX_OPERATION_SET_TOTAL_CONSUMPTION,
                });
            }
        }
        Ok(())
    }

    fn empty_side_rate(
        &self,
        consumes_requested_side: impl Fn(&OperationType) -> Option<usize>,
    ) -> Result<EmptySideRate, ScaleError> {
        let scale = CostScale::for_operations(self)?;
        let mut best = EmptySideRate::Infinite;
        for operation in &self.operations {
            let Some(consumed) = consumes_requested_side(operation) else {
                continue;
            };
            let candidate =
                EmptySideRate::finite(scale.to_scaled(operation.weight())?, consumed, scale)?;
            best = best.min(candidate);
        }
        Ok(best)
    }

    /// Derive $`\rho_{\mathrm{del}}`$: the least exact cost per source scalar
    /// among operations that consume a non-empty source slice and no target
    /// scalars. Returns [`EmptySideRate::Infinite`] when deletion is absent.
    pub fn rho_del(&self) -> Result<EmptySideRate, ScaleError> {
        self.empty_side_rate(|operation| {
            (operation.consume_y() == 0 && operation.consume_x() > 0).then(|| operation.consume_x())
        })
    }

    /// Derive $`\rho_{\mathrm{ins}}`$: the least exact cost per target scalar
    /// among operations that consume a non-empty target slice and no source
    /// scalars. Returns [`EmptySideRate::Infinite`] when insertion is absent.
    pub fn rho_ins(&self) -> Result<EmptySideRate, ScaleError> {
        self.empty_side_rate(|operation| {
            (operation.consume_x() == 0 && operation.consume_y() > 0).then(|| operation.consume_y())
        })
    }

    /// Create an empty operation set.
    ///
    /// This is rarely useful directly - prefer [`OperationSetBuilder`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::new();
    /// assert_eq!(ops.operations().len(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Create an operation set with expected capacity.
    ///
    /// Pre-allocates space for `capacity` operations to avoid reallocations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::with_capacity(10);
    /// assert_eq!(ops.operations().len(), 0);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            operations: Vec::with_capacity(capacity),
        }
    }

    /// Add an operation to the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationType};
    /// let mut ops = OperationSet::new();
    /// ops.add(OperationType::new(1, 1, 0.0, "match"));
    /// assert_eq!(ops.operations().len(), 1);
    /// ```
    pub fn add(&mut self, op: OperationType) {
        self.operations.push(op);
    }

    /// Get a slice of all operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
    /// let ops = OperationSetBuilder::new().with_standard_ops().build();
    /// assert_eq!(ops.operations().len(), 4);
    /// ```
    #[inline]
    pub fn operations(&self) -> &[OperationType] {
        &self.operations
    }

    /// Get the number of operations in the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
    /// let ops = OperationSetBuilder::new().with_match().build();
    /// assert_eq!(ops.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the operation set is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::new();
    /// assert!(ops.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Iterate over operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
    /// let ops = OperationSetBuilder::new().with_standard_ops().build();
    /// for op in ops.iter() {
    ///     println!("{}", op);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &OperationType> {
        self.operations.iter()
    }

    /// Create a standard Levenshtein operation set.
    ///
    /// Includes: match, substitution, insertion, deletion.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::standard();
    /// assert_eq!(ops.len(), 4);
    /// ```
    pub fn standard() -> Self {
        OperationSetBuilder::new().with_standard_ops().build()
    }

    /// Create a Levenshtein operation set with transposition.
    ///
    /// Includes: match, substitution, insertion, deletion, transposition.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::with_transposition();
    /// assert_eq!(ops.len(), 5);
    /// ```
    pub fn with_transposition() -> Self {
        OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .build()
    }

    /// Create a Levenshtein operation set with merge and split.
    ///
    /// Includes: match, substitution, insertion, deletion, merge, split.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSet;
    /// let ops = OperationSet::with_merge_split();
    /// assert_eq!(ops.len(), 6);
    /// ```
    pub fn with_merge_split() -> Self {
        OperationSetBuilder::new()
            .with_standard_ops()
            .with_merge()
            .with_split()
            .build()
    }
}

impl Default for OperationSet {
    /// Create a standard Levenshtein operation set (default).
    #[inline]
    fn default() -> Self {
        Self::standard()
    }
}

/// Builder for composing operation sets with a fluent API.
///
/// # Examples
///
/// ## Standard Levenshtein
///
/// ```rust
/// # use liblevenshtein::transducer::OperationSetBuilder;
/// let ops = OperationSetBuilder::new()
///     .with_match()
///     .with_substitution()
///     .with_insertion()
///     .with_deletion()
///     .build();
/// ```
///
/// ## Weighted Operations
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationSetBuilder, OperationType};
/// let ops = OperationSetBuilder::new()
///     .with_match()
///     .with_operation(OperationType::new(1, 1, 0.5, "cheap_subst"))
///     .with_insertion()
///     .with_deletion()
///     .build();
/// ```
///
/// ## Phonetic Corrections
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationSetBuilder, OperationType, SubstitutionSet};
/// let mut phonetic = SubstitutionSet::new();
/// phonetic.allow_str("ph", "f");
///
/// let ops = OperationSetBuilder::new()
///     .with_match()
///     .with_operation(OperationType::with_restriction(
///         2, 1, 0.15,
///         phonetic,
///         "ph_to_f"
///     ))
///     .with_standard_ops()
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct OperationSetBuilder {
    /// Operations being built.
    operations: Vec<OperationType>,
}

impl OperationSetBuilder {
    /// Create a new builder with no operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let builder = OperationSetBuilder::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add a custom operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationSetBuilder, OperationType};
    /// let ops = OperationSetBuilder::new()
    ///     .with_operation(OperationType::new(1, 1, 0.5, "custom"))
    ///     .build();
    /// ```
    pub fn with_operation(mut self, op: OperationType) -> Self {
        self.operations.push(op);
        self
    }

    /// Add match operation: $`\langle 1,1,0.0\rangle`$.
    ///
    /// Matches a single character with zero cost.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_match()
    ///     .build();
    /// ```
    pub fn with_match(self) -> Self {
        self.with_operation(OperationType::new(1, 1, 0.0, "match"))
    }

    /// Add substitution operation: $`\langle 1,1,1.0\rangle`$.
    ///
    /// Substitutes one character for another with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_substitution()
    ///     .build();
    /// ```
    pub fn with_substitution(self) -> Self {
        self.with_operation(OperationType::new(1, 1, 1.0, "substitute"))
    }

    /// Add insertion operation: $`\langle 0,1,1.0\rangle`$.
    ///
    /// Inserts a character from the query with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_insertion()
    ///     .build();
    /// ```
    pub fn with_insertion(self) -> Self {
        self.with_operation(OperationType::new(0, 1, 1.0, "insert"))
    }

    /// Add deletion operation: $`\langle 1,0,1.0\rangle`$.
    ///
    /// Deletes a character from the dictionary with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_deletion()
    ///     .build();
    /// ```
    pub fn with_deletion(self) -> Self {
        self.with_operation(OperationType::new(1, 0, 1.0, "delete"))
    }

    /// Add transposition operation: $`\langle 2,2,1.0\rangle`$.
    ///
    /// Swaps two adjacent characters with cost 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_transposition()
    ///     .build();
    /// ```
    pub fn with_transposition(self) -> Self {
        self.with_operation(OperationType::adjacent_transposition(1.0, "transpose"))
    }

    /// Add merge operation: $`\langle 1,2,1.0\rangle`$.
    ///
    /// Merges two characters in the query into one in the dictionary.
    /// Example: "every one" → "everyone"
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_merge()
    ///     .build();
    /// ```
    pub fn with_merge(self) -> Self {
        self.with_operation(OperationType::new(1, 2, 1.0, "merge"))
    }

    /// Add split operation: $`\langle 2,1,1.0\rangle`$.
    ///
    /// Splits one character in the dictionary into two in the query.
    /// Example: "everyone" → "every one"
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_split()
    ///     .build();
    /// ```
    pub fn with_split(self) -> Self {
        self.with_operation(OperationType::new(2, 1, 1.0, "split"))
    }

    /// Add all standard operations: match, substitution, insertion, deletion.
    ///
    /// This is a convenience method equivalent to calling:
    /// - `with_match()`
    /// - `with_substitution()`
    /// - `with_insertion()`
    /// - `with_deletion()`
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_standard_ops()
    ///     .build();
    ///
    /// assert_eq!(ops.len(), 4);
    /// ```
    pub fn with_standard_ops(self) -> Self {
        self.with_match()
            .with_substitution()
            .with_insertion()
            .with_deletion()
    }

    /// Build the final operation set.
    ///
    /// Consumes the builder and returns the constructed [`OperationSet`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationSetBuilder;
    /// let ops = OperationSetBuilder::new()
    ///     .with_standard_ops()
    ///     .build();
    /// ```
    #[inline]
    pub fn build(self) -> OperationSet {
        OperationSet {
            operations: self.operations,
        }
    }
}

impl Default for OperationSetBuilder {
    /// Create a new empty builder (default).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::SubstitutionSet;

    #[test]
    fn test_empty_operation_set() {
        let ops = OperationSet::new();
        assert_eq!(ops.len(), 0);
        assert!(ops.is_empty());
    }

    #[test]
    fn test_standard_operations() {
        let ops = OperationSet::standard();
        assert_eq!(ops.len(), 4);
        assert!(!ops.is_empty());

        let names: Vec<_> = ops.iter().map(|op| op.name()).collect();
        assert!(names.contains(&"match"));
        assert!(names.contains(&"substitute"));
        assert!(names.contains(&"insert"));
        assert!(names.contains(&"delete"));
    }

    #[test]
    fn test_with_transposition() {
        let ops = OperationSet::with_transposition();
        assert_eq!(ops.len(), 5);

        let names: Vec<_> = ops.iter().map(|op| op.name()).collect();
        assert!(names.contains(&"transpose"));
    }

    #[test]
    fn test_with_merge_split() {
        let ops = OperationSet::with_merge_split();
        assert_eq!(ops.len(), 6);

        let names: Vec<_> = ops.iter().map(|op| op.name()).collect();
        assert!(names.contains(&"merge"));
        assert!(names.contains(&"split"));
    }

    #[test]
    fn test_builder_standard() {
        let ops = OperationSetBuilder::new().with_standard_ops().build();

        assert_eq!(ops.len(), 4);
    }

    #[test]
    fn test_builder_custom() {
        let ops = OperationSetBuilder::new()
            .with_match()
            .with_operation(OperationType::new(1, 1, 0.5, "cheap_subst"))
            .with_insertion()
            .build();

        assert_eq!(ops.len(), 3);

        let custom_op = ops
            .operations()
            .iter()
            .find(|op| op.name() == "cheap_subst")
            .expect("Custom operation should exist");

        assert_eq!(custom_op.weight(), 0.5);
    }

    #[test]
    fn test_builder_phonetic() {
        let mut phonetic = SubstitutionSet::new();
        phonetic.allow_str("ph", "f");

        let ops = OperationSetBuilder::new()
            .with_match()
            .with_operation(OperationType::with_restriction(
                2, 1, 0.15, phonetic, "ph_to_f",
            ))
            .with_standard_ops()
            .build();

        // Builder allows duplicates: match (explicit) + ph_to_f + standard_ops (match + substitute + insert + delete)
        assert_eq!(ops.len(), 6);

        let ph_op = ops
            .operations()
            .iter()
            .find(|op| op.name() == "ph_to_f")
            .expect("Phonetic operation should exist");

        assert!(ph_op.is_restricted());
        assert_eq!(ph_op.consume_x(), 2);
        assert_eq!(ph_op.consume_y(), 1);
    }

    #[test]
    fn test_default_is_standard() {
        let ops = OperationSet::default();
        assert_eq!(ops.len(), 4);
    }

    #[test]
    fn test_iteration() {
        let ops = OperationSet::standard();
        let count = ops.iter().count();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_add_operation() {
        let mut ops = OperationSet::new();
        ops.add(OperationType::new(1, 1, 0.0, "match"));
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn hamming_has_infinite_empty_side_rates() {
        let operations = OperationSetBuilder::new()
            .with_match()
            .with_substitution()
            .build();

        assert_eq!(operations.rho_del(), Ok(EmptySideRate::Infinite));
        assert_eq!(operations.rho_ins(), Ok(EmptySideRate::Infinite));
        assert!(operations.rho_del().unwrap().fits_budget(0, 3));
        assert!(!operations.rho_del().unwrap().fits_budget(1, 3));
    }

    #[test]
    fn complete_set_validation_checks_progress_weights_and_aggregate_bound() {
        assert!(OperationSet::standard().validate().is_ok());
        assert!(OperationSet::hamming().validate().is_ok());
        assert!(OperationSet::indel().validate().is_ok());
        assert!(OperationSet::bounded_skip().validate().is_ok());

        let no_progress = OperationSetBuilder::new()
            .with_operation(OperationType::new(0, 0, 1.0, "cycle"))
            .build();
        assert!(matches!(
            no_progress.validate(),
            Err(OperationSetValidationError::NoProgress { .. })
        ));

        let non_finite = OperationSetBuilder::new()
            .with_operation(OperationType::new(1, 1, f64::INFINITY, "infinite"))
            .build();
        assert!(matches!(
            non_finite.validate(),
            Err(OperationSetValidationError::InvalidWeight { .. })
        ));

        let overflowing = OperationSetBuilder::new()
            .with_operation(OperationType::new(usize::MAX, 1, 1.0, "overflow"))
            .build();
        assert!(matches!(
            overflowing.validate(),
            Err(OperationSetValidationError::ConsumptionOverflow { .. })
        ));

        let mut excessive = OperationSet::new();
        for _ in 0..=(MAX_OPERATION_SET_TOTAL_CONSUMPTION / 2) {
            excessive.add(OperationType::new(1, 1, 1.0, "bounded"));
        }
        assert!(matches!(
            excessive.validate(),
            Err(OperationSetValidationError::ConsumptionLimit { .. })
        ));
    }

    #[test]
    fn empty_side_rates_choose_the_least_exact_per_scalar_cost() {
        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(1, 0, 1.0, "delete_one"))
            .with_operation(OperationType::new(3, 0, 1.5, "delete_three"))
            .with_operation(OperationType::new(0, 2, 0.5, "insert_two"))
            .build();

        assert_eq!(
            operations.rho_del(),
            Ok(EmptySideRate::Finite {
                numerator: 1,
                denominator: 2,
            })
        );
        assert_eq!(
            operations.rho_ins(),
            Ok(EmptySideRate::Finite {
                numerator: 1,
                denominator: 4,
            })
        );
        assert!(operations.rho_del().unwrap().fits_budget(4, 2));
        assert!(!operations.rho_del().unwrap().fits_budget(5, 2));
        assert_eq!(operations.rho_ins().unwrap().max_consumable(1), 4);
    }
}
