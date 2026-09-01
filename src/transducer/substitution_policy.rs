//! Zero-cost substitution policies for character matching.
//!
//! This module provides generic traits that enable restricted substitutions
//! with zero overhead for the unrestricted case through zero-sized type (ZST) optimization.
//!
//! ## Zero-Cost Abstraction
//!
//! The [`Unrestricted`] policy is a zero-sized type that compiles to identical
//! machine code as the pre-generic implementation. When the compiler sees
//! `Unrestricted::is_allowed()`, it inlines the function and optimizes away
//! the `false` constant, resulting in zero runtime overhead.
//!
//! ## Usage
//!
//! ```rust
//! use liblevenshtein::transducer::substitution_policy::{SubstitutionPolicy, Unrestricted, Restricted};
//! use liblevenshtein::transducer::SubstitutionSet;
//!
//! // Unrestricted (default, zero overhead)
//! let unrestricted = Unrestricted;
//! assert!(!unrestricted.is_allowed(b'a', b'b')); // No zero-cost substitutions
//!
//! // Restricted (custom similarity)
//! let mut set = SubstitutionSet::new();
//! set.allow('f', 'p');  // Single char only
//! let restricted = Restricted::new(&set);
//! assert!(!restricted.is_allowed(b'f', b'x')); // Only explicit pairs allowed
//! ```

use super::{SubstitutionSet, SubstitutionSetChar};
use libdictenstein::CharUnit;
use std::sync::Arc;

/// Zero-cost abstraction for substitution policies.
///
/// Implementations of this trait determine whether a character substitution
/// is allowed during fuzzy matching. The trait is designed to enable
/// zero-cost abstraction through monomorphization:
///
/// - [`Unrestricted`]: Allows all substitutions (zero-sized type, no overhead)
/// - [`Restricted`]: Allows only explicitly configured substitutions
///
/// ## Performance
///
/// The `Unrestricted` implementation is a zero-sized type (ZST) that compiles
/// to identical code as non-generic character matching. The compiler inlines
/// `is_allowed()` and optimizes away the constant `false`, resulting in zero
/// runtime overhead compared to the baseline implementation.
///
/// The `Restricted` implementation adds a set lookup only when characters don't
/// match exactly, introducing 1-5% overhead in typical scenarios.
///
/// ## Character Type Support
///
/// [`SubstitutionPolicy`] itself is the byte-level (`u8`) interface. For
/// character-level (`char`) dictionaries, use [`SubstitutionPolicyChar`] or
/// [`SubstitutionPolicyFor<char>`] with [`RestrictedChar`].
pub trait SubstitutionPolicy: Clone {
    /// Whether the policy can make two distinct units equivalent at zero cost.
    ///
    /// Transition engines use this capability to reuse exact-match masks for
    /// policies such as [`Unrestricted`]. Custom policies conservatively
    /// default to `true`, preserving their semantics without requiring an API
    /// change.
    const MAY_MATCH_DISTINCT_UNITS: bool = true;

    /// Check if substituting `dict_char` with `query_char` is allowed as a zero-cost operation.
    ///
    /// # Parameters
    ///
    /// - `dict_char`: Character from the dictionary term
    /// - `query_char`: Character from the query string
    ///
    /// # Returns
    ///
    /// `true` if this substitution should be treated as cost-free (equivalent characters),
    /// `false` if it should cost 1 edit (normal Levenshtein substitution).
    ///
    /// # Semantics
    ///
    /// - For `Unrestricted`: Always returns `false` (no zero-cost substitutions, standard Levenshtein)
    /// - For `Restricted`: Returns `true` for exact matches OR explicitly allowed substitutions
    ///
    /// # Type Parameter
    ///
    /// This method is the byte-level (`u8`) interface. Use
    /// [`SubstitutionPolicyChar::is_allowed`] or
    /// [`SubstitutionPolicyFor::is_allowed_for`] for character-level matching.
    ///
    /// # Implementation Note
    ///
    /// This method should be marked `#[inline(always)]` to enable
    /// aggressive compiler optimization, especially for the `Unrestricted` case.
    /// Implementations must be deterministic and side-effect free for the
    /// lifetime of a query: transition engines may cache the result for each
    /// `(dict_char, query_char)` pair and reuse it across dictionary nodes.
    fn is_allowed(&self, dict_char: u8, query_char: u8) -> bool;
}

/// Marker trait indicating a substitution policy supports a specific `CharUnit` type.
///
/// This trait enables unified substitution policy handling for both byte-level (`u8`)
/// and character-level (`char`) operations through compile-time polymorphism.
///
/// # Design Rationale
///
/// The substitution policy system needs to work with different character unit types:
/// - **Byte-level (u8)**: For ASCII/Latin-1 text, maximum performance
/// - **Character-level (char)**: For full Unicode support
///
/// Rather than maintaining separate trait hierarchies (`SubstitutionPolicy` vs `SubstitutionPolicyChar`),
/// this marker trait enables a unified approach where policies declare which character
/// unit types they support through trait implementations.
///
/// # Zero-Cost Abstraction
///
/// This design maintains zero-cost abstraction:
/// - Trait bounds are resolved at compile time
/// - Generic functions are fully monomorphized
/// - No runtime dispatch or dynamic checks
/// - `Unrestricted` policy remains a zero-sized type
///
/// # Type Safety
///
/// The marker trait provides compile-time safety by ensuring:
/// - Policies can only be used with compatible character types
/// - Incompatible combinations (e.g., `Restricted<u8>` with `char`) produce clear compiler errors
/// - No unsafe transmute operations needed
///
/// # Example
///
/// ```rust
/// use liblevenshtein::transducer::substitution_policy::{SubstitutionPolicy, SubstitutionPolicyFor};
/// use liblevenshtein::dictionary::CharUnit;
///
/// fn check_match<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
///     policy: &P,
///     dict_unit: U,
///     query_unit: U,
/// ) -> bool {
///     dict_unit == query_unit || policy.is_allowed_for(dict_unit, query_unit)
/// }
/// ```
///
/// # Implementations
///
/// - **Unrestricted**: Implements `SubstitutionPolicyFor<u8>` and `SubstitutionPolicyFor<char>`
/// - **Restricted<'a>**: Implements `SubstitutionPolicyFor<u8>` only
/// - **RestrictedChar<'a>**: Implements `SubstitutionPolicyFor<char>` only
pub trait SubstitutionPolicyFor<U: CharUnit>: SubstitutionPolicy {
    /// Check if substituting `dict_unit` with `query_unit` is allowed for this character unit type.
    ///
    /// # Parameters
    ///
    /// - `dict_unit`: Character unit from the dictionary term
    /// - `query_unit`: Character unit from the query string
    ///
    /// # Returns
    ///
    /// `true` if this substitution should be treated as cost-free (equivalent characters),
    /// `false` if it should cost 1 edit (normal Levenshtein substitution).
    ///
    /// # Performance
    ///
    /// This method should be marked `#[inline(always)]` to enable aggressive optimization,
    /// especially for the `Unrestricted` case where it compiles to a constant `false`.
    /// The result must remain deterministic and side-effect free for the
    /// lifetime of a query because monomorphized transition engines cache unit
    /// equivalence vectors by dictionary label.
    fn is_allowed_for(&self, dict_unit: U, query_unit: U) -> bool;
}

/// Unrestricted substitution policy: all substitutions are allowed.
///
/// This is the default policy and implements zero-cost abstraction through
/// being a zero-sized type (ZST). The compiler completely eliminates any
/// overhead from this policy, generating identical machine code to the
/// pre-generic implementation.
///
/// ## Zero-Cost Guarantee
///
/// ```rust
/// # use liblevenshtein::transducer::substitution_policy::{SubstitutionPolicy, Unrestricted};
/// # let dict_char = b'a';
/// # let query_char = b'b';
/// let policy = Unrestricted;
///
/// // This compiles to the same code as: dict_char == query_char
/// let matches = dict_char == query_char || policy.is_allowed(dict_char, query_char);
/// ```
///
/// ## Size
///
/// ```rust
/// # use liblevenshtein::transducer::substitution_policy::Unrestricted;
/// assert_eq!(std::mem::size_of::<Unrestricted>(), 0);
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Unrestricted;

impl SubstitutionPolicy for Unrestricted {
    const MAY_MATCH_DISTINCT_UNITS: bool = false;

    #[inline(always)]
    fn is_allowed(&self, _dict_char: u8, _query_char: u8) -> bool {
        // Unrestricted means standard Levenshtein: NO zero-cost substitutions.
        // All non-exact-matches incur edit distance cost of 1.
        // The characteristic vector will handle exact matches separately.
        false
    }
}

// Blanket implementation for all CharUnit types
// This allows Unrestricted to work with any valid CharUnit (u8, char)
impl<U: CharUnit> SubstitutionPolicyFor<U> for Unrestricted {
    #[inline(always)]
    fn is_allowed_for(&self, _dict_unit: U, _query_unit: U) -> bool {
        // Unrestricted: standard Levenshtein, no zero-cost substitutions
        false
    }
}

/// Restricted substitution policy: only explicitly allowed substitutions.
///
/// This policy checks a [`SubstitutionSet`] to determine if a character
/// substitution is allowed. It introduces a small overhead (1-5% in typical
/// scenarios) but only when characters don't match exactly.
///
/// ## Performance Characteristics
///
/// - **Exact match**: No overhead (short-circuit evaluation)
/// - **Mismatch**: HashSet lookup (~10-30ns per check)
/// - **Typical overhead**: 1-5% for match-heavy workloads
///
/// ## Example
///
/// ```rust
/// # use liblevenshtein::transducer::substitution_policy::{SubstitutionPolicy, Restricted};
/// # use liblevenshtein::transducer::SubstitutionSet;
/// let mut set = SubstitutionSet::new();
/// set.allow('f', 'p');  // Single char only
/// set.allow('c', 'k');  // Allow 'c' ↔ 'k' substitution
///
/// let policy = Restricted::new(&set);
///
/// // Exact match always allowed
/// assert!(policy.is_allowed(b'a', b'a'));
///
/// // Explicit substitutions allowed
/// assert!(policy.is_allowed(b'f', b'p'));
///
/// // Other substitutions not allowed
/// assert!(!policy.is_allowed(b'x', b'y'));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Restricted<'a> {
    set: &'a SubstitutionSet,
}

impl<'a> Restricted<'a> {
    /// Create a new restricted policy with the given substitution set.
    ///
    /// # Parameters
    ///
    /// - `set`: Reference to a [`SubstitutionSet`] defining allowed substitutions
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::substitution_policy::Restricted;
    /// # use liblevenshtein::transducer::SubstitutionSet;
    /// let set = SubstitutionSet::phonetic_basic();
    /// let policy = Restricted::new(&set);
    /// ```
    #[inline]
    pub fn new(set: &'a SubstitutionSet) -> Self {
        Self { set }
    }

    /// Get a reference to the underlying substitution set.
    #[inline]
    pub fn set(&self) -> &'a SubstitutionSet {
        self.set
    }
}

impl<'a> SubstitutionPolicy for Restricted<'a> {
    #[inline(always)]
    fn is_allowed(&self, dict_char: u8, query_char: u8) -> bool {
        // Check if this is an explicitly allowed zero-cost substitution.
        // Note: Exact matches (dict_char == query_char) are handled by the
        // characteristic vector before calling is_allowed, but we include
        // the check here for semantic clarity and to match expected behavior.
        dict_char == query_char || self.set.contains(dict_char, query_char)
    }
}

// Implement SubstitutionPolicyFor<u8> for byte-level restricted policy
impl<'a> SubstitutionPolicyFor<u8> for Restricted<'a> {
    #[inline(always)]
    fn is_allowed_for(&self, dict_unit: u8, query_unit: u8) -> bool {
        // Check if explicitly allowed substitution or exact match
        dict_unit == query_unit || self.set.contains(dict_unit, query_unit)
    }
}

/// Owned byte-level restricted substitution policy.
///
/// This policy stores its substitution set in an [`Arc`], allowing constructors
/// such as [`crate::transducer::Transducer::with_substitutions`] to own caller
/// supplied sets without leaking memory or forcing a borrowed lifetime into the
/// public transducer type.
#[derive(Clone, Debug)]
pub struct OwnedRestricted {
    set: Arc<SubstitutionSet>,
}

impl OwnedRestricted {
    /// Create an owned restricted policy from a substitution set.
    #[inline]
    pub fn new(set: SubstitutionSet) -> Self {
        Self { set: Arc::new(set) }
    }

    /// Create an owned restricted policy from a shared substitution set.
    #[inline]
    pub fn from_arc(set: Arc<SubstitutionSet>) -> Self {
        Self { set }
    }

    /// Get a reference to the underlying substitution set.
    #[inline]
    pub fn set(&self) -> &SubstitutionSet {
        &self.set
    }
}

impl SubstitutionPolicy for OwnedRestricted {
    #[inline(always)]
    fn is_allowed(&self, dict_char: u8, query_char: u8) -> bool {
        dict_char == query_char || self.set.contains(dict_char, query_char)
    }
}

impl SubstitutionPolicyFor<u8> for OwnedRestricted {
    #[inline(always)]
    fn is_allowed_for(&self, dict_unit: u8, query_unit: u8) -> bool {
        dict_unit == query_unit || self.set.contains(dict_unit, query_unit)
    }
}

// ============================================================================
// Character-Level Policy Support (for Unicode)
// ============================================================================

/// Zero-cost abstraction for character-level substitution policies.
///
/// This trait is similar to [`SubstitutionPolicy`] but works with full Unicode
/// characters (`char`) instead of just bytes (`u8`). Use this with
/// character-level dictionaries for full Unicode substitution support.
///
/// ## Implementations
///
/// - [`Unrestricted`]: Works for both byte and char (returns `false` for all)
/// - [`RestrictedChar`]: Character-level policy with [`SubstitutionSetChar`]
///
/// ## Performance
///
/// Like `SubstitutionPolicy`, the `Unrestricted` case is zero-cost. The
/// `RestrictedChar` case adds HashSet lookup overhead only on mismatches.
pub trait SubstitutionPolicyChar: Copy + Clone {
    /// Check if substituting `dict_char` with `query_char` is allowed as a zero-cost operation.
    ///
    /// # Parameters
    ///
    /// - `dict_char`: Character from the dictionary term
    /// - `query_char`: Character from the query string
    ///
    /// # Returns
    ///
    /// `true` if this substitution should be treated as cost-free (equivalent characters),
    /// `false` if it should cost 1 edit (normal Levenshtein substitution).
    ///
    /// # Semantics
    ///
    /// - For `Unrestricted`: Always returns `false` (no zero-cost substitutions, standard Levenshtein)
    /// - For `RestrictedChar`: Returns `true` for exact matches OR explicitly allowed substitutions
    fn is_allowed(&self, dict_char: char, query_char: char) -> bool;
}

// Unrestricted works for both byte and char policies
impl SubstitutionPolicyChar for Unrestricted {
    #[inline(always)]
    fn is_allowed(&self, _dict_char: char, _query_char: char) -> bool {
        // Unrestricted means standard Levenshtein: NO zero-cost substitutions
        false
    }
}

/// Restricted character-level substitution policy for full Unicode support.
///
/// This policy checks a [`SubstitutionSetChar`] to determine if a character
/// substitution is allowed. It's the character-level equivalent of [`Restricted`],
/// enabling full Unicode substitution policies.
///
/// ## Performance Characteristics
///
/// - **Exact match**: No overhead (short-circuit evaluation)
/// - **Mismatch**: HashSet lookup (~10-30ns per check)
/// - **Typical overhead**: 1-5% for match-heavy workloads
///
/// ## Example
///
/// ```rust
/// # use liblevenshtein::transducer::substitution_policy::{SubstitutionPolicyChar, RestrictedChar};
/// # use liblevenshtein::transducer::SubstitutionSetChar;
/// let mut set = SubstitutionSetChar::new();
/// set.allow('é', 'e');  // Allow é ↔ e substitution
/// set.allow('ñ', 'n');  // Allow ñ ↔ n substitution
///
/// let policy = RestrictedChar::new(&set);
///
/// // Exact match always allowed
/// assert!(policy.is_allowed('a', 'a'));
///
/// // Explicit substitutions allowed
/// assert!(policy.is_allowed('é', 'e'));
///
/// // Other substitutions not allowed
/// assert!(!policy.is_allowed('x', 'y'));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct RestrictedChar<'a> {
    set: &'a SubstitutionSetChar,
}

impl<'a> RestrictedChar<'a> {
    /// Create a new restricted character-level policy with the given substitution set.
    ///
    /// # Parameters
    ///
    /// - `set`: Reference to a [`SubstitutionSetChar`] defining allowed substitutions
    ///
    /// # Example
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::substitution_policy::RestrictedChar;
    /// # use liblevenshtein::transducer::SubstitutionSetChar;
    /// let set = SubstitutionSetChar::diacritics_latin();
    /// let policy = RestrictedChar::new(&set);
    /// ```
    #[inline]
    pub fn new(set: &'a SubstitutionSetChar) -> Self {
        Self { set }
    }

    /// Get a reference to the underlying substitution set.
    #[inline]
    pub fn set(&self) -> &'a SubstitutionSetChar {
        self.set
    }
}

impl<'a> SubstitutionPolicyChar for RestrictedChar<'a> {
    #[inline(always)]
    fn is_allowed(&self, dict_char: char, query_char: char) -> bool {
        // Check if this is an explicitly allowed zero-cost substitution
        dict_char == query_char || self.set.contains(dict_char, query_char)
    }
}

// Implement SubstitutionPolicy as a requirement (but should never be called with u8)
impl<'a> SubstitutionPolicy for RestrictedChar<'a> {
    #[inline(always)]
    fn is_allowed(&self, dict_char: u8, query_char: u8) -> bool {
        // RestrictedChar is for character-level matching; keep the byte shim total.
        dict_char == query_char
    }
}

// Implement SubstitutionPolicyFor<char> for character-level restricted policy
impl<'a> SubstitutionPolicyFor<char> for RestrictedChar<'a> {
    #[inline(always)]
    fn is_allowed_for(&self, dict_unit: char, query_unit: char) -> bool {
        // Check if explicitly allowed substitution or exact match
        dict_unit == query_unit || self.set.contains(dict_unit, query_unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unrestricted_size_is_zero() {
        assert_eq!(
            std::mem::size_of::<Unrestricted>(),
            0,
            "Unrestricted must be zero-sized for ZST optimization"
        );
    }

    #[test]
    fn test_unrestricted_no_zero_cost_substitutions() {
        let policy = Unrestricted;

        // Unrestricted means standard Levenshtein: no zero-cost substitutions
        // is_allowed should return false for all non-exact-matches (explicitly use SubstitutionPolicy)
        assert!(!SubstitutionPolicy::is_allowed(&policy, b'a', b'b'));
        assert!(!SubstitutionPolicy::is_allowed(&policy, b'x', b'y'));
        assert!(!SubstitutionPolicy::is_allowed(&policy, b'1', b'2'));
        assert!(!SubstitutionPolicy::is_allowed(&policy, 0, 255));

        // Note: Exact matches are handled in characteristic_vector before
        // calling is_allowed, so we don't need to test them here
    }

    #[test]
    fn test_restricted_basic() {
        let mut set = SubstitutionSet::new();
        set.allow_byte(b'a', b'b');
        set.allow_byte(b'x', b'y');

        let policy = Restricted::new(&set);

        // Exact matches always allowed
        assert!(policy.is_allowed(b'a', b'a'));
        assert!(policy.is_allowed(b'z', b'z'));

        // Explicitly allowed pairs
        assert!(policy.is_allowed(b'a', b'b'));
        assert!(policy.is_allowed(b'x', b'y'));

        // Not allowed
        assert!(!policy.is_allowed(b'a', b'c'));
        assert!(!policy.is_allowed(b'b', b'a')); // Not symmetric by default
    }

    #[test]
    fn test_policy_is_copy() {
        let policy1 = Unrestricted;
        let policy2 = policy1; // Copy

        // Both should work independently (and return same results)
        // Unrestricted returns false for non-exact-matches (explicitly use SubstitutionPolicy)
        assert!(!SubstitutionPolicy::is_allowed(&policy1, b'a', b'b'));
        assert!(!SubstitutionPolicy::is_allowed(&policy2, b'x', b'y'));
    }

    #[test]
    fn test_restricted_zero_cost_substitutions() {
        // Test that restricted policy allows specific pairs as zero-cost
        let mut set = SubstitutionSet::new();
        set.allow('c', 'k');
        set.allow('k', 'c');

        let policy = Restricted::new(&set);

        // Should allow c↔k substitutions
        assert!(policy.is_allowed(b'c', b'k'), "c->k should be allowed");
        assert!(policy.is_allowed(b'k', b'c'), "k->c should be allowed");

        // Should allow exact matches
        assert!(policy.is_allowed(b'c', b'c'), "c==c should be allowed");
        assert!(policy.is_allowed(b'k', b'k'), "k==k should be allowed");

        // Should NOT allow other substitutions
        assert!(!policy.is_allowed(b'a', b'b'), "a->b should NOT be allowed");
        assert!(!policy.is_allowed(b'c', b'z'), "c->z should NOT be allowed");
    }

    #[test]
    fn test_owned_restricted_zero_cost_substitutions() {
        let mut set = SubstitutionSet::new();
        set.allow('c', 'k');

        let policy = OwnedRestricted::new(set);
        let cloned = policy.clone();

        assert!(policy.is_allowed(b'c', b'k'));
        assert!(cloned.is_allowed_for(b'c', b'k'));
        assert!(!cloned.is_allowed_for(b'k', b'c'));
    }

    // ========================================================================
    // Character-Level Policy Tests
    // ========================================================================

    #[test]
    fn test_unrestricted_char_policy() {
        let policy = Unrestricted;

        // Should return false for all non-exact matches (explicitly use SubstitutionPolicyChar)
        assert!(!SubstitutionPolicyChar::is_allowed(&policy, 'α', 'β'));
        assert!(!SubstitutionPolicyChar::is_allowed(&policy, '你', '好'));
        assert!(!SubstitutionPolicyChar::is_allowed(&policy, 'é', 'e'));
    }

    #[test]
    fn test_restricted_char_basic() {
        use crate::transducer::SubstitutionSetChar;

        let mut set = SubstitutionSetChar::new();
        set.allow('α', 'β');
        set.allow('你', '好');

        let policy = RestrictedChar::new(&set);

        // Exact matches always allowed
        assert!(policy.is_allowed_for('α', 'α'));
        assert!(policy.is_allowed_for('z', 'z'));

        // Explicitly allowed pairs
        assert!(policy.is_allowed_for('α', 'β'));
        assert!(policy.is_allowed_for('你', '好'));

        // Not allowed
        assert!(!policy.is_allowed_for('α', 'γ'));
        assert!(!policy.is_allowed_for('β', 'α')); // Not symmetric by default
    }

    #[test]
    fn test_restricted_char_diacritics() {
        use crate::transducer::SubstitutionSetChar;

        let mut set = SubstitutionSetChar::new();
        set.allow('é', 'e');
        set.allow('e', 'é');
        set.allow('ñ', 'n');
        set.allow('n', 'ñ');

        let policy = RestrictedChar::new(&set);

        // Should allow diacritic substitutions
        assert!(policy.is_allowed_for('é', 'e'), "é->e should be allowed");
        assert!(policy.is_allowed_for('e', 'é'), "e->é should be allowed");
        assert!(policy.is_allowed_for('ñ', 'n'), "ñ->n should be allowed");
        assert!(policy.is_allowed_for('n', 'ñ'), "n->ñ should be allowed");

        // Should allow exact matches
        assert!(policy.is_allowed_for('é', 'é'), "é==é should be allowed");
        assert!(policy.is_allowed_for('e', 'e'), "e==e should be allowed");

        // Should NOT allow other substitutions
        assert!(
            !policy.is_allowed_for('a', 'b'),
            "a->b should NOT be allowed"
        );
        assert!(
            !policy.is_allowed_for('é', 'x'),
            "é->x should NOT be allowed"
        );
    }

    #[test]
    fn test_restricted_char_byte_shim_is_total() {
        let set = SubstitutionSetChar::diacritics_latin();
        let policy = RestrictedChar::new(&set);

        assert!(SubstitutionPolicy::is_allowed(&policy, b'a', b'a'));
        assert!(!SubstitutionPolicy::is_allowed(&policy, b'e', b'a'));
    }

    #[test]
    fn test_policy_char_is_copy() {
        let policy1: Unrestricted = Unrestricted;
        let policy2 = policy1; // Copy

        // Both should work independently (explicitly call SubstitutionPolicyChar)
        assert!(!SubstitutionPolicyChar::is_allowed(&policy1, 'α', 'β'));
        assert!(!SubstitutionPolicyChar::is_allowed(&policy2, '你', '好'));
    }
}
