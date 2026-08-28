//! Phonetic operations for English approximate string matching.
//!
//! This module provides preset operation sets for English phonetic corrections,
//! targeting 60-70% coverage of common English phonetic transformations.
//!
//! # Phase 1: Basic Phonetic Operations
//!
//! The current implementation includes:
//! - **Consonant digraphs**: ch→k, sh→s, ph→f, th→t, etc.
//! - **Initial clusters**: wr→r, wh→w, kn→n, ps→s, etc.
//! - **Common confusions**: c/k, c/s, s/z, g/j
//!
//! These rules are fully modelable within the TCS 2011 generalized operations
//! framework and provide good coverage for common English words.
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::phonetic::phonetic_english_basic;
//! use liblevenshtein::transducer::OperationSetBuilder;
//!
//! // Build operation set with standard ops + phonetic corrections
//! let mut builder = OperationSetBuilder::new().with_standard_ops();
//! for op in phonetic_english_basic().operations() {
//!     builder = builder.with_operation(op.clone());
//! }
//! let ops = builder.build();
//! ```
//!
//! # Future Phases
//!
//! - **Phase 2**: Context-dependent operations (c→s before front vowels)
//! - **Phase 3**: Vowel reductions and schwa handling
//!
//! See: `docs/research/phonetic-corrections/ENGLISH_PHONETIC_FEASIBILITY.md`

use crate::transducer::{OperationSet, OperationSetBuilder, OperationType, SubstitutionSet};

/// English consonant digraphs (2→1 and 1→2 mappings).
///
/// Maps common two-letter consonant combinations to their single-character
/// phonetic equivalents. These are fully modelable and context-free.
///
/// # Mappings
///
/// - `ch → k`: church → kurk
/// - `sh → s`: ship → sip
/// - `ph → f`: phone → fon
/// - `th → t`: think → tink
/// - `qu → kw`: queen → kwen  (2→2, handled separately)
///
/// # Coverage
///
/// Affects ~15-20% of English words (high-frequency patterns).
pub fn consonant_digraphs() -> OperationSet {
    // 2→1 mappings (digraph to single char)
    let mut digraphs_2_to_1 = SubstitutionSet::new();
    digraphs_2_to_1.allow_str("ch", "k");
    digraphs_2_to_1.allow_str("sh", "s");
    digraphs_2_to_1.allow_str("ph", "f");
    digraphs_2_to_1.allow_str("th", "t");

    // 1→2 mappings (single char to digraph)
    let mut digraphs_1_to_2 = SubstitutionSet::new();
    digraphs_1_to_2.allow_str("k", "ch");
    digraphs_1_to_2.allow_str("s", "sh");
    digraphs_1_to_2.allow_str("f", "ph");
    digraphs_1_to_2.allow_str("t", "th");

    // 2→2 mapping (qu ↔ kw)
    let mut digraphs_2_to_2 = SubstitutionSet::new();
    digraphs_2_to_2.allow_str("qu", "kw");
    digraphs_2_to_2.allow_str("kw", "qu");

    OperationSetBuilder::new()
        .with_operation(OperationType::with_restriction(
            2,
            1,
            0.15,
            digraphs_2_to_1,
            "consonant_digraphs_2to1",
        ))
        .with_operation(OperationType::with_restriction(
            1,
            2,
            0.15,
            digraphs_1_to_2,
            "consonant_digraphs_1to2",
        ))
        .with_operation(OperationType::with_restriction(
            2,
            2,
            0.15,
            digraphs_2_to_2,
            "consonant_digraphs_2to2",
        ))
        .build()
}

/// Initial consonant clusters with silent letters.
///
/// Handles word-initial consonant clusters where one letter is typically silent.
/// These are position-dependent but can be approximated as context-free with
/// slightly higher cost.
///
/// # Mappings
///
/// - `wr → r`: write → rite
/// - `wh → w`: white → wite
/// - `kn → n`: knight → nite
/// - `ps → s`: psychology → sicology
/// - `pn → n`: pneumonia → numonia
/// - `gn → n`: gnome → nome
///
/// # Coverage
///
/// Affects ~5-8% of English words (less common but distinctive).
pub fn initial_clusters() -> OperationSet {
    // 2→1 mappings (cluster to single char)
    let mut clusters_2_to_1 = SubstitutionSet::new();
    clusters_2_to_1.allow_str("wr", "r");
    clusters_2_to_1.allow_str("wh", "w");
    clusters_2_to_1.allow_str("kn", "n");
    clusters_2_to_1.allow_str("ps", "s");
    clusters_2_to_1.allow_str("pn", "n");
    clusters_2_to_1.allow_str("gn", "n");
    clusters_2_to_1.allow_str("rh", "r");

    // 1→2 mappings (single char to cluster)
    let mut clusters_1_to_2 = SubstitutionSet::new();
    clusters_1_to_2.allow_str("r", "wr");
    clusters_1_to_2.allow_str("w", "wh");
    clusters_1_to_2.allow_str("n", "kn");
    clusters_1_to_2.allow_str("s", "ps");
    clusters_1_to_2.allow_str("n", "pn");
    clusters_1_to_2.allow_str("n", "gn");
    clusters_1_to_2.allow_str("r", "rh");

    OperationSetBuilder::new()
        .with_operation(OperationType::with_restriction(
            2,
            1,
            0.20,
            clusters_2_to_1,
            "initial_clusters_2to1",
        ))
        .with_operation(OperationType::with_restriction(
            1,
            2,
            0.20,
            clusters_1_to_2,
            "initial_clusters_1to2",
        ))
        .build()
}

/// Common single-character phonetic confusions.
///
/// Single-character substitutions for phonetically similar sounds.
/// These complement the standard substitution operation with phonetic awareness.
///
/// # Mappings
///
/// - `c ↔ k`: cat ↔ kat
/// - `c ↔ s`: cent ↔ sent (before front vowels)
/// - `s ↔ z`: dogs ↔ dogz
/// - `g ↔ j`: giant ↔ jiant
/// - `f ↔ v`: leaf ↔ leav
///
/// # Note
///
/// Some mappings are context-dependent (c→s before e/i/y) but are approximated
/// as context-free with appropriate weights.
pub fn phonetic_confusions() -> OperationSet {
    let mut confusions = SubstitutionSet::new();

    // Hard/soft C
    confusions.allow('c', 'k');
    confusions.allow('k', 'c');
    confusions.allow('c', 's');
    confusions.allow('s', 'c');

    // S/Z voicing
    confusions.allow('s', 'z');
    confusions.allow('z', 's');

    // G/J
    confusions.allow('g', 'j');
    confusions.allow('j', 'g');

    // F/V voicing
    confusions.allow('f', 'v');
    confusions.allow('v', 'f');

    // Common vowel reductions (simplified)
    confusions.allow('a', 'e');
    confusions.allow('e', 'a');
    confusions.allow('i', 'e');
    confusions.allow('e', 'i');

    let op = OperationType::with_restriction(
        1,
        1,
        0.25, // Moderate cost (context-dependent approximations)
        confusions,
        "phonetic_confusions",
    );

    OperationSetBuilder::new().with_operation(op).build()
}

/// Bidirectional double-consonant equivalence.
///
/// Allows matching between doubled and single consonants, which often sound
/// identical in English.
///
/// # Mappings
///
/// - Simplification: `bb → b`, `dd → d`, `ff → f`, `gg → g`, etc.
/// - Expansion: `b → bb`, `d → dd`, `f → ff`, `g → gg`, etc.
///
/// # Coverage
///
/// Affects ~10-15% of English words (very common pattern).
pub fn double_consonants() -> OperationSet {
    let mut simplifications = SubstitutionSet::new();
    let mut expansions = SubstitutionSet::new();

    // Common doubled consonants
    let consonants = [
        'b', 'c', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'z',
    ];

    for &c in &consonants {
        let double = format!("{}{}", c, c);
        let single = format!("{}", c);
        simplifications.allow_str(&double, &single);
        expansions.allow_str(&single, &double);
    }

    OperationSetBuilder::new()
        .with_operation(OperationType::with_restriction(
            2,
            1,
            0.10, // Very low cost (extremely common)
            simplifications,
            "double_consonants_2to1",
        ))
        .with_operation(OperationType::with_restriction(
            1,
            2,
            0.10,
            expansions,
            "double_consonants_1to2",
        ))
        .build()
}

/// Comprehensive English phonetic operation set (Phase 1).
///
/// Combines all Phase 1 phonetic operations for maximum coverage (~60-70%
/// of common English phonetic transformations).
///
/// # Includes
///
/// - Consonant digraphs (ch, sh, ph, th, qu)
/// - Initial clusters (wr, wh, kn, ps, etc.)
/// - Phonetic confusions (c/k, s/z, g/j, etc.)
/// - Double consonant simplification
///
/// # Example
///
/// ```rust
/// use liblevenshtein::transducer::phonetic::phonetic_english_basic;
/// use liblevenshtein::transducer::OperationSetBuilder;
///
/// let phonetic = phonetic_english_basic();
/// phonetic.validate().expect("the built-in preset is valid");
///
/// // Add a base metric when complete-string alignment also needs exact-match,
/// // insertion, deletion, and ordinary-substitution edges.
/// let mut builder = OperationSetBuilder::new().with_standard_ops();
/// for operation in phonetic.operations() {
///     builder = builder.with_operation(operation.clone());
/// }
/// let complete_metric = builder.build();
/// complete_metric.validate().expect("the combined metric is valid");
/// ```
///
/// # Performance
///
/// The preset contains eight operation classes. Each class groups its allowed
/// source/target pairs in one restriction table rather than creating one
/// operation value per spelling pair.
///
/// # Limitations
///
/// - ASCII-only (no special phonetic characters)
/// - Context-free approximations (some rules are context-dependent)
/// - Contains phonetic rules only; combine it with standard operations for a
///   complete edit metric
pub fn phonetic_english_basic() -> OperationSet {
    let mut builder = OperationSetBuilder::new();

    // Add all operations from each component set
    for op in consonant_digraphs().operations() {
        builder = builder.with_operation(op.clone());
    }
    for op in initial_clusters().operations() {
        builder = builder.with_operation(op.clone());
    }
    for op in phonetic_confusions().operations() {
        builder = builder.with_operation(op.clone());
    }
    for op in double_consonants().operations() {
        builder = builder.with_operation(op.clone());
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consonant_digraphs() {
        let ops = consonant_digraphs();
        assert!(!ops.is_empty());
        assert_eq!(ops.len(), 3); // Three operations: 2→1, 1→2, 2→2

        // Verify the operations exist
        let names: Vec<_> = ops.operations().iter().map(|op| op.name()).collect();
        assert!(names.contains(&"consonant_digraphs_2to1"));
        assert!(names.contains(&"consonant_digraphs_1to2"));
        assert!(names.contains(&"consonant_digraphs_2to2"));
    }

    #[test]
    fn test_initial_clusters() {
        let ops = initial_clusters();
        assert!(!ops.is_empty());
        assert_eq!(ops.len(), 2); // Two operations: 2→1, 1→2

        let names: Vec<_> = ops.operations().iter().map(|op| op.name()).collect();
        assert!(names.contains(&"initial_clusters_2to1"));
        assert!(names.contains(&"initial_clusters_1to2"));
    }

    #[test]
    fn test_phonetic_confusions() {
        let ops = phonetic_confusions();
        assert!(!ops.is_empty());
        assert_eq!(ops.len(), 1);

        let op = &ops.operations()[0];
        assert_eq!(op.name(), "phonetic_confusions");
        assert_eq!(op.consume_x(), 1);
        assert_eq!(op.consume_y(), 1);
    }

    #[test]
    fn test_double_consonants() {
        let ops = double_consonants();
        assert!(!ops.is_empty());
        assert_eq!(ops.len(), 2);
        ops.validate()
            .expect("both double-consonant directions have matching arity");

        let simplify = &ops.operations()[0];
        assert_eq!(simplify.name(), "double_consonants_2to1");
        assert_eq!((simplify.consume_x(), simplify.consume_y()), (2, 1));
        assert!(simplify.can_apply(b"ll", b"l"));

        let expand = &ops.operations()[1];
        assert_eq!(expand.name(), "double_consonants_1to2");
        assert_eq!((expand.consume_x(), expand.consume_y()), (1, 2));
        assert!(expand.can_apply(b"l", b"ll"));
    }

    #[test]
    fn test_phonetic_english_basic() {
        let ops = phonetic_english_basic();
        assert!(!ops.is_empty());
        // 3 digraph + 2 initial-cluster + 1 confusion + 2 double-consonant classes.
        assert_eq!(ops.len(), 8);
        ops.validate().expect("the built-in preset must be valid");

        // Verify all operations are included
        let names: Vec<_> = ops.operations().iter().map(|op| op.name()).collect();
        assert!(names.contains(&"consonant_digraphs_2to1"));
        assert!(names.contains(&"consonant_digraphs_1to2"));
        assert!(names.contains(&"consonant_digraphs_2to2"));
        assert!(names.contains(&"initial_clusters_2to1"));
        assert!(names.contains(&"initial_clusters_1to2"));
        assert!(names.contains(&"phonetic_confusions"));
        assert!(names.contains(&"double_consonants_2to1"));
        assert!(names.contains(&"double_consonants_1to2"));
    }

    #[test]
    fn test_can_apply_consonant_digraphs() {
        let ops = consonant_digraphs();

        // Find the 2→1 operation
        let op_2to1 = ops
            .operations()
            .iter()
            .find(|op| op.name() == "consonant_digraphs_2to1")
            .expect("Should have 2to1 operation");

        // Should be able to apply to "ph" → "f"
        assert!(op_2to1.can_apply(b"ph", b"f"));

        // Find the 1→2 operation
        let op_1to2 = ops
            .operations()
            .iter()
            .find(|op| op.name() == "consonant_digraphs_1to2")
            .expect("Should have 1to2 operation");

        // Should be able to apply bidirectionally
        assert!(op_1to2.can_apply(b"f", "ph".as_bytes()));

        // Should NOT apply to unrelated pairs
        assert!(!op_2to1.can_apply(b"xy", b"z"));
    }

    #[test]
    fn test_can_apply_initial_clusters() {
        let ops = initial_clusters();

        // Find the 2→1 operation
        let op_2to1 = ops
            .operations()
            .iter()
            .find(|op| op.name() == "initial_clusters_2to1")
            .expect("Should have 2to1 operation");

        // Should be able to apply to "wr" → "r"
        assert!(op_2to1.can_apply(b"wr", b"r"));

        // Should be able to apply to "kn" → "n"
        assert!(op_2to1.can_apply(b"kn", b"n"));

        // Find the 1→2 operation
        let op_1to2 = ops
            .operations()
            .iter()
            .find(|op| op.name() == "initial_clusters_1to2")
            .expect("Should have 1to2 operation");

        // Bidirectional
        assert!(op_1to2.can_apply(b"r", b"wr"));
    }

    #[test]
    fn test_operation_weights() {
        let digraphs = consonant_digraphs();
        let clusters = initial_clusters();
        let confusions = phonetic_confusions();
        let doubles = double_consonants();

        // Verify weight hierarchy (lower = more common/confident)
        assert!(doubles.operations()[0].weight() < digraphs.operations()[0].weight());
        assert!(digraphs.operations()[0].weight() < clusters.operations()[0].weight());
        assert!(clusters.operations()[0].weight() < confusions.operations()[0].weight());
    }
}
