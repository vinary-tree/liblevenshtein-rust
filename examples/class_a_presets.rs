//! Alignment-expressible Hamming, indel, and bounded-skip semantics.

use liblevenshtein::distance::{hamming_distance, indel_distance, indel_distance_bounded};
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{OperationSet, OperationSetBuilder, OperationType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Hamming has no insertion or deletion path, so unequal lengths are outside
    // its domain rather than merely expensive.
    assert_eq!(hamming_distance("abc", "bca"), Some(3));
    assert_eq!(hamming_distance("abc", "ab"), None);
    let hamming = GeneralizedAutomaton::try_with_operations(3, OperationSet::hamming())?;
    assert_eq!(hamming.scaled_distance("abc", "bca")?, Some(3));
    assert_eq!(hamming.scaled_distance("abc", "ab")?, None);

    // Replacing one scalar under indel costs delete plus insert.
    assert_eq!(indel_distance("a", "b"), 2);
    assert_eq!(indel_distance_bounded("abc", "bca", 1), None);
    assert_eq!(indel_distance_bounded("abc", "bca", 2), Some(2));
    let indel = GeneralizedAutomaton::try_with_operations(2, OperationSet::indel())?;
    assert_eq!(indel.scaled_distance("abc", "bca")?, Some(2));

    // Direction matters: the second argument must be a subsequence of the
    // first, and the cost is the number of skipped source scalars.
    let skip = GeneralizedAutomaton::try_with_operations(2, OperationSet::bounded_skip())?;
    assert_eq!(skip.scaled_distance("crate", "cat")?, Some(2));
    assert_eq!(skip.scaled_distance("cat", "crate")?, None);

    // Generated or untrusted rule collections cross an explicit validation
    // boundary. A positive-cost cycle is rejected before traversal.
    let invalid = OperationSetBuilder::new()
        .with_operation(OperationType::new(0, 0, 1.0, "cycle"))
        .build();
    assert!(invalid.validate().is_err());

    println!("Hamming, indel, bounded skip, and validation examples passed");
    Ok(())
}
