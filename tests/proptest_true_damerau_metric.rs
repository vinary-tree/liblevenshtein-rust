use liblevenshtein::distance::damerau_levenshtein_distance;
use proptest::prelude::*;

fn word() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(vec!['a', 'b', 'c', 'é']), 0..=7)
        .prop_map(|characters| characters.into_iter().collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn identity(value in word()) {
        prop_assert_eq!(damerau_levenshtein_distance(&value, &value), 0);
    }

    #[test]
    fn indiscernibility(left in word(), right in word()) {
        let distance = damerau_levenshtein_distance(&left, &right);
        prop_assert_eq!(distance == 0, left == right);
    }

    #[test]
    fn symmetry(left in word(), right in word()) {
        prop_assert_eq!(
            damerau_levenshtein_distance(&left, &right),
            damerau_levenshtein_distance(&right, &left)
        );
    }

    #[test]
    fn triangle_inequality(left in word(), middle in word(), right in word()) {
        let direct = damerau_levenshtein_distance(&left, &right);
        let via = damerau_levenshtein_distance(&left, &middle)
            .saturating_add(damerau_levenshtein_distance(&middle, &right));
        prop_assert!(direct <= via);
    }
}
