//! Property-based tests for the online phonetic scanner (`src/phonetic/online_scanner.rs`).
//!
//! Mirrors the TLA+ `OnlineScanner` model (`docs/verification/tla/`), which
//! models bounded active matches, position monotonicity, and `NoMissedMatches`,
//! but abstracts the matching.
//!
//! We drive the real scanner with an EMPTY rewrite-rule set, so normalization is
//! the identity and the scanner performs plain fuzzy substring matching.
//!
//! Gated on the `phonetic-rules` feature; run with
//! `cargo test --features phonetic-rules`.
#![cfg(feature = "phonetic-rules")]

use liblevenshtein::phonetic::online_scanner::OnlinePhoneticScannerChar;
use proptest::prelude::*;

fn arb_query() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-c]{1,5}").unwrap()
}

fn arb_doc() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-c]{0,16}").unwrap()
}

fn arb_nonmatching_context() -> impl Strategy<Value = String> {
    prop::string::string_regex("[x-z]{0,8}").unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Boundedness and source correspondence: every completed match is within
    /// the configured error bound and reproduces its exact document slice.
    #[test]
    fn scanner_matches_within_bound(
        query in arb_query(),
        document in arb_doc(),
        max_distance in 0u8..=2,
    ) {
        let mut scanner = OnlinePhoneticScannerChar::new(&query, &[], max_distance);
        let matches = scanner.scan(&document);
        for m in &matches {
            prop_assert!(m.distance <= max_distance, "match distance {} exceeds bound {}", m.distance, max_distance);
            prop_assert!(m.byte_range.0 <= m.byte_range.1, "inverted byte range");
            prop_assert!(m.byte_range.1 <= document.len(), "byte range past end of document");
            prop_assert!(m.char_range.0 <= m.char_range.1, "inverted char range");
            prop_assert_eq!(&document[m.byte_range.0..m.byte_range.1], m.original_text.as_str());
        }
    }

    /// Determinism: scanning the same document with a fresh scanner is repeatable.
    #[test]
    fn scanner_is_deterministic(
        query in arb_query(),
        document in arb_doc(),
        max_distance in 0u8..=2,
    ) {
        let m1 = OnlinePhoneticScannerChar::new(&query, &[], max_distance).scan(&document);
        let m2 = OnlinePhoneticScannerChar::new(&query, &[], max_distance).scan(&document);
        prop_assert_eq!(m1, m2);
    }

    /// PositionMonotonicity: completed matches are ordered by start position,
    /// then by distance.
    #[test]
    fn scanner_matches_position_monotonic(
        query in arb_query(),
        document in arb_doc(),
        max_distance in 0u8..=2,
    ) {
        let matches = OnlinePhoneticScannerChar::new(&query, &[], max_distance).scan(&document);
        for w in matches.windows(2) {
            let a = (w[0].byte_range.0, w[0].distance);
            let b = (w[1].byte_range.0, w[1].distance);
            prop_assert!(a <= b, "matches not ordered by (start_byte, distance): {:?} then {:?}", a, b);
        }
    }

    /// NoMissedMatches for an exact occurrence surrounded by symbols that do
    /// not occur in the query alphabet.
    #[test]
    fn scanner_finds_exact_embedded_occurrence(
        query in arb_query(),
        prefix in arb_nonmatching_context(),
        suffix in arb_nonmatching_context(),
    ) {
        let expected_start = prefix.len();
        let expected_end = expected_start + query.len();
        let document = format!("{prefix}{query}{suffix}");
        let matches = OnlinePhoneticScannerChar::new(&query, &[], 0).scan(&document);

        let found = matches.iter().any(|m| {
            m.byte_range == (expected_start, expected_end)
                && m.original_text == query
                && m.distance == 0
        });
        prop_assert!(found, "exact occurrence was not returned: {matches:?}");
    }
}
