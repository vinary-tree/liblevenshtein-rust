//! Property-based tests for the online phonetic scanner (`src/phonetic/online_scanner.rs`).
//!
//! Mirrors the TLA+ `OnlineScanner` model (`docs/verification/tla/`), which
//! models bounded active matches, position monotonicity, and (aspirationally)
//! `NoMissedMatches`, but abstracts the matching.
//!
//! We drive the real scanner with an EMPTY rewrite-rule set, so normalization is
//! the identity and the scanner performs plain fuzzy substring matching.
//!
//! NOTE / discovered inconsistency (left for a follow-up design decision rather
//! than an ad-hoc fix): a completed `ScanMatch` does not satisfy the natural
//! invariant `original_text == document[byte_range]`, and its `distance` is the
//! minimum over accepting *prefixes* of the consumed window rather than the edit
//! distance of `normalized_text` as a whole. Concretely, `PotentialMatch`
//! accumulates `original_chars`/`normalized_chars` over the full consumed window
//! (`online_scanner.rs:64-67, 91-97`) while `end_byte` / `min_distance` track the
//! best accepting position, so for query "c" over document "ca" a reported match
//! can have `byte_range` covering "c" but `original_text` "ca". The properties
//! below therefore assert only the invariants that genuinely hold; tightening
//! the `ScanMatch` field contract is a separate change.
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Boundedness: every completed match is within the configured error bound,
    /// and its byte range is a well-formed span of the document.
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
    /// then by distance (the scanner sorts by `(byte_range.0, distance)` in
    /// `finalize_matches`, `online_scanner.rs:459`).
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
}

// NOTE: a "no missed matches" property over arbitrary proper substrings is NOT
// asserted here. The scanner's substring-matching contract is not pinned down by
// its in-source tests (which all match the query against the whole normalized
// document), and an exploratory check found that scanning "ba" for query "b" at
// max_distance 0 reports no match. Whether proper-substring occurrences must be
// reported is a semantic question for the scanner's owner; it is recorded here
// as an open question rather than asserted as a (possibly incorrect) contract.
