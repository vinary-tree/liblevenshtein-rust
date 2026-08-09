//! Shared integration-test support.
//!
//! The interop provider modules require the `bindings-core` feature (they
//! implement the `vinary_tree_interop` vtables); the conformance-fixture
//! parser is feature-free so ungated native tests can replay the same
//! canonical script.

#[cfg(feature = "bindings-core")]
pub mod fault_dictionary;
#[cfg(feature = "bindings-core")]
pub mod high_degree_dictionary;
#[cfg(feature = "bindings-core")]
pub mod interop_dictionary;

/// Parser for the canonical cross-language query-start snapshot fixture at
/// `vinary-tree-interop/conformance/query_start_snapshot.tsv`.
///
/// The fixture is the shared oracle for query-start snapshot semantics: an
/// `initial` phase populates the dictionary, a cursor is started, and a
/// `mutation` phase runs while that cursor is still live. Every language
/// binding replays the same script and must observe the same frozen result
/// set. The parser is deliberately strict — malformed rows panic with the
/// offending line — so fixture drift breaks loudly in every replaying suite.
#[allow(dead_code)]
pub mod query_start_fixture {
    /// Repository-relative fixture location (also cited by binding suites in
    /// other languages).
    pub const FIXTURE_PATH: &str = "vinary-tree-interop/conformance/query_start_snapshot.tsv";

    /// Which half of the replay a step belongs to.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum FixturePhase {
        /// Applied before the long-lived cursor starts.
        Initial,
        /// Applied while the long-lived cursor is mid-iteration.
        Mutation,
    }

    /// One fixture row.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct FixtureStep {
        /// Replay phase.
        pub phase: FixturePhase,
        /// Provider operation (`insert`, `update`, `remove`, `compact`,
        /// `checkpoint`).
        pub operation: String,
        /// Term operand; empty for structural operations.
        pub term: String,
        /// Optional value operand.
        pub id: Option<u64>,
    }

    /// Load and parse the canonical fixture.
    pub fn load() -> Vec<FixtureStep> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("fixture {} must be readable: {error}", path.display()));
        parse(&text)
    }

    fn parse(text: &str) -> Vec<FixtureStep> {
        let mut lines = text.lines();
        let header = lines.next().expect("fixture has a header line");
        assert_eq!(
            header, "phase\toperation\tterm\tid",
            "fixture header drifted from the published schema"
        );
        let mut steps = Vec::with_capacity(text.lines().count().saturating_sub(1));
        for (index, line) in lines.enumerate() {
            if line.is_empty() {
                continue;
            }
            let cells: Vec<&str> = line.split('\t').collect();
            assert_eq!(
                cells.len(),
                4,
                "fixture row {} must have 4 tab-separated cells: {line:?}",
                index + 2
            );
            let phase = match cells[0] {
                "initial" => FixturePhase::Initial,
                "mutation" => FixturePhase::Mutation,
                other => panic!("fixture row {} has unknown phase {other:?}", index + 2),
            };
            let id = match cells[3] {
                "" => None,
                digits => Some(digits.parse::<u64>().unwrap_or_else(|error| {
                    panic!("fixture row {} id {digits:?}: {error}", index + 2)
                })),
            };
            steps.push(FixtureStep {
                phase,
                operation: cells[1].to_owned(),
                term: cells[2].to_owned(),
                id,
            });
        }
        steps
    }
}
