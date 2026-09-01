//! Constrained-stack and stable-retention regressions for the online string
//! reference machines.

use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::universal::{
    MergeAndSplit, PositionVariant, Standard, Transposition, UniversalAutomaton,
};

const PREFIX_LEN: usize = 100_000;

fn consume_universal<V: PositionVariant>() {
    let automaton = UniversalAutomaton::<V>::new(2);
    let mut online = automaton.online("fixed word");
    let word_length = online.word_length();
    for _ in 0..PREFIX_LEN {
        online.advance('x');
        assert_eq!(online.word_length(), word_length);
    }
    assert_eq!(online.input_length(), PREFIX_LEN);
    assert!(online.state().is_none());
}

#[test]
fn every_universal_online_variant_is_iterative_and_prefix_stable() {
    std::thread::Builder::new()
        .name("universal-online-small-stack".into())
        .stack_size(128 * 1024)
        .spawn(|| {
            consume_universal::<Standard>();
            consume_universal::<Transposition>();
            consume_universal::<MergeAndSplit>();
        })
        .expect("small-stack test thread must start")
        .join()
        .expect("universal online transitions must not consume the process call stack");
}

#[test]
fn generalized_online_ring_is_iterative_and_prefix_stable() {
    std::thread::Builder::new()
        .name("generalized-online-small-stack".into())
        .stack_size(128 * 1024)
        .spawn(|| {
            let automaton = GeneralizedAutomaton::new(2);
            let mut online = automaton
                .online("fixed word")
                .expect("the bounded generalized machine must construct");
            let retained_cells = online.retained_cells();
            for _ in 0..PREFIX_LEN {
                online
                    .advance('x')
                    .expect("fixed per-transition work must remain in budget");
                assert_eq!(online.retained_cells(), retained_cells);
            }
            assert_eq!(online.observation().consumed_target_len, PREFIX_LEN);
        })
        .expect("small-stack test thread must start")
        .join()
        .expect("generalized online transitions must not consume the process call stack");
}
