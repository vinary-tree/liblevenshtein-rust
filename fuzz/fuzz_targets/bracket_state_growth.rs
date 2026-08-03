#![no_main]

use libfuzzer_sys::fuzz_target;
use liblevenshtein::transducer::language::{balanced_depth_dfa, BRACKET_DFA_MAX_STATES};

fuzz_target!(|data: &[u8]| {
    let kinds = data.first().copied().map_or(0, usize::from);
    let depth = data.get(1).copied().map_or(0, usize::from);

    if let Ok(dfa) = balanced_depth_dfa(kinds, depth) {
        assert!(dfa.state_count() <= BRACKET_DFA_MAX_STATES);
    }
});
