//! Reproduce the preregistered true-Damerau frontier and successor profile.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example damerau_state_profile
//! ```

use liblevenshtein::transducer::transition::{
    initial_state, transition_position, transition_state,
};
use liblevenshtein::transducer::{Algorithm, Position, Unrestricted};

fn main() {
    println!("k\tmax_state\tmax_state_over_k_squared\tsuccessors\tspilled");

    for k in 1usize..=3 {
        let length = 2 * k + 4;
        let query = vec![b'a'; length];
        let mut state = initial_state(query.len(), k, Algorithm::DamerauLevenshtein);
        let mut maximum = state.len();

        for dictionary_unit in vec![b'a'; length] {
            state = transition_state(
                &state,
                Unrestricted,
                dictionary_unit,
                &query,
                k,
                Algorithm::DamerauLevenshtein,
                false,
            )
            .expect("the repeated-unit path must remain reachable");
            maximum = maximum.max(state.len());
        }

        let characteristic_vector = vec![true; k + 1];
        let successors = transition_position(
            &Position::new(0, 0),
            &characteristic_vector,
            query.len(),
            k,
            Algorithm::DamerauLevenshtein,
            false,
        );

        println!(
            "{k}\t{maximum}\t{:.6}\t{}\t{}",
            maximum as f64 / (k * k) as f64,
            successors.len(),
            successors.spilled(),
        );
    }
}
