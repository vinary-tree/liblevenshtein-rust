//! Public-Rust-API oracle for the Julia standalone-automata qualification.
//!
//! Output is a deterministic tab-separated stream of complete cases. Each
//! observation field contains the initial state followed by every target
//! prefix, so the Julia test compares transitions, not only final acceptance.

use libdictenstein::CharUnit;
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::universal::{
    MergeAndSplit, PositionVariant, Standard, Transposition, UniversalAutomaton,
};
use liblevenshtein::transducer::{
    OperationApplicability, OperationSet, OperationType, OwnedRestricted, OwnedRestrictedChar,
    SubstitutionPolicy, SubstitutionPolicyFor, SubstitutionSet, SubstitutionSetChar, Unrestricted,
};
use std::hint::black_box;
use std::time::Instant;

fn operation(id: u8) -> OperationType {
    let (source, target, weight, applicability) = match id {
        0 => (0, 1, 1.0, OperationApplicability::Any),
        1 => (1, 0, 1.0, OperationApplicability::Any),
        2 => (1, 1, 0.0, OperationApplicability::Equal),
        3 => (1, 1, 0.5, OperationApplicability::Any),
        4 => (2, 2, 0.75, OperationApplicability::AdjacentTranspose),
        5 => {
            let mut pairs = SubstitutionSet::new();
            pairs.allow_str("éa", "δ");
            pairs.allow_str("ph", "f");
            (2, 1, 0.25, OperationApplicability::Listed(pairs))
        }
        6 => (2, 2, 0.0, OperationApplicability::Equal),
        7 => (2, 1, 1.25, OperationApplicability::Any),
        8 => (1, 2, 1.25, OperationApplicability::Any),
        9 => {
            let mut pairs = SubstitutionSet::new();
            pairs.allow_str("a", "é");
            (1, 1, 0.0, OperationApplicability::Listed(pairs))
        }
        _ => unreachable!(),
    };
    OperationType::with_owned_applicability(
        source,
        target,
        weight,
        applicability,
        format!("qualification_{id}"),
    )
}

fn encode_units(values: impl IntoIterator<Item = u64>) -> String {
    values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn emit_generalized(ids: &[u8], budget: u8, source: &str, target: &str) {
    let mut operations = OperationSet::new();
    for &id in ids {
        operations.add(operation(id));
    }
    let automaton = GeneralizedAutomaton::try_with_operations(budget, operations).unwrap();
    let denominator = automaton.cost_scale().unwrap().denominator();
    let mut online = automaton.online(source).unwrap();
    let mut observations = Vec::new();
    let mut append =
        |value: liblevenshtein::transducer::generalized::GeneralizedOnlineObservation| {
            observations.push(format!(
                "{},{},{},{}",
                value.consumed_target_len,
                value.active_positions,
                value
                    .distance_within_budget
                    .map_or("x".to_owned(), |n| n.to_string()),
                u8::from(value.distance_within_budget.is_some()),
            ));
        };
    append(online.observation());
    for unit in target.chars() {
        append(online.advance(unit).unwrap());
    }
    println!(
        "G\t{}\t{budget}\t{}\t{}\t{denominator}\t{}",
        encode_units(ids.iter().map(|&id| u64::from(id))),
        encode_units(source.chars().map(|unit| u64::from(u32::from(unit)))),
        encode_units(target.chars().map(|unit| u64::from(u32::from(unit)))),
        observations.join(";"),
    );
}

trait WireUnit: CharUnit + Copy {
    fn wire(self) -> u64;
}
impl WireUnit for char {
    fn wire(self) -> u64 {
        u64::from(u32::from(self))
    }
}
impl WireUnit for u8 {
    fn wire(self) -> u64 {
        u64::from(self)
    }
}
impl WireUnit for u64 {
    fn wire(self) -> u64 {
        self
    }
}

#[derive(Clone, Debug)]
struct RestrictedU64;
impl SubstitutionPolicy for RestrictedU64 {
    fn is_allowed(&self, source: u8, target: u8) -> bool {
        source == target
    }
}
impl SubstitutionPolicyFor<u64> for RestrictedU64 {
    fn is_allowed_for(&self, source: u64, target: u64) -> bool {
        source == target || (source == u64::MAX && target == 0)
    }
}

fn emit_universal<V, P, U>(
    variant: u8,
    policy_id: u8,
    domain: char,
    budget: u8,
    policy: P,
    source: &[U],
    target: &[U],
) where
    V: PositionVariant,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    U: WireUnit,
{
    let automaton = UniversalAutomaton::<V, P>::with_policy(budget, policy);
    let mut online = automaton.online_units(source);
    let mut observations = vec![format!(
        "0,{},{},{}",
        online.word_length(),
        u8::from(online.state().is_some()),
        u8::from(online.is_accepting()),
    )];
    for &unit in target {
        online.advance(unit);
        observations.push(format!(
            "{},{},{},{}",
            online.input_length(),
            online.word_length(),
            u8::from(online.state().is_some()),
            u8::from(online.is_accepting()),
        ));
    }
    println!(
        "U\t{variant}\t{policy_id}\t{domain}\t{budget}\t{}\t{}\t{}",
        encode_units(source.iter().map(|&unit| unit.wire())),
        encode_units(target.iter().map(|&unit| unit.wire())),
        observations.join(";"),
    );
}

fn run_variant<V: PositionVariant>(variant: u8) {
    let text: &[&[char]] = &[
        &[],
        &['a'],
        &['é'],
        &['δ'],
        &['a', 'é'],
        &['é', 'a'],
        &['a', 'δ', 'é'],
        &['a', 'a', 'a', 'a'],
    ];
    let bytes: &[&[u8]] = &[
        &[],
        &[0],
        &[0xff],
        &[b'p'],
        &[b'f'],
        &[0, 0xff],
        &[b'p', b'f'],
        &[0xff, 0, b'p'],
    ];
    let tokens: &[&[u64]] = &[
        &[],
        &[0],
        &[u64::MAX],
        &[7],
        &[9],
        &[u64::MAX, 0],
        &[0, u64::MAX],
        &[7, 9, u64::MAX],
    ];
    let mut char_set = SubstitutionSetChar::new();
    char_set.allow('a', 'é');
    let mut byte_set = SubstitutionSet::new();
    byte_set.allow_byte(b'p', b'f');
    for budget in 0..=2 {
        for &source in text {
            for &target in text {
                emit_universal::<V, _, _>(variant, 0, 'T', budget, Unrestricted, source, target);
                emit_universal::<V, _, _>(
                    variant,
                    1,
                    'T',
                    budget,
                    OwnedRestrictedChar::new(char_set.clone()),
                    source,
                    target,
                );
            }
        }
        for &source in bytes {
            for &target in bytes {
                emit_universal::<V, _, _>(variant, 0, 'B', budget, Unrestricted, source, target);
                emit_universal::<V, _, _>(
                    variant,
                    1,
                    'B',
                    budget,
                    OwnedRestricted::new(byte_set.clone()),
                    source,
                    target,
                );
            }
        }
        for &source in tokens {
            for &target in tokens {
                emit_universal::<V, _, _>(variant, 0, 'U', budget, Unrestricted, source, target);
                emit_universal::<V, _, _>(variant, 1, 'U', budget, RestrictedU64, source, target);
            }
        }
    }
}

fn measure(mut operation: impl FnMut()) -> u128 {
    const ITERATIONS: u128 = 200;
    for _ in 0..100 {
        operation();
    }
    let mut samples = Vec::new();
    for _ in 0..7 {
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            operation();
        }
        samples.push(started.elapsed().as_nanos() / ITERATIONS);
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn benchmark() {
    let mut operations = OperationSet::new();
    for id in [0, 1, 2, 3] {
        operations.add(operation(id));
    }
    let generalized = GeneralizedAutomaton::try_with_operations(2, operations.clone()).unwrap();
    let universal = UniversalAutomaton::<Standard>::new(2);
    let source = "a".repeat(32);
    let complete = source.clone();
    let early = "z".repeat(32);
    let late = format!("{}zzzzz", "a".repeat(27));
    let targets: Vec<String> = (0..32)
        .map(|index| {
            if index % 2 == 0 {
                complete.clone()
            } else {
                early.clone()
            }
        })
        .collect();
    let emit = |name: &str, operation: &mut dyn FnMut()| {
        println!("B\t{name}\t{}", measure(operation));
    };
    emit("generalized_construction", &mut || {
        black_box(GeneralizedAutomaton::try_with_operations(2, operations.clone()).unwrap());
    });
    emit("universal_construction", &mut || {
        black_box(UniversalAutomaton::<Standard>::new(2));
    });
    let general_match = |target: &str| {
        let mut state = generalized.online(&source).unwrap();
        for unit in target.chars() {
            state.advance(unit).unwrap();
        }
        black_box(state.observation());
    };
    let universal_match = |target: &str| {
        let mut state = universal.online(&source);
        for unit in target.chars() {
            state.advance(unit);
        }
        black_box(state.is_accepting());
    };
    emit("generalized_complete", &mut || general_match(&complete));
    emit("universal_complete", &mut || universal_match(&complete));
    emit("generalized_early_reject", &mut || general_match(&early));
    emit("universal_early_reject", &mut || universal_match(&early));
    emit("generalized_late_reject", &mut || general_match(&late));
    emit("universal_late_reject", &mut || universal_match(&late));
    emit("generalized_traversal_32", &mut || general_match(&complete));
    emit("universal_traversal_32", &mut || universal_match(&complete));
    emit("generalized_batch_32", &mut || {
        for target in &targets {
            general_match(target);
        }
    });
    emit("universal_batch_32", &mut || {
        for target in &targets {
            universal_match(target);
        }
    });
}

fn main() {
    if std::env::args().any(|argument| argument == "--bench") {
        benchmark();
        return;
    }
    let corpus = ["", "a", "é", "δ", "aa", "aé", "éa", "δa", "ph", "f"];
    let fixed_sets: &[&[u8]] = &[
        &[],
        &[2],
        &[0, 1, 2, 3],
        &[0, 1, 2, 3, 4],
        &[5],
        &[6],
        &[9],
        &[2, 5, 6, 9],
        &[0, 1, 2, 7, 8],
    ];
    for (index, ids) in fixed_sets.iter().enumerate() {
        let budget = (index % 3) as u8;
        for &source in &corpus {
            for &target in &corpus {
                emit_generalized(ids, budget, source, target);
            }
        }
    }
    // Fixed-seed randomized grammars and operands exercise combinations not
    // selected by the hand-written corpus while remaining byte-reproducible.
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let alphabet = ['a', 'é', 'δ', 'p', 'h', 'f'];
    for _ in 0..192 {
        let mask = next();
        let ids: Vec<u8> = (0..10).filter(|id| mask & (1 << id) != 0).collect();
        let budget = (next() % 3) as u8;
        let source: String = (0..next() % 5)
            .map(|_| alphabet[(next() as usize) % alphabet.len()])
            .collect();
        let target: String = (0..next() % 5)
            .map(|_| alphabet[(next() as usize) % alphabet.len()])
            .collect();
        emit_generalized(&ids, budget, &source, &target);
    }
    run_variant::<Standard>(0);
    run_variant::<Transposition>(1);
    run_variant::<MergeAndSplit>(2);
}
