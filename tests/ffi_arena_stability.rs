//! Arena-stability correspondence for the leased-batch fill path.
//!
//! Spec: docs/verification/verus/ffi_batch_arena.rs (Verus; committed),
//! which proves the two-pass `fill_batch` offset arithmetic of
//! `src/ffi/index.rs`: descriptors record OFFSETS while the arenas grow, and
//! pointers are minted only after the arenas stop moving, so no view ever
//! dangles into a pre-reallocation allocation. This file is its executable
//! face against the REAL C ABI, with corpora engineered to force MANY arena
//! reallocations inside a single fill (term lengths up to 2048 units, term
//! counts in the hundreds) across all three unit domains, driven through
//! real providers: the coordinator's `HighDegreeDictionary::long_terms`
//! trie and genuine libdictenstein `DynamicDawgBinding` resources.
//!
//! INVARIANT-HOOK: LLEV-ARENA-1 — windows are contiguous prefix sums: every
//! view in a batch starts exactly where the previous one ended
//! (`copy_and_check_batch` asserts the tiling on every batch of every
//! test).
//! INVARIANT-HOOK: LLEV-ARENA-2 — every window lies within the final arena:
//! after multi-reallocation fills, every `term_data` dereferences its full
//! `byte_len` to the exact expected bytes and u64 views keep 8-byte
//! alignment (`long_terms_survive_multi_realloc_fills_across_batch_sizes`,
//! `u64_views_are_eight_byte_aligned_with_byte_len_eight_times_term_len`).
//! INVARIANT-HOOK: LLEV-ARENA-3 — slicing at a term's window recovers that
//! exact term, and distinct windows recover distinct terms (every content
//! comparison below, most sharply the 300 x 2048 corpus and the
//! per-domain byte_len laws).
//!
//! Pinned byte_len law per domain (`LlevMatch` contract):
//! - Byte: `byte_len == term_len`;
//! - UnicodeScalar: `term_len` is the scalar count, `byte_len` the UTF-8
//!   length (strictly larger for multibyte corpora);
//! - U64: `byte_len == 8 * term_len` with `term_data` 8-byte aligned.

#![cfg(feature = "binding-integration-tests")]

mod support;

use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};
use liblevenshtein::ffi::*;
use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr;
use support::fault_dictionary::FaultDictionary;
use support::high_degree_dictionary::HighDegreeDictionary;
use vinary_tree_interop::{VtResource, VtUnitDomain};

// ---------------------------------------------------------------------------
// View recovery with the arena laws asserted inline.
// ---------------------------------------------------------------------------

/// One match copied OUT of a leased view, in unit-domain-neutral form.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Recovered {
    units: Vec<u64>,
    distance: usize,
    id: Option<u64>,
}

/// Copy every descriptor of one leased batch, asserting the per-domain
/// byte_len law (LLEV-ARENA-3's recovery face), u64 alignment
/// (LLEV-ARENA-2), and contiguous tiling (LLEV-ARENA-1).
unsafe fn copy_and_check_batch(
    view: &LlevMatchBatchView,
    expected_domain: VtUnitDomain,
) -> Vec<Recovered> {
    let items = std::slice::from_raw_parts(view.matches, view.len);
    let mut recovered = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        assert_eq!(item.unit_domain, expected_domain, "view {index} domain");
        assert_eq!(item.reserved, [0u8; 3], "view {index} reserved bytes");
        assert!(!item.term_data.is_null(), "view {index} term_data");
        assert!(matches!(item.has_id, 0 | 1), "view {index} has_id");
        let units: Vec<u64> = match expected_domain {
            VtUnitDomain::Byte => {
                assert_eq!(item.byte_len, item.term_len, "byte domain byte_len law");
                std::slice::from_raw_parts(item.term_data.cast::<u8>(), item.byte_len)
                    .iter()
                    .map(|unit| u64::from(*unit))
                    .collect()
            }
            VtUnitDomain::UnicodeScalar => {
                let bytes = std::slice::from_raw_parts(item.term_data.cast::<u8>(), item.byte_len);
                let text = std::str::from_utf8(bytes).expect("scalar terms are valid UTF-8");
                assert_eq!(text.chars().count(), item.term_len, "scalar count law");
                assert_eq!(text.len(), item.byte_len, "UTF-8 byte_len law");
                text.chars()
                    .map(|unit| u64::from(u32::from(unit)))
                    .collect()
            }
            VtUnitDomain::U64 => {
                assert_eq!(
                    item.byte_len,
                    item.term_len * std::mem::size_of::<u64>(),
                    "u64 domain byte_len law"
                );
                assert_eq!(
                    item.term_data as usize % std::mem::align_of::<u64>(),
                    0,
                    "u64 views must be 8-byte aligned (LLEV-ARENA-2)"
                );
                std::slice::from_raw_parts(item.term_data.cast::<u64>(), item.term_len).to_vec()
            }
        };
        if index + 1 < items.len() {
            let next = &items[index + 1];
            assert_eq!(
                next.term_data as usize,
                item.term_data as usize + item.byte_len,
                "views must tile the arena contiguously (LLEV-ARENA-1)"
            );
        }
        recovered.push(Recovered {
            units,
            distance: item.distance,
            id: (item.has_id == 1).then_some(item.id),
        });
    }
    recovered
}

unsafe fn drain_with_batch(
    cursor: *mut LlevQueryCursor,
    batch: usize,
    domain: VtUnitDomain,
) -> Vec<Recovered> {
    let mut all = Vec::new();
    loop {
        let mut view = LlevMatchBatchView::default();
        match llev_query_cursor_next_batch(cursor, batch, &mut view) {
            LlevStatus::Ok => {
                assert!(view.len >= 1 && view.len <= batch, "batch length bound");
                all.extend(copy_and_check_batch(&view, domain));
                assert_eq!(
                    llev_query_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => return all,
            status => panic!("arena drain hit {status:?}"),
        }
    }
}

unsafe fn transducer_over(resource: &VtResource) -> *mut LlevTransducer {
    let mut transducer = ptr::null_mut();
    assert_eq!(
        llev_transducer_new(resource, LlevAlgorithm::Standard as u32, &mut transducer),
        LlevStatus::Ok
    );
    transducer
}

fn utf8_units(term: &str) -> Vec<u64> {
    term.chars()
        .map(|unit| u64::from(u32::from(unit)))
        .collect()
}

// ---------------------------------------------------------------------------
// The realloc-forcing corpus: 300 terms x 2048 scalars in one fill.
// ---------------------------------------------------------------------------

/// Banded (Ukkonen) Levenshtein oracle: `Some(d)` exactly when the edit
/// distance is `d <= bound`; `None` when it exceeds the bound. O(len x
/// bound), so it stays an honest in-test reference even for 2048-unit
/// terms.
fn bounded_levenshtein(a: &[u64], b: &[u64], bound: usize) -> Option<usize> {
    let infinity = usize::MAX / 2;
    if a.len().abs_diff(b.len()) > bound {
        return None;
    }
    let width = b.len() + 1;
    let mut previous: Vec<usize> = vec![infinity; width];
    let mut current: Vec<usize> = vec![infinity; width];
    for (column, cell) in previous.iter_mut().enumerate().take(bound + 1) {
        *cell = column;
    }
    for row in 1..=a.len() {
        current.fill(infinity);
        let low = row.saturating_sub(bound);
        let high = (row + bound).min(b.len());
        if low == 0 {
            current[0] = row;
        }
        for column in low.max(1)..=high {
            let substitution_cost = usize::from(a[row - 1] != b[column - 1]);
            let mut best = previous[column - 1].saturating_add(substitution_cost);
            best = best.min(previous[column].saturating_add(1));
            best = best.min(current[column - 1].saturating_add(1));
            current[column] = best;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let distance = previous[b.len()];
    (distance <= bound).then_some(distance)
}

/// LLEV-ARENA-2/3 under maximal pressure: one fill of up to 300 x 2048
/// bytes grows the byte arena through many reallocations, and every view
/// still dereferences to its exact term afterward — at every batch size,
/// with the multiset invariant across batch sizes. The query anchors on the
/// first stored term at distance 3, which covers the whole corpus (every
/// suffix differs from `0000` by at most three digits) while keeping the
/// automaton narrow; the banded DP above supplies the exact per-term
/// distances.
#[test]
fn long_terms_survive_multi_realloc_fills_across_batch_sizes() {
    let (dictionary, terms) = HighDegreeDictionary::long_terms(300, 2048);
    let query = terms.first().expect("corpus is nonempty").clone();
    let query_units = utf8_units(&query);
    let expected: BTreeSet<Recovered> = terms
        .iter()
        .enumerate()
        .map(|(rank, term)| {
            let units = utf8_units(term);
            let distance = bounded_levenshtein(&units, &query_units, 3)
                .expect("every 0000..0299 suffix is within three edits of 0000");
            Recovered {
                units,
                distance,
                id: Some(rank as u64),
            }
        })
        .collect();
    assert_eq!(expected.len(), 300, "corpus terms are distinct");
    assert!(
        expected.iter().any(|item| item.distance == 0)
            && expected.iter().any(|item| item.distance == 3),
        "the corpus spans heterogeneous distances"
    );

    unsafe {
        let resource = dictionary.resource();
        let transducer = transducer_over(&resource);
        for batch in [1usize, 255, 256, 257, 4096] {
            let mut cursor = ptr::null_mut();
            assert_eq!(
                llev_transducer_query_utf8(
                    transducer,
                    query.as_ptr().cast(),
                    query.len(),
                    3,
                    LlevQueryOrder::Traversal as u32,
                    &mut cursor,
                ),
                LlevStatus::Ok
            );
            let recovered = drain_with_batch(cursor, batch, VtUnitDomain::UnicodeScalar);
            assert_eq!(recovered.len(), 300, "batch {batch} total");
            let observed: BTreeSet<Recovered> = recovered.into_iter().collect();
            assert_eq!(
                observed, expected,
                "batch {batch}: every term must be recovered byte-exactly"
            );
            assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        }
        llev_transducer_free(transducer);
    }
}

// ---------------------------------------------------------------------------
// Domain-specific corpora over REAL libdictenstein resources.
// ---------------------------------------------------------------------------

/// Forty distinct non-'α' scalars spanning all four UTF-8 widths.
fn multibyte_variants() -> Vec<char> {
    let mut variants = Vec::with_capacity(40);
    for index in 0u32..10 {
        variants.push(char::from_u32(u32::from('a') + index).expect("ASCII"));
    }
    for index in 0u32..10 {
        variants.push(char::from_u32(0x00E0 + index).expect("two-byte scalar"));
    }
    for index in 0u32..10 {
        variants.push(char::from_u32(0x4E00 + index).expect("three-byte scalar"));
    }
    for index in 0u32..10 {
        variants.push(char::from_u32(0x10330 + index).expect("four-byte scalar"));
    }
    variants
}

/// UnicodeScalar over a real libdictenstein DAWG: `term_len` counts scalars
/// while `byte_len` counts UTF-8 bytes, so multibyte corpora pin the two
/// apart (pure-'α' terms have `byte_len == 2 * term_len` exactly).
#[test]
fn multibyte_utf8_terms_pin_the_byte_len_law() {
    for length in [1usize, 255, 257] {
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
        let chain: String = "α".repeat(length);
        let stem: String = "α".repeat(length - 1);
        let mut expected: BTreeSet<Recovered> = BTreeSet::new();
        assert!(dictionary
            .insert_text(chain.as_bytes(), Some(7000))
            .expect("insert chain"));
        expected.insert(Recovered {
            units: utf8_units(&chain),
            distance: 0,
            id: Some(7000),
        });
        if length > 1 {
            assert!(dictionary
                .insert_text(stem.as_bytes(), Some(7001))
                .expect("insert stem"));
            expected.insert(Recovered {
                units: utf8_units(&stem),
                distance: 1,
                id: Some(7001),
            });
        }
        for (index, variant) in multibyte_variants().into_iter().enumerate() {
            let mut term = stem.clone();
            term.push(variant);
            assert!(dictionary
                .insert_text(term.as_bytes(), Some(index as u64))
                .expect("insert variant"));
            expected.insert(Recovered {
                units: utf8_units(&term),
                distance: 1,
                id: Some(index as u64),
            });
        }

        unsafe {
            let resource = dictionary.resource();
            let raw = resource.as_raw();
            let transducer = transducer_over(&raw);
            for batch in [1usize, 257] {
                let mut cursor = ptr::null_mut();
                assert_eq!(
                    llev_transducer_query_utf8(
                        transducer,
                        chain.as_ptr().cast(),
                        chain.len(),
                        1,
                        LlevQueryOrder::Traversal as u32,
                        &mut cursor,
                    ),
                    LlevStatus::Ok
                );
                let observed: BTreeSet<Recovered> =
                    drain_with_batch(cursor, batch, VtUnitDomain::UnicodeScalar)
                        .into_iter()
                        .collect();
                assert_eq!(observed, expected, "length {length} batch {batch}");
                assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            }
            llev_transducer_free(transducer);
        }
    }
}

/// Byte domain over a real libdictenstein DAWG: `byte_len == term_len`
/// exactly, raw high bytes included, across arena-pressuring lengths.
#[test]
fn byte_domain_views_pin_byte_len_equals_term_len() {
    for length in [1usize, 300, 2048] {
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::Byte);
        let chain: Vec<u8> = vec![0xE9; length];
        let stem: Vec<u8> = vec![0xE9; length - 1];
        let mut expected: BTreeSet<Recovered> = BTreeSet::new();
        assert!(dictionary
            .insert_text(&chain, Some(9000))
            .expect("insert chain"));
        expected.insert(Recovered {
            units: chain.iter().map(|unit| u64::from(*unit)).collect(),
            distance: 0,
            id: Some(9000),
        });
        for index in 0u8..48 {
            let mut term = stem.clone();
            term.push(0x80 + index);
            assert!(dictionary
                .insert_text(&term, Some(u64::from(index)))
                .expect("insert variant"));
            expected.insert(Recovered {
                units: term.iter().map(|unit| u64::from(*unit)).collect(),
                distance: 1,
                id: Some(u64::from(index)),
            });
        }

        unsafe {
            let resource = dictionary.resource();
            let raw = resource.as_raw();
            let transducer = transducer_over(&raw);
            for batch in [1usize, 256, 4096] {
                let mut cursor = ptr::null_mut();
                assert_eq!(
                    llev_transducer_query_bytes(
                        transducer,
                        chain.as_ptr(),
                        chain.len(),
                        1,
                        LlevQueryOrder::Traversal as u32,
                        &mut cursor,
                    ),
                    LlevStatus::Ok
                );
                let observed: BTreeSet<Recovered> =
                    drain_with_batch(cursor, batch, VtUnitDomain::Byte)
                        .into_iter()
                        .collect();
                assert_eq!(observed, expected, "length {length} batch {batch}");
                assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            }
            llev_transducer_free(transducer);
        }
    }
}

/// U64 domain over a real libdictenstein DAWG: every view 8-byte aligned,
/// `byte_len == 8 * term_len`, extreme token values recovered exactly.
#[test]
fn u64_views_are_eight_byte_aligned_with_byte_len_eight_times_term_len() {
    for length in [1usize, 256, 2048] {
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::U64);
        let chain: Vec<u64> = vec![7; length];
        let stem: Vec<u64> = vec![7; length - 1];
        let mut expected: BTreeSet<Recovered> = BTreeSet::new();
        assert!(dictionary
            .insert_u64(&chain, Some(4000))
            .expect("insert chain"));
        expected.insert(Recovered {
            units: chain.clone(),
            distance: 0,
            id: Some(4000),
        });
        for index in 0u64..32 {
            let mut term = stem.clone();
            // Extreme labels: half near u64::MAX, half small-but-distinct.
            let token = if index % 2 == 0 {
                u64::MAX - index
            } else {
                1000 + index
            };
            term.push(token);
            assert!(dictionary
                .insert_u64(&term, Some(index))
                .expect("insert variant"));
            expected.insert(Recovered {
                units: term,
                distance: 1,
                id: Some(index),
            });
        }

        unsafe {
            let resource = dictionary.resource();
            let raw = resource.as_raw();
            let transducer = transducer_over(&raw);
            for batch in [1usize, 4096] {
                let mut cursor = ptr::null_mut();
                assert_eq!(
                    llev_transducer_query_u64(
                        transducer,
                        chain.as_ptr(),
                        chain.len(),
                        1,
                        LlevQueryOrder::Traversal as u32,
                        &mut cursor,
                    ),
                    LlevStatus::Ok
                );
                let observed: BTreeSet<Recovered> =
                    drain_with_batch(cursor, batch, VtUnitDomain::U64)
                        .into_iter()
                        .collect();
                assert_eq!(observed, expected, "length {length} batch {batch}");
                assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            }
            llev_transducer_free(transducer);
        }
    }
}

/// The degenerate windows: a zero-length term yields a valid non-null view
/// of zero bytes that still tiles with its successors, and `has_id == 0`
/// flows through untouched.
#[test]
fn empty_and_single_unit_terms_have_valid_views() {
    let dictionary = FaultDictionary::new([(String::new(), Some(9)), ("x".to_owned(), None)]);
    unsafe {
        let resource = dictionary.resource();
        let transducer = transducer_over(&resource);

        // Distance 0: exactly the empty term.
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_utf8(
                transducer,
                b"".as_ptr().cast(),
                0,
                0,
                LlevQueryOrder::Traversal as u32,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        let only = drain_with_batch(cursor, 8, VtUnitDomain::UnicodeScalar);
        assert_eq!(
            only,
            vec![Recovered {
                units: vec![],
                distance: 0,
                id: Some(9),
            }]
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        // Distance 1: the empty term tiles with a one-unit term in one batch.
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_utf8(
                transducer,
                b"".as_ptr().cast(),
                0,
                1,
                LlevQueryOrder::Traversal as u32,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        let observed: BTreeSet<Recovered> =
            drain_with_batch(cursor, 8, VtUnitDomain::UnicodeScalar)
                .into_iter()
                .collect();
        let expected: BTreeSet<Recovered> = [
            Recovered {
                units: vec![],
                distance: 0,
                id: Some(9),
            },
            Recovered {
                units: utf8_units("x"),
                distance: 1,
                id: None,
            },
        ]
        .into_iter()
        .collect();
        assert_eq!(observed, expected);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        llev_transducer_free(transducer);
    }
}

// ---------------------------------------------------------------------------
// Property: arbitrary ladder corpora round-trip exactly in every domain.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// For an arbitrary unit domain, term length (1..=2048), fanout, and
    /// batch size: a distance-1 ladder corpus built on a REAL libdictenstein
    /// resource is recovered from the C ABI byte-exactly, with the tiling,
    /// alignment, and byte_len laws holding on every batch.
    #[test]
    fn arbitrary_ladders_round_trip_exactly(
        domain_index in 0usize..3,
        length in prop_oneof![
            Just(1usize), Just(2), Just(3), Just(255), Just(256), Just(257),
            Just(300), Just(1024), Just(2048),
        ],
        fanout in 1usize..48,
        batch in prop_oneof![
            Just(1usize), Just(255), Just(256), Just(257), Just(4096),
        ],
        include_exact in any::<bool>(),
    ) {
        let domain = [
            BindingUnitDomain::Byte,
            BindingUnitDomain::UnicodeScalar,
            BindingUnitDomain::U64,
        ][domain_index];
        let vt_domain = [
            VtUnitDomain::Byte,
            VtUnitDomain::UnicodeScalar,
            VtUnitDomain::U64,
        ][domain_index];
        let dictionary = DynamicDawgBinding::new(domain);

        // Chain unit and per-domain distinct variant units.
        let chain_unit: u64 = match vt_domain {
            VtUnitDomain::Byte => 0xE9,
            VtUnitDomain::UnicodeScalar => u64::from(u32::from('α')),
            VtUnitDomain::U64 => 7,
        };
        let variant_unit = |index: usize| -> u64 {
            match vt_domain {
                VtUnitDomain::Byte => 0x80 + index as u64,
                VtUnitDomain::UnicodeScalar => u64::from(0x4E00 + index as u32),
                VtUnitDomain::U64 => 1000 + index as u64,
            }
        };

        let query: Vec<u64> = vec![chain_unit; length];
        let mut expected: BTreeSet<Recovered> = BTreeSet::new();
        {
            // Scoped so the closure's borrows of `dictionary` and `expected`
            // end before the drain below reads them.
            let mut insert = |units: &[u64], id: Option<u64>, distance: usize| {
                match vt_domain {
                    VtUnitDomain::Byte => {
                        let bytes: Vec<u8> = units
                            .iter()
                            .map(|unit| u8::try_from(*unit).expect("byte label"))
                            .collect();
                        assert!(dictionary.insert_text(&bytes, id).expect("insert bytes"));
                    }
                    VtUnitDomain::UnicodeScalar => {
                        let text: String = units
                            .iter()
                            .map(|unit| {
                                char::from_u32(u32::try_from(*unit).expect("scalar"))
                                    .expect("scalar label")
                            })
                            .collect();
                        assert!(dictionary
                            .insert_text(text.as_bytes(), id)
                            .expect("insert text"));
                    }
                    VtUnitDomain::U64 => {
                        assert!(dictionary.insert_u64(units, id).expect("insert tokens"));
                    }
                }
                expected.insert(Recovered {
                    units: units.to_vec(),
                    distance,
                    id,
                });
            };

            if include_exact {
                insert(&query, Some(u64::MAX), 0);
            }
            for index in 0..fanout {
                let mut term = vec![chain_unit; length - 1];
                term.push(variant_unit(index));
                let id = (index % 3 != 0).then_some(index as u64);
                insert(&term, id, 1);
            }
        }

        unsafe {
            let resource = dictionary.resource();
            let raw = resource.as_raw();
            let transducer = transducer_over(&raw);
            let mut cursor = ptr::null_mut();
            let status = match vt_domain {
                VtUnitDomain::Byte => {
                    let bytes: Vec<u8> = query
                        .iter()
                        .map(|unit| u8::try_from(*unit).expect("byte label"))
                        .collect();
                    llev_transducer_query_bytes(
                        transducer,
                        bytes.as_ptr(),
                        bytes.len(),
                        1,
                        LlevQueryOrder::Traversal as u32,
                        &mut cursor,
                    )
                }
                VtUnitDomain::UnicodeScalar => {
                    let text: String = query
                        .iter()
                        .map(|unit| {
                            char::from_u32(u32::try_from(*unit).expect("scalar"))
                                .expect("scalar label")
                        })
                        .collect();
                    llev_transducer_query_utf8(
                        transducer,
                        text.as_ptr().cast(),
                        text.len(),
                        1,
                        LlevQueryOrder::Traversal as u32,
                        &mut cursor,
                    )
                }
                VtUnitDomain::U64 => llev_transducer_query_u64(
                    transducer,
                    query.as_ptr(),
                    query.len(),
                    1,
                    LlevQueryOrder::Traversal as u32,
                    &mut cursor,
                ),
            };
            prop_assert_eq!(status, LlevStatus::Ok);
            let observed: BTreeSet<Recovered> =
                drain_with_batch(cursor, batch, vt_domain).into_iter().collect();
            let counts: BTreeMap<usize, usize> =
                observed.iter().fold(BTreeMap::new(), |mut map, item| {
                    *map.entry(item.distance).or_insert(0) += 1;
                    map
                });
            prop_assert_eq!(&observed, &expected, "distance histogram {:?}", counts);
            prop_assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            llev_transducer_free(transducer);
        }
    }
}
