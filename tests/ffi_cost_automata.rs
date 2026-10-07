#![cfg(feature = "binding-integration-tests")]

use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};
use libdictenstein::dynamic_dawg::DynamicDawgU64;
#[cfg(target_pointer_width = "64")]
use liblevenshtein::ffi::LlevCostMatch;
use liblevenshtein::ffi::{
    llev_cost_cursor_free, llev_cost_cursor_next_batch, llev_cost_cursor_release_batch,
    llev_transducer_free, llev_transducer_new, llev_transducer_query_affine,
    llev_transducer_query_weighted, LlevAffineCosts, LlevAlgorithm, LlevCostBatchView,
    LlevCostCursor, LlevOperationCostsF64, LlevStatus, LlevTransducer,
};
use liblevenshtein::transducer::{AffineGapParams, Algorithm, OperationCostsF64, Transducer};
use std::collections::BTreeMap;
use std::ffi::c_void;
#[cfg(target_pointer_width = "64")]
use std::mem::{offset_of, size_of};
use std::ptr;
use vinary_tree_interop::VtUnitDomain;

#[cfg(target_pointer_width = "64")]
#[test]
fn cost_abi_layout_is_stable_on_lp64() {
    assert_eq!(size_of::<LlevAffineCosts>(), 32);
    assert_eq!(size_of::<LlevOperationCostsF64>(), 56);
    assert_eq!(size_of::<LlevCostMatch>(), 56);
    assert_eq!(size_of::<LlevCostBatchView>(), 24);
    assert_eq!(offset_of!(LlevCostMatch, scaled_cost), 32);
    assert_eq!(offset_of!(LlevCostMatch, scale_denominator), 40);
    assert_eq!(offset_of!(LlevCostMatch, unit_domain), 44);
}

unsafe fn collect_costs(
    cursor: *mut LlevCostCursor,
) -> BTreeMap<Vec<u64>, (f64, Option<usize>, u32)> {
    let mut output = BTreeMap::new();
    loop {
        let mut view = LlevCostBatchView::default();
        match llev_cost_cursor_next_batch(cursor, 2, &mut view) {
            LlevStatus::End => break,
            LlevStatus::Ok => {
                for item in std::slice::from_raw_parts(view.matches, view.len) {
                    assert_eq!(item.unit_domain, VtUnitDomain::U64);
                    assert_eq!(item.byte_len, item.term_len * 8);
                    let term =
                        std::slice::from_raw_parts(item.term_data.cast::<u64>(), item.term_len)
                            .to_vec();
                    output.insert(
                        term,
                        (
                            item.cost,
                            (item.has_scaled_cost != 0).then_some(item.scaled_cost),
                            item.scale_denominator,
                        ),
                    );
                }
                assert_eq!(
                    llev_cost_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                );
            }
            status => panic!("cost cursor failed: {status:?}"),
        }
    }
    assert_eq!(llev_cost_cursor_free(cursor), LlevStatus::Ok);
    output
}

#[test]
fn c_cost_cursor_matches_native_affine_and_weighted_token_automata() {
    let terms: [&[u64]; 5] = [
        &[10, 20, 30],
        &[10, 20, 40],
        &[10, 20],
        &[10, 20, 30, 50],
        &[99, 98, 97],
    ];
    let native_dict: DynamicDawgU64 = DynamicDawgU64::new();
    let foreign_dict = DynamicDawgBinding::new(BindingUnitDomain::U64);
    for (index, term) in terms.iter().enumerate() {
        native_dict.insert_sequence(term);
        foreign_dict.insert_u64(term, Some(index as u64)).unwrap();
    }
    let native = Transducer::new(native_dict, Algorithm::Standard);
    let resource = foreign_dict.resource();
    let raw = resource.as_raw();
    let mut transducer: *mut LlevTransducer = ptr::null_mut();
    unsafe {
        assert_eq!(
            llev_transducer_new(&raw, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );
        let affine = LlevAffineCosts {
            gap_open: 0.5,
            gap_extend: 0.25,
            substitution: 1.0,
            scale_denominator: 0,
            reserved: 0,
        };
        let params = AffineGapParams::new(0.5, 0.25, 1.0).unwrap();
        let expected: BTreeMap<_, _> = native
            .query_units_affine_scaled(&[10, 20, 30], 4, params)
            .map(|item| {
                (
                    item.term,
                    (params.unscale_cost(item.distance), Some(item.distance), 4),
                )
            })
            .collect();
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_affine(
                transducer,
                VtUnitDomain::U64 as u32,
                [10_u64, 20, 30].as_ptr().cast::<c_void>(),
                3,
                1.0,
                &affine,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        assert_eq!(collect_costs(cursor), expected);

        let mut native_costs = OperationCostsF64::standard();
        native_costs.substitution = 2.0;
        let weighted = LlevOperationCostsF64 {
            match_cost: 0.0,
            substitution: 2.0,
            insertion: 1.0,
            deletion: 1.0,
            transposition: 1.0,
            split: 1.0,
            merge: 1.0,
        };
        let expected: BTreeMap<_, _> = native
            .query_units_weighted(&[10, 20, 30], 1.0, native_costs)
            .map(|item| (item.term, (item.distance, None, 1)))
            .collect();
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_weighted(
                transducer,
                VtUnitDomain::U64 as u32,
                [10_u64, 20, 30].as_ptr().cast::<c_void>(),
                3,
                1.0,
                &weighted,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        assert_eq!(collect_costs(cursor), expected);
        llev_transducer_free(transducer);
    }
}

#[test]
fn c_cost_queries_reject_inexact_budgets_and_unsupported_kernel_without_output() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    dictionary.insert_text(b"cat", Some(1)).unwrap();
    let resource = dictionary.resource();
    let raw = resource.as_raw();
    unsafe {
        let mut transducer: *mut LlevTransducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&raw, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );
        let affine = LlevAffineCosts {
            gap_open: 0.5,
            gap_extend: 0.25,
            substitution: 1.0,
            scale_denominator: 0,
            reserved: 0,
        };
        let sentinel = ptr::dangling_mut::<LlevCostCursor>();
        let mut cursor = sentinel;
        assert_eq!(
            llev_transducer_query_affine(
                transducer,
                VtUnitDomain::UnicodeScalar as u32,
                b"cat".as_ptr().cast(),
                3,
                0.3,
                &affine,
                &mut cursor,
            ),
            LlevStatus::InvalidArgument
        );
        assert_eq!(cursor, sentinel);
        llev_transducer_free(transducer);

        let mut unsupported: *mut LlevTransducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(
                &raw,
                LlevAlgorithm::DamerauLevenshtein as u32,
                &mut unsupported,
            ),
            LlevStatus::Ok
        );
        let weighted: LlevOperationCostsF64 = OperationCostsF64::standard().into();
        assert_eq!(
            llev_transducer_query_weighted(
                unsupported,
                VtUnitDomain::UnicodeScalar as u32,
                b"cat".as_ptr().cast(),
                3,
                1.0,
                &weighted,
                &mut cursor,
            ),
            LlevStatus::InvalidArgument
        );
        assert_eq!(cursor, sentinel);
        llev_transducer_free(unsupported);
    }
}
