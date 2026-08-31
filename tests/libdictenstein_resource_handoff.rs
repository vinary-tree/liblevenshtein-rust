#![cfg(feature = "binding-integration-tests")]

use libdictenstein::bindings::{
    BindingUnitDomain, DoubleArrayTrieBinding, DynamicDawgBinding, ScdawgBinding,
};
use liblevenshtein::bindings::{
    Match, MatchBatch, MatchTerm, QueryCursor, QueryOrder, ResourceQueryCache, ResourceTransducer,
};
use liblevenshtein::ffi::{
    llev_query_cache_clear, llev_query_cache_free, llev_query_cache_new,
    llev_query_cache_query_utf8, llev_query_cache_reset_stats, llev_query_cache_stats,
    llev_query_cursor_free, llev_query_cursor_next_batch, llev_query_cursor_release_batch,
    llev_transducer_free, llev_transducer_new, LlevAlgorithm, LlevMatchBatchView, LlevQueryCache,
    LlevQueryCacheStats, LlevQueryCursor, LlevQueryOrder, LlevStatus, LlevTransducer,
};
use liblevenshtein::transducer::{Algorithm, QueryCacheLimits};
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::ptr;

fn drain(cursor: &mut QueryCursor) -> Vec<Match> {
    let mut output = Vec::new();
    let mut batch = MatchBatch::default();
    loop {
        let written = cursor.next_batch(&mut batch, 3).unwrap();
        if written == 0 {
            return output;
        }
        output.extend_from_slice(batch.as_slice());
    }
}

fn map(matches: impl IntoIterator<Item = Match>) -> BTreeMap<String, Option<u64>> {
    matches
        .into_iter()
        .map(|item| match item.term {
            MatchTerm::Utf8(term) => (term, item.id),
            other => panic!("unexpected term domain: {other:?}"),
        })
        .collect()
}

unsafe fn drain_ffi_cursor(cursor: *mut LlevQueryCursor) -> usize {
    let mut count = 0;
    loop {
        let mut view = LlevMatchBatchView::default();
        match llev_query_cursor_next_batch(cursor, 2, &mut view) {
            LlevStatus::Ok => {
                count += view.len;
                assert_eq!(
                    llev_query_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => break,
            status => panic!("unexpected cursor status {status:?}"),
        }
    }
    assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    count
}

#[test]
fn c_query_cache_owns_results_reports_policy_and_leaves_cursors_independent() {
    unsafe {
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
        for (term, value) in [("cat", 1), ("cot", 2), ("cut", 3)] {
            dictionary
                .insert_text(term.as_bytes(), Some(value))
                .expect("seed dictionary");
        }
        let resource = dictionary.resource();
        let raw = resource.as_raw();
        let mut transducer: *mut LlevTransducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&raw, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );
        let mut cache: *mut LlevQueryCache = ptr::null_mut();
        assert_eq!(
            llev_query_cache_new(transducer, 8, 1 << 20, &mut cache),
            LlevStatus::Ok
        );

        let mut cold: *mut LlevQueryCursor = ptr::null_mut();
        assert_eq!(
            llev_query_cache_query_utf8(
                cache,
                b"cat".as_ptr().cast(),
                3,
                1,
                LlevQueryOrder::Traversal as u32,
                &mut cold,
            ),
            LlevStatus::Ok
        );
        assert_eq!(drain_ffi_cursor(cold), 3);

        let mut hit: *mut LlevQueryCursor = ptr::null_mut();
        assert_eq!(
            llev_query_cache_query_utf8(
                cache,
                b"cat".as_ptr().cast(),
                3,
                1,
                LlevQueryOrder::Traversal as u32,
                &mut hit,
            ),
            LlevStatus::Ok
        );
        let mut stats = LlevQueryCacheStats::default();
        assert_eq!(llev_query_cache_stats(cache, &mut stats), LlevStatus::Ok);
        assert_eq!((stats.requests, stats.hits, stats.misses), (2, 1, 1));
        assert_eq!(stats.resident_entries, 1);
        assert!(stats.resident_weight > 0);

        assert_eq!(llev_query_cache_reset_stats(cache), LlevStatus::Ok);
        assert_eq!(llev_query_cache_stats(cache, &mut stats), LlevStatus::Ok);
        assert_eq!(stats.requests, 0);
        assert_eq!(stats.resident_entries, 1);
        assert_eq!(llev_query_cache_clear(cache), LlevStatus::Ok);
        assert_eq!(llev_query_cache_stats(cache, &mut stats), LlevStatus::Ok);
        assert_eq!(stats.resident_entries, 0);

        llev_query_cache_free(cache);
        llev_transducer_free(transducer);
        drop(resource);
        drop(dictionary);
        assert_eq!(drain_ffi_cursor(hit), 3, "cursor retains its own result");
    }
}

#[test]
fn c_query_cache_rejects_null_handles_without_writing_outputs() {
    unsafe {
        let mut cache = std::ptr::dangling_mut::<LlevQueryCache>();
        assert_eq!(
            llev_query_cache_new(ptr::null(), 8, 1024, &mut cache),
            LlevStatus::NullPointer
        );
        let mut cursor = std::ptr::dangling_mut::<LlevQueryCursor>();
        assert_eq!(
            llev_query_cache_query_utf8(
                ptr::null_mut(),
                ptr::null(),
                0,
                0,
                LlevQueryOrder::Traversal as u32,
                &mut cursor,
            ),
            LlevStatus::NullPointer
        );
        assert_eq!(
            llev_query_cache_stats(ptr::null(), ptr::null_mut()),
            LlevStatus::NullPointer
        );
        assert_eq!(
            llev_query_cache_clear(ptr::null_mut()),
            LlevStatus::NullPointer
        );
        assert_eq!(
            llev_query_cache_reset_stats(ptr::null_mut()),
            LlevStatus::NullPointer
        );
        llev_query_cache_free(ptr::null_mut());
    }
}

#[test]
fn real_libdictenstein_resource_has_one_query_start_revision() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for (term, value) in [("cat", 1), ("cot", 2), ("cut", 3), ("scat", 4)] {
        assert!(dictionary
            .insert_text(term.as_bytes(), Some(value))
            .unwrap());
    }
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let mut one = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut one, 1).unwrap(), 1);
    let first = one.as_slice()[0].clone();

    assert!(dictionary.remove_text(b"cot").unwrap());
    assert!(!dictionary.insert_text(b"cut", Some(30)).unwrap());
    assert!(dictionary.insert_text(b"cit", Some(5)).unwrap());
    dictionary.compact();
    dictionary.clear();
    assert!(dictionary.insert_text(b"new", Some(99)).unwrap());

    let fresh = map(drain(
        &mut transducer
            .query_utf8("cat", 8, QueryOrder::Traversal)
            .unwrap(),
    ));
    assert_eq!(fresh, BTreeMap::from([("new".to_owned(), Some(99))]));

    drop(transducer);
    drop(resource);
    drop(dictionary);
    let frozen = map(std::iter::once(first).chain(drain(&mut cursor)));
    assert_eq!(
        frozen,
        BTreeMap::from([
            ("cat".to_owned(), Some(1)),
            ("cot".to_owned(), Some(2)),
            ("cut".to_owned(), Some(3)),
            ("scat".to_owned(), Some(4)),
        ])
    );
}

#[test]
fn bounded_resource_cache_hits_and_invalidates_on_revision_change() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for (term, value) in [("cat", 1), ("cot", 2), ("cut", 3)] {
        dictionary
            .insert_text(term.as_bytes(), Some(value))
            .expect("seed dictionary");
    }
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cache = ResourceQueryCache::new(transducer, QueryCacheLimits::new(8, 1 << 20));

    let first = map(drain(
        &mut cache
            .query_utf8("cat", 1, QueryOrder::Traversal)
            .expect("cold cached query"),
    ));
    let second = map(drain(
        &mut cache
            .query_utf8("cat", 1, QueryOrder::Traversal)
            .expect("resident cached query"),
    ));
    assert_eq!(second, first);
    assert_eq!(cache.traversal_stats().requests(), 2);
    assert_eq!(cache.traversal_stats().hits(), 1);
    assert_eq!(cache.traversal_stats().misses(), 1);
    assert_eq!(cache.len(), 1);

    dictionary.remove_text(b"cot").expect("mutate dictionary");
    dictionary
        .insert_text(b"cit", Some(4))
        .expect("mutate dictionary");
    let revised = map(drain(
        &mut cache
            .query_utf8("cat", 1, QueryOrder::Traversal)
            .expect("revision-invalidated query"),
    ));
    assert_ne!(revised, first);
    assert_eq!(cache.traversal_stats().misses(), 2);
    assert_eq!(
        cache.len(),
        1,
        "stale residency is cleared before admission"
    );
}

#[test]
fn resource_cache_keeps_result_orders_in_independent_policy_shards() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for (term, value) in [("cat", 1), ("cot", 2), ("cut", 3)] {
        dictionary
            .insert_text(term.as_bytes(), Some(value))
            .expect("seed dictionary");
    }
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cache = ResourceQueryCache::new(transducer, QueryCacheLimits::new(8, 1 << 20));

    let _ = drain(&mut cache.query_utf8("cat", 1, QueryOrder::Traversal).unwrap());
    let ordered = drain(
        &mut cache
            .query_utf8("cat", 1, QueryOrder::DistanceThenTerm)
            .unwrap(),
    );
    let ordered_hit = drain(
        &mut cache
            .query_utf8("cat", 1, QueryOrder::DistanceThenTerm)
            .unwrap(),
    );

    assert_eq!(ordered_hit, ordered);
    assert_eq!(cache.traversal_stats().misses(), 1);
    assert_eq!(cache.ordered_stats().misses(), 1);
    assert_eq!(cache.ordered_stats().hits(), 1);
    assert_eq!(cache.len(), 2);

    dictionary.remove_text(b"cot").expect("mutate dictionary");
    let _ = drain(
        &mut cache
            .query_utf8("cat", 1, QueryOrder::DistanceThenTerm)
            .expect("revision-invalidated ordered query"),
    );
    assert_eq!(
        cache.len(),
        1,
        "observing a new revision clears stale results from every order shard"
    );
}

#[test]
#[cfg(feature = "perf-instrumentation")]
fn distinct_snapshot_resources_share_identity_keyed_node_caches() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for term in ["cat", "cot", "cut", "scat"] {
        dictionary
            .insert_text(term.as_bytes(), None)
            .expect("seed insert");
    }
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };

    let first = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let second = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    assert_eq!(first.snapshot_identity(), second.snapshot_identity());
    assert!(first.snapshot_identity().is_some());
    assert!(first.shares_node_cache_with(&second));

    dictionary.insert_text(b"cit", None).expect("mutate");
    let next = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let (producer, revision) = first.snapshot_identity().unwrap();
    let (next_producer, next_revision) = next.snapshot_identity().unwrap();
    assert_eq!(producer, next_producer);
    assert!(next_revision > revision);
    assert!(!first.shares_node_cache_with(&next));
}

#[test]
fn double_array_trie_resource_composes_without_serialization() {
    let dictionary = DoubleArrayTrieBinding::from_unicode_terms([
        ("café", Some(7)),
        ("caff", None),
        ("tea", Some(9)),
    ]);
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let observed = map(drain(
        &mut transducer
            .query_utf8("cafe", 2, QueryOrder::Traversal)
            .unwrap(),
    ));
    assert_eq!(
        observed,
        BTreeMap::from([("caff".to_owned(), None), ("café".to_owned(), Some(7)),])
    );
}

#[test]
fn scdawg_long_lived_cursor_keeps_its_published_revision() {
    let dictionary = ScdawgBinding::new_unicode();
    assert!(dictionary.insert("cat", Some(1)));
    assert!(dictionary.insert("cot", Some(2)));
    assert!(dictionary.insert("cut", None));
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let mut one = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut one, 1).unwrap(), 1);
    let first = one.as_slice()[0].clone();

    assert!(dictionary.insert("cit", Some(4)));
    assert!(dictionary.contains_substring("it"));
    assert_eq!(dictionary.frequency("it"), 1);

    let frozen = map(std::iter::once(first).chain(drain(&mut cursor)));
    assert_eq!(
        frozen,
        BTreeMap::from([
            ("at".to_owned(), Some(1)),
            ("cat".to_owned(), Some(1)),
            ("cot".to_owned(), Some(2)),
            ("cut".to_owned(), None),
            ("ot".to_owned(), Some(2)),
            ("ut".to_owned(), None),
        ])
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn arbitrary_real_producer_mutations_do_not_change_a_partially_consumed_cursor(
        initial in prop::collection::vec(("[a-z]{1,8}", any::<u64>()), 2..40),
        mutations in prop::collection::vec((0u8..4, "[a-z]{1,8}", any::<u64>()), 0..40),
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<String, u64> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
        for (term, value) in &initial {
            dictionary.insert_text(term.as_bytes(), Some(*value)).unwrap();
        }
        let resource = dictionary.resource();
        let transducer = unsafe {
            ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
        };
        let mut cursor = transducer.query_utf8("", 16, QueryOrder::Traversal).unwrap();
        let prefix_len = 1 + prefix_seed % (initial.len() - 1);
        let mut prefix = Vec::new();
        let mut batch = MatchBatch::default();
        for _ in 0..prefix_len {
            prop_assert_eq!(cursor.next_batch(&mut batch, 1).unwrap(), 1);
            prefix.push(batch.as_slice()[0].clone());
        }

        for (operation, term, value) in mutations {
            match operation {
                0 | 1 => {
                    dictionary.insert_text(term.as_bytes(), Some(value)).unwrap();
                }
                2 => {
                    dictionary.remove_text(term.as_bytes()).unwrap();
                }
                _ => {
                    dictionary.compact();
                }
            }
        }

        let observed = map(prefix.into_iter().chain(drain(&mut cursor)));
        let expected = initial
            .into_iter()
            .map(|(term, value)| (term, Some(value)))
            .collect::<BTreeMap<_, _>>();
        prop_assert_eq!(observed, expected);
    }
}
