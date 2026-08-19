#![cfg(all(
    feature = "binding-integration-tests",
    feature = "perf-instrumentation"
))]

use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};
use libdictenstein::{causal_construction_stats, reset_causal_construction_stats};
#[cfg(feature = "bindings-phonetic")]
use liblevenshtein::bindings::PhoneticPattern;
use liblevenshtein::bindings::{
    Match, MatchBatch, MatchTerm, QueryCursor, QueryOrder, ResourceTransducer,
};
use liblevenshtein::transducer::Algorithm;
use liblevenshtein::{causal_perf_stats, reset_causal_perf_stats};
use std::sync::{Arc, Barrier};

fn drain(cursor: &mut QueryCursor) -> Vec<Match> {
    let mut output = Vec::new();
    let mut batch = MatchBatch::default();
    loop {
        let written = cursor.next_batch(&mut batch, 2).expect("drain graph query");
        if written == 0 {
            return output;
        }
        output.extend_from_slice(batch.as_slice());
    }
}

fn transducer(dictionary: &DynamicDawgBinding) -> ResourceTransducer {
    let resource = dictionary.resource();
    unsafe { ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard) }
        .expect("retain real DynamicDawg resource")
}

#[test]
fn concurrent_cold_queries_import_one_graph_without_registry_locking() {
    const THREADS: usize = 16;
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for index in 0..512 {
        dictionary
            .insert_text(format!("term-{index:04}").as_bytes(), Some(index))
            .expect("seed insert");
    }
    let transducer = Arc::new(transducer(&dictionary));
    let barrier = Arc::new(Barrier::new(THREADS));

    reset_causal_perf_stats();
    reset_causal_construction_stats();
    let workers = (0..THREADS)
        .map(|_| {
            let transducer = Arc::clone(&transducer);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                drain(
                    &mut transducer
                        .query_utf8("term-0010", 1, QueryOrder::Traversal)
                        .expect("concurrent graph query"),
                )
            })
        })
        .collect::<Vec<_>>();

    let mut reference = None;
    for worker in workers {
        let matches = worker.join().expect("query thread");
        if let Some(reference) = &reference {
            assert_eq!(&matches, reference);
        } else {
            reference = Some(matches);
        }
    }
    let consumer = causal_perf_stats();
    assert_eq!(
        consumer.foreign_graph_decodes, 1,
        "one revision cell must single-flight the complete graph import: {consumer:?}"
    );
    assert_eq!(consumer.foreign_node_cache_hits, 0);
    assert_eq!(consumer.foreign_node_cache_misses, 0);

    let provider = causal_construction_stats();
    assert_eq!(provider.resource_snapshots_created, 1);
    assert_eq!(provider.resource_graph_projections, 1);
    assert_eq!(provider.resource_graph_calls, 1);
    assert_eq!(provider.resource_edges_calls, 0);
    assert_eq!(provider.resource_is_final_calls, 0);
}

#[test]
fn real_dynamic_dawg_queries_share_one_graph_per_revision_in_all_unit_domains() {
    let bytes = DynamicDawgBinding::new(BindingUnitDomain::Byte);
    bytes.insert_text(&[0], Some(10)).unwrap();
    bytes.insert_text(b"a", None).unwrap();
    bytes.insert_text(&[u8::MAX], Some(12)).unwrap();

    let unicode = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    unicode.insert_text("a".as_bytes(), Some(20)).unwrap();
    unicode.insert_text("é".as_bytes(), None).unwrap();
    unicode
        .insert_text("\u{10ffff}".as_bytes(), Some(22))
        .unwrap();

    let tokens = DynamicDawgBinding::new(BindingUnitDomain::U64);
    tokens.insert_u64(&[0], Some(30)).unwrap();
    tokens.insert_u64(&[42], None).unwrap();
    tokens.insert_u64(&[u64::MAX], Some(32)).unwrap();

    let byte_transducer = transducer(&bytes);
    let unicode_transducer = transducer(&unicode);
    let token_transducer = transducer(&tokens);

    reset_causal_perf_stats();
    reset_causal_construction_stats();

    for _ in 0..2 {
        let byte_matches = drain(
            &mut byte_transducer
                .query_bytes(&[], 1, QueryOrder::Traversal)
                .unwrap(),
        );
        assert_eq!(
            byte_matches
                .iter()
                .map(|item| (item.term.clone(), item.id))
                .collect::<Vec<_>>(),
            vec![
                (MatchTerm::Bytes(vec![0]), Some(10)),
                (MatchTerm::Bytes(b"a".to_vec()), None),
                (MatchTerm::Bytes(vec![u8::MAX]), Some(12)),
            ]
        );

        let unicode_matches = drain(
            &mut unicode_transducer
                .query_utf8("", 1, QueryOrder::Traversal)
                .unwrap(),
        );
        assert_eq!(
            unicode_matches
                .iter()
                .map(|item| (item.term.clone(), item.id))
                .collect::<Vec<_>>(),
            vec![
                (MatchTerm::Utf8("a".to_owned()), Some(20)),
                (MatchTerm::Utf8("é".to_owned()), None),
                (MatchTerm::Utf8("\u{10ffff}".to_owned()), Some(22)),
            ]
        );

        let token_matches = drain(
            &mut token_transducer
                .query_u64(&[], 1, QueryOrder::Traversal)
                .unwrap(),
        );
        assert_eq!(
            token_matches
                .iter()
                .map(|item| (item.term.clone(), item.id))
                .collect::<Vec<_>>(),
            vec![
                (MatchTerm::U64(vec![0]), Some(30)),
                (MatchTerm::U64(vec![42]), None),
                (MatchTerm::U64(vec![u64::MAX]), Some(32)),
            ]
        );
    }

    assert!(unicode.insert_text(b"b", Some(23)).unwrap());
    let revised = drain(
        &mut unicode_transducer
            .query_utf8("", 1, QueryOrder::Traversal)
            .unwrap(),
    );
    assert_eq!(revised.len(), 4);
    assert!(revised
        .iter()
        .any(|item| item.term == MatchTerm::Utf8("b".to_owned()) && item.id == Some(23)));

    let consumer = causal_perf_stats();
    assert_eq!(
        consumer.foreign_graph_decodes,
        4,
        "consumer={consumer:?}; provider={:?}",
        causal_construction_stats()
    );
    assert_eq!(consumer.foreign_is_final_callbacks, 0);
    assert_eq!(consumer.foreign_edge_callbacks, 0);
    assert_eq!(consumer.foreign_edge_pages, 0);
    assert_eq!(consumer.foreign_edge_descriptors, 0);

    let provider = causal_construction_stats();
    assert_eq!(provider.resource_snapshots_created, 4);
    assert_eq!(provider.resource_graph_projections, 4);
    assert_eq!(provider.resource_graph_calls, 4);
    assert!(provider.resource_graph_value_calls > 0);
    assert_eq!(
        provider.resource_value_calls, provider.resource_graph_value_calls,
        "every value must resolve through a graph value cursor, never the base node callback"
    );
    assert_eq!(provider.resource_is_final_calls, 0);
    assert_eq!(provider.resource_edges_calls, 0);
    assert_eq!(provider.resource_edge_cache_misses, 0);
    assert_eq!(provider.resource_native_edges_enumerated, 0);

    #[cfg(feature = "bindings-phonetic")]
    {
        // The language-product surface returns values rather than nodes, so it
        // can share the same compact graph session. Reset only the counters:
        // the current revision remains memoized and must require neither a new
        // projection nor a consumer decode.
        reset_causal_perf_stats();
        reset_causal_construction_stats();
        let pattern = PhoneticPattern::from_regex("a").expect("literal pattern");
        let matches = drain(
            &mut unicode_transducer
                .query_pattern(&pattern, 0)
                .expect("phonetic graph query"),
        );
        assert_eq!(
            matches
                .iter()
                .map(|item| (item.term.clone(), item.id))
                .collect::<Vec<_>>(),
            vec![(MatchTerm::Utf8("a".to_owned()), Some(20))]
        );

        let consumer = causal_perf_stats();
        assert_eq!(consumer.foreign_graph_decodes, 0);
        assert_eq!(consumer.foreign_is_final_callbacks, 0);
        assert_eq!(consumer.foreign_edge_callbacks, 0);

        let provider = causal_construction_stats();
        assert_eq!(provider.resource_graph_projections, 0);
        assert_eq!(provider.resource_graph_calls, 0);
        assert_eq!(provider.resource_graph_value_calls, 1);
        assert_eq!(provider.resource_value_calls, 1);
        assert_eq!(provider.resource_is_final_calls, 0);
        assert_eq!(provider.resource_edges_calls, 0);
    }
}
