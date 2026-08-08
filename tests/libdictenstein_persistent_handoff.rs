#![cfg(feature = "binding-persistent-integration-tests")]

use libdictenstein::bindings::{BindingUnitDomain, PersistentARTrieBinding};
use liblevenshtein::bindings::{
    Match, MatchBatch, MatchTerm, QueryCursor, QueryOrder, ResourceTransducer,
};
use liblevenshtein::transducer::Algorithm;
use std::collections::BTreeMap;

fn drain(cursor: &mut QueryCursor) -> Vec<Match> {
    let mut output = Vec::new();
    let mut batch = MatchBatch::default();
    loop {
        if cursor.next_batch(&mut batch, 2).unwrap() == 0 {
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

#[test]
fn persistent_artrie_checkpoint_does_not_change_a_long_lived_cursor() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dictionary.part");
    let dictionary =
        PersistentARTrieBinding::create(&path, BindingUnitDomain::UnicodeScalar).unwrap();
    for (term, value) in [("cat", Some(1)), ("cot", Some(2)), ("cut", None)] {
        assert!(dictionary.insert_text(term.as_bytes(), value).unwrap());
    }
    dictionary.checkpoint().unwrap();

    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let mut first_batch = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut first_batch, 1).unwrap(), 1);
    let first = first_batch.as_slice()[0].clone();

    assert!(dictionary.remove_text(b"cot").unwrap());
    assert!(!dictionary.insert_text(b"cut", Some(30)).unwrap());
    assert!(dictionary.insert_text(b"cit", Some(4)).unwrap());
    dictionary.checkpoint().unwrap();

    drop(transducer);
    drop(resource);
    let frozen = map(std::iter::once(first).chain(drain(&mut cursor)));
    assert_eq!(
        frozen,
        BTreeMap::from([
            ("cat".to_owned(), Some(1)),
            ("cot".to_owned(), Some(2)),
            ("cut".to_owned(), None),
        ])
    );
    drop(dictionary);

    let reopened = PersistentARTrieBinding::open(&path, BindingUnitDomain::UnicodeScalar).unwrap();
    assert!(reopened.contains_text(b"cat").unwrap());
    assert!(!reopened.contains_text(b"cot").unwrap());
    assert_eq!(reopened.value_text(b"cut").unwrap(), Some(Some(30)));
    assert_eq!(reopened.value_text(b"cit").unwrap(), Some(Some(4)));
}
