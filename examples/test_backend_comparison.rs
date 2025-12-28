use liblevenshtein::prelude::*;

fn test_dict<D: Dictionary>(name: &str, dict: D) {
    println!("\n=== {} ===", name);
    println!("Contains 'n': {}", dict.contains("n"));
    println!("Contains 'a': {}", dict.contains("a"));
    println!("Contains 'ag': {}", dict.contains("ag"));

    let transducer = Transducer::new(dict, Algorithm::Standard);
    let results: Vec<_> = transducer.query("a", 1).collect();
    println!("Query 'a' (dist=1): {:?}", results);
}

fn main() {
    let terms = vec!["n", "a", "ag"];

    test_dict(
        "DoubleArrayTrie",
        DoubleArrayTrie::from_terms(terms.clone()),
    );
    test_dict("DynamicDawg", DynamicDawg::<()>::from_terms(terms));
}
