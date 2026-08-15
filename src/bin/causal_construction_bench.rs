//! Uninstrumented construction controls for the Java parity campaign.

use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgU64};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    dictionary: PathBuf,
    domain: String,
    constructor: String,
    warmups: usize,
    reps: usize,
}

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("causal_construction_bench: {}", message.as_ref());
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut dictionary = None;
    let mut domain = "byte".to_owned();
    let mut constructor = "from_terms".to_owned();
    let mut warmups = 5usize;
    let mut reps = 30usize;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv
            .next()
            .unwrap_or_else(|| fail(format!("{flag} requires a value")));
        match flag.as_str() {
            "--dictionary" => dictionary = Some(PathBuf::from(value)),
            "--domain" => domain = value,
            "--constructor" => constructor = value,
            "--warmups" => {
                warmups = value
                    .parse()
                    .unwrap_or_else(|_| fail("--warmups must be an integer"))
            }
            "--reps" => {
                reps = value
                    .parse()
                    .unwrap_or_else(|_| fail("--reps must be a positive integer"))
            }
            _ => fail(format!("unknown flag {flag:?}")),
        }
    }
    if reps == 0 {
        fail("--reps must be positive");
    }
    Args {
        dictionary: dictionary.unwrap_or_else(|| fail("--dictionary is required")),
        domain,
        constructor,
        warmups,
        reps,
    }
}

fn read_lines(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| fail(format!("cannot read {}: {error}", path.display())));
    let lines: Vec<_> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect();
    if lines.is_empty() {
        fail(format!("{} has no non-empty lines", path.display()));
    }
    lines
}

fn measure<D, F>(args: &Args, terms: &[String], build: F)
where
    F: Fn(&[String]) -> D,
{
    for _ in 0..args.warmups {
        std::hint::black_box(build(terms));
    }
    let mut samples = Vec::with_capacity(args.reps);
    for _ in 0..args.reps {
        let start = Instant::now();
        let dictionary = build(terms);
        let elapsed = start.elapsed().as_nanos() as u64;
        std::hint::black_box(&dictionary);
        samples.push(elapsed);
        drop(dictionary);
    }
    let mut json = String::with_capacity(1024);
    writeln!(&mut json, "{{").unwrap();
    writeln!(
        &mut json,
        "  \"schema\": \"liblevenshtein.causal-construction.v1\","
    )
    .unwrap();
    writeln!(&mut json, "  \"domain\": {:?},", args.domain).unwrap();
    writeln!(&mut json, "  \"constructor\": {:?},", args.constructor).unwrap();
    writeln!(&mut json, "  \"term_count\": {},", terms.len()).unwrap();
    writeln!(&mut json, "  \"warmups\": {},", args.warmups).unwrap();
    writeln!(&mut json, "  \"samples_ns\": [").unwrap();
    for (index, sample) in samples.iter().enumerate() {
        let comma = if index + 1 == samples.len() { "" } else { "," };
        writeln!(&mut json, "    {sample}{comma}").unwrap();
    }
    writeln!(&mut json, "  ]").unwrap();
    writeln!(&mut json, "}}").unwrap();
    print!("{json}");
}

fn main() {
    let args = parse_args();
    let terms = read_lines(&args.dictionary);
    match (args.domain.as_str(), args.constructor.as_str()) {
        ("byte", "from_terms") => measure(&args, &terms, |items| {
            DynamicDawg::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("byte", "from_sorted_terms") => measure(&args, &terms, |items| {
            DynamicDawg::<()>::from_sorted_terms(items.iter().map(String::as_str))
        }),
        ("byte", "stream") => measure(&args, &terms, |items| {
            let dictionary = DynamicDawg::<()>::new();
            for item in items {
                dictionary.insert(item);
            }
            dictionary
        }),
        ("byte", "dat_static") => measure(&args, &terms, |items| {
            DoubleArrayTrie::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("byte", "dat_incremental") => measure(&args, &terms, |items| {
            let mut builder = DoubleArrayTrieBuilder::<()>::new();
            for item in items {
                builder.insert(item);
            }
            builder.build()
        }),
        ("unicode", "from_terms") => measure(&args, &terms, |items| {
            DynamicDawgChar::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("unicode", "from_sorted_terms") => measure(&args, &terms, |items| {
            DynamicDawgChar::<()>::from_sorted_terms(items.iter().map(String::as_str))
        }),
        ("unicode", "stream") => measure(&args, &terms, |items| {
            let dictionary = DynamicDawgChar::<()>::new();
            for item in items {
                dictionary.insert(item);
            }
            dictionary
        }),
        ("u64", "from_terms") => measure(&args, &terms, |items| {
            DynamicDawgU64::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("u64", "from_sorted_terms") => measure(&args, &terms, |items| {
            DynamicDawgU64::<()>::from_sorted_terms(items.iter().map(String::as_str))
        }),
        ("u64", "stream") => measure(&args, &terms, |items| {
            let dictionary = DynamicDawgU64::<()>::new();
            for item in items {
                dictionary.insert(item);
            }
            dictionary
        }),
        ("unicode", "dat_static") => measure(&args, &terms, |items| {
            DoubleArrayTrieChar::<()>::from_terms(items.iter().map(String::as_str))
        }),
        ("byte" | "unicode" | "u64", "clone_input") => {
            measure(&args, &terms, |items| items.to_vec())
        }
        ("byte" | "unicode" | "u64", "clone_and_sort_input") => measure(&args, &terms, |items| {
            let mut prepared = items.to_vec();
            prepared.sort_unstable();
            prepared
        }),
        ("byte" | "unicode" | "u64", _) => {
            fail("unsupported constructor for this domain; expected from_terms, from_sorted_terms, stream, dat_static, dat_incremental (byte only), clone_input, or clone_and_sort_input")
        }
        _ => fail("--domain must be byte, unicode, or u64"),
    }
}
