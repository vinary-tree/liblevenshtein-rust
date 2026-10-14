//! Native control for the Raku phonetic online-scan boundary benchmark.

use liblevenshtein::phonetic::grep_online::PhoneticGrepOnline;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let input = "cat dog ".repeat(128);
    let matcher = PhoneticGrepOnline::without_rules("cat", 0);
    let mut samples = Vec::new();
    for iteration in 0..25 {
        let start = Instant::now();
        let matches = matcher.scan(black_box(&input));
        assert_eq!(matches.len(), 128);
        if iteration >= 5 {
            samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "native_rust_scan median_us={:.3} matches=128 bytes={} runs=20",
        (samples[9] + samples[10]) / 2.0,
        input.len()
    );
}
