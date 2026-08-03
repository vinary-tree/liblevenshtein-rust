#![no_main]

use libfuzzer_sys::fuzz_target;
use liblevenshtein::time_series::DtwConfig;

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .take(64)
        .map(|chunk| f64::from_bits(u64::from_le_bytes(chunk.try_into().expect("eight bytes"))))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let Some((&band, body)) = data.split_first() else {
        return;
    };
    let samples = values(body);
    let middle = samples.len() / 2;
    let (left, right) = samples.split_at(middle);
    let config = DtwConfig::new(usize::from(band));

    let forward = config.distance_squared(left, right);
    let reverse = config.distance_squared(right, left);
    assert!(!forward.is_nan());
    assert_eq!(forward, reverse);
});
