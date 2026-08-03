#![no_main]

use libfuzzer_sys::fuzz_target;
use liblevenshtein::time_series::msm_interval::step_interval_column_into_with_bound;

fn decode(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks(8)
        .take(70)
        .map(|chunk| {
            if chunk.len() == 8 {
                f64::from_bits(u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
            } else {
                // Make even one-byte mutations exercise the security classes;
                // complete chunks still cover every IEEE-754 bit pattern.
                match chunk[0] % 6 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    3 => f64::MAX,
                    4 => -f64::MAX,
                    _ => f64::from(i8::from_ne_bytes([chunk[0]])),
                }
            }
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let mut values = decode(data);
    let boundary_values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        1.0,
        -1.0,
        f64::MAX,
    ];
    while values.len() < boundary_values.len() {
        values.push(boundary_values[values.len()]);
    }
    let query_len = (values[0].to_bits() as usize) % (values.len() - 6);
    let query = &values[1..1 + query_len];
    let tail = &values[query_len + 1..];
    let curr = (tail[0], tail[1]);
    let prev = Some((tail[2], tail[3]));
    let c_const = tail[4];
    let previous_column = &tail[5..];
    let mut output = vec![f64::NAN; query.len().saturating_add(1)];

    let bound = step_interval_column_into_with_bound(
        previous_column,
        query,
        curr,
        prev,
        c_const,
        &mut output,
    );
    assert!(!bound.is_nan());
    assert!(output.iter().all(|value| !value.is_nan()));
});
