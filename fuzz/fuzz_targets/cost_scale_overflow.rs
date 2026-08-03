#![no_main]

use libfuzzer_sys::fuzz_target;
use liblevenshtein::cost::CostScale;

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }
    let denominator = u32::from_le_bytes(data[0..4].try_into().expect("four bytes"));
    let target_denominator = u32::from_le_bytes(data[4..8].try_into().expect("four bytes"));
    let weight = f64::from_bits(u64::from_le_bytes(
        data[8..16].try_into().expect("eight bytes"),
    ));
    let cost = usize::try_from(u64::from_le_bytes(
        data[16..24].try_into().expect("eight bytes"),
    ))
    .unwrap_or(usize::MAX);

    let Ok(scale) = CostScale::new(denominator) else {
        return;
    };
    if let Ok(scaled) = scale.to_scaled(weight) {
        assert!(weight.is_finite() && weight >= 0.0);
        assert_eq!(scale.to_scaled(weight), Ok(scaled));
    }
    if let Ok(target) = CostScale::new(target_denominator) {
        let _ = scale.common(target);
        let _ = scale.rescale(cost, target);
    }
});
