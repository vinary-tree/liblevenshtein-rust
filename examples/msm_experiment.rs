//! Deterministic MSM experiment harness used for pgmcp-recorded optimization runs.
//!
//! The binary intentionally avoids external data and extra dependencies so that
//! control and treatment arms can be run from any checked-out commit:
//!
//! ```text
//! cargo run --release --example msm_experiment -- exact-range 33 3
//! cargo run --release --example msm_experiment -- exact-knn 33 3
//! cargo run --release --example msm_experiment -- variable-range 33 3
//! cargo run --release --example msm_experiment -- approx-control-knn 33 3
//! cargo run --release --example msm_experiment -- approx-paa-knn 33 3
//! cargo run --release --example msm_experiment -- ucr-1nn-latency 5 1 target/msm-corpora/ItalyPowerDemand ItalyPowerDemand
//! cargo run --release --example msm_experiment -- ucr-1nn-outcomes 0 0 target/msm-corpora/ItalyPowerDemand ItalyPowerDemand
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use liblevenshtein::time_series::{
    msm_distance_automaton, msm_distance_wavefront, ApproxMsmConfig, ApproxMsmIndex, MsmConfig,
    MsmTransducer, QuantizationConfig,
};

#[derive(Debug, Clone)]
struct LabeledSeries {
    label: String,
    series: Vec<f64>,
}

fn generate_series(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut series = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = ((state >> 33) as f64) / (u32::MAX as f64) * 100.0;
        series.push(value);
    }
    series
}

fn generate_prefix_shared_database(db_size: usize, len: usize) -> Vec<Vec<f64>> {
    let base = generate_series(len, 42);
    (0..db_size)
        .map(|i| {
            let mut series = base.clone();
            let pivot = len.saturating_mul(3) / 4;
            for (j, value) in series.iter_mut().enumerate().skip(pivot) {
                let perturb = ((i as f64 + 1.0) * (j as f64 + 0.5)).sin() * 2.0;
                *value = (*value + perturb).clamp(0.0, 100.0);
            }
            series
        })
        .collect()
}

fn build_case() -> (MsmTransducer<usize>, Vec<f64>) {
    let database = generate_prefix_shared_database(512, 48);
    let query = database[0]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i % 7 == 0 {
                (*v + 1.5).clamp(0.0, 100.0)
            } else {
                *v
            }
        })
        .collect::<Vec<_>>();
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 100.0),
        MsmConfig::new(1.0),
        &database,
    );
    (index, query)
}

fn generate_variable_length_database(db_size: usize, query_len: usize) -> Vec<Vec<f64>> {
    let base = generate_series(query_len, 2026);
    (0..db_size)
        .map(|i| {
            let len = query_len + (i % 96);
            let mut series = Vec::with_capacity(len);
            for j in 0..len {
                let anchor = base[j.min(query_len - 1)];
                let drift = ((i as f64 + 0.25) * (j as f64 + 1.0)).cos() * 0.01;
                series.push((anchor + drift).clamp(0.0, 100.0));
            }
            series
        })
        .collect()
}

fn build_variable_case() -> (MsmTransducer<usize>, Vec<f64>, f64) {
    let query_len = 16;
    let database = generate_variable_length_database(768, query_len);
    let query = database[0][..query_len].to_vec();
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 100.0),
        MsmConfig::new(1.0),
        &database,
    );
    (index, query, 8.0)
}

fn generate_approx_database(db_size: usize, len: usize) -> Vec<Vec<f64>> {
    (0..db_size)
        .map(|i| {
            let phase = (i % 32) as f64 * 0.07;
            let amplitude = 15.0 + (i % 11) as f64 * 0.35;
            let offset = 50.0 + ((i / 32) % 7) as f64 * 0.8;
            (0..len)
                .map(|j| {
                    let t = j as f64 / len as f64;
                    let wave = (t * std::f64::consts::TAU * 3.0 + phase).sin();
                    let harmonic = (t * std::f64::consts::TAU * 7.0 + phase * 0.5).cos() * 2.0;
                    (offset + amplitude * wave + harmonic).clamp(0.0, 100.0)
                })
                .collect()
        })
        .collect()
}

fn exact_brute_knn(
    database: &[Vec<f64>],
    query: &[f64],
    msm: MsmConfig,
    k: usize,
) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = database
        .iter()
        .enumerate()
        .map(|(idx, series)| (idx, msm.distance(query, series)))
        .filter(|(_, distance)| distance.is_finite())
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    out
}

fn recall_at_k(approx: &[(usize, f64)], exact: &[(usize, f64)]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let hits = approx
        .iter()
        .filter(|(id, _)| exact.iter().any(|(exact_id, _)| exact_id == id))
        .count();
    hits as f64 / exact.len() as f64
}

fn build_approx_case() -> (
    ApproxMsmIndex<usize>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<(usize, f64)>,
) {
    let database = generate_approx_database(1024, 96);
    let mut query = database[321].clone();
    for (i, value) in query.iter_mut().enumerate() {
        if i % 9 == 0 {
            *value = (*value + 0.25).clamp(0.0, 100.0);
        }
    }
    let msm = MsmConfig::new(1.0);
    let exact = exact_brute_knn(&database, &query, msm, 8);
    let index = ApproxMsmIndex::from_series(ApproxMsmConfig::new(16, 128, msm), &database);
    (index, database, query, exact)
}

fn parse_ucr_ts(path: &Path) -> Result<Vec<LabeledSeries>, String> {
    let content =
        fs::read_to_string(path).map_err(|err| format!("failed to read {path:?}: {err}"))?;
    let mut in_data = false;
    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !in_data {
            if line.eq_ignore_ascii_case("@data") {
                in_data = true;
            }
            continue;
        }

        let (values, label) = line
            .rsplit_once(':')
            .ok_or_else(|| format!("{path:?}:{} missing ':' label separator", line_no + 1))?;
        let series = values
            .split(',')
            .map(|value| {
                value.trim().parse::<f64>().map_err(|err| {
                    format!(
                        "{path:?}:{} invalid numeric value {value:?}: {err}",
                        line_no + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(LabeledSeries {
            label: label.trim().to_string(),
            series,
        });
    }
    Ok(rows)
}

fn parse_ucr_txt(path: &Path) -> Result<Vec<LabeledSeries>, String> {
    let content =
        fs::read_to_string(path).map_err(|err| format!("failed to read {path:?}: {err}"))?;
    let mut rows = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let label = fields
            .next()
            .ok_or_else(|| format!("{path:?}:{} missing label", line_no + 1))?;
        let series = fields
            .map(|value| {
                value.parse::<f64>().map_err(|err| {
                    format!(
                        "{path:?}:{} invalid numeric value {value:?}: {err}",
                        line_no + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(LabeledSeries {
            label: label.to_string(),
            series,
        });
    }
    Ok(rows)
}

fn find_ucr_split_paths(
    dataset_dir: &Path,
    dataset_name: &str,
) -> Result<(PathBuf, PathBuf), String> {
    for extension in ["ts", "txt"] {
        let train = dataset_dir.join(format!("{dataset_name}_TRAIN.{extension}"));
        let test = dataset_dir.join(format!("{dataset_name}_TEST.{extension}"));
        if train.exists() && test.exists() {
            return Ok((train, test));
        }
    }
    Err(format!(
        "could not find {dataset_name}_TRAIN/{{ts,txt}} and {dataset_name}_TEST/{{ts,txt}} under {dataset_dir:?}"
    ))
}

fn load_ucr_dataset(
    dataset_dir: &Path,
    dataset_name: &str,
) -> Result<(Vec<LabeledSeries>, Vec<LabeledSeries>), String> {
    let (train_path, test_path) = find_ucr_split_paths(dataset_dir, dataset_name)?;
    let parse = |path: &Path| match path.extension().and_then(|ext| ext.to_str()) {
        Some("ts") => parse_ucr_ts(path),
        Some("txt") => parse_ucr_txt(path),
        other => Err(format!("unsupported UCR extension for {path:?}: {other:?}")),
    };
    Ok((parse(&train_path)?, parse(&test_path)?))
}

fn ucr_1nn_accuracy(
    train: &[LabeledSeries],
    test: &[LabeledSeries],
    msm: MsmConfig,
) -> (usize, usize, f64) {
    let mut correct = 0usize;
    for probe in test {
        let predicted = train
            .iter()
            .map(|candidate| {
                (
                    candidate.label.as_str(),
                    msm.distance(&probe.series, &candidate.series),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label)
            .unwrap_or("");
        if predicted == probe.label {
            correct += 1;
        }
    }
    let total = test.len();
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    };
    (correct, total, accuracy)
}

fn majority_label(train: &[LabeledSeries]) -> &str {
    let mut counts = BTreeMap::<&str, usize>::new();
    for row in train {
        *counts.entry(row.label.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|(left_label, left_count), (right_label, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_label.cmp(left_label))
        })
        .map(|(label, _)| label)
        .unwrap_or("")
}

fn ucr_1nn_outcomes(
    train: &[LabeledSeries],
    test: &[LabeledSeries],
    msm: MsmConfig,
) -> Vec<(bool, bool)> {
    let majority = majority_label(train);
    test.iter()
        .map(|probe| {
            let predicted = train
                .iter()
                .map(|candidate| {
                    (
                        candidate.label.as_str(),
                        msm.distance(&probe.series, &candidate.series),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(label, _)| label)
                .unwrap_or("");
            (majority == probe.label, predicted == probe.label)
        })
        .collect()
}

fn measure_legacy_ratio(scenario: &str) -> f64 {
    let x = generate_series(24, 12345);
    let y = generate_series(24, 67890);
    let config = MsmConfig::new(1.0);
    let iterations = 128;

    let started = Instant::now();
    let mut optimized_checksum = 0.0;
    for _ in 0..iterations {
        optimized_checksum += config.distance_optimized(&x, &y);
    }
    let optimized_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let mut candidate_checksum = 0.0;
    for _ in 0..iterations {
        candidate_checksum += match scenario {
            "legacy-wavefront-ratio" => {
                msm_distance_wavefront(&x, &y, &config, f64::INFINITY).unwrap()
            }
            "legacy-automaton-ratio" => {
                msm_distance_automaton(&x, &y, &config, f64::INFINITY).unwrap()
            }
            other => panic!("unknown legacy scenario: {other}"),
        };
    }
    let candidate_ms = started.elapsed().as_secs_f64() * 1000.0;

    assert!(
        (optimized_checksum - candidate_checksum).abs() < 1e-6,
        "legacy path diverged from optimized DP"
    );
    candidate_ms / optimized_ms
}

fn main() {
    let mut args = env::args().skip(1);
    let scenario = args.next().unwrap_or_else(|| "exact-range".to_string());
    let measured_runs = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30);
    let warmup_runs = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let total_runs = warmup_runs + measured_runs;

    let (index, query) = build_case();
    let (variable_index, variable_query, variable_threshold) = build_variable_case();
    let approx_case = scenario.starts_with("approx-").then(build_approx_case);
    let ucr_case = if scenario.starts_with("ucr-") {
        let dataset_dir = args
            .next()
            .unwrap_or_else(|| "target/msm-corpora/ItalyPowerDemand".to_string());
        let dataset_name = args
            .next()
            .unwrap_or_else(|| "ItalyPowerDemand".to_string());
        Some(
            load_ucr_dataset(Path::new(&dataset_dir), &dataset_name)
                .unwrap_or_else(|err| panic!("failed to load UCR dataset: {err}")),
        )
    } else {
        None
    };
    let threshold = 24.0;
    let k = 8;

    if scenario == "ucr-1nn-outcomes" {
        let (train, test) = ucr_case.as_ref().unwrap();
        println!("scenario,arm,case,correct");
        for (case, (control_correct, treatment_correct)) in
            ucr_1nn_outcomes(train, test, MsmConfig::new(1.0))
                .into_iter()
                .enumerate()
        {
            println!("{scenario},majority,{case},{}", u8::from(control_correct));
            println!(
                "{scenario},exact_msm_1nn,{case},{}",
                u8::from(treatment_correct)
            );
        }
        return;
    }

    println!("scenario,phase,run,elapsed_ms,result_len,checksum");
    for run in 0..total_runs {
        let started = Instant::now();
        let (metric_value, result_len, checksum) = match scenario.as_str() {
            "exact-knn" => {
                let results = index.search_knn(&query, k, 1.0);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let checksum = results.iter().fold(0.0, |acc, (id, distance)| {
                    acc + *id as f64 * 0.001 + *distance
                });
                (elapsed_ms, results.len(), checksum)
            }
            "exact-range" => {
                let results = index.search_range(&query, threshold);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let checksum = results.iter().fold(0.0, |acc, (id, distance)| {
                    acc + *id as f64 * 0.001 + *distance
                });
                (elapsed_ms, results.len(), checksum)
            }
            "variable-range" => {
                let results = variable_index.search_range(&variable_query, variable_threshold);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let checksum = results.iter().fold(0.0, |acc, (id, distance)| {
                    acc + *id as f64 * 0.001 + *distance
                });
                (elapsed_ms, results.len(), checksum)
            }
            "approx-control-knn" => {
                let (_, database, approx_query, _) = approx_case.as_ref().unwrap();
                let results = exact_brute_knn(database, approx_query, MsmConfig::new(1.0), k);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let checksum = results.iter().fold(0.0, |acc, (id, distance)| {
                    acc + *id as f64 * 0.001 + *distance
                });
                (elapsed_ms, results.len(), checksum)
            }
            "approx-paa-knn" => {
                let (approx_index, _, approx_query, _) = approx_case.as_ref().unwrap();
                let results = approx_index.search_knn(approx_query, k);
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                let checksum = results.iter().fold(0.0, |acc, (id, distance)| {
                    acc + *id as f64 * 0.001 + *distance
                });
                (elapsed_ms, results.len(), checksum)
            }
            "approx-paa-recall" => {
                let (approx_index, _, approx_query, exact) = approx_case.as_ref().unwrap();
                let results = approx_index.search_knn(approx_query, k);
                let recall = recall_at_k(&results, exact);
                (recall, results.len(), recall)
            }
            "ucr-1nn-latency" => {
                let (train, test) = ucr_case.as_ref().unwrap();
                let (correct, _, accuracy) = ucr_1nn_accuracy(train, test, MsmConfig::new(1.0));
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                (elapsed_ms, correct, accuracy)
            }
            "ucr-1nn-accuracy" => {
                let (train, test) = ucr_case.as_ref().unwrap();
                let (correct, _, accuracy) = ucr_1nn_accuracy(train, test, MsmConfig::new(1.0));
                (accuracy, correct, accuracy)
            }
            "legacy-wavefront-ratio" | "legacy-automaton-ratio" => {
                let ratio = measure_legacy_ratio(&scenario);
                (ratio, 1, ratio)
            }
            other => panic!("unknown scenario: {other}"),
        };
        let phase = if run < warmup_runs {
            "warmup"
        } else {
            "measure"
        };
        println!(
            "{scenario},{phase},{},{metric_value:.6},{result_len},{checksum}",
            run.saturating_sub(warmup_runs),
        );
    }
}
