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
    length_lb, msm_distance_automaton, msm_distance_wavefront, ApproxMsmConfig, ApproxMsmIndex,
    MsmConfig, MsmTransducer, QuantizationConfig,
};

#[derive(Debug, Clone)]
struct LabeledSeries {
    label: String,
    series: Vec<f64>,
}

#[derive(Debug, Clone)]
struct DatasetCandidate {
    name: String,
    dir: PathBuf,
    train_count: usize,
    test_count: usize,
    series_len: usize,
    estimated_cells: u128,
}

#[derive(Debug, Clone)]
struct DatasetEvaluation {
    candidate: DatasetCandidate,
    majority_correct: usize,
    msm_correct: usize,
    total: usize,
    exact_evaluations: usize,
    lb_pruned: usize,
    cutoff_abandoned: usize,
    elapsed_ms: f64,
    outcomes: Vec<(bool, bool)>,
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

fn impute_missing_linear(series: &mut [f64]) {
    let known: Vec<(usize, f64)> = series
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect();
    if known.is_empty() {
        series.fill(0.0);
        return;
    }

    let (first_idx, first_value) = known[0];
    for value in &mut series[..first_idx] {
        *value = first_value;
    }

    for window in known.windows(2) {
        let (left_idx, left_value) = window[0];
        let (right_idx, right_value) = window[1];
        series[left_idx] = left_value;
        for idx in left_idx + 1..right_idx {
            let ratio = (idx - left_idx) as f64 / (right_idx - left_idx) as f64;
            series[idx] = left_value + (right_value - left_value) * ratio;
        }
    }

    let (last_idx, last_value) = *known.last().unwrap();
    series[last_idx] = last_value;
    for value in &mut series[last_idx + 1..] {
        *value = last_value;
    }
}

fn parse_series_value(path: &Path, line_no: usize, value: &str) -> Result<f64, String> {
    let trimmed = value.trim();
    if trimmed == "?" || trimmed.eq_ignore_ascii_case("nan") {
        return Ok(f64::NAN);
    }
    trimmed.parse::<f64>().map_err(|err| {
        format!(
            "{path:?}:{} invalid numeric value {value:?}: {err}",
            line_no
        )
    })
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
        let mut series = values
            .split(',')
            .map(|value| parse_series_value(path, line_no + 1, value))
            .collect::<Result<Vec<_>, _>>()?;
        impute_missing_linear(&mut series);
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
        let mut series = fields
            .map(|value| parse_series_value(path, line_no + 1, value))
            .collect::<Result<Vec<_>, _>>()?;
        impute_missing_linear(&mut series);
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

fn estimate_dataset_candidate(
    dataset_dir: &Path,
    dataset_name: &str,
) -> Result<DatasetCandidate, String> {
    let (train, test) = load_ucr_dataset(dataset_dir, dataset_name)?;
    let series_len = train
        .first()
        .or_else(|| test.first())
        .map(|row| row.series.len())
        .unwrap_or(0);
    let estimated_cells =
        train.len() as u128 * test.len() as u128 * series_len as u128 * series_len as u128;
    Ok(DatasetCandidate {
        name: dataset_name.to_string(),
        dir: dataset_dir.to_path_buf(),
        train_count: train.len(),
        test_count: test.len(),
        series_len,
        estimated_cells,
    })
}

fn discover_ucr_archive(root: &Path) -> Result<Vec<DatasetCandidate>, String> {
    let mut candidates = Vec::new();
    for entry in fs::read_dir(root).map_err(|err| format!("failed to read {root:?}: {err}"))? {
        let entry = entry.map_err(|err| format!("failed to read entry in {root:?}: {err}"))?;
        if !entry
            .file_type()
            .map_err(|err| format!("failed to read file type for {:?}: {err}", entry.path()))?
            .is_dir()
        {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let candidate = estimate_dataset_candidate(&entry.path(), &name)?;
        candidates.push(candidate);
    }
    candidates.sort_by(|left, right| {
        left.estimated_cells
            .cmp(&right.estimated_cells)
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(candidates)
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

fn predict_1nn_cutoff<'a>(
    train: &'a [LabeledSeries],
    probe: &[f64],
    msm: MsmConfig,
    exact_evaluations: &mut usize,
    lb_pruned: &mut usize,
    cutoff_abandoned: &mut usize,
) -> &'a str {
    let mut best_label = "";
    let mut best_distance = f64::INFINITY;

    for candidate in train {
        if length_lb(probe, &candidate.series, msm.c) > best_distance {
            *lb_pruned += 1;
            continue;
        }
        *exact_evaluations += 1;
        match msm.distance_with_cutoff(probe, &candidate.series, best_distance) {
            Some(distance) => {
                if distance < best_distance {
                    best_distance = distance;
                    best_label = candidate.label.as_str();
                }
            }
            None => {
                *cutoff_abandoned += 1;
            }
        }
    }

    best_label
}

fn evaluate_ucr_dataset(
    candidate: &DatasetCandidate,
    msm: MsmConfig,
) -> Result<DatasetEvaluation, String> {
    let (train, test) = load_ucr_dataset(&candidate.dir, &candidate.name)?;
    let majority = majority_label(&train).to_string();
    let started = Instant::now();
    let mut majority_correct = 0usize;
    let mut msm_correct = 0usize;
    let mut exact_evaluations = 0usize;
    let mut lb_pruned = 0usize;
    let mut cutoff_abandoned = 0usize;
    let mut outcomes = Vec::with_capacity(test.len());

    for probe in &test {
        let majority_hit = majority == probe.label;
        if majority_hit {
            majority_correct += 1;
        }
        let predicted = predict_1nn_cutoff(
            &train,
            &probe.series,
            msm,
            &mut exact_evaluations,
            &mut lb_pruned,
            &mut cutoff_abandoned,
        );
        let msm_hit = predicted == probe.label;
        if msm_hit {
            msm_correct += 1;
        }
        outcomes.push((majority_hit, msm_hit));
    }

    Ok(DatasetEvaluation {
        candidate: candidate.clone(),
        majority_correct,
        msm_correct,
        total: test.len(),
        exact_evaluations,
        lb_pruned,
        cutoff_abandoned,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        outcomes,
    })
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

#[cfg(test)]
fn ucr_1nn_outcomes_cutoff(
    train: &[LabeledSeries],
    test: &[LabeledSeries],
    msm: MsmConfig,
) -> Vec<(bool, bool)> {
    let majority = majority_label(train).to_string();
    let mut exact_evaluations = 0usize;
    let mut lb_pruned = 0usize;
    let mut cutoff_abandoned = 0usize;
    test.iter()
        .map(|probe| {
            let predicted = predict_1nn_cutoff(
                train,
                &probe.series,
                msm,
                &mut exact_evaluations,
                &mut lb_pruned,
                &mut cutoff_abandoned,
            );
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
    if scenario == "ucr-archive-summary" {
        let archive_root = args
            .next()
            .unwrap_or_else(|| "target/msm-corpora/Univariate_ts".to_string());
        let max_dataset_cells = args
            .next()
            .and_then(|s| s.parse::<u128>().ok())
            .unwrap_or(u128::MAX);
        let max_datasets = args
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        println!("dataset,train_count,test_count,series_len,estimated_cells,selected");
        for candidate in discover_ucr_archive(Path::new(&archive_root))
            .unwrap_or_else(|err| panic!("failed to discover UCR archive: {err}"))
            .into_iter()
            .take(max_datasets)
        {
            println!(
                "{},{},{},{},{},{}",
                candidate.name,
                candidate.train_count,
                candidate.test_count,
                candidate.series_len,
                candidate.estimated_cells,
                u8::from(candidate.estimated_cells <= max_dataset_cells),
            );
        }
        return;
    }
    if scenario == "ucr-archive-1nn" {
        let archive_root = args
            .next()
            .unwrap_or_else(|| "target/msm-corpora/Univariate_ts".to_string());
        let max_dataset_cells = args
            .next()
            .and_then(|s| s.parse::<u128>().ok())
            .unwrap_or(1_000_000_000);
        let max_datasets = args
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let candidates = discover_ucr_archive(Path::new(&archive_root))
            .unwrap_or_else(|err| panic!("failed to discover UCR archive: {err}"));
        println!("record_type,dataset,field_a,field_b,field_c,field_d,field_e,field_f,field_g,field_h,field_i,field_j,field_k");
        for candidate in candidates
            .into_iter()
            .filter(|candidate| candidate.estimated_cells <= max_dataset_cells)
            .take(max_datasets)
        {
            let evaluation = evaluate_ucr_dataset(&candidate, MsmConfig::new(1.0))
                .unwrap_or_else(|err| panic!("failed to evaluate {}: {err}", candidate.name));
            let accuracy = if evaluation.total == 0 {
                0.0
            } else {
                evaluation.msm_correct as f64 / evaluation.total as f64
            };
            println!(
                "summary,{},{},{},{},{},{},{},{},{:.12},{},{},{},{}",
                evaluation.candidate.name,
                evaluation.candidate.train_count,
                evaluation.candidate.test_count,
                evaluation.candidate.series_len,
                evaluation.candidate.estimated_cells,
                evaluation.majority_correct,
                evaluation.msm_correct,
                evaluation.total,
                accuracy,
                evaluation.exact_evaluations,
                evaluation.lb_pruned,
                evaluation.cutoff_abandoned,
                evaluation.elapsed_ms
            );
            for (case, (majority_correct, msm_correct)) in evaluation.outcomes.iter().enumerate() {
                println!(
                    "case,{},majority,{},{},,,,,,,,,",
                    evaluation.candidate.name,
                    case,
                    u8::from(*majority_correct)
                );
                println!(
                    "case,{},exact_msm_1nn,{},{},,,,,,,,,",
                    evaluation.candidate.name,
                    case,
                    u8::from(*msm_correct)
                );
            }
        }
        return;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SCRATCH_COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct ScratchDir {
        path: PathBuf,
    }

    impl ScratchDir {
        fn new(name: &str) -> Self {
            let id = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = PathBuf::from("target")
                .join("test-scratch")
                .join("msm-experiment")
                .join(name)
                .join(format!("{}-{}", std::process::id(), id));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("failed to create scratch dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
            if let Some(parent) = self.path.parent() {
                let _ = fs::remove_dir(parent);
            }
        }
    }

    #[test]
    fn approximate_paa_case_preserves_recall_floor() {
        let (index, _database, query, exact) = build_approx_case();
        let approximate = index.search_knn(&query, exact.len());
        let recall = recall_at_k(&approximate, &exact);

        assert_eq!(approximate.len(), exact.len());
        assert!(
            recall >= 0.875,
            "deterministic approximate MSM recall floor regressed: {recall}"
        );
    }

    #[test]
    fn ucr_txt_loader_and_1nn_outcomes_are_deterministic() {
        let scratch = ScratchDir::new("txt");
        let dataset = "ToyPowerDemand";
        fs::write(
            scratch.path().join(format!("{dataset}_TRAIN.txt")),
            "A 0.0 0.0 0.0\nB 10.0 10.0 10.0\n",
        )
        .expect("write train split");
        fs::write(
            scratch.path().join(format!("{dataset}_TEST.txt")),
            "A 0.0 0.0 1.0\nB 10.0 9.0 10.0\n",
        )
        .expect("write test split");

        let (train, test) = load_ucr_dataset(scratch.path(), dataset).expect("load UCR txt");
        let (correct, total, accuracy) = ucr_1nn_accuracy(&train, &test, MsmConfig::new(1.0));
        assert_eq!((correct, total), (2, 2));
        assert!((accuracy - 1.0).abs() < 1e-12);

        let outcomes = ucr_1nn_outcomes(&train, &test, MsmConfig::new(1.0));
        assert_eq!(outcomes, vec![(true, true), (false, true)]);

        let cutoff_outcomes = ucr_1nn_outcomes_cutoff(&train, &test, MsmConfig::new(1.0));
        assert_eq!(cutoff_outcomes, outcomes);
    }

    #[test]
    fn ucr_ts_loader_accepts_uea_style_data_section() {
        let scratch = ScratchDir::new("ts");
        let dataset = "ToyUeaArchive";
        fs::write(
            scratch.path().join(format!("{dataset}_TRAIN.ts")),
            "@problemName ToyUeaArchive\n@classLabel true A B\n@data\n0.0,0.0,0.0:A\n10.0,10.0,10.0:B\n",
        )
        .expect("write train split");
        fs::write(
            scratch.path().join(format!("{dataset}_TEST.ts")),
            "@problemName ToyUeaArchive\n@classLabel true A B\n@data\n0.0,?,1.0:A\n10.0,9.0,10.0:B\n",
        )
        .expect("write test split");

        let (train, test) = load_ucr_dataset(scratch.path(), dataset).expect("load UCR ts");
        assert_eq!(train.len(), 2);
        assert_eq!(test.len(), 2);
        assert_eq!(train[0].label, "A");
        assert_eq!(train[1].label, "B");
        assert_eq!(test[0].series, vec![0.0, 0.5, 1.0]);

        let (correct, total, accuracy) = ucr_1nn_accuracy(&train, &test, MsmConfig::new(1.0));
        assert_eq!((correct, total), (2, 2));
        assert!((accuracy - 1.0).abs() < 1e-12);

        let candidate =
            estimate_dataset_candidate(scratch.path(), dataset).expect("estimate dataset");
        assert_eq!(candidate.train_count, 2);
        assert_eq!(candidate.test_count, 2);
        assert_eq!(candidate.series_len, 3);
        assert_eq!(candidate.estimated_cells, 36);

        let evaluation = evaluate_ucr_dataset(&candidate, MsmConfig::new(1.0)).expect("evaluate");
        assert_eq!(evaluation.majority_correct, 1);
        assert_eq!(evaluation.msm_correct, 2);
        assert_eq!(evaluation.total, 2);
        assert_eq!(evaluation.outcomes, vec![(true, true), (false, true)]);
    }
}
