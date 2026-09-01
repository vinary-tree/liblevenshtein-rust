//! Controlled Criterion cells for the exact temporal-product campaign.
//!
//! The suite registers bounded experiments; it does not turn an exploratory
//! local run into release evidence. Corpus construction, dictionary insertion,
//! certificate preparation for replay, online-machine construction, and
//! snapshot publication are outside their respective timed regions.

use std::hint::black_box;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use liblevenshtein::time_series::elastic::ElasticTransducer;
use liblevenshtein::time_series::{
    ElasticCertificateLimits, ErpConfig, GroundMetric, L1GroundMetric, L2GroundMetric,
    LinfGroundMetric, MetricTimestampedTwedConfig, OnlineAutomatonLimits, OperationOutcome,
    PageBudget, QuantizationConfig, ResourceLimits, TimestampUnit, TimestampedSeries,
    TimestampedTwedIndex, TimestampedTwedProductLimits, TimestampedTwedQuantizer,
    TimestampedTwedRangeOutcome, VectorFrechetOnlineAutomaton, VectorFrechetPath, VectorSample,
};

#[cfg(feature = "persistent-artrie")]
use liblevenshtein::time_series::ElasticSnapshotMetadata;
#[cfg(feature = "persistent-artrie")]
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const FULL_PAGE: PageBudget = PageBudget {
    max_work_units: usize::MAX,
    max_results: usize::MAX,
};

#[derive(Clone, Copy)]
enum Selectivity {
    Narrow,
    Broad,
}

impl Selectivity {
    fn name(self) -> &'static str {
        match self {
            Self::Narrow => "narrow",
            Self::Broad => "broad",
        }
    }

    fn cutoff(self) -> f64 {
        match self {
            Self::Narrow => 0.0,
            Self::Broad => 1.0e12,
        }
    }
}

#[derive(Clone, Copy)]
struct TimestampRangeCell {
    depth: usize,
    fanout: usize,
    query_len: usize,
    selectivity: Selectivity,
    cache_entries: usize,
    collision_members: usize,
}

impl TimestampRangeCell {
    fn label(self) -> String {
        format!(
            "depth_{}/fanout_{}/query_{}/cutoff_{}/cache_{}/bucket_{}",
            self.depth,
            self.fanout,
            self.query_len,
            self.selectivity.name(),
            if self.cache_entries == 0 { "off" } else { "on" },
            self.collision_members,
        )
    }
}

struct TimestampRangeFixture {
    index: TimestampedTwedIndex<u64>,
    query: TimestampedSeries,
    cutoff: f64,
    limits: TimestampedTwedProductLimits,
}

fn timestamp_quantizer() -> TimestampedTwedQuantizer {
    TimestampedTwedQuantizer::try_new(
        TimestampUnit::Seconds,
        0.0,
        (-128.0, 128.0),
        (0.0, 2_048.0),
        128,
        2_048,
    )
    .expect("benchmark quantizer is valid")
}

fn bin_center(bin: usize) -> f64 {
    -127.0 + 2.0 * f64::from(u16::try_from(bin % 128).expect("bin is below 128"))
}

fn timestamp_series(
    len: usize,
    branch: usize,
    suffix: usize,
    collision: usize,
) -> TimestampedSeries {
    let perturbation = f64::from(
        u16::try_from(collision).expect("benchmark collision bucket is deliberately small"),
    ) * 0.01;
    let values: Vec<_> = (0..len)
        .map(|position| {
            let bin = match position {
                0 => branch,
                1 => suffix,
                _ => position.wrapping_mul(13).wrapping_add(suffix * 7),
            };
            bin_center(bin) + perturbation
        })
        .collect();
    let timestamps: Vec<_> = (0..len)
        .map(|position| {
            f64::from(u32::try_from(position + 1).expect("benchmark depth fits u32")) + 0.25
        })
        .collect();
    TimestampedSeries::try_new_with_origin(
        &values,
        &timestamps,
        TimestampUnit::Seconds,
        0.0,
        ResourceLimits::default(),
    )
    .expect("deterministic timestamp benchmark series is valid")
}

fn timestamp_fixture(cell: TimestampRangeCell) -> TimestampRangeFixture {
    const SUFFIXES_PER_BRANCH: usize = 8;
    let config = MetricTimestampedTwedConfig::try_new(0.5, 1.0)
        .expect("benchmark TWED parameters are metric");
    let mut index = TimestampedTwedIndex::new(timestamp_quantizer(), config);
    let mut stable_id = 0_u64;
    for branch in 0..cell.fanout {
        for suffix in 0..SUFFIXES_PER_BRANCH {
            for collision in 0..cell.collision_members {
                index
                    .insert(
                        stable_id,
                        timestamp_series(cell.depth, branch, suffix, collision),
                    )
                    .expect("benchmark episode is inside the typed quantizer");
                stable_id += 1;
            }
        }
    }
    let limits = TimestampedTwedProductLimits {
        max_transition_cache_entries: cell.cache_entries,
        ..TimestampedTwedProductLimits::default()
    };
    TimestampRangeFixture {
        index,
        query: timestamp_series(cell.query_len, 0, 0, 0),
        cutoff: cell.selectivity.cutoff(),
        limits,
    }
}

fn drain_timestamped_range<V>(mut outcome: TimestampedTwedRangeOutcome<'_, V>) -> usize {
    loop {
        match outcome {
            OperationOutcome::Complete { value, .. } => return value.len(),
            OperationOutcome::Incomplete {
                continuation: Some(continuation),
                ..
            } => outcome = continuation.resume(FULL_PAGE),
            OperationOutcome::Incomplete { reason, .. } => {
                panic!("benchmark range query failed closed: {reason:?}")
            }
        }
    }
}

fn timestamped_twed_lazy_range_product(c: &mut Criterion) {
    let baseline = TimestampRangeCell {
        depth: 64,
        fanout: 8,
        query_len: 64,
        selectivity: Selectivity::Narrow,
        cache_entries: 1_000_000,
        collision_members: 1,
    };
    let cells = [
        baseline,
        TimestampRangeCell {
            depth: 256,
            ..baseline
        },
        TimestampRangeCell {
            fanout: 32,
            ..baseline
        },
        TimestampRangeCell {
            query_len: 256,
            ..baseline
        },
        TimestampRangeCell {
            selectivity: Selectivity::Broad,
            ..baseline
        },
        TimestampRangeCell {
            selectivity: Selectivity::Broad,
            cache_entries: 0,
            ..baseline
        },
        TimestampRangeCell {
            collision_members: 8,
            ..baseline
        },
    ];

    let mut group = c.benchmark_group("timestamped_twed_lazy_range_product");
    group.sample_size(10);
    for cell in cells {
        let fixture = timestamp_fixture(cell);
        group.throughput(Throughput::Elements(
            u64::try_from(fixture.index.len()).expect("benchmark index length fits u64"),
        ));
        group.bench_function(BenchmarkId::from_parameter(cell.label()), |b| {
            b.iter(|| {
                let outcome = fixture
                    .index
                    .search_range_bounded(
                        black_box(&fixture.query),
                        black_box(fixture.cutoff),
                        fixture.limits,
                        FULL_PAGE,
                    )
                    .expect("benchmark request is valid");
                black_box(drain_timestamped_range(outcome))
            });
        });
    }
    group.finish();
}

fn quantized_bin_center(bin: usize) -> f64 {
    -63.0 + 2.0 * f64::from(u16::try_from(bin % 64).expect("bin is below 64"))
}

fn elastic_series(len: usize, path: usize, collision: usize) -> Vec<f64> {
    let perturbation = f64::from(
        u16::try_from(collision).expect("benchmark collision bucket is deliberately small"),
    ) * 0.01;
    (0..len)
        .map(|position| {
            let bin = match position {
                0 => path % 64,
                1 => path / 64,
                _ => path.wrapping_mul(17).wrapping_add(position * 11),
            };
            quantized_bin_center(bin) + perturbation
        })
        .collect()
}

fn elastic_fixture(
    query_len: usize,
    unique_paths: usize,
    collision_members: usize,
) -> (ElasticTransducer<ErpConfig, u64>, Vec<f64>) {
    let mut index = ElasticTransducer::new(
        QuantizationConfig::uniform(-64.0, 64.0, 64),
        ErpConfig::new(0.0),
    );
    let mut stable_id = 0_u64;
    for path in 0..unique_paths {
        for collision in 0..collision_members {
            assert!(index.insert(stable_id, &elastic_series(query_len, path, collision)));
            stable_id += 1;
        }
    }
    (index, elastic_series(query_len, 0, 0))
}

#[derive(Clone, Copy)]
struct KnnCell {
    query_len: usize,
    unique_paths: usize,
    collision_members: usize,
    k: usize,
}

impl KnnCell {
    fn label(self) -> String {
        format!(
            "query_{}/paths_{}/bucket_{}/k_{}",
            self.query_len, self.unique_paths, self.collision_members, self.k,
        )
    }
}

fn bounded_exact_knn(c: &mut Criterion) {
    let baseline = KnnCell {
        query_len: 32,
        unique_paths: 128,
        collision_members: 1,
        k: 8,
    };
    let cells = [
        baseline,
        KnnCell {
            query_len: 128,
            ..baseline
        },
        KnnCell { k: 32, ..baseline },
        KnnCell {
            collision_members: 8,
            ..baseline
        },
    ];
    let mut group = c.benchmark_group("bounded_exact_knn");
    group.sample_size(10);
    for cell in cells {
        let (index, query) =
            elastic_fixture(cell.query_len, cell.unique_paths, cell.collision_members);
        group.throughput(Throughput::Elements(
            u64::try_from(index.len()).expect("benchmark index length fits u64"),
        ));
        group.bench_function(BenchmarkId::from_parameter(cell.label()), |b| {
            b.iter(|| {
                black_box(
                    index
                        .search_knn_bounded(
                            black_box(&query),
                            black_box(cell.k),
                            ResourceLimits::default(),
                        )
                        .expect("benchmark query is valid"),
                )
            });
        });
    }
    group.finish();
}

#[derive(Clone, Copy)]
struct CertificateCell {
    query_len: usize,
    unique_paths: usize,
    collision_members: usize,
    selectivity: Selectivity,
}

impl CertificateCell {
    fn label(self) -> String {
        format!(
            "query_{}/paths_{}/bucket_{}/cutoff_{}",
            self.query_len,
            self.unique_paths,
            self.collision_members,
            self.selectivity.name(),
        )
    }
}

fn replayable_elastic_range_certificates(c: &mut Criterion) {
    let baseline = CertificateCell {
        query_len: 32,
        unique_paths: 64,
        collision_members: 1,
        selectivity: Selectivity::Narrow,
    };
    let cells = [
        baseline,
        CertificateCell {
            query_len: 128,
            ..baseline
        },
        CertificateCell {
            selectivity: Selectivity::Broad,
            ..baseline
        },
        CertificateCell {
            collision_members: 8,
            ..baseline
        },
    ];
    let mut group = c.benchmark_group("replayable_elastic_range_certificates");
    group.sample_size(10);
    for cell in cells {
        let (index, query) =
            elastic_fixture(cell.query_len, cell.unique_paths, cell.collision_members);
        let cutoff = cell.selectivity.cutoff();
        let limits = ElasticCertificateLimits::default();
        let (_, certificate) = index
            .search_range_with_certificate(&query, cutoff, limits)
            .expect("benchmark certificate fits default limits");
        assert!(index
            .verify_range_certificate(&query, cutoff, &certificate, limits)
            .expect("prepared benchmark certificate replays"));
        let label = cell.label();

        group.bench_function(BenchmarkId::new("construct", &label), |b| {
            b.iter(|| {
                black_box(
                    index
                        .search_range_with_certificate(black_box(&query), black_box(cutoff), limits)
                        .expect("benchmark certificate construction completes"),
                )
            });
        });
        group.bench_function(BenchmarkId::new("replay", &label), |b| {
            b.iter(|| {
                black_box(
                    index
                        .verify_range_certificate(
                            black_box(&query),
                            black_box(cutoff),
                            black_box(&certificate),
                            limits,
                        )
                        .expect("benchmark certificate replay completes"),
                )
            });
        });
    }
    group.finish();
}

#[derive(Clone, Copy)]
struct VectorOnlineCell {
    query_len: usize,
    target_len: usize,
    dimension: usize,
    selectivity: Selectivity,
}

impl VectorOnlineCell {
    fn label(self, metric: &str) -> String {
        format!(
            "metric_{metric}/query_{}/target_{}/dimension_{}/cutoff_{}",
            self.query_len,
            self.target_len,
            self.dimension,
            self.selectivity.name(),
        )
    }
}

fn vector_sample(row: usize, dimension: usize, phase: usize) -> VectorSample {
    let coordinates: Vec<_> = (0..dimension)
        .map(|coordinate| {
            f64::from(
                u16::try_from((row * 7 + coordinate * 11 + phase) % 97)
                    .expect("coordinate phase is below 97"),
            ) - 48.0
        })
        .collect();
    VectorSample::try_new(&coordinates, ResourceLimits::default())
        .expect("benchmark vector sample is finite and nonempty")
}

fn vector_path(len: usize, dimension: usize, phase: usize) -> VectorFrechetPath {
    VectorFrechetPath::try_new(
        (0..len)
            .map(|row| vector_sample(row, dimension, phase))
            .collect(),
        ResourceLimits::default(),
    )
    .expect("benchmark path has a fixed dimension and no invalid points")
}

fn register_vector_online_metric<M: GroundMetric>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    metric_name: &str,
    ground: M,
    cell: VectorOnlineCell,
) {
    let query = vector_path(cell.query_len, cell.dimension, 0);
    let target: Vec<_> = (0..cell.target_len)
        .map(|row| vector_sample(row, cell.dimension, 3))
        .collect();
    let cutoff = match cell.selectivity {
        Selectivity::Narrow => 1.0,
        Selectivity::Broad => 1.0e12,
    };
    group.throughput(Throughput::Elements(
        u64::try_from(cell.target_len * cell.dimension)
            .expect("benchmark vector coordinate count fits u64"),
    ));
    group.bench_function(BenchmarkId::from_parameter(cell.label(metric_name)), |b| {
        b.iter_batched(
            || {
                VectorFrechetOnlineAutomaton::new(
                    query.clone(),
                    ground.clone(),
                    cutoff,
                    OnlineAutomatonLimits::default(),
                )
                .expect("benchmark online machine fits default limits")
            },
            |mut automaton| {
                for point in &target {
                    let _ = black_box(
                        automaton
                            .advance(black_box(point))
                            .expect("benchmark point has the fixed dimension"),
                    );
                }
                black_box(automaton.observation())
            },
            BatchSize::SmallInput,
        );
    });
}

fn vector_online_kernels(c: &mut Criterion) {
    let baseline = VectorOnlineCell {
        query_len: 64,
        target_len: 64,
        dimension: 4,
        selectivity: Selectivity::Broad,
    };
    let mut group = c.benchmark_group("vector_frechet_online_kernels");
    group.sample_size(10);

    register_vector_online_metric(&mut group, "l1", L1GroundMetric, baseline);
    register_vector_online_metric(&mut group, "l2", L2GroundMetric, baseline);
    register_vector_online_metric(&mut group, "linf", LinfGroundMetric, baseline);
    register_vector_online_metric(
        &mut group,
        "l1",
        L1GroundMetric,
        VectorOnlineCell {
            query_len: 256,
            target_len: 64,
            ..baseline
        },
    );
    register_vector_online_metric(
        &mut group,
        "l1",
        L1GroundMetric,
        VectorOnlineCell {
            dimension: 16,
            ..baseline
        },
    );
    register_vector_online_metric(
        &mut group,
        "l1",
        L1GroundMetric,
        VectorOnlineCell {
            selectivity: Selectivity::Narrow,
            ..baseline
        },
    );
    group.finish();
}

#[cfg(feature = "persistent-artrie")]
static SNAPSHOT_FIXTURE_NONCE: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "persistent-artrie")]
struct SnapshotFixture {
    root: PathBuf,
    manifest: PathBuf,
    quantizer: QuantizationConfig,
    kernel: ErpConfig,
    metadata: ElasticSnapshotMetadata,
    query: Vec<f64>,
}

#[cfg(feature = "persistent-artrie")]
impl Drop for SnapshotFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[cfg(feature = "persistent-artrie")]
fn snapshot_fixture_root() -> PathBuf {
    let nonce = SNAPSHOT_FIXTURE_NONCE.fetch_add(1, Ordering::Relaxed);
    std::env::current_dir()
        .expect("benchmark has a working directory")
        .join("target")
        .join("temporal-campaign-benchmark-state")
        .join(format!("process-{}-{nonce}", std::process::id()))
}

#[cfg(feature = "persistent-artrie")]
fn snapshot_fixture() -> SnapshotFixture {
    let root = snapshot_fixture_root();
    fs::create_dir_all(&root).expect("create disk-backed benchmark state directory");
    let manifest = root.join("erp.snapshot");
    let quantizer = QuantizationConfig::uniform(-64.0, 64.0, 64);
    let kernel = ErpConfig::new(0.0);
    let metadata = ElasticSnapshotMetadata::try_new(
        "benchmark-training-fold",
        "temporal-campaign-benchmark-v1",
        vec![1.0],
        vec![1.0],
    )
    .expect("benchmark snapshot metadata is valid");
    let (index, query) = elastic_fixture(64, 128, 4);
    index
        .write_complete_snapshot(&manifest, &metadata, 256 * 1024 * 1024)
        .expect("publish benchmark snapshot before timing");
    SnapshotFixture {
        root,
        manifest,
        quantizer,
        kernel,
        metadata,
        query,
    }
}

#[cfg(feature = "persistent-artrie")]
fn persistent_snapshot_load_and_query(c: &mut Criterion) {
    let fixture = snapshot_fixture();
    let mut load = c.benchmark_group("persistent_elastic_snapshot_load");
    load.sample_size(10);
    load.bench_function("entries_512/depth_64/bucket_4", |b| {
        b.iter_with_large_drop(|| {
            black_box(
                ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
                    black_box(fixture.manifest.as_path()),
                    &fixture.quantizer,
                    &fixture.kernel,
                    &fixture.metadata,
                    256 * 1024 * 1024,
                )
                .expect("benchmark snapshot verifies and loads"),
            )
        });
    });
    load.finish();

    let loaded = ElasticTransducer::<ErpConfig, u64>::load_complete_snapshot(
        &fixture.manifest,
        &fixture.quantizer,
        &fixture.kernel,
        &fixture.metadata,
        256 * 1024 * 1024,
    )
    .expect("prepare persistent query fixture outside timing");
    let mut query = c.benchmark_group("persistent_elastic_snapshot_bounded_query");
    query.sample_size(10);
    query.throughput(Throughput::Elements(512));
    query.bench_function("knn_8/query_64/bucket_4", |b| {
        b.iter(|| {
            let outcome = loaded
                .index
                .search_knn_bounded(
                    black_box(&fixture.query),
                    black_box(8),
                    ResourceLimits::default(),
                )
                .expect("persistent benchmark query is valid");
            match outcome {
                OperationOutcome::Complete { value, .. } => black_box(value.len()),
                OperationOutcome::Incomplete { reason, .. } => {
                    panic!("persistent benchmark query failed closed: {reason:?}")
                }
            }
        });
    });
    query.finish();
}

#[cfg(not(feature = "persistent-artrie"))]
fn persistent_snapshot_load_and_query(_: &mut Criterion) {}

criterion_group!(
    temporal_campaign_benches,
    timestamped_twed_lazy_range_product,
    bounded_exact_knn,
    replayable_elastic_range_certificates,
    vector_online_kernels,
    persistent_snapshot_load_and_query,
);
criterion_main!(temporal_campaign_benches);
