//! Comparative persistence benchmark for the two OperationSet formats and
//! their optional gzip wrappers.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::{OperationSet, OperationType};
use std::hint::black_box;
use std::time::Duration;

fn corpus() -> Vec<(&'static str, OperationSet)> {
    let standard = OperationSet::standard();

    let mut repetitive = OperationSet::new();
    for index in 0..256 {
        repetitive.add(OperationType::new_owned(
            1,
            1,
            1.0,
            format!("repetitive_operation_family_{index:03}"),
        ));
    }
    vec![("standard", standard), ("repetitive_256", repetitive)]
}

fn bench_operation_set_persistence(criterion: &mut Criterion) {
    for (name, operations) in corpus() {
        let bincode = operations.to_binary().expect("benchmark bincode encodes");
        let protobuf = operations
            .to_protobuf()
            .expect("benchmark protobuf encodes");
        let bincode_gzip = operations
            .to_binary_gzip()
            .expect("benchmark bincode gzip encodes");
        let protobuf_gzip = operations
            .to_protobuf_gzip()
            .expect("benchmark protobuf gzip encodes");

        eprintln!(
            "OperationSet size/{name}: bincode={} protobuf={} bincode-gzip={} protobuf-gzip={}",
            bincode.len(),
            protobuf.len(),
            bincode_gzip.len(),
            protobuf_gzip.len()
        );

        let mut encode = criterion.benchmark_group(format!("operation_set_encode/{name}"));
        encode.throughput(Throughput::Elements(operations.len() as u64));
        encode.bench_function("bincode", |bencher| {
            bencher.iter(|| black_box(&operations).to_binary().expect("bincode encodes"));
        });
        encode.bench_function("protobuf", |bencher| {
            bencher.iter(|| {
                black_box(&operations)
                    .to_protobuf()
                    .expect("protobuf encodes")
            });
        });
        encode.bench_function("bincode-gzip", |bencher| {
            bencher.iter(|| {
                black_box(&operations)
                    .to_binary_gzip()
                    .expect("bincode gzip encodes")
            });
        });
        encode.bench_function("protobuf-gzip", |bencher| {
            bencher.iter(|| {
                black_box(&operations)
                    .to_protobuf_gzip()
                    .expect("protobuf gzip encodes")
            });
        });
        encode.finish();

        let mut decode = criterion.benchmark_group(format!("operation_set_decode/{name}"));
        for (format, bytes) in [
            ("bincode", &bincode),
            ("protobuf", &protobuf),
            ("bincode-gzip", &bincode_gzip),
            ("protobuf-gzip", &protobuf_gzip),
        ] {
            decode.throughput(Throughput::Bytes(bytes.len() as u64));
            decode.bench_with_input(
                BenchmarkId::from_parameter(format),
                bytes,
                |bencher, input| {
                    bencher.iter(|| {
                        match format {
                            "bincode" => OperationSet::from_binary(black_box(input))
                                .map_err(|error| error.to_string()),
                            "protobuf" => OperationSet::from_protobuf(black_box(input))
                                .map_err(|error| error.to_string()),
                            "bincode-gzip" => OperationSet::from_binary_gzip(black_box(input))
                                .map_err(|error| error.to_string()),
                            "protobuf-gzip" => OperationSet::from_protobuf_gzip(black_box(input))
                                .map_err(|error| error.to_string()),
                            _ => unreachable!("benchmark enumerates every format"),
                        }
                        .expect("benchmark payload decodes")
                    });
                },
            );
        }
        decode.finish();
    }
}

criterion_group! {
    name = operation_set_persistence;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(2));
    targets = bench_operation_set_persistence
}
criterion_main!(operation_set_persistence);
