#![cfg(feature = "persistent-artrie")]

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;

use liblevenshtein::time_series::elastic::ElasticTransducer;
use liblevenshtein::time_series::{
    ElasticSnapshotError, ElasticSnapshotLimits, ElasticSnapshotMetadata, MsmConfig, MsmKernel,
    QuantizationConfig,
};

const SNAPSHOT_LIMIT: u64 = 1 << 20;

fn metadata(fold: &str) -> ElasticSnapshotMetadata {
    ElasticSnapshotMetadata::try_new(
        fold,
        "commit=0123456789abcdef;lock=sha256:feedface;rust=1.95.0",
        vec![2.0, 4.0],
        vec![0.25, 0.75],
    )
    .expect("test metadata is valid")
}

fn scratch(prefix: &str) -> tempfile::TempDir {
    fs::create_dir_all("target/test-tmp").expect("create disk-backed test scratch root");
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in("target/test-tmp")
        .expect("create disk-backed snapshot scratch directory")
}

fn index(order: &[u64]) -> ElasticTransducer<MsmKernel, u64> {
    let quantizer = QuantizationConfig::for_u8(0.0, 100.0);
    let mut index = ElasticTransducer::new(quantizer, MsmConfig::new(1.0));
    for stable_id in order {
        let series = match stable_id {
            7 => &[10.01, 20.0][..],
            11 => &[10.02, 20.0][..],
            19 => &[80.0, 90.0, 95.0][..],
            _ => panic!("unknown fixture stable id"),
        };
        assert!(index.insert(*stable_id, series));
    }
    index
}

fn generation_path(path: &Path, identity: impl std::fmt::Display) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .expect("snapshot file name")
        .to_string_lossy();
    parent
        .join(format!(".{name}.elastic-generations"))
        .join(identity.to_string())
}

#[test]
fn complete_snapshot_is_deterministic_and_exact_across_collisions() {
    let directory = scratch("elastic-snapshot-test-");
    let first_path = directory.path().join("first.snapshot");
    let second_path = directory.path().join("second.snapshot");
    let snapshot_metadata = metadata("training-fold-3");
    let first = index(&[19, 7, 11]);
    let second = index(&[11, 19, 7]);

    let first_identity = first
        .write_complete_snapshot(&first_path, &snapshot_metadata, SNAPSHOT_LIMIT)
        .expect("write complete first snapshot");
    let second_identity = second
        .write_complete_snapshot(&second_path, &snapshot_metadata, SNAPSHOT_LIMIT)
        .expect("write complete second snapshot");
    assert_eq!(first_identity, second_identity);
    assert_eq!(
        fs::read(&first_path).expect("read first snapshot"),
        fs::read(&second_path).expect("read second snapshot")
    );

    let loaded = ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
        &first_path,
        first.quant_config(),
        first.kernel(),
        &snapshot_metadata,
        SNAPSHOT_LIMIT,
    )
    .expect("load matching complete snapshot");
    assert_eq!(loaded.identity, first_identity);
    assert_eq!(loaded.metadata, snapshot_metadata);
    assert_eq!(loaded.index.len(), 3);
    assert_eq!(loaded.index.get_original(&7), Some(&[10.01, 20.0][..]));
    assert_eq!(loaded.index.get_original(&11), Some(&[10.02, 20.0][..]));

    // IDs 7 and 11 share the same byte key, but exact verification must retain
    // both originals and admit only the actual zero-distance member.
    assert_eq!(
        loaded.index.search_range(&[10.01, 20.0], 0.0),
        vec![(7, 0.0)]
    );

    let mut expected_range = first.search_range(&[10.0, 20.0], 1.0);
    let mut loaded_range = loaded.index.search_range(&[10.0, 20.0], 1.0);
    expected_range.sort_by_key(|(stable_id, _)| *stable_id);
    loaded_range.sort_by_key(|(stable_id, _)| *stable_id);
    assert_eq!(loaded_range, expected_range);
    assert_eq!(
        loaded.index.search_knn(&[10.0, 20.0], 3, 1000.0),
        first.search_knn(&[10.0, 20.0], 3, 1000.0)
    );
}

#[test]
fn complete_snapshot_rejects_corruption_and_every_expected_binding_mismatch() {
    let directory = scratch("elastic-snapshot-rejection-test-");
    let path = directory.path().join("index.snapshot");
    let corrupt_path = directory.path().join("corrupt.snapshot");
    let index = index(&[7, 11, 19]);
    let snapshot_metadata = metadata("training-fold-3");
    index
        .write_complete_snapshot(&path, &snapshot_metadata, SNAPSHOT_LIMIT)
        .expect("write complete snapshot");

    let mut corrupt = fs::read(&path).expect("read snapshot to corrupt");
    let offset = corrupt.len() / 2;
    corrupt[offset] ^= 0x80;
    fs::write(&corrupt_path, corrupt).expect("write corrupt test snapshot");
    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &corrupt_path,
            index.quant_config(),
            index.kernel(),
            &snapshot_metadata,
            SNAPSHOT_LIMIT,
        ),
        Err(ElasticSnapshotError::ChecksumMismatch)
    ));

    // Corrupt the first attacker-controlled length field. Checksum verification
    // must precede UTF-8/length decoding, so this is still exactly a checksum
    // failure rather than a parser or allocator outcome.
    let mut early_corrupt = fs::read(&path).expect("read snapshot for early corruption");
    early_corrupt[12] ^= 0x80;
    let early_path = directory.path().join("early-corrupt.snapshot");
    fs::write(&early_path, early_corrupt).expect("write early corrupt snapshot");
    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &early_path,
            index.quant_config(),
            index.kernel(),
            &snapshot_metadata,
            SNAPSHOT_LIMIT,
        ),
        Err(ElasticSnapshotError::ChecksumMismatch)
    ));

    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &path,
            index.quant_config(),
            index.kernel(),
            &metadata("training-fold-4"),
            SNAPSHOT_LIMIT,
        ),
        Err(ElasticSnapshotError::ConfigurationMismatch)
    ));
    let different_kernel = MsmKernel::new(MsmConfig::new(2.0));
    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &path,
            index.quant_config(),
            &different_kernel,
            &snapshot_metadata,
            SNAPSHOT_LIMIT,
        ),
        Err(ElasticSnapshotError::ConfigurationMismatch)
    ));
    let different_quantizer = QuantizationConfig::for_u8(-1.0, 100.0);
    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &path,
            &different_quantizer,
            index.kernel(),
            &snapshot_metadata,
            SNAPSHOT_LIMIT,
        ),
        Err(ElasticSnapshotError::ConfigurationMismatch)
    ));
}

#[test]
fn snapshot_identity_binds_every_logical_configuration() {
    let directory = scratch("elastic-snapshot-identity-invalidation-");
    let base = index(&[7, 11, 19]);
    let base_metadata = metadata("training-fold-3");
    let base_path = directory.path().join("base.snapshot");
    let base_identity = base
        .write_complete_snapshot(&base_path, &base_metadata, SNAPSHOT_LIMIT)
        .expect("write base snapshot");

    let mut variants: Vec<(ElasticTransducer<MsmKernel, u64>, ElasticSnapshotMetadata)> =
        Vec::new();
    let mut changed_original = index(&[7, 11, 19]);
    assert!(!changed_original.insert(7, &[10.01_f64.next_up(), 20.0]));
    variants.push((changed_original, base_metadata.clone()));

    let mut changed_id =
        ElasticTransducer::new(QuantizationConfig::for_u8(0.0, 100.0), MsmConfig::new(1.0));
    assert!(changed_id.insert(8_u64, &[10.01, 20.0]));
    assert!(changed_id.insert(11, &[10.02, 20.0]));
    assert!(changed_id.insert(19, &[80.0, 90.0, 95.0]));
    variants.push((changed_id, base_metadata.clone()));

    let mut changed_kernel =
        ElasticTransducer::new(QuantizationConfig::for_u8(0.0, 100.0), MsmConfig::new(2.0));
    for id in [7, 11, 19] {
        let series = base.get_original(&id).expect("base original");
        assert!(changed_kernel.insert(id, series));
    }
    variants.push((changed_kernel, base_metadata.clone()));

    let mut changed_quantizer =
        ElasticTransducer::new(QuantizationConfig::for_u8(-1.0, 100.0), MsmConfig::new(1.0));
    for id in [7, 11, 19] {
        let series = base.get_original(&id).expect("base original");
        assert!(changed_quantizer.insert(id, series));
    }
    variants.push((changed_quantizer, base_metadata.clone()));

    variants.push((index(&[7, 11, 19]), metadata("training-fold-4")));
    variants.push((
        index(&[7, 11, 19]),
        ElasticSnapshotMetadata::try_new(
            "training-fold-3",
            "commit=different;lock=sha256:feedface;rust=1.95.0",
            vec![2.0, 4.0],
            vec![0.25, 0.75],
        )
        .expect("changed provenance metadata"),
    ));
    variants.push((
        index(&[7, 11, 19]),
        ElasticSnapshotMetadata::try_new(
            "training-fold-3",
            "commit=0123456789abcdef;lock=sha256:feedface;rust=1.95.0",
            vec![2.0, 5.0],
            vec![0.25, 0.75],
        )
        .expect("changed scale metadata"),
    ));
    variants.push((
        index(&[7, 11, 19]),
        ElasticSnapshotMetadata::try_new(
            "training-fold-3",
            "commit=0123456789abcdef;lock=sha256:feedface;rust=1.95.0",
            vec![2.0, 4.0],
            vec![0.5, 0.5],
        )
        .expect("changed weight metadata"),
    ));

    for (variant, (variant_index, variant_metadata)) in variants.iter().enumerate() {
        let identity = variant_index
            .write_complete_snapshot(
                directory.path().join(format!("variant-{variant}.snapshot")),
                variant_metadata,
                SNAPSHOT_LIMIT,
            )
            .expect("write identity variant");
        assert_ne!(
            identity, base_identity,
            "variant {variant} must invalidate identity"
        );
    }

    let loaded = ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
        &base_path,
        base.quant_config(),
        base.kernel(),
        &base_metadata,
        SNAPSHOT_LIMIT,
    )
    .expect("load base snapshot");
    assert_eq!(loaded.index.snapshot_identity(), Some(base_identity));
    assert_eq!(loaded.index.len(), base.len());
}

#[test]
fn physical_generation_corruption_and_missing_generation_fail_closed() {
    let directory = scratch("elastic-snapshot-generation-corruption-");
    let path = directory.path().join("index.snapshot");
    let source = index(&[7, 11, 19]);
    let snapshot_metadata = metadata("training-fold-3");
    let identity = source
        .write_complete_snapshot(&path, &snapshot_metadata, SNAPSHOT_LIMIT)
        .expect("write sealed generation");
    let generation = generation_path(&path, identity);
    let dictionary = generation.join("dictionary.part");
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dictionary)
        .expect("open dictionary component");
    let offset = file.metadata().expect("dictionary metadata").len() / 2;
    file.seek(SeekFrom::Start(offset))
        .expect("seek dictionary corruption offset");
    let mut original = [0_u8; 1];
    std::io::Read::read_exact(&mut file, &mut original).expect("read dictionary byte");
    file.seek(SeekFrom::Start(offset))
        .expect("rewind dictionary corruption offset");
    file.write_all(&[original[0] ^ 0xA5])
        .expect("corrupt dictionary byte");
    file.sync_all().expect("sync corruption");
    drop(file);
    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &path,
            source.quant_config(),
            source.kernel(),
            &snapshot_metadata,
            SNAPSHOT_LIMIT,
        ),
        Err(ElasticSnapshotError::ChecksumMismatch)
    ));

    fs::remove_dir_all(&generation).expect("remove test generation");
    assert!(ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
        &path,
        source.quant_config(),
        source.kernel(),
        &snapshot_metadata,
        SNAPSHOT_LIMIT,
    )
    .is_err());
}

#[test]
fn deep_key_snapshot_reopen_and_drop_are_stack_safe_and_bounded() {
    const DEPTH: usize = 100_000;
    let directory = scratch("elastic-snapshot-deep-stack-");
    let path = directory.path().join("deep.snapshot");
    let worker = thread::Builder::new()
        .name("elastic-snapshot-deep-stack".into())
        .stack_size(128 * 1024)
        .spawn(move || {
            let quantizer = QuantizationConfig::for_u8(0.0, 1.0);
            let mut source = ElasticTransducer::new(quantizer, MsmConfig::new(1.0));
            let first = vec![0.25; DEPTH];
            let second = vec![0.25_f64.next_up(); DEPTH];
            assert!(source.insert(1_u64, &first));
            assert!(source.insert(2_u64, &second));
            let metadata = metadata("deep-training-fold");
            let limits = ElasticSnapshotLimits {
                max_manifest_bytes: 4 * 1024 * 1024,
                max_bundle_bytes: 256 * 1024 * 1024,
                max_entries: 2,
                max_series_len: DEPTH,
                max_total_samples: 2 * DEPTH,
                max_backend_memory_bytes: 1024 * 1024,
            };
            let identity = source
                .write_complete_snapshot_with_limits(&path, &metadata, limits)
                .expect("write deep snapshot on constrained stack");
            let loaded = ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot_with_limits(
                &path,
                source.quant_config(),
                source.kernel(),
                &metadata,
                limits,
            )
            .expect("reopen deep disk-backed snapshot on constrained stack");
            assert_eq!(loaded.identity, identity);
            assert_eq!(loaded.index.get_original(&1).map(<[f64]>::len), Some(DEPTH));
            assert_eq!(loaded.index.get_original(&2).map(<[f64]>::len), Some(DEPTH));
            drop(loaded);
            drop(source);
        })
        .expect("spawn constrained-stack snapshot worker");
    worker.join().expect("constrained-stack snapshot worker");
}

#[test]
fn complete_snapshot_byte_ceiling_is_hard_and_precedes_publication() {
    let directory = scratch("elastic-snapshot-limit-test-");
    let path = directory.path().join("too-small.snapshot");
    let index = index(&[7, 11, 19]);
    let error = index
        .write_complete_snapshot(&path, &metadata("training-fold-3"), 16)
        .expect_err("sixteen bytes cannot hold a complete snapshot");
    assert!(matches!(error, ElasticSnapshotError::ResourceLimit { .. }));
    assert!(!path.exists());

    let valid_path = directory.path().join("valid.snapshot");
    index
        .write_complete_snapshot(&valid_path, &metadata("training-fold-3"), SNAPSHOT_LIMIT)
        .expect("write complete snapshot");
    assert!(matches!(
        ElasticTransducer::<MsmKernel, u64>::load_complete_snapshot(
            &valid_path,
            index.quant_config(),
            index.kernel(),
            &metadata("training-fold-3"),
            16,
        ),
        Err(ElasticSnapshotError::ResourceLimit { .. })
    ));
}
