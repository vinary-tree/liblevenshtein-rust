use std::fs;

use liblevenshtein::time_series::elastic::ElasticTransducer;
use liblevenshtein::time_series::{
    ElasticSnapshotError, ElasticSnapshotMetadata, MsmConfig, MsmKernel, QuantizationConfig,
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

#[test]
fn complete_snapshot_is_deterministic_and_exact_across_collisions() {
    let directory = tempfile::Builder::new()
        .prefix(".elastic-snapshot-test-")
        .tempdir_in(".")
        .expect("create small snapshot test directory beside the worktree");
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
    let directory = tempfile::Builder::new()
        .prefix(".elastic-snapshot-rejection-test-")
        .tempdir_in(".")
        .expect("create small snapshot test directory beside the worktree");
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
fn complete_snapshot_byte_ceiling_is_hard_and_precedes_publication() {
    let directory = tempfile::Builder::new()
        .prefix(".elastic-snapshot-limit-test-")
        .tempdir_in(".")
        .expect("create small snapshot test directory beside the worktree");
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
