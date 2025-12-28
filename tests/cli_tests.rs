//! Integration tests for CLI functionality

#[cfg(feature = "cli")]
mod cli_integration_tests {
    use std::fs;
    use tempfile::TempDir;

    use liblevenshtein::cli::args::SerializationFormat;
    use liblevenshtein::cli::detect::{detect_format, DetectionMethod};
    use liblevenshtein::cli::paths::{file_extension, PersistentConfig};
    use liblevenshtein::repl::state::DictionaryBackend;

    #[test]
    fn test_detect_text_format() {
        let temp_dir = TempDir::new().unwrap();
        let dict_path = temp_dir.path().join("test.txt");
        fs::write(&dict_path, "hello\nworld\ntest\n").unwrap();

        let detection = detect_format(&dict_path, None, None).unwrap();
        assert_eq!(detection.format.format, SerializationFormat::Text);
        assert!(matches!(
            detection.method,
            DetectionMethod::Extension | DetectionMethod::Content
        ));
    }

    #[test]
    fn test_detect_format_by_extension() {
        let temp_dir = TempDir::new().unwrap();
        let dict_path = temp_dir.path().join("test.json");
        fs::write(&dict_path, "[]").unwrap();

        let detection = detect_format(&dict_path, None, None).unwrap();
        assert_eq!(detection.format.format, SerializationFormat::Json);
    }

    #[test]
    fn test_user_specified_format_override() {
        let temp_dir = TempDir::new().unwrap();
        let dict_path = temp_dir.path().join("test.txt");
        fs::write(&dict_path, "hello\nworld\n").unwrap();

        let detection = detect_format(
            &dict_path,
            Some(DictionaryBackend::DynamicDawg),
            Some(SerializationFormat::Text),
        )
        .unwrap();

        assert_eq!(detection.format.backend, DictionaryBackend::DynamicDawg);
        assert_eq!(detection.format.format, SerializationFormat::Text);
    }

    #[test]
    fn test_persistent_config_default() {
        let config = PersistentConfig::default();
        assert_eq!(
            config.algorithm,
            Some(liblevenshtein::transducer::Algorithm::Standard)
        );
        assert_eq!(config.max_distance, Some(2));
        assert_eq!(config.prefix_mode, Some(false));
        assert_eq!(config.show_distances, Some(false));
        assert_eq!(config.auto_sync, Some(false));
    }

    #[test]
    fn test_file_extensions() {
        assert_eq!(file_extension(SerializationFormat::Text), "txt");
        assert_eq!(file_extension(SerializationFormat::Bincode), "bin");
        assert_eq!(file_extension(SerializationFormat::Json), "json");
    }

    #[test]
    fn test_config_merge() {
        let base_config = PersistentConfig::default();
        let cli_overrides = PersistentConfig {
            dict_path: None,
            backend: Some(DictionaryBackend::DynamicDawg),
            format: None,
            algorithm: Some(liblevenshtein::transducer::Algorithm::Transposition),
            max_distance: Some(3),
            prefix_mode: Some(true),
            show_distances: None,
            result_limit: None,
            auto_sync: None,
        };
        let merged = base_config.merge_with_cli(&cli_overrides);

        assert_eq!(merged.backend, Some(DictionaryBackend::DynamicDawg));
        assert_eq!(
            merged.algorithm,
            Some(liblevenshtein::transducer::Algorithm::Transposition)
        );
        assert_eq!(merged.max_distance, Some(3));
        assert_eq!(merged.prefix_mode, Some(true));
        assert_eq!(merged.show_distances, Some(false)); // unchanged
    }
}
