//! Integration tests for REPL functionality

#[cfg(feature = "cli")]
mod repl_integration_tests {
    use liblevenshtein::repl::state::DictionaryBackend;
    use liblevenshtein::repl::{Command, ReplState};
    use liblevenshtein::transducer::Algorithm;

    #[test]
    fn test_parse_query_command() {
        let cmd = Command::parse("query test 2").unwrap();
        match cmd {
            Command::Query {
                term,
                distance,
                prefix,
                limit,
            } => {
                assert_eq!(term, "test");
                assert_eq!(distance, Some(2));
                assert!(!prefix);
                assert_eq!(limit, None);
            }
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_parse_query_with_flags() {
        let cmd = Command::parse("query test 2 --prefix --limit 10").unwrap();
        match cmd {
            Command::Query {
                term,
                distance,
                prefix,
                limit,
            } => {
                assert_eq!(term, "test");
                assert_eq!(distance, Some(2));
                assert!(prefix);
                assert_eq!(limit, Some(10));
            }
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_parse_insert_command() {
        let cmd = Command::parse("insert hello world test").unwrap();
        match cmd {
            Command::Insert { terms } => {
                assert_eq!(terms, vec!["hello", "world", "test"]);
            }
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_parse_backend_command() {
        let cmd = Command::parse("backend dynamic-dawg").unwrap();
        match cmd {
            Command::Backend { backend } => {
                assert_eq!(backend, DictionaryBackend::DynamicDawg);
            }
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_parse_algorithm_command() {
        let cmd = Command::parse("algorithm transposition").unwrap();
        match cmd {
            Command::Algorithm { algorithm } => {
                assert_eq!(algorithm, Algorithm::Transposition);
            }
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_parse_algorithm_alias() {
        let cmd = Command::parse("algo merge-and-split").unwrap();
        match cmd {
            Command::Algorithm { algorithm } => {
                assert_eq!(algorithm, Algorithm::MergeAndSplit);
            }
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_parse_settings_aliases() {
        assert!(matches!(
            Command::parse("settings").unwrap(),
            Command::Settings
        ));
        assert!(matches!(
            Command::parse("options").unwrap(),
            Command::Settings
        ));
        assert!(matches!(Command::parse("opts").unwrap(), Command::Settings));
    }

    #[test]
    fn test_parse_config_command() {
        // Config without arguments shows current config file
        assert!(matches!(
            Command::parse("config").unwrap(),
            Command::Config { path: None }
        ));
        // Config with path switches to new config file
        match Command::parse("config /tmp/test.json").unwrap() {
            Command::Config { path: Some(p) } => {
                assert_eq!(p.to_str().unwrap(), "/tmp/test.json");
            }
            _ => panic!("Expected Config command with path"),
        }
    }

    #[test]
    fn test_parse_help_command() {
        let cmd = Command::parse("help").unwrap();
        assert!(matches!(cmd, Command::Help { topic: None }));

        let cmd = Command::parse("help query").unwrap();
        match cmd {
            Command::Help { topic: Some(t) } => assert_eq!(t, "query"),
            _ => panic!("Wrong command type"),
        }
    }

    #[test]
    fn test_insert_and_query() {
        let mut state = ReplState::new();
        state.max_distance = 1;

        // Insert terms
        let cmd = Command::parse("insert test testing tester").unwrap();
        let result = cmd.execute(&mut state).unwrap();
        assert!(matches!(
            result,
            liblevenshtein::repl::CommandResult::Continue(_)
        ));

        // Verify insertion
        assert_eq!(state.dictionary.len(), 3);

        // Query
        let results = state.query("test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "test");
    }

    #[test]
    fn test_backend_migration() {
        let mut state = ReplState::new();

        // Insert terms into PathMap
        state.dictionary.insert("hello").unwrap();
        state.dictionary.insert("world").unwrap();
        assert_eq!(state.backend, DictionaryBackend::PathMap);

        // Migrate to DynamicDawg
        let cmd = Command::parse("backend dynamic-dawg").unwrap();
        cmd.execute(&mut state).unwrap();

        assert_eq!(state.backend, DictionaryBackend::DynamicDawg);
        assert_eq!(state.dictionary.len(), 2);
        assert!(state.dictionary.contains("hello"));
        assert!(state.dictionary.contains("world"));
    }

    #[test]
    fn test_algorithm_change() {
        let mut state = ReplState::new();
        assert_eq!(state.algorithm, Algorithm::Standard);

        let cmd = Command::parse("algorithm transposition").unwrap();
        cmd.execute(&mut state).unwrap();

        assert_eq!(state.algorithm, Algorithm::Transposition);
    }

    #[test]
    fn test_prefix_mode_toggle() {
        let mut state = ReplState::new();
        assert!(!state.prefix_mode);

        let cmd = Command::parse("prefix on").unwrap();
        cmd.execute(&mut state).unwrap();
        assert!(state.prefix_mode);

        let cmd = Command::parse("prefix off").unwrap();
        cmd.execute(&mut state).unwrap();
        assert!(!state.prefix_mode);
    }

    #[test]
    fn test_show_distances_toggle() {
        let mut state = ReplState::new();
        assert!(!state.show_distances);

        let cmd = Command::parse("show-distances on").unwrap();
        cmd.execute(&mut state).unwrap();
        assert!(state.show_distances);
    }

    #[test]
    fn test_max_distance_change() {
        let mut state = ReplState::new();
        assert_eq!(state.max_distance, 2);

        let cmd = Command::parse("distance 3").unwrap();
        cmd.execute(&mut state).unwrap();

        assert_eq!(state.max_distance, 3);
    }

    #[test]
    fn test_result_limit() {
        let mut state = ReplState::new();
        assert_eq!(state.result_limit, None);

        let cmd = Command::parse("limit 10").unwrap();
        cmd.execute(&mut state).unwrap();
        assert_eq!(state.result_limit, Some(10));

        let cmd = Command::parse("limit none").unwrap();
        cmd.execute(&mut state).unwrap();
        assert_eq!(state.result_limit, None);
    }

    #[test]
    fn test_contains_command() {
        let mut state = ReplState::new();
        state.dictionary.insert("hello").unwrap();

        assert!(state.dictionary.contains("hello"));
        assert!(!state.dictionary.contains("world"));
    }

    #[test]
    fn test_delete_command() {
        let mut state = ReplState::new();
        state.dictionary.insert("hello").unwrap();
        state.dictionary.insert("world").unwrap();

        assert_eq!(state.dictionary.len(), 2);

        let cmd = Command::parse("delete hello").unwrap();
        cmd.execute(&mut state).unwrap();

        assert_eq!(state.dictionary.len(), 1);
        assert!(!state.dictionary.contains("hello"));
        assert!(state.dictionary.contains("world"));
    }
}
