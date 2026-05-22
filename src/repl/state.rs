//! REPL state management
//!
//! Manages the dictionary backend, algorithm selection, and query parameters.

use crate::cli::args::SerializationFormat;
use crate::commands::core::QueryParams;
use crate::commands::handlers::query::execute_query;
use crate::transducer::Algorithm;
use anyhow::Result;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::suffix_automaton::SuffixAutomaton;
use libdictenstein::{Dictionary, DictionaryNode};
use std::path::Path;

// NOTE: For caching, use the eviction wrappers from cache::eviction module.
// Example: Lru<PathMapDictionary>, Ttl<DynamicDawg>, etc.
// See src/cache/eviction/mod.rs for available wrappers.

/// Helper to extract all terms from any dictionary using DFS
fn extract_terms<D>(dict: &D) -> Vec<String>
where
    D: Dictionary,
    D::Node: DictionaryNode<Unit = u8>,
{
    let est_size = dict.len().unwrap_or(100);
    let mut terms = Vec::with_capacity(est_size);
    let mut current_term = Vec::with_capacity(32);

    fn dfs<N: DictionaryNode<Unit = u8>>(
        node: &N,
        current_term: &mut Vec<u8>,
        terms: &mut Vec<String>,
    ) {
        if node.is_final() {
            if let Ok(term) = String::from_utf8(current_term.clone()) {
                terms.push(term);
            }
        }

        for (byte, child) in node.edges() {
            current_term.push(byte);
            dfs(&child, current_term, terms);
            current_term.pop();
        }
    }

    let root = dict.root();
    dfs(&root, &mut current_term, &mut terms);
    terms
}

/// Dictionary backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "cli",
    derive(clap::ValueEnum, serde::Serialize, serde::Deserialize)
)]
pub enum DictionaryBackend {
    /// PathMap-based trie (default, fast insertion/deletion)
    PathMap,
    /// Double-Array Trie (recommended default, fast and compact)
    DoubleArrayTrie,
    /// Dynamic DAWG (supports modifications, compressed)
    DynamicDawg,
    /// Suffix automaton (substring matching, dynamic)
    SuffixAutomaton,
}

impl std::fmt::Display for DictionaryBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathMap => write!(f, "path-map"),
            Self::DoubleArrayTrie => write!(f, "double-array-trie"),
            Self::DynamicDawg => write!(f, "dynamic-dawg"),
            Self::SuffixAutomaton => write!(f, "suffix-automaton"),
        }
    }
}

impl std::str::FromStr for DictionaryBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pathmap" | "path-map" => Ok(Self::PathMap),
            "double-array-trie" | "doublearraytrie" | "dat" => Ok(Self::DoubleArrayTrie),
            "dynamic-dawg" | "dynamicdawg" => Ok(Self::DynamicDawg),
            "suffix-automaton" | "suffixautomaton" => Ok(Self::SuffixAutomaton),
            _ => Err(anyhow::anyhow!(
                "Unknown backend: {}. Valid options: path-map, double-array-trie (dat), dynamic-dawg, suffix-automaton",
                s
            )),
        }
    }
}

/// Unified dictionary container
pub enum DictContainer {
    /// PathMap-based trie dictionary
    PathMap(PathMapDictionary),
    /// Double-Array Trie dictionary
    DoubleArrayTrie(DoubleArrayTrie),
    /// Dynamic DAWG dictionary
    DynamicDawg(DynamicDawg),
    /// Suffix automaton dictionary
    SuffixAutomaton(SuffixAutomaton),
}

impl DictContainer {
    /// Get the backend type
    pub fn backend(&self) -> DictionaryBackend {
        match self {
            Self::PathMap(_) => DictionaryBackend::PathMap,
            Self::DoubleArrayTrie(_) => DictionaryBackend::DoubleArrayTrie,
            Self::DynamicDawg(_) => DictionaryBackend::DynamicDawg,
            Self::SuffixAutomaton(_) => DictionaryBackend::SuffixAutomaton,
        }
    }

    /// Check if term exists
    pub fn contains(&self, term: &str) -> bool {
        match self {
            Self::PathMap(d) => d.contains(term),
            Self::DoubleArrayTrie(d) => d.contains(term),
            Self::DynamicDawg(d) => d.contains(term),
            Self::SuffixAutomaton(d) => d.contains(term),
        }
    }

    /// Insert a term (only for mutable backends)
    pub fn insert(&mut self, term: &str) -> Result<bool> {
        match self {
            Self::PathMap(d) => Ok(d.insert(term)),
            Self::DoubleArrayTrie(_) => Err(anyhow::anyhow!("DoubleArrayTrie dictionary is read-only. Use 'backend dynamic-dawg', 'backend pathmap', or 'backend suffix-automaton' for modifications.")),
            Self::DynamicDawg(d) => Ok(d.insert(term)),
            Self::SuffixAutomaton(d) => Ok(d.insert(term)),
        }
    }

    /// Remove a term (only for mutable backends)
    pub fn remove(&mut self, term: &str) -> Result<bool> {
        match self {
            Self::PathMap(d) => Ok(d.remove(term)),
            Self::DoubleArrayTrie(_) => Err(anyhow::anyhow!("DoubleArrayTrie dictionary is read-only. Use 'backend dynamic-dawg', 'backend pathmap', or 'backend suffix-automaton' for modifications.")),
            Self::DynamicDawg(d) => Ok(d.remove(term)),
            Self::SuffixAutomaton(d) => Ok(d.remove(term)),
        }
    }

    /// Get term count
    pub fn len(&self) -> usize {
        match self {
            Self::PathMap(d) => d.len().unwrap_or(0),
            Self::DoubleArrayTrie(d) => d.len().unwrap_or(0),
            Self::DynamicDawg(d) => d.len().unwrap_or(0),
            Self::SuffixAutomaton(d) => d.string_count(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collect all terms
    pub fn terms(&self) -> Vec<String> {
        match self {
            Self::PathMap(d) => extract_terms(d),
            Self::DoubleArrayTrie(d) => extract_terms(d),
            Self::DynamicDawg(d) => extract_terms(d),
            Self::SuffixAutomaton(d) => d.source_texts(),
        }
    }

    /// Migrate to a different backend
    pub fn migrate_to(&self, backend: DictionaryBackend) -> Result<Self> {
        let terms: Vec<String> = self.terms();

        let new_dict = match backend {
            DictionaryBackend::PathMap => {
                let dict = PathMapDictionary::from_terms(terms.iter().map(|s| s.as_str()));
                Self::PathMap(dict)
            }
            DictionaryBackend::DoubleArrayTrie => {
                let dict = DoubleArrayTrie::from_terms(terms);
                Self::DoubleArrayTrie(dict)
            }
            DictionaryBackend::DynamicDawg => {
                let dict = DynamicDawg::new();
                for term in &terms {
                    dict.insert(term);
                }
                Self::DynamicDawg(dict)
            }
            DictionaryBackend::SuffixAutomaton => {
                let dict = SuffixAutomaton::from_texts(terms);
                Self::SuffixAutomaton(dict)
            }
        };

        Ok(new_dict)
    }

    /// Clear all terms (only for mutable backends)
    pub fn clear(&mut self) -> Result<()> {
        match self {
            Self::PathMap(d) => {
                d.clear();
                Ok(())
            }
            Self::DoubleArrayTrie(_) => {
                Err(anyhow::anyhow!("DoubleArrayTrie dictionary is read-only"))
            }
            Self::DynamicDawg(_) => {
                // Replace with new empty DynamicDawg
                *self = Self::DynamicDawg(DynamicDawg::new());
                Ok(())
            }
            Self::SuffixAutomaton(d) => {
                d.clear();
                Ok(())
            }
        }
    }

    /// Compact/minimize (for dynamic backends)
    pub fn compact(&mut self) -> Result<()> {
        match self {
            Self::PathMap(_) => {
                // PathMap doesn't need compaction
                Ok(())
            }
            Self::DoubleArrayTrie(_) => Err(anyhow::anyhow!(
                "DoubleArrayTrie dictionary is already minimized"
            )),
            Self::DynamicDawg(d) => {
                d.minimize();
                Ok(())
            }
            Self::SuffixAutomaton(d) => {
                d.compact();
                Ok(())
            }
        }
    }
}

/// REPL state
pub struct ReplState {
    /// Dictionary container
    pub dictionary: DictContainer,
    /// Current backend type
    pub backend: DictionaryBackend,
    /// Serialization format for save/load operations
    pub serialization_format: SerializationFormat,
    /// Levenshtein algorithm
    pub algorithm: Algorithm,
    /// Maximum edit distance
    pub max_distance: usize,
    /// Whether to show distances in query results
    pub show_distances: bool,
    /// Whether to use prefix matching by default
    pub prefix_mode: bool,
    /// Result limit for queries
    pub result_limit: Option<usize>,
    /// Whether to auto-save after modifications
    pub auto_sync: bool,
    /// Path for auto-sync operations
    pub auto_sync_path: Option<std::path::PathBuf>,
    /// Custom config file path
    pub config_file_path: Option<std::path::PathBuf>,
}

impl ReplState {
    /// Create new REPL state with empty PathMap dictionary
    pub fn new() -> Self {
        // Default to Protobuf if available, otherwise Bincode
        #[cfg(feature = "protobuf")]
        let default_format = SerializationFormat::Protobuf;
        #[cfg(not(feature = "protobuf"))]
        let default_format = SerializationFormat::Bincode;

        Self {
            dictionary: DictContainer::PathMap(PathMapDictionary::new()),
            backend: DictionaryBackend::PathMap,
            serialization_format: default_format,
            algorithm: Algorithm::Standard,
            max_distance: 2,
            show_distances: false,
            prefix_mode: false,
            result_limit: None,
            auto_sync: false,
            auto_sync_path: None,
            config_file_path: None,
        }
    }

    /// Load dictionary from file
    pub fn load_from_file(&mut self, path: &Path, backend: DictionaryBackend) -> Result<usize> {
        #[cfg(feature = "cli")]
        {
            use crate::cli::commands::load_dictionary;
            use crate::cli::detect::detect_format;

            // Detect format
            let detection = detect_format(path, Some(backend), None)?;

            // Update serialization format based on detection
            self.serialization_format = detection.format.format;

            // Load dictionary using CLI function
            self.dictionary = load_dictionary(path, detection.format)?;
            self.backend = detection.format.backend;

            let count = self.dictionary.len();
            Ok(count)
        }

        #[cfg(not(feature = "cli"))]
        {
            // Fallback to plain text loading if CLI feature is not enabled
            let contents = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read dictionary file: {}", path.display()))?;

            let terms: Vec<&str> = contents
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .collect();

            if terms.is_empty() {
                return Err(anyhow::anyhow!("Dictionary file is empty"));
            }

            let term_count = terms.len();

            self.dictionary = match backend {
                DictionaryBackend::PathMap => {
                    DictContainer::PathMap(PathMapDictionary::from_terms(terms))
                }
                DictionaryBackend::DoubleArrayTrie => {
                    DictContainer::DoubleArrayTrie(DoubleArrayTrie::from_terms(terms))
                }
                DictionaryBackend::DynamicDawg => {
                    let dict = DynamicDawg::new();
                    for term in &terms {
                        dict.insert(term);
                    }
                    DictContainer::DynamicDawg(dict)
                }
                DictionaryBackend::SuffixAutomaton => {
                    let dict =
                        SuffixAutomaton::from_texts(terms.iter().map(|s| s.to_string()).collect());
                    DictContainer::SuffixAutomaton(dict)
                }
            };

            self.backend = backend;
            Ok(term_count)
        }
    }

    /// Save dictionary to file
    pub fn save_to_file(&self, path: &Path) -> Result<usize> {
        #[cfg(feature = "cli")]
        {
            use crate::cli::commands::save_dictionary;

            let count = self.dictionary.len();
            save_dictionary(&self.dictionary, path, self.serialization_format)?;
            Ok(count)
        }

        #[cfg(not(feature = "cli"))]
        {
            // Fallback to plain text saving if CLI feature is not enabled
            let terms = self.dictionary.terms();
            let content = terms.join("\n");

            std::fs::write(path, content)
                .with_context(|| format!("Failed to write dictionary file: {}", path.display()))?;

            Ok(terms.len())
        }
    }

    /// Change backend (migrates data)
    pub fn change_backend(&mut self, backend: DictionaryBackend) -> Result<()> {
        if self.backend == backend {
            return Ok(()); // Already using this backend
        }

        self.dictionary = self.dictionary.migrate_to(backend)?;
        self.backend = backend;
        Ok(())
    }

    /// Query the dictionary
    pub fn query(&self, term: &str) -> Vec<(String, usize)> {
        // NOTE: Cache functionality disabled - needs refactoring for new cache API
        #[cfg(all(feature = "pathmap-backend", not(feature = "pathmap-backend")))]
        if let (Some(cache), DictContainer::PathMap(_)) = (&self.cache, &self.dictionary) {
            // Use cache.query() which returns Vec<Candidate>
            let candidates = match cache {
                CacheContainer::Lru(c) => c.query(term, self.max_distance),
                CacheContainer::Lfu(c) => c.query(term, self.max_distance),
                CacheContainer::Ttl(c) => c.query(term, self.max_distance),
                CacheContainer::Age(c) => c.query(term, self.max_distance),
                CacheContainer::CostAware(c) => c.query(term, self.max_distance),
                CacheContainer::MemoryPressure(c) => c.query(term, self.max_distance),
                CacheContainer::Manual(c) => c.query(term, self.max_distance),
            };

            // Convert Candidate to (String, usize) and apply limit
            let mut results: Vec<(String, usize)> = candidates
                .into_iter()
                .map(|c| (c.term, c.distance))
                .collect();

            // Apply result limit if set
            if let Some(limit) = self.result_limit {
                results.truncate(limit);
            }

            return results;
        }

        // Fall back to direct dictionary query
        let params = QueryParams {
            term: term.to_string(),
            max_distance: self.max_distance,
            algorithm: self.algorithm,
            prefix: self.prefix_mode,
            show_distances: false, // Not used by execute_query
            limit: self.result_limit,
        };

        match &self.dictionary {
            DictContainer::PathMap(d) => execute_query(d, &params),
            DictContainer::DoubleArrayTrie(d) => execute_query(d, &params),
            DictContainer::DynamicDawg(d) => execute_query(d, &params),
            DictContainer::SuffixAutomaton(d) => execute_query(d, &params),
        }
    }

    /// Get dictionary statistics
    pub fn stats(&self) -> DictStats {
        DictStats {
            backend: self.backend,
            term_count: self.dictionary.len(),
            node_count: self.node_count(),
        }
    }

    fn node_count(&self) -> Option<usize> {
        match &self.dictionary {
            DictContainer::PathMap(_) => None,
            DictContainer::DoubleArrayTrie(_) => None,
            DictContainer::DynamicDawg(d) => Some(d.node_count()),
            DictContainer::SuffixAutomaton(d) => Some(d.state_count()),
        }
    }

    /// Enable fuzzy cache with specified strategy
    /// NOTE: Cache functionality disabled - needs refactoring for new cache API
    #[cfg(all(feature = "pathmap-backend", not(feature = "pathmap-backend")))]
    pub fn enable_cache(&mut self, strategy: &str, max_size: Option<usize>) -> Result<()> {
        use std::time::Duration;

        let default_max_size = 1000;
        let max_size = max_size.unwrap_or(default_max_size);

        let cache = match strategy.to_lowercase().as_str() {
            "lru" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .lru();
                CacheContainer::Lru(c)
            }
            "lfu" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .lfu();
                CacheContainer::Lfu(c)
            }
            "ttl" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .ttl(Duration::from_secs(300));
                CacheContainer::Ttl(c)
            }
            "age" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .age();
                CacheContainer::Age(c)
            }
            "cost-aware" | "cost" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .cost_aware();
                CacheContainer::CostAware(c)
            }
            "memory-pressure" | "memory" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .memory_pressure();
                CacheContainer::MemoryPressure(c)
            }
            "manual" | "fifo" => {
                let c = FuzzyCacheBuilder::<String>::new()
                    .max_size(max_size)
                    .algorithm(self.algorithm)
                    .manual();
                CacheContainer::Manual(c)
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown cache strategy: '{}'. Valid: lru, lfu, ttl, age, cost-aware, memory-pressure, manual",
                    strategy
                ));
            }
        };

        self.cache = Some(cache);
        Ok(())
    }

    /// Disable fuzzy cache
    /// NOTE: Cache functionality disabled - needs refactoring for new cache API
    #[cfg(all(feature = "pathmap-backend", not(feature = "pathmap-backend")))]
    pub fn disable_cache(&mut self) {
        self.cache = None;
    }

    /// Get cache statistics
    /// NOTE: Cache functionality disabled - needs refactoring for new cache API
    #[cfg(all(feature = "pathmap-backend", not(feature = "pathmap-backend")))]
    pub fn cache_stats(&self) -> String {
        if let Some(cache) = &self.cache {
            let (capacity, size, metrics_report) = match cache {
                CacheContainer::Lru(c) => (c.capacity(), c.len(), c.metrics().report()),
                CacheContainer::Lfu(c) => (c.capacity(), c.len(), c.metrics().report()),
                CacheContainer::Ttl(c) => (c.capacity(), c.len(), c.metrics().report()),
                CacheContainer::Age(c) => (c.capacity(), c.len(), c.metrics().report()),
                CacheContainer::CostAware(c) => (c.capacity(), c.len(), c.metrics().report()),
                CacheContainer::MemoryPressure(c) => (c.capacity(), c.len(), c.metrics().report()),
                CacheContainer::Manual(c) => (c.capacity(), c.len(), c.metrics().report()),
            };

            format!(
                "Cache Status: Enabled\nCapacity: {}\nCurrent Size: {}\n\n{}",
                capacity, size, metrics_report
            )
        } else {
            "Cache Status: Disabled".to_string()
        }
    }

    /// Clear cache
    /// NOTE: Cache functionality disabled - needs refactoring for new cache API
    #[cfg(all(feature = "pathmap-backend", not(feature = "pathmap-backend")))]
    pub fn clear_cache(&mut self) -> Result<()> {
        if let Some(cache) = &mut self.cache {
            match cache {
                CacheContainer::Lru(c) => c.clear(),
                CacheContainer::Lfu(c) => c.clear(),
                CacheContainer::Ttl(c) => c.clear(),
                CacheContainer::Age(c) => c.clear(),
                CacheContainer::CostAware(c) => c.clear(),
                CacheContainer::MemoryPressure(c) => c.clear(),
                CacheContainer::Manual(c) => c.clear(),
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Cache is not enabled"))
        }
    }

    /// Convert current state to PersistentConfig
    #[cfg(feature = "cli")]
    pub fn to_persistent_config(&self) -> crate::cli::paths::PersistentConfig {
        crate::cli::paths::PersistentConfig {
            dict_path: self.auto_sync_path.clone(),
            backend: Some(self.backend),
            format: Some(self.serialization_format),
            algorithm: Some(self.algorithm),
            max_distance: Some(self.max_distance),
            prefix_mode: Some(self.prefix_mode),
            show_distances: Some(self.show_distances),
            result_limit: Some(self.result_limit),
            auto_sync: Some(self.auto_sync),
        }
    }

    /// Save current state to configuration file
    #[cfg(feature = "cli")]
    pub fn save_config(&self) -> Result<()> {
        let config = self.to_persistent_config();
        config.save_to(self.config_file_path.clone())
    }
}

impl Default for ReplState {
    fn default() -> Self {
        Self::new()
    }
}

/// Dictionary statistics
#[derive(Debug)]
pub struct DictStats {
    /// Dictionary backend type
    pub backend: DictionaryBackend,
    /// Number of terms in the dictionary
    pub term_count: usize,
    /// Number of nodes (if available)
    pub node_count: Option<usize>,
}

impl std::fmt::Display for DictStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Backend:    {}", self.backend)?;
        writeln!(f, "Terms:      {}", self.term_count)?;
        if let Some(nodes) = self.node_count {
            writeln!(f, "Nodes:      {}", nodes)?;
            if self.term_count > 0 {
                let ratio = nodes as f64 / self.term_count as f64;
                writeln!(f, "Compression: {:.2}x (nodes/terms)", ratio)?;
            }
        }
        Ok(())
    }
}
