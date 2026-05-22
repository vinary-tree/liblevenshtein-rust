//! Shared I/O handler for dictionary serialization/deserialization.
//!
//! This module provides unified I/O operations that can be used by both the CLI
//! and the REPL, avoiding code duplication.

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::cli::args::SerializationFormat;
use crate::cli::detect::{detect_format, DictFormat};
use crate::commands::core::{
    DeserializeParams, DeserializeResult, DictInfo, SerializeParams, SerializeResult,
};
use crate::repl::state::{DictContainer, DictionaryBackend};

#[cfg(feature = "serialization")]
use crate::serialization::{BincodeSerializer, DictionarySerializer, JsonSerializer};

// ============================================================================
// Serialization (Save) Operations
// ============================================================================

/// Execute a serialize (save) operation.
///
/// Saves the dictionary container to the specified path using the given format.
/// Returns information about the serialization result including term count and
/// file size.
pub fn execute_serialize(
    container: &DictContainer,
    params: &SerializeParams,
) -> Result<SerializeResult> {
    // Check if file exists and overwrite is not allowed
    if params.path.exists() && !params.overwrite {
        bail!(
            "File already exists: {}. Use overwrite=true to replace.",
            params.path.display()
        );
    }

    // Create parent directories if needed
    if let Some(parent) = params.path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
    }

    let term_count = container.len();

    // Perform serialization based on format
    save_dictionary_impl(container, &params.path, params.format)?;

    // Get file size
    let byte_size = std::fs::metadata(&params.path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(SerializeResult {
        term_count,
        byte_size,
        format: params.format,
    })
}

/// Internal implementation for saving dictionaries.
fn save_dictionary_impl(
    container: &DictContainer,
    path: &Path,
    format: SerializationFormat,
) -> Result<()> {
    match format {
        SerializationFormat::Text => save_text_dict(container, path),
        #[cfg(feature = "serialization")]
        SerializationFormat::Bincode => save_bincode_dict(container, path),
        #[cfg(feature = "serialization")]
        SerializationFormat::Json => save_json_dict(container, path),
        #[cfg(all(feature = "protobuf", feature = "serialization"))]
        SerializationFormat::Protobuf => {
            bail!("Protobuf format not yet implemented for saving")
        }
        #[cfg(feature = "compression")]
        SerializationFormat::BincodeGzip => save_bincode_gzip_dict(container, path),
        #[cfg(feature = "compression")]
        SerializationFormat::JsonGzip => save_json_gzip_dict(container, path),
        #[cfg(all(feature = "protobuf", feature = "compression"))]
        SerializationFormat::ProtobufGzip => {
            bail!("Protobuf-Gzip format not yet implemented for saving")
        }
        SerializationFormat::PathsNative => save_paths_native_dict(container, path),
        #[cfg(not(feature = "serialization"))]
        _ => bail!("Serialization feature not enabled"),
    }
}

/// Save as plain text (one term per line).
fn save_text_dict(container: &DictContainer, path: &Path) -> Result<()> {
    let terms = container.terms();
    let content = terms.join("\n");
    std::fs::write(path, content)
        .with_context(|| format!("Failed to write file: {}", path.display()))
}

#[cfg(feature = "serialization")]
fn save_bincode_dict(container: &DictContainer, path: &Path) -> Result<()> {
    let file = std::fs::File::create(path)?;
    match container {
        DictContainer::PathMap(d) => BincodeSerializer::serialize(d, file)?,
        DictContainer::DoubleArrayTrie(d) => BincodeSerializer::serialize(d, file)?,
        DictContainer::DynamicDawg(d) => BincodeSerializer::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => {
            BincodeSerializer::serialize_suffix_automaton(d, file)?
        }
    }
    Ok(())
}

#[cfg(feature = "serialization")]
fn save_json_dict(container: &DictContainer, path: &Path) -> Result<()> {
    let file = std::fs::File::create(path)?;
    match container {
        DictContainer::PathMap(d) => JsonSerializer::serialize(d, file)?,
        DictContainer::DoubleArrayTrie(d) => JsonSerializer::serialize(d, file)?,
        DictContainer::DynamicDawg(d) => JsonSerializer::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => JsonSerializer::serialize(d, file)?,
    }
    Ok(())
}

#[cfg(feature = "compression")]
fn save_bincode_gzip_dict(container: &DictContainer, path: &Path) -> Result<()> {
    use crate::serialization::GzipSerializer;

    let file = std::fs::File::create(path)?;
    match container {
        DictContainer::PathMap(d) => GzipSerializer::<BincodeSerializer>::serialize(d, file)?,
        DictContainer::DoubleArrayTrie(d) => {
            GzipSerializer::<BincodeSerializer>::serialize(d, file)?
        }
        DictContainer::DynamicDawg(d) => GzipSerializer::<BincodeSerializer>::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => {
            GzipSerializer::<BincodeSerializer>::serialize(d, file)?
        }
    }
    Ok(())
}

#[cfg(feature = "compression")]
fn save_json_gzip_dict(container: &DictContainer, path: &Path) -> Result<()> {
    use crate::serialization::GzipSerializer;

    let file = std::fs::File::create(path)?;
    match container {
        DictContainer::PathMap(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
        DictContainer::DoubleArrayTrie(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
        DictContainer::DynamicDawg(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
    }
    Ok(())
}

/// Save as PathMap's native .paths format.
fn save_paths_native_dict(container: &DictContainer, path: &Path) -> Result<()> {
    match container {
        DictContainer::PathMap(d) => {
            let mut file = std::fs::File::create(path)
                .with_context(|| format!("Failed to create file: {}", path.display()))?;
            d.serialize_paths(&mut file)
                .map_err(|e| anyhow::anyhow!("Failed to serialize paths: {}", e))
                .with_context(|| format!("Failed to serialize paths to: {}", path.display()))?;
            Ok(())
        }
        _ => bail!("PathsNative format only supports PathMap backend"),
    }
}

// ============================================================================
// Deserialization (Load) Operations
// ============================================================================

/// Execute a deserialize (load) operation.
///
/// Loads a dictionary from the specified path, optionally using hints for the
/// backend and format. Returns the loaded container along with metadata.
pub fn execute_deserialize(
    params: &DeserializeParams,
) -> Result<(DictContainer, DeserializeResult)> {
    if !params.path.exists() {
        bail!("Dictionary file does not exist: {}", params.path.display());
    }

    // Detect format (or use hints)
    let detection = detect_format(&params.path, params.backend, params.format)?;
    let dict_format = detection.format;

    // Load dictionary
    let container = load_dictionary_impl(&params.path, dict_format)?;
    let term_count = container.len();

    let result = DeserializeResult {
        term_count,
        backend: dict_format.backend,
        format: dict_format.format,
    };

    Ok((container, result))
}

/// Internal implementation for loading dictionaries.
fn load_dictionary_impl(path: &Path, format: DictFormat) -> Result<DictContainer> {
    match format.format {
        SerializationFormat::Text => load_text_dict(path, format.backend),
        #[cfg(feature = "serialization")]
        SerializationFormat::Bincode => load_bincode_dict(path, format.backend),
        #[cfg(feature = "serialization")]
        SerializationFormat::Json => load_json_dict(path, format.backend),
        #[cfg(all(feature = "protobuf", feature = "serialization"))]
        SerializationFormat::Protobuf => {
            bail!("Protobuf format not yet implemented for loading")
        }
        #[cfg(feature = "compression")]
        SerializationFormat::BincodeGzip => load_bincode_gzip_dict(path, format.backend),
        #[cfg(feature = "compression")]
        SerializationFormat::JsonGzip => load_json_gzip_dict(path, format.backend),
        #[cfg(all(feature = "protobuf", feature = "compression"))]
        SerializationFormat::ProtobufGzip => {
            bail!("Protobuf-Gzip format not yet implemented for loading")
        }
        SerializationFormat::PathsNative => load_paths_native_dict(path),
        #[cfg(not(feature = "serialization"))]
        _ => bail!("Serialization feature not enabled"),
    }
}

/// Load plain text dictionary.
fn load_text_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use std::io::BufRead;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    let terms: Vec<String> = reader
        .lines()
        .filter_map(|line| {
            line.ok().and_then(|l| {
                let trimmed = l.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
        })
        .collect();

    create_dict_from_terms(terms, backend)
}

#[cfg(feature = "serialization")]
fn load_bincode_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::suffix_automaton::SuffixAutomaton;

    let file = std::fs::File::open(path)?;
    let container = match backend {
        DictionaryBackend::PathMap => {
            let dict: PathMapDictionary = BincodeSerializer::deserialize(file)?;
            DictContainer::PathMap(dict)
        }
        DictionaryBackend::DoubleArrayTrie => {
            let dict: DoubleArrayTrie = BincodeSerializer::deserialize(file)?;
            DictContainer::DoubleArrayTrie(dict)
        }
        DictionaryBackend::DynamicDawg => {
            // DynamicDawg doesn't implement direct deserialization, so extract terms and rebuild
            let dict: PathMapDictionary = BincodeSerializer::deserialize(file)?;
            let terms = extract_terms_from_pathmap(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict: SuffixAutomaton = BincodeSerializer::deserialize_suffix_automaton(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

#[cfg(feature = "serialization")]
fn load_json_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::suffix_automaton::SuffixAutomaton;

    let file = std::fs::File::open(path)?;
    let container = match backend {
        DictionaryBackend::PathMap => {
            let dict: PathMapDictionary = JsonSerializer::deserialize(file)?;
            DictContainer::PathMap(dict)
        }
        DictionaryBackend::DoubleArrayTrie => {
            let dict: DoubleArrayTrie = JsonSerializer::deserialize(file)?;
            DictContainer::DoubleArrayTrie(dict)
        }
        DictionaryBackend::DynamicDawg => {
            // DynamicDawg doesn't implement direct deserialization, so extract terms and rebuild
            let dict: PathMapDictionary = JsonSerializer::deserialize(file)?;
            let terms = extract_terms_from_pathmap(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict: SuffixAutomaton = JsonSerializer::deserialize(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

#[cfg(feature = "compression")]
fn load_bincode_gzip_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use crate::serialization::GzipSerializer;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::suffix_automaton::SuffixAutomaton;

    let file = std::fs::File::open(path)?;
    let container = match backend {
        DictionaryBackend::PathMap => {
            let dict: PathMapDictionary = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            DictContainer::PathMap(dict)
        }
        DictionaryBackend::DoubleArrayTrie => {
            let dict: DoubleArrayTrie = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            DictContainer::DoubleArrayTrie(dict)
        }
        DictionaryBackend::DynamicDawg => {
            let dict: PathMapDictionary = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            let terms = extract_terms_from_pathmap(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict: SuffixAutomaton = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

#[cfg(feature = "compression")]
fn load_json_gzip_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use crate::serialization::GzipSerializer;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::suffix_automaton::SuffixAutomaton;

    let file = std::fs::File::open(path)?;
    let container = match backend {
        DictionaryBackend::PathMap => {
            let dict: PathMapDictionary = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            DictContainer::PathMap(dict)
        }
        DictionaryBackend::DoubleArrayTrie => {
            let dict: DoubleArrayTrie = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            DictContainer::DoubleArrayTrie(dict)
        }
        DictionaryBackend::DynamicDawg => {
            let dict: PathMapDictionary = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            let terms = extract_terms_from_pathmap(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict: SuffixAutomaton = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

/// Load PathMap's native .paths format.
fn load_paths_native_dict(path: &Path) -> Result<DictContainer> {
    use libdictenstein::pathmap::PathMapDictionary;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    let dict = PathMapDictionary::deserialize_paths(file)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize paths: {}", e))
        .with_context(|| format!("Failed to load PathMap from: {}", path.display()))?;
    Ok(DictContainer::PathMap(dict))
}

// ============================================================================
// Dictionary Information
// ============================================================================

/// Get information about a dictionary file without fully loading it.
///
/// This is useful for displaying file metadata and statistics.
pub fn get_dict_info(path: &Path) -> Result<DictInfo> {
    if !path.exists() {
        bail!("Dictionary file does not exist: {}", path.display());
    }

    // Get file size
    let file_size = std::fs::metadata(path)
        .with_context(|| format!("Failed to read file metadata: {}", path.display()))?
        .len();

    // Detect format
    let detection = detect_format(path, None, None)?;
    let dict_format = detection.format;

    // Load dictionary to get term count
    let container = load_dictionary_impl(path, dict_format)?;
    let term_count = container.len();

    Ok(DictInfo {
        path: path.to_path_buf(),
        term_count,
        backend: dict_format.backend,
        format: dict_format.format,
        file_size,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract terms from a PathMapDictionary using DFS traversal.
#[cfg(feature = "serialization")]
fn extract_terms_from_pathmap(dict: &libdictenstein::pathmap::PathMapDictionary) -> Vec<String> {
    use libdictenstein::{Dictionary, DictionaryNode};

    let mut terms = Vec::new();
    let mut current_term = Vec::new();

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

/// Create a dictionary container from a list of terms.
fn create_dict_from_terms(terms: Vec<String>, backend: DictionaryBackend) -> Result<DictContainer> {
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::suffix_automaton::SuffixAutomaton;

    let container = match backend {
        DictionaryBackend::PathMap => DictContainer::PathMap(PathMapDictionary::from_terms(
            terms.iter().map(|s| s.as_str()),
        )),
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
            DictContainer::SuffixAutomaton(SuffixAutomaton::from_texts(terms))
        }
    };

    Ok(container)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_and_load_text_dict() {
        use libdictenstein::pathmap::PathMapDictionary;

        // Create a dictionary
        let terms = vec!["apple", "banana", "cherry"];
        let dict = PathMapDictionary::from_terms(terms.iter().cloned());
        let container = DictContainer::PathMap(dict);

        // Save to temp file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        let params = SerializeParams {
            path: path.clone(),
            format: SerializationFormat::Text,
            overwrite: true,
        };

        let result = execute_serialize(&container, &params).expect("Failed to serialize");
        assert_eq!(result.term_count, 3);
        assert!(result.byte_size > 0);

        // Load back
        let load_params = DeserializeParams {
            path: path.clone(),
            backend: Some(DictionaryBackend::PathMap),
            format: Some(SerializationFormat::Text),
        };

        let (loaded_container, load_result) =
            execute_deserialize(&load_params).expect("Failed to deserialize");
        assert_eq!(load_result.term_count, 3);
        assert!(loaded_container.contains("apple"));
        assert!(loaded_container.contains("banana"));
        assert!(loaded_container.contains("cherry"));
    }

    #[test]
    fn test_load_text_dict_with_comments() {
        // Create a text file with comments
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "# This is a comment").expect("write failed");
        writeln!(temp_file, "apple").expect("write failed");
        writeln!(temp_file, "").expect("write failed");
        writeln!(temp_file, "# Another comment").expect("write failed");
        writeln!(temp_file, "banana").expect("write failed");
        temp_file.flush().expect("flush failed");

        let path = temp_file.path().to_path_buf();

        let params = DeserializeParams {
            path,
            backend: Some(DictionaryBackend::PathMap),
            format: Some(SerializationFormat::Text),
        };

        let (container, result) = execute_deserialize(&params).expect("Failed to deserialize");
        assert_eq!(result.term_count, 2);
        assert!(container.contains("apple"));
        assert!(container.contains("banana"));
    }

    #[test]
    fn test_serialize_no_overwrite() {
        use libdictenstein::pathmap::PathMapDictionary;

        // Create a temp file that already exists
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        let dict = PathMapDictionary::from_terms(["test"].iter().cloned());
        let container = DictContainer::PathMap(dict);

        let params = SerializeParams {
            path,
            format: SerializationFormat::Text,
            overwrite: false, // Should fail because file exists
        };

        let result = execute_serialize(&container, &params);
        assert!(result.is_err());
    }
}
