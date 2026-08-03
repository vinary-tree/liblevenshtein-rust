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
use crate::serialization::{BincodeSerializer, DictionarySerializer};
#[cfg(feature = "protobuf")]
use crate::serialization::{ProtobufSerializer, SuffixAutomatonProtobufSerializer};

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
        #[cfg(feature = "serialization")]
        SerializationFormat::Bincode => save_bincode_dict(container, path),
        #[cfg(all(feature = "protobuf", feature = "serialization"))]
        SerializationFormat::Protobuf => save_protobuf_dict(container, path),
        #[cfg(feature = "compression")]
        SerializationFormat::BincodeGzip => save_bincode_gzip_dict(container, path),
        #[cfg(all(feature = "protobuf", feature = "compression"))]
        SerializationFormat::ProtobufGzip => save_protobuf_gzip_dict(container, path),
        #[cfg(not(feature = "serialization"))]
        _ => bail!("Serialization feature not enabled"),
    }
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

#[cfg(feature = "protobuf")]
fn save_protobuf_dict(container: &DictContainer, path: &Path) -> Result<()> {
    let file = std::fs::File::create(path)?;
    match container {
        DictContainer::PathMap(d) => ProtobufSerializer::serialize(d, file)?,
        DictContainer::DoubleArrayTrie(d) => ProtobufSerializer::serialize(d, file)?,
        DictContainer::DynamicDawg(d) => ProtobufSerializer::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => {
            SuffixAutomatonProtobufSerializer::serialize_suffix_automaton(d, file)?
        }
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

#[cfg(all(feature = "protobuf", feature = "compression"))]
fn save_protobuf_gzip_dict(container: &DictContainer, path: &Path) -> Result<()> {
    use crate::serialization::GzipSerializer;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let file = std::fs::File::create(path)?;
    match container {
        DictContainer::PathMap(d) => GzipSerializer::<ProtobufSerializer>::serialize(d, file)?,
        DictContainer::DoubleArrayTrie(d) => {
            GzipSerializer::<ProtobufSerializer>::serialize(d, file)?
        }
        DictContainer::DynamicDawg(d) => GzipSerializer::<ProtobufSerializer>::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => {
            let mut encoder = GzEncoder::new(file, Compression::default());
            SuffixAutomatonProtobufSerializer::serialize_suffix_automaton(d, &mut encoder)?;
            encoder.finish()?;
        }
    }
    Ok(())
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
        #[cfg(feature = "serialization")]
        SerializationFormat::Bincode => load_bincode_dict(path, format.backend),
        #[cfg(all(feature = "protobuf", feature = "serialization"))]
        SerializationFormat::Protobuf => load_protobuf_dict(path, format.backend),
        #[cfg(feature = "compression")]
        SerializationFormat::BincodeGzip => load_bincode_gzip_dict(path, format.backend),
        #[cfg(all(feature = "protobuf", feature = "compression"))]
        SerializationFormat::ProtobufGzip => load_protobuf_gzip_dict(path, format.backend),
        #[cfg(not(feature = "serialization"))]
        _ => bail!("Serialization feature not enabled"),
    }
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

#[cfg(feature = "protobuf")]
fn load_protobuf_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use libdictenstein::pathmap::PathMapDictionary;

    let file = std::fs::File::open(path)?;
    let container = match backend {
        DictionaryBackend::PathMap => {
            let dict: PathMapDictionary = ProtobufSerializer::deserialize(file)?;
            DictContainer::PathMap(dict)
        }
        DictionaryBackend::DoubleArrayTrie => {
            let dict: DoubleArrayTrie = ProtobufSerializer::deserialize(file)?;
            DictContainer::DoubleArrayTrie(dict)
        }
        DictionaryBackend::DynamicDawg => {
            let dict: DynamicDawg = ProtobufSerializer::deserialize(file)?;
            DictContainer::DynamicDawg(dict)
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict = SuffixAutomatonProtobufSerializer::deserialize_suffix_automaton(file)?;
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

#[cfg(all(feature = "protobuf", feature = "compression"))]
fn load_protobuf_gzip_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use crate::serialization::GzipSerializer;
    use flate2::read::GzDecoder;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use libdictenstein::pathmap::PathMapDictionary;

    let file = std::fs::File::open(path)?;
    let container = match backend {
        DictionaryBackend::PathMap => {
            let dict: PathMapDictionary = GzipSerializer::<ProtobufSerializer>::deserialize(file)?;
            DictContainer::PathMap(dict)
        }
        DictionaryBackend::DoubleArrayTrie => {
            let dict: DoubleArrayTrie = GzipSerializer::<ProtobufSerializer>::deserialize(file)?;
            DictContainer::DoubleArrayTrie(dict)
        }
        DictionaryBackend::DynamicDawg => {
            let dict: DynamicDawg = GzipSerializer::<ProtobufSerializer>::deserialize(file)?;
            DictContainer::DynamicDawg(dict)
        }
        DictionaryBackend::SuffixAutomaton => {
            let decoder = GzDecoder::new(file);
            let dict = SuffixAutomatonProtobufSerializer::deserialize_suffix_automaton(decoder)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
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
    use tempfile::NamedTempFile;

    #[cfg(feature = "serialization")]
    #[test]
    fn test_save_and_load_bincode_dict() {
        use libdictenstein::pathmap::PathMapDictionary;

        // Create a dictionary
        let terms = ["apple", "banana", "cherry"];
        let dict = PathMapDictionary::from_terms(terms.iter().cloned());
        let container = DictContainer::PathMap(dict);

        // Save to temp file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        let params = SerializeParams {
            path: path.clone(),
            format: SerializationFormat::Bincode,
            overwrite: true,
        };

        let result = execute_serialize(&container, &params).expect("Failed to serialize");
        assert_eq!(result.term_count, 3);
        assert!(result.byte_size > 0);

        // Load back
        let load_params = DeserializeParams {
            path: path.clone(),
            backend: Some(DictionaryBackend::PathMap),
            format: Some(SerializationFormat::Bincode),
        };

        let (loaded_container, load_result) =
            execute_deserialize(&load_params).expect("Failed to deserialize");
        assert_eq!(load_result.term_count, 3);
        assert!(loaded_container.contains("apple"));
        assert!(loaded_container.contains("banana"));
        assert!(loaded_container.contains("cherry"));
    }

    #[cfg(feature = "serialization")]
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
            format: SerializationFormat::Bincode,
            overwrite: false, // Should fail because file exists
        };

        let result = execute_serialize(&container, &params);
        assert!(result.is_err());
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_save_and_load_protobuf_pathmap_dict() {
        use libdictenstein::pathmap::PathMapDictionary;

        let terms = ["alpha", "beta", "gamma"];
        let dict = PathMapDictionary::from_terms(terms.iter().cloned());
        let container = DictContainer::PathMap(dict);
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        let save_params = SerializeParams {
            path: path.clone(),
            format: SerializationFormat::Protobuf,
            overwrite: true,
        };
        execute_serialize(&container, &save_params).expect("Failed to serialize protobuf");

        let load_params = DeserializeParams {
            path,
            backend: Some(DictionaryBackend::PathMap),
            format: Some(SerializationFormat::Protobuf),
        };
        let (loaded, result) =
            execute_deserialize(&load_params).expect("Failed to deserialize protobuf");

        assert_eq!(result.term_count, 3);
        assert!(loaded.contains("alpha"));
        assert!(loaded.contains("beta"));
        assert!(loaded.contains("gamma"));
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_save_and_load_protobuf_suffix_automaton_dict() {
        use libdictenstein::suffix_automaton::SuffixAutomaton;

        let container =
            DictContainer::SuffixAutomaton(SuffixAutomaton::from_texts(["abracadabra", "banana"]));
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        let save_params = SerializeParams {
            path: path.clone(),
            format: SerializationFormat::Protobuf,
            overwrite: true,
        };
        execute_serialize(&container, &save_params)
            .expect("Failed to serialize suffix automaton protobuf");

        let load_params = DeserializeParams {
            path,
            backend: Some(DictionaryBackend::SuffixAutomaton),
            format: Some(SerializationFormat::Protobuf),
        };
        let (loaded, result) = execute_deserialize(&load_params)
            .expect("Failed to deserialize suffix automaton protobuf");

        assert_eq!(result.term_count, 2);
        assert!(loaded.contains("abra"));
        assert!(loaded.contains("nana"));
    }

    #[cfg(all(feature = "protobuf", feature = "compression"))]
    #[test]
    fn test_save_and_load_protobuf_gzip_pathmap_dict() {
        use libdictenstein::pathmap::PathMapDictionary;

        let terms = ["delta", "epsilon", "zeta"];
        let dict = PathMapDictionary::from_terms(terms.iter().cloned());
        let container = DictContainer::PathMap(dict);
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        let save_params = SerializeParams {
            path: path.clone(),
            format: SerializationFormat::ProtobufGzip,
            overwrite: true,
        };
        execute_serialize(&container, &save_params).expect("Failed to serialize protobuf gzip");

        let load_params = DeserializeParams {
            path,
            backend: Some(DictionaryBackend::PathMap),
            format: Some(SerializationFormat::ProtobufGzip),
        };
        let (loaded, result) =
            execute_deserialize(&load_params).expect("Failed to deserialize protobuf gzip");

        assert_eq!(result.term_count, 3);
        assert!(loaded.contains("delta"));
        assert!(loaded.contains("epsilon"));
        assert!(loaded.contains("zeta"));
    }
}
