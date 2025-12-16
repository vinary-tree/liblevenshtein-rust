//! CLI command implementations

use anyhow::{bail, Context, Result};
use colored::Colorize;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use crate::commands::core::QueryParams;
use crate::commands::handlers::query::execute_query;
use crate::dictionary::dawg::DawgDictionary;
use crate::dictionary::dawg_optimized::OptimizedDawg;
use crate::dictionary::double_array_trie::DoubleArrayTrie;
use crate::dictionary::dynamic_dawg::DynamicDawg;
use crate::dictionary::pathmap::PathMapDictionary;
use crate::dictionary::suffix_automaton::SuffixAutomaton;
use crate::dictionary::{Dictionary, DictionaryNode};
use crate::repl::state::{DictContainer, DictionaryBackend};
use crate::transducer::Algorithm;

#[cfg(feature = "serialization")]
use crate::serialization::{BincodeSerializer, DictionarySerializer, JsonSerializer};

// Phonetic rule imports
#[cfg(feature = "phonetic-rules")]
use crate::phonetic::{apply_rules_seq, apply_rules_seq_char};

use super::args::{Cli, SerializationFormat};
use super::detect::{detect_format, DictFormat};
use super::paths::{default_dict_path, PersistentConfig};

/// Execute a CLI command based on operation flags
pub fn execute(cli: &Cli) -> Result<()> {
    if cli.repl {
        // REPL is handled in main.rs
        unreachable!("REPL operation should be handled in main");
    } else if cli.compile {
        let input = cli.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--input is required for --compile operation")
        })?;
        cmd_compile(input, cli.output.clone(), cli.unicode, cli.verify)
    } else if cli.phonetic {
        let text = cli.text.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--text is required for --phonetic operation")
        })?;
        let rules = cli.rules.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--rules is required for --phonetic operation")
        })?;
        cmd_phonetic(text, rules, cli.compiled)
    } else if cli.query {
        let term = cli.text.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--text is required for --query operation (use --text for query term)")
        })?;
        cmd_query(QueryOptions {
            term,
            dict_path: cli.dict.clone(),
            backend: cli.backend,
            format: cli.format,
            max_distance: cli.max_distance,
            algorithm: cli.algorithm,
            prefix: cli.prefix,
            show_distances: cli.show_distances,
            limit: cli.limit,
        })
    } else if cli.info {
        cmd_info(cli.dict.clone())
    } else if cli.convert {
        let input = cli.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--input is required for --convert operation")
        })?;
        let output = cli.output.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--output is required for --convert operation")
        })?;
        cmd_convert(
            input,
            output,
            cli.from_backend,
            cli.to_backend,
            cli.from_format,
            cli.to_format,
        )
    } else if cli.insert {
        if cli.terms.is_empty() {
            bail!("At least one term is required for --insert operation");
        }
        cmd_insert(&cli.terms, cli.dict.clone(), cli.backend, cli.format)
    } else if cli.delete {
        if cli.terms.is_empty() {
            bail!("At least one term is required for --delete operation");
        }
        cmd_delete(&cli.terms, cli.dict.clone(), cli.backend, cli.format)
    } else if cli.clear {
        cmd_clear(cli.dict.clone(), cli.backend, cli.format)
    } else if cli.minimize {
        cmd_minimize(cli.dict.clone(), cli.backend, cli.format)
    } else if cli.settings {
        cmd_settings(
            cli.set_dict.clone(),
            cli.set_backend,
            cli.set_format,
            cli.set_algorithm,
            cli.set_max_distance,
            cli.reset,
        )
    } else if cli.config_mgmt {
        cmd_config(cli.switch.clone(), cli.show)
    } else if cli.compile_regex {
        let input = cli.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--input is required for --compile-regex operation")
        })?;
        cmd_compile_regex(input, cli.output.clone(), cli.verify)
    } else if cli.match_regex {
        let text = cli.text.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--text is required for --match-regex operation")
        })?;
        let rules = cli.rules.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--rules is required for --match-regex operation")
        })?;
        cmd_match_regex(text, rules, cli.multiline, cli.dotall, cli.compiled)
    } else {
        bail!("No operation specified. Use --help for available operations.")
    }
}

/// Query command options
struct QueryOptions<'a> {
    term: &'a str,
    dict_path: Option<PathBuf>,
    backend: Option<DictionaryBackend>,
    format: Option<SerializationFormat>,
    max_distance: usize,
    algorithm: Algorithm,
    prefix: bool,
    show_distances: bool,
    limit: Option<usize>,
}

/// Query command
fn cmd_query(opts: QueryOptions) -> Result<()> {
    let (path, dict_format) =
        resolve_dict_path_and_format(opts.dict_path, opts.backend, opts.format)?;

    // Load dictionary
    let container = load_dictionary(&path, dict_format)?;

    // Create query parameters
    let params = QueryParams {
        term: opts.term.to_string(),
        max_distance: opts.max_distance,
        algorithm: opts.algorithm,
        prefix: opts.prefix,
        show_distances: false, // Not used by execute_query
        limit: opts.limit,
    };

    // Perform query using shared handler
    let results: Vec<(String, usize)> = match &container {
        DictContainer::PathMap(d) => execute_query(d, &params),
        DictContainer::DoubleArrayTrie(d) => execute_query(d, &params),
        DictContainer::Dawg(d) => execute_query(d, &params),
        DictContainer::OptimizedDawg(d) => execute_query(d, &params),
        DictContainer::DynamicDawg(d) => execute_query(d, &params),
        DictContainer::SuffixAutomaton(d) => execute_query(d, &params),
    };

    // Print results
    if results.is_empty() {
        println!("{}", "No matches found".yellow());
    } else {
        for (i, (candidate, distance)) in results.iter().enumerate() {
            if opts.show_distances {
                println!("   {}. {} (d={})", i + 1, candidate.green(), distance);
            } else {
                println!("   {}. {}", i + 1, candidate.green());
            }
        }
        println!();
        println!("{} match(es) found", results.len());
    }

    Ok(())
}

/// Info command
fn cmd_info(dict_path: Option<PathBuf>) -> Result<()> {
    let (path, dict_format) = resolve_dict_path_and_format(dict_path, None, None)?;

    println!("{}", "Dictionary Information".bold().underline());
    println!();
    println!("  Path:    {}", path.display().to_string().cyan());

    // Detect format
    let detection = detect_format(&path, None, None)?;
    println!(
        "  Backend: {} (detected via {})",
        detection.format.backend.to_string().green(),
        detection.method.to_string().yellow()
    );
    println!(
        "  Format:  {} (detected via {})",
        detection.format.format.to_string().green(),
        detection.method.to_string().yellow()
    );

    // Load and get stats
    let container = load_dictionary(&path, dict_format)?;
    println!("  Terms:   {}", container.len().to_string().green());

    match &container {
        DictContainer::Dawg(d) => {
            let nodes = d.node_count();
            let ratio = nodes as f64 / container.len() as f64;
            println!("  Nodes:   {}", nodes.to_string().green());
            println!("  Ratio:   {:.2}x", ratio);
        }
        DictContainer::OptimizedDawg(d) => {
            let nodes = d.node_count();
            let ratio = nodes as f64 / container.len() as f64;
            println!("  Nodes:   {}", nodes.to_string().green());
            println!("  Ratio:   {:.2}x", ratio);
        }
        DictContainer::DynamicDawg(d) => {
            let nodes = d.node_count();
            let ratio = nodes as f64 / container.len() as f64;
            println!("  Nodes:   {}", nodes.to_string().green());
            println!("  Ratio:   {:.2}x", ratio);
        }
        _ => {}
    }

    println!();

    Ok(())
}

/// Convert command
fn cmd_convert(
    input: &Path,
    output: &Path,
    from_backend: Option<DictionaryBackend>,
    to_backend: DictionaryBackend,
    from_format: Option<SerializationFormat>,
    to_format: SerializationFormat,
) -> Result<()> {
    // Detect input format
    let detection = detect_format(input, from_backend, from_format)?;
    println!(
        "{}  Input:  {} ({}, detected via {})",
        "→".cyan(),
        input.display().to_string().yellow(),
        detection.format.backend.to_string().green(),
        detection.method
    );

    // Load dictionary
    let container = load_dictionary(input, detection.format)?;
    println!("  Loaded {} terms", container.len().to_string().green());

    // Convert backend if needed
    let output_container = if detection.format.backend != to_backend {
        println!(
            "  Converting from {} to {}...",
            detection.format.backend.to_string().yellow(),
            to_backend.to_string().green()
        );
        container.migrate_to(to_backend)?
    } else {
        container
    };

    // Save to output
    println!(
        "{}  Output: {} ({})",
        "→".cyan(),
        output.display().to_string().yellow(),
        to_backend.to_string().green()
    );
    save_dictionary(&output_container, output, to_format)?;

    println!();
    println!("{}", "Conversion complete!".green().bold());

    Ok(())
}

/// Insert command
fn cmd_insert(
    terms: &[String],
    dict_path: Option<PathBuf>,
    backend: Option<DictionaryBackend>,
    format: Option<SerializationFormat>,
) -> Result<()> {
    let (path, dict_format) = resolve_dict_path_and_format(dict_path, backend, format)?;

    // Load or create dictionary
    let mut container = if path.exists() {
        load_dictionary(&path, dict_format)?
    } else {
        println!(
            "  Creating new dictionary at {}",
            path.display().to_string().cyan()
        );
        create_empty_dict(dict_format.backend)
    };

    // Insert terms
    let mut inserted = 0;
    for term in terms {
        if container.insert(term)? {
            inserted += 1;
        }
    }

    // Save
    save_dictionary(&container, &path, dict_format.format)?;

    println!();
    println!("  Inserted {} new term(s)", inserted.to_string().green());
    println!("  Total terms: {}", container.len().to_string().cyan());

    Ok(())
}

/// Delete command
fn cmd_delete(
    terms: &[String],
    dict_path: Option<PathBuf>,
    backend: Option<DictionaryBackend>,
    format: Option<SerializationFormat>,
) -> Result<()> {
    let (path, dict_format) = resolve_dict_path_and_format(dict_path, backend, format)?;

    if !path.exists() {
        bail!("Dictionary file does not exist: {}", path.display());
    }

    // Load dictionary
    let mut container = load_dictionary(&path, dict_format)?;

    // Delete terms
    let mut deleted = 0;
    for term in terms {
        if container.remove(term)? {
            deleted += 1;
        }
    }

    // Save
    save_dictionary(&container, &path, dict_format.format)?;

    println!();
    println!("  Deleted {} term(s)", deleted.to_string().green());
    println!("  Remaining terms: {}", container.len().to_string().cyan());

    Ok(())
}

/// Minimize command
fn cmd_minimize(
    dict_path: Option<PathBuf>,
    backend: Option<DictionaryBackend>,
    format: Option<SerializationFormat>,
) -> Result<()> {
    let (path, dict_format) = resolve_dict_path_and_format(dict_path, backend, format)?;

    if !path.exists() {
        bail!("Dictionary file does not exist: {}", path.display());
    }

    // Load dictionary
    let mut container = load_dictionary(&path, dict_format)?;

    println!(
        "  Minimizing dictionary at {}...",
        path.display().to_string().cyan()
    );

    // Try to compact/minimize
    match container.compact() {
        Ok(()) => {
            // Save
            save_dictionary(&container, &path, dict_format.format)?;
            println!();
            println!("  {}", "Dictionary minimized successfully".green().bold());

            // Show stats if available
            if let DictContainer::DynamicDawg(d) = &container {
                let nodes = d.node_count();
                let terms = container.len();
                let ratio = nodes as f64 / terms as f64;
                println!(
                    "  Terms: {}, Nodes: {}, Ratio: {:.2}x",
                    terms.to_string().cyan(),
                    nodes.to_string().cyan(),
                    ratio
                );
            }
        }
        Err(e) => {
            // Check if it's a "not supported" error
            let error_msg = e.to_string();
            if error_msg.contains("read-only") || error_msg.contains("already minimized") {
                println!();
                println!("  {}: {}", "Warning".yellow(), e);
                println!("  {}", "No action taken".yellow());
            } else {
                return Err(e);
            }
        }
    }

    println!();
    Ok(())
}

/// Clear command
fn cmd_clear(
    dict_path: Option<PathBuf>,
    backend: Option<DictionaryBackend>,
    format: Option<SerializationFormat>,
) -> Result<()> {
    let (path, dict_format) = resolve_dict_path_and_format(dict_path, backend, format)?;

    if !path.exists() {
        bail!("Dictionary file does not exist: {}", path.display());
    }

    // Load dictionary
    let mut container = load_dictionary(&path, dict_format)?;
    let original_count = container.len();

    if original_count == 0 {
        println!();
        println!("  Dictionary is already empty");
        return Ok(());
    }

    // Confirm
    print!(
        "  {} {} ",
        format!("Clear {} term(s) from {}?", original_count, path.display()).yellow(),
        "(y/N):".yellow().bold()
    );
    std::io::Write::flush(&mut std::io::stdout())?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
        println!();
        println!("  Clear cancelled");
        return Ok(());
    }

    // Clear
    container.clear()?;

    // Save
    save_dictionary(&container, &path, dict_format.format)?;

    println!();
    println!("  Cleared {} term(s)", original_count.to_string().green());

    Ok(())
}

/// Resolve dictionary path and format
fn resolve_dict_path_and_format(
    dict_path: Option<PathBuf>,
    backend: Option<DictionaryBackend>,
    format: Option<SerializationFormat>,
) -> Result<(PathBuf, DictFormat)> {
    // Try to get from config if not specified
    let config = PersistentConfig::load().unwrap_or_default();

    let path = if let Some(p) = dict_path {
        p
    } else if let Some(p) = config.dict_path {
        p
    } else {
        // Use default path
        let backend = backend
            .or(config.backend)
            .unwrap_or(DictionaryBackend::PathMap);
        let format = format
            .or(config.format)
            .unwrap_or(SerializationFormat::Text);
        default_dict_path(backend, format)?
    };

    // Detect or use specified format
    // Only pass explicitly user-provided backend/format to detect_format for validation.
    // Config values are used as fallback when detection fails, not as constraints.
    let detection = detect_format(&path, backend, format)?;

    Ok((path, detection.format))
}

/// Load dictionary from file
pub fn load_dictionary(path: &Path, format: DictFormat) -> Result<DictContainer> {
    if !path.exists() {
        bail!("Dictionary file does not exist: {}", path.display());
    }

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

/// Load plain text dictionary
fn load_text_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
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
        DictionaryBackend::Dawg => {
            let dict: DawgDictionary = BincodeSerializer::deserialize(file)?;
            DictContainer::Dawg(dict)
        }
        DictionaryBackend::OptimizedDawg => {
            // OptimizedDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = BincodeSerializer::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::DynamicDawg => {
            // DynamicDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = BincodeSerializer::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict = BincodeSerializer::deserialize_suffix_automaton(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

#[cfg(feature = "serialization")]
fn load_json_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
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
        DictionaryBackend::Dawg => {
            let dict: DawgDictionary = JsonSerializer::deserialize(file)?;
            DictContainer::Dawg(dict)
        }
        DictionaryBackend::OptimizedDawg => {
            // OptimizedDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = JsonSerializer::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::DynamicDawg => {
            // DynamicDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = JsonSerializer::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            // Use generic serialization
            let dict: SuffixAutomaton = JsonSerializer::deserialize(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

#[cfg(feature = "compression")]
fn load_bincode_gzip_dict(path: &Path, backend: DictionaryBackend) -> Result<DictContainer> {
    use crate::serialization::GzipSerializer;

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
        DictionaryBackend::Dawg => {
            let dict: DawgDictionary = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            DictContainer::Dawg(dict)
        }
        DictionaryBackend::OptimizedDawg => {
            // OptimizedDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::DynamicDawg => {
            // DynamicDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = GzipSerializer::<BincodeSerializer>::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
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
        DictionaryBackend::Dawg => {
            let dict: DawgDictionary = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            DictContainer::Dawg(dict)
        }
        DictionaryBackend::OptimizedDawg => {
            // OptimizedDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::DynamicDawg => {
            // DynamicDawg doesn't implement DictionaryFromTerms, so extract terms and rebuild
            let dict: PathMapDictionary = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            let terms = extract_terms_from_dict(&dict);
            create_dict_from_terms(terms, backend)?
        }
        DictionaryBackend::SuffixAutomaton => {
            let dict: SuffixAutomaton = GzipSerializer::<JsonSerializer>::deserialize(file)?;
            DictContainer::SuffixAutomaton(dict)
        }
    };
    Ok(container)
}

/// Extract terms from a dictionary
fn extract_terms_from_dict<D>(dict: &D) -> Vec<String>
where
    D: Dictionary,
    D::Node: DictionaryNode<Unit = u8>,
{
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

/// Create dictionary from terms
fn create_dict_from_terms(terms: Vec<String>, backend: DictionaryBackend) -> Result<DictContainer> {
    let container = match backend {
        DictionaryBackend::PathMap => DictContainer::PathMap(PathMapDictionary::from_terms(
            terms.iter().map(|s| s.as_str()),
        )),
        DictionaryBackend::DoubleArrayTrie => {
            DictContainer::DoubleArrayTrie(DoubleArrayTrie::from_terms(terms))
        }
        DictionaryBackend::Dawg => {
            DictContainer::Dawg(DawgDictionary::from_iter(terms.iter().map(|s| s.as_str())))
        }
        DictionaryBackend::OptimizedDawg => {
            DictContainer::OptimizedDawg(OptimizedDawg::from_terms(terms))
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

/// Create empty dictionary
fn create_empty_dict(backend: DictionaryBackend) -> DictContainer {
    match backend {
        DictionaryBackend::PathMap => DictContainer::PathMap(PathMapDictionary::new()),
        DictionaryBackend::DoubleArrayTrie => {
            DictContainer::DoubleArrayTrie(DoubleArrayTrie::from_terms(Vec::<String>::new()))
        }
        DictionaryBackend::Dawg => {
            DictContainer::Dawg(DawgDictionary::from_iter(Vec::<&str>::new()))
        }
        DictionaryBackend::OptimizedDawg => {
            DictContainer::OptimizedDawg(OptimizedDawg::from_terms(Vec::<String>::new()))
        }
        DictionaryBackend::DynamicDawg => DictContainer::DynamicDawg(DynamicDawg::new()),
        DictionaryBackend::SuffixAutomaton => {
            DictContainer::SuffixAutomaton(SuffixAutomaton::new())
        }
    }
}

/// Save dictionary to file
pub fn save_dictionary(
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

/// Save as plain text
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
        DictContainer::Dawg(d) => BincodeSerializer::serialize(d, file)?,
        DictContainer::OptimizedDawg(d) => BincodeSerializer::serialize(d, file)?,
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
        DictContainer::Dawg(d) => JsonSerializer::serialize(d, file)?,
        DictContainer::OptimizedDawg(d) => JsonSerializer::serialize(d, file)?,
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
        DictContainer::Dawg(d) => GzipSerializer::<BincodeSerializer>::serialize(d, file)?,
        DictContainer::OptimizedDawg(d) => GzipSerializer::<BincodeSerializer>::serialize(d, file)?,
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
        DictContainer::Dawg(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
        DictContainer::OptimizedDawg(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
        DictContainer::DynamicDawg(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
        DictContainer::SuffixAutomaton(d) => GzipSerializer::<JsonSerializer>::serialize(d, file)?,
    }
    Ok(())
}

/// Save as PathsNative format
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
        _ => bail!("PathsNative format is only supported for PathMap backend"),
    }
}

/// Load from PathsNative format
fn load_paths_native_dict(path: &Path) -> Result<DictContainer> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;

    let dict = PathMapDictionary::deserialize_paths(file)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize paths: {}", e))
        .with_context(|| format!("Failed to deserialize paths from: {}", path.display()))?;

    Ok(DictContainer::PathMap(dict))
}

/// Config command
fn cmd_settings(
    set_dict: Option<PathBuf>,
    set_backend: Option<DictionaryBackend>,
    set_format: Option<SerializationFormat>,
    set_algorithm: Option<Algorithm>,
    set_max_distance: Option<usize>,
    reset: bool,
) -> Result<()> {
    if reset {
        let config = PersistentConfig::default();
        config.save()?;
        println!("{}", "Configuration reset to defaults".green().bold());
        println!();
        print_config(&config);
        return Ok(());
    }

    let mut config = PersistentConfig::load().unwrap_or_default();
    let mut changed = false;

    if let Some(dict) = set_dict {
        config.dict_path = Some(dict.clone());
        println!(
            "  Set default dictionary path: {}",
            dict.display().to_string().cyan()
        );
        changed = true;
    }

    if let Some(backend) = set_backend {
        config.backend = Some(backend);
        println!("  Set default backend: {}", backend.to_string().green());
        changed = true;
    }

    if let Some(format) = set_format {
        // If we have a dict_path, validate it matches the new format
        if let Some(ref dict_path) = config.dict_path {
            use super::paths::{change_extension, file_extension, validate_dict_path};

            if validate_dict_path(dict_path, format).is_err() {
                let new_path = change_extension(dict_path, format);
                let expected_ext = file_extension(format);

                eprintln!(
                    "  {}: Dictionary path extension doesn't match new format",
                    "Warning".yellow().bold()
                );
                eprintln!(
                    "    Current path: {}",
                    dict_path.display().to_string().yellow()
                );
                eprintln!("    Expected extension: .{}", expected_ext);
                eprintln!(
                    "    Suggested path: {}",
                    new_path.display().to_string().cyan()
                );
                return Err(anyhow::anyhow!(
                    "Dictionary path must have .{} extension for {} format. Use --set-dict to update the path.",
                    expected_ext,
                    format
                ));
            }
        }

        config.format = Some(format);
        println!("  Set default format: {}", format.to_string().green());
        changed = true;
    }

    if let Some(algorithm) = set_algorithm {
        config.algorithm = Some(algorithm);
        println!("  Set default algorithm: {}", algorithm.to_string().green());
        changed = true;
    }

    if let Some(distance) = set_max_distance {
        config.max_distance = Some(distance);
        println!(
            "  Set default max distance: {}",
            distance.to_string().green()
        );
        changed = true;
    }

    if changed {
        config.save()?;
        println!();
        println!("{}", "Configuration saved".green().bold());
    }

    println!();
    print_config(&config);

    Ok(())
}

/// Config file management command
fn cmd_config(switch: Option<PathBuf>, show: bool) -> Result<()> {
    use super::paths::AppConfig;

    if let Some(new_path) = switch {
        // Switch to a different config file
        use super::paths::validate_config_path;

        // Validate config file extension first
        validate_config_path(&new_path)?;

        let mut app_config = AppConfig::load()?;

        // If the new config file doesn't exist, create it with current settings
        if !new_path.exists() {
            let current_config = PersistentConfig::load()?;
            current_config.copy_to(&new_path)?;
            println!(
                "  Copied current configuration to {}",
                new_path.display().to_string().cyan()
            );
        }

        app_config.switch_user_config(new_path.clone())?;
        println!();
        println!("{}", "Config file switched successfully".green().bold());
        println!("  Now using: {}", new_path.display().to_string().cyan());

        // Load and display the new config
        let config = PersistentConfig::load()?;
        println!();
        print_config(&config);
    } else if show {
        // Just show current config file path
        let app_config = AppConfig::load()?;
        println!("{}", "Config File Location:".bold().underline());
        println!();
        println!(
            "  App Config:  {}",
            super::paths::app_config_path()?
                .display()
                .to_string()
                .cyan()
        );
        println!(
            "  User Config: {}",
            app_config.user_config_path.display().to_string().cyan()
        );
    } else {
        // No arguments - show help
        println!("{}", "Config File Management".bold().underline());
        println!();
        println!("Usage:");
        println!("  liblevenshtein config --show          Show config file paths");
        println!("  liblevenshtein config --switch <path> Switch to different config file");
    }

    Ok(())
}

/// Print current configuration
fn print_config(config: &PersistentConfig) {
    println!("{}", "Current Configuration:".bold().underline());
    println!();

    // Dictionary Path - show actual default if not set
    let dict_path = if let Some(ref path) = config.dict_path {
        path.display().to_string()
    } else {
        let default_backend = config.backend.unwrap_or(DictionaryBackend::PathMap);
        let default_format = if let Some(fmt) = config.format {
            fmt
        } else {
            #[cfg(feature = "protobuf")]
            {
                SerializationFormat::Protobuf
            }
            #[cfg(not(feature = "protobuf"))]
            {
                SerializationFormat::Bincode
            }
        };

        super::paths::default_dict_path(default_backend, default_format)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "(auto)".to_string())
    };
    println!("  Dictionary Path: {}", dict_path.cyan());

    // Backend
    let backend = config.backend.unwrap_or(DictionaryBackend::PathMap);
    println!("  Backend:         {}", backend.to_string().yellow());

    // Serialization
    let format = config.format.unwrap_or({
        #[cfg(feature = "protobuf")]
        {
            SerializationFormat::Protobuf
        }
        #[cfg(not(feature = "protobuf"))]
        {
            SerializationFormat::Bincode
        }
    });
    println!("  Serialization:   {}", format.to_string().yellow());

    println!(
        "  Algorithm:       {}",
        config
            .algorithm
            .unwrap_or(Algorithm::Standard)
            .to_string()
            .yellow()
    );
    println!(
        "  Max Distance:    {}",
        config.max_distance.unwrap_or(2).to_string().yellow()
    );
    println!(
        "  Prefix Mode:     {}",
        if config.prefix_mode.unwrap_or(false) {
            "enabled".green()
        } else {
            "disabled".red()
        }
    );
    println!(
        "  Show Distances:  {}",
        if config.show_distances.unwrap_or(false) {
            "enabled".green()
        } else {
            "disabled".red()
        }
    );

    if let Some(Some(limit)) = config.result_limit {
        println!("  Result Limit:    {}", limit.to_string().yellow());
    } else {
        println!("  Result Limit:    {}", "none".yellow());
    }

    println!(
        "  Auto-sync:       {}",
        if config.auto_sync.unwrap_or(false) {
            "enabled".green()
        } else {
            "disabled".red()
        }
    );

    println!();
    println!(
        "  Config file: {}",
        super::paths::config_file_path()
            .unwrap()
            .display()
            .to_string()
            .cyan()
    );
}

/// Compile phonetic rules from .llev to binary format
#[cfg(all(feature = "phonetic-rules", feature = "serialization"))]
fn cmd_compile(
    input: &Path,
    output: Option<PathBuf>,
    unicode: bool,
    verify: bool,
) -> Result<()> {
    use crate::phonetic::llev::{load_file_with_includes, RuleSet, RuleSetChar};
    use crate::phonetic::{save, save_char};

    println!("{}", "Compiling Phonetic Rules".bold().underline());
    println!();

    // Parse the .llev file
    println!(
        "  {} Loading {}...",
        "→".cyan(),
        input.display().to_string().yellow()
    );

    // Use empty include paths - all includes should be relative to the input file
    let include_paths: &[&Path] = &[];
    let llev_file = load_file_with_includes(input, include_paths)
        .with_context(|| format!("Failed to parse {}", input.display()))?;

    // Display file metadata
    if let Some(name) = &llev_file.metadata.name {
        println!("    Name:    {}", name.green());
    }
    if let Some(version) = &llev_file.metadata.version {
        println!("    Version: {}", version.green());
    }
    println!("    Rules:   {}", llev_file.rules.len().to_string().green());

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let mut out = input.to_path_buf();
        let ext = if unicode { "llev.bin" } else { "llev.bin" };
        out.set_extension(ext);
        out
    });

    println!();
    println!(
        "  {} Writing {}...",
        "→".cyan(),
        output_path.display().to_string().yellow()
    );

    if unicode {
        // Character-level (Unicode) rules
        let ruleset = RuleSetChar::from_llev(&llev_file)
            .map_err(|e| anyhow::anyhow!("Failed to convert rules: {}", e))?;

        println!("    Mode:    {}", "Unicode (char-level)".green());
        println!(
            "    Rules:   {}",
            ruleset.rules.len().to_string().green()
        );

        save_char(&ruleset, &output_path)
            .map_err(|e| anyhow::anyhow!("Failed to save compiled rules: {}", e))?;

        if verify {
            println!();
            println!("  {} Verifying...", "→".cyan());
            let loaded = crate::phonetic::load_char(&output_path)
                .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;

            if loaded.rules.len() == ruleset.rules.len() {
                println!("    {}", "Verification passed".green().bold());
            } else {
                bail!(
                    "Verification failed: expected {} rules, got {}",
                    ruleset.rules.len(),
                    loaded.rules.len()
                );
            }
        }
    } else {
        // Byte-level (ASCII) rules
        let ruleset = RuleSet::from_llev(&llev_file)
            .map_err(|e| anyhow::anyhow!("Failed to convert rules: {}", e))?;

        println!("    Mode:    {}", "ASCII (byte-level)".green());
        println!(
            "    Rules:   {}",
            ruleset.rules.len().to_string().green()
        );

        save(&ruleset, &output_path)
            .map_err(|e| anyhow::anyhow!("Failed to save compiled rules: {}", e))?;

        if verify {
            println!();
            println!("  {} Verifying...", "→".cyan());
            let loaded = crate::phonetic::load(&output_path)
                .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;

            if loaded.rules.len() == ruleset.rules.len() {
                println!("    {}", "Verification passed".green().bold());
            } else {
                bail!(
                    "Verification failed: expected {} rules, got {}",
                    ruleset.rules.len(),
                    loaded.rules.len()
                );
            }
        }
    }

    // Show file size
    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!();
    println!(
        "  {} Compiled {} bytes",
        "✓".green().bold(),
        file_size.to_string().cyan()
    );

    Ok(())
}

#[cfg(not(all(feature = "phonetic-rules", feature = "serialization")))]
fn cmd_compile(
    _input: &Path,
    _output: Option<PathBuf>,
    _unicode: bool,
    _verify: bool,
) -> Result<()> {
    bail!(
        "Compile command requires both 'phonetic-rules' and 'serialization' features.\n\
         Rebuild with: cargo build --features \"phonetic-rules,serialization\""
    );
}

/// Apply phonetic rules to transform text
#[cfg(all(feature = "phonetic-rules", feature = "serialization"))]
fn cmd_phonetic(text: &str, rules_path: &Path, compiled: bool) -> Result<()> {
    use crate::phonetic::llev::{load_file_with_includes, RuleSetChar};
    use crate::phonetic::{load_char, PhoneChar};

    println!("{}", "Phonetic Transformation".bold().underline());
    println!();

    // Determine if compiled based on extension or flag
    let is_compiled = compiled
        || rules_path
            .extension()
            .map(|e| e == "bin")
            .unwrap_or(false);

    let ruleset: RuleSetChar = if is_compiled {
        println!(
            "  {} Loading compiled rules from {}...",
            "→".cyan(),
            rules_path.display().to_string().yellow()
        );
        load_char(rules_path).map_err(|e| anyhow::anyhow!("Failed to load compiled rules: {}", e))?
    } else {
        println!(
            "  {} Parsing rules from {}...",
            "→".cyan(),
            rules_path.display().to_string().yellow()
        );
        // Use empty include paths - all includes should be relative to the rules file
        let include_paths: &[&Path] = &[];
        let llev_file = load_file_with_includes(rules_path, include_paths)
            .with_context(|| format!("Failed to parse {}", rules_path.display()))?;
        RuleSetChar::from_llev(&llev_file)
            .map_err(|e| anyhow::anyhow!("Failed to convert rules: {}", e))?
    };

    println!("    Rules:   {}", ruleset.rules.len().to_string().green());

    // Convert input string to Vec<PhoneChar>
    let vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
    let input_phones: Vec<PhoneChar> = text
        .chars()
        .map(|c| {
            if vowels.contains(&c) {
                PhoneChar::Vowel(c)
            } else {
                PhoneChar::Consonant(c)
            }
        })
        .collect();

    // Apply rules with sufficient fuel (max 1000 iterations)
    println!();
    println!("  {} Applying rules...", "→".cyan());
    let result = apply_rules_seq_char(&ruleset.rules, &input_phones, 1000);

    // Convert result back to string
    let output = match result {
        Some(phones) => {
            let mut s = String::new();
            for p in &phones {
                match p {
                    PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => s.push(*c),
                    PhoneChar::Digraph(c1, c2) => {
                        s.push(*c1);
                        s.push(*c2);
                    }
                    PhoneChar::Silent => {}
                }
            }
            s
        }
        None => text.to_string(),
    };

    println!();
    println!("  Input:  {}", text.yellow());
    println!("  Output: {}", output.green().bold());

    Ok(())
}

#[cfg(not(all(feature = "phonetic-rules", feature = "serialization")))]
fn cmd_phonetic(_text: &str, _rules_path: &Path, _compiled: bool) -> Result<()> {
    bail!(
        "Phonetic command requires both 'phonetic-rules' and 'serialization' features.\n\
         Rebuild with: cargo build --features \"phonetic-rules,serialization\""
    );
}

/// Compile .llre regex file to binary format
#[cfg(all(feature = "phonetic-rules", feature = "serialization"))]
fn cmd_compile_regex(input: &Path, output: Option<PathBuf>, verify: bool) -> Result<()> {
    use crate::phonetic::llre::{compile, load_file, save, to_bytes};

    println!("{}", "Compiling LLRE Regex".bold().underline());
    println!();

    // Load the .llre file
    println!(
        "  {} Loading {}...",
        "→".cyan(),
        input.display().to_string().yellow()
    );

    let llre_file = load_file(input)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", input.display(), e))?;

    // Display file metadata
    if let Some(name) = &llre_file.metadata.name {
        println!("    Name:    {}", name.green());
    }
    if let Some(version) = &llre_file.metadata.version {
        println!("    Version: {}", version.green());
    }
    if llre_file.is_multiline() {
        println!("    Flags:   {}", "multiline".cyan());
    }
    if llre_file.is_dotall() {
        println!("    Flags:   {}", "dotall".cyan());
    }

    // Compile to NFA
    println!();
    println!("  {} Compiling to NFA...", "→".cyan());

    let compiled = compile(&llre_file)
        .map_err(|e| anyhow::anyhow!("Failed to compile regex: {}", e))?;

    println!("    States:      {}", compiled.state_count().to_string().green());
    println!("    Transitions: {}", compiled.transition_count().to_string().green());

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let mut out = input.to_path_buf();
        out.set_extension("llre.bin");
        out
    });

    println!();
    println!(
        "  {} Writing {}...",
        "→".cyan(),
        output_path.display().to_string().yellow()
    );

    save(&compiled, &output_path)
        .map_err(|e| anyhow::anyhow!("Failed to save compiled regex: {}", e))?;

    if verify {
        println!();
        println!("  {} Verifying...", "→".cyan());

        let loaded = crate::phonetic::llre::load(&output_path)
            .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;

        // Test that the loaded NFA has the same state count
        if loaded.state_count() == compiled.state_count() {
            println!("    {}", "Verification passed".green().bold());
        } else {
            bail!(
                "Verification failed: expected {} states, got {}",
                compiled.state_count(),
                loaded.state_count()
            );
        }
    }

    // Show file size
    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!();
    println!(
        "  {} Compiled {} bytes",
        "✓".green().bold(),
        file_size.to_string().cyan()
    );

    Ok(())
}

#[cfg(not(all(feature = "phonetic-rules", feature = "serialization")))]
fn cmd_compile_regex(_input: &Path, _output: Option<PathBuf>, _verify: bool) -> Result<()> {
    bail!(
        "Compile-regex command requires both 'phonetic-rules' and 'serialization' features.\n\
         Rebuild with: cargo build --features \"phonetic-rules,serialization\""
    );
}

/// Match text against a .llre regex pattern
#[cfg(all(feature = "phonetic-rules", feature = "serialization"))]
fn cmd_match_regex(
    text: &str,
    rules_path: &Path,
    multiline: bool,
    dotall: bool,
    compiled: bool,
) -> Result<()> {
    use crate::phonetic::llre::{compile, load, load_file, CompiledNFA};

    println!("{}", "LLRE Regex Match".bold().underline());
    println!();

    // Determine if compiled based on extension or flag
    let is_compiled = compiled
        || rules_path
            .extension()
            .map(|e| e == "bin")
            .unwrap_or(false);

    let mut nfa: CompiledNFA = if is_compiled {
        println!(
            "  {} Loading compiled regex from {}...",
            "→".cyan(),
            rules_path.display().to_string().yellow()
        );
        load(rules_path)
            .map_err(|e| anyhow::anyhow!("Failed to load compiled regex: {}", e))?
    } else {
        println!(
            "  {} Parsing regex from {}...",
            "→".cyan(),
            rules_path.display().to_string().yellow()
        );
        let llre_file = load_file(rules_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", rules_path.display(), e))?;
        compile(&llre_file)
            .map_err(|e| anyhow::anyhow!("Failed to compile regex: {}", e))?
    };

    // Override flags from command line if specified
    if multiline {
        nfa.multiline = true;
    }
    if dotall {
        nfa.dotall = true;
    }

    println!(
        "    States:      {}",
        nfa.state_count().to_string().green()
    );
    println!(
        "    Transitions: {}",
        nfa.transition_count().to_string().green()
    );
    if nfa.multiline {
        println!("    Flags:       {}", "multiline".cyan());
    }
    if nfa.dotall {
        println!("    Flags:       {}", "dotall".cyan());
    }

    // Perform match
    println!();
    println!("  {} Testing match...", "→".cyan());
    println!("    Input:  {}", text.yellow());

    let matches = nfa.matches(text);

    if matches {
        println!("    Result: {}", "MATCH".green().bold());
    } else {
        println!("    Result: {}", "NO MATCH".red().bold());
    }

    // Also try full match
    let full_matches = nfa.matches_full(text);
    if full_matches != matches {
        println!();
        if full_matches {
            println!("    Full match: {}", "YES".green());
        } else {
            println!("    Full match: {}", "NO".yellow());
        }
    }

    Ok(())
}

#[cfg(not(all(feature = "phonetic-rules", feature = "serialization")))]
fn cmd_match_regex(
    _text: &str,
    _rules_path: &Path,
    _multiline: bool,
    _dotall: bool,
    _compiled: bool,
) -> Result<()> {
    bail!(
        "Match-regex command requires both 'phonetic-rules' and 'serialization' features.\n\
         Rebuild with: cargo build --features \"phonetic-rules,serialization\""
    );
}
