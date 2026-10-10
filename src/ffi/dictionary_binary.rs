//! Bounded C bridge to native dictionary persistence formats.

use super::{index::boundary, LlevOwnedBytes, LlevStatus};
#[cfg(feature = "compression")]
use libdictenstein::serialization::GzipSerializer;
#[cfg(feature = "protobuf")]
use libdictenstein::serialization::{
    DatProtobufSerializer, OptimizedProtobufSerializer, ProtobufSerializer,
};
#[cfg(feature = "serialization")]
use libdictenstein::{
    double_array_trie::DoubleArrayTrie,
    serialization::{extract_terms, BincodeSerializer, DictionarySerializer},
};
#[cfg(feature = "protobuf")]
use prost::Message;
#[cfg(feature = "compression")]
use std::io::Read;
#[cfg(feature = "serialization")]
use std::io::{self, Write};

#[cfg(feature = "protobuf")]
pub(super) mod protobuf_wire {
    #![allow(dead_code)] // The shared schema also generates specialized formats.
    include!(concat!(env!("OUT_DIR"), "/liblevenshtein.proto.rs"));
}

/// Borrowed UTF-8 term valid for the duration of a call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevUtf8Slice {
    /// Borrowed term bytes; may be null when length is zero.
    pub data: *const u8,
    /// Number of bytes in the UTF-8 term.
    pub len: usize,
}

/// Caller-selected input and output ceilings for dictionary persistence.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevDictionaryLimits {
    /// Maximum number of terms accepted or returned.
    pub max_terms: usize,
    /// Maximum UTF-8 bytes in one term.
    pub max_term_bytes: usize,
    /// Maximum UTF-8 bytes across all terms.
    pub max_total_term_bytes: usize,
    /// Maximum binary payload bytes read or written.
    pub max_payload_bytes: usize,
}

/// Immutable decoded term snapshot; borrowed term views last until free.
pub struct LlevDecodedDictionaryTerms {
    pub(super) terms: Vec<String>,
}

#[cfg(feature = "serialization")]
const BINCODE_V1: u32 = 1;
#[cfg(feature = "serialization")]
const PROTOBUF_V1: u32 = 2;
#[cfg(feature = "serialization")]
const PROTOBUF_V2: u32 = 3;
#[cfg(feature = "serialization")]
const GZIP_BINCODE_V1: u32 = 4;
#[cfg(feature = "serialization")]
const GZIP_PROTOBUF_V1: u32 = 5;
#[cfg(feature = "serialization")]
const GZIP_PROTOBUF_V2: u32 = 6;
#[cfg(feature = "serialization")]
const PROTOBUF_DAT_V1: u32 = 7;

#[cfg(feature = "serialization")]
pub(super) fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

#[cfg(feature = "serialization")]
pub(super) fn limits(
    raw: *const LlevDictionaryLimits,
) -> Result<LlevDictionaryLimits, (LlevStatus, String)> {
    let value = unsafe { raw.as_ref() }
        .ok_or((LlevStatus::NullPointer, "bincode limits are null".into()))?;
    if value.max_payload_bytes < 8 || value.max_term_bytes > value.max_total_term_bytes {
        return Err(invalid("invalid bincode resource ceilings"));
    }
    Ok(*value)
}

#[cfg(feature = "serialization")]
pub(super) struct BoundedWriter {
    pub(super) bytes: Vec<u8>,
    pub(super) max_bytes: usize,
}

#[cfg(feature = "serialization")]
impl Write for BoundedWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let requested = self
            .bytes
            .len()
            .checked_add(input.len())
            .ok_or_else(|| io::Error::other("bincode output length overflow"))?;
        if requested > self.max_bytes {
            return Err(io::Error::other("bincode output byte ceiling"));
        }
        self.bytes
            .try_reserve(input.len())
            .map_err(|_| io::Error::other("bincode output allocation failed"))?;
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "serialization")]
pub(super) fn read_u64(input: &[u8], offset: &mut usize) -> Result<u64, (LlevStatus, String)> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| invalid("bincode offset overflow"))?;
    let bytes: [u8; 8] = input
        .get(*offset..end)
        .ok_or_else(|| invalid("truncated bincode length"))?
        .try_into()
        .map_err(|_| invalid("truncated bincode length"))?;
    *offset = end;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(feature = "serialization")]
pub(super) fn preflight(
    input: &[u8],
    ceilings: LlevDictionaryLimits,
) -> Result<(), (LlevStatus, String)> {
    let mut offset = 0;
    let count = usize::try_from(read_u64(input, &mut offset)?).map_err(|_| {
        (
            LlevStatus::LimitExceeded,
            "bincode term count overflow".into(),
        )
    })?;
    if count > ceilings.max_terms {
        return Err((
            LlevStatus::LimitExceeded,
            "bincode term count ceiling".into(),
        ));
    }
    let mut total = 0usize;
    for _ in 0..count {
        let len = usize::try_from(read_u64(input, &mut offset)?).map_err(|_| {
            (
                LlevStatus::LimitExceeded,
                "bincode term length overflow".into(),
            )
        })?;
        total = total.checked_add(len).ok_or((
            LlevStatus::LimitExceeded,
            "bincode term bytes overflow".into(),
        ))?;
        if len > ceilings.max_term_bytes || total > ceilings.max_total_term_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "bincode term byte ceiling".into(),
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| invalid("bincode offset overflow"))?;
        let bytes = input
            .get(offset..end)
            .ok_or_else(|| invalid("truncated bincode term"))?;
        std::str::from_utf8(bytes).map_err(|_| invalid("bincode term is not UTF-8"))?;
        offset = end;
    }
    if offset != input.len() {
        return Err(invalid("bincode payload has trailing bytes"));
    }
    Ok(())
}

#[cfg(feature = "protobuf")]
fn preflight_graph(
    root: u64,
    declared_size: u64,
    finals: impl IntoIterator<Item = u64>,
    edges: impl IntoIterator<Item = (u64, u8, u64)>,
    ceilings: LlevDictionaryLimits,
) -> Result<(), (LlevStatus, String)> {
    use std::collections::{HashMap, HashSet};

    let expected = usize::try_from(declared_size).map_err(|_| {
        (
            LlevStatus::LimitExceeded,
            "protobuf term count overflow".into(),
        )
    })?;
    if expected > ceilings.max_terms {
        return Err((
            LlevStatus::LimitExceeded,
            "protobuf term count ceiling".into(),
        ));
    }
    let final_nodes: HashSet<u64> = finals.into_iter().collect();
    let mut adjacency: HashMap<u64, Vec<(u8, u64)>> = HashMap::new();
    for (source, label, target) in edges {
        adjacency.entry(source).or_default().push((label, target));
    }
    for branches in adjacency.values_mut() {
        branches.sort_unstable_by_key(|(label, _)| *label);
        if branches.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("protobuf graph has duplicate outgoing labels"));
        }
    }

    struct Frame {
        node: u64,
        next_edge: usize,
        entered: bool,
    }
    let mut frames = vec![Frame {
        node: root,
        next_edge: 0,
        entered: false,
    }];
    let mut active = HashSet::new();
    let mut path = Vec::new();
    let mut count = 0usize;
    let mut total = 0usize;
    while let Some(frame) = frames.last_mut() {
        if !frame.entered {
            if !active.insert(frame.node) {
                return Err(invalid("protobuf graph contains a reachable cycle"));
            }
            frame.entered = true;
            if final_nodes.contains(&frame.node) {
                std::str::from_utf8(&path)
                    .map_err(|_| invalid("protobuf terminal path is not UTF-8"))?;
                count = count.checked_add(1).ok_or((
                    LlevStatus::LimitExceeded,
                    "protobuf term count overflow".into(),
                ))?;
                total = total.checked_add(path.len()).ok_or((
                    LlevStatus::LimitExceeded,
                    "protobuf term bytes overflow".into(),
                ))?;
                if count > ceilings.max_terms || total > ceilings.max_total_term_bytes {
                    return Err((LlevStatus::LimitExceeded, "protobuf term ceiling".into()));
                }
            }
        }
        let branches = adjacency.get(&frame.node).map_or(&[][..], Vec::as_slice);
        if frame.next_edge == branches.len() {
            active.remove(&frame.node);
            frames.pop();
            if !frames.is_empty() {
                path.pop();
            }
            continue;
        }
        let (label, target) = branches[frame.next_edge];
        frame.next_edge += 1;
        if path.len() >= ceilings.max_term_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "protobuf path byte ceiling".into(),
            ));
        }
        path.push(label);
        frames.push(Frame {
            node: target,
            next_edge: 0,
            entered: false,
        });
    }
    if count != expected {
        return Err(invalid("protobuf declared term count does not match graph"));
    }
    Ok(())
}

#[cfg(feature = "protobuf")]
fn preflight_protobuf(
    input: &[u8],
    format_id: u32,
    ceilings: LlevDictionaryLimits,
) -> Result<(), (LlevStatus, String)> {
    use std::collections::HashSet;

    match format_id {
        PROTOBUF_V1 => {
            let graph = protobuf_wire::Dictionary::decode(input)
                .map_err(|error| invalid(format!("malformed protobuf V1: {error}")))?;
            let nodes: HashSet<u64> = graph.node_id.into_iter().collect();
            if !nodes.contains(&graph.root_id)
                || graph.final_node_id.iter().any(|node| !nodes.contains(node))
                || graph.edge.iter().any(|edge| {
                    !nodes.contains(&edge.source_id) || !nodes.contains(&edge.target_id)
                })
            {
                return Err(invalid("protobuf V1 graph refers to an undeclared node"));
            }
            let mut edges = Vec::new();
            edges.try_reserve(graph.edge.len()).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "protobuf edge allocation failed".into(),
                )
            })?;
            for edge in graph.edge {
                let label = u8::try_from(edge.label)
                    .map_err(|_| invalid("protobuf V1 edge label exceeds byte domain"))?;
                edges.push((edge.source_id, label, edge.target_id));
            }
            preflight_graph(
                graph.root_id,
                graph.size,
                graph.final_node_id,
                edges,
                ceilings,
            )
        }
        PROTOBUF_V2 => {
            let graph = protobuf_wire::DictionaryV2::decode(input)
                .map_err(|error| invalid(format!("malformed protobuf V2: {error}")))?;
            if graph.edge_data.len() % 3 != 0
                || usize::try_from(graph.edge_count).ok() != Some(graph.edge_data.len() / 3)
            {
                return Err(invalid("protobuf V2 edge count is inconsistent"));
            }
            let mut finals = Vec::new();
            finals
                .try_reserve(graph.final_node_delta.len())
                .map_err(|_| {
                    (
                        LlevStatus::LimitExceeded,
                        "protobuf final allocation failed".into(),
                    )
                })?;
            let mut node = 0_u64;
            for delta in graph.final_node_delta {
                node = node
                    .checked_add(delta)
                    .ok_or_else(|| invalid("protobuf V2 final-node delta overflow"))?;
                finals.push(node);
            }
            let mut edges = Vec::new();
            edges.try_reserve(graph.edge_data.len() / 3).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "protobuf edge allocation failed".into(),
                )
            })?;
            for edge in graph.edge_data.chunks_exact(3) {
                let label = u8::try_from(edge[1])
                    .map_err(|_| invalid("protobuf V2 edge label exceeds byte domain"))?;
                edges.push((edge[0], label, edge[2]));
            }
            preflight_graph(graph.root_id, graph.size, finals, edges, ceilings)
        }
        PROTOBUF_DAT_V1 => {
            let graph = protobuf_wire::DoubleArrayTrie::decode(input)
                .map_err(|error| invalid(format!("malformed DAT protobuf: {error}")))?;
            let expected = usize::try_from(graph.term_count)
                .map_err(|_| (LlevStatus::LimitExceeded, "DAT term count overflow".into()))?;
            if expected > ceilings.max_terms {
                return Err((LlevStatus::LimitExceeded, "DAT term count ceiling".into()));
            }
            let payload = graph.edge_data;
            if !payload.starts_with(b"LDT1") {
                return Err(invalid("DAT term payload has invalid magic"));
            }
            let mut offset = 4usize;
            let mut count = 0usize;
            let mut total = 0usize;
            while offset < payload.len() {
                let end = offset
                    .checked_add(4)
                    .ok_or_else(|| invalid("DAT offset overflow"))?;
                let length: [u8; 4] = payload
                    .get(offset..end)
                    .ok_or_else(|| invalid("truncated DAT term length"))?
                    .try_into()
                    .map_err(|_| invalid("truncated DAT term length"))?;
                offset = end;
                let len = u32::from_le_bytes(length) as usize;
                total = total
                    .checked_add(len)
                    .ok_or((LlevStatus::LimitExceeded, "DAT term bytes overflow".into()))?;
                count = count
                    .checked_add(1)
                    .ok_or((LlevStatus::LimitExceeded, "DAT term count overflow".into()))?;
                if count > ceilings.max_terms
                    || len > ceilings.max_term_bytes
                    || total > ceilings.max_total_term_bytes
                {
                    return Err((LlevStatus::LimitExceeded, "DAT term ceiling".into()));
                }
                let end = offset
                    .checked_add(len)
                    .ok_or_else(|| invalid("DAT offset overflow"))?;
                let term = payload
                    .get(offset..end)
                    .ok_or_else(|| invalid("truncated DAT term bytes"))?;
                std::str::from_utf8(term).map_err(|_| invalid("DAT term is not UTF-8"))?;
                offset = end;
            }
            if count != expected {
                return Err(invalid("DAT declared term count does not match payload"));
            }
            Ok(())
        }
        _ => Err(invalid("unsupported protobuf format ID")),
    }
}

#[cfg(feature = "compression")]
fn decompress_gzip(
    input: &[u8],
    ceilings: LlevDictionaryLimits,
) -> Result<Vec<u8>, (LlevStatus, String)> {
    use flate2::bufread::GzDecoder;

    let limit = ceilings.max_payload_bytes.saturating_add(1) as u64;
    let mut bounded = GzDecoder::new(input).take(limit);
    let mut bytes = Vec::new();
    bounded
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("malformed gzip stream: {error}")))?;
    if bytes.len() > ceilings.max_payload_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "gzip inflated byte ceiling".into(),
        ));
    }
    let decoder = bounded.into_inner();
    if !decoder.into_inner().is_empty() {
        return Err(invalid("gzip stream has trailing compressed bytes"));
    }
    Ok(bytes)
}

#[cfg(feature = "serialization")]
fn require_format(format_id: u32) -> Result<(), (LlevStatus, String)> {
    match format_id {
        BINCODE_V1 => Ok(()),
        PROTOBUF_V1 | PROTOBUF_V2 | PROTOBUF_DAT_V1 if !cfg!(feature = "protobuf") => Err((
            LlevStatus::Unsupported,
            "dictionary protobuf requires protobuf feature".into(),
        )),
        GZIP_BINCODE_V1 if !cfg!(feature = "compression") => Err((
            LlevStatus::Unsupported,
            "dictionary gzip requires compression feature".into(),
        )),
        GZIP_PROTOBUF_V1 | GZIP_PROTOBUF_V2
            if !cfg!(all(feature = "compression", feature = "protobuf")) =>
        {
            Err((
                LlevStatus::Unsupported,
                "dictionary gzip protobuf requires compression and protobuf features".into(),
            ))
        }
        PROTOBUF_V1 | PROTOBUF_V2 | PROTOBUF_DAT_V1 | GZIP_BINCODE_V1 | GZIP_PROTOBUF_V1
        | GZIP_PROTOBUF_V2 => Ok(()),
        _ => Err(invalid("unsupported dictionary format ID")),
    }
}

#[cfg(feature = "serialization")]
fn preflight_format(
    input: &[u8],
    format_id: u32,
    ceilings: LlevDictionaryLimits,
) -> Result<(), (LlevStatus, String)> {
    require_format(format_id)?;
    if input.len() > ceilings.max_payload_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "dictionary payload byte ceiling".into(),
        ));
    }
    match format_id {
        BINCODE_V1 => preflight(input, ceilings),
        #[cfg(feature = "protobuf")]
        PROTOBUF_V1 | PROTOBUF_V2 | PROTOBUF_DAT_V1 => {
            preflight_protobuf(input, format_id, ceilings)
        }
        #[cfg(feature = "compression")]
        GZIP_BINCODE_V1 => preflight(&decompress_gzip(input, ceilings)?, ceilings),
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V1 | GZIP_PROTOBUF_V2 => {
            let inner = if format_id == GZIP_PROTOBUF_V1 {
                PROTOBUF_V1
            } else {
                PROTOBUF_V2
            };
            preflight_protobuf(&decompress_gzip(input, ceilings)?, inner, ceilings)
        }
        _ => Err(invalid("unsupported dictionary format ID")),
    }
}

#[cfg(feature = "serialization")]
fn serialize_native(
    dictionary: &DoubleArrayTrie,
    format_id: u32,
    writer: &mut BoundedWriter,
) -> Result<(), (LlevStatus, String)> {
    require_format(format_id)?;
    let result = match format_id {
        BINCODE_V1 => BincodeSerializer::serialize(dictionary, writer),
        #[cfg(feature = "protobuf")]
        PROTOBUF_V1 => ProtobufSerializer::serialize(dictionary, writer),
        #[cfg(feature = "protobuf")]
        PROTOBUF_V2 => OptimizedProtobufSerializer::serialize(dictionary, writer),
        #[cfg(feature = "protobuf")]
        PROTOBUF_DAT_V1 => DatProtobufSerializer::serialize_dat(dictionary, writer),
        #[cfg(feature = "compression")]
        GZIP_BINCODE_V1 => GzipSerializer::<BincodeSerializer>::serialize(dictionary, writer),
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V1 => GzipSerializer::<ProtobufSerializer>::serialize(dictionary, writer),
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V2 => {
            GzipSerializer::<OptimizedProtobufSerializer>::serialize(dictionary, writer)
        }
        _ => return Err(invalid("unsupported dictionary format ID")),
    };
    result.map_err(|error| {
        (
            LlevStatus::LimitExceeded,
            format!("bounded dictionary serialization failed: {error}"),
        )
    })
}

#[cfg(feature = "serialization")]
fn deserialize_native(
    input: &[u8],
    format_id: u32,
) -> Result<DoubleArrayTrie, (LlevStatus, String)> {
    require_format(format_id)?;
    let result = match format_id {
        BINCODE_V1 => BincodeSerializer::deserialize(input),
        #[cfg(feature = "protobuf")]
        PROTOBUF_V1 => ProtobufSerializer::deserialize(input),
        #[cfg(feature = "protobuf")]
        PROTOBUF_V2 => OptimizedProtobufSerializer::deserialize(input),
        #[cfg(feature = "protobuf")]
        PROTOBUF_DAT_V1 => DatProtobufSerializer::deserialize_dat(input),
        #[cfg(feature = "compression")]
        GZIP_BINCODE_V1 => GzipSerializer::<BincodeSerializer>::deserialize(input),
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V1 => GzipSerializer::<ProtobufSerializer>::deserialize(input),
        #[cfg(all(feature = "compression", feature = "protobuf"))]
        GZIP_PROTOBUF_V2 => GzipSerializer::<OptimizedProtobufSerializer>::deserialize(input),
        _ => return Err(invalid("unsupported dictionary format ID")),
    };
    result.map_err(|error| invalid(format!("native dictionary decode failed: {error}")))
}

#[cfg(feature = "serialization")]
pub(super) unsafe fn input_strings(
    source: *const LlevUtf8Slice,
    count: usize,
    ceilings: LlevDictionaryLimits,
) -> Result<Vec<String>, (LlevStatus, String)> {
    if count > ceilings.max_terms {
        return Err((LlevStatus::LimitExceeded, "string count ceiling".into()));
    }
    let slices = if count == 0 {
        &[][..]
    } else {
        if source.is_null() {
            return Err((LlevStatus::NullPointer, "string array is null".into()));
        }
        std::slice::from_raw_parts(source, count)
    };
    let mut owned = Vec::new();
    owned
        .try_reserve(count)
        .map_err(|_| (LlevStatus::LimitExceeded, "string allocation failed".into()))?;
    let mut total = 0usize;
    for entry in slices {
        total = total
            .checked_add(entry.len)
            .ok_or((LlevStatus::LimitExceeded, "string bytes overflow".into()))?;
        if entry.len > ceilings.max_term_bytes || total > ceilings.max_total_term_bytes {
            return Err((LlevStatus::LimitExceeded, "string byte ceiling".into()));
        }
        let bytes = if entry.len == 0 {
            &[][..]
        } else {
            if entry.data.is_null() {
                return Err((LlevStatus::NullPointer, "string data is null".into()));
            }
            std::slice::from_raw_parts(entry.data, entry.len)
        };
        let value = std::str::from_utf8(bytes).map_err(|_| invalid("string is not UTF-8"))?;
        owned.push(value.to_owned());
    }
    Ok(owned)
}

/// Serialize accepted UTF-8 terms with a selected native binary format.
///
/// # Safety
/// Nonempty input slices and the output pointer must be valid. The output is
/// released with `llev_owned_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn llev_dictionary_serialize(
    format_id: u32,
    terms: *const LlevUtf8Slice,
    term_count: usize,
    raw_limits: *const LlevDictionaryLimits,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    boundary(|| {
        let output = out_bytes
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "dictionary output is null".into()))?;
        *output = LlevOwnedBytes::default();
        #[cfg(feature = "serialization")]
        {
            require_format(format_id)?;
            let ceilings = limits(raw_limits)?;
            let dictionary =
                DoubleArrayTrie::from_terms(input_strings(terms, term_count, ceilings)?);
            // Bincode is a fixed-width Vec<String>. Compute its exact native
            // size first: its writer adapter can otherwise report success after
            // a rejected bounded write.
            let bincode_expected = if format_id == BINCODE_V1 || format_id == GZIP_BINCODE_V1 {
                let native_terms = extract_terms(&dictionary);
                let mut expected = 8usize;
                for term in &native_terms {
                    expected = expected
                        .checked_add(8)
                        .and_then(|size| size.checked_add(term.len()))
                        .ok_or((
                            LlevStatus::LimitExceeded,
                            "bincode payload length overflow".into(),
                        ))?;
                }
                if expected > ceilings.max_payload_bytes {
                    return Err((
                        LlevStatus::LimitExceeded,
                        "bincode payload byte ceiling".into(),
                    ));
                }
                Some(expected)
            } else {
                None
            };
            let mut writer = BoundedWriter {
                bytes: Vec::new(),
                max_bytes: ceilings.max_payload_bytes,
            };
            serialize_native(&dictionary, format_id, &mut writer)?;
            if let Some(expected) = bincode_expected {
                if format_id == BINCODE_V1 && writer.bytes.len() != expected {
                    return Err(invalid(
                        "native bincode encoder returned an incomplete payload",
                    ));
                }
                #[cfg(feature = "compression")]
                if format_id == GZIP_BINCODE_V1
                    && decompress_gzip(&writer.bytes, ceilings)?.len() != expected
                {
                    return Err(invalid(
                        "native gzip bincode encoder returned an incomplete payload",
                    ));
                }
            }
            preflight_format(&writer.bytes, format_id, ceilings)?;
            let boxed = writer.bytes.into_boxed_slice();
            output.len = boxed.len();
            output.data = Box::into_raw(boxed) as *mut u8;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (format_id, terms, term_count, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "dictionary binary formats require serialization feature".into(),
            ))
        }
    })
}

/// Decode native dictionary bytes into an independently owned term snapshot.
///
/// # Safety
/// Nonempty input and output pointer must be valid; free the handle once.
#[no_mangle]
pub unsafe extern "C" fn llev_dictionary_deserialize(
    format_id: u32,
    data: *const u8,
    len: usize,
    raw_limits: *const LlevDictionaryLimits,
    out_terms: *mut *mut LlevDecodedDictionaryTerms,
) -> LlevStatus {
    boundary(|| {
        let output = out_terms.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded terms output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        #[cfg(feature = "serialization")]
        {
            require_format(format_id)?;
            let ceilings = limits(raw_limits)?;
            if len > ceilings.max_payload_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "dictionary payload byte ceiling".into(),
                ));
            }
            let input = if len == 0 {
                &[][..]
            } else {
                if data.is_null() {
                    return Err((LlevStatus::NullPointer, "dictionary data is null".into()));
                }
                std::slice::from_raw_parts(data, len)
            };
            preflight_format(input, format_id, ceilings)?;
            let dictionary = deserialize_native(input, format_id)?;
            let terms = extract_terms(&dictionary);
            let total = terms
                .iter()
                .try_fold(0usize, |sum, term| sum.checked_add(term.len()))
                .ok_or((
                    LlevStatus::LimitExceeded,
                    "decoded term bytes overflow".into(),
                ))?;
            if terms.len() > ceilings.max_terms
                || total > ceilings.max_total_term_bytes
                || terms
                    .iter()
                    .any(|term| term.len() > ceilings.max_term_bytes)
            {
                return Err((
                    LlevStatus::LimitExceeded,
                    "decoded dictionary term ceiling".into(),
                ));
            }
            *output = Box::into_raw(Box::new(LlevDecodedDictionaryTerms { terms }));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "serialization"))]
        {
            let _ = (format_id, data, len, raw_limits, output);
            Err((
                LlevStatus::Unsupported,
                "dictionary binary formats require serialization feature".into(),
            ))
        }
    })
}

/// Return the number of terms in the decoded immutable snapshot.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_dictionary_terms_len(
    terms: *const LlevDecodedDictionaryTerms,
    out_len: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let source = terms
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "decoded terms are null".into()))?;
        let output = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded length output is null".into(),
        ))?;
        *output = source.terms.len();
        Ok(LlevStatus::Ok)
    })
}

/// Return a borrowed UTF-8 term view valid until the snapshot is freed.
///
/// # Safety
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_dictionary_term_at(
    terms: *const LlevDecodedDictionaryTerms,
    index: usize,
    out_term: *mut LlevUtf8Slice,
) -> LlevStatus {
    boundary(|| {
        let source = terms
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "decoded terms are null".into()))?;
        let output = out_term.as_mut().ok_or((
            LlevStatus::NullPointer,
            "decoded term output is null".into(),
        ))?;
        let value = source.terms.get(index).ok_or((
            LlevStatus::InvalidArgument,
            "decoded term index is out of range".into(),
        ))?;
        *output = LlevUtf8Slice {
            data: value.as_ptr(),
            len: value.len(),
        };
        Ok(LlevStatus::Ok)
    })
}

/// Free a decoded immutable term snapshot; null is a no-op.
///
/// # Safety
/// Pointer must be a live handle from this library and freed only once.
#[no_mangle]
pub unsafe extern "C" fn llev_decoded_dictionary_terms_free(
    terms: *mut LlevDecodedDictionaryTerms,
) {
    if !terms.is_null() {
        drop(Box::from_raw(terms));
    }
}
