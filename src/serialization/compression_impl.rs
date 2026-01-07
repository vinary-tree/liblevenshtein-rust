//! Gzip compression wrapper for serializers.

use libdictenstein::{Dictionary, DictionaryNode};
use std::io::{Read, Write};

use super::{DictionaryFromTerms, DictionarySerializer, SerializationError};

/// Gzip-compressed serializer wrapper.
///
/// This wrapper applies gzip compression to any underlying serializer,
/// reducing file size by 40-60% with minimal performance cost.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::serialization::{GzipSerializer, BincodeSerializer};
/// use std::fs::File;
///
/// let dict = PathMapDictionary::from_terms(vec!["test", "testing"]);
///
/// // Serialize with gzip compression
/// let file = File::create("dict.bin.gz")?;
/// GzipSerializer::<BincodeSerializer>::serialize(&dict, file)?;
///
/// // Deserialize
/// let file = File::open("dict.bin.gz")?;
/// let loaded: PathMapDictionary =
///     GzipSerializer::<BincodeSerializer>::deserialize(file)?;
/// ```
pub struct GzipSerializer<S> {
    _inner: std::marker::PhantomData<S>,
}

impl<S: DictionarySerializer> DictionarySerializer for GzipSerializer<S> {
    fn serialize<D, W>(dict: &D, writer: W) -> Result<(), SerializationError>
    where
        D: Dictionary,
        D::Node: DictionaryNode<Unit = u8>,
        W: Write,
    {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut encoder = GzEncoder::new(writer, Compression::default());
        S::serialize(dict, &mut encoder)?;
        encoder.finish().map_err(SerializationError::Io)?;
        Ok(())
    }

    fn deserialize<D, R>(reader: R) -> Result<D, SerializationError>
    where
        D: DictionaryFromTerms,
        R: Read,
    {
        use flate2::read::GzDecoder;

        let decoder = GzDecoder::new(reader);
        S::deserialize(decoder)
    }
}
