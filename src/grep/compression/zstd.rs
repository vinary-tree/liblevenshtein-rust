//! Zstandard decompression support.
//!
//! This module provides streaming zstd decompression using the `zstd` crate.

use std::io::{self, BufReader, Read};

/// Streaming zstd decompressor.
///
/// Wraps a reader and provides decompressed data through the `Read` trait.
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::io::Read;
/// use liblevenshtein::grep::compression::zstd::ZstdDecompressor;
///
/// let file = File::open("data.zst")?;
/// let mut decompressor = ZstdDecompressor::new(file)?;
///
/// let mut content = String::new();
/// decompressor.read_to_string(&mut content)?;
/// ```
pub struct ZstdDecompressor<'a, R: Read> {
    decoder: zstd::stream::Decoder<'a, BufReader<R>>,
}

impl<'a, R: Read> ZstdDecompressor<'a, R> {
    /// Create a new zstd decompressor wrapping the given reader.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder cannot be initialized.
    pub fn new(reader: R) -> io::Result<Self> {
        let decoder = zstd::stream::Decoder::new(reader)?;
        Ok(Self { decoder })
    }

    /// Create a new zstd decompressor with a custom buffer capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder cannot be initialized.
    pub fn with_buffer_size(reader: R, capacity: usize) -> io::Result<Self> {
        let decoder = zstd::stream::Decoder::with_buffer(BufReader::with_capacity(capacity, reader))?;
        Ok(Self { decoder })
    }
}

impl<R: Read> Read for ZstdDecompressor<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.decoder.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn compress_data(data: &[u8]) -> Vec<u8> {
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3).expect("encoder should create");
        encoder.write_all(data).expect("write should succeed");
        encoder.finish().expect("finish should succeed")
    }

    #[test]
    fn test_decompress_simple() {
        let original = b"Hello, World!";
        let compressed = compress_data(original);

        let mut decompressor =
            ZstdDecompressor::new(&compressed[..]).expect("decompressor should create");
        let mut result = Vec::new();
        decompressor
            .read_to_end(&mut result)
            .expect("read should succeed");

        assert_eq!(result, original);
    }

    #[test]
    fn test_decompress_empty() {
        let original = b"";
        let compressed = compress_data(original);

        let mut decompressor =
            ZstdDecompressor::new(&compressed[..]).expect("decompressor should create");
        let mut result = Vec::new();
        decompressor
            .read_to_end(&mut result)
            .expect("read should succeed");

        assert_eq!(result, original);
    }

    #[test]
    fn test_decompress_large() {
        let original: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let compressed = compress_data(&original);

        let mut decompressor =
            ZstdDecompressor::new(&compressed[..]).expect("decompressor should create");
        let mut result = Vec::new();
        decompressor
            .read_to_end(&mut result)
            .expect("read should succeed");

        assert_eq!(result, original);
    }

    #[test]
    fn test_decompress_chunked() {
        let original = b"Hello, World! This is a test of chunked reading.";
        let compressed = compress_data(original);

        let mut decompressor =
            ZstdDecompressor::new(&compressed[..]).expect("decompressor should create");
        let mut result = Vec::new();
        let mut buf = [0u8; 8]; // Small buffer to force multiple reads

        loop {
            let n = decompressor.read(&mut buf).expect("read should succeed");
            if n == 0 {
                break;
            }
            result.extend_from_slice(&buf[..n]);
        }

        assert_eq!(result, original);
    }
}
