//! Gzip decompression support.
//!
//! This module provides streaming gzip decompression using the `flate2` crate.

use std::io::{self, Read};

use flate2::read::GzDecoder;

/// Streaming gzip decompressor.
///
/// Wraps a reader and provides decompressed data through the `Read` trait.
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::io::Read;
/// use liblevenshtein::grep::compression::gzip::GzipDecompressor;
///
/// let file = File::open("data.gz")?;
/// let mut decompressor = GzipDecompressor::new(file);
///
/// let mut content = String::new();
/// decompressor.read_to_string(&mut content)?;
/// ```
pub struct GzipDecompressor<R: Read> {
    decoder: GzDecoder<R>,
}

impl<R: Read> GzipDecompressor<R> {
    /// Create a new gzip decompressor wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            decoder: GzDecoder::new(reader),
        }
    }

    /// Get a reference to the underlying reader.
    pub fn get_ref(&self) -> &R {
        self.decoder.get_ref()
    }

    /// Get a mutable reference to the underlying reader.
    pub fn get_mut(&mut self) -> &mut R {
        self.decoder.get_mut()
    }

    /// Consume the decompressor and return the underlying reader.
    pub fn into_inner(self) -> R {
        self.decoder.into_inner()
    }

    /// Get the gzip header if available.
    ///
    /// Returns `None` if the header hasn't been read yet or is invalid.
    pub fn header(&self) -> Option<&flate2::GzHeader> {
        self.decoder.header()
    }
}

impl<R: Read> Read for GzipDecompressor<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.decoder.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn compress_data(data: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).expect("write should succeed");
        encoder.finish().expect("finish should succeed")
    }

    #[test]
    fn test_decompress_simple() {
        let original = b"Hello, World!";
        let compressed = compress_data(original);

        let mut decompressor = GzipDecompressor::new(&compressed[..]);
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

        let mut decompressor = GzipDecompressor::new(&compressed[..]);
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

        let mut decompressor = GzipDecompressor::new(&compressed[..]);
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

        let mut decompressor = GzipDecompressor::new(&compressed[..]);
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
