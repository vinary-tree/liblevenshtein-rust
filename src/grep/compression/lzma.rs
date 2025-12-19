//! XZ/LZMA decompression support.
//!
//! This module provides streaming XZ and LZMA decompression using the `xz2` crate.

use std::io::{self, Read};

use xz2::read::XzDecoder;

/// Streaming XZ/LZMA decompressor.
///
/// Wraps a reader and provides decompressed data through the `Read` trait.
/// Supports both XZ (.xz) and legacy LZMA (.lzma) formats.
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::io::Read;
/// use liblevenshtein::grep::compression::lzma::XzDecompressor;
///
/// let file = File::open("data.xz")?;
/// let mut decompressor = XzDecompressor::new(file);
///
/// let mut content = String::new();
/// decompressor.read_to_string(&mut content)?;
/// ```
pub struct XzDecompressor<R: Read> {
    decoder: XzDecoder<R>,
}

impl<R: Read> XzDecompressor<R> {
    /// Create a new XZ decompressor wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            decoder: XzDecoder::new(reader),
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

    /// Get the total number of bytes read from the compressed stream.
    pub fn total_in(&self) -> u64 {
        self.decoder.total_in()
    }

    /// Get the total number of decompressed bytes produced.
    pub fn total_out(&self) -> u64 {
        self.decoder.total_out()
    }
}

impl<R: Read> Read for XzDecompressor<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.decoder.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use xz2::write::XzEncoder;

    fn compress_data(data: &[u8]) -> Vec<u8> {
        let mut encoder = XzEncoder::new(Vec::new(), 6);
        encoder.write_all(data).expect("write should succeed");
        encoder.finish().expect("finish should succeed")
    }

    #[test]
    fn test_decompress_simple() {
        let original = b"Hello, World!";
        let compressed = compress_data(original);

        let mut decompressor = XzDecompressor::new(&compressed[..]);
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

        let mut decompressor = XzDecompressor::new(&compressed[..]);
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

        let mut decompressor = XzDecompressor::new(&compressed[..]);
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

        let mut decompressor = XzDecompressor::new(&compressed[..]);
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

    #[test]
    fn test_total_counts() {
        let original = b"Hello, World!";
        let compressed = compress_data(original);

        let mut decompressor = XzDecompressor::new(&compressed[..]);
        let mut result = Vec::new();
        decompressor
            .read_to_end(&mut result)
            .expect("read should succeed");

        assert!(decompressor.total_in() > 0);
        assert_eq!(decompressor.total_out(), original.len() as u64);
    }
}
