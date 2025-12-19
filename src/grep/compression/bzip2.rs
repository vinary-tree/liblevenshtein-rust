//! Bzip2 decompression support.
//!
//! This module provides streaming bzip2 decompression using the `bzip2` crate.

use std::io::{self, Read};

use bzip2::read::BzDecoder;

/// Streaming bzip2 decompressor.
///
/// Wraps a reader and provides decompressed data through the `Read` trait.
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::io::Read;
/// use liblevenshtein::grep::compression::bzip2::Bzip2Decompressor;
///
/// let file = File::open("data.bz2")?;
/// let mut decompressor = Bzip2Decompressor::new(file);
///
/// let mut content = String::new();
/// decompressor.read_to_string(&mut content)?;
/// ```
pub struct Bzip2Decompressor<R: Read> {
    decoder: BzDecoder<R>,
}

impl<R: Read> Bzip2Decompressor<R> {
    /// Create a new bzip2 decompressor wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            decoder: BzDecoder::new(reader),
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
}

impl<R: Read> Read for Bzip2Decompressor<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.decoder.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use std::io::Write;

    fn compress_data(data: &[u8]) -> Vec<u8> {
        let mut encoder = BzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).expect("write should succeed");
        encoder.finish().expect("finish should succeed")
    }

    #[test]
    fn test_decompress_simple() {
        let original = b"Hello, World!";
        let compressed = compress_data(original);

        let mut decompressor = Bzip2Decompressor::new(&compressed[..]);
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

        let mut decompressor = Bzip2Decompressor::new(&compressed[..]);
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

        let mut decompressor = Bzip2Decompressor::new(&compressed[..]);
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

        let mut decompressor = Bzip2Decompressor::new(&compressed[..]);
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
