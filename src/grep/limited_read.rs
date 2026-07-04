//! Bounded in-memory reads for grep inputs.

use std::io::{self, Read};

pub(crate) const DEFAULT_INITIAL_READ_CAPACITY: usize = 1024;
pub(crate) const MAX_INITIAL_READ_CAPACITY: usize = 1024 * 1024;

const READ_CHUNK_SIZE: usize = 8 * 1024;

pub(crate) fn initial_read_capacity(size: Option<u64>) -> usize {
    match size {
        Some(size) => usize::try_from(size)
            .unwrap_or(usize::MAX)
            .min(MAX_INITIAL_READ_CAPACITY),
        None => DEFAULT_INITIAL_READ_CAPACITY,
    }
}

pub(crate) fn bounded_size_hint(size: Option<u64>, max_size: Option<u64>) -> Option<u64> {
    match (size, max_size) {
        (Some(size), Some(max_size)) => Some(size.min(max_size)),
        (Some(size), None) => Some(size),
        (None, Some(max_size)) => Some(max_size),
        (None, None) => None,
    }
}

pub(crate) fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(crate) fn len_exceeds_limit(len: usize, max_size: u64) -> bool {
    u64::try_from(len).map_or(true, |len| len > max_size)
}

fn read_len_after_chunk_exceeds_limit(
    content_len: usize,
    bytes_read: usize,
    max_size: u64,
) -> Option<u64> {
    let Some(new_len) = content_len.checked_add(bytes_read) else {
        return Some(u64::MAX);
    };

    len_exceeds_limit(new_len, max_size).then(|| usize_to_u64_saturating(new_len))
}

pub(crate) fn file_too_large_error(label: &str, size: u64, max_size: u64) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{label} too large: {size} bytes (limit: {max_size} bytes)"),
    )
}

pub(crate) fn validate_size(
    label: &str,
    size: Option<u64>,
    max_size: Option<u64>,
) -> io::Result<()> {
    let Some((size, max_size)) = size.zip(max_size) else {
        return Ok(());
    };

    if size > max_size {
        return Err(file_too_large_error(label, size, max_size));
    }
    Ok(())
}

pub(crate) fn read_to_end_limited<R: Read + ?Sized>(
    reader: &mut R,
    content: &mut Vec<u8>,
    max_size: Option<u64>,
    label: &str,
) -> io::Result<()> {
    let Some(max_size) = max_size else {
        reader.read_to_end(content)?;
        return Ok(());
    };

    let mut chunk = [0; READ_CHUNK_SIZE];
    loop {
        let bytes_read = reader.read(&mut chunk)?;
        if bytes_read == 0 {
            return Ok(());
        }

        if let Some(new_len) =
            read_len_after_chunk_exceeds_limit(content.len(), bytes_read, max_size)
        {
            return Err(file_too_large_error(label, new_len, max_size));
        }
        content.extend_from_slice(&chunk[..bytes_read]);
    }
}

pub(crate) fn read_to_vec_limited<R: Read + ?Sized>(
    reader: &mut R,
    size: Option<u64>,
    max_size: Option<u64>,
    label: &str,
) -> io::Result<Vec<u8>> {
    validate_size(label, size, max_size)?;
    let mut buf = Vec::with_capacity(initial_read_capacity(bounded_size_hint(size, max_size)));
    read_to_end_limited(reader, &mut buf, max_size, label)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_exceeds_limit_uses_checked_usize_to_u64_conversion() {
        assert!(!len_exceeds_limit(5, 5));
        assert!(len_exceeds_limit(6, 5));

        if usize::BITS > u64::BITS {
            assert!(len_exceeds_limit(usize::MAX, u64::MAX));
        } else {
            assert!(!len_exceeds_limit(usize::MAX, u64::MAX));
        }
    }

    #[test]
    fn read_len_after_chunk_detects_limit_and_usize_overflow() {
        assert_eq!(read_len_after_chunk_exceeds_limit(2, 3, 5), None);
        assert_eq!(read_len_after_chunk_exceeds_limit(2, 4, 5), Some(6));
        assert_eq!(
            read_len_after_chunk_exceeds_limit(usize::MAX, 1, u64::MAX),
            Some(u64::MAX)
        );
    }
}
