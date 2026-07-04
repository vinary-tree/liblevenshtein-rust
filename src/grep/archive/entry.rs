//! Archive entry abstraction.
//!
//! This module provides types for representing entries within archives.

use std::borrow::Cow;
use std::io::Read;
use std::time::SystemTime;

use crate::grep::limited_read;

pub(crate) fn read_to_vec_limited<R: Read + ?Sized>(
    reader: &mut R,
    size: Option<u64>,
    max_size: Option<u64>,
) -> std::io::Result<Vec<u8>> {
    limited_read::read_to_vec_limited(reader, size, max_size, "archive entry")
}

fn normalized_prefix_len(path: &str) -> usize {
    let mut prefix_len = 0;
    loop {
        let remaining = &path[prefix_len..];
        let without_slashes = remaining.trim_start_matches('/');
        let slash_prefix_len = remaining.len() - without_slashes.len();
        if slash_prefix_len > 0 {
            prefix_len += slash_prefix_len;
            continue;
        }

        if path[prefix_len..].starts_with("./") {
            prefix_len += 2;
            continue;
        }

        break;
    }
    prefix_len
}

pub(crate) fn normalize_archive_path(path: &str) -> Cow<'_, str> {
    if path.contains('\\') {
        return Cow::Owned(normalize_archive_path_owned(path.to_string()));
    }

    let prefix_len = normalized_prefix_len(path);
    if prefix_len == 0 {
        Cow::Borrowed(path)
    } else {
        Cow::Owned(path[prefix_len..].to_string())
    }
}

pub(crate) fn normalize_archive_path_owned(mut path: String) -> String {
    if path.contains('\\') {
        path = path.replace('\\', "/");
    }

    let prefix_len = normalized_prefix_len(&path);
    if prefix_len > 0 {
        path.drain(..prefix_len);
    }
    path
}

/// Metadata about an entry within an archive.
#[derive(Debug, Clone)]
pub struct ArchiveEntryMeta {
    /// Path within the archive (uses forward slashes).
    pub path: String,

    /// Size in bytes (uncompressed), if known.
    pub size: Option<u64>,

    /// Entry type.
    pub entry_type: EntryType,

    /// Modification time, if available.
    pub mtime: Option<SystemTime>,

    /// Unix permissions mode (if available).
    pub mode: Option<u32>,
}

/// Type of archive entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryType {
    /// Regular file.
    File,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
    /// Hard link.
    Hardlink,
    /// Other/unknown type.
    Other,
}

impl ArchiveEntryMeta {
    /// Create metadata for a regular file.
    pub fn file(path: impl Into<String>, size: Option<u64>) -> Self {
        Self {
            path: path.into(),
            size,
            entry_type: EntryType::File,
            mtime: None,
            mode: None,
        }
    }

    /// Create metadata for a directory.
    pub fn directory(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            size: None,
            entry_type: EntryType::Directory,
            mtime: None,
            mode: None,
        }
    }

    /// Check if this entry is a regular file.
    pub fn is_file(&self) -> bool {
        matches!(self.entry_type, EntryType::File)
    }

    /// Check if this entry is a directory.
    pub fn is_dir(&self) -> bool {
        matches!(self.entry_type, EntryType::Directory)
    }

    /// Check if this entry is a symlink.
    pub fn is_symlink(&self) -> bool {
        matches!(self.entry_type, EntryType::Symlink)
    }

    /// Get the file name (last component of path).
    pub fn file_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    /// Get the parent directory path.
    pub fn parent(&self) -> Option<&str> {
        let trimmed = self.path.trim_end_matches('/');
        trimmed.rfind('/').map(|pos| &trimmed[..pos])
    }

    /// Get the file extension (if any).
    pub fn extension(&self) -> Option<&str> {
        let name = self.file_name();
        let dot_pos = name.rfind('.')?;
        if dot_pos == 0 || dot_pos == name.len() - 1 {
            None
        } else {
            Some(&name[dot_pos + 1..])
        }
    }

    /// Normalize the path (remove leading slashes, convert backslashes).
    pub fn normalize_path(&mut self) {
        self.path = normalize_archive_path_owned(std::mem::take(&mut self.path));
    }
}

impl std::fmt::Display for ArchiveEntryMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path)?;
        if let Some(size) = self.size {
            write!(f, " ({} bytes)", size)?;
        }
        Ok(())
    }
}

impl EntryType {
    /// Get a character representation (like `ls -l`).
    pub fn as_char(&self) -> char {
        match self {
            EntryType::File => '-',
            EntryType::Directory => 'd',
            EntryType::Symlink => 'l',
            EntryType::Hardlink => 'h',
            EntryType::Other => '?',
        }
    }
}

/// A readable archive entry with its content.
///
/// The entry content is available through the `Read` trait.
pub struct ArchiveEntry<'a> {
    /// Entry metadata.
    pub meta: ArchiveEntryMeta,

    /// Reader for entry content.
    reader: Box<dyn Read + 'a>,
}

impl<'a> ArchiveEntry<'a> {
    /// Create a new archive entry.
    pub fn new(meta: ArchiveEntryMeta, reader: impl Read + 'a) -> Self {
        Self {
            meta,
            reader: Box::new(reader),
        }
    }

    /// Get the entry path.
    pub fn path(&self) -> &str {
        &self.meta.path
    }

    /// Get the entry size (if known).
    pub fn size(&self) -> Option<u64> {
        self.meta.size
    }

    /// Check if this entry is a regular file.
    pub fn is_file(&self) -> bool {
        self.meta.is_file()
    }

    /// Consume the entry and return just the reader.
    pub fn into_reader(self) -> Box<dyn Read + 'a> {
        self.reader
    }

    /// Read all content as bytes.
    pub fn read_to_vec(&mut self) -> std::io::Result<Vec<u8>> {
        self.read_to_vec_limited(None)
    }

    /// Read all content as bytes, enforcing a maximum byte length if provided.
    pub fn read_to_vec_limited(&mut self, max_size: Option<u64>) -> std::io::Result<Vec<u8>> {
        read_to_vec_limited(&mut self.reader, self.meta.size, max_size)
    }

    /// Read all content as a UTF-8 string.
    pub fn read_to_string(&mut self) -> std::io::Result<String> {
        let bytes = self.read_to_vec()?;
        String::from_utf8(bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid UTF-8: {}", e),
            )
        })
    }
}

impl Read for ArchiveEntry<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_meta_file() {
        let meta = ArchiveEntryMeta::file("path/to/file.txt", Some(1024));
        assert!(meta.is_file());
        assert!(!meta.is_dir());
        assert_eq!(meta.file_name(), "file.txt");
        assert_eq!(meta.parent(), Some("path/to"));
        assert_eq!(meta.extension(), Some("txt"));
    }

    #[test]
    fn test_entry_meta_directory() {
        let meta = ArchiveEntryMeta::directory("path/to/dir/");
        assert!(meta.is_dir());
        assert!(!meta.is_file());
    }

    #[test]
    fn test_entry_meta_no_extension() {
        let meta = ArchiveEntryMeta::file("Makefile", None);
        assert_eq!(meta.extension(), None);

        let meta = ArchiveEntryMeta::file(".gitignore", None);
        assert_eq!(meta.extension(), None);
    }

    #[test]
    fn test_normalize_path() {
        let mut meta = ArchiveEntryMeta::file("/path\\to\\file.txt", None);
        meta.normalize_path();
        assert_eq!(meta.path, "path/to/file.txt");

        let mut meta = ArchiveEntryMeta::file("./relative/path", None);
        meta.normalize_path();
        assert_eq!(meta.path, "relative/path");
    }

    #[test]
    fn test_normalize_path_repeated_prefixes() {
        let cases = [
            ("////././path\\to\\file.txt", "path/to/file.txt"),
            (".//relative\\path", "relative/path"),
            ("././relative/path", "relative/path"),
            ("already/clean", "already/clean"),
            ("/", ""),
        ];

        for (input, expected) in cases {
            let mut meta = ArchiveEntryMeta::file(input, None);
            meta.normalize_path();
            assert_eq!(meta.path, expected);
        }
    }

    #[test]
    fn test_entry_type_char() {
        assert_eq!(EntryType::File.as_char(), '-');
        assert_eq!(EntryType::Directory.as_char(), 'd');
        assert_eq!(EntryType::Symlink.as_char(), 'l');
    }

    #[test]
    fn test_archive_entry_read() {
        let content = b"Hello, World!";
        let meta = ArchiveEntryMeta::file(
            "test.txt",
            Some(limited_read::usize_to_u64_saturating(content.len())),
        );
        let mut entry = ArchiveEntry::new(meta, &content[..]);

        let result = entry.read_to_string().expect("should read");
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_archive_entry_limited_read_accepts_exact_limit() {
        let content = b"Hello";
        let meta = ArchiveEntryMeta::file(
            "test.txt",
            Some(limited_read::usize_to_u64_saturating(content.len())),
        );
        let mut entry = ArchiveEntry::new(meta, &content[..]);

        let result = entry
            .read_to_vec_limited(Some(limited_read::usize_to_u64_saturating(content.len())))
            .expect("read at exact limit");

        assert_eq!(result, content);
    }

    #[test]
    fn test_archive_entry_limited_read_rejects_metadata_over_limit() {
        let content = b"Hello";
        let meta = ArchiveEntryMeta::file(
            "test.txt",
            Some(limited_read::usize_to_u64_saturating(content.len())),
        );
        let mut entry = ArchiveEntry::new(meta, &content[..]);

        let err = entry
            .read_to_vec_limited(Some(4))
            .expect_err("metadata should exceed limit");

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_archive_entry_limited_read_rejects_stream_over_limit_without_size_hint() {
        let content = b"Hello";
        let meta = ArchiveEntryMeta::file("test.txt", None);
        let mut entry = ArchiveEntry::new(meta, &content[..]);

        let err = entry
            .read_to_vec_limited(Some(4))
            .expect_err("stream should exceed limit");

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_initial_read_capacity_is_capped() {
        assert_eq!(
            limited_read::initial_read_capacity(None),
            limited_read::DEFAULT_INITIAL_READ_CAPACITY
        );
        assert_eq!(limited_read::initial_read_capacity(Some(42)), 42);
        assert_eq!(
            limited_read::initial_read_capacity(Some(
                limited_read::usize_to_u64_saturating(limited_read::MAX_INITIAL_READ_CAPACITY) + 1
            )),
            limited_read::MAX_INITIAL_READ_CAPACITY
        );
        assert_eq!(
            limited_read::initial_read_capacity(Some(u64::MAX)),
            limited_read::MAX_INITIAL_READ_CAPACITY
        );
    }
}
