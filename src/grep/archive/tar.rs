//! Tar archive support.
//!
//! This module provides tar archive reading with optional compression support.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use tar::Archive;

use crate::grep::archive::entry::{ArchiveEntryMeta, EntryType};
use crate::grep::archive::filter::EntryFilter;
use crate::grep::compression::{create_decompressor, CompressionFormat};
use crate::grep::error::{GrepError, GrepResult};
use crate::grep::result::SourceId;

/// Reader for tar archives (optionally compressed).
pub struct TarArchiveReader {
    /// Path to the archive.
    path: std::path::PathBuf,

    /// Compression format of the tar archive.
    compression: CompressionFormat,

    /// Entry filter (if any).
    filter: Option<EntryFilter>,
}

impl TarArchiveReader {
    /// Open a tar archive.
    pub fn open(path: impl AsRef<Path>, compression: CompressionFormat) -> GrepResult<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(GrepError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("archive not found: {}", path.display()),
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
            compression,
            filter: None,
        })
    }

    /// Set an entry filter.
    pub fn with_filter(mut self, filter: EntryFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a filter from a pattern string.
    pub fn with_pattern(self, pattern: &str) -> GrepResult<Self> {
        let filter = EntryFilter::new(pattern)?;
        Ok(self.with_filter(filter))
    }

    /// Get the archive path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Iterate over entries in the archive.
    ///
    /// Returns an iterator that yields `(SourceId, ArchiveEntryMeta, content)` tuples.
    /// Only regular files are returned; directories and symlinks are skipped.
    ///
    /// Note: This method reads all matching entries into memory. For very large
    /// archives, consider using `read_entry` for specific files instead.
    pub fn entries(&self) -> GrepResult<TarEntryIterator> {
        let file = File::open(&self.path)?;
        let reader = create_decompressor(file, self.compression)?;
        let mut archive = Archive::new(reader);

        // Collect all matching entries eagerly to avoid lifetime issues
        let mut collected = Vec::new();

        for entry_result in archive.entries()? {
            match entry_result {
                Ok(mut entry) => {
                    // Extract all header values first before any mutable borrow
                    let entry_type = header_to_entry_type(entry.header());
                    let size = entry.header().size().ok();
                    let mtime = entry
                        .header()
                        .mtime()
                        .ok()
                        .map(|t| std::time::UNIX_EPOCH + std::time::Duration::from_secs(t));
                    let mode = entry.header().mode().ok();

                    // Skip non-files
                    if !matches!(entry_type, EntryType::File) {
                        continue;
                    }

                    let path = match entry.path() {
                        Ok(p) => p.to_string_lossy().to_string(),
                        Err(e) => {
                            collected.push(Err(GrepError::Io(e)));
                            continue;
                        }
                    };

                    // Normalize path
                    let normalized = path.trim_start_matches('/').trim_start_matches("./");

                    // Apply filter
                    if let Some(ref filter) = self.filter {
                        if !filter.matches(normalized) {
                            continue;
                        }
                    }

                    // Skip binary files by extension
                    if super::filter::should_skip_entry(normalized) {
                        continue;
                    }

                    // Read content
                    let mut content = Vec::with_capacity(size.unwrap_or(1024) as usize);
                    if let Err(e) = entry.read_to_end(&mut content) {
                        collected.push(Err(GrepError::Io(e)));
                        continue;
                    }

                    let meta = ArchiveEntryMeta {
                        path: normalized.to_string(),
                        size,
                        entry_type: EntryType::File,
                        mtime,
                        mode,
                    };

                    let source_id =
                        SourceId::archive_entry(self.path.clone(), normalized.to_string());

                    collected.push(Ok((source_id, meta, content)));
                }
                Err(e) => {
                    collected.push(Err(GrepError::Io(e)));
                }
            }
        }

        Ok(TarEntryIterator {
            entries: collected.into_iter(),
        })
    }

    /// Read a specific entry by path.
    pub fn read_entry(&self, entry_path: &str) -> GrepResult<(ArchiveEntryMeta, Vec<u8>)> {
        let file = File::open(&self.path)?;
        let reader = create_decompressor(file, self.compression)?;
        let mut archive = Archive::new(reader);

        for entry_result in archive.entries()? {
            let mut entry = entry_result?;
            let path = entry.path()?.to_string_lossy().to_string();

            // Normalize for comparison
            let normalized = path.trim_start_matches('/').trim_start_matches("./");

            if normalized == entry_path || path == entry_path {
                let header = entry.header();
                let meta = ArchiveEntryMeta {
                    path: normalized.to_string(),
                    size: Some(header.size()?),
                    entry_type: header_to_entry_type(header),
                    mtime: header
                        .mtime()
                        .ok()
                        .map(|t| std::time::UNIX_EPOCH + std::time::Duration::from_secs(t)),
                    mode: header.mode().ok(),
                };

                let mut content = Vec::new();
                entry.read_to_end(&mut content)?;
                return Ok((meta, content));
            }
        }

        Err(GrepError::EntryNotFound {
            archive: self.path.clone(),
            entry: entry_path.to_string(),
        })
    }

    /// List all entries (for debugging/display).
    pub fn list_entries(&self) -> GrepResult<Vec<ArchiveEntryMeta>> {
        let file = File::open(&self.path)?;
        let reader = create_decompressor(file, self.compression)?;
        let mut archive = Archive::new(reader);

        let mut entries = Vec::new();
        for entry_result in archive.entries()? {
            let entry = entry_result?;
            let header = entry.header();
            let path = entry.path()?.to_string_lossy().to_string();

            let mut meta = ArchiveEntryMeta {
                path,
                size: header.size().ok(),
                entry_type: header_to_entry_type(header),
                mtime: header
                    .mtime()
                    .ok()
                    .map(|t| std::time::UNIX_EPOCH + std::time::Duration::from_secs(t)),
                mode: header.mode().ok(),
            };
            meta.normalize_path();
            entries.push(meta);
        }

        Ok(entries)
    }
}

/// Iterator over tar archive entries.
pub struct TarEntryIterator {
    entries: std::vec::IntoIter<GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>>,
}

impl Iterator for TarEntryIterator {
    type Item = GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next()
    }
}

/// Convert tar header entry type to our EntryType.
fn header_to_entry_type(header: &tar::Header) -> EntryType {
    match header.entry_type() {
        tar::EntryType::Regular => EntryType::File,
        tar::EntryType::Directory => EntryType::Directory,
        tar::EntryType::Symlink => EntryType::Symlink,
        tar::EntryType::Link => EntryType::Hardlink,
        _ => EntryType::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_tar() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempdir().expect("create temp dir");
        let tar_path = dir.path().join("test.tar");

        let file = File::create(&tar_path).expect("create tar file");
        let mut builder = tar::Builder::new(file);

        // Add some test files
        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "file1.txt", b"hello" as &[u8])
            .expect("add file1");

        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "dir/file2.txt", b"world" as &[u8])
            .expect("add file2");

        builder.finish().expect("finish tar");

        (dir, tar_path)
    }

    #[test]
    fn test_list_entries() {
        let (_dir, tar_path) = create_test_tar();
        let reader = TarArchiveReader::open(&tar_path, CompressionFormat::None).expect("open tar");

        let entries = reader.list_entries().expect("list entries");
        assert_eq!(entries.len(), 2);

        let paths: Vec<_> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"file1.txt"));
        assert!(paths.contains(&"dir/file2.txt"));
    }

    #[test]
    fn test_read_entry() {
        let (_dir, tar_path) = create_test_tar();
        let reader = TarArchiveReader::open(&tar_path, CompressionFormat::None).expect("open tar");

        let (meta, content) = reader.read_entry("file1.txt").expect("read entry");
        assert_eq!(meta.path, "file1.txt");
        assert_eq!(content, b"hello");
    }

    #[test]
    fn test_iterate_entries() {
        let (_dir, tar_path) = create_test_tar();
        let reader = TarArchiveReader::open(&tar_path, CompressionFormat::None).expect("open tar");

        let entries: Vec<_> = reader.entries().expect("get entries").collect();
        assert_eq!(entries.len(), 2);

        for result in entries {
            let (source_id, meta, content) = result.expect("entry should be ok");
            assert!(source_id.is_archive_entry());
            assert!(meta.is_file());
            assert!(!content.is_empty());
        }
    }

    #[test]
    fn test_filter() {
        let (_dir, tar_path) = create_test_tar();
        let reader = TarArchiveReader::open(&tar_path, CompressionFormat::None)
            .expect("open tar")
            .with_pattern("dir/*")
            .expect("set filter");

        let entries: Vec<_> = reader.entries().expect("get entries").collect();
        assert_eq!(entries.len(), 1);

        let (_, meta, _) = entries[0].as_ref().expect("entry should be ok");
        assert_eq!(meta.path, "dir/file2.txt");
    }
}
