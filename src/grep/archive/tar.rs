//! Tar archive support.
//!
//! This module provides tar archive reading with optional compression support.

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;

use tar::Archive;

use crate::grep::archive::entry::{
    normalize_archive_path_owned, read_to_vec_limited, ArchiveEntryMeta, EntryType,
};
use crate::grep::archive::filter::EntryFilter;
use crate::grep::compression::{create_decompressor, CompressionFormat};
use crate::grep::error::{GrepError, GrepResult};
use crate::grep::result::SourceId;

const TAR_ENTRY_CHANNEL_BOUND: usize = 1;

/// Reader for tar archives (optionally compressed).
pub struct TarArchiveReader {
    /// Path to the archive.
    path: std::path::PathBuf,

    /// Compression format of the tar archive.
    compression: CompressionFormat,

    /// Entry filter (if any).
    filter: Option<EntryFilter>,

    /// Maximum uncompressed entry size to read, if any.
    max_entry_size: Option<u64>,
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
            max_entry_size: None,
        })
    }

    /// Set an entry filter.
    pub fn with_filter(mut self, filter: EntryFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set the maximum uncompressed entry size to read.
    pub fn with_max_entry_size(mut self, bytes: Option<u64>) -> Self {
        self.max_entry_size = bytes;
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
    pub fn entries(&self) -> GrepResult<TarEntryIterator> {
        let file = File::open(&self.path)?;
        let reader = create_decompressor(file, self.compression)?;
        let archive_path = self.path.clone();
        let filter = self.filter.clone();
        let max_entry_size = self.max_entry_size;
        let (sender, receiver) = sync_channel(TAR_ENTRY_CHANNEL_BOUND);

        thread::spawn(move || {
            let mut archive = Archive::new(reader);
            let entries = match archive.entries() {
                Ok(entries) => entries,
                Err(e) => {
                    let _ = sender.send(Err(GrepError::Io(e)));
                    return;
                }
            };

            for entry_result in entries {
                if let Some(item) =
                    read_entry_result(&archive_path, filter.as_ref(), max_entry_size, entry_result)
                {
                    if sender.send(item).is_err() {
                        break;
                    }
                }
            }
        });

        Ok(TarEntryIterator { entries: receiver })
    }

    /// Read a specific entry by path.
    pub fn read_entry(&self, entry_path: &str) -> GrepResult<(ArchiveEntryMeta, Vec<u8>)> {
        let file = File::open(&self.path)?;
        let reader = create_decompressor(file, self.compression)?;
        let mut archive = Archive::new(reader);

        for entry_result in archive.entries()? {
            let mut entry = entry_result?;
            let path = entry.path()?.to_string_lossy().to_string();
            let raw_matches = path == entry_path;

            // Normalize for comparison
            let normalized = normalize_archive_path_owned(path);

            if normalized == entry_path || raw_matches {
                let header = entry.header();
                let size = header.size()?;
                let meta = ArchiveEntryMeta {
                    path: normalized,
                    size: Some(size),
                    entry_type: header_to_entry_type(header),
                    mtime: header
                        .mtime()
                        .ok()
                        .map(|t| std::time::UNIX_EPOCH + std::time::Duration::from_secs(t)),
                    mode: header.mode().ok(),
                };

                let content = read_to_vec_limited(&mut entry, Some(size), self.max_entry_size)?;
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

            let meta = ArchiveEntryMeta {
                path: normalize_archive_path_owned(path),
                size: header.size().ok(),
                entry_type: header_to_entry_type(header),
                mtime: header
                    .mtime()
                    .ok()
                    .map(|t| std::time::UNIX_EPOCH + std::time::Duration::from_secs(t)),
                mode: header.mode().ok(),
            };
            entries.push(meta);
        }

        Ok(entries)
    }
}

/// Iterator over tar archive entries.
pub struct TarEntryIterator {
    entries: Receiver<GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>>,
}

impl Iterator for TarEntryIterator {
    type Item = GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.recv().ok()
    }
}

fn read_entry_result<R: io::Read>(
    archive_path: &Path,
    filter: Option<&EntryFilter>,
    max_entry_size: Option<u64>,
    entry_result: io::Result<tar::Entry<'_, R>>,
) -> Option<GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>> {
    let mut entry = match entry_result {
        Ok(entry) => entry,
        Err(e) => return Some(Err(GrepError::Io(e))),
    };

    // Extract all header values before the mutable content read.
    let entry_type = header_to_entry_type(entry.header());
    let size = entry.header().size().ok();
    let mtime = entry
        .header()
        .mtime()
        .ok()
        .map(|t| std::time::UNIX_EPOCH + std::time::Duration::from_secs(t));
    let mode = entry.header().mode().ok();

    if !matches!(entry_type, EntryType::File) {
        return None;
    }

    let path = match entry.path() {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(e) => return Some(Err(GrepError::Io(e))),
    };
    let normalized = normalize_archive_path_owned(path);

    if let Some(filter) = filter {
        if !filter.matches(&normalized) {
            return None;
        }
    }

    if super::filter::should_skip_entry(&normalized) {
        return None;
    }

    let content = match read_to_vec_limited(&mut entry, size, max_entry_size) {
        Ok(content) => content,
        Err(e) => return Some(Err(GrepError::Io(e))),
    };

    let meta = ArchiveEntryMeta {
        path: normalized.clone(),
        size,
        entry_type: EntryType::File,
        mtime,
        mode,
    };
    let source_id = SourceId::archive_entry(archive_path.to_path_buf(), normalized);

    Some(Ok((source_id, meta, content)))
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
    fn test_read_entry_respects_max_entry_size() {
        let (_dir, tar_path) = create_test_tar();
        let reader = TarArchiveReader::open(&tar_path, CompressionFormat::None)
            .expect("open tar")
            .with_max_entry_size(Some(4));

        let err = reader
            .read_entry("file1.txt")
            .expect_err("entry should exceed max entry size");

        assert!(
            matches!(err, GrepError::Io(ref io_err) if io_err.kind() == io::ErrorKind::InvalidData)
        );
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
    fn test_iterate_entries_respects_max_entry_size() {
        let (_dir, tar_path) = create_test_tar();
        let reader = TarArchiveReader::open(&tar_path, CompressionFormat::None)
            .expect("open tar")
            .with_max_entry_size(Some(4));

        let entries: Vec<_> = reader.entries().expect("get entries").collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|entry| {
            matches!(entry, Err(GrepError::Io(io_err)) if io_err.kind() == io::ErrorKind::InvalidData)
        }));
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

    #[test]
    fn test_normalizes_archive_paths() {
        let dir = tempdir().expect("create temp dir");
        let tar_path = dir.path().join("test.tar");

        let file = File::create(&tar_path).expect("create tar file");
        let mut builder = tar::Builder::new(file);

        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, ".//dir\\file3.txt", b"slash" as &[u8])
            .expect("add normalized path file");

        builder.finish().expect("finish tar");

        let reader =
            TarArchiveReader::open(&tar_path, CompressionFormat::None).expect("open tar archive");
        let entries = reader.list_entries().expect("list entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "dir/file3.txt");

        let (meta, content) = reader
            .read_entry("dir/file3.txt")
            .expect("read normalized entry");
        assert_eq!(meta.path, "dir/file3.txt");
        assert_eq!(content, b"slash");
    }
}
