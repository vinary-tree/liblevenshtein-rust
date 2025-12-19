//! Zip archive support.
//!
//! This module provides zip archive reading with automatic per-entry decompression.

use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;

use zip::ZipArchive;

use crate::grep::archive::entry::{ArchiveEntryMeta, EntryType};
use crate::grep::archive::filter::EntryFilter;
use crate::grep::error::{GrepError, GrepResult};
use crate::grep::result::SourceId;

/// Reader for zip archives.
pub struct ZipArchiveReader {
    /// Path to the archive.
    path: std::path::PathBuf,

    /// Entry filter (if any).
    filter: Option<EntryFilter>,
}

impl ZipArchiveReader {
    /// Open a zip archive.
    pub fn open(path: impl AsRef<Path>) -> GrepResult<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(GrepError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("archive not found: {}", path.display()),
            )));
        }

        // Verify it's a valid zip
        let file = File::open(path)?;
        let _ = ZipArchive::new(file).map_err(|e| {
            GrepError::archive(path, format!("invalid zip archive: {}", e))
        })?;

        Ok(Self {
            path: path.to_path_buf(),
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

    /// Get the number of entries in the archive.
    pub fn entry_count(&self) -> GrepResult<usize> {
        let file = File::open(&self.path)?;
        let archive = ZipArchive::new(file).map_err(|e| {
            GrepError::archive(&self.path, e.to_string())
        })?;
        Ok(archive.len())
    }

    /// Iterate over entries in the archive.
    ///
    /// Returns an iterator that yields `(SourceId, ArchiveEntryMeta, content)` tuples.
    /// Only regular files are returned; directories are skipped.
    pub fn entries(&self) -> GrepResult<impl Iterator<Item = GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>>> {
        let file = File::open(&self.path)?;
        let archive = ZipArchive::new(file).map_err(|e| {
            GrepError::archive(&self.path, e.to_string())
        })?;

        Ok(ZipEntryIterator {
            archive,
            archive_path: self.path.clone(),
            index: 0,
            filter: self.filter.clone(),
        })
    }

    /// Read a specific entry by path.
    pub fn read_entry(&self, entry_path: &str) -> GrepResult<(ArchiveEntryMeta, Vec<u8>)> {
        let file = File::open(&self.path)?;
        let mut archive = ZipArchive::new(file).map_err(|e| {
            GrepError::archive(&self.path, e.to_string())
        })?;

        let mut entry = archive.by_name(entry_path).map_err(|e| match e {
            zip::result::ZipError::FileNotFound => GrepError::EntryNotFound {
                archive: self.path.clone(),
                entry: entry_path.to_string(),
            },
            _ => GrepError::archive(&self.path, e.to_string()),
        })?;

        let meta = zip_file_to_meta(&entry);
        let mut content = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut content)?;

        Ok((meta, content))
    }

    /// Read a specific entry by index.
    pub fn read_entry_by_index(&self, index: usize) -> GrepResult<(ArchiveEntryMeta, Vec<u8>)> {
        let file = File::open(&self.path)?;
        let mut archive = ZipArchive::new(file).map_err(|e| {
            GrepError::archive(&self.path, e.to_string())
        })?;

        let mut entry = archive.by_index(index).map_err(|e| {
            GrepError::archive(&self.path, e.to_string())
        })?;

        let meta = zip_file_to_meta(&entry);
        let mut content = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut content)?;

        Ok((meta, content))
    }

    /// List all entries (for debugging/display).
    pub fn list_entries(&self) -> GrepResult<Vec<ArchiveEntryMeta>> {
        let file = File::open(&self.path)?;
        let archive = ZipArchive::new(file).map_err(|e| {
            GrepError::archive(&self.path, e.to_string())
        })?;

        let mut entries = Vec::with_capacity(archive.len());
        for i in 0..archive.len() {
            let file = File::open(&self.path)?;
            let mut archive = ZipArchive::new(file).map_err(|e| {
                GrepError::archive(&self.path, e.to_string())
            })?;

            let entry = archive.by_index(i).map_err(|e| {
                GrepError::archive(&self.path, e.to_string())
            })?;

            entries.push(zip_file_to_meta(&entry));
        }

        Ok(entries)
    }
}

/// Iterator over zip archive entries.
struct ZipEntryIterator<R: Read + Seek> {
    archive: ZipArchive<R>,
    archive_path: std::path::PathBuf,
    index: usize,
    filter: Option<EntryFilter>,
}

impl<R: Read + Seek> Iterator for ZipEntryIterator<R> {
    type Item = GrepResult<(SourceId, ArchiveEntryMeta, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.index >= self.archive.len() {
                return None;
            }

            let index = self.index;
            self.index += 1;

            let entry = match self.archive.by_index(index) {
                Ok(e) => e,
                Err(e) => {
                    return Some(Err(GrepError::archive(&self.archive_path, e.to_string())));
                }
            };

            // Skip directories
            if entry.is_dir() {
                continue;
            }

            let meta = zip_file_to_meta(&entry);
            let path = meta.path.clone();

            // Apply filter
            if let Some(ref filter) = self.filter {
                if !filter.matches(&path) {
                    continue;
                }
            }

            // Skip binary files
            if super::filter::should_skip_entry(&path) {
                continue;
            }

            // Re-open archive to get a fresh entry reader
            // (ZipFile borrows the archive)
            let file = match File::open(&self.archive_path) {
                Ok(f) => f,
                Err(e) => return Some(Err(GrepError::Io(e))),
            };

            let mut archive = match ZipArchive::new(file) {
                Ok(a) => a,
                Err(e) => {
                    return Some(Err(GrepError::archive(&self.archive_path, e.to_string())));
                }
            };

            let mut entry = match archive.by_index(index) {
                Ok(e) => e,
                Err(e) => {
                    return Some(Err(GrepError::archive(&self.archive_path, e.to_string())));
                }
            };

            let mut content = Vec::with_capacity(entry.size() as usize);
            if let Err(e) = entry.read_to_end(&mut content) {
                return Some(Err(GrepError::Io(e)));
            }

            let source_id = SourceId::archive_entry(
                self.archive_path.clone(),
                path,
            );

            return Some(Ok((source_id, meta, content)));
        }
    }
}

/// Convert a zip file entry to our metadata type.
fn zip_file_to_meta(entry: &zip::read::ZipFile) -> ArchiveEntryMeta {
    let path = entry.name().to_string();
    let normalized = path
        .trim_start_matches('/')
        .trim_start_matches("./")
        .replace('\\', "/");

    let entry_type = if entry.is_dir() {
        EntryType::Directory
    } else if entry.is_symlink() {
        EntryType::Symlink
    } else {
        EntryType::File
    };

    ArchiveEntryMeta {
        path: normalized,
        size: Some(entry.size()),
        entry_type,
        mtime: entry.last_modified().and_then(|dt| {
            // Convert zip DateTime to SystemTime
            use std::time::{Duration, SystemTime, UNIX_EPOCH};

            // Approximate conversion - zip times are local, not UTC
            let year = dt.year() as u64;
            let month = dt.month() as u64;
            let day = dt.day() as u64;
            let hour = dt.hour() as u64;
            let minute = dt.minute() as u64;
            let second = dt.second() as u64;

            // Very rough approximation (ignores leap years, etc.)
            let days_since_epoch = (year - 1970) * 365 + (month - 1) * 30 + day;
            let secs = days_since_epoch * 86400 + hour * 3600 + minute * 60 + second;

            Some(UNIX_EPOCH + Duration::from_secs(secs))
        }),
        mode: entry.unix_mode(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    use zip::write::SimpleFileOptions;

    fn create_test_zip() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempdir().expect("create temp dir");
        let zip_path = dir.path().join("test.zip");

        let file = File::create(&zip_path).expect("create zip file");
        let mut zip = zip::ZipWriter::new(file);

        let options = SimpleFileOptions::default();

        zip.start_file("file1.txt", options).expect("start file1");
        zip.write_all(b"hello").expect("write file1");

        zip.start_file("dir/file2.txt", options).expect("start file2");
        zip.write_all(b"world").expect("write file2");

        zip.finish().expect("finish zip");

        (dir, zip_path)
    }

    #[test]
    fn test_list_entries() {
        let (_dir, zip_path) = create_test_zip();
        let reader = ZipArchiveReader::open(&zip_path).expect("open zip");

        let entries = reader.list_entries().expect("list entries");
        assert_eq!(entries.len(), 2);

        let paths: Vec<_> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"file1.txt"));
        assert!(paths.contains(&"dir/file2.txt"));
    }

    #[test]
    fn test_read_entry() {
        let (_dir, zip_path) = create_test_zip();
        let reader = ZipArchiveReader::open(&zip_path).expect("open zip");

        let (meta, content) = reader.read_entry("file1.txt").expect("read entry");
        assert_eq!(meta.path, "file1.txt");
        assert_eq!(content, b"hello");
    }

    #[test]
    fn test_iterate_entries() {
        let (_dir, zip_path) = create_test_zip();
        let reader = ZipArchiveReader::open(&zip_path).expect("open zip");

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
    fn test_entry_count() {
        let (_dir, zip_path) = create_test_zip();
        let reader = ZipArchiveReader::open(&zip_path).expect("open zip");
        assert_eq!(reader.entry_count().expect("count"), 2);
    }

    #[test]
    fn test_filter() {
        let (_dir, zip_path) = create_test_zip();
        let reader = ZipArchiveReader::open(&zip_path)
            .expect("open zip")
            .with_pattern("dir/*")
            .expect("set filter");

        let entries: Vec<_> = reader.entries().expect("get entries").collect();
        assert_eq!(entries.len(), 1);

        let (_, meta, _) = entries[0].as_ref().expect("entry should be ok");
        assert_eq!(meta.path, "dir/file2.txt");
    }
}
