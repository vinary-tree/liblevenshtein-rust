//! DjVu text extraction using DjVuLibre's `djvutxt`.
//!
//! The Rust ecosystem does not currently provide a maintained pure-Rust DjVu
//! text extractor. This module integrates with the standard DjVuLibre command
//! line tool when it is available on `PATH`.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::grep::error::{GrepError, GrepResult};

const DJVUTXT: &str = "djvutxt";
static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempDjvuFile {
    path: PathBuf,
}

impl TempDjvuFile {
    fn create(data: &[u8]) -> GrepResult<Self> {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();

        for attempt in 0..16 {
            let path = dir.join(format!(
                "liblevenshtein-djvu-{pid}-{nanos}-{counter}-{attempt}.djvu"
            ));
            match File::options().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    file.write_all(data)
                        .map_err(|err| GrepError::DocumentExtraction {
                            file_path: path.clone(),
                            message: format!("failed to write temporary DjVu data: {err}"),
                        })?;
                    return Ok(Self { path });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => {
                    return Err(GrepError::DocumentExtraction {
                        file_path: path,
                        message: format!("failed to create temporary DjVu file: {err}"),
                    });
                }
            }
        }

        Err(GrepError::DocumentExtraction {
            file_path: dir,
            message: "failed to allocate a unique temporary DjVu file name".to_string(),
        })
    }
}

impl Drop for TempDjvuFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Check whether the `djvutxt` executable is available on `PATH`.
pub fn is_available() -> bool {
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(DJVUTXT))
                .find(|candidate| is_executable(candidate))
        })
        .is_some()
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    path.is_file()
        && path
            .metadata()
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Extract text from DjVu document bytes.
///
/// Requires DjVuLibre's `djvutxt` executable to be installed and visible on
/// `PATH`.
///
/// # Arguments
///
/// * `data` - The raw DjVu file bytes
///
/// # Returns
///
/// The extracted text content, or an error if `djvutxt` is unavailable or
/// extraction fails.
pub fn extract_text(data: &[u8]) -> GrepResult<String> {
    let temp = TempDjvuFile::create(data)?;
    extract_text_from_file(&temp.path)
}

/// Extract text from a DjVu file path using `djvutxt`.
pub fn extract_text_from_file(path: &Path) -> GrepResult<String> {
    let output =
        Command::new(DJVUTXT)
            .arg(path)
            .output()
            .map_err(|err| GrepError::DocumentExtraction {
                file_path: path.to_path_buf(),
                message: format!("failed to execute {DJVUTXT}: {err}"),
            })?;

    if output.status.success() {
        String::from_utf8(output.stdout).map_err(|err| GrepError::DocumentExtraction {
            file_path: path.to_path_buf(),
            message: format!("{DJVUTXT} returned non-UTF-8 text: {err}"),
        })
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(GrepError::DocumentExtraction {
            file_path: path.to_path_buf(),
            message: format!("{DJVUTXT} failed: {}", stderr.trim()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_djvu_availability_check_matches_path_lookup() {
        let expected = std::env::var_os("PATH")
            .map(|paths| std::env::split_paths(&paths).any(|dir| is_executable(&dir.join(DJVUTXT))))
            .unwrap_or(false);
        assert_eq!(is_available(), expected);
    }

    #[test]
    fn test_invalid_djvu_reports_extraction_error() {
        let data = b"AT&TFORM invalid djvu data";
        let result = extract_text(data);
        assert!(result.is_err());

        if let Err(GrepError::DocumentExtraction { message, .. }) = result {
            assert!(
                message.contains(DJVUTXT)
                    || message.contains("temporary DjVu")
                    || message.contains("failed")
            );
        } else {
            panic!("Expected DocumentExtraction error");
        }
    }
}
