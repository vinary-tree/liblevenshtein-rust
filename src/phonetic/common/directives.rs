//! Shared directive and metadata types for phonetic parsing modules.
//!
//! This module provides common types for file-level metadata directives
//! that are shared between the LLev and LLRE parsers.

use crate::phonetic::common::Position;

/// File-level metadata from `@name`, `@version`, etc. directives.
///
/// This structure is shared between LLev (.llev) and LLRE (.llre) file formats,
/// which both support the same set of metadata directives.
///
/// # Example
///
/// ```text
/// @name "English Phonetic Rules"
/// @version "2.0"
/// @author "LibLevenshtein Team"
/// @description "Standard English phonetic transformation rules"
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileMetadata {
    /// Human-readable name (`@name "..."`)
    pub name: Option<String>,

    /// Version string (`@version "..."`)
    pub version: Option<String>,

    /// Author information (`@author "..."`)
    pub author: Option<String>,

    /// Description (`@description "..."`)
    pub description: Option<String>,
}

impl FileMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create new metadata with a name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Check if all metadata fields are empty.
    pub fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.version.is_none()
            && self.author.is_none()
            && self.description.is_none()
    }

    /// Set the name field (builder pattern).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the version field (builder pattern).
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Set the author field (builder pattern).
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the description field (builder pattern).
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Common directive kinds for file-level metadata.
///
/// Both LLev and LLRE support these metadata directives, though they may
/// have additional format-specific directives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataDirective {
    /// `@name "..."` - Human-readable name
    Name(String, Position),

    /// `@version "..."` - Version string
    Version(String, Position),

    /// `@author "..."` - Author information
    Author(String, Position),

    /// `@description "..."` - File description
    Description(String, Position),
}

impl MetadataDirective {
    /// Get the directive name as a string.
    pub fn directive_name(&self) -> &'static str {
        match self {
            Self::Name(..) => "name",
            Self::Version(..) => "version",
            Self::Author(..) => "author",
            Self::Description(..) => "description",
        }
    }

    /// Get the value of the directive.
    pub fn value(&self) -> &str {
        match self {
            Self::Name(v, _)
            | Self::Version(v, _)
            | Self::Author(v, _)
            | Self::Description(v, _) => v,
        }
    }

    /// Get the position of the directive.
    pub fn position(&self) -> Position {
        match self {
            Self::Name(_, p)
            | Self::Version(_, p)
            | Self::Author(_, p)
            | Self::Description(_, p) => *p,
        }
    }

    /// Apply this directive to a FileMetadata instance.
    ///
    /// Returns `Ok(())` if successful, or `Err` with the existing value's
    /// position if this is a duplicate directive.
    pub fn apply_to(&self, metadata: &mut FileMetadata) -> Result<(), Position> {
        match self {
            Self::Name(value, _) => {
                if metadata.name.is_some() {
                    // We don't track position in FileMetadata, so just return current position
                    return Err(self.position());
                }
                metadata.name = Some(value.clone());
            }
            Self::Version(value, _) => {
                if metadata.version.is_some() {
                    return Err(self.position());
                }
                metadata.version = Some(value.clone());
            }
            Self::Author(value, _) => {
                if metadata.author.is_some() {
                    return Err(self.position());
                }
                metadata.author = Some(value.clone());
            }
            Self::Description(value, _) => {
                if metadata.description.is_some() {
                    return Err(self.position());
                }
                metadata.description = Some(value.clone());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_metadata_default() {
        let metadata = FileMetadata::default();
        assert!(metadata.is_empty());
        assert!(metadata.name.is_none());
        assert!(metadata.version.is_none());
        assert!(metadata.author.is_none());
        assert!(metadata.description.is_none());
    }

    #[test]
    fn test_file_metadata_with_name() {
        let metadata = FileMetadata::with_name("Test Rules");
        assert!(!metadata.is_empty());
        assert_eq!(metadata.name, Some("Test Rules".to_string()));
    }

    #[test]
    fn test_file_metadata_builder() {
        let metadata = FileMetadata::new()
            .name("Test Rules")
            .version("1.0")
            .author("Test Author")
            .description("Test description");

        assert!(!metadata.is_empty());
        assert_eq!(metadata.name, Some("Test Rules".to_string()));
        assert_eq!(metadata.version, Some("1.0".to_string()));
        assert_eq!(metadata.author, Some("Test Author".to_string()));
        assert_eq!(metadata.description, Some("Test description".to_string()));
    }

    #[test]
    fn test_metadata_directive_apply() {
        let mut metadata = FileMetadata::new();

        let name_directive = MetadataDirective::Name("Test".into(), Position::start());
        assert!(name_directive.apply_to(&mut metadata).is_ok());
        assert_eq!(metadata.name, Some("Test".to_string()));

        // Duplicate should fail
        let name_directive2 = MetadataDirective::Name("Test2".into(), Position::new(2, 1, 10));
        assert!(name_directive2.apply_to(&mut metadata).is_err());
    }

    #[test]
    fn test_metadata_directive_accessors() {
        let directive = MetadataDirective::Name("Test".into(), Position::new(1, 1, 0));

        assert_eq!(directive.directive_name(), "name");
        assert_eq!(directive.value(), "Test");
        assert_eq!(directive.position(), Position::new(1, 1, 0));
    }
}
