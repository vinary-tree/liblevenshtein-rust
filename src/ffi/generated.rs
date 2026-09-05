//! Generated ABI constants and enums.

/// Stable liblevenshtein native ABI version.
pub const LLEV_ABI_VERSION: u32 = 1;
/// Additive API revision within this ABI version.
pub const LLEV_API_REVISION: u32 = 4;

/// Compiled binding feature: core.
pub const LLEV_BUILD_FEATURE_CORE: u64 = 1;
/// Compiled binding feature: phonetic.
pub const LLEV_BUILD_FEATURE_PHONETIC: u64 = 2;

/// Result of a fallible native operation.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlevStatus {
    /// ok.
    Ok = 0,
    /// end.
    End = 1,
    /// invalid argument.
    InvalidArgument = 2,
    /// invalid utf8.
    InvalidUtf8 = 3,
    /// null pointer.
    NullPointer = 4,
    /// panic.
    Panic = 5,
    /// unsupported.
    Unsupported = 6,
    /// io error.
    IoError = 7,
    /// closed.
    Closed = 8,
    /// limit exceeded.
    LimitExceeded = 9,
    /// provider error.
    ProviderError = 10,
    /// batch in use.
    BatchInUse = 11,
    /// domain mismatch.
    DomainMismatch = 12,
}

impl TryFrom<u32> for LlevStatus {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Ok),
            1 => Ok(Self::End),
            2 => Ok(Self::InvalidArgument),
            3 => Ok(Self::InvalidUtf8),
            4 => Ok(Self::NullPointer),
            5 => Ok(Self::Panic),
            6 => Ok(Self::Unsupported),
            7 => Ok(Self::IoError),
            8 => Ok(Self::Closed),
            9 => Ok(Self::LimitExceeded),
            10 => Ok(Self::ProviderError),
            11 => Ok(Self::BatchInUse),
            12 => Ok(Self::DomainMismatch),
            _ => Err(()),
        }
    }
}

/// Edit-distance algorithm.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlevAlgorithm {
    /// standard.
    Standard = 0,
    /// transposition.
    Transposition = 1,
    /// merge and split.
    MergeAndSplit = 2,
    /// damerau levenshtein.
    DamerauLevenshtein = 3,
}

impl TryFrom<u32> for LlevAlgorithm {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Transposition),
            2 => Ok(Self::MergeAndSplit),
            3 => Ok(Self::DamerauLevenshtein),
            _ => Err(()),
        }
    }
}

/// Lazy result order.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlevQueryOrder {
    /// traversal.
    Traversal = 0,
    /// distance then term.
    DistanceThenTerm = 1,
}

impl TryFrom<u32> for LlevQueryOrder {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Traversal),
            1 => Ok(Self::DistanceThenTerm),
            _ => Err(()),
        }
    }
}

/// Built-in phonetic rewrite-rule set.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlevPhoneticRuleSetKind {
    /// english orthography.
    EnglishOrthography = 0,
    /// english phonetic.
    EnglishPhonetic = 1,
}

impl TryFrom<u32> for LlevPhoneticRuleSetKind {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::EnglishOrthography),
            1 => Ok(Self::EnglishPhonetic),
            _ => Err(()),
        }
    }
}
