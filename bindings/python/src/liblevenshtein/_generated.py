"""Generated from bindings/api.json; do not edit numeric values manually."""
from enum import IntEnum

ABI_VERSION = 1
API_REVISION = 5
DEFAULT_MATCH_BATCH = 256

class Status(IntEnum):
    """Result of a fallible native operation."""
    OK = 0
    """The operation completed successfully."""
    END = 1
    """A finite cursor reached the end of its stream."""
    INVALID_ARGUMENT = 2
    """An argument violated the operation's contract."""
    INVALID_UTF8 = 3
    """Input advertised as text was not valid UTF-8."""
    NULL_POINTER = 4
    """A required native pointer was null."""
    PANIC = 5
    """A contained Rust panic crossed the failure boundary."""
    UNSUPPORTED = 6
    """The requested capability is unavailable in this build."""
    IO_ERROR = 7
    """An input/output operation failed."""
    CLOSED = 8
    """The target resource was already closed."""
    LIMIT_EXCEEDED = 9
    """A configured resource or traversal limit was exceeded."""
    PROVIDER_ERROR = 10
    """A foreign dictionary provider reported a failure."""
    BATCH_IN_USE = 11
    """A cursor was advanced while its previous batch remained borrowed."""
    DOMAIN_MISMATCH = 12
    """The query and dictionary use different unit domains."""

class Algorithm(IntEnum):
    """Edit-distance algorithm."""
    STANDARD = 0
    """Standard insert/delete/substitute distance."""
    TRANSPOSITION = 1
    """Optimal string alignment with adjacent transposition."""
    MERGE_AND_SPLIT = 2
    """Merge-and-split edit distance."""
    DAMERAU_LEVENSHTEIN = 3
    """Unrestricted Damerau-Levenshtein distance."""

class QueryOrder(IntEnum):
    """Lazy result ordering."""
    TRAVERSAL = 0
    """Provider traversal order with bounded buffering."""
    DISTANCE_THEN_TERM = 1
    """Distance then term, buffering at most one distance layer."""

class PhoneticRuleSetKind(IntEnum):
    """Built-in phonetic rewrite-rule set."""
    ENGLISH_ORTHOGRAPHY = 0
    """English orthography normalization."""
    ENGLISH_PHONETIC = 1
    """English phonetic transformation."""

class OperationApplicability(IntEnum):
    """Runtime generalized-operation applicability predicate."""
    ANY = 0
    """Apply without inspecting consumed units."""
    EQUAL = 1
    """Apply only when the consumed source and target slices are equal."""
    ADJACENT_TRANSPOSE = 2
    """Apply only to an adjacent two-unit transposition."""
    LISTED = 3
    """Apply only to a configured directional source/target pair."""

class UniversalVariant(IntEnum):
    """Universal edit-automaton variant."""
    STANDARD = 0
    """Standard insert/delete/substitute universal automaton."""
    TRANSPOSITION = 1
    """Universal automaton with adjacent transposition."""
    MERGE_AND_SPLIT = 2
    """Universal automaton with merge-and-split edits."""
