"""Generated from bindings/api.json; do not edit numeric values manually."""
from enum import IntEnum

ABI_VERSION = 1
API_REVISION = 2
DEFAULT_MATCH_BATCH = 256

class Status(IntEnum):
    OK = 0
    END = 1
    INVALID_ARGUMENT = 2
    INVALID_UTF8 = 3
    NULL_POINTER = 4
    PANIC = 5
    UNSUPPORTED = 6
    IO_ERROR = 7
    CLOSED = 8
    LIMIT_EXCEEDED = 9
    PROVIDER_ERROR = 10
    BATCH_IN_USE = 11
    DOMAIN_MISMATCH = 12

class Algorithm(IntEnum):
    STANDARD = 0
    TRANSPOSITION = 1
    MERGE_AND_SPLIT = 2
    DAMERAU_LEVENSHTEIN = 3

class QueryOrder(IntEnum):
    TRAVERSAL = 0
    DISTANCE_THEN_TERM = 1

class PhoneticRuleSetKind(IntEnum):
    ENGLISH_ORTHOGRAPHY = 0
    ENGLISH_PHONETIC = 1
