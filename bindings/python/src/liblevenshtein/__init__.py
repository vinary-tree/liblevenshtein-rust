"""Fast streaming fuzzy search over live, versioned dictionary resources.

The package consumes a :class:`vinary_tree_interop.DictionaryResource`
provided by libdictenstein rather than owning dictionary construction. A
:class:`Transducer` retains that provider, and each :meth:`Transducer.query`
captures one immutable revision for an independent :class:`QueryCursor`.

Use context managers for deterministic native-resource release. Ordinary
iteration materializes safe :class:`Match` values; :meth:`QueryCursor.reduce`
is the allocation-sensitive path and confines zero-copy borrowed views to one
callback invocation.
"""

from ._native import (
    Algorithm,
    BorrowedBatch,
    BorrowedMatch,
    Match,
    NativeError,
    PhoneticPattern,
    PhoneticRuleSet,
    PhoneticRuleSetKind,
    QueryCursor,
    QueryOrder,
    Status,
    Transducer,
)

__all__ = [
    "Algorithm",
    "BorrowedBatch",
    "BorrowedMatch",
    "Match",
    "NativeError",
    "PhoneticPattern",
    "PhoneticRuleSet",
    "PhoneticRuleSetKind",
    "QueryCursor",
    "QueryOrder",
    "Status",
    "Transducer",
]
