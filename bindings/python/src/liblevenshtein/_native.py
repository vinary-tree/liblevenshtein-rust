"""Batch-oriented ctypes facade over liblevenshtein's resource ABI."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypeVar

from vinary_tree_interop import DictionaryResource, UnitDomain, VtResource

from ._generated import (
    ABI_VERSION,
    DEFAULT_MATCH_BATCH,
    Algorithm,
    PhoneticRuleSetKind,
    QueryOrder,
    Status,
)


class NativeError(RuntimeError):
    """Failure reported by the stable native ABI."""

    def __init__(self, status: int | Status, message: str) -> None:
        super().__init__(message)
        try:
            self.status: Status | int = Status(status)
        except ValueError:
            # Preserve forward-compatible diagnostics when a newer native
            # library reports a status unknown to this package revision.
            self.status = int(status)


class _MatchView(ctypes.Structure):
    _fields_ = [
        ("term_data", ctypes.c_void_p),
        ("term_len", ctypes.c_size_t),
        ("byte_len", ctypes.c_size_t),
        ("distance", ctypes.c_size_t),
        ("id", ctypes.c_uint64),
        ("unit_domain", ctypes.c_uint32),
        ("has_id", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class _BatchView(ctypes.Structure):
    _fields_ = [
        ("matches", ctypes.POINTER(_MatchView)),
        ("len", ctypes.c_size_t),
        ("generation", ctypes.c_uint64),
    ]


class _OwnedString(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def _library_names() -> tuple[str, ...]:
    system = platform.system()
    if system == "Windows":
        return ("liblevenshtein.dll",)
    if system == "Darwin":
        return ("libliblevenshtein.dylib",)
    return ("libliblevenshtein.so",)


def _load_library() -> ctypes.CDLL:
    candidates: list[str] = []
    if explicit := os.environ.get("LIBLEVENSHTEIN_LIBRARY"):
        candidates.append(explicit)
    package = Path(__file__).resolve().parent
    candidates.extend(str(package / "native" / name) for name in _library_names())
    if discovered := ctypes.util.find_library("liblevenshtein"):
        candidates.append(discovered)
    candidates.extend(_library_names())
    failures = []
    for candidate in candidates:
        try:
            # CDLL releases the GIL while first-party native search is running.
            return ctypes.CDLL(candidate)
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    raise ImportError(
        "could not load liblevenshtein; set LIBLEVENSHTEIN_LIBRARY\n"
        + "\n".join(failures)
    )


_lib = _load_library()
_lib.llev_abi_version.restype = ctypes.c_uint32
_lib.llev_last_error_message.restype = ctypes.c_char_p
_lib.llev_transducer_new.argtypes = [
    ctypes.POINTER(VtResource),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.llev_transducer_new.restype = ctypes.c_uint32
_lib.llev_transducer_free.argtypes = [ctypes.c_void_p]
_lib.llev_transducer_query_utf8.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.llev_transducer_query_utf8.restype = ctypes.c_uint32
_lib.llev_transducer_query_bytes.argtypes = list(
    _lib.llev_transducer_query_utf8.argtypes
)
_lib.llev_transducer_query_bytes.restype = ctypes.c_uint32
_lib.llev_transducer_query_u64.argtypes = list(_lib.llev_transducer_query_utf8.argtypes)
_lib.llev_transducer_query_u64.restype = ctypes.c_uint32
_lib.llev_query_cursor_next_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(_BatchView),
]
_lib.llev_query_cursor_next_batch.restype = ctypes.c_uint32
_lib.llev_query_cursor_release_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_lib.llev_query_cursor_release_batch.restype = ctypes.c_uint32
_lib.llev_query_cursor_free.argtypes = [ctypes.c_void_p]
_lib.llev_query_cursor_free.restype = ctypes.c_uint32

_REDUCER = ctypes.CFUNCTYPE(
    ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(_MatchView), ctypes.c_size_t
)
_lib.llev_query_cursor_reduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    _REDUCER,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.llev_query_cursor_reduce.restype = ctypes.c_uint32

_lib.llev_phonetic_pattern_compile_regex.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.llev_phonetic_pattern_compile_regex.restype = ctypes.c_uint32
_lib.llev_phonetic_pattern_compile_llre.argtypes = list(
    _lib.llev_phonetic_pattern_compile_regex.argtypes
)
_lib.llev_phonetic_pattern_compile_llre.restype = ctypes.c_uint32
_lib.llev_phonetic_pattern_free.argtypes = [ctypes.c_void_p]
_lib.llev_phonetic_pattern_matches.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
]
_lib.llev_phonetic_pattern_matches.restype = ctypes.c_uint32
_lib.llev_phonetic_pattern_size.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.llev_phonetic_pattern_size.restype = ctypes.c_uint32
_lib.llev_transducer_query_pattern.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint8,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.llev_transducer_query_pattern.restype = ctypes.c_uint32
_lib.llev_phonetic_rules_parse.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.llev_phonetic_rules_parse.restype = ctypes.c_uint32
_lib.llev_phonetic_rules_builtin.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.llev_phonetic_rules_builtin.restype = ctypes.c_uint32
_lib.llev_phonetic_rules_len.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.llev_phonetic_rules_len.restype = ctypes.c_uint32
_lib.llev_phonetic_rules_free.argtypes = [ctypes.c_void_p]
_lib.llev_phonetic_rules_apply.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(_OwnedString),
]
_lib.llev_phonetic_rules_apply.restype = ctypes.c_uint32
_lib.llev_owned_string_free.argtypes = [ctypes.POINTER(_OwnedString)]

if _lib.llev_abi_version() != ABI_VERSION:
    raise ImportError("liblevenshtein native ABI version mismatch")


def _error() -> str:
    value = _lib.llev_last_error_message()
    return value.decode("utf-8", "replace") if value else "native operation failed"


def _check(status: int, *, end: bool = False) -> bool:
    if status == Status.OK:
        return True
    if end and status == Status.END:
        return False
    raise NativeError(status, _error())


@dataclass(frozen=True, slots=True)
class Match:
    """One safely materialized fuzzy match."""

    term: str | bytes | tuple[int, ...]
    distance: int
    id: int | None


class BorrowedMatch:
    """Zero-copy descriptor valid only during a reducer callback."""

    __slots__ = ("_active", "_view")

    def __init__(self, view: _MatchView, active: list[bool]) -> None:
        self._view = view
        self._active = active

    def _ensure(self) -> None:
        if not self._active[0]:
            raise RuntimeError("borrowed match escaped its batch callback")

    @property
    def distance(self) -> int:
        self._ensure()
        return self._view.distance

    @property
    def id(self) -> int | None:
        self._ensure()
        return self._view.id if self._view.has_id else None

    def bytes_view(self) -> memoryview:
        """Borrow raw bytes for byte or Unicode terms."""
        self._ensure()
        if self._view.unit_domain == UnitDomain.U64:
            raise TypeError("u64 term has no byte view")
        array = (ctypes.c_ubyte * self._view.byte_len).from_address(
            self._view.term_data
        )
        return memoryview(array)

    def utf8(self) -> str:
        """Decode a Unicode term only when requested."""
        if self._view.unit_domain != UnitDomain.UNICODE_SCALAR:
            raise TypeError("term is not Unicode")
        return bytes(self.bytes_view()).decode("utf-8")

    def u64_view(self) -> memoryview:
        """Borrow aligned u64 terms."""
        self._ensure()
        if self._view.unit_domain != UnitDomain.U64:
            raise TypeError("term is not u64")
        array = (ctypes.c_uint64 * self._view.term_len).from_address(
            self._view.term_data
        )
        return memoryview(array)

    def materialize(self) -> Match:
        self._ensure()
        domain = UnitDomain(self._view.unit_domain)
        if domain == UnitDomain.UNICODE_SCALAR:
            term: str | bytes | tuple[int, ...] = self.utf8()
        elif domain == UnitDomain.BYTE:
            term = bytes(self.bytes_view())
        else:
            term = tuple(self.u64_view())
        return Match(term, self.distance, self.id)


class BorrowedBatch(Sequence[BorrowedMatch]):
    """Zero-copy batch valid only during one reducer invocation."""

    def __init__(
        self, views: ctypes.POINTER(_MatchView), length: int, active: list[bool]
    ) -> None:
        self._views, self._length, self._active = views, length, active

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> BorrowedMatch:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return BorrowedMatch(self._views[index], self._active)


T = TypeVar("T")


class QueryCursor(Iterator[Match]):
    """One-shot lazy cursor retaining its query-start snapshot."""

    def __init__(self, handle: int) -> None:
        self._handle = ctypes.c_void_p(handle)
        self._pending: deque[Match] = deque()

    def __iter__(self) -> QueryCursor:
        return self

    def __next__(self) -> Match:
        if not self._pending:
            view = _BatchView()
            if not _check(
                _lib.llev_query_cursor_next_batch(
                    self._handle, DEFAULT_MATCH_BATCH, ctypes.byref(view)
                ),
                end=True,
            ):
                self.close()
                raise StopIteration
            try:
                active = [True]
                self._pending.extend(
                    BorrowedMatch(view.matches[index], active).materialize()
                    for index in range(view.len)
                )
            finally:
                active[0] = False
                _check(
                    _lib.llev_query_cursor_release_batch(self._handle, view.generation)
                )
        return self._pending.popleft()

    def reduce(
        self,
        function: Callable[[T, BorrowedBatch], T],
        initial: T,
        *,
        batch_size: int = DEFAULT_MATCH_BATCH,
    ) -> T:
        """Reduce native batches without allocating Python match objects."""
        accumulator = initial
        failure: BaseException | None = None

        @_REDUCER
        def callback(
            _context: int, views: ctypes.POINTER(_MatchView), length: int
        ) -> int:
            nonlocal accumulator, failure
            active = [True]
            try:
                accumulator = function(
                    accumulator, BorrowedBatch(views, length, active)
                )
                return Status.OK
            # No Python exception may unwind through the C callback trampoline;
            # preserve even cancellation/system-exit exceptions for re-raising.
            except BaseException as error:  # noqa: BLE001
                failure = error
                return Status.END
            finally:
                active[0] = False

        count = ctypes.c_size_t()
        try:
            _check(
                _lib.llev_query_cursor_reduce(
                    self._handle, batch_size, callback, None, ctypes.byref(count)
                )
            )
        finally:
            self.close()
        if failure is not None:
            raise failure
        return accumulator

    def close(self) -> None:
        if self._handle:
            _check(_lib.llev_query_cursor_free(self._handle))
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            # Destructors cannot report errors, especially during interpreter
            # shutdown. Deterministic users should use close()/with.
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class Transducer:
    """Levenshtein automaton configuration over a live dictionary resource."""

    def __init__(
        self,
        dictionary: DictionaryResource | VtResource,
        algorithm: Algorithm = Algorithm.STANDARD,
    ) -> None:
        resource = (
            dictionary.native_resource
            if isinstance(dictionary, DictionaryResource)
            else dictionary
        )
        self._handle = ctypes.c_void_p()
        _check(
            _lib.llev_transducer_new(
                ctypes.byref(resource), int(algorithm), ctypes.byref(self._handle)
            )
        )

    def query(
        self,
        query: str | bytes | Sequence[int] | PhoneticPattern,
        max_distance: int,
        *,
        order: QueryOrder = QueryOrder.TRAVERSAL,
    ) -> QueryCursor:
        out = ctypes.c_void_p()
        if isinstance(query, PhoneticPattern):
            _check(
                _lib.llev_transducer_query_pattern(
                    self._handle, query._handle, max_distance, ctypes.byref(out)
                )
            )
        elif isinstance(query, str):
            data = query.encode()
            _check(
                _lib.llev_transducer_query_utf8(
                    self._handle,
                    data,
                    len(data),
                    max_distance,
                    int(order),
                    ctypes.byref(out),
                )
            )
        elif isinstance(query, bytes):
            _check(
                _lib.llev_transducer_query_bytes(
                    self._handle,
                    query,
                    len(query),
                    max_distance,
                    int(order),
                    ctypes.byref(out),
                )
            )
        else:
            values = (ctypes.c_uint64 * len(query))(*query)
            _check(
                _lib.llev_transducer_query_u64(
                    self._handle,
                    values,
                    len(query),
                    max_distance,
                    int(order),
                    ctypes.byref(out),
                )
            )
        return QueryCursor(out.value)

    def close(self) -> None:
        if self._handle:
            _lib.llev_transducer_free(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PhoneticPattern:
    """Reusable Unicode phonetic language automaton."""

    def __init__(self, source: str, *, llre: bool = False) -> None:
        data = source.encode()
        self._handle = ctypes.c_void_p()
        constructor = (
            _lib.llev_phonetic_pattern_compile_llre
            if llre
            else _lib.llev_phonetic_pattern_compile_regex
        )
        _check(constructor(data, len(data), ctypes.byref(self._handle)))

    def matches(self, text: str) -> bool:
        data = text.encode()
        output = ctypes.c_uint8()
        _check(
            _lib.llev_phonetic_pattern_matches(
                self._handle, data, len(data), ctypes.byref(output)
            )
        )
        return bool(output.value)

    @property
    def size(self) -> tuple[int, int]:
        """Compiled ``(states, transitions)`` counts of the pattern automaton."""
        states = ctypes.c_size_t()
        transitions = ctypes.c_size_t()
        _check(
            _lib.llev_phonetic_pattern_size(
                self._handle, ctypes.byref(states), ctypes.byref(transitions)
            )
        )
        return (states.value, transitions.value)

    def close(self) -> None:
        if self._handle:
            _lib.llev_phonetic_pattern_free(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PhoneticRuleSet:
    """Reusable Unicode phonetic rewrite-rule set."""

    def __init__(self, source: str | PhoneticRuleSetKind) -> None:
        self._handle = ctypes.c_void_p()
        if isinstance(source, PhoneticRuleSetKind):
            _check(
                _lib.llev_phonetic_rules_builtin(
                    int(source), ctypes.byref(self._handle)
                )
            )
        else:
            data = source.encode()
            _check(
                _lib.llev_phonetic_rules_parse(
                    data, len(data), ctypes.byref(self._handle)
                )
            )

    def apply(self, text: str) -> str:
        data = text.encode()
        output = _OwnedString()
        _check(
            _lib.llev_phonetic_rules_apply(
                self._handle, data, len(data), ctypes.byref(output)
            )
        )
        try:
            return ctypes.string_at(output.data, output.len).decode()
        finally:
            _lib.llev_owned_string_free(ctypes.byref(output))

    def __len__(self) -> int:
        """Number of enabled rewrite rules in the set."""
        count = ctypes.c_size_t()
        _check(_lib.llev_phonetic_rules_len(self._handle, ctypes.byref(count)))
        return count.value

    def close(self) -> None:
        if self._handle:
            _lib.llev_phonetic_rules_free(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
