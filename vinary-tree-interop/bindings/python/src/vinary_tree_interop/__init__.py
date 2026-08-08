"""Shared ctypes layouts for modular vinary-tree bindings."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Protocol, runtime_checkable


class UnitDomain(IntEnum):
    """Dictionary edge-label domain."""

    BYTE = 1
    UNICODE_SCALAR = 2
    U64 = 3


class VtResource(ctypes.Structure):
    """Two-word retained resource passed between native project libraries."""

    _fields_ = [("context", ctypes.c_void_p), ("vtable", ctypes.c_void_p)]


class UnicodeDictionarySnapshot(Protocol):
    """Immutable node graph captured by a custom Python provider."""

    def root(self) -> int:
        """Return the root node identifier."""

    def __len__(self) -> int:
        """Return the number of final terms."""

    def is_final(self, node: int) -> bool:
        """Return whether a node terminates a term."""

    def value(self, node: int) -> int | None:
        """Return a final node's optional u64 value."""

    def edges(self, node: int) -> Sequence[tuple[str | int, int]]:
        """Return outgoing Unicode-scalar edges and child node IDs."""


@runtime_checkable
class DictionaryResource(Protocol):
    """A live resource implementing dictionary interface v1."""

    @property
    def native_resource(self) -> VtResource:
        """Borrow the two-word resource for one synchronous native call."""

    def close(self) -> None:
        """Release this facade's owned retain."""


ABI_VERSION = 1
DICTIONARY_INTERFACE_VERSION = 1
DICTIONARY_INTERFACE_ID = b"vt.dictionary.v1"


class _InterfaceId(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_uint8 * 16)]


class _OptionalU64(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint64),
        ("has_value", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 7),
    ]


class _Edge(ctypes.Structure):
    _fields_ = [("label", ctypes.c_uint64), ("node", ctypes.c_uint64)]


_RETAIN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_QUERY_INTERFACE = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.POINTER(_InterfaceId),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
)
_SNAPSHOT = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(VtResource))
_ROOT = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64))
_LEN = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_uint8),
)
_IS_FINAL = ctypes.CFUNCTYPE(
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8)
)
_VALUE = ctypes.CFUNCTYPE(
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(_OptionalU64)
)
_TRANSITION = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint8),
)
_EDGES = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_size_t,
    ctypes.POINTER(_Edge),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
)


class _ResourceVTable(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("abi_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("retain", _RETAIN),
        ("release", _RELEASE),
        ("query_interface", _QUERY_INTERFACE),
    ]


class _DictionaryVTable(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("interface_version", ctypes.c_uint32),
        ("unit_domain", ctypes.c_uint32),
        ("value_domain", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("snapshot", _SNAPSHOT),
        ("root", _ROOT),
        ("len", _LEN),
        ("node_is_final", _IS_FINAL),
        ("node_value_u64", _VALUE),
        ("node_transition", _TRANSITION),
        ("node_edges", _EDGES),
    ]


class _Holder:
    __slots__ = ("source", "immutable", "parallel", "refs", "errors")

    def __init__(
        self,
        source: Callable[[], UnicodeDictionarySnapshot] | UnicodeDictionarySnapshot,
        *,
        immutable: bool,
        parallel: bool,
        errors: list[BaseException | None] | None = None,
    ) -> None:
        self.source = source
        self.immutable = immutable
        self.parallel = parallel
        self.refs = 1
        self.errors = errors if errors is not None else [None]

    def snapshot(self) -> UnicodeDictionarySnapshot:
        return self.source if self.immutable else self.source()  # type: ignore[operator,return-value]


_registry: dict[int, _Holder] = {}
_registry_lock = threading.RLock()


def _register(holder: _Holder) -> int:
    key = id(holder)
    with _registry_lock:
        _registry[key] = holder
    return key


def _holder(context: int) -> _Holder:
    with _registry_lock:
        return _registry[context]


def _status(holder: _Holder, operation: Callable[[], None]) -> int:
    try:
        operation()
        return 0
    except BaseException as error:
        holder.errors[0] = error
        return 8


@_RETAIN
def _retain(context: int) -> None:
    with _registry_lock:
        _registry[context].refs += 1


@_RELEASE
def _release(context: int) -> None:
    with _registry_lock:
        holder = _registry.get(context)
        if holder is None:
            return
        holder.refs -= 1
        if holder.refs == 0:
            del _registry[context]


def _dictionary_vtable(holder: _Holder) -> _DictionaryVTable:
    return _DICTIONARY_VTABLES[(holder.immutable, holder.parallel)]


@_QUERY_INTERFACE
def _query_interface(
    context: int,
    interface_id: ctypes.POINTER(_InterfaceId),
    minimum_version: int,
    out_vtable: ctypes.POINTER(ctypes.c_void_p),
) -> int:
    if not interface_id or not out_vtable:
        return 3
    if bytes(interface_id.contents.bytes) != DICTIONARY_INTERFACE_ID or minimum_version > 1:
        return 4
    table = _dictionary_vtable(_holder(context))
    out_vtable[0] = ctypes.cast(ctypes.pointer(table), ctypes.c_void_p)
    return 0


@_SNAPSHOT
def _snapshot(context: int, output: ctypes.POINTER(VtResource)) -> int:
    holder = _holder(context)

    def run() -> None:
        snapshot = holder.snapshot()
        child = _Holder(
            snapshot,
            immutable=True,
            parallel=holder.parallel,
            errors=holder.errors,
        )
        key = _register(child)
        output[0] = VtResource(key, ctypes.addressof(_RESOURCE_VTABLE))

    return _status(holder, run)


@_ROOT
def _root(context: int, output: ctypes.POINTER(ctypes.c_uint64)) -> int:
    holder = _holder(context)
    return _status(holder, lambda: output.__setitem__(0, holder.snapshot().root()))


@_LEN
def _len(context: int, output: ctypes.POINTER(ctypes.c_size_t), known: ctypes.POINTER(ctypes.c_uint8)) -> int:
    holder = _holder(context)

    def run() -> None:
        output[0] = len(holder.snapshot())
        known[0] = 1

    return _status(holder, run)


@_IS_FINAL
def _is_final(context: int, node: int, output: ctypes.POINTER(ctypes.c_uint8)) -> int:
    holder = _holder(context)
    return _status(holder, lambda: output.__setitem__(0, holder.snapshot().is_final(node)))


@_VALUE
def _value(context: int, node: int, output: ctypes.POINTER(_OptionalU64)) -> int:
    holder = _holder(context)

    def run() -> None:
        value = holder.snapshot().value(node)
        if value is not None and not 0 <= value < 2**64:
            raise ValueError("dictionary value is outside u64")
        output[0] = _OptionalU64(value or 0, value is not None, (ctypes.c_uint8 * 7)())

    return _status(holder, run)


def _label(value: str | int) -> int:
    if isinstance(value, str):
        if len(value) != 1:
            raise ValueError("Unicode edge labels must contain one scalar")
        return ord(value)
    if not 0 <= value <= 0x10FFFF or 0xD800 <= value <= 0xDFFF:
        raise ValueError("edge label is not a Unicode scalar")
    return value


@_TRANSITION
def _transition(
    context: int,
    node: int,
    label: int,
    child: ctypes.POINTER(ctypes.c_uint64),
    found: ctypes.POINTER(ctypes.c_uint8),
) -> int:
    holder = _holder(context)

    def run() -> None:
        for edge_label, edge_child in holder.snapshot().edges(node):
            if _label(edge_label) == label:
                child[0], found[0] = edge_child, 1
                return
        child[0], found[0] = 0, 0

    return _status(holder, run)


@_EDGES
def _edges(
    context: int,
    node: int,
    start: int,
    output: ctypes.POINTER(_Edge),
    capacity: int,
    written: ctypes.POINTER(ctypes.c_size_t),
    total: ctypes.POINTER(ctypes.c_size_t),
) -> int:
    holder = _holder(context)

    def run() -> None:
        values = holder.snapshot().edges(node)
        total[0] = len(values)
        page = values[start : start + capacity]
        for index, (label, child) in enumerate(page):
            output[index] = _Edge(_label(label), child)
        written[0] = len(page)

    return _status(holder, run)


_RESOURCE_VTABLE = _ResourceVTable(
    ctypes.sizeof(_ResourceVTable), ABI_VERSION, 0, _retain, _release, _query_interface
)


def _make_dictionary_vtable(immutable: bool, parallel: bool) -> _DictionaryVTable:
    flags = (1 if parallel else 0) | (4 if immutable else 0)
    return _DictionaryVTable(
        ctypes.sizeof(_DictionaryVTable),
        DICTIONARY_INTERFACE_VERSION,
        UnitDomain.UNICODE_SCALAR,
        1,
        flags,
        _snapshot,
        _root,
        _len,
        _is_final,
        _value,
        _transition,
        _edges,
    )


_DICTIONARY_VTABLES = {
    (immutable, parallel): _make_dictionary_vtable(immutable, parallel)
    for immutable in (False, True)
    for parallel in (False, True)
}


class UnicodeDictionaryResource:
    """Adapt a host-defined Unicode snapshot provider to dictionary ABI v1.

    Callbacks are serialized by consumers unless ``parallel_reentrant`` is true.
    ``capture`` must return an immutable, O(1)-captured revision.
    """

    def __init__(
        self,
        capture: Callable[[], UnicodeDictionarySnapshot],
        *,
        parallel_reentrant: bool = False,
    ) -> None:
        self._holder = _Holder(capture, immutable=False, parallel=parallel_reentrant)
        self._key = _register(self._holder)
        self._resource = VtResource(self._key, ctypes.addressof(_RESOURCE_VTABLE))

    @property
    def native_resource(self) -> VtResource:
        """Borrow the two-word custom-provider resource."""
        if not self._key:
            raise RuntimeError("dictionary resource is closed")
        return self._resource

    @property
    def last_callback_error(self) -> BaseException | None:
        """Last exception converted to ``VT_STATUS_PROVIDER_ERROR``."""
        return self._holder.errors[0]

    def close(self) -> None:
        """Release the resource facade; retained transducers remain valid."""
        if self._key:
            _release(self._key)
            self._key = 0

    def __enter__(self) -> UnicodeDictionaryResource:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

__all__ = [
    "ABI_VERSION",
    "DICTIONARY_INTERFACE_ID",
    "DICTIONARY_INTERFACE_VERSION",
    "DictionaryResource",
    "UnicodeDictionaryResource",
    "UnicodeDictionarySnapshot",
    "UnitDomain",
    "VtResource",
]
