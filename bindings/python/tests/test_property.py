"""C8 native property-based testing for the Python facade (Hypothesis).

Every property is checked against an in-language brute-force oracle, so a
regression in the native transducer -- or in this facade's marshalling of it --
surfaces as a shrunk counterexample rather than a silent contract drift.

Properties covered (task C8):

* (a) distance symmetry ``d(a,b) == d(b,a)``, identity ``d(a,a) == 0``, and
  threshold consistency (a term is reported at distance ``d`` iff ``d <= k``).
  The Python facade is transducer-only (it exposes no standalone ``llev_distance``
  entry point -- see the reasoned absences in ``scripts/check-bindings.py``), so
  ``d`` is realized through a singleton-dictionary query whose bound
  ``max(len a, len b)`` provably admits the peer (Levenshtein distance never
  exceeds the longer operand's length).
* (b) a query's result set at distance ``k`` equals
  ``{t in dict : levenshtein(query, t) <= k}`` over a small random dictionary.
* (c) round-trips for the u64 value channel, including the ``0`` and ``2**64-1``
  boundaries.

Determinism: the ``pbt`` profile runs with ``derandomize=True`` and no on-disk
example database, so the seeded PRNG makes every run reproducible from source.
The ``@example`` corpus below is the committed regression seed (Hypothesis's
equivalent of ``.proptest-regressions``): the historically interesting cases are
inlined so they are always exercised regardless of random sampling.
"""

from __future__ import annotations

import gc

import liblevenshtein as L
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from vinary_tree_interop import UnicodeDictionaryResource

settings.register_profile(
    "pbt",
    derandomize=True,
    database=None,
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile("pbt")

# "é" is one Unicode scalar but two UTF-8 bytes, so mixing it into the alphabet
# proves the facade counts scalars (not bytes) end to end.
ALPHABET = "abcé"
U64_MAX = 2**64 - 1

_terms = st.text(alphabet=ALPHABET, min_size=0, max_size=6)
_values = st.one_of(st.integers(min_value=0, max_value=U64_MAX), st.none())
_entries = st.dictionaries(keys=_terms, values=_values, max_size=8)
_k = st.integers(min_value=0, max_value=3)


class Snapshot:
    """Deterministic in-language trie oracle over ``{term: u64 | None}``."""

    def __init__(self, entries: dict[str, int | None]) -> None:
        self.nodes: list[dict[str, object]] = [{"edges": {}, "final": False, "value": None}]
        for term, value in entries.items():
            node = 0
            for unit in term:
                edges: dict[str, int] = self.nodes[node]["edges"]  # type: ignore[assignment]
                if unit not in edges:
                    edges[unit] = len(self.nodes)
                    self.nodes.append({"edges": {}, "final": False, "value": None})
                node = edges[unit]
            self.nodes[node]["final"] = True
            self.nodes[node]["value"] = value
        self.length = len(entries)

    def root(self) -> int:
        return 0

    def __len__(self) -> int:
        return self.length

    def is_final(self, node: int) -> bool:
        return bool(self.nodes[node]["final"])

    def value(self, node: int) -> int | None:
        return self.nodes[node]["value"]  # type: ignore[return-value]

    def edges(self, node: int) -> tuple[tuple[str, int], ...]:
        edges: dict[str, int] = self.nodes[node]["edges"]  # type: ignore[assignment]
        return tuple(sorted(edges.items()))


def levenshtein(a: str, b: str) -> int:
    """Reference Levenshtein distance over Unicode scalars (the oracle)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[len(b)]


def _query(entries: dict[str, int | None], query: str, k: int, order: L.QueryOrder = L.QueryOrder.TRAVERSAL):
    snapshot = Snapshot(entries)
    with UnicodeDictionaryResource(lambda: snapshot) as dictionary:
        with L.Transducer(dictionary) as transducer:
            return list(transducer.query(query, k, order=order))


def _transducer_distance(a: str, b: str) -> int:
    """Realize ``d(a, b)`` through a singleton dictionary containing only ``b``.

    ``max(len a, len b)`` is a valid upper bound on ``levenshtein(a, b)``, so
    ``b`` is guaranteed to appear in the result set and carry the true distance.
    """
    bound = max(len(a), len(b))
    for match in _query({b: 0}, a, bound):
        if str(match.term) == b:
            return match.distance
    raise AssertionError(f"peer {b!r} not within bound {bound} of {a!r}")


# --- (b) result set equals the brute-force oracle ---------------------------

@given(entries=_entries, query=_terms, k=_k)
@example(entries={"cat": 1, "cot": 2, "cut": 3, "scat": None}, query="cat", k=2)
@example(entries={"": 0, "cat": U64_MAX, "café": 5}, query="", k=2)
@example(entries={"café": 7}, query="cafe", k=1)
def test_query_result_set_equals_oracle(entries: dict[str, int | None], query: str, k: int) -> None:
    expected = {term: value for term, value in entries.items() if levenshtein(query, term) <= k}
    got = {str(m.term): (m.distance, m.id) for m in _query(entries, query, k)}
    assert set(got) == set(expected)
    for term, (distance, identifier) in got.items():
        assert distance == levenshtein(query, term)  # (a) reported distance is exact
        assert identifier == expected[term]  # (c) value channel round-trips (incl. None)


# --- (a) symmetry and identity ---------------------------------------------

@given(a=_terms, b=_terms)
@example(a="kitten", b="sitting")
@example(a="", b="abc")
def test_distance_symmetry_and_identity(a: str, b: str) -> None:
    forward = _transducer_distance(a, b)
    backward = _transducer_distance(b, a)
    assert forward == backward == levenshtein(a, b)
    assert _transducer_distance(a, a) == 0


# --- (a) threshold consistency ---------------------------------------------

@given(a=_terms, b=_terms, k=_k)
@example(a="cat", b="cot", k=0)
@example(a="cat", b="cot", k=1)
def test_threshold_consistency(a: str, b: str, k: int) -> None:
    distance = levenshtein(a, b)
    reported = {str(m.term): m.distance for m in _query({b: 0}, a, k)}
    if distance <= k:
        assert reported.get(b) == distance
    else:
        assert b not in reported


# --- (c) u64 value round-trip, boundaries pinned ----------------------------

@given(value=st.integers(min_value=0, max_value=U64_MAX))
@example(value=0)
@example(value=U64_MAX)
@example(value=2**63)
def test_u64_value_round_trip(value: int) -> None:
    matches = _query({"term": value}, "term", 0)
    assert len(matches) == 1
    assert matches[0].id == value


# --- bonus: ordered queries are monotone in (distance, term) ----------------

@given(entries=_entries, query=_terms, k=_k)
def test_distance_then_term_ordering(entries: dict[str, int | None], query: str, k: int) -> None:
    sequence = [
        (m.distance, str(m.term))
        for m in _query(entries, query, k, order=L.QueryOrder.DISTANCE_THEN_TERM)
    ]
    assert sequence == sorted(sequence)


def teardown_module(_module: object) -> None:
    # Custom-provider holders are torn down through ``with`` blocks; force a
    # collection so no per-example resource lingers into the leak suite.
    gc.collect()
