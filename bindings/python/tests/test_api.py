from __future__ import annotations

import liblevenshtein
import pytest
from vinary_tree_interop import UnicodeDictionaryResource


class Snapshot:
    def __init__(self, entries: dict[str, int | None]) -> None:
        self.nodes: list[dict[str, object]] = [
            {"edges": {}, "final": False, "value": None}
        ]
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


def test_dictionary_constructors_are_not_reexported() -> None:
    assert not hasattr(liblevenshtein, "Index")
    assert not hasattr(liblevenshtein, "PersistentArTrie")
    assert hasattr(liblevenshtein, "Transducer")


def test_project_owned_phonetic_resource() -> None:
    pattern = liblevenshtein.PhoneticPattern("c(at|ot)")
    try:
        assert pattern.matches("cat")
        assert pattern.matches("cot")
    finally:
        pattern.close()


def test_phonetic_pattern_size_is_exposed() -> None:
    # LLEV-B14: _native.py now binds llev_phonetic_pattern_size, so the
    # automaton (states, transitions) counts surface as PhoneticPattern.size,
    # matching the JVM and every Tier-2/3 facade.
    with liblevenshtein.PhoneticPattern("c(at|ot)") as pattern:
        size = pattern.size
        assert isinstance(size, tuple) and len(size) == 2
        states, transitions = size
        assert states > 0 and transitions > 0


def test_phonetic_rule_set_len_is_exposed() -> None:
    # LLEV-B14: _native.py now binds llev_phonetic_rules_len, so the enabled-rule
    # count surfaces through the Pythonic len() protocol.
    with liblevenshtein.PhoneticRuleSet(
        liblevenshtein.PhoneticRuleSetKind.ENGLISH_ORTHOGRAPHY
    ) as rules:
        assert len(rules) > 0
        assert isinstance(rules.apply("phone"), str)


def test_native_error_exposes_a_typed_status_and_diagnostic() -> None:
    with pytest.raises(liblevenshtein.NativeError) as captured:
        liblevenshtein.PhoneticPattern("(")
    assert captured.value.status is liblevenshtein.Status.INVALID_ARGUMENT
    assert str(captured.value)


def test_query_cursor_next_and_reduce_enforce_owned_and_borrowed_lifetimes() -> None:
    current = Snapshot({"cat": 1, "cot": 2, "cut": 3})
    with (
        UnicodeDictionaryResource(lambda: current) as dictionary,
        liblevenshtein.Transducer(dictionary) as automaton,
    ):
        with automaton.query("cat", 1) as cursor:
            first = cursor.__next__()
            remainder = list(cursor)
        assert sorted(match.term for match in [first, *remainder]) == [
            "cat",
            "cot",
            "cut",
        ]

        escaped = []

        def collect(
            accumulator: list[str], batch: liblevenshtein.BorrowedBatch
        ) -> list[str]:
            if batch:
                escaped.append(batch[0])
            accumulator.extend(match.materialize().term for match in batch)
            return accumulator

        with automaton.query("cat", 1) as cursor:
            reduced = cursor.reduce(collect, [], batch_size=2)
        assert sorted(reduced) == ["cat", "cot", "cut"]
        with pytest.raises(RuntimeError, match="escaped"):
            _ = escaped[0].distance


def test_algorithm_and_query_order_selectors_have_distinguishing_semantics() -> None:
    current = Snapshot({"ab": 1, "c": 2, "abc": 3, "bat": 4, "cat": 5, "cats": 6})

    def query(
        dictionary: UnicodeDictionaryResource,
        algorithm: liblevenshtein.Algorithm,
        text: str,
        maximum: int,
        order: liblevenshtein.QueryOrder = liblevenshtein.QueryOrder.TRAVERSAL,
    ) -> list[liblevenshtein.Match]:
        with (
            liblevenshtein.Transducer(dictionary, algorithm) as automaton,
            automaton.query(text, maximum, order=order) as cursor,
        ):
            return list(cursor)

    with UnicodeDictionaryResource(lambda: current) as dictionary:
        standard = query(dictionary, liblevenshtein.Algorithm.STANDARD, "ba", 1)
        assert all(match.term != "ab" for match in standard)
        transposed = query(dictionary, liblevenshtein.Algorithm.TRANSPOSITION, "ba", 1)
        assert any(match.term == "ab" and match.distance == 1 for match in transposed)
        merged = query(dictionary, liblevenshtein.Algorithm.MERGE_AND_SPLIT, "ab", 1)
        assert any(match.term == "c" and match.distance == 1 for match in merged)
        damerau = query(
            dictionary, liblevenshtein.Algorithm.DAMERAU_LEVENSHTEIN, "ca", 2
        )
        assert any(match.term == "abc" and match.distance == 2 for match in damerau)

        traversal = query(
            dictionary,
            liblevenshtein.Algorithm.STANDARD,
            "cat",
            1,
            liblevenshtein.QueryOrder.TRAVERSAL,
        )
        ranked = query(
            dictionary,
            liblevenshtein.Algorithm.STANDARD,
            "cat",
            1,
            liblevenshtein.QueryOrder.DISTANCE_THEN_TERM,
        )
        assert [(match.term, match.distance) for match in traversal] == [
            ("bat", 1),
            ("cat", 0),
            ("cats", 1),
        ]
        assert [(match.term, match.distance) for match in ranked] == [
            ("cat", 0),
            ("bat", 1),
            ("cats", 1),
        ]


def test_one_long_lived_custom_provider_cursor_keeps_query_start_snapshot() -> None:
    current = Snapshot({"cat": 1, "cot": 2, "cut": 3, "scat": None})
    with (
        UnicodeDictionaryResource(lambda: current) as dictionary,
        liblevenshtein.Transducer(dictionary) as automaton,
    ):
        frozen = sorted(automaton.query("cat", 2), key=lambda match: str(match.term))
        cursor = automaton.query("cat", 2)
        first = next(cursor)

        # Publish insert/remove/update/compact/checkpoint-equivalent revisions
        # after partial consumption. Each assignment is one immutable revision.
        current = Snapshot({"cat": 1, "cit": 5, "cut": 30, "scat": None})
        current = Snapshot(
            {
                key: value
                for key, value in {
                    "cat": 1,
                    "cit": 5,
                    "cut": 30,
                    "scat": None,
                }.items()
            }
        )

        observed = sorted([first, *cursor], key=lambda match: str(match.term))
        assert observed == frozen
        fresh = sorted(automaton.query("cat", 2), key=lambda match: str(match.term))
        assert fresh != frozen
