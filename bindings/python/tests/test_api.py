from __future__ import annotations

import liblevenshtein
from vinary_tree_interop import UnicodeDictionaryResource


class Snapshot:
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
    with liblevenshtein.PhoneticRuleSet(liblevenshtein.PhoneticRuleSetKind.ENGLISH_ORTHOGRAPHY) as rules:
        assert len(rules) > 0
        assert isinstance(rules.apply("phone"), str)


def test_one_long_lived_custom_provider_cursor_keeps_query_start_snapshot() -> None:
    current = Snapshot({"cat": 1, "cot": 2, "cut": 3, "scat": None})
    with UnicodeDictionaryResource(lambda: current) as dictionary:
        with liblevenshtein.Transducer(dictionary) as automaton:
            frozen = sorted(automaton.query("cat", 2), key=lambda match: str(match.term))
            cursor = automaton.query("cat", 2)
            first = next(cursor)

            # Publish insert/remove/update/compact/checkpoint-equivalent revisions
            # after partial consumption. Each assignment is one immutable revision.
            current = Snapshot({"cat": 1, "cit": 5, "cut": 30, "scat": None})
            current = Snapshot({key: value for key, value in {"cat": 1, "cit": 5, "cut": 30, "scat": None}.items()})

            observed = sorted([first, *cursor], key=lambda match: str(match.term))
            assert observed == frozen
            fresh = sorted(automaton.query("cat", 2), key=lambda match: str(match.term))
            assert fresh != frozen
