"""C9 leak-discipline suite for the Python facade (tracemalloc + gc).

A >=10,000-cycle create/use/free loop must reach a memory steady state: neither
the interop's provider registry (an exact handle-count signal) nor the Python
heap (measured by ``tracemalloc``) may grow across the cycles. A retained
transducer handle, an unreleased query cursor, or a provider holder that never
leaves ``vinary_tree_interop._registry`` would all show up here.

Measured baseline on this native build (10,000 cycles): registry delta ``0`` and
~0.14 B/cycle of ``tracemalloc`` churn -- i.e. flat. The thresholds below leave a
>100x margin over that noise floor while still catching a real per-cycle leak,
which would accumulate megabytes.
"""

from __future__ import annotations

import gc
import tracemalloc

import liblevenshtein as L
import vinary_tree_interop as interop
from vinary_tree_interop import UnicodeDictionaryResource

CYCLES = 10_000
WARMUP = 1_000
# Generous absolute ceiling: measured churn is ~1.4 KiB total; a genuine
# per-cycle leak would blow through this by orders of magnitude.
MAX_GROWTH_BYTES = 256 * 1024


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


def _measure(loop) -> int:
    for _ in range(WARMUP):
        loop()
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    for _ in range(CYCLES):
        loop()
    gc.collect()
    end = tracemalloc.take_snapshot()
    growth = sum(stat.size_diff for stat in end.compare_to(base, "filename"))
    tracemalloc.stop()
    return growth


def test_transducer_cycle_has_no_handle_or_memory_leak() -> None:
    snapshot = Snapshot({"cat": 1, "cot": 2, "cut": 3, "scat": None})

    def loop() -> None:
        with UnicodeDictionaryResource(lambda: snapshot) as dictionary:
            with L.Transducer(dictionary) as transducer:
                for _ in transducer.query("cat", 2):
                    pass

    for _ in range(WARMUP):
        loop()
    gc.collect()
    registry_before = len(interop._registry)
    growth = _measure(loop)
    registry_after = len(interop._registry)

    # Exact handle-leak signal: every provider holder is released each cycle.
    assert registry_after == registry_before, (
        f"provider registry grew from {registry_before} to {registry_after}"
    )
    assert growth < MAX_GROWTH_BYTES, f"heap grew {growth} bytes over {CYCLES} cycles"


def test_query_reduce_cycle_has_no_memory_leak() -> None:
    snapshot = Snapshot({"cat": 1, "cot": 2, "cut": 3, "scat": None})

    def loop() -> None:
        with UnicodeDictionaryResource(lambda: snapshot) as dictionary:
            with L.Transducer(dictionary) as transducer:
                cursor = transducer.query("cat", 2)
                cursor.reduce(lambda count, batch: count + len(batch), 0)

    growth = _measure(loop)
    assert growth < MAX_GROWTH_BYTES, f"heap grew {growth} bytes over {CYCLES} cycles"


def test_phonetic_pattern_cycle_has_no_memory_leak() -> None:
    def loop() -> None:
        with L.PhoneticPattern("c(at|ot)") as pattern:
            pattern.matches("cat")

    growth = _measure(loop)
    assert growth < MAX_GROWTH_BYTES, f"heap grew {growth} bytes over {CYCLES} cycles"


def test_phonetic_rules_cycle_has_no_memory_leak() -> None:
    def loop() -> None:
        with L.PhoneticRuleSet(L.PhoneticRuleSetKind.ENGLISH_ORTHOGRAPHY) as rules:
            rules.apply("phone")

    growth = _measure(loop)
    assert growth < MAX_GROWTH_BYTES, f"heap grew {growth} bytes over {CYCLES} cycles"
