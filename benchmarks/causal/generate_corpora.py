#!/usr/bin/env python3
"""Generate deterministic corpus shapes for the Java-parity causal campaign.

The committed cross-language dictionary remains the published anchor. This
tool derives smaller anchor strata and creates synthetic shapes that isolate
prefix sharing, suffix sharing, insertion order, and Unicode width. Generated
corpora are intentionally not committed; their manifest pins every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
SEED = 0xCA55_A1_2026_0814
SYNTHETIC_TERMS = 25_000
QUERY_COUNT = 1_000


class SplitMix64:
    """Small, specified PRNG so results do not depend on Python's RNG."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E37_79B9_7F4A_7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
        return value ^ (value >> 31)

    def below(self, limit: int) -> int:
        if limit <= 0:
            raise ValueError("limit must be positive")
        threshold = (1 << 64) - ((1 << 64) % limit)
        while True:
            value = self.next()
            if value < threshold:
                return value % limit


def shuffled(values: list[str], seed: int) -> list[str]:
    out = values.copy()
    rng = SplitMix64(seed)
    for index in range(len(out) - 1, 0, -1):
        other = rng.below(index + 1)
        out[index], out[other] = out[other], out[index]
    return out


def packed_u64_key(value: str) -> tuple[int, ...]:
    """Match `CharUnit for u64`: little-endian values for each 8-byte chunk."""

    encoded = value.encode("utf-8")
    return tuple(
        int.from_bytes(encoded[offset : offset + 8].ljust(8, b"\0"), "little")
        for offset in range(0, len(encoded), 8)
    )


def base26(value: int, width: int = 5) -> str:
    chars = ["a"] * width
    for index in range(width - 1, -1, -1):
        chars[index] = chr(ord("a") + value % 26)
        value //= 26
    return "".join(chars)


def mutate_at_distance_two(term: str, ordinal: int) -> str:
    """Make two scalar substitutions, preserving validity and length."""

    chars = list(term)
    if len(chars) < 2:
        return term + "qx"
    first = ordinal % len(chars)
    second = (first + max(1, len(chars) // 2)) % len(chars)
    if second == first:
        second = (first + 1) % len(chars)
    for position, candidates in ((first, "qx"), (second, "zv")):
        replacement = candidates[0] if chars[position] != candidates[0] else candidates[1]
        chars[position] = replacement
    return "".join(chars)


def common_prefix_length(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def common_suffix_length(left: str, right: str) -> int:
    return common_prefix_length(left[::-1], right[::-1])


@dataclass(frozen=True)
class CorpusRecord:
    name: str
    unit_domain: str
    order: str
    term_count: int
    utf8_bytes: int
    scalar_units: int
    adjacent_prefix_units: int
    adjacent_suffix_units: int
    dictionary: str
    dictionary_sha256: str
    queries: str
    queries_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def record_corpus(
    output: Path, name: str, unit_domain: str, order: str, terms: list[str]
) -> CorpusRecord:
    dictionary_path = output / f"{name}-{order}.txt"
    queries_path = output / f"{name}-{order}-queries-std-d2.txt"
    canonical = sorted(terms, key=packed_u64_key) if unit_domain == "u64" else sorted(terms)
    ordered = canonical if order == "sorted" else shuffled(canonical, SEED ^ len(name))
    write_lines(dictionary_path, ordered)

    source_order = shuffled(canonical, SEED ^ 0xD2 ^ len(name))
    queries = [
        mutate_at_distance_two(term, ordinal)
        for ordinal, term in enumerate(source_order[: min(QUERY_COUNT, len(source_order))])
    ]
    write_lines(queries_path, queries)

    adjacent = list(zip(ordered, ordered[1:]))
    return CorpusRecord(
        name=name,
        unit_domain=unit_domain,
        order=order,
        term_count=len(ordered),
        utf8_bytes=sum(len(term.encode("utf-8")) for term in ordered),
        scalar_units=sum(len(term) for term in ordered),
        adjacent_prefix_units=sum(common_prefix_length(a, b) for a, b in adjacent),
        adjacent_suffix_units=sum(common_suffix_length(a, b) for a, b in adjacent),
        dictionary=dictionary_path.name,
        dictionary_sha256=file_sha256(dictionary_path),
        queries=queries_path.name,
        queries_sha256=file_sha256(queries_path),
    )


def unique_nonempty_lines(path: Path) -> list[str]:
    values = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate non-empty terms")
    if not values:
        raise ValueError(f"{path} contains no terms")
    return values


def corpus_shapes(anchor: list[str]) -> list[tuple[str, str, list[str]]]:
    sampled = shuffled(anchor, SEED ^ 0xA11C)
    shapes: list[tuple[str, str, list[str]]] = []
    for count in (1_000, 10_000, len(anchor)):
        if count > len(anchor):
            continue
        label = "full" if count == len(anchor) else str(count)
        shapes.append((f"anchor-{label}", "byte", sampled[:count]))

    codes = [base26(index) for index in range(SYNTHETIC_TERMS)]
    shapes.append(("prefix-heavy-25k", "byte", [f"commonprefix{code}" for code in codes]))
    shapes.append(("suffix-heavy-25k", "byte", [f"{code}commonsuffix" for code in codes]))

    markers = ("é", "λ", "Ж", "中", "🙂")
    unicode_terms = [
        f"{markers[index % len(markers)]}{base26(index)}{markers[(index // len(markers)) % len(markers)]}"
        for index in range(SYNTHETIC_TERMS)
    ]
    shapes.append(("unicode-mixed-25k", "unicode", unicode_terms))
    # Packed-u64 ordering is intentionally different from UTF-8 lexical order;
    # this anchor proves that the optimized constructor and corpus generator
    # agree on the backend's actual edge order.
    shapes.append(("u64-anchor-full", "u64", anchor))
    return shapes


def generate(anchor_path: Path, output: Path) -> int:
    anchor = unique_nonempty_lines(anchor_path)
    output.mkdir(parents=True, exist_ok=True)
    records = [
        record_corpus(output, name, unit_domain, order, terms)
        for name, unit_domain, terms in corpus_shapes(anchor)
        for order in ("sorted", "shuffled")
    ]
    manifest = {
        "schema": "liblevenshtein.java-parity-corpora.v1",
        "seed": SEED,
        "synthetic_term_count": SYNTHETIC_TERMS,
        "query_count": QUERY_COUNT,
        "anchor": str(anchor_path.resolve()),
        "anchor_sha256": file_sha256(anchor_path),
        "records": [asdict(record) for record in records],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "manifest.json")
    return 0


def verify(output: Path) -> int:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["records"]:
        for path_key, digest_key in (
            ("dictionary", "dictionary_sha256"),
            ("queries", "queries_sha256"),
        ):
            path = output / record[path_key]
            actual = file_sha256(path) if path.is_file() else "missing"
            if actual != record[digest_key]:
                failures.append(f"{path.name}: expected {record[digest_key]}, got {actual}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"verified {len(manifest['records'])} corpus/order cells")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cross-language/workload/dictionary.txt",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return verify(args.output) if args.verify else generate(args.anchor, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
