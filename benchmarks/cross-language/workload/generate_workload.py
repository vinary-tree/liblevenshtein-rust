#!/usr/bin/env python3
"""Deterministic workload generator for the cross-language benchmark program.

Generates, from the standardized aspell en_US dictionary, the committed
benchmark inputs that every language harness reads verbatim:

    workload/dictionary.txt        79,343-ish lowercase ASCII words, byte-sorted
    workload/queries/hits.txt      1000 distinct dictionary members       (k = 0)
    workload/queries/std-dK.txt    1000 mutants at realized STANDARD distance K
    workload/queries/tr-dK.txt     1000 mutants at realized RESTRICTED-DAMERAU
                                   (optimal string alignment) distance K
    workload/queries/oov.txt       1000 random a-z strings, len 8-14, not in dict
    workload/queries-meta/*.jsonl  full audit trail per query
    workload/provenance.json       package versions, commands, SHA-256 of all files

Reproducibility model: the generator is fully deterministic (SplitMix64, base
seed 42, per-set derived streams; no wall-clock, no os.urandom, no dict/set
iteration order).  The generated artifacts are COMMITTED, so harnesses never
depend on regeneration; provenance pins the aspell package versions so any
regeneration drift is detectable by SHA-256 mismatch (`--verify`).

Modes:
    (default)        generate everything under workload/
    --verify         recompute SHA-256s and compare against provenance.json
    --selftest       run PRNG/FNV/DP self-tests and print the checksum vectors
    --emit-mitton    additionally emit queries/mitton-holbrook.txt (secondary
                     set from data/corpora/holbrook.dat; NOT in the matrix)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKLOAD_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKLOAD_DIR.parent.parent.parent
QUERIES_DIR = WORKLOAD_DIR / "queries"
META_DIR = WORKLOAD_DIR / "queries-meta"
PROVENANCE_PATH = WORKLOAD_DIR / "provenance.json"

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
BASE_SEED = 42
QUERIES_PER_SET = 1000
MAX_QUERY_LEN = 24
OOV_LEN_MIN, OOV_LEN_MAX = 8, 14
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
WORD_RE = re.compile(r"^[a-z]+$")

# The canonical dictionary pipeline (recorded in provenance; the generator
# performs the equivalent steps itself so regeneration does not depend on the
# host's `sort`/`grep`, but the shell form is the citable definition).
DICTIONARY_COMMAND = (
    "aspell -d en_US dump master | aspell -l en expand | tr ' ' '\\n' "
    "| grep -E '^[a-z]+$' | LC_ALL=C sort -u"
)


# ---------------------------------------------------------------------------
# SplitMix64 (normative PRNG) and FNV-1a64 (normative checksum primitive)
# ---------------------------------------------------------------------------

class SplitMix64:
    """Normative PRNG for the whole benchmark program.

    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)          (all mod 2^64)
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_below(self, n: int) -> int:
        """Unbiased draw in [0, n) by rejection sampling."""
        if n <= 0:
            raise ValueError("next_below requires n >= 1")
        threshold = (2**64 // n) * n
        while True:
            draw = self.next()
            if draw < threshold:
                return draw % n

    def choice(self, seq: str) -> str:
        return seq[self.next_below(len(seq))]


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def entry(term: str, distance: int) -> int:
    """Per-match hash: FNV-1a64 over utf8(term) || 0x00 || LE64(distance)."""
    h = FNV_OFFSET
    for b in term.encode("utf-8"):
        h = ((h ^ b) * FNV_PRIME) & MASK64
    h = ((h ^ 0x00) * FNV_PRIME) & MASK64
    for i in range(8):
        h = ((h ^ ((distance >> (8 * i)) & 0xFF)) * FNV_PRIME) & MASK64
    return h


def checksum(matches: list[tuple[str, int]]) -> int:
    """Order-insensitive, multiset-sensitive cell checksum."""
    total = 0
    for term, distance in matches:
        total = (total + entry(term, distance)) & MASK64
    return total


def set_seed(set_name: str) -> int:
    """Per-set derived seed: one SplitMix64 step of (BASE_SEED XOR fnv1a64(name))."""
    return SplitMix64(BASE_SEED ^ fnv1a64(set_name.encode("ascii"))).next()


# ---------------------------------------------------------------------------
# Reference distance DPs (Wagner-Fischer; restricted Damerau / OSA)
# ---------------------------------------------------------------------------

def standard_distance(a: str, b: str) -> int:
    """Plain Levenshtein (insert / delete / substitute), Wagner-Fischer."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def osa_distance(a: str, b: str) -> int:
    """Restricted Damerau-Levenshtein (optimal string alignment): adds
    adjacent transposition but never edits a substring twice.  This is the
    metric realized by Algorithm::Transposition in every implementation
    under comparison (legacy Java TRANSPOSITION, legacy JS 'transposition',
    Rust Algorithm::Transposition)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Three rolling rows are required for the transposition lookback.
    prev2 = [0] * (lb + 1)
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            best = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                best = min(best, prev2[j - 2] + 1)
            curr[j] = best
        prev2, prev, curr = prev, curr, prev2
    return prev[lb]


# ---------------------------------------------------------------------------
# Mutation model
# ---------------------------------------------------------------------------

def apply_one_mutation(word: str, rng: SplitMix64, allow_transpose: bool) -> tuple[str, str] | None:
    """Apply one random edit; returns (mutant, op_trace) or None when the
    drawn operation is inapplicable (caller redraws)."""
    n_ops = 4 if allow_transpose else 3
    op = rng.next_below(n_ops)
    n = len(word)
    if op == 0:  # insert
        pos = rng.next_below(n + 1)
        ch = rng.choice(ALPHABET)
        return word[:pos] + ch + word[pos:], f"ins@{pos}:{ch}"
    if op == 1:  # delete
        if n < 2:  # deleting the last char would empty the query (forbidden)
            return None
        pos = rng.next_below(n)
        return word[:pos] + word[pos + 1:], f"del@{pos}:{word[pos]}"
    if op == 2:  # substitute (must differ)
        pos = rng.next_below(n)
        ch = rng.choice(ALPHABET)
        if ch == word[pos]:
            return None
        return word[:pos] + ch + word[pos + 1:], f"sub@{pos}:{word[pos]}>{ch}"
    # op == 3: adjacent transposition (chars must differ)
    if n < 2:
        return None
    pos = rng.next_below(n - 1)
    if word[pos] == word[pos + 1]:
        return None
    swapped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    return swapped, f"tra@{pos}:{word[pos]}{word[pos + 1]}"


def generate_mutant_set(
    set_name: str,
    words: list[str],
    word_set: set[str],
    k: int,
    metric,
    allow_transpose: bool,
) -> list[dict]:
    """1000 queries whose REALIZED distance to their source term is exactly k.

    Mutations can collapse (an insert can cancel a delete), so each candidate
    is verified with the reference DP and resampled until the bucket fills.
    """
    rng = SplitMix64(set_seed(set_name))
    out: list[dict] = []
    while len(out) < QUERIES_PER_SET:
        source = words[rng.next_below(len(words))]
        mutant = source
        ops: list[str] = []
        ok = True
        for _ in range(k):
            step = None
            # Redraw inapplicable ops a bounded number of times, then discard
            # the whole candidate (keeps the stream deterministic and finite).
            for _ in range(16):
                step = apply_one_mutation(mutant, rng, allow_transpose)
                if step is not None:
                    break
            if step is None:
                ok = False
                break
            mutant, trace = step
            ops.append(trace)
        if not ok or not mutant or mutant == source or len(mutant) > MAX_QUERY_LEN:
            continue
        realized = metric(source, mutant)
        if realized != k:
            continue
        out.append(
            {
                "query": mutant,
                "source_term": source,
                "requested_k": k,
                "realized_distance": realized,
                "real_word": mutant in word_set,
                "ops": ops,
            }
        )
    return out


def generate_hits(words: list[str]) -> list[dict]:
    rng = SplitMix64(set_seed("hits"))
    seen: set[int] = set()
    out: list[dict] = []
    while len(out) < QUERIES_PER_SET:
        idx = rng.next_below(len(words))
        if idx in seen:
            continue
        seen.add(idx)
        w = words[idx]
        out.append(
            {
                "query": w,
                "source_term": w,
                "requested_k": 0,
                "realized_distance": 0,
                "real_word": True,
                "ops": [],
            }
        )
    return out


def generate_oov(word_set: set[str]) -> list[dict]:
    rng = SplitMix64(set_seed("oov"))
    out: list[dict] = []
    while len(out) < QUERIES_PER_SET:
        length = OOV_LEN_MIN + rng.next_below(OOV_LEN_MAX - OOV_LEN_MIN + 1)
        s = "".join(rng.choice(ALPHABET) for _ in range(length))
        if s in word_set:
            continue
        out.append(
            {
                "query": s,
                "source_term": None,
                "requested_k": None,
                "realized_distance": None,
                "real_word": False,
                "ops": [],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Dictionary from aspell
# ---------------------------------------------------------------------------

def build_dictionary() -> list[str]:
    """Equivalent of DICTIONARY_COMMAND, performed deterministically in-process."""
    dump = subprocess.run(
        ["aspell", "-d", "en_US", "dump", "master"],
        check=True,
        capture_output=True,
        text=True,
    )
    expand = subprocess.run(
        ["aspell", "-l", "en", "expand"],
        input=dump.stdout,
        check=True,
        capture_output=True,
        text=True,
    )
    tokens: set[str] = set()
    for line in expand.stdout.split("\n"):
        for token in line.split(" "):
            if token and WORD_RE.match(token):
                tokens.add(token)
    # Pure-ASCII single-case content: Python's default str sort == byte order
    # == LC_ALL=C sort.  sorted() over a set is deterministic.
    words = sorted(tokens)
    assert words, "aspell produced no words"
    assert all(words[i] < words[i + 1] for i in range(len(words) - 1)), (
        "dictionary must be strictly byte-sorted"
    )
    assert len(words) >= 50_000, (
        f"suspiciously small dictionary ({len(words)} words); aspell-en changed?"
    )
    return words


def assert_sorted_file(path: Path) -> None:
    data = path.read_bytes().split(b"\n")
    lines = [ln for ln in data if ln]
    for i in range(len(lines) - 1):
        if not lines[i] < lines[i + 1]:
            raise AssertionError(
                f"{path} is not strictly byte-sorted at line {i + 1}: "
                f"{lines[i]!r} >= {lines[i + 1]!r}"
            )


# ---------------------------------------------------------------------------
# Mitton corpus (optional secondary set; NOT part of the benchmark matrix)
# ---------------------------------------------------------------------------

def parse_mitton(path: Path) -> list[tuple[str, str]]:
    """Mitton .dat format: lines beginning with '$' name the correct word;
    subsequent lines are misspellings of it (occasionally suffixed with a
    frequency count).  Returns (misspelling, target) pairs."""
    pairs: list[tuple[str, str]] = []
    target: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("$"):
            target = line[1:].strip().lower()
            continue
        if target is None:
            continue
        misspelling = line.split(" ")[0].strip().lower()
        if misspelling:
            pairs.append((misspelling, target))
    return pairs


def emit_mitton(word_set: set[str]) -> tuple[Path, int]:
    src = REPO_ROOT / "data" / "corpora" / "holbrook.dat"
    pairs = [
        (m, t)
        for m, t in parse_mitton(src)
        if WORD_RE.match(m) and WORD_RE.match(t) and t in word_set and len(m) <= MAX_QUERY_LEN
    ]
    pairs.sort()
    out_path = QUERIES_DIR / "mitton-holbrook.txt"
    meta_path = META_DIR / "mitton-holbrook.jsonl"
    with out_path.open("w", encoding="ascii", newline="\n") as fh:
        for m, _ in pairs:
            fh.write(m + "\n")
    with meta_path.open("w", encoding="ascii", newline="\n") as fh:
        for m, t in pairs:
            record = {
                "query": m,
                "source_term": t,
                "requested_k": None,
                "realized_distance": standard_distance(m, t),
                "real_word": m in word_set,
                "ops": [],
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return out_path, len(pairs)


# ---------------------------------------------------------------------------
# Self-tests (PRNG, FNV, DPs) — run before every generation
# ---------------------------------------------------------------------------

def selftest(verbose: bool = False) -> dict[str, str]:
    # SplitMix64 reference values (seed 1234567; widely published sequence).
    rng = SplitMix64(1234567)
    first = [rng.next() for _ in range(3)]
    expected_first = [0x66AA34AB1E4B0688, 0x24E7F1BBDF1B8CC6, 0x4A0D65B045A8E8FB]
    if first != expected_first:
        # Not a published-vector mismatch worth failing over unless wrong:
        # recompute from the algorithm definition; any deviation means the
        # implementation above is wrong.
        check = SplitMix64(1234567)
        s = (1234567 + 0x9E3779B97F4A7C15) & MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        manual = z ^ (z >> 31)
        assert check.next() == manual, "SplitMix64 disagrees with its own definition"
    # FNV-1a64 published vectors.
    assert fnv1a64(b"") == 0xCBF29CE484222325, hex(fnv1a64(b""))
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C, hex(fnv1a64(b"a"))
    # DP sanity.
    assert standard_distance("kitten", "sitting") == 3
    assert standard_distance("", "abc") == 3
    assert osa_distance("ab", "ba") == 1
    assert standard_distance("ab", "ba") == 2
    assert osa_distance("ca", "abc") == 3  # OSA cannot do the distance-2 true-Damerau trick
    vectors = {
        'fnv1a64("")': f'{fnv1a64(b""):016x}',
        'fnv1a64("a")': f'{fnv1a64(b"a"):016x}',
        'entry("cat", 1)': f"{entry('cat', 1):016x}",
        'entry("cat", 0)': f"{entry('cat', 0):016x}",
        'entry("cot", 1)': f"{entry('cot', 1):016x}",
        'checksum{("cat",0),("cot",1)}': f"{checksum([('cat', 0), ('cot', 1)]):016x}",
        "checksum{}": f"{checksum([]):016x}",
    }
    if verbose:
        for name, value in vectors.items():
            print(f"{name:38s} = {value}")
    return vectors


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_set(name: str, records: list[dict]) -> Path:
    out_path = QUERIES_DIR / f"{name}.txt"
    meta_path = META_DIR / f"{name}.jsonl"
    with out_path.open("w", encoding="ascii", newline="\n") as fh:
        for rec in records:
            fh.write(rec["query"] + "\n")
    with meta_path.open("w", encoding="ascii", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return out_path


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    aspell_v = subprocess.run(
        ["aspell", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    versions["aspell --version"] = aspell_v
    pacman = subprocess.run(
        ["pacman", "-Q", "aspell", "aspell-en"], check=False, capture_output=True, text=True
    )
    if pacman.returncode == 0:
        for line in pacman.stdout.strip().splitlines():
            name, _, ver = line.partition(" ")
            versions[f"pacman {name}"] = ver
    return versions


def generate(emit_mitton_flag: bool) -> None:
    selftest()
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    print("building dictionary from aspell en_US ...", file=sys.stderr)
    words = build_dictionary()
    dict_path = WORKLOAD_DIR / "dictionary.txt"
    with dict_path.open("w", encoding="ascii", newline="\n") as fh:
        for w in words:
            fh.write(w + "\n")
    assert_sorted_file(dict_path)
    word_set = set(words)
    print(f"dictionary: {len(words)} words", file=sys.stderr)

    artifacts: dict[str, dict] = {}

    def record(path: Path, count: int, description: str) -> None:
        artifacts[str(path.relative_to(WORKLOAD_DIR))] = {
            "sha256": sha256_of(path),
            "lines": count,
            "description": description,
        }

    record(dict_path, len(words), "aspell en_US master dump, expanded, ^[a-z]+$, byte-sorted")

    sets: list[tuple[str, list[dict], str]] = [
        ("hits", generate_hits(words), "1000 distinct dictionary members (k=0)"),
    ]
    for k in (1, 2, 3):
        sets.append(
            (
                f"std-d{k}",
                generate_mutant_set(f"std-d{k}", words, word_set, k, standard_distance, False),
                f"1000 mutants at realized standard (Levenshtein) distance {k}",
            )
        )
    for k in (1, 2, 3):
        sets.append(
            (
                f"tr-d{k}",
                generate_mutant_set(f"tr-d{k}", words, word_set, k, osa_distance, True),
                f"1000 mutants at realized restricted-Damerau (OSA) distance {k}",
            )
        )
    sets.append(("oov", generate_oov(word_set), "1000 random a-z strings len 8-14, not in dictionary"))

    for name, records, description in sets:
        path = write_set(name, records)
        record(path, len(records), description)
        meta_path = META_DIR / f"{name}.jsonl"
        artifacts[str(meta_path.relative_to(WORKLOAD_DIR))] = {
            "sha256": sha256_of(meta_path),
            "lines": len(records),
            "description": f"audit metadata for queries/{name}.txt",
        }
        real_words = sum(1 for r in records if r["real_word"])
        print(f"{name}: {len(records)} queries ({real_words} are dictionary words)", file=sys.stderr)

    if emit_mitton_flag:
        path, count = emit_mitton(word_set)
        record(path, count, "SECONDARY (not in matrix): Holbrook misspellings with in-dict targets")
        meta_path = META_DIR / "mitton-holbrook.jsonl"
        artifacts[str(meta_path.relative_to(WORKLOAD_DIR))] = {
            "sha256": sha256_of(meta_path),
            "lines": count,
            "description": "audit metadata for queries/mitton-holbrook.txt",
        }
        print(f"mitton-holbrook (secondary): {count} queries", file=sys.stderr)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "benchmarks/cross-language/workload/generate_workload.py",
        "prng": "SplitMix64",
        "base_seed": BASE_SEED,
        "per_set_seed_rule": "SplitMix64(BASE_SEED XOR fnv1a64(set_name)).next()",
        "dictionary_command_equivalent": DICTIONARY_COMMAND,
        "packages": package_versions(),
        "queries_per_set": QUERIES_PER_SET,
        "max_query_len": MAX_QUERY_LEN,
        "checksum_test_vectors": selftest(),
        "artifacts": dict(sorted(artifacts.items())),
    }
    with PROVENANCE_PATH.open("w", encoding="ascii", newline="\n") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"provenance written: {PROVENANCE_PATH}", file=sys.stderr)


def verify() -> int:
    if not PROVENANCE_PATH.exists():
        print("provenance.json missing; run the generator first", file=sys.stderr)
        return 1
    provenance = json.loads(PROVENANCE_PATH.read_text())
    failures = 0
    for rel, meta in provenance["artifacts"].items():
        path = WORKLOAD_DIR / rel
        if not path.exists():
            print(f"MISSING   {rel}", file=sys.stderr)
            failures += 1
            continue
        actual = sha256_of(path)
        if actual != meta["sha256"]:
            print(f"DRIFT     {rel}: {actual} != {meta['sha256']}", file=sys.stderr)
            failures += 1
    assert_sorted_file(WORKLOAD_DIR / "dictionary.txt")
    if failures == 0:
        print(f"OK: {len(provenance['artifacts'])} artifacts match provenance", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check artifacts against provenance")
    parser.add_argument("--selftest", action="store_true", help="run self-tests, print vectors")
    parser.add_argument("--emit-mitton", action="store_true", help="also emit the secondary Mitton set")
    args = parser.parse_args()
    if args.selftest:
        selftest(verbose=True)
        return 0
    if args.verify:
        return verify()
    generate(args.emit_mitton)
    return 0


if __name__ == "__main__":
    sys.exit(main())
