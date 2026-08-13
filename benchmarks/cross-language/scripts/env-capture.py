#!/usr/bin/env python3
"""Capture the benchmark run environment into <results-dir>/environment.json.

Conforms to schema/environment.schema.json. Every fact that could explain a
performance delta between runs is pinned here once per results directory:
toolchain versions, git commits (and dirtiness), artifact SHA-256s, CPU
topology and governor, cpusets, and the workload digests.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
XL = SCRIPTS_DIR.parent
LR = XL.parent.parent
LD = (LR.parent / "libdictenstein").resolve()

TOOLCHAIN_PROBES: dict[str, list[str]] = {
    "rustc": ["rustc", "--version"],
    "cargo": ["cargo", "--version"],
    "cc": ["cc", "--version"],
    "c++": ["c++", "--version"],
    "java": ["java", "--version"],
    "gradle-wrapper": ["bash", "-c", "cd '%s/bindings/jvm' && ./gradlew --version 2>/dev/null | grep -m1 Gradle" % LR],
    "node": ["node", "--version"],
    "npm": ["npm", "--version"],
    "python3": ["python3", "--version"],
    "ruby": ["ruby", "--version"],
    "go": ["go", "version"],
    "dotnet": ["dotnet", "--version"],
    "swift": ["swift", "--version"],
    "ghc": ["ghc", "--version"],
    "cabal": ["cabal", "--version"],
    "ocaml": ["ocaml", "-version"],
    "dune": ["dune", "--version"],
    "opam": ["opam", "--version"],
    "gfortran": ["gfortran", "--version"],
    "fpm": ["fpm", "--version"],
    "lua5.4": ["lua5.4", "-v"],
    "luarocks": ["luarocks", "--version"],
    "clojure": ["clojure", "--version"],
    "wasm-pack": ["wasm-pack", "--version"],
    "wasm-opt": ["wasm-opt", "--version"],
    "node-gyp": ["node-gyp", "--version"],
    "taskset": ["taskset", "--version"],
    "time": ["/usr/bin/time", "--version"],
}

NATIVE_ARTIFACTS = {
    "libliblevenshtein.so": LR / "target" / "release" / "libliblevenshtein.so",
    "liblibdictenstein.so": LD / "target" / "release" / "liblibdictenstein.so",
}

# Optional artifacts appear once their phase has built them (absence is not an
# error at capture time; doctor enforces presence per requested target).
OPTIONAL_ARTIFACTS = {
    "vinary_tree_native.node": LR
    / "bindings"
    / "javascript-runtime"
    / "native"
    / "build"
    / "Release"
    / "vinary_tree_native.node",
    "vinary_tree_bg.wasm": LR
    / "bindings"
    / "javascript-runtime"
    / "generated"
    / "wasm"
    / "vinary_tree_bg.wasm",
    "vinary_tree_wasi.wasm": LR
    / "bindings"
    / "javascript-runtime"
    / "generated"
    / "wasi"
    / "vinary_tree.wasm",
}


def probe(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc.__class__.__name__})"
    stream = result.stdout if result.stdout.strip() else result.stderr
    first = stream.strip().splitlines()
    return first[0].strip() if first else f"unavailable (exit {result.returncode})"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_first_line(path: str) -> str:
    try:
        with open(path, encoding="ascii") as fh:
            return fh.readline().strip()
    except OSError:
        return ""


def git_state(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }


def cpu_info() -> dict:
    model = ""
    cores = 0
    threads = 0
    with open("/proc/cpuinfo", encoding="ascii", errors="replace") as fh:
        physical: set[tuple[str, str]] = set()
        phys = "0"
        for line in fh:
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            if line.startswith("processor"):
                threads += 1
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            if line.startswith("core id"):
                physical.add((phys, line.split(":", 1)[1].strip()))
        cores = len(physical) if physical else threads
    sibling_map = {}
    for cpu in (2, 3, 4, 5, 6, 7, 8, 9):
        sibling = read_first_line(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        if sibling:
            sibling_map[str(cpu)] = sibling
    return {
        "model": model,
        "cores": cores,
        "threads": threads,
        "scaling_driver": read_first_line(
            "/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver"
        ),
        "boost": read_first_line("/sys/devices/system/cpu/cpufreq/boost") or "unknown",
        "smt_sibling_map": sibling_map,
    }


def governor_map() -> dict[str, str]:
    governors = {}
    for cpu in (2, 3, 4, 5, 6, 7, 8, 9):
        value = read_first_line(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
        )
        if value:
            governors[str(cpu)] = value
    return governors


def mem_total_kib() -> int:
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: env-capture.py <results-dir>", file=sys.stderr)
        return 2
    results_dir = Path(sys.argv[1])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = results_dir.name

    native = {}
    for name, path in NATIVE_ARTIFACTS.items():
        if not path.exists():
            print(f"env-capture: required artifact missing: {path}", file=sys.stderr)
            return 1
        native[name] = {"path": str(path), "sha256": sha256_of(path)}
    for name, path in OPTIONAL_ARTIFACTS.items():
        if path.exists():
            native[name] = {"path": str(path), "sha256": sha256_of(path)}

    legacy: dict = {}
    # The legacy jar digest is recorded by the Phase 1 harness once Gradle has
    # resolved it; gate/aggregate read it from the results of that phase. Here
    # we record the vendored JS artifacts if present.
    vendor_dir = XL / "legacy" / "javascript" / "vendor"
    provenance = vendor_dir / "provenance.json"
    if provenance.exists():
        vendored = json.loads(provenance.read_text())
        legacy["js_ghpages_commit"] = vendored["git_commit"]
        legacy["js_vendor_sha256"] = {
            entry["name"]: entry["sha256"] for entry in vendored["files"]
        }
        if "npm_cross_check" in vendored:
            legacy["js_npm_tarball_sha1"] = vendored["npm_cross_check"]["sha1"]
    legacy["java_maven_gav"] = "com.github.universal-automata:liblevenshtein:3.0.0"

    workload_provenance = XL / "workload" / "provenance.json"
    environment = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": os.uname().nodename,
        "kernel": " ".join(os.uname()),
        "cpu": cpu_info(),
        "memory": {"total_kib": mem_total_kib()},
        "governor": governor_map(),
        "cpusets": {"single_threaded": "2", "vm_runtimes": "2-9"},
        "toolchains": {name: probe(cmd) for name, cmd in TOOLCHAIN_PROBES.items()},
        "rustflags": "-C target-cpu=native",
        "repositories": {
            "liblevenshtein-rust": git_state(LR),
            "libdictenstein": git_state(LD),
        },
        "native_artifacts": native,
        "legacy_artifacts": legacy,
        "workload": {
            "provenance_sha256": sha256_of(workload_provenance),
            "dictionary_sha256": sha256_of(XL / "workload" / "dictionary.txt"),
        },
        "notes": [],
    }

    # Sibling WFST repos are part of the JS umbrella static link; record them
    # when present so the addon provenance is complete.
    for sibling in ("lling-llang", "duallity"):
        repo = LR.parent / sibling
        if (repo / ".git").exists():
            environment["repositories"][sibling] = git_state(repo)

    out = results_dir / "environment.json"
    out.write_text(json.dumps(environment, indent=2, sort_keys=False) + "\n")
    print(f"environment captured: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
