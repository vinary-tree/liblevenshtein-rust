#!/usr/bin/env python3
"""Keep executable Rust API examples from silently becoming ignored doctests."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"

# This is a ratchet, not a target. A change that repairs ignored examples must
# lower this number in the same commit; a change that adds ignored examples is
# rejected. The remaining debt is catalogued in the Rustdoc example policy.
IGNORED_EXAMPLE_BASELINE = 33

# Every cache example was compiled and executed successfully during the
# 2026-08-28 audit. This subtree therefore has a stricter zero-ignore policy.
CACHE_EXECUTABLE_BASELINE = 46

FENCE_RE = re.compile(r"```(?P<info>[^`]*)\s*$")


def fail(message: str) -> None:
    print(f"rustdoc-examples: {message}", file=sys.stderr)
    raise SystemExit(1)


def fence_info(line: str) -> str | None:
    match = FENCE_RE.search(line)
    if match is None:
        return None
    info = match.group("info").strip().replace(" ", "")
    return info or None


def is_ignored_rust(info: str) -> bool:
    attributes = info.split(",")
    return info == "ignore" or (attributes[0] == "rust" and "ignore" in attributes[1:])


def is_executable_rust(info: str) -> bool:
    attributes = info.split(",")
    return attributes[0] == "rust" and "ignore" not in attributes[1:]


def main() -> None:
    ignored: list[str] = []
    cache_ignored: list[str] = []
    cache_executable = 0

    for path in sorted(SOURCE_ROOT.rglob("*.rs")):
        relative = path.relative_to(ROOT)
        in_cache = relative.parts[:2] == ("src", "cache")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            info = fence_info(line)
            if info is None:
                continue
            location = f"{relative}:{line_number}"
            if is_ignored_rust(info):
                ignored.append(location)
                if in_cache:
                    cache_ignored.append(location)
            elif in_cache and is_executable_rust(info):
                cache_executable += 1

    if cache_ignored:
        fail(
            "cache API examples must remain executable; ignored fences: "
            + ", ".join(cache_ignored)
        )
    if cache_executable < CACHE_EXECUTABLE_BASELINE:
        fail(
            "cache executable-example coverage regressed: "
            f"found {cache_executable}, require at least {CACHE_EXECUTABLE_BASELINE}"
        )
    if len(ignored) > IGNORED_EXAMPLE_BASELINE:
        fail(
            "ignored Rustdoc debt grew: "
            f"found {len(ignored)}, baseline {IGNORED_EXAMPLE_BASELINE}; "
            "make the new example executable instead"
        )
    if len(ignored) < IGNORED_EXAMPLE_BASELINE:
        fail(
            "ignored Rustdoc debt decreased without lowering the ratchet: "
            f"found {len(ignored)}, baseline {IGNORED_EXAMPLE_BASELINE}; "
            "update IGNORED_EXAMPLE_BASELINE in this script"
        )

    print(
        "rustdoc-examples: ok "
        f"({cache_executable} executable cache examples; "
        f"{len(ignored)} globally ignored examples, ratchet enforced)"
    )


if __name__ == "__main__":
    main()
