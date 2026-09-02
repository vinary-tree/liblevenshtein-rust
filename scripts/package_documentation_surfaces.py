"""Canonical generated-reference layout shared by package-documentation tools."""

from __future__ import annotations

from typing import Final

# Values are (path relative to target/package-documentation, entry point relative
# to that path). Keeping this inventory outside either producer or packager makes
# it impossible for a successfully generated surface to be silently omitted from
# the immutable release archive.
GENERATED_SURFACE_LAYOUT: Final[dict[str, tuple[str, str]]] = {
    "native": ("native/html", "index.html"),
    "python": ("python", "liblevenshtein.html"),
    "javascript": ("javascript", "index.html"),
    "julia": ("julia", "index.html"),
    "raku": ("raku", "index.html"),
}
