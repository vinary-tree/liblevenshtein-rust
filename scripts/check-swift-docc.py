"""Verify that the generated Swift DocC site covers every public facade type."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWIFT_SOURCE = (
    ROOT / "bindings" / "swift" / "liblevenshtein" / "Sources" / "Liblevenshtein"
)
PUBLIC_TYPE_RE = re.compile(
    r"^public\s+(?:(?:final|indirect)\s+)?(?:class|struct|enum)\s+([A-Za-z]\w*)",
    re.MULTILINE,
)


def fail(message: str) -> None:
    raise SystemExit(f"swift-docc: {message}")


def public_types() -> set[str]:
    names: set[str] = set()
    for path in sorted(SWIFT_SOURCE.glob("*.swift")):
        names.update(PUBLIC_TYPE_RE.findall(path.read_text(encoding="utf-8")))
    if not names:
        fail("Swift facade source contains no public types")
    return names


def required_static_assets(site: Path) -> set[Path]:
    index = site / "index.html"
    if not index.is_file():
        fail("DocC did not produce index.html")
    body = index.read_text(encoding="utf-8")
    match = re.search(r'var baseUrl = "(/[^"]*/?)"', body)
    if match is None:
        fail("DocC index does not declare its static-hosting base path")
    base = match.group(1)
    assets: set[Path] = set()
    for value in re.findall(r'(?:src|href)="([^"]+)"', body):
        if not value.startswith(base):
            continue
        relative = Path(value.removeprefix(base).split("?", 1)[0])
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"unsafe DocC asset reference: {value}")
        assets.add(relative)
    if not any(path.suffix == ".js" for path in assets) or not any(
        path.suffix == ".css" for path in assets
    ):
        fail("DocC index does not reference both JavaScript and CSS assets")
    return assets


def toolchain_render_template() -> Path:
    information = json.loads(
        subprocess.check_output(["swiftc", "-print-target-info"], text=True)
    )
    resource = Path(information["paths"]["runtimeResourcePath"])
    compiler = Path(shutil.which("swiftc") or "swiftc").resolve()
    candidates = (
        resource.parents[1] / "share" / "docc" / "render",
        compiler.parents[1] / "share" / "docc" / "render",
    )
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    fail("cannot locate the active Swift toolchain's DocC render template")


def complete_toolchain_assets(site: Path, template: Path | None = None) -> None:
    """Fill a toolchain archive's omitted renderer files from its exact template.

    Some Linux Swift 6.3 DocC archives omit the JavaScript and icons even
    though their generated HTML references them. A matching CSS asset guards
    against accidentally mixing two renderer versions.
    """
    required = required_static_assets(site)
    if all((site / path).is_file() for path in required):
        return
    template = template or toolchain_render_template()
    matching_css = [
        path
        for path in required
        if path.suffix == ".css"
        and (site / path).is_file()
        and (template / path).is_file()
        and (site / path).read_bytes() == (template / path).read_bytes()
    ]
    if not matching_css:
        fail("DocC toolchain renderer does not match the generated CSS")
    for source in sorted(template.rglob("*")):
        if source.is_symlink():
            fail(f"DocC renderer contains a symlink: {source}")
        if not source.is_file():
            continue
        relative = source.relative_to(template)
        if relative.name in {"index.html", "index-template.html"}:
            continue
        destination = site / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    missing = sorted(str(path) for path in required if not (site / path).is_file())
    if missing:
        fail(f"DocC renderer template lacks required assets: {missing}")


def verify(site: Path) -> None:
    index = site / "index.html"
    if not index.is_file() or "<html" not in index.read_text(encoding="utf-8").lower():
        fail("DocC did not produce a browsable index.html")
    data = site / "data" / "documentation"
    module = data / "liblevenshtein.json"
    if not module.is_file():
        fail("DocC did not produce the Liblevenshtein module page")
    module_page = json.loads(module.read_text(encoding="utf-8"))
    if "dictionary" not in json.dumps(module_page).lower():
        fail("DocC module page omits the dictionary-search overview")

    documented: set[str] = set()
    for path in sorted(data.rglob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        identifier = page.get("identifier", {})
        url = identifier.get("url") if isinstance(identifier, dict) else None
        if isinstance(url, str) and url:
            documented.add(url.rsplit("/", 1)[-1].casefold())
    missing = sorted(
        name for name in public_types() if name.casefold() not in documented
    )
    if missing:
        fail(f"DocC omits public Swift types: {missing}")
    missing_assets = sorted(
        str(path)
        for path in required_static_assets(site)
        if not (site / path).is_file()
    )
    if missing_assets:
        fail(f"DocC site lacks browser assets: {missing_assets}")
    print(
        f"swift-docc: verified {len(public_types())} public types and browsable assets"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="generated Swift DocC static-site root")
    parser.add_argument(
        "--complete-toolchain-assets",
        action="store_true",
        help="copy renderer files omitted by DocC from the matching Swift toolchain",
    )
    arguments = parser.parse_args()
    if arguments.complete_toolchain_assets:
        complete_toolchain_assets(arguments.site)
    verify(arguments.site)


if __name__ == "__main__":
    main()
