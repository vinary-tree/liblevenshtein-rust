"""Render a complete Lua facade reference and reject registration/documentation drift.

The JSON inventory is prose, not authority for the exported names: the C
``luaL_Reg`` arrays are parsed and compared before anything is published.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bindings/lua/src/liblevenshtein_lua.c"
SPEC = ROOT / "bindings/lua/api-reference.json"
REGISTERED_ARRAY = re.compile(
    r"(?:static\s+)?(?:const\s+)?luaL_Reg\s+(\w+)\s*\[\]\s*=\s*\{(.*?)\};",
    re.DOTALL,
)
REGISTERED_ENTRY = re.compile(r'\{\s*"([^"]+)"\s*,\s*\w+\s*\}')
CONSTANT = re.compile(r'(?:string_constant|status_constant)\(state,\s*"([^"]+)"')


def fail(message: str) -> None:
    raise ValueError(f"Lua API reference: {message}")


def load_and_validate() -> dict[str, object]:
    specification = json.loads(SPEC.read_text(encoding="utf-8"))
    source = SOURCE.read_text(encoding="utf-8")
    arrays = {
        name: set(REGISTERED_ENTRY.findall(body))
        for name, body in REGISTERED_ARRAY.findall(source)
    }
    expected_arrays = {group["sourceArray"] for group in specification["groups"]}
    if set(arrays) != expected_arrays:
        fail(
            f"registration arrays differ: source={sorted(arrays)}, docs={sorted(expected_arrays)}"
        )
    for group in specification["groups"]:
        name = group["sourceArray"]
        documented = [entry["name"] for entry in group["entries"]]
        if len(documented) != len(set(documented)) or set(documented) != arrays[name]:
            fail(
                f"{name} differs: source={sorted(arrays[name])}, docs={sorted(documented)}"
            )
        for entry in group["entries"]:
            for field in ("signature", "description", "returns"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    fail(f"{name}.{entry['name']} has no {field}")
    constants = set(CONSTANT.findall(source))
    documented_constants = [entry["name"] for entry in specification["constants"]]
    if (
        len(documented_constants) != len(set(documented_constants))
        or set(documented_constants) != constants
    ):
        fail(
            f"constants differ: source={sorted(constants)}, docs={sorted(documented_constants)}"
        )
    for entry in specification["constants"]:
        if not entry.get("description"):
            fail(f"constant {entry['name']} has no description")
    for entry in specification["protocols"]:
        if entry["sourceMarker"] not in source:
            fail(f"protocol source marker missing: {entry['sourceMarker']}")
        if not entry.get("description"):
            fail(f"protocol {entry['name']} has no description")
    for example in specification["examples"]:
        path = ROOT / example["path"]
        if not path.is_file() or not example.get("description"):
            fail(f"example is missing or undocumented: {example['path']}")
        completed = subprocess.run(
            ["luac5.4", "-p", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            fail(f"example is not valid Lua: {example['path']}: {completed.stderr}")
    return specification


def render(specification: dict[str, object], version: str, source_ref: str) -> str:
    if re.fullmatch(r"[0-9A-Za-z.-]+", version) is None:
        fail("invalid version")
    if re.fullmatch(r"[0-9A-Za-z.-]+", source_ref) is None:
        fail("invalid source reference")
    escape = html.escape
    repository = specification["repository"]
    base = f"{repository}/blob/{source_ref}/"
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{escape(specification['title'])} {escape(version)} API</title>",
        (
            "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:75rem;margin:2rem auto;padding:0 1rem;color:#18212b}"
            "code,pre{font-family:ui-monospace,monospace}pre{overflow:auto;background:#f1f4f8;padding:1rem}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #c9d2dd;padding:.45rem;text-align:left;vertical-align:top}"
            "nav a{margin-right:1rem}a{color:#06539b}</style></head><body><main>"
        ),
        f"<h1>{escape(specification['title'])} {escape(version)} API</h1>",
        f"<p>{escape(specification['summary'])}</p>",
        (
            f'<nav><a href="{escape(base + "bindings/lua/README.md", quote=True)}">Usage guide</a>'
            f'<a href="{escape(base + "bindings/lua/src/" + SOURCE.name, quote=True)}">C facade source</a></nav>'
        ),
        '<h2 id="examples">Common usage</h2>',
    ]
    for example in specification["examples"]:
        path = ROOT / example["path"]
        parts.extend(
            (
                f"<h3>{escape(example['title'])}</h3>",
                (
                    f"<p>{escape(example['description'])} "
                    f'<a href="{escape(base + example["path"], quote=True)}">Source</a>.</p>'
                ),
                f'<pre><code class="language-lua">{escape(path.read_text(encoding="utf-8"))}</code></pre>',
            )
        )
    for group in specification["groups"]:
        parts.extend(
            (
                f"<h2>{escape(group['title'])}</h2>",
                "<table><thead><tr><th>Call</th><th>Behavior</th><th>Result</th></tr></thead><tbody>",
            )
        )
        for entry in group["entries"]:
            parts.append(
                "<tr><td><code>"
                + escape(entry["signature"])
                + "</code></td><td>"
                + escape(entry["description"])
                + "</td><td>"
                + escape(entry["returns"])
                + "</td></tr>"
            )
        parts.append("</tbody></table>")
    parts.extend(
        (
            "<h2>Constants and protocols</h2>",
            "<table><thead><tr><th>Name</th><th>Meaning</th></tr></thead><tbody>",
        )
    )
    for entry in specification["constants"] + specification["protocols"]:
        parts.append(
            "<tr><td><code>"
            + escape(entry["name"])
            + "</code></td><td>"
            + escape(entry["description"])
            + "</td></tr>"
        )
    parts.extend(
        (
            "</tbody></table>",
            f"<p>Source revision: <code>{escape(source_ref)}</code>. This reference is generated from a source-validated API inventory.</p>",
            "</main></body></html>",
            "",
        )
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    specification = load_and_validate()
    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render(specification, arguments.version, arguments.source_ref), encoding="utf-8"
    )
    print(f"Lua API reference: wrote {output}")


if __name__ == "__main__":
    main()
