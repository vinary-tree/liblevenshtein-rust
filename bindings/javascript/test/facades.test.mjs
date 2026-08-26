import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url)));

test("all project facades share the exact shared JavaScript runtime", () => {
  assert.equal(packageJson.dependencies["@vinary-tree/javascript-runtime"], packageJson.version);
  assert.equal(packageJson.dependencies["@vinary-tree/vinary-tree-interop"], "4.0.0-rc.5");
  for (const path of [".", "./typescript", "./clojurescript", "./wasm", "./wasi"]) {
    assert.ok(packageJson.exports[path]);
  }
});

test("project package does not own dictionary constructors", async () => {
  const declarations = await readFile(new URL("../index.d.ts", import.meta.url), "utf8");
  assert.doesNotMatch(declarations, /class\s+(?:Dynamic|Persistent|Scdawg|DoubleArray)/);
  assert.match(declarations, /DictionaryResource/);
  assert.match(declarations, /reduceBatches/);
});

// C1/C3/C5: a delegated shim earns its keep only if it re-exports the entire
// contract surface. Pin the streaming cursor (pull iterator + push reducer +
// batch paging + disposal), the phonetic automata, and the borrowed-resource
// transducer entry point, so a future trim of the umbrella surface cannot slip
// past this facade unnoticed.
test("delegated facade re-exports the full streaming and phonetic surface", async () => {
  const declarations = await readFile(new URL("../index.d.ts", import.meta.url), "utf8");
  // C5: one cursor is both a pull iterator and a push reducer, with paging.
  assert.match(declarations, /interface QueryCursor extends IterableIterator<Match>/);
  for (const member of ["nextBatch(", "reduceBatches<", "close()"]) {
    assert.ok(declarations.includes(member), `QueryCursor must declare ${member}`);
  }
  // C3/C6: phonetic pattern matching and rewrite-rule application travel here.
  assert.match(declarations, /interface PhoneticPattern[\s\S]*?matches\(input: string\): boolean/);
  assert.match(declarations, /interface PhoneticRuleSet[\s\S]*?apply\(input: string\): string/);
  // C1: the transducer only ever borrows a DictionaryResource; it never
  // constructs storage, keeping ownership on the delegated shared JavaScript runtime.
  assert.match(declarations, /transducer\(dictionary: DictionaryResource/);
});
