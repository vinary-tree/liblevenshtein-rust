import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url)));

test("all project facades share the exact umbrella runtime", () => {
  assert.equal(packageJson.dependencies["@vinary-tree/vinary-tree"], packageJson.version);
  assert.equal(packageJson.dependencies["@vinary-tree/interop"], "0.1.0");
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
