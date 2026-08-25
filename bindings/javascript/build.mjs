import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const packageJson = JSON.parse(await readFile(new URL("./package.json", import.meta.url)));
assert.equal(packageJson.dependencies["@vinary-tree/vinary-tree"], packageJson.version);
assert.equal(packageJson.dependencies["@vinary-tree/interop"], "4.0.0-rc.4");
for (const path of [".", "./typescript", "./clojurescript", "./wasm", "./wasi"]) {
  assert.ok(packageJson.exports[path], `missing ${path} facade`);
}
console.log("validated shared-runtime liblevenshtein facades");
