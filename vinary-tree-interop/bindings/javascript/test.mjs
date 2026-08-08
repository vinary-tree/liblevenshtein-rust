import assert from "node:assert/strict";
import test from "node:test";

import { assertDictionaryResource, assertSameRuntime, assertWfstResource } from "./index.mjs";

test("runtime identity prevents cross-instance handles", () => {
  const runtimeIdentity = {};
  const resource = {
    runtimeIdentity,
    interfaceId: "vt.dictionary.v1",
    unitDomain: "unicode",
    valueDomain: "optional-u64",
    close() {},
  };
  assert.doesNotThrow(() => assertSameRuntime(resource, runtimeIdentity));
  assert.throws(() => assertSameRuntime(resource, {}), /different/);
  assert.doesNotThrow(() => assertDictionaryResource(resource));
  assert.throws(
    () => assertDictionaryResource({ ...resource, interfaceId: "vt.dictionary.v2" }),
    /vt\.dictionary\.v1/,
  );
});

test("scalar WFST resources are independently versioned", () => {
  const resource = {
    runtimeIdentity: {},
    interfaceId: "vt.scalar-wfst.1",
    weightDomain: "tropical-f64",
    close() {},
  };
  assert.doesNotThrow(() => assertWfstResource(resource));
  assert.throws(
    () => assertWfstResource({ ...resource, interfaceId: "vt.scalar-wfst.2" }),
    /vt\.scalar-wfst\.1/,
  );
  assert.throws(() => assertDictionaryResource(resource), /vt\.dictionary\.v1/);
});
