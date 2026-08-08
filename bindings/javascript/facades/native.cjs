"use strict";
const namespace = require("@vinary-tree/vinary-tree").liblevenshtein;
const { assertDictionaryResource, assertSameRuntime } = require("@vinary-tree/interop");
if (!namespace || !namespace.runtimeIdentity) {
  throw new Error("@vinary-tree/vinary-tree does not expose a runtime identity");
}
function transducer(dictionary, algorithm) {
  assertDictionaryResource(dictionary);
  assertSameRuntime(dictionary, namespace.runtimeIdentity);
  return namespace.transducer(dictionary, algorithm);
}
const facade = Object.assign({}, namespace, { transducer });
module.exports = Object.assign(facade, { default: facade });
