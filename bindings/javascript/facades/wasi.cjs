"use strict";
const namespace = require("@vinary-tree/vinary-tree/wasi").liblevenshtein;
const { assertDictionaryResource, assertSameRuntime } = require("@vinary-tree/interop");
function transducer(dictionary, algorithm) {
  assertDictionaryResource(dictionary);
  assertSameRuntime(dictionary, namespace.runtimeIdentity);
  return namespace.transducer(dictionary, algorithm);
}
const facade = Object.assign({}, namespace, { transducer });
module.exports = Object.assign(facade, { default: facade });
