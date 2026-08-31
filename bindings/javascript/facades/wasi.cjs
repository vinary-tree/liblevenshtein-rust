"use strict";
const namespace = require("@vinary-tree/javascript-runtime/wasi").liblevenshtein;
const { assertDictionaryResource, assertSameRuntime } = require("@vinary-tree/vinary-tree-interop");
function transducer(dictionary, algorithm) {
  assertDictionaryResource(dictionary);
  assertSameRuntime(dictionary, namespace.runtimeIdentity);
  return namespace.transducer(dictionary, algorithm);
}
const queryCache = namespace.queryCache.bind(namespace);
const facade = Object.assign({}, namespace, { transducer, queryCache });
module.exports = Object.assign(facade, { default: facade });
