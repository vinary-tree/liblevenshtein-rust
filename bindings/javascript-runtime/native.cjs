"use strict";
const { existsSync } = require("node:fs");
const { join } = require("node:path");
const platform = `${process.platform}-${process.arch}`;
const prebuiltAddon = join(__dirname, "native", "prebuilds", platform, "vinary_tree_native.node");
const developmentAddon = join(__dirname, "native", "build", "Release", "vinary_tree_native.node");
const addon = existsSync(prebuiltAddon) ? prebuiltAddon : developmentAddon;
let ffi;
try {
  ffi = require(addon);
} catch (cause) {
  throw new Error(
    `@vinary-tree/vinary-tree has no usable native addon for ${platform}; `
      + "install a supported prebuilt package or build the addon from source",
    { cause },
  );
}
const runtimeIdentity = Object.freeze({ implementation: "vinary-tree-node-napi-v1" });
const domains = new Map([["byte", 1], ["unicode", 2], ["u64", 3]]);
const algorithms = new Map([
  ["standard", 0], ["transposition", 1], ["merge-and-split", 2], ["damerau-levenshtein", 3],
]);
const orders = new Map([["traversal", 0], ["distance-then-term", 1]]);
const duallityKinds = new Map([
  ["levenshtein", 0], ["universal-standard", 1], ["universal-transposition", 2],
  ["universal-merge-and-split", 3], ["generalized-standard", 4],
  ["generalized-transposition", 5], ["generalized-merge-and-split", 6],
  ["generalized-phonetic", 7], ["fzf", 8],
]);
const weightDomains = new Map([
  [1, "tropical-f64"], [2, "log-f64"], [3, "probability-f64"], [4, "arctic-f64"],
  [5, "signed-tropical-f64"], [6, "count-f64"], [7, "boolean-f64"],
]);

function select(table, value, kind) {
  const selected = table.get(value);
  if (selected === undefined) throw new TypeError(`unknown ${kind}: ${value}`);
  return selected;
}

class Dictionary {
  #handle;
  #kind;
  constructor(handle, unitDomain, kind) {
    this.#handle = handle;
    this.#kind = kind;
    Object.defineProperties(this, {
      interfaceId: { value: "vt.dictionary.v1", enumerable: true },
      runtimeIdentity: { value: runtimeIdentity, enumerable: true },
      unitDomain: { value: unitDomain, enumerable: true },
      valueDomain: { value: "optional-u64", enumerable: true },
    });
  }
  get _handle() {
    if (this.#handle === null) throw new Error("dictionary is closed");
    return this.#handle;
  }
  get size() { return ffi.dictionaryLen(this._handle); }
  put(term, value = null) { return ffi.dictionaryPutText(this._handle, term, value); }
  putU64(term, value = null) { return ffi.dictionaryPutU64(this._handle, term, value); }
  remove(term) { return ffi.dictionaryRemoveText(this._handle, term); }
  removeU64(term) { return ffi.dictionaryRemoveU64(this._handle, term); }
  get(term) { return ffi.dictionaryGetText(this._handle, term); }
  getU64(term) { return ffi.dictionaryGetU64(this._handle, term); }
  has(term) { return this.get(term).found; }
  hasU64(term) { return this.getU64(term).found; }
  clear() { return ffi.dictionaryClear(this._handle); }
  compact() { return ffi.dictionaryCompact(this._handle); }
  checkpoint() { return ffi.dictionaryCheckpoint(this._handle); }
  containsSubstring(term) { return ffi.containsSubstring(this._handle, term); }
  substringFrequency(term) { return ffi.substringFrequency(this._handle, term); }
  close() {
    if (this.#handle !== null) {
      ffi.dictionaryClose(this.#handle);
      this.#handle = null;
    }
  }
  get kind() { return this.#kind; }
}

class QueryCursor {
  #handle;
  #pending = [];
  constructor(handle) { this.#handle = handle; }
  [Symbol.iterator]() { return this; }
  next() {
    if (this.#pending.length === 0) this.#pending = this.nextBatch(256);
    return this.#pending.length === 0
      ? { done: true, value: undefined }
      : { done: false, value: this.#pending.shift() };
  }
  nextBatch(maximum) {
    if (!Number.isSafeInteger(maximum) || maximum <= 0) throw new RangeError("batch size must be positive");
    return this.#handle === null ? [] : ffi.cursorNextBatch(this.#handle, maximum);
  }
  reduceBatches(reducer, initial, batchSize = 256) {
    let accumulator = initial;
    for (;;) {
      const batch = this.nextBatch(batchSize);
      if (batch.length === 0) return accumulator;
      accumulator = reducer(accumulator, batch);
    }
  }
  close() {
    if (this.#handle !== null) {
      ffi.cursorClose(this.#handle);
      this.#handle = null;
      this.#pending = [];
    }
  }
}

class PhoneticPattern {
  #handle;
  constructor(handle) { this.#handle = handle; }
  get _handle() {
    if (this.#handle === null) throw new Error("phonetic pattern is closed");
    return this.#handle;
  }
  get size() { return ffi.patternSize(this._handle); }
  matches(input) { return ffi.patternMatches(this._handle, input); }
  close() {
    if (this.#handle !== null) {
      ffi.patternClose(this.#handle);
      this.#handle = null;
    }
  }
}

class PhoneticRuleSet {
  #handle;
  constructor(handle) { this.#handle = handle; }
  get size() {
    if (this.#handle === null) throw new Error("phonetic rules are closed");
    return ffi.rulesLen(this.#handle);
  }
  apply(input) {
    if (this.#handle === null) throw new Error("phonetic rules are closed");
    return ffi.rulesApply(this.#handle, input);
  }
  close() {
    if (this.#handle !== null) {
      ffi.rulesClose(this.#handle);
      this.#handle = null;
    }
  }
}

class Transducer {
  #handle;
  constructor(dictionary, algorithm) {
    if (dictionary?.runtimeIdentity !== runtimeIdentity || dictionary.interfaceId !== "vt.dictionary.v1") {
      throw new TypeError("dictionary belongs to a different Vinary Tree runtime");
    }
    this.#handle = ffi.transducerNew(dictionary._handle, select(algorithms, algorithm, "algorithm"));
  }
  query(input, maximumDistance, order = "traversal") {
    if (input instanceof PhoneticPattern) {
      return new QueryCursor(ffi.queryPattern(this.#handle, input._handle, maximumDistance));
    }
    const selectedOrder = select(orders, order, "query order");
    if (typeof input === "string") {
      return new QueryCursor(ffi.queryText(this.#handle, input, maximumDistance, selectedOrder));
    }
    if (input instanceof BigUint64Array) {
      return new QueryCursor(ffi.queryU64(this.#handle, input, maximumDistance, selectedOrder));
    }
    if (input instanceof Uint8Array) {
      return new QueryCursor(ffi.queryBytes(this.#handle, input, maximumDistance, selectedOrder));
    }
    throw new TypeError("query requires text, Uint8Array, BigUint64Array, or a phonetic pattern");
  }
  close() {
    if (this.#handle !== null) {
      ffi.transducerClose(this.#handle);
      this.#handle = null;
    }
  }
}

class Wfst {
  #handle;
  constructor(handle) {
    this.#handle = handle;
    Object.defineProperties(this, {
      interfaceId: { value: "vt.scalar-wfst.1", enumerable: true },
      runtimeIdentity: { value: runtimeIdentity, enumerable: true },
    });
  }
  get _handle() {
    if (this.#handle === null) throw new Error("WFST is closed");
    return this.#handle;
  }
  get weightDomain() {
    const value = ffi.wfstWeightDomain(this._handle);
    const name = weightDomains.get(value);
    if (name === undefined) throw new Error(`unknown WFST weight domain ${value}`);
    return name;
  }
  start() { return ffi.wfstStart(this._handle); }
  state(state) { return ffi.wfstState(this._handle, state); }
  close() {
    if (this.#handle !== null) {
      ffi.wfstClose(this.#handle);
      this.#handle = null;
    }
  }
}

class WfstBuilder {
  #handle = ffi.wfstBuilderNew();
  get _handle() {
    if (this.#handle === null) throw new Error("WFST builder is closed");
    return this.#handle;
  }
  addState() { return ffi.wfstBuilderAddState(this._handle); }
  setStart(state) { ffi.wfstBuilderSetStart(this._handle, state); }
  setFinal(state, weight = 0) { ffi.wfstBuilderSetFinal(this._handle, state, weight); }
  addArc(from, input, output, to, weight = 0) {
    const label = (value) => {
      if (value === null || value === undefined) return [0n, 0];
      if (typeof value !== "string" || [...value].length !== 1) {
        throw new TypeError("arc label must contain one Unicode scalar or be null");
      }
      return [BigInt(value.codePointAt(0)), 1];
    };
    const [inputLabel, hasInput] = label(input);
    const [outputLabel, hasOutput] = label(output);
    ffi.wfstBuilderAddArc(
      this._handle, from, inputLabel, hasInput, outputLabel, hasOutput, to, weight,
    );
  }
  build() { return new Wfst(ffi.wfstBuilderBuild(this._handle)); }
  close() {
    if (this.#handle !== null) {
      ffi.wfstBuilderClose(this.#handle);
      this.#handle = null;
    }
  }
}

const libdictenstein = Object.freeze({
  runtimeIdentity,
  dynamicDawg(unitDomain = "unicode") {
    return new Dictionary(ffi.dynamicDawgNew(select(domains, unitDomain, "unit domain")), unitDomain, "dynamic-dawg");
  },
  doubleArrayTrie(entries, unitDomain = "unicode") {
    return new Dictionary(ffi.doubleArrayTrieNew(select(domains, unitDomain, "unit domain"), entries), unitDomain, "double-array-trie");
  },
  scdawg(unitDomain = "unicode") {
    return new Dictionary(ffi.scdawgNew(select(domains, unitDomain, "unit domain")), unitDomain, "scdawg");
  },
  createPersistentARTrie(path, unitDomain = "unicode") {
    return new Dictionary(ffi.persistentARTrieCreate(select(domains, unitDomain, "unit domain"), path), unitDomain, "persistent-artrie");
  },
  openPersistentARTrie(path, unitDomain = "unicode") {
    return new Dictionary(ffi.persistentARTrieOpen(select(domains, unitDomain, "unit domain"), path), unitDomain, "persistent-artrie");
  },
});

const liblevenshtein = Object.freeze({
  runtimeIdentity,
  transducer(dictionary, algorithm = "standard") { return new Transducer(dictionary, algorithm); },
  phoneticPattern(source) { return new PhoneticPattern(ffi.patternCompileRegex(source)); },
  llrePattern(source) { return new PhoneticPattern(ffi.patternCompileLlre(source)); },
  phoneticRules(source) { return new PhoneticRuleSet(ffi.rulesCompile(source)); },
  levenshteinDistance: ffi.levenshteinDistance,
  levenshteinDistanceThreshold: ffi.levenshteinDistanceThreshold,
  damerauDistance: ffi.damerauDistance,
  damerauDistanceThreshold: ffi.damerauDistanceThreshold,
  trueDamerauDistance: ffi.trueDamerauDistance,
  trueDamerauDistanceThreshold: ffi.trueDamerauDistanceThreshold,
});

const llingLlang = Object.freeze({
  runtimeIdentity,
  vectorWfst() { return new WfstBuilder(); },
  compose(first, second) {
    if (first?.runtimeIdentity !== runtimeIdentity || second?.runtimeIdentity !== runtimeIdentity) {
      throw new TypeError("WFST belongs to a different Vinary Tree runtime");
    }
    return new Wfst(ffi.wfstCompose(first._handle, second._handle));
  },
});

const duallity = Object.freeze({
  runtimeIdentity,
  wfst(dictionary, query, maximumDistance, algorithm = "standard", kind = "levenshtein") {
    if (dictionary?.runtimeIdentity !== runtimeIdentity || dictionary.interfaceId !== "vt.dictionary.v1") {
      throw new TypeError("dictionary belongs to a different Vinary Tree runtime");
    }
    return new Wfst(ffi.duallityWfstNew(
      dictionary._handle,
      query,
      maximumDistance,
      select(algorithms, algorithm, "algorithm"),
      select(duallityKinds, kind, "duallity WFST kind"),
    ));
  },
});

const runtime = Object.freeze({ runtimeIdentity, libdictenstein, liblevenshtein, llingLlang, duallity });
module.exports = Object.assign({}, runtime, { default: runtime });
