const DEFAULT_BATCH_SIZE = 256;

function defineResourceMetadata(resource, runtimeIdentity, unitDomain) {
  Object.defineProperties(resource, {
    interfaceId: { value: "vt.dictionary.v1", enumerable: true },
    runtimeIdentity: { value: runtimeIdentity, enumerable: true },
    valueDomain: { value: "optional-u64", enumerable: true },
    unitDomain: { value: unitDomain, enumerable: true },
  });
  return resource;
}

function requireDictionary(dictionary, runtimeIdentity) {
  if (dictionary?.interfaceId !== "vt.dictionary.v1") {
    throw new TypeError("resource does not implement vt.dictionary.v1");
  }
  if (dictionary.runtimeIdentity !== runtimeIdentity) {
    throw new TypeError("resource belongs to a different Vinary Tree runtime");
  }
  return dictionary;
}

function requireWfst(wfst, runtimeIdentity) {
  if (wfst?.interfaceId !== "vt.scalar-wfst.1") {
    throw new TypeError("resource does not implement vt.scalar-wfst.1");
  }
  if (wfst.runtimeIdentity !== runtimeIdentity) {
    throw new TypeError("WFST belongs to a different Vinary Tree runtime");
  }
  return wfst;
}

function installCursorProtocol(raw) {
  if (raw.QueryCursor.prototype[Symbol.iterator]) return;
  const rawNext = raw.QueryCursor.prototype.next;
  Object.defineProperties(raw.QueryCursor.prototype, {
    [Symbol.iterator]: {
      value() { return this; },
    },
    next: {
      value() {
        const result = rawNext.call(this);
        if (result.done) this.close();
        return result;
      },
    },
    reduceBatches: {
      value(reducer, initial, batchSize = DEFAULT_BATCH_SIZE) {
        if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
          throw new RangeError("batchSize must be a positive safe integer");
        }
        let accumulator = initial;
        try {
          for (;;) {
            const batch = this.nextBatch(batchSize);
            if (batch.length === 0) return accumulator;
            accumulator = reducer(accumulator, batch);
          }
        } finally {
          this.close();
        }
      },
    },
    return: {
      value() { this.close(); return { done: true, value: undefined }; },
    },
    [Symbol.dispose]: { value() { this.close(); } },
  });
}

class MaterializedEntryCursor {
  #entries;
  #offset = 0;
  constructor(entries) {
    this.#entries = entries;
    this.size = entries.length;
    this.identity = null;
  }
  [Symbol.iterator]() { return this; }
  next() {
    if (this.#offset >= this.#entries.length) {
      this.close();
      return { done: true, value: undefined };
    }
    return { done: false, value: this.#entries[this.#offset++] };
  }
  nextBatch(maximum) {
    if (!Number.isSafeInteger(maximum) || maximum <= 0) {
      throw new RangeError("batch size must be a positive safe integer");
    }
    const batch = this.#entries.slice(this.#offset, this.#offset + maximum);
    this.#offset += batch.length;
    return batch;
  }
  reduceBatches(reducer, initial, batchSize = DEFAULT_BATCH_SIZE) {
    let accumulator = initial;
    try {
      for (;;) {
        const batch = this.nextBatch(batchSize);
        if (batch.length === 0) return accumulator;
        accumulator = reducer(accumulator, batch);
      }
    } finally {
      this.close();
    }
  }
  return() { this.close(); return { done: true, value: undefined }; }
  close() { this.#offset = this.#entries.length; }
  [Symbol.dispose]() { this.close(); }
}

class WasmEntryCursor {
  #raw;
  #pending = [];
  #offset = 0;
  constructor(raw) {
    this.#raw = raw;
    this.size = raw.size;
    this.identity = null;
  }
  [Symbol.iterator]() { return this; }
  next() {
    if (this.#offset >= this.#pending.length) {
      this.#pending = this.#fetch(DEFAULT_BATCH_SIZE);
      this.#offset = 0;
    }
    if (this.#pending.length === 0) {
      this.close();
      return { done: true, value: undefined };
    }
    return { done: false, value: this.#pending[this.#offset++] };
  }
  nextBatch(maximum) {
    if (!Number.isSafeInteger(maximum) || maximum <= 0) {
      throw new RangeError("batch size must be a positive safe integer");
    }
    const result = [];
    while (result.length < maximum) {
      while (this.#offset < this.#pending.length && result.length < maximum) {
        result.push(this.#pending[this.#offset++]);
      }
      if (result.length === maximum || this.#raw === null) break;
      this.#pending = this.#fetch(maximum - result.length);
      this.#offset = 0;
      if (this.#pending.length === 0) {
        this.close();
        break;
      }
    }
    return result;
  }
  #fetch(maximum) {
    return this.#raw === null ? [] : Array.from(this.#raw.nextBatch(maximum));
  }
  reduceBatches(reducer, initial, batchSize = DEFAULT_BATCH_SIZE) {
    let accumulator = initial;
    try {
      for (;;) {
        const batch = this.nextBatch(batchSize);
        if (batch.length === 0) return accumulator;
        accumulator = reducer(accumulator, batch);
      }
    } finally {
      this.close();
    }
  }
  return() { this.close(); return { done: true, value: undefined }; }
  close() {
    if (this.#raw !== null) {
      this.#raw.close();
      this.#raw = null;
      this.#pending = [];
      this.#offset = 0;
    }
  }
  [Symbol.dispose]() { this.close(); }
}

function installDictionaryProtocol(raw) {
  const prototype = raw.Dictionary.prototype;
  if (prototype.lookup) return;
  const rawPut = prototype.put;
  const rawPutBytes = prototype.putBytes;
  const rawPutU64 = prototype.putU64;
  const rawRemove = prototype.remove;
  const rawRemoveBytes = prototype.removeBytes;
  const rawRemoveU64 = prototype.removeU64;
  const rawLookup = prototype.get;
  const rawLookupBytes = prototype.getBytes;
  const rawLookupU64 = prototype.getU64;
  const rawSnapshotEntries = prototype.snapshotEntries;
  const rawOpenEntryStream = prototype.openEntryStream;

  const select = (term, text, bytes, u64) => {
    if (term instanceof BigUint64Array) return u64;
    if (term instanceof Uint8Array) return bytes;
    if (typeof term === "string") return text;
    throw new TypeError("dictionary key must be a string, Uint8Array, or BigUint64Array");
  };

  Object.defineProperties(prototype, {
    put: {
      value(term, value = null) {
        return select(term, rawPut, rawPutBytes, rawPutU64).call(this, term, value);
      },
    },
    set: {
      value(term, value = null) { this.put(term, value); return this; },
    },
    remove: {
      value(term) {
        return select(term, rawRemove, rawRemoveBytes, rawRemoveU64).call(this, term);
      },
    },
    delete: { value(term) { return this.remove(term); } },
    lookup: {
      value(term) {
        return select(term, rawLookup, rawLookupBytes, rawLookupU64).call(this, term);
      },
    },
    lookupU64: { value(term) { return rawLookupU64.call(this, term); } },
    get: {
      value(term) {
        const result = this.lookup(term);
        return result.found ? result.value : undefined;
      },
    },
    getU64: {
      value(term) {
        const result = this.lookupU64(term);
        return result.found ? result.value : undefined;
      },
    },
    has: { value(term) { return this.lookup(term).found; } },
    hasU64: { value(term) { return this.lookupU64(term).found; } },
    snapshotEntries: {
      value() {
        return Object.freeze(Array.from(
          rawSnapshotEntries.call(this),
          ([key, value]) => Object.freeze([key, value]),
        ));
      },
    },
    entries: { value() { return this.snapshotEntries()[Symbol.iterator](); } },
    [Symbol.iterator]: { value() { return this.entries(); } },
    keys: {
      value: function* keys() { for (const [key] of this) yield key; },
    },
    values: {
      value: function* values() { for (const [, value] of this) yield value; },
    },
    streamEntries: {
      value() {
        return typeof rawOpenEntryStream === "function"
          ? new WasmEntryCursor(rawOpenEntryStream.call(this))
          : new MaterializedEntryCursor(this.snapshotEntries());
      },
    },
    forEach: {
      value(callback, thisArg = undefined) {
        for (const [key, value] of this) callback.call(thisArg, value, key, this);
      },
    },
    toMap: {
      value() {
        if (this.unitDomain !== "unicode") {
          throw new TypeError("toMap is defined only for value-equal JavaScript string keys");
        }
        return new Map(this);
      },
    },
    [Symbol.dispose]: { value() { this.close(); } },
  });
}

function query(transducer, input, maximumDistance, order = "traversal") {
  if (typeof input === "string") {
    return transducer.queryText(input, maximumDistance, order);
  }
  if (input instanceof Uint8Array && !(input instanceof BigUint64Array)) {
    return transducer.queryBytes(input, maximumDistance);
  }
  if (input instanceof BigUint64Array) {
    return transducer.queryU64(input, maximumDistance);
  }
  if (input instanceof transducer.constructor.__phoneticPatternClass) {
    return transducer.queryPattern(input, maximumDistance);
  }
  throw new TypeError("query must be a string, Uint8Array, BigUint64Array, or PhoneticPattern");
}

/** Build all public project namespaces over exactly one initialized runtime. */
export function createRuntime(raw) {
  installCursorProtocol(raw);
  installDictionaryProtocol(raw);
  const runtimeIdentity = Object.freeze({ implementation: "vinary-tree-wasm-v1" });

  Object.defineProperty(raw.Transducer, "__phoneticPatternClass", {
    value: raw.PhoneticPattern,
  });
  if (!raw.Transducer.prototype.query) {
    Object.defineProperty(raw.Transducer.prototype, "query", {
      value(input, maximumDistance, order) {
        return query(this, input, maximumDistance, order);
      },
    });
  }

  Object.defineProperties(raw.Wfst.prototype, {
    interfaceId: { value: "vt.scalar-wfst.1" },
    runtimeIdentity: { value: runtimeIdentity },
  });

  const libdictenstein = Object.freeze({
    runtimeIdentity,
    dynamicDawg(unitDomain = "unicode") {
      return defineResourceMetadata(raw.Dictionary.dynamicDawg(unitDomain), runtimeIdentity, unitDomain);
    },
    doubleArrayTrie(entries, unitDomain = "unicode") {
      return defineResourceMetadata(
        raw.Dictionary.doubleArrayTrie(entries, unitDomain),
        runtimeIdentity,
        unitDomain,
      );
    },
    scdawg(unitDomain = "unicode") {
      return defineResourceMetadata(raw.Dictionary.scdawg(unitDomain), runtimeIdentity, unitDomain);
    },
  });

  const liblevenshtein = Object.freeze({
    runtimeIdentity,
    transducer(dictionary, algorithm = "standard") {
      return new raw.Transducer(requireDictionary(dictionary, runtimeIdentity), algorithm);
    },
    phoneticPattern(source) {
      return raw.PhoneticPattern.compileRegex(source);
    },
    llrePattern(source) {
      return raw.PhoneticPattern.compileLlre(source);
    },
    phoneticRules(source) {
      return raw.PhoneticRuleSet.compile(source);
    },
    levenshteinDistance: raw.levenshteinDistance,
    levenshteinDistanceThreshold: raw.levenshteinDistanceThreshold,
    damerauDistance: raw.damerauDistance,
    damerauDistanceThreshold: raw.damerauDistanceThreshold,
    trueDamerauDistance: raw.trueDamerauDistance,
    trueDamerauDistanceThreshold: raw.trueDamerauDistanceThreshold,
  });

  const llingLlang = Object.freeze({
    runtimeIdentity,
    vectorWfst() { return new raw.WfstBuilder(); },
    compose(first, second) {
      return raw.composeWfst(
        requireWfst(first, runtimeIdentity),
        requireWfst(second, runtimeIdentity),
      );
    },
  });

  const duallity = Object.freeze({
    runtimeIdentity,
    wfst(
      dictionary,
      query,
      maximumDistance,
      algorithm = "standard",
      kind = "levenshtein",
    ) {
      return raw.createDuallityWfst(
        requireDictionary(dictionary, runtimeIdentity),
        query,
        maximumDistance,
        algorithm,
        kind,
      );
    },
  });

  return Object.freeze({ runtimeIdentity, libdictenstein, liblevenshtein, llingLlang, duallity });
}
