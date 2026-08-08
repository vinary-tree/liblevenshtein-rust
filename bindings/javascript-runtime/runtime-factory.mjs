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
  Object.defineProperties(raw.QueryCursor.prototype, {
    [Symbol.iterator]: {
      value() { return this; },
    },
    reduceBatches: {
      value(reducer, initial, batchSize = DEFAULT_BATCH_SIZE) {
        if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
          throw new RangeError("batchSize must be a positive safe integer");
        }
        let accumulator = initial;
        for (;;) {
          const batch = this.nextBatch(batchSize);
          if (batch.length === 0) return accumulator;
          accumulator = reducer(accumulator, batch);
        }
      },
    },
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
