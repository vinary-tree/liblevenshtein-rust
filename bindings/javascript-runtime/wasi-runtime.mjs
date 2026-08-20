import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";

const FAILURE = 0xffff_ffff;
const RECORD_SIZE = 32;
const ENTRY_RECORD_HEADER_SIZE = 24;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

const DOMAINS = new Map([["byte", 0], ["unicode", 1], ["u64", 2]]);
const ALGORITHMS = new Map([
  ["standard", 0],
  ["transposition", 1],
  ["merge-and-split", 2],
  ["damerau-levenshtein", 3],
]);
const ORDERS = new Map([["traversal", 0], ["distance-then-term", 1]]);
const DUALLITY_KINDS = new Map([
  ["levenshtein", 0], ["universal-standard", 1], ["universal-transposition", 2],
  ["universal-merge-and-split", 3], ["generalized-standard", 4],
  ["generalized-transposition", 5], ["generalized-merge-and-split", 6],
  ["generalized-phonetic", 7], ["fzf", 8],
]);
const WEIGHT_DOMAINS = new Map([
  [1, "tropical-f64"], [2, "log-f64"], [3, "probability-f64"], [4, "arctic-f64"],
  [5, "signed-tropical-f64"], [6, "count-f64"], [7, "boolean-f64"],
]);

/** Instantiate an isolated WASI runtime with explicit guest-to-host preopens. */
export async function createWasiRuntime({
  preopens = { "/workspace": process.cwd() },
  wasm = new URL("./generated/wasi/vinary_tree.wasm", import.meta.url),
} = {}) {
  const wasi = new WASI({ version: "preview1", args: [], env: {}, preopens });
  const source = wasm instanceof WebAssembly.Module ? wasm : await readFile(wasm);
  const module = source instanceof WebAssembly.Module ? source : await WebAssembly.compile(source);
  const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
  wasi.initialize(instance);
  const ffi = instance.exports;
  const runtimeIdentity = Object.freeze({ implementation: "vinary-tree-wasi-preview1-v1", instance });

  function view() { return new DataView(ffi.memory.buffer); }
  function bytes() { return new Uint8Array(ffi.memory.buffer); }
  function failure(value) {
    const unsigned = value >>> 0;
    if (unsigned !== FAILURE) return unsigned;
    const pointer = ffi.vt_error_pointer();
    const length = ffi.vt_error_length();
    throw new Error(decoder.decode(bytes().slice(pointer, pointer + length)));
  }
  function withBytes(value, operation) {
    const encoded = typeof value === "string" ? encoder.encode(value) : value;
    const pointer = ffi.vt_alloc(encoded.length);
    try {
      bytes().set(encoded, pointer);
      return operation(pointer, encoded.length);
    } finally {
      ffi.vt_dealloc(pointer, encoded.length);
    }
  }
  function withTokens(value, operation) {
    if (!(value instanceof BigUint64Array)) {
      throw new TypeError("u64 dictionary keys require BigUint64Array");
    }
    const encoded = new Uint8Array(value.length * 8);
    const data = new DataView(encoded.buffer);
    for (let index = 0; index < value.length; index += 1) {
      data.setBigUint64(index * 8, value[index], true);
    }
    return withBytes(encoded, (pointer) => operation(pointer, value.length));
  }
  function select(table, value, kind) {
    const selected = table.get(value);
    if (selected === undefined) throw new TypeError(`unknown ${kind}: ${value}`);
    return selected;
  }

  class Dictionary {
    #handle;
    #persistent;
    constructor(handle, unitDomain, persistent) {
      this.#handle = failure(handle);
      this.#persistent = persistent;
      Object.defineProperties(this, {
        interfaceId: { value: "vt.dictionary.v1", enumerable: true },
        runtimeIdentity: { value: runtimeIdentity, enumerable: true },
        unitDomain: { value: unitDomain, enumerable: true },
        valueDomain: { value: "optional-u64", enumerable: true },
      });
    }
    get _handle() {
      if (this.#handle === 0) throw new Error("dictionary is closed");
      return this.#handle;
    }
    get size() { return failure(ffi.vt_dictionary_len(this._handle)); }
    put(term, value = null) {
      if (value !== null && (typeof value !== "bigint" || value < 0n || value > 0xffff_ffff_ffff_ffffn)) {
        throw new RangeError("dictionary value must be null or an unsigned 64-bit bigint");
      }
      if (term instanceof BigUint64Array) {
        return withTokens(term, (pointer, length) => Boolean(failure(
          ffi.vt_dictionary_put_u64(this._handle, pointer, length, Number(value !== null), value ?? 0n),
        )));
      }
      if (typeof term !== "string" && !(term instanceof Uint8Array)) {
        throw new TypeError("dictionary key must be a string, Uint8Array, or BigUint64Array");
      }
      return withBytes(term, (pointer, length) => Boolean(failure(
        ffi.vt_dictionary_put_text(this._handle, pointer, length, Number(value !== null), value ?? 0n),
      )));
    }
    putU64(term, value = null) { return this.put(term, value); }
    set(term, value = null) { this.put(term, value); return this; }
    remove(term) {
      if (term instanceof BigUint64Array) {
        return withTokens(term, (pointer, length) => Boolean(failure(
          ffi.vt_dictionary_remove_u64(this._handle, pointer, length),
        )));
      }
      return withBytes(term, (pointer, length) => Boolean(failure(
        ffi.vt_dictionary_remove_text(this._handle, pointer, length),
      )));
    }
    removeU64(term) { return this.remove(term); }
    delete(term) { return this.remove(term); }
    lookup(term) {
      const decode = (termPointer, termLength, operation) => {
        const output = ffi.vt_alloc(16);
        try {
          failure(operation(this._handle, termPointer, termLength, output));
          const memory = view();
          const found = memory.getUint32(output, true) !== 0;
          const present = memory.getUint32(output + 4, true) !== 0;
          return { found, value: present ? memory.getBigUint64(output + 8, true) : null };
        } finally {
          ffi.vt_dealloc(output, 16);
        }
      };
      return term instanceof BigUint64Array
        ? withTokens(term, (pointer, length) => decode(pointer, length, ffi.vt_dictionary_get_u64))
        : withBytes(term, (pointer, length) => decode(pointer, length, ffi.vt_dictionary_get_text));
    }
    lookupU64(term) { return this.lookup(term); }
    get(term) { const result = this.lookup(term); return result.found ? result.value : undefined; }
    getU64(term) { return this.get(term); }
    has(term) { return this.lookup(term).found; }
    hasU64(term) { return this.has(term); }
    streamEntries() { return new DictionaryEntryCursor(ffi.vt_dictionary_entries_open(this._handle)); }
    snapshotEntries() {
      const result = [];
      const cursor = this.streamEntries();
      try {
        for (const [key, value] of cursor) result.push(Object.freeze([key, value]));
      } finally {
        cursor.close();
      }
      return Object.freeze(result);
    }
    entries() { return this.snapshotEntries()[Symbol.iterator](); }
    [Symbol.iterator]() { return this.entries(); }
    *keys() { for (const [key] of this) yield key; }
    *values() { for (const [, value] of this) yield value; }
    forEach(callback, thisArg = undefined) {
      for (const [key, value] of this) callback.call(thisArg, value, key, this);
    }
    toMap() {
      if (this.unitDomain !== "unicode") {
        throw new TypeError("toMap is defined only for value-equal JavaScript string keys");
      }
      return new Map(this);
    }
    clear() { failure(ffi.vt_dictionary_clear(this._handle)); }
    compact() { return failure(ffi.vt_dictionary_compact(this._handle)); }
    checkpoint() {
      if (!this.#persistent) throw new Error("in-memory dictionaries do not checkpoint");
      failure(ffi.vt_dictionary_checkpoint(this._handle));
    }
    close() {
      if (this.#handle !== 0) {
        ffi.vt_handle_close(this.#handle);
        this.#handle = 0;
      }
    }
    [Symbol.dispose]() { this.close(); }
  }

  class DictionaryEntryCursor {
    #handle;
    #pending = [];
    #offset = 0;
    constructor(handle) {
      this.#handle = failure(handle);
      this.size = failure(ffi.vt_entry_cursor_len(this.#handle));
      this.identity = null;
    }
    [Symbol.iterator]() { return this; }
    next() {
      if (this.#offset >= this.#pending.length) {
        this.#pending = this.#fetch(256);
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
        if (result.length === maximum || this.#handle === 0) break;
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
      if (this.#handle === 0) return [];
      const count = failure(ffi.vt_entry_cursor_next_batch(this.#handle, maximum));
      if (count === 0) return [];
      let record = failure(ffi.vt_entry_cursor_batch_pointer(this.#handle));
      const memory = view();
      const output = [];
      for (let index = 0; index < count; index += 1) {
        const payloadLength = memory.getUint32(record, true);
        const domain = memory.getUint32(record + 4, true);
        const present = memory.getUint32(record + 8, true) !== 0;
        const value = present ? memory.getBigUint64(record + 16, true) : null;
        const payload = record + ENTRY_RECORD_HEADER_SIZE;
        let key;
        if (domain === 0) {
          key = bytes().slice(payload, payload + payloadLength);
        } else if (domain === 1) {
          key = decoder.decode(bytes().slice(payload, payload + payloadLength));
        } else {
          if (payloadLength % 8 !== 0) throw new Error("invalid u64 dictionary entry payload");
          key = new BigUint64Array(payloadLength / 8);
          for (let token = 0; token < key.length; token += 1) {
            key[token] = memory.getBigUint64(payload + token * 8, true);
          }
        }
        output.push([key, value]);
        record = payload + payloadLength;
      }
      return output;
    }
    reduceBatches(reducer, initial, batchSize = 256) {
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
      if (this.#handle !== 0) {
        ffi.vt_handle_close(this.#handle);
        this.#handle = 0;
        this.#pending = [];
        this.#offset = 0;
      }
    }
    [Symbol.dispose]() { this.close(); }
  }

  class QueryCursor {
    #handle;
    #pending = [];
    #offset = 0;
    constructor(handle) { this.#handle = failure(handle); }
    [Symbol.iterator]() { return this; }
    next() {
      if (this.#offset >= this.#pending.length) {
        this.#pending = this.#fetch(256);
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
        if (result.length === maximum || this.#handle === 0) break;
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
      if (this.#handle === 0) return [];
      const count = failure(ffi.vt_cursor_next_batch(this.#handle, maximum));
      if (count === 0) return [];
      const pointer = failure(ffi.vt_cursor_batch_pointer(this.#handle));
      const memory = view();
      const output = [];
      for (let index = 0; index < count; index += 1) {
        const record = pointer + index * RECORD_SIZE;
        const termPointer = memory.getUint32(record, true);
        const termLength = memory.getUint32(record + 4, true);
        const domain = memory.getUint32(record + 8, true);
        let term;
        if (domain === 0) {
          term = { domain: "byte", value: bytes().slice(termPointer, termPointer + termLength) };
        } else if (domain === 1) {
          term = { domain: "unicode", value: decoder.decode(bytes().slice(termPointer, termPointer + termLength)) };
        } else {
          const values = new BigUint64Array(termLength);
          for (let offset = 0; offset < termLength; offset += 1) {
            values[offset] = memory.getBigUint64(termPointer + offset * 8, true);
          }
          term = { domain: "u64", value: values };
        }
        output.push({
          term,
          distance: memory.getUint32(record + 12, true),
          id: memory.getUint32(record + 16, true) === 0
            ? null
            : memory.getBigUint64(record + 24, true),
        });
      }
      return output;
    }
    reduceBatches(reducer, initial, batchSize = 256) {
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
    close() {
      if (this.#handle !== 0) {
        ffi.vt_handle_close(this.#handle);
        this.#handle = 0;
        this.#pending = [];
        this.#offset = 0;
      }
    }
    return() { this.close(); return { done: true, value: undefined }; }
    [Symbol.dispose]() { this.close(); }
  }

  class Transducer {
    #handle;
    constructor(dictionary, algorithm = "standard") {
      if (dictionary?.runtimeIdentity !== runtimeIdentity || dictionary.interfaceId !== "vt.dictionary.v1") {
        throw new TypeError("dictionary belongs to a different Vinary Tree runtime");
      }
      this.#handle = failure(ffi.vt_transducer_new(
        dictionary._handle,
        select(ALGORITHMS, algorithm, "algorithm"),
      ));
    }
    query(query, maximumDistance, order = "traversal") {
      if (typeof query !== "string") throw new TypeError("this WASI query entry point requires text");
      return withBytes(query, (pointer, length) => new QueryCursor(ffi.vt_query_text(
        this.#handle,
        pointer,
        length,
        maximumDistance,
        select(ORDERS, order, "query order"),
      )));
    }
    close() {
      if (this.#handle !== 0) {
        ffi.vt_handle_close(this.#handle);
        this.#handle = 0;
      }
    }
  }

  class Wfst {
    #handle;
    constructor(handle) {
      this.#handle = failure(handle);
      Object.defineProperties(this, {
        interfaceId: { value: "vt.scalar-wfst.1", enumerable: true },
        runtimeIdentity: { value: runtimeIdentity, enumerable: true },
      });
    }
    get _handle() {
      if (this.#handle === 0) throw new Error("WFST is closed");
      return this.#handle;
    }
    get weightDomain() {
      const value = failure(ffi.vt_wfst_weight_domain(this._handle));
      const name = WEIGHT_DOMAINS.get(value);
      if (name === undefined) throw new Error(`unknown WFST weight domain ${value}`);
      return name;
    }
    start() {
      const output = ffi.vt_alloc(8);
      try {
        failure(ffi.vt_wfst_start(this._handle, output));
        return view().getBigUint64(output, true);
      } finally {
        ffi.vt_dealloc(output, 8);
      }
    }
    state(state) {
      if (typeof state !== "bigint" || state < 0n || state > 0xffff_ffff_ffff_ffffn) {
        throw new RangeError("WFST state must be an unsigned 64-bit bigint");
      }
      const count = failure(ffi.vt_wfst_state(this._handle, state));
      const pointer = failure(ffi.vt_wfst_state_pointer(this._handle));
      const memory = view();
      const arcs = [];
      for (let index = 0; index < count; index += 1) {
        const record = pointer + 16 + index * 40;
        const hasInput = memory.getUint32(record + 32, true) !== 0;
        const hasOutput = memory.getUint32(record + 36, true) !== 0;
        arcs.push({
          input: hasInput ? String.fromCodePoint(Number(memory.getBigUint64(record, true))) : null,
          output: hasOutput ? String.fromCodePoint(Number(memory.getBigUint64(record + 8, true))) : null,
          target: memory.getBigUint64(record + 16, true),
          weight: memory.getFloat64(record + 24, true),
        });
      }
      return {
        valid: memory.getUint32(pointer, true) !== 0,
        final: memory.getUint32(pointer + 4, true) !== 0,
        finalWeight: memory.getFloat64(pointer + 8, true),
        arcs,
      };
    }
    close() {
      if (this.#handle !== 0) {
        ffi.vt_handle_close(this.#handle);
        this.#handle = 0;
      }
    }
  }

  class WfstBuilder {
    #handle = failure(ffi.vt_wfst_builder_new());
    get _handle() {
      if (this.#handle === 0) throw new Error("WFST builder is closed");
      return this.#handle;
    }
    addState() { return failure(ffi.vt_wfst_builder_add_state(this._handle)); }
    setStart(state) { failure(ffi.vt_wfst_builder_set_start(this._handle, state)); }
    setFinal(state, weight = 0) { failure(ffi.vt_wfst_builder_set_final(this._handle, state, weight)); }
    addArc(from, input, output, to, weight = 0) {
      const label = (value) => {
        if (value === null || value === undefined) return [0, 0];
        if (typeof value !== "string" || [...value].length !== 1) {
          throw new TypeError("arc label must contain one Unicode scalar or be null");
        }
        return [value.codePointAt(0), 1];
      };
      const [inputLabel, hasInput] = label(input);
      const [outputLabel, hasOutput] = label(output);
      failure(ffi.vt_wfst_builder_add_arc(
        this._handle, from, inputLabel, hasInput, outputLabel, hasOutput, to, weight,
      ));
    }
    build() {
      const handle = this._handle;
      const result = new Wfst(ffi.vt_wfst_builder_build(handle));
      this.#handle = 0;
      return result;
    }
    close() {
      if (this.#handle !== 0) {
        ffi.vt_handle_close(this.#handle);
        this.#handle = 0;
      }
    }
  }

  const libdictenstein = Object.freeze({
    runtimeIdentity,
    dynamicDawg(unitDomain = "unicode") {
      return new Dictionary(
        ffi.vt_dynamic_dawg_new(select(DOMAINS, unitDomain, "unit domain")),
        unitDomain,
        false,
      );
    },
    createPersistentARTrie(path, unitDomain = "unicode") {
      return withBytes(path, (pointer, length) => new Dictionary(
        ffi.vt_persistent_artrie_create(pointer, length, select(DOMAINS, unitDomain, "unit domain")),
        unitDomain,
        true,
      ));
    },
    openPersistentARTrie(path, unitDomain = "unicode") {
      return withBytes(path, (pointer, length) => new Dictionary(
        ffi.vt_persistent_artrie_open(pointer, length, select(DOMAINS, unitDomain, "unit domain")),
        unitDomain,
        true,
      ));
    },
  });
  const liblevenshtein = Object.freeze({
    runtimeIdentity,
    transducer(dictionary, algorithm = "standard") { return new Transducer(dictionary, algorithm); },
  });

  const llingLlang = Object.freeze({
    runtimeIdentity,
    vectorWfst() { return new WfstBuilder(); },
    compose(first, second) {
      if (first?.runtimeIdentity !== runtimeIdentity || second?.runtimeIdentity !== runtimeIdentity) {
        throw new TypeError("WFST belongs to a different Vinary Tree runtime");
      }
      return new Wfst(ffi.vt_wfst_compose(first._handle, second._handle));
    },
  });

  const duallity = Object.freeze({
    runtimeIdentity,
    wfst(dictionary, query, maximumDistance, algorithm = "standard", kind = "levenshtein") {
      if (dictionary?.runtimeIdentity !== runtimeIdentity || dictionary.interfaceId !== "vt.dictionary.v1") {
        throw new TypeError("dictionary belongs to a different Vinary Tree runtime");
      }
      return withBytes(query, (pointer, length) => new Wfst(ffi.vt_duallity_wfst_new(
        dictionary._handle,
        pointer,
        length,
        maximumDistance,
        select(ALGORITHMS, algorithm, "algorithm"),
        select(DUALLITY_KINDS, kind, "duallity WFST kind"),
      )));
    },
  });

  return Object.freeze({ runtimeIdentity, libdictenstein, liblevenshtein, llingLlang, duallity });
}
