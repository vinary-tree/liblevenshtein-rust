#!/usr/bin/env node
// JavaScript harness for the cross-language benchmark program.
// Implements harnesses/common/PROTOCOL.md for FOUR targets, selected by the
// XL_JS_BACKEND environment variable (the runner's per-target wrappers set
// it; the CLI surface stays exactly the PROTOCOL contract):
//   native  @vinary-tree/liblevenshtein            (N-API addon)
//   wasm    @vinary-tree/liblevenshtein/wasm       (wasm-bindgen)
//   wasi    @vinary-tree/liblevenshtein/wasi       (wasm32-wasip1)
//   legacy  vendored 2.0.4 compiled-CoffeeScript build, loaded same-realm
//           through a Function("window", src) shim so it runs the exact
//           browser code path the old interactive demo used (ESM has no
//           global `require`, so the bundle's CommonJS branch never fires).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, basename, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HARNESS_DIR = dirname(fileURLToPath(import.meta.url));
const XL = resolve(HARNESS_DIR, "../..");
const WALL_CAP_SECONDS = 300;
const SAMPLE_DEFINITION =
  "one full pass over the query set; every cursor fully drained and " +
  "(term, distance) materialized";

// ---------------------------------------------------------------------------
// Checksum primitives (PROTOCOL.md §8) — BigInt masked to 64 bits
// ---------------------------------------------------------------------------

const MASK64 = (1n << 64n) - 1n;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;

function fnvUpdate(hash, byte) {
  return ((hash ^ BigInt(byte)) * FNV_PRIME) & MASK64;
}

function fnv1a64Bytes(bytes) {
  let hash = FNV_OFFSET;
  for (const b of bytes) hash = fnvUpdate(hash, b);
  return hash;
}

function entryAscii(term, distance) {
  let hash = FNV_OFFSET;
  for (let i = 0; i < term.length; i++) {
    const code = term.charCodeAt(i);
    if (code > 0x7f) {
      return entryBytes(Buffer.from(term, "utf8"), distance);
    }
    hash = fnvUpdate(hash, code);
  }
  return finishEntry(hash, distance);
}

function entryBytes(bytes, distance) {
  let hash = FNV_OFFSET;
  for (const b of bytes) hash = fnvUpdate(hash, b);
  return finishEntry(hash, distance);
}

function finishEntry(hash, distance) {
  hash = fnvUpdate(hash, 0);
  let d = BigInt(distance);
  for (let i = 0; i < 8; i++) {
    hash = fnvUpdate(hash, Number(d & 0xffn));
    d >>= 8n;
  }
  return hash;
}

function selfTest() {
  const expect = (actual, wanted, label) => {
    if (actual !== wanted) {
      throw new Error(
        `checksum self-test failed for ${label}: got ${actual.toString(16)}, want ${wanted.toString(16)}`,
      );
    }
  };
  expect(fnv1a64Bytes([]), 0xcbf29ce484222325n, 'fnv1a64("")');
  expect(fnv1a64Bytes([0x61]), 0xaf63dc4c8601ec8cn, 'fnv1a64("a")');
  expect(entryAscii("cat", 1), 0x9697fa3e50464bc4n, "entry(cat,1)");
  expect(entryAscii("cat", 0), 0xb592c1475b3595e5n, "entry(cat,0)");
  expect(entryAscii("cot", 1), 0xb8acc5d3816bcdean, "entry(cot,1)");
  expect(
    (entryAscii("cat", 0) + entryAscii("cot", 1)) & MASK64,
    0x6e3f871adca163cfn,
    "checksum{2}",
  );
}

// ---------------------------------------------------------------------------
// CLI (PROTOCOL.md §1)
// ---------------------------------------------------------------------------

function die(message) {
  process.stderr.write(`bench-cross-js: ${message}\n`);
  process.exit(2);
}

function parseArgs(argv) {
  const args = {
    mode: "",
    algorithm: null,
    maxDistance: -1,
    dictionary: null,
    queries: null,
    backend: "",
    out: null,
    samples: 30,
    warmupSeconds: 3,
    gateLimit: 200,
    reps: 10,
    cells: null,
  };
  for (let i = 0; i + 1 < argv.length; i += 2) {
    const flag = argv[i];
    const value = argv[i + 1];
    switch (flag) {
      case "--mode": args.mode = value; break;
      case "--algorithm": args.algorithm = value; break;
      case "--max-distance": args.maxDistance = Number(value); break;
      case "--dictionary": args.dictionary = value; break;
      case "--queries": args.queries = value; break;
      case "--backend": args.backend = value; break;
      case "--out": args.out = value; break;
      case "--samples": args.samples = Number(value); break;
      case "--warmup-seconds": args.warmupSeconds = Number(value); break;
      case "--gate-limit": args.gateLimit = Number(value); break;
      case "--reps": args.reps = Number(value); break;
      case "--cells": args.cells = value; break;
      default: die(`unknown flag: ${flag}`);
    }
  }
  if (!args.mode || !args.dictionary || !args.backend) {
    die("--mode, --dictionary, --backend are required");
  }
  return args;
}

// ---------------------------------------------------------------------------
// Input loading (PROTOCOL.md §3)
// ---------------------------------------------------------------------------

function readLines(path) {
  const raw = readFileSync(path, "utf8");
  const lines = [];
  let start = 0;
  for (let i = 0; i <= raw.length; i++) {
    if (i === raw.length || raw[i] === "\n") {
      if (i > start) lines.push(raw.slice(start, i));
      start = i + 1;
    }
  }
  if (lines.length === 0) die(`${path} contains no lines`);
  return lines;
}

function assertStrictlySorted(lines, path) {
  for (let i = 0; i + 1 < lines.length; i++) {
    // ASCII workload: JS string comparison over ASCII equals byte order.
    if (!(lines[i] < lines[i + 1])) {
      die(`${path} is not strictly byte-sorted at line ${i + 1}: "${lines[i]}" >= "${lines[i + 1]}"`);
    }
  }
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

const JS_BACKEND = process.env.XL_JS_BACKEND ?? "native";

const NEW_STACK_ALGORITHMS = {
  standard: "standard",
  transposition: "transposition",
  merge_and_split: "merge-and-split",
  damerau_levenshtein: "damerau-levenshtein",
};

async function loadNamespaces() {
  // Dictionaries come from the UMBRELLA's libdictenstein namespace rather
  // than the sibling repo's facade package: that facade is a one-line
  // re-export of exactly this namespace, and importing the umbrella directly
  // (a) needs no node_modules provisioning inside the libdictenstein repo
  // (which must stay untouched) and (b) realpath-unifies with the umbrella
  // instance the liblevenshtein facade holds, so assertSameRuntime passes.
  switch (JS_BACKEND) {
    case "native":
      return {
        levenshtein: await import("@vinary-tree/liblevenshtein"),
        dictionaries: (await import("@vinary-tree/javascript-runtime")).libdictenstein,
      };
    case "wasm": {
      // The umbrella's ./wasm entry initializes via browser fetch(); under
      // Node the repo's own conformance test (test/browser-wasm.test.mjs)
      // loads the module bytes explicitly and builds the runtime through
      // runtime-factory. Replicate exactly that sanctioned pattern; the
      // generated/ and runtime-factory paths are not in the export map, so
      // they are imported by file URL through the farm symlink.
      const runtimeDir = pathToFileURL(
        resolve(HARNESS_DIR, "node_modules/@vinary-tree/javascript-runtime") + "/",
      );
      const raw = await import(new URL("generated/wasm/vinary_tree.js", runtimeDir));
      raw.initSync({
        module: await readFile(new URL("generated/wasm/vinary_tree_bg.wasm", runtimeDir)),
      });
      const { createRuntime } = await import(new URL("runtime-factory.mjs", runtimeDir));
      const runtime = createRuntime(raw);
      return { levenshtein: runtime.liblevenshtein, dictionaries: runtime.libdictenstein };
    }
    case "wasi": {
      const runtime = await import("@vinary-tree/javascript-runtime/wasi");
      return { levenshtein: runtime.liblevenshtein, dictionaries: runtime.libdictenstein };
    }
    case "legacy": {
      const bundlePath = resolve(XL, "legacy/javascript/vendor/levenshtein-transducer.js");
      const source = readFileSync(bundlePath, "utf8");
      const windowShim = {};
      // Same-realm browser-path load: no `exports`/`require` in scope.
      // Security note: the Function body is the committed, SHA-256-pinned
      // vendored artifact (legacy/javascript/vendor/provenance.json) — fixed
      // bytes with zero interpolation, executed deliberately as the
      // benchmark's legacy baseline. Nothing user-controlled reaches this.
      new Function("window", source)(windowShim);
      return { legacy: windowShim.levenshtein };
    }
    default:
      die(`unknown XL_JS_BACKEND: ${JS_BACKEND}`);
      return null;
  }
}

class NewStackSide {
  constructor(namespaces) {
    this.namespaces = namespaces;
    this.dictionary = null;
    this.transducer = null;
  }

  buildDictionary(terms, backend) {
    if (backend === "dynamic_dawg") {
      // The JS facade exposes no batch insert (unlike putAllBytes/
      // update_many elsewhere); the put loop below IS the facade's bulk
      // path and is recorded in the cell notes.
      const dawg = this.namespaces.dictionaries.dynamicDawg();
      for (const term of terms) dawg.put(term);
      this.dictionary = dawg;
    } else if (backend === "double_array_trie") {
      this.dictionary = this.namespaces.dictionaries.doubleArrayTrie(
        this.preparedEntries ?? (this.preparedEntries = terms.map((term) => ({ term }))),
      );
    } else {
      die(`unknown backend for new stack: ${backend}`);
    }
  }

  freeDictionary() {
    if (this.transducer) { this.transducer.close(); this.transducer = null; }
    if (this.dictionary) { this.dictionary.close(); this.dictionary = null; }
  }

  createTransducer(algorithm) {
    if (this.transducer) this.transducer.close();
    const mapped = NEW_STACK_ALGORITHMS[algorithm];
    if (!mapped) die(`unknown algorithm: ${algorithm}`);
    this.transducer = this.namespaces.levenshtein.transducer(this.dictionary, mapped);
  }

  pass(queries, maxDistance, withChecksum) {
    let matches = 0n;
    let bytes = 0n;
    let distanceSum = 0n;
    let checksum = 0n;
    for (const query of queries) {
      const cursor = this.transducer.query(query, maxDistance);
      for (const match of cursor) {
        // Term is a tagged union: { domain: "unicode", value: string } here.
        const term = match.term.value;
        matches += 1n;
        bytes += BigInt(term.length); // ASCII workload: length == UTF-8 bytes
        distanceSum += BigInt(match.distance);
        if (withChecksum) {
          checksum = (checksum + entryAscii(term, match.distance)) & MASK64;
        }
      }
      cursor.close();
    }
    return { matches, bytes, distanceSum, checksum };
  }

  targetInfo(backend) {
    return {
      implementation: "vinary-tree",
      backendLabel: JS_BACKEND === "native" ? "napi" : JS_BACKEND,
      libraryVersion: "0.10.0",
      artifactKind: "local-build",
      artifactId: `@vinary-tree/liblevenshtein@0.10.0 (${JS_BACKEND})`,
      structure: backend,
      unitDomain: "unicode_scalar",
      batchSize: 256,
      notes: [
        "dynamic_dawg construction uses the facade's put() loop (no batch API in the JS facade)",
      ],
    };
  }
}

class LegacySide {
  constructor(namespaces) {
    this.levenshtein = namespaces.legacy;
    this.builder = null;
    this.transducer = null;
  }

  buildDictionary(terms, backend) {
    if (backend !== "own") die(`legacy target only supports backend 'own', got ${backend}`);
    // Dawg construction happens inside the dictionary setter; sorted=true is
    // the legacy fast path (sortedness asserted at load).
    this.builder = new this.levenshtein.Builder()
      .dictionary(terms, true)
      .include_distance(true)
      .sort_candidates(false);
  }

  freeDictionary() {
    this.builder = null;      // GC-managed; the legacy API has no close()
    this.transducer = null;
  }

  createTransducer(algorithm) {
    if (algorithm === "damerau_levenshtein") {
      die("legacy 2.0.4 does not support damerau_levenshtein");
    }
    this.transducer = this.builder.algorithm(algorithm).transducer();
  }

  pass(queries, maxDistance, withChecksum) {
    let matches = 0n;
    let bytes = 0n;
    let distanceSum = 0n;
    let checksum = 0n;
    for (const query of queries) {
      const results = this.transducer.transduce(query, maxDistance); // eager array
      for (const [term, distance] of results) {
        matches += 1n;
        bytes += BigInt(term.length);
        distanceSum += BigInt(distance);
        if (withChecksum) {
          checksum = (checksum + entryAscii(term, distance)) & MASK64;
        }
      }
    }
    return { matches, bytes, distanceSum, checksum };
  }

  targetInfo(backend) {
    return {
      implementation: "legacy",
      backendLabel: "node-legacy-js",
      libraryVersion: "2.0.4",
      artifactKind: "vendored",
      artifactId: "gh-pages 59934da8 javascripts/2.0.4/levenshtein-transducer.js",
      structure: "legacy_js_dawg",
      unitDomain: "js_string",
      batchSize: null,
      notes: [
        "sort_candidates(false) pinned: the new stack returns traversal order, so result sorting would be uncompensated extra work",
        "loaded same-realm via Function('window', source) — the browser code path the demo used",
      ],
    };
  }
}

// ---------------------------------------------------------------------------
// Protocol driver (PROTOCOL.md §5–7)
// ---------------------------------------------------------------------------

const nowNs = () => process.hrtime.bigint();

function tripleEquals(a, b) {
  return a.matches === b.matches && a.bytes === b.bytes && a.distanceSum === b.distanceSum;
}

function runQueryCell(side, args, queries, algorithm, maxDistance, queriesPath, outPath, meta) {
  const gate = side.pass(queries, maxDistance, true);

  const warmStart = nowNs();
  const warmupNs = BigInt(Math.round(args.warmupSeconds * 1e9));
  let warmupPasses = 0;
  let lastPassNs = 0n;
  while (nowNs() - warmStart < warmupNs || warmupPasses < 2) {
    const t0 = nowNs();
    const triple = side.pass(queries, maxDistance, false);
    lastPassNs = nowNs() - t0;
    if (!tripleEquals(triple, gate)) die("nondeterministic result during warmup");
    warmupPasses++;
  }

  let sampleCount = args.samples;
  let status = "ok";
  const notes = [...meta.notes];
  const lastPassSeconds = Number(lastPassNs) / 1e9;
  if (sampleCount * lastPassSeconds > WALL_CAP_SECONDS) {
    const reduced = Math.max(10, Math.floor(WALL_CAP_SECONDS / lastPassSeconds));
    notes.push(
      `samples reduced from ${sampleCount} to ${reduced} by the ${WALL_CAP_SECONDS}s wall cap (estimated pass ${lastPassSeconds.toFixed(3)}s)`,
    );
    sampleCount = reduced;
    status = "degraded";
  }

  const samplesNs = new Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    const t0 = nowNs();
    const triple = side.pass(queries, maxDistance, false);
    samplesNs[i] = Number(nowNs() - t0);
    if (!tripleEquals(triple, gate)) die("nondeterministic result during measurement");
  }

  emitResult(outPath, args, "query", algorithm, maxDistance, queriesPath, queries.length, meta, {
    warmupPasses,
    samplesNs,
    gate,
    status,
    notes,
  });
}

function emitResult(outPath, args, mode, algorithm, maxDistance, queriesPath, queryCount, meta, extra) {
  const {
    warmupPasses = 0,
    samplesNs = [],
    gate = { matches: 0n, bytes: 0n, distanceSum: 0n, checksum: 0n },
    constructTimes = null,
    status = "ok",
    notes = meta.notes,
  } = extra;
  const queryset = basename(queriesPath).replace(/\.txt$/, "");
  const result = {
    schema_version: "1.0.0",
    suite: "cross-language-v1",
    timestamp_utc: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
    target: {
      language: "javascript",
      implementation: meta.implementation,
      backend: meta.backendLabel,
      runtime_version: `node ${process.version}`,
      library_version: meta.libraryVersion,
      artifact: { kind: meta.artifactKind, id: meta.artifactId },
    },
    dictionary: {
      file: args.dictionary,
      term_count: meta.termCount,
      structure: meta.structure,
      unit_domain: meta.unitDomain,
      ...(meta.constructNs != null ? { construct_ns: meta.constructNs } : {}),
    },
    workload: { queryset, file: queriesPath, query_count: queryCount },
    algorithm,
    max_distance: maxDistance,
    mode: mode === "memory-child" ? "memory" : mode,
    protocol: {
      timer: "monotonic",
      harness: "self-timed",
      warmup_seconds_min: args.warmupSeconds,
      warmup_passes: warmupPasses,
      samples_requested: mode === "construct" ? args.reps : mode === "query" ? args.samples : 0,
      sample_definition: SAMPLE_DEFINITION,
      batch_size: meta.batchSize,
      wall_cap_seconds: WALL_CAP_SECONDS,
    },
    ...(constructTimes
      ? {
          construct: {
            reps: constructTimes.length,
            times_ns: constructTimes,
            term_count: meta.termCount,
          },
        }
      : {
          measurements: {
            samples_ns: samplesNs,
            sample_count: samplesNs.length,
            matches_per_pass: Number(gate.matches),
            term_bytes_per_pass: Number(gate.bytes),
            distance_sum_per_pass: Number(gate.distanceSum),
            checksum_hex: gate.checksum.toString(16).padStart(16, "0"),
          },
        }),
    status,
    notes,
  };
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(result, null, 2) + "\n");
}

async function main() {
  selfTest();
  const args = parseArgs(process.argv.slice(2));
  const namespaces = await loadNamespaces();
  const side = JS_BACKEND === "legacy" ? new LegacySide(namespaces) : new NewStackSide(namespaces);

  const terms = readLines(args.dictionary);
  assertStrictlySorted(terms, args.dictionary);

  const info = side.targetInfo(args.backend);
  const meta = { ...info, notes: [...info.notes], termCount: terms.length, constructNs: null };

  if (args.mode === "construct") {
    if (!args.out) die("--out is required for construct mode");
    side.buildDictionary(terms, args.backend); // warmup build
    side.freeDictionary();
    const times = new Array(args.reps);
    for (let r = 0; r < args.reps; r++) {
      const t0 = nowNs();
      side.buildDictionary(terms, args.backend);
      times[r] = Number(nowNs() - t0);
      side.freeDictionary();
    }
    meta.notes.push(
      "construct mode: timed region is the build from the pre-sorted in-memory list only",
    );
    emitResult(args.out, args, "construct", "standard", 1,
      args.queries ?? `${XL}/workload/queries/hits.txt`, 1, meta, { constructTimes: times });
    return;
  }

  const buildStart = nowNs();
  side.buildDictionary(terms, args.backend);
  meta.constructNs = Number(nowNs() - buildStart);

  const runOne = (algorithm, maxDistance, queriesPath, outPath) => {
    side.createTransducer(algorithm);
    const queries = readLines(queriesPath);
    if (args.mode === "verify") {
      const limit = Math.min(args.gateLimit, queries.length);
      const subset = queries.slice(0, limit);
      const gate = side.pass(subset, maxDistance, true);
      emitResult(outPath, args, "verify", algorithm, maxDistance, queriesPath, limit, meta, { gate });
    } else if (args.mode === "memory-child") {
      const gate = side.pass(queries, maxDistance, true);
      emitResult(outPath, args, "memory-child", algorithm, maxDistance, queriesPath,
        queries.length, meta, { gate });
    } else if (args.mode === "query") {
      runQueryCell(side, args, queries, algorithm, maxDistance, queriesPath, outPath, meta);
    } else {
      die(`unknown mode: ${args.mode}`);
    }
  };

  if (args.cells) {
    const rows = readFileSync(args.cells, "utf8")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((line) => line.split("\t"));
    for (const [algorithm, distance, queriesPath, outPath] of rows) {
      runOne(algorithm, Number(distance), queriesPath, outPath);
    }
  } else {
    if (!args.algorithm || args.maxDistance < 0 || !args.queries || !args.out) {
      die("--algorithm, --max-distance, --queries, --out are required");
    }
    runOne(args.algorithm, args.maxDistance, args.queries, args.out);
  }
}

await main();
