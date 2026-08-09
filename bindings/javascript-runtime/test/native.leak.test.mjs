// C9 leak-discipline suite for the javascript-runtime facade.
//
// Run with `node --expose-gc --test` (see the `test:leak` npm script). A
// >=10,000-cycle create/use/free loop must reach a memory steady state: the V8
// heap (process.memoryUsage().heapUsed) and the N-API external memory that
// backs native handles (process.memoryUsage().external) must not grow across
// the cycles. RSS is deliberately not asserted on -- it reflects allocator
// retention and JIT code, not handle leaks, and does not shrink on gc().
//
// Measured baseline on this native build (10,000 cycles): heapUsed ~0.6 B/cycle
// and external ~40 bytes total -- i.e. flat. The ceilings below leave a wide
// margin over that noise while still catching a real per-cycle leak, which
// would accrue megabytes.

import assert from "node:assert/strict";
import test from "node:test";
import { libdictenstein, liblevenshtein } from "../native.mjs";

const CYCLES = 10_000;
const WARMUP = 2_000;
const MAX_HEAP_GROWTH = 4 * 1024 * 1024;
const MAX_EXTERNAL_GROWTH = 1 * 1024 * 1024;

function requireGc() {
  if (typeof global.gc !== "function") {
    throw new Error("run the leak suite with `node --expose-gc` (npm run test:leak)");
  }
}

function settled() {
  for (let i = 0; i < 4; i += 1) global.gc();
  return process.memoryUsage();
}

function measure(cycle) {
  requireGc();
  for (let i = 0; i < WARMUP; i += 1) cycle();
  const base = settled();
  for (let i = 0; i < CYCLES; i += 1) cycle();
  const end = settled();
  return { heap: end.heapUsed - base.heapUsed, external: end.external - base.external };
}

function assertSteady(label, cycle) {
  const growth = measure(cycle);
  assert.ok(
    growth.heap < MAX_HEAP_GROWTH,
    `${label}: heap grew ${growth.heap} bytes over ${CYCLES} cycles`,
  );
  assert.ok(
    growth.external < MAX_EXTERNAL_GROWTH,
    `${label}: external grew ${growth.external} bytes over ${CYCLES} cycles`,
  );
}

const ENTRIES = [
  ["cat", 1n],
  ["cot", 2n],
  ["cut", 3n],
  ["scat", null],
];

function buildDictionary() {
  const dictionary = libdictenstein.dynamicDawg("unicode");
  for (const [term, value] of ENTRIES) dictionary.put(term, value);
  return dictionary;
}

test("transducer iterator cycle reaches memory steady state", () => {
  assertSteady("iterator", () => {
    const dictionary = buildDictionary();
    const transducer = liblevenshtein.transducer(dictionary);
    const cursor = transducer.query("cat", 2);
    for (const _match of cursor) {
      // drain
    }
    cursor.close();
    transducer.close();
    dictionary.close();
  });
});

test("transducer reduceBatches cycle reaches memory steady state", () => {
  assertSteady("reduceBatches", () => {
    const dictionary = buildDictionary();
    const transducer = liblevenshtein.transducer(dictionary);
    const cursor = transducer.query("cat", 2);
    cursor.reduceBatches((count, batch) => count + batch.length, 0);
    cursor.close();
    transducer.close();
    dictionary.close();
  });
});

test("phonetic pattern cycle reaches memory steady state", () => {
  assertSteady("phoneticPattern", () => {
    const pattern = liblevenshtein.phoneticPattern("c[ao]t");
    pattern.matches("cat");
    pattern.close();
  });
});

test("phonetic rules cycle reaches memory steady state", () => {
  assertSteady("phoneticRules", () => {
    const rules = liblevenshtein.phoneticRules("english-orthography");
    rules.apply("phone");
    rules.close();
  });
});
