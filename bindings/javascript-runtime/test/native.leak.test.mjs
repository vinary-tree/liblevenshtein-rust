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
import { libdictenstein, liblevenshtein, llingLlang, duallity } from "../native.mjs";

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

// Build a small two-state chain WFST (start -a:a/0-> final) via the lling-llang
// builder. Returns a built, live Wfst the caller must close.
function buildSmallWfst() {
  const builder = llingLlang.vectorWfst();
  const start = builder.addState();
  const final = builder.addState();
  builder.setStart(start);
  builder.setFinal(final, 0);
  builder.addArc(start, "a", "a", final, 0);
  const wfst = builder.build();
  builder.close();
  return wfst;
}

test("lling-llang vector WFST build cycle reaches memory steady state", () => {
  assertSteady("vectorWfst", () => {
    const builder = llingLlang.vectorWfst();
    const start = builder.addState();
    const final = builder.addState();
    builder.setStart(start);
    builder.setFinal(final, 0);
    builder.addArc(start, "a", "a", final, 0);
    const wfst = builder.build();
    wfst.state(wfst.start()); // touch the native handle
    wfst.close();
    builder.close();
  });
});

test("lling-llang WFST composition cycle reaches memory steady state", () => {
  assertSteady("composeWfst", () => {
    const first = buildSmallWfst();
    const second = buildSmallWfst();
    // compose captures snapshots of both inputs, which remain owned here.
    const composed = llingLlang.compose(first, second);
    composed.close();
    first.close();
    second.close();
  });
});

test("duallity levenshtein WFST cycle reaches memory steady state", () => {
  // Directly guards the handle class behind DUAL-B10 (a construction leak in the
  // duallity WFST boundary) at the JS runtime level.
  assertSteady("duallityWfst", () => {
    const dictionary = buildDictionary();
    const wfst = duallity.wfst(dictionary, "cat", 2);
    wfst.state(wfst.start()); // touch the native handle
    wfst.close();
    dictionary.close();
  });
});
