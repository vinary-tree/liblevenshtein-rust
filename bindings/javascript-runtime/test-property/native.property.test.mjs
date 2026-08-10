// C8 native property-based tests for the javascript-runtime facade (fast-check).
//
// This suite lives in an isolated package whose only dependency is fast-check,
// so `npm install` never has to resolve the unpublished @vinary-tree/interop
// sibling that the umbrella package declares. It imports the native runtime by
// relative path (`../native.mjs`), which loads the prebuilt N-API addon.
//
// Properties, each checked against an in-language brute-force oracle:
//   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, and threshold
//       consistency dThreshold(a,b,k) == (d<=k ? d : undefined) -- exercised
//       directly against the exposed Levenshtein/Damerau/true-Damerau entry
//       points, plus standard-distance agreement with the oracle;
//   (b) a query's result set at distance k equals {t in dict : lev(query,t)<=k}
//       over a small random dictionary, with exact distances and value
//       round-trips;
//   (c) u64 value-channel round-trips, with 0 and 2**64-1 pinned as examples.
//
// Determinism: every fc.assert runs with a fixed seed, and the `examples`
// arrays are the committed regression corpus (fast-check keeps no on-disk DB).

import assert from "node:assert/strict";
import test from "node:test";
import fc from "fast-check";
import { libdictenstein, liblevenshtein, llingLlang, duallity } from "../native.mjs";

const U64_MAX = 2n ** 64n - 1n;
// "é" (U+00E9) is one Unicode scalar but two UTF-8 bytes: including it proves
// the facade counts scalars, not bytes, end to end.
const CHARS = ["a", "b", "c", "é"];

const term = fc.array(fc.constantFrom(...CHARS), { maxLength: 6 }).map((chars) => chars.join(""));
const value = fc.option(fc.bigInt({ min: 0n, max: U64_MAX }), { nil: null });
const dictionary = fc.uniqueArray(fc.tuple(term, value), {
  selector: (pair) => pair[0],
  maxLength: 8,
});
const distance = fc.integer({ min: 0, max: 3 });
const SEED = 20260809;

/** Reference Levenshtein distance over Unicode scalars (the standard oracle). */
function levenshtein(a, b) {
  if (a === b) return 0;
  const left = [...a];
  const right = [...b];
  if (left.length === 0) return right.length;
  if (right.length === 0) return left.length;
  let previous = Array.from({ length: right.length + 1 }, (_unused, index) => index);
  for (let i = 1; i <= left.length; i += 1) {
    const current = new Array(right.length + 1);
    current[0] = i;
    for (let j = 1; j <= right.length; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost);
    }
    previous = current;
  }
  return previous[right.length];
}

function runQuery(entries, query, k, order = "traversal") {
  const dict = libdictenstein.dynamicDawg("unicode");
  try {
    for (const [key, storedValue] of entries) dict.put(key, storedValue);
    const transducer = liblevenshtein.transducer(dict);
    try {
      const result = [];
      for (const match of transducer.query(query, k, order)) {
        result.push({ term: match.term.value, distance: match.distance, id: match.id });
      }
      return result;
    } finally {
      transducer.close();
    }
  } finally {
    dict.close();
  }
}

// --- (a) distance self-consistency across all exposed variants --------------

const DISTANCE_VARIANTS = [
  ["levenshtein", liblevenshtein.levenshteinDistance, liblevenshtein.levenshteinDistanceThreshold],
  ["damerau", liblevenshtein.damerauDistance, liblevenshtein.damerauDistanceThreshold],
  ["trueDamerau", liblevenshtein.trueDamerauDistance, liblevenshtein.trueDamerauDistanceThreshold],
];

for (const [name, dist, distThreshold] of DISTANCE_VARIANTS) {
  test(`${name} distance is symmetric with zero identity`, () => {
    fc.assert(
      fc.property(term, term, (a, b) => {
        const forward = dist(a, b);
        const backward = dist(b, a);
        return forward === backward && dist(a, a) === 0;
      }),
      { seed: SEED, numRuns: 300, examples: [["", "abc"], ["kitten", "sitting"], ["ca", "abc"]] },
    );
  });

  test(`${name} threshold matches the unthresholded distance`, () => {
    fc.assert(
      fc.property(term, term, distance, (a, b, k) => {
        const full = dist(a, b);
        const bounded = distThreshold(a, b, k);
        return full <= k ? bounded === full : bounded === undefined;
      }),
      { seed: SEED, numRuns: 300, examples: [["cat", "cot", 0], ["cat", "cot", 1]] },
    );
  });
}

test("standard distance agrees with the brute-force oracle", () => {
  fc.assert(
    fc.property(term, term, (a, b) => liblevenshtein.levenshteinDistance(a, b) === levenshtein(a, b)),
    { seed: SEED, numRuns: 300, examples: [["", ""], ["café", "cafe"], ["abcabc", "aaa"]] },
  );
});

// --- (b) query result set equals the oracle ---------------------------------

test("query result set equals the brute-force oracle", () => {
  fc.assert(
    fc.property(dictionary, term, distance, (entries, query, k) => {
      const expected = new Map();
      for (const [key, storedValue] of entries) {
        if (levenshtein(query, key) <= k) expected.set(key, storedValue);
      }
      const got = new Map();
      for (const match of runQuery(entries, query, k)) got.set(match.term, match);
      if (got.size !== expected.size) return false;
      for (const [key, match] of got) {
        if (!expected.has(key)) return false;
        if (match.distance !== levenshtein(query, key)) return false; // (a) exact distance
        if (match.id !== expected.get(key)) return false; // (c) value round-trip incl. null
      }
      return true;
    }),
    {
      seed: SEED,
      numRuns: 150,
      examples: [
        [[["cat", 1n], ["cot", 2n], ["cut", 3n], ["scat", null]], "cat", 2],
        [[["", 0n], ["cat", U64_MAX], ["café", 5n]], "", 2],
      ],
    },
  );
});

// --- (b) ordered queries are monotone in (distance, term) -------------------

test("distance-then-term order is monotone", () => {
  fc.assert(
    fc.property(dictionary, term, distance, (entries, query, k) => {
      const observed = runQuery(entries, query, k, "distance-then-term");
      for (let i = 1; i < observed.length; i += 1) {
        const previous = observed[i - 1];
        const current = observed[i];
        if (
          previous.distance > current.distance ||
          (previous.distance === current.distance && previous.term > current.term)
        ) {
          return false;
        }
      }
      return true;
    }),
    { seed: SEED, numRuns: 150 },
  );
});

// --- (c) u64 value channel round-trips, boundaries pinned -------------------

test("u64 values round-trip, including 0 and 2**64-1", () => {
  fc.assert(
    fc.property(fc.bigInt({ min: 0n, max: U64_MAX }), (storedValue) => {
      const matches = runQuery([["term", storedValue]], "term", 0);
      return matches.length === 1 && matches[0].id === storedValue;
    }),
    { seed: SEED, numRuns: 200, examples: [[0n], [U64_MAX], [2n ** 63n]] },
  );
});

// --- lling-llang WFST facade: structural round-trip through the N-API boundary

// A chain of arcs described via the builder survives build() with its start,
// arc labels/targets/weights, and final state/weight intact -- the JS <-> native
// marshalling of the vt.scalar-wfst.1 surface preserves the WFST structurally.
const label = fc.constantFrom("a", "b", "c", null);
const chainSpec = fc.record({
  labels: fc.array(label, { minLength: 1, maxLength: 5 }),
  finalWeight: fc.integer({ min: 0, max: 5 }),
});

test("lling-llang vector WFST round-trips its structure through the native boundary", () => {
  fc.assert(
    fc.property(chainSpec, ({ labels, finalWeight }) => {
      const builder = llingLlang.vectorWfst();
      const states = [builder.addState()];
      for (let i = 0; i < labels.length; i += 1) states.push(builder.addState());
      builder.setStart(states[0]);
      builder.setFinal(states[labels.length], finalWeight);
      for (let i = 0; i < labels.length; i += 1) {
        builder.addArc(states[i], labels[i], labels[i], states[i + 1], i);
      }
      const wfst = builder.build();
      try {
        if (wfst.start() !== BigInt(states[0])) return false;
        for (let i = 0; i < labels.length; i += 1) {
          const node = wfst.state(BigInt(states[i]));
          if (!node.valid || node.arcs.length !== 1) return false;
          const arc = node.arcs[0];
          if (arc.input !== labels[i] || arc.output !== labels[i]) return false;
          if (arc.target !== BigInt(states[i + 1]) || arc.weight !== i) return false;
        }
        const finalNode = wfst.state(BigInt(states[labels.length]));
        return finalNode.valid && finalNode.final && finalNode.finalWeight === finalWeight;
      } finally {
        wfst.close();
        builder.close();
      }
    }),
    { seed: SEED, numRuns: 200, examples: [[{ labels: ["a"], finalWeight: 0 }], [{ labels: ["a", null, "c"], finalWeight: 3 }]] },
  );
});

// --- duallity levenshtein WFST facade: deterministic and valid for its query --

// Building the same (dictionary, query, distance) levenshtein WFST twice yields
// the same start state, and that start state is valid -- the duallity boundary is
// deterministic and always produces a well-formed automaton for valid input.
test("duallity levenshtein WFST is deterministic and its start state is valid", () => {
  fc.assert(
    fc.property(dictionary, term, distance, (entries, query, k) => {
      const dict = libdictenstein.dynamicDawg("unicode");
      try {
        for (const [key, storedValue] of entries) dict.put(key, storedValue);
        const first = duallity.wfst(dict, query, k);
        const second = duallity.wfst(dict, query, k);
        try {
          return first.start() === second.start() && first.state(first.start()).valid;
        } finally {
          first.close();
          second.close();
        }
      } finally {
        dict.close();
      }
    }),
    { seed: SEED, numRuns: 100, examples: [[[["cat", 1n], ["cot", 2n]], "cat", 1], [[["a", 0n]], "", 2]] },
  );
});

// A trivial node:test assertion keeps the runner honest if fc.assert is ever
// disabled: the runtime must at least load and answer one exact query.
test("native runtime loads and answers a known query", () => {
  assert.deepEqual(runQuery([["cat", 1n]], "bat", 1), [{ term: "cat", distance: 1, id: 1n }]);
});
