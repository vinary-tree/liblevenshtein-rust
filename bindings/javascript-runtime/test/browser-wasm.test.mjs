import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import * as raw from "../generated/wasm/vinary_tree.js";
import { createRuntime } from "../runtime-factory.mjs";

raw.initSync({
  module: await readFile(new URL("../generated/wasm/vinary_tree_bg.wasm", import.meta.url)),
});
const { libdictenstein, liblevenshtein, llingLlang, duallity, runtimeIdentity } = createRuntime(raw);

function collect(cursor) {
  try {
    return [...cursor].map(({ term, distance, id }) => ({
      term: term.value,
      distance,
      id,
    }));
  } finally {
    cursor.close();
  }
}

test("all namespaces share one identity and transfer dictionaries in O(1)", () => {
  assert.equal(libdictenstein.runtimeIdentity, runtimeIdentity);
  assert.equal(liblevenshtein.runtimeIdentity, runtimeIdentity);
  const dictionary = libdictenstein.dynamicDawg();
  dictionary.put("kitten", 7n);
  dictionary.put("sitting", null);
  const transducer = liblevenshtein.transducer(dictionary);
  assert.deepEqual(collect(transducer.query("kitten", 3, "distance-then-term")), [
    { term: "kitten", distance: 0, id: 7n },
    { term: "sitting", distance: 3, id: null },
  ]);
  assert.equal(liblevenshtein.levenshteinDistance("kitten", "sitting"), 3);
  transducer.close();
  dictionary.close();
});

test("one long-lived iterator has exact query-start snapshot semantics", () => {
  let state = 0x9e3779b9;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state >>> 0;
  };

  for (let trace = 0; trace < 64; trace += 1) {
    const dictionary = libdictenstein.dynamicDawg();
    const terms = new Set();
    while (terms.size < 16) terms.add(`t${trace}-${random().toString(36)}`);
    let id = 0n;
    for (const term of terms) dictionary.put(term, id++);
    const transducer = liblevenshtein.transducer(dictionary);
    const expected = collect(transducer.query("", 64, "distance-then-term"));
    const cursor = transducer.query("", 64, "distance-then-term");
    const actual = [];

    actual.push(cursor.next().value);
    const first = terms.values().next().value;
    dictionary.remove(first);
    dictionary.put(first, 999n);
    actual.push(cursor.next().value);
    dictionary.clear();
    dictionary.compact();
    actual.push(cursor.next().value);
    dictionary.put(`after-${trace}`, 1000n);
    for (const match of cursor) actual.push(match);

    assert.deepEqual(
      actual.map(({ term, distance, id: matchId }) => ({ term: term.value, distance, id: matchId })),
      expected,
      `trace ${trace}`,
    );
    assert.deepEqual(
      collect(transducer.query("", 64, "distance-then-term")).map(({ term }) => term),
      [`after-${trace}`],
    );
    cursor.close();
    transducer.close();
    dictionary.close();
  }
});

test("u64 dictionaries and batch reduction remain streaming", () => {
  const dictionary = libdictenstein.dynamicDawg("u64");
  dictionary.putU64(new BigUint64Array([1n, 2n]), 8n);
  assert.deepEqual(dictionary.lookupU64(new BigUint64Array([1n, 2n])), { found: true, value: 8n });
  const transducer = liblevenshtein.transducer(dictionary);
  const count = transducer
    .query(new BigUint64Array([1n, 3n]), 1)
    .reduceBatches((sum, batch) => sum + batch.length, 0, 1);
  assert.equal(count, 1);
  transducer.close();
  dictionary.close();
});

test("browser dictionaries expose host-owned Map collection snapshots", () => {
  const dictionary = libdictenstein.dynamicDawg();
  dictionary.set("cat", 1n).set("caff", null).set("dog", 3n);
  assert.equal(dictionary.get("cat"), 1n);
  assert.equal(dictionary.get("caff"), null);
  assert.equal(dictionary.get("absent"), undefined);

  const entries = dictionary.entries();
  const cursor = dictionary.streamEntries();
  assert.equal(cursor.size, 3);
  dictionary.delete("cat");
  dictionary.set("zebra", 9n);
  assert.deepEqual([...entries], [["caff", null], ["cat", 1n], ["dog", 3n]]);
  assert.deepEqual([...cursor], [["caff", null], ["cat", 1n], ["dog", 3n]]);
  assert.equal(Object.isFrozen(dictionary.snapshotEntries()), true);
  assert.deepEqual([...dictionary.keys()], ["caff", "dog", "zebra"]);
  assert.deepEqual([...dictionary.values()], [null, 3n, 9n]);
  assert.deepEqual([...dictionary.toMap()], [...dictionary]);
  dictionary.close();

  const bytes = libdictenstein.dynamicDawg("byte");
  bytes.set(new Uint8Array([0, 255]), 4n);
  const [[byteKey, byteValue]] = [...bytes];
  assert.deepEqual([...byteKey], [0, 255]);
  assert.equal(byteValue, 4n);
  bytes.close();

  const tokens = libdictenstein.dynamicDawg("u64");
  tokens.set(new BigUint64Array([0n, 2n ** 63n]), null);
  const [[tokenKey, tokenValue]] = [...tokens];
  assert.deepEqual([...tokenKey], [0n, 2n ** 63n]);
  assert.equal(tokenValue, null);
  tokens.close();
});

test("duallity WFST composes lazily with a lling-llang VectorWfst", () => {
  const dictionary = libdictenstein.dynamicDawg();
  dictionary.put("cat", 1n);
  const edit = duallity.wfst(dictionary, "cat", 1);
  dictionary.clear();
  dictionary.close();

  const builder = llingLlang.vectorWfst();
  const states = [builder.addState(), builder.addState(), builder.addState(), builder.addState()];
  builder.setStart(states[0]);
  builder.setFinal(states[3], 0);
  for (const [index, [input, output]] of [["c", "C"], ["a", "A"], ["t", "T"]].entries()) {
    builder.addArc(states[index], input, output, states[index + 1], 0);
  }
  const uppercase = builder.build();
  builder.close();
  const composed = llingLlang.compose(edit, uppercase);
  edit.close();
  uppercase.close();

  const seen = new Set();
  const pending = [[composed.start(), ""]];
  let accepted = false;
  while (pending.length > 0) {
    const [state, output] = pending.pop();
    const key = `${state}:${output}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const expanded = composed.state(state);
    if (expanded.final && output === "CAT") accepted = true;
    for (const arc of expanded.arcs) pending.push([arc.target, output + (arc.output ?? "")]);
  }
  assert.equal(accepted, true);
  composed.close();
});
