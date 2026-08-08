import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { libdictenstein, liblevenshtein, llingLlang, duallity, runtimeIdentity } from "../native.mjs";

function collect(cursor) {
  try { return [...cursor].map(({ term, distance, id }) => [term.value, distance, id]); }
  finally { cursor.close(); }
}

test("native N-API uses one cross-project runtime and exact snapshots", () => {
  assert.equal(libdictenstein.runtimeIdentity, runtimeIdentity);
  assert.equal(liblevenshtein.runtimeIdentity, runtimeIdentity);
  for (let trace = 0; trace < 64; trace += 1) {
    const dictionary = libdictenstein.dynamicDawg();
    for (let index = 0; index < 16; index += 1) dictionary.put(`t${trace}-${index}`, BigInt(index));
    const transducer = liblevenshtein.transducer(dictionary);
    const expected = collect(transducer.query("", 64, "distance-then-term"));
    const cursor = transducer.query("", 64, "distance-then-term");
    const actual = [cursor.next().value];
    dictionary.remove(`t${trace}-1`);
    dictionary.put(`t${trace}-2`, 999n);
    actual.push(cursor.next().value);
    dictionary.clear();
    dictionary.compact();
    dictionary.put(`after-${trace}`, 1000n);
    actual.push(...cursor);
    assert.deepEqual(actual.map(({ term, distance, id }) => [term.value, distance, id]), expected);
    cursor.close();
    dictionary.close();
    const after = `after-${trace}`;
    assert.deepEqual(collect(transducer.query("", 64)), [[after, [...after].length, 1000n]]);
    transducer.close();
  }
});

test("native DAT, SCDAWG, phonetic, distances, and persistent ARTrie", async () => {
  const dat = libdictenstein.doubleArrayTrie([{ term: "café", value: 7n }, { term: "caff", value: null }]);
  assert.deepEqual(dat.get("caff"), { found: true, value: null });
  dat.close();
  const suffixes = libdictenstein.scdawg();
  suffixes.put("cat", 1n);
  suffixes.put("cot", 2n);
  assert.equal(suffixes.containsSubstring("ot"), true);
  assert.equal(suffixes.substringFrequency("t"), 2);
  suffixes.close();
  assert.equal(liblevenshtein.levenshteinDistance("kitten", "sitting"), 3);
  const pattern = liblevenshtein.phoneticPattern("c[ao]t");
  assert.equal(pattern.matches("cat"), true);
  pattern.close();

  const directory = await mkdtemp(join(tmpdir(), "vinary-tree-native-"));
  const path = join(directory, "words.artrie");
  try {
    let dictionary = libdictenstein.createPersistentARTrie(path);
    dictionary.put("cat", 1n);
    dictionary.checkpoint();
    dictionary.close();
    dictionary = libdictenstein.openPersistentARTrie(path);
    assert.deepEqual(dictionary.get("cat"), { found: true, value: 1n });
    dictionary.close();
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("native byte and u64 domains remain typed and streaming", () => {
  const bytes = libdictenstein.dynamicDawg("byte");
  bytes.put("cat", 1n);
  const byteTransducer = liblevenshtein.transducer(bytes);
  assert.deepEqual(collect(byteTransducer.query(new Uint8Array([99, 117, 116]), 1)), [
    [new Uint8Array([99, 97, 116]), 1, 1n],
  ]);
  byteTransducer.close();
  bytes.close();

  const tokens = libdictenstein.dynamicDawg("u64");
  tokens.putU64(new BigUint64Array([1n, 2n]), 8n);
  assert.deepEqual(tokens.getU64(new BigUint64Array([1n, 2n])), { found: true, value: 8n });
  const tokenTransducer = liblevenshtein.transducer(tokens);
  assert.deepEqual(collect(tokenTransducer.query(new BigUint64Array([1n, 3n]), 1)), [
    [new BigUint64Array([1n, 2n]), 1, 8n],
  ]);
  tokenTransducer.close();
  tokens.close();
});

test("native duallity and lling-llang share retained scalar WFST resources", () => {
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
