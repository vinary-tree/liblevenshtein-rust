import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createWasiRuntime } from "../wasi-runtime.mjs";

function values(cursor) {
  try { return [...cursor].map(({ term, id }) => [term.value, id]); }
  finally { cursor.close(); }
}

test("WASI persistent ARTrie checkpoints, reopens, and retains query snapshots", async () => {
  const directory = await mkdtemp(join(tmpdir(), "vinary-tree-wasi-"));
  try {
    const runtime = await createWasiRuntime({ preopens: { "/data": directory } });
    let dictionary = runtime.libdictenstein.createPersistentARTrie("/data/words.artrie");
    for (const [term, id] of [["cat", 1n], ["cot", 2n], ["cut", null]]) dictionary.put(term, id);
    dictionary.checkpoint();

    const transducer = runtime.liblevenshtein.transducer(dictionary);
    const cursor = transducer.query("cat", 2, "distance-then-term");
    const first = cursor.next().value;
    dictionary.remove("cot");
    dictionary.put("cit", 4n);
    dictionary.checkpoint();
    assert.deepEqual(
      [first, ...cursor].map(({ term, id }) => [term.value, id]),
      [["cat", 1n], ["cot", 2n], ["cut", null]],
    );
    cursor.close();
    dictionary.close();
    assert.deepEqual(values(transducer.query("cat", 2, "distance-then-term")), [
      ["cat", 1n], ["cit", 4n], ["cut", null],
    ]);
    transducer.close();

    dictionary = runtime.libdictenstein.openPersistentARTrie("/data/words.artrie");
    assert.deepEqual(dictionary.lookup("cit"), { found: true, value: 4n });
    assert.deepEqual(dictionary.lookup("cot"), { found: false, value: null });
    dictionary.close();
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("WASI dictionaries expose closeable snapshot collection iterators", async () => {
  const runtime = await createWasiRuntime({ preopens: {} });
  const dictionary = runtime.libdictenstein.dynamicDawg();
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

  const bytes = runtime.libdictenstein.dynamicDawg("byte");
  bytes.set(new Uint8Array([0, 255]), 4n);
  const [[byteKey, byteValue]] = [...bytes];
  assert.deepEqual([...byteKey], [0, 255]);
  assert.equal(byteValue, 4n);
  bytes.close();

  const tokens = runtime.libdictenstein.dynamicDawg("u64");
  tokens.set(new BigUint64Array([0n, 2n ** 63n]), null);
  assert.equal(tokens.has(new BigUint64Array([0n, 2n ** 63n])), true);
  const [[tokenKey, tokenValue]] = [...tokens];
  assert.deepEqual([...tokenKey], [0n, 2n ** 63n]);
  assert.equal(tokenValue, null);
  tokens.close();
});

test("WASI dynamic cursor survives remove, update, clear, and compact", async () => {
  const runtime = await createWasiRuntime({ preopens: {} });
  for (let trace = 0; trace < 32; trace += 1) {
    const dictionary = runtime.libdictenstein.dynamicDawg();
    for (let index = 0; index < 12; index += 1) dictionary.put(`t${trace}-${index}`, BigInt(index));
    const transducer = runtime.liblevenshtein.transducer(dictionary);
    const expected = values(transducer.query("", 32, "distance-then-term"));
    const cursor = transducer.query("", 32, "distance-then-term");
    const actual = [cursor.next().value];
    dictionary.remove(`t${trace}-1`);
    dictionary.put(`t${trace}-2`, 99n);
    actual.push(cursor.next().value);
    dictionary.clear();
    dictionary.compact();
    dictionary.put(`after-${trace}`, 100n);
    actual.push(...cursor);
    assert.deepEqual(actual.map(({ term, id }) => [term.value, id]), expected, `trace ${trace}`);
    cursor.close();
    transducer.close();
    dictionary.close();
  }
});

test("WASI reducer and iterator drain a query to the same matches (C5)", async () => {
  const runtime = await createWasiRuntime({ preopens: {} });
  const dictionary = runtime.libdictenstein.dynamicDawg();
  for (const [term, id] of [["cat", 1n], ["cot", 2n], ["cut", 3n]]) dictionary.put(term, id);
  const transducer = runtime.liblevenshtein.transducer(dictionary);
  try {
    const byIterator = values(transducer.query("cat", 1))
      .map(([value, id]) => `${value}:${id}`)
      .sort();
    const reducerCursor = transducer.query("cat", 1);
    const byReducer = reducerCursor
      .reduceBatches((accumulator, batch) => {
        for (const { term, id } of batch) accumulator.push(`${term.value}:${id}`);
        return accumulator;
      }, [])
      .sort();
    reducerCursor.close();
    assert.deepEqual(byIterator, byReducer);
    assert.equal(byIterator.length, 3);
  } finally {
    transducer.close();
    dictionary.close();
  }
});

test("WASI duallity snapshots compose with lling-llang in the same instance", async () => {
  const runtime = await createWasiRuntime({ preopens: {} });
  const dictionary = runtime.libdictenstein.dynamicDawg();
  dictionary.put("cat", 1n);
  const edit = runtime.duallity.wfst(dictionary, "cat", 1);
  dictionary.clear();
  dictionary.close();

  const builder = runtime.llingLlang.vectorWfst();
  const states = [builder.addState(), builder.addState(), builder.addState(), builder.addState()];
  builder.setStart(states[0]);
  builder.setFinal(states[3], 0);
  for (const [index, [input, output]] of [["c", "C"], ["a", "A"], ["t", "T"]].entries()) {
    builder.addArc(states[index], input, output, states[index + 1], 0);
  }
  const uppercase = builder.build();
  const composed = runtime.llingLlang.compose(edit, uppercase);
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
