import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { libdictenstein, liblevenshtein, llingLlang, duallity, runtimeIdentity } from "../native.mjs";

function collect(cursor) {
  try { return [...cursor].map(({ term, distance, id }) => [term.value, distance, id]); }
  finally { cursor.close(); }
}

/** Extract every `export interface` body from a .d.ts source. */
function interfaceBodies(declarations) {
  const result = new Map();
  for (const match of declarations.matchAll(/export interface (\w+)[^{]*\{/g)) {
    let depth = 1;
    let index = match.index + match[0].length;
    const start = index;
    while (index < declarations.length && depth > 0) {
      if (declarations[index] === "{") depth += 1;
      if (declarations[index] === "}") depth -= 1;
      index += 1;
    }
    result.set(match[1], declarations.slice(start, index - 1));
  }
  return result;
}

/** Split one interface body into declared method and property names. */
function declaredMembers(body) {
  const methods = new Set();
  const properties = new Set();
  // Members terminate at semicolons outside every nested brace, paren, and
  // bracket, so inline object types and parameter lists never leak members.
  const fragments = [];
  let depth = 0;
  let current = "";
  for (const character of body) {
    if ("{([".includes(character)) depth += 1;
    if ("})]".includes(character)) depth -= 1;
    if (character === ";" && depth === 0) {
      fragments.push(current);
      current = "";
    } else {
      current += character;
    }
  }
  fragments.push(current);
  for (const fragment of fragments) {
    const declaration = fragment.trim();
    const method = declaration.match(/^(\w+)\s*[<(]/);
    const property = declaration.match(/^readonly\s+(\w+)\??:/)
      ?? declaration.match(/^(\w+)\??:/);
    if (method) methods.add(method[1]);
    else if (property) properties.add(property[1]);
  }
  return { methods, properties };
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

test("native thresholded distances mirror their unthresholded siblings", () => {
  // Early-exit semantics: a plain number whenever the true distance fits the
  // maximum (including exactly at it), undefined once the bound is exceeded.
  assert.equal(liblevenshtein.levenshteinDistance("kitten", "sitting"), 3);
  assert.equal(liblevenshtein.levenshteinDistanceThreshold("kitten", "sitting", 3), 3);
  assert.equal(liblevenshtein.levenshteinDistanceThreshold("kitten", "sitting", 64), 3);
  assert.equal(liblevenshtein.levenshteinDistanceThreshold("kitten", "sitting", 2), undefined);

  // OSA counts ca -> abc as 3 while unrestricted Damerau-Levenshtein reaches
  // 2, so the pair also pins each threshold variant to the right algorithm.
  assert.equal(liblevenshtein.damerauDistance("ca", "abc"), 3);
  assert.equal(liblevenshtein.damerauDistanceThreshold("ca", "abc", 3), 3);
  assert.equal(liblevenshtein.damerauDistanceThreshold("ca", "abc", 2), undefined);
  assert.equal(liblevenshtein.trueDamerauDistance("ca", "abc"), 2);
  assert.equal(liblevenshtein.trueDamerauDistanceThreshold("ca", "abc", 2), 2);
  assert.equal(liblevenshtein.trueDamerauDistanceThreshold("ca", "abc", 1), undefined);

  assert.equal(liblevenshtein.damerauDistance("abcd", "acbd"), 1);
  assert.equal(liblevenshtein.damerauDistanceThreshold("abcd", "acbd", 1), 1);
  assert.equal(liblevenshtein.damerauDistanceThreshold("abcd", "acbd", 0), undefined);
});

test("native phonetic pattern size and rule-set size", () => {
  // Any automaton accepting exactly {cat, cot} needs at least four states
  // (a length-three path plus the start) and four labelled transitions.
  const pattern = liblevenshtein.phoneticPattern("c[ao]t");
  const size = pattern.size;
  assert.ok(Number.isSafeInteger(size.states) && size.states >= 4, `states=${size.states}`);
  assert.ok(
    Number.isSafeInteger(size.transitions) && size.transitions >= 4,
    `transitions=${size.transitions}`,
  );
  pattern.close();
  assert.throws(() => pattern.size, /closed/);

  const rules = liblevenshtein.phoneticRules("ph -> f; gh -> ;");
  assert.equal(rules.size, 2);
  assert.equal(rules.apply("graph"), "graf");
  rules.close();
  assert.throws(() => rules.size, /closed/);

  const builtin = liblevenshtein.phoneticRules("english-orthography");
  assert.ok(Number.isSafeInteger(builtin.size) && builtin.size > 0);
  builtin.close();
});

test("every index.d.ts member exists on the native path", async () => {
  const declarations = await readFile(new URL("../index.d.ts", import.meta.url), "utf8");
  const interfaces = interfaceBodies(declarations);
  assert.ok(interfaces.size >= 12, "index.d.ts interface scan looks truncated");

  const dictionary = libdictenstein.dynamicDawg();
  dictionary.put("cat", 1n);
  const transducer = liblevenshtein.transducer(dictionary);
  const cursor = transducer.query("cat", 1);
  const pattern = liblevenshtein.phoneticPattern("c[ao]t");
  const rules = liblevenshtein.phoneticRules("english-orthography");
  const builder = llingLlang.vectorWfst();
  builder.setStart(builder.addState());
  const wfst = duallity.wfst(dictionary, "cat", 1);

  // Interfaces describing plain data records need no live instance; every
  // interface that declares behavior must map to one so a new declaration
  // cannot ship without a runtime member behind it.
  const instances = new Map([
    ["Dictionary", dictionary],
    ["QueryCursor", cursor],
    ["PhoneticPattern", pattern],
    ["PhoneticRuleSet", rules],
    ["Transducer", transducer],
    ["Wfst", wfst],
    ["WfstBuilder", builder],
    ["LibdictensteinNamespace", libdictenstein],
    ["LiblevenshteinNamespace", liblevenshtein],
    ["LlingLlangNamespace", llingLlang],
    ["DuallityNamespace", duallity],
  ]);
  try {
    for (const [name, body] of interfaces) {
      const { methods, properties } = declaredMembers(body);
      const instance = instances.get(name);
      if (instance === undefined) {
        assert.equal(
          methods.size,
          0,
          `interface ${name} declares methods but maps to no live instance in this scan`,
        );
        continue;
      }
      for (const method of methods) {
        assert.equal(
          typeof instance[method],
          "function",
          `${name}.${method} must be a function on the native path`,
        );
      }
      for (const property of properties) {
        assert.ok(property in instance, `${name}.${property} must exist on the native path`);
      }
    }
  } finally {
    cursor.close();
    wfst.close();
    builder.close();
    transducer.close();
    pattern.close();
    rules.close();
    dictionary.close();
  }
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
