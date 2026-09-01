import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url)));

test("all project facades share the exact shared JavaScript runtime", () => {
  assert.equal(packageJson.dependencies["@vinary-tree/javascript-runtime"], packageJson.version);
  assert.equal(packageJson.dependencies["@vinary-tree/vinary-tree-interop"], "4.0.0-rc.6");
  for (const path of [".", "./typescript", "./clojurescript", "./wasm", "./wasi"]) {
    assert.ok(packageJson.exports[path]);
  }
});

test("project package does not own dictionary constructors", async () => {
  const declarations = await readFile(new URL("../index.d.ts", import.meta.url), "utf8");
  assert.doesNotMatch(declarations, /class\s+(?:Dynamic|Persistent|Scdawg|DoubleArray)/);
  assert.match(declarations, /DictionaryResource/);
  assert.match(declarations, /reduceBatches/);
});

// C1/C3/C5: a delegated shim earns its keep only if it re-exports the entire
// contract surface. Pin the streaming cursor (pull iterator + push reducer +
// batch paging + disposal), the phonetic automata, and the borrowed-resource
// transducer entry point, so a future trim of the umbrella surface cannot slip
// past this facade unnoticed.
test("delegated facade re-exports the full streaming and phonetic surface", async () => {
  const declarations = await readFile(new URL("../index.d.ts", import.meta.url), "utf8");
  // C5: one cursor is both a pull iterator and a push reducer, with paging.
  assert.match(declarations, /interface QueryCursor extends IterableIterator<Match>/);
  for (const member of ["nextBatch(", "reduceBatches<", "close()"]) {
    assert.ok(declarations.includes(member), `QueryCursor must declare ${member}`);
  }
  // C3/C6: phonetic pattern matching and rewrite-rule application travel here.
  assert.match(declarations, /interface PhoneticPattern[\s\S]*?matches\(input: string\): boolean/);
  assert.match(declarations, /interface PhoneticRuleSet[\s\S]*?apply\(input: string\): string/);
  // C1: the transducer only ever borrows a DictionaryResource; it never
  // constructs storage, keeping ownership on the delegated shared JavaScript runtime.
  assert.match(declarations, /transducer\(dictionary: DictionaryResource/);
});

test("public facade executes Algorithm, QueryOrder, domain, and phonetic contracts", async () => {
  const facade = await import("../facades/native.mjs");
  const { libdictenstein } = await import("@vinary-tree/javascript-runtime");
  assert.equal(facade.runtimeIdentity, libdictenstein.runtimeIdentity);

  const collect = (cursor) => {
    try {
      return [...cursor];
    } finally {
      cursor.close();
    }
  };
  const queryWith = (dictionary, algorithm, input, maximumDistance, order) => {
    const automaton = facade.transducer(dictionary, algorithm);
    try {
      return collect(automaton.query(input, maximumDistance, order));
    } finally {
      automaton.close();
    }
  };

  const dictionary = libdictenstein.dynamicDawg();
  for (const [term, id] of [
    ["ab", 1n], ["c", 2n], ["abc", 3n],
    ["bat", 4n], ["cat", 5n], ["cats", 6n],
  ]) {
    dictionary.put(term, id);
  }
  try {
    assert.equal(
      queryWith(dictionary, "standard", "ba", 1)
        .some(({ term }) => term.value === "ab"),
      false,
    );
    assert.ok(
      queryWith(dictionary, "transposition", "ba", 1)
        .some(({ term, distance }) => term.value === "ab" && distance === 1),
    );
    assert.ok(
      queryWith(dictionary, "merge-and-split", "ab", 1)
        .some(({ term, distance }) => term.value === "c" && distance === 1),
    );
    assert.ok(
      queryWith(dictionary, "damerau-levenshtein", "ca", 2)
        .some(({ term, distance }) => term.value === "abc" && distance === 2),
    );

    assert.deepEqual(
      queryWith(dictionary, "standard", "cat", 1, "traversal")
        .map(({ term, distance }) => [term.value, distance]),
      [["bat", 1], ["cat", 0], ["cats", 1]],
    );
    assert.deepEqual(
      queryWith(dictionary, "standard", "cat", 1, "distance-then-term")
        .map(({ term, distance }) => [term.value, distance]),
      [["cat", 0], ["bat", 1], ["cats", 1]],
    );

    const regex = facade.phoneticPattern("c[ao]t");
    try {
      assert.ok(regex.size.states > 0);
      assert.ok(regex.size.transitions > 0);
      assert.equal(regex.matches("cat"), true);
      assert.deepEqual(
        queryWith(dictionary, "standard", regex, 0)
          .map(({ term }) => term.value),
        ["cat"],
      );
    } finally {
      regex.close();
    }

    const llre = facade.llrePattern('@name "Greeting"\n^hello$');
    try {
      assert.equal(llre.matches("hello"), true);
      assert.equal(llre.matches("world"), false);
    } finally {
      llre.close();
    }

    const parsedRules = facade.phoneticRules("ph -> f; gh -> ;");
    try {
      assert.equal(parsedRules.size, 2);
      assert.equal(parsedRules.apply("graph"), "graf");
    } finally {
      parsedRules.close();
    }
    const builtInRules = facade.phoneticRules("english-orthography");
    try {
      assert.ok(builtInRules.size > 0);
    } finally {
      builtInRules.close();
    }
  } finally {
    dictionary.close();
  }

  const byteDictionary = libdictenstein.dynamicDawg("byte");
  const bytes = new Uint8Array([0, 1, 255]);
  byteDictionary.put(bytes, 7n);
  try {
    const [match] = queryWith(byteDictionary, "standard", bytes, 0);
    assert.equal(match.term.domain, "byte");
    assert.deepEqual(match.term.value, bytes);
  } finally {
    byteDictionary.close();
  }

  const u64Dictionary = libdictenstein.dynamicDawg("u64");
  const tokens = new BigUint64Array([0n, 1n, 0xffff_ffff_ffff_ffffn]);
  u64Dictionary.putU64(tokens, 8n);
  try {
    const [match] = queryWith(u64Dictionary, "standard", tokens, 0);
    assert.equal(match.term.domain, "u64");
    assert.deepEqual(match.term.value, tokens);
  } finally {
    u64Dictionary.close();
  }
});
