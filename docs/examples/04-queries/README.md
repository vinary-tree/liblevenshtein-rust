# 04 · Queries, Unicode & Custom Substitutions

**What you'll learn.** How `liblevenshtein` handles **Unicode** and how to declare that
certain character pairs should be treated as *free* (zero-cost) substitutions — so that
`é` matches `e`, `Α` matches $`\alpha`$, or `あ` matches `ア` without spending an edit. You'll
use `SubstitutionSetChar`, both via ready-made presets (Latin diacritics, Greek /
Cyrillic case-folding, Japanese kana) and by building a custom set pair-by-pair.

---

## The concept

### Why Unicode needs more than bytes

Over a byte alphabet (`u8`), `é` is *two* bytes (`U+00E9` is `0xC3 0xA9` in UTF-8), so a
byte-level automaton would mis-count edits across multi-byte characters. The `…Char`
family operates over **Unicode scalar values** (`char` / `u32`) instead, so one accented
letter is one symbol and edit distances are computed in *characters*, not bytes.

> Terms defined. A **Unicode scalar value** is a single `char` (any code point except
> surrogates). A **diacritic** is an accent mark such as the acute in `é` or the tilde in
> `ñ`. **Case-folding** treats an upper- and lower-case letter as equivalent ($`A \equiv \alpha`$).

### What a substitution set does

By default, replacing one symbol with a *different* symbol costs one edit. A
**substitution set** is a relation $`\chi \subseteq \Sigma \times \Sigma`$ of ordered pairs `(a, b)` that the
automaton is allowed to treat as a **zero-cost** substitution — i.e. `a` and `b` are
considered "the same" for matching purposes. `SubstitutionSetChar` is the Unicode
(`char`) form. This is the mechanism behind *restricted* and *generalized* edits: instead
of "any symbol may substitute for any other at cost 1", you whitelist exactly which
pairs are free.

`SubstitutionSetChar` ships **preset builders** for common scripts and exposes a
two-method core:

- `allow(a, b)` — add the ordered pair `(a, b)` (call both directions for symmetry).
- `contains(a, b)` — test whether `(a, b)` is in the set.
- `len()` — how many pairs the set holds.

![Levenshtein NFA: the non-deterministic automaton for a query, showing the match, insertion, deletion, and substitution transitions out of each (position, errors) state — the substitution edges are the ones a substitution set can make free.](../../diagrams/automata/levenshtein-nfa.svg)

### Why presets *and* custom sets

International search ("résumé" should match "resume"), case-insensitive matching in
non-Latin scripts, and script-bridging (Greek $`\alpha`$ ↔ Latin `a`) are common enough to
warrant batteries-included presets — but domain glossaries (chemical symbols, currency
signs, emoji skin-tone variants) need ad-hoc pairs, so you can also build a set by hand.

---

## Walking through `examples/unicode_diacritics.rs`

### 1 · Latin diacritics preset

`diacritics_latin()` returns a set where accented Latin letters are equivalent to their
unaccented bases, in both directions:

```rust
use liblevenshtein::transducer::SubstitutionSetChar;

let diacritics = SubstitutionSetChar::diacritics_latin();
println!("{} pairs", diacritics.len());

assert!(diacritics.contains('é', 'e'));   // accented ↔ base
assert!(diacritics.contains('e', 'é'));   // reverse direction included
assert!(diacritics.contains('ñ', 'n'));
assert!(!diacritics.contains('x', 'y'));  // unrelated pair: still a real edit
```

### 2 · Case-folding presets for Greek and Cyrillic

`greek_case_insensitive()` and `cyrillic_case_insensitive()` fold upper- and lower-case
across the whole alphabet — including special forms such as Greek final sigma $`\varsigma`$:

```rust
let greek = SubstitutionSetChar::greek_case_insensitive();
assert!(greek.contains('Α', 'α'));   // Alpha
assert!(greek.contains('Σ', 'ς'));   // Sigma, final form
assert!(!greek.contains('Α', 'Β'));  // different letters: not free

let cyrillic = SubstitutionSetChar::cyrillic_case_insensitive();
assert!(cyrillic.contains('Я', 'я'));   // Ya
```

### 3 · Japanese Hiragana ↔ Katakana

`japanese_hiragana_katakana()` makes the two kana syllabaries interchangeable, so a query
in one script matches dictionary entries in the other:

```rust
let japanese = SubstitutionSetChar::japanese_hiragana_katakana();
assert!(japanese.contains('あ', 'ア'));   // a
assert!(japanese.contains('か', 'カ'));   // ka
assert!(!japanese.contains('あ', 'か'));  // different syllables
```

### 4 · Build a custom set pair-by-pair

When you need bridges the presets don't cover — here Greek letters to their Latin
look-alikes — call `new()` then `allow(a, b)` for each direction:

```rust
let mut custom = SubstitutionSetChar::new();
custom.allow('α', 'a'); custom.allow('a', 'α');   // Greek alpha ↔ Latin a
custom.allow('β', 'b'); custom.allow('b', 'β');   // Greek beta  ↔ Latin b
custom.allow('π', 'p'); custom.allow('p', 'π');   // Greek pi    ↔ Latin p

assert!(custom.contains('α', 'a'));
assert!(custom.contains('π', 'p'));
```

A `SubstitutionSetChar` built this way is then handed to the Unicode transducer
machinery (see the crate's *Restricted & Custom Substitutions* and the `…Char`
dictionaries) so those whitelisted swaps cost zero edits during a fuzzy query.

---

## Run it

No features required:

```bash
cargo run --example unicode_diacritics
```

The program prints each preset's pair count and verifies a battery of equivalences for
Latin, Greek, Cyrillic, and Japanese, then demonstrates a hand-built Greek↔Latin set.

> Related: `examples/custom_substitutions.rs` shows the byte-level `SubstitutionSet`
> (combining sets for domain-specific matching), and `examples/dynamic_dawg_unicode.rs`
> shows the Unicode `DynamicDawgChar` dictionary the substitution sets pair with.

---

## Key takeaways

- Use the **`…Char`** types for Unicode so edit distance counts *characters*, not bytes.
- A **`SubstitutionSetChar`** is a relation of ordered pairs the automaton may substitute
  **at zero cost** — the knob for diacritic-insensitive, case-insensitive, and
  script-bridging matching.
- Reach for a **preset** (`diacritics_latin`, `greek_case_insensitive`,
  `cyrillic_case_insensitive`, `japanese_hiragana_katakana`) when one fits; otherwise
  `new()` + `allow(a, b)` builds exactly the pairs you need.

---

[← 03 · Algorithms](../03-algorithms/README.md) · Next: [05 · Values & Fuzzy Maps →](../05-values/README.md)

[← Documentation Index](../../README.md)
