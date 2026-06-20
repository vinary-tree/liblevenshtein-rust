# Compositional Spelling Correction: Phonetic NFAs + Levenshtein Automata

*A Pedagogical Guide to Finding Spelling Candidates*

---

## Introduction: The Spelling Correction Problem

When a user types "fone", we want to suggest "phone". When they type "elefant", we want to suggest "elephant". This seems simple, but the underlying algorithms are surprisingly elegant.

The key insight is that spelling errors come in two flavors:

1. **Phonetic errors**: The user typed what they *heard* ("fone" sounds like "phone")
2. **Typographical errors**: The user's fingers hit wrong keys ("teh" instead of "the")

A robust spell checker must handle both. This guide explains how to combine two powerful automata:
- **Phonetic NFAs**: Match sound-alike patterns
- **Levenshtein automata**: Tolerate typos

> **Implementation Status:** Approach 1 (PhoneticNormalizedDictionary) is fully
> implemented in `src/dictionary/phonetic_normalized.rs`. Approach 3 (Product
> Automaton) is implemented in `src/phonetic/nfa/product.rs`. Approach 2
> (Pre-process Dictionary Only) is documented but not yet implemented.

---

## Chapter 1: Foundational Concepts

### 1.1 What is a Finite Automaton?

A finite automaton is a simple computing device that reads input one character at a time and decides whether to accept or reject the input.

```
Example: An automaton that accepts "cat"

    ┌───┐  c   ┌───┐  a   ┌───┐  t   ╔═══╗
    │ 0 │─────▶│ 1 │─────▶│ 2 │─────▶║ 3 ║
    └───┘      └───┘      └───┘      ╚═══╝
   start                            accept

State 0: Haven't seen anything yet
State 1: Saw "c"
State 2: Saw "ca"
State 3: Saw "cat" - ACCEPT!
```

If we're in state 2 and see "t", we move to state 3 (accepting). If we see anything else, we fail.

### 1.2 NFAs: Non-determinism Gives Us Choices

A **Deterministic Finite Automaton (DFA)** has exactly one next state for each input character. A **Non-deterministic Finite Automaton (NFA)** can have:

- **Multiple transitions** on the same character (try all paths)
- **Epsilon (ε) transitions**: Move without consuming input

```
Example: NFA that accepts "phone" OR "fone"

    ┌───┐  p   ┌───┐  h   ┌───┐  o   ┌───┐  n   ┌───┐  e   ╔═══╗
    │ 0 │─────▶│ 1 │─────▶│ 2 │─────▶│ 3 │─────▶│ 4 │─────▶║ 5 ║
    └───┘      └───┘      └───┘      └───┘      └───┘      ╚═══╝
      │                     ▲
      │  f   ┌───┐    ε     │
      └─────▶│ 6 │──────────┘
             └───┘

State 0: Start - can take 'p' OR 'f' path
States 1,2: The "ph" path (consuming 'p' then 'h')
State 6: The "f" path (ε-transitions to state 2)
State 2: Implicit merge point (both paths converge here)
States 3,4,5: Shared "one" suffix (5 is accepting)

From state 0, we can try BOTH paths simultaneously!
```

NFAs are perfect for representing phonetic patterns because sounds can be spelled multiple ways.

### 1.3 Phonetic Rewrite Rules

Phonetic rules express sound equivalences:

```
ph → f       "phone" sounds like "fone"
c → s / _[ei]  "c" before "e" or "i" sounds like "s" (like "cent" → "sent")
tion → shun  "nation" sounds like "nashun"
```

Each rule has:
- **Pattern**: What characters to match ("ph")
- **Replacement**: What to transform to ("f")
- **Context**: Optional constraints (only before vowels, only at word start, etc.)

### 1.4 Levenshtein Distance

The **Levenshtein distance** (edit distance) between two strings is the minimum number of single-character edits to transform one into the other.

**Edit operations**:
- **Insertion**: Add a character ("cat" → "cats")
- **Deletion**: Remove a character ("cats" → "cat")
- **Substitution**: Replace a character ("cat" → "car")

```
Example: "kitten" → "sitting" (distance = 3)

kitten
sitten  (substitution: k → s)
sittin  (substitution: e → i)
sitting (insertion: g)
```

### 1.5 Levenshtein Automata

A Levenshtein automaton accepts all strings within edit distance `n` of a query word. Rather than computing distance for every dictionary word, we build an automaton once and intersect it with the dictionary.

```
Query: "cat", max distance: 1

The automaton accepts:
- "cat" (distance 0)
- "cats", "scat" (1 insertion)
- "at", "ca" (1 deletion)
- "bat", "cot", "cap" (1 substitution)
```

---

## Chapter 2: The Three Compositional Approaches

Now we understand the building blocks. How do we combine phonetic patterns with typo tolerance?

### 2.1 The Core Insight: Composition

We have two automata:
1. **Phonetic NFA**: Accepts strings matching sound patterns
2. **Levenshtein automaton**: Accepts strings within edit distance

We want to accept strings that satisfy *both* conditions (with some combination of costs).

There are three ways to compose them, depending on when we apply phonetic transformations:

| Approach | Pre-process Query? | Pre-process Dictionary? | When to Use |
|----------|-------------------|------------------------|-------------|
| 1 | Yes | Yes | Maximum speed, less precision |
| 2 | No | Yes | Balance of speed and precision |
| 3 | No | No | Maximum precision, runtime flexibility |

---

## Chapter 3: Approach 1 — Pre-process Both Query and Dictionary

### 3.1 The Intuition

Imagine you're organizing a phone book by *pronunciation* rather than spelling. All words that sound alike go together:

```
Pronunciation "fon":
  - phone
  - fone
  - phon

Pronunciation "elefant":
  - elephant
```

At query time, normalize the query to its pronunciation, then look up matches.

### 3.2 Step-by-Step Walkthrough

**Build Time** (do once for the dictionary):

```
Step 1: Define phonetic normalization rules
  ph → f
  ough → o
  tion → shun
  silent e at end → (remove)
  ...

Step 2: Normalize every dictionary word
  "phone"    → normalize → "fon"
  "elephant" → normalize → "elefant"
  "knight"   → normalize → "nit"
  "night"    → normalize → "nit"  ← Same as knight!

Step 3: Build a trie indexed by normalized form
  Store: normalized_form → [original_word1, original_word2, ...]

  "fon" → ["phone", "phon"]
  "nit" → ["knight", "night", "nit"]
```

**Query Time** (do for each search):

```
User types: "fone"

Step 1: Normalize the query
  "fone" → normalize → "fon"

Step 2: Levenshtein search in normalized trie
  Find all normalized forms within distance n of "fon"
  Matches: "fon" (distance 0)

Step 3: Map back to original words
  "fon" → ["phone", "phon"]

Step 4: Return candidates
  "phone", "phon"
```

### 3.3 Worked Example

```
Dictionary: ["phone", "elephant", "elegance", "phony", "foe", "bone"]

Rules:
  ph → f
  silent e → (remove)

Build normalized index:
  "fon"     → ["phone"]
  "elefant" → ["elephant"]
  "elegans" → ["elegance"]
  "foni"    → ["phony"]
  "fo"      → ["foe"]
  "bon"     → ["bone"]

Query: "fone" (user meant "phone" but typed phonetically)

Step 1: normalize("fone") = "fon"
Step 2: Levenshtein search for "fon" with distance 1
        Matches: "fon" (d=0), "fo" (d=1), "bon" (d=1), "foni" (d=1)
Step 3: Map back:
        "fon"  → "phone"     (best match!)
        "fo"   → "foe"       (1 typo)
        "bon"  → "bone"      (1 typo)
        "foni" → "phony"     (1 typo)
```

### 3.4 Pros and Cons

**Advantages**:
- **Fastest query time**: Just one normalization + one Levenshtein search
- **Simplest implementation**: Reuses existing Levenshtein machinery
- **Small memory**: Normalized trie is same size as original

**Disadvantages**:
- **Information loss**: "phone" and "fone" become indistinguishable
- **Can't separate costs**: Don't know if match was phonetic or typo
- **Rebuild required**: Changing rules requires re-indexing

### 3.5 Pseudocode

```python
class PhoneticNormalizedDictionary:
    """
    Uses a FuzzyMultiMap (trie-based multimap with Levenshtein search) to store
    normalized forms mapped to their original terms. This is more efficient
    than maintaining a separate trie + hashmap because:

    1. Unified structure - no separate data structures to synchronize
    2. Memory efficient - common prefixes in normalized forms share storage
    3. Built-in fuzzy search - FuzzyMultiMap already supports Levenshtein lookup
    4. Native multimap - multiple originals per normalized form handled naturally
    """

    def __init__(self, terms, rules):
        self.rules = rules
        # FuzzyMultiMap stores: normalized_form → [original_terms]
        # Uses trie structure internally for efficient prefix sharing
        # Multimap semantics: same key can have multiple values
        self.normalized_map = FuzzyMultiMap()

        for term in terms:
            normalized = self.normalize(term)
            # FuzzyMultiMap handles multiple values per key natively
            self.normalized_map.insert(normalized, term)

    def normalize(self, word):
        """Apply rules until fixed point"""
        while True:
            new_word = apply_rules_once(word, self.rules)
            if new_word == word:
                return word
            word = new_word

    def query(self, query, max_distance):
        """Find candidates for misspelled query"""
        normalized_query = self.normalize(query)

        # FuzzyMultiMap.fuzzy_get returns (key, distance, values) tuples
        # The trie structure enables efficient Levenshtein search
        # while the associated values give us the original terms directly
        candidates = []
        for norm_form, distance, originals in self.normalized_map.fuzzy_get(
            normalized_query, max_distance
        ):
            for original in originals:
                candidates.append(Candidate(term=original, distance=distance))

        return candidates
```

**Why FuzzyMultiMap instead of HashMap + Trie?**

The naive approach uses two separate structures:
- A `Trie` for storing normalized forms and enabling Levenshtein search
- A `HashMap<String, Vec<String>>` for mapping normalized forms back to originals

Using liblevenshtein's `FuzzyMultiMap` (a trie-based multimap) combines both:
- The trie structure stores normalized forms with shared prefixes
- Each key can have multiple associated values (the original terms)
- Levenshtein search returns both the matched key AND all its values
- Native multimap semantics eliminate manual list management

This eliminates redundancy and keeps the implementation consistent with
liblevenshtein's existing patterns.

### 3.6 Rust Implementation

The `PhoneticNormalizedDictionary` in `src/dictionary/phonetic_normalized.rs`
implements this approach. Here's how to use it:

```rust
use liblevenshtein::prelude::*;

// Create dictionary with default Zompist rules
let dict = PhoneticNormalizedDictionary::from_terms([
    "phone", "fone", "elephant", "elegance"
]);

// Query - "fone" normalizes to same as "phone"
let results = dict.query("fone", 0);
// Returns both "phone" and "fone"

// With edit distance tolerance
let results = dict.query("elefant", 1);
// Returns "elephant"

// Regex query against normalized forms
let results = dict.query_regex("(f|b)on", 0)?;
// Returns terms whose normalized forms match pattern

// Auto-expand query to match phonetic variants
let results = dict.query_phonetic_pattern("fone", 1)?;
// Expands "fone" to "(ph|f)one" pattern, matches variants
```

The implementation provides:
- `from_terms()` / `from_terms_with_rules()` - construct from term iterator
- `query()` - fuzzy search with Levenshtein tolerance
- `query_regex()` - grep-like pattern matching against normalized forms
- `query_phonetic_pattern()` - auto-expand queries to match phonetic variants
- `normalize()` - get the normalized form of any term

---

## Chapter 4: Approach 2 — Pre-process Dictionary Only

### 4.1 The Intuition

Instead of collapsing words to one pronunciation, *expand* each word to all possible spellings:

```
"phone" → ["phone", "fone", "phon", "fon", ...]

Now we can search for the user's exact input among all variants.
```

This preserves the distinction between variants while still doing heavy work at build time.

### 4.2 Step-by-Step Walkthrough

**Build Time**:

```
Step 1: For each dictionary word, enumerate ALL phonetic variants
  "phone" → rule ph→f → "fone"
          → rule silent e → "phon"
          → both rules → "fon"
  Variants of "phone": {"phone", "fone", "phon", "fon"}

Step 2: Build trie of all variants
  Store: variant → (original_word, phonetic_cost)

  "phone" → ("phone", cost=0.0)
  "fone"  → ("phone", cost=0.1)  # ph→f costs 0.1
  "phon"  → ("phone", cost=0.1)  # silent e costs 0.1
  "fon"   → ("phone", cost=0.2)  # both transformations
```

**Query Time**:

```
User types: "fone"

Step 1: Search for "fone" in variant trie (exact or with Levenshtein)
  Found: "fone" → ("phone", phonetic_cost=0.1)

Step 2: Return with combined cost
  Candidate: "phone"
    - Phonetic cost: 0.1 (for ph→f transformation)
    - Edit distance: 0 (exact match to variant)
    - Total: 0.1
```

### 4.3 Worked Example with Cost Tracking

```
Dictionary: ["phone", "cat"]

Rules (with costs):
  ph → f  (cost 0.1)
  c → k   (cost 0.1)
  silent e → (remove)  (cost 0.05)

Expand dictionary:
  From "phone":
    "phone" (cost 0.0)
    "fone"  (cost 0.1)     ← ph→f
    "phon"  (cost 0.05)    ← drop e
    "fon"   (cost 0.15)    ← ph→f + drop e

  From "cat":
    "cat" (cost 0.0)
    "kat" (cost 0.1)       ← c→k

Query: "fon" with max_distance=1

Step 1: Search variant trie for "fon"
        Exact: "fon" → ("phone", phonetic=0.15, edit=0)

Step 2: Also check Levenshtein neighbors
        "fo" → no match
        "kon" → no match
        "fan" → no match
        "fone" → ("phone", phonetic=0.1, edit=1)  ← insertion of 'e'

Step 3: Combine and rank
        "phone" via "fon":  total = 0.15 + 0 = 0.15  ✓ best
        "phone" via "fone": total = 0.1 + 1 = 1.1
```

### 4.4 Handling Variant Explosion

Some words have many variants. "through" with common rules might generate dozens. We must limit expansion:

```python
def expand_with_limit(word, rules, max_variants=100):
    """Enumerate variants with combinatorial explosion protection"""
    variants = {word: 0.0}  # variant → cost
    queue = [(word, 0.0)]

    while queue and len(variants) < max_variants:
        current, cost = queue.pop(0)

        for rule in rules:
            if rule.matches(current):
                new_variant = rule.apply(current)
                new_cost = cost + rule.weight

                if new_variant not in variants or variants[new_variant] > new_cost:
                    variants[new_variant] = new_cost
                    queue.append((new_variant, new_cost))

    return variants
```

### 4.5 Pros and Cons

**Advantages**:
- **Separates phonetic and edit costs**: Know exactly why a match occurred
- **Query uses raw input**: No normalization step needed
- **Supports asymmetric rules**: "ph→f" doesn't require "f→ph"

**Disadvantages**:
- **Memory explosion**: Each word becomes many variants (5-20× typical)
- **Long build time**: Must enumerate all variants
- **Complex deduplication**: Same original found via multiple paths

### 4.6 Pseudocode

```python
class PhoneticExpandedDictionary:
    def __init__(self, terms, rules, max_variants_per_term=100):
        self.variant_trie = Trie()
        self.variant_to_original = {}  # variant → [(original, cost), ...]

        for term in terms:
            variants = expand_with_limit(term, rules, max_variants_per_term)

            for variant, phonetic_cost in variants.items():
                if variant not in self.variant_to_original:
                    self.variant_to_original[variant] = []
                    self.variant_trie.insert(variant)
                self.variant_to_original[variant].append((term, phonetic_cost))

    def query(self, query, max_distance):
        """Find candidates with cost decomposition"""
        results = []

        # Levenshtein search in variant trie
        matches = levenshtein_search(self.variant_trie, query, max_distance)

        for variant, edit_distance in matches:
            for original, phonetic_cost in self.variant_to_original[variant]:
                total_cost = edit_distance + phonetic_cost
                results.append(Candidate(
                    term=original,
                    edit_distance=edit_distance,
                    phonetic_cost=phonetic_cost,
                    total_cost=total_cost
                ))

        # Deduplicate: keep best path to each original term
        return deduplicate_by_original(results)
```

---

## Chapter 5: Approach 3 — No Pre-processing (Product Automaton)

### 5.1 The Intuition

What if we could search the dictionary *while simultaneously considering* both phonetic patterns and edit operations?

The **product automaton** does exactly this. We compose:
- Phonetic NFA (accepts sound-alike patterns)
- Levenshtein automaton (tolerates typos)

The result is a single automaton whose states track *both* where we are in the phonetic pattern *and* how many errors we've used.

![Phonetic NFA product pipeline: phonetic rules normalize a term, a Thompson-constructed NFA is intersected with a Levenshtein automaton, and the product automaton walks the dictionary trie lock-step to yield ranked candidates.](../diagrams/phonetic/nfa-product-pipeline.svg)

### 5.2 Product Automaton States

A state in the product automaton is a pair:

```
ProductState = (S_nfa, d)

Where:
  S_nfa = set of active NFA states (which phonetic paths are alive)
  d     = edit distance consumed so far
```

**Example**:

```
Query pattern: "(ph|f)one"  (matches "phone" or "fone")
Max distance: 1

Initial state: ({0}, 0)
  - NFA is at start (state 0)
  - Zero errors used

After seeing 'p':
  - Path 1: ({1}, 0)  ← NFA matched 'p', going toward "phone"
  - Path 2: ({0}, 1)  ← NFA stayed (insertion error), still at start

After seeing 'h' from Path 1:
  - ({2}, 0)  ← NFA matched 'h', now at "ph"

And so on...
```

### 5.3 The Seven Transition Types

When processing dictionary character `c` from state `(S, d)`:

#### Transition 1: Match (no cost)
```
If NFA can transition on 'c':
  S' = nfa_step(S, c)  ← advance NFA
  d' = d               ← same distance

This is a perfect match - the dictionary has what the pattern expects.
```

#### Transition 2: Substitution (cost +1)
```
If d < max_distance:
  S' = nfa_advance(S)  ← advance NFA by any edge
  d' = d + 1           ← pay for the mismatch

Dictionary has 'c', but NFA expected something else.
We pretend NFA got what it wanted, costing 1 edit.
```

#### Transition 3: Insertion (cost +1)
```
If d < max_distance:
  S' = S      ← NFA stays put
  d' = d + 1  ← pay for extra dictionary character

Dictionary has an extra character the pattern doesn't expect.
Consume 'c' without advancing the NFA.
```

#### Transition 4: Deletion (cost +1)
```
If d < max_distance:
  S' = nfa_advance(S)  ← NFA advances
  d' = d + 1           ← pay for missing dictionary character

Pattern expects a character the dictionary doesn't have.
Advance NFA without consuming dictionary character.
```

#### Transition 5: Transposition (cost +1, optional)
```
If d < max_distance AND next dictionary char exists:
  Look at c and c_next (adjacent characters)
  Try matching pattern "xy" against dictionary "yx"
  S' = nfa_step(nfa_step(S, c_next), c)  ← swapped order
  d' = d + 1
  Consume BOTH c and c_next

Handles common typos like "teh" → "the".
```

#### Transition 6: Merge (cost +1, optional)
```
If d < max_distance AND next dictionary char exists:
  Two dictionary chars → one pattern transition
  S' = nfa_advance(S)  ← one NFA step
  d' = d + 1
  Consume BOTH dictionary chars

Handles OCR errors like "rn" being scanned as "m".
```

#### Transition 7: Split (cost +1, optional)
```
If d < max_distance:
  One dictionary char → two pattern transitions
  S' = nfa_advance(nfa_advance(S))  ← two NFA steps
  d' = d + 1
  Consume ONE dictionary char

Handles cases like "ä" expanding to "ae".
```

### 5.4 Articulatory-Weighted Substitutions

The seven transition types above all use **fixed costs** (1.0 for each edit operation). However, not all substitutions are equally likely. A user who types "b" instead of "p" probably misheard or mistyped a similar sound, while "h" instead of "p" is less likely.

**Articulatory distance** provides gradient substitution costs based on IPA phonetic features:

| Substitution | Phonetic Difference | Cost |
|--------------|---------------------|------|
| p → b | Voicing only | ~0.1 |
| p → t | Adjacent place | ~0.45 |
| p → k | Distant place | ~1.0 |
| p → h | Place + manner | ~1.0 |

To enable articulatory-weighted substitutions, use `ProductAutomatonChar::with_articulatory_costs()`:

```rust
use liblevenshtein::phonetic::nfa::compiler::compile;
use liblevenshtein::phonetic::nfa::product::ProductAutomatonChar;
use liblevenshtein::phonetic::regex::parse;
use liblevenshtein::transducer::{Algorithm, ArticulatoryCosts};

// Compile phonetic pattern
let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();

// Create articulatory cost configuration
let costs = ArticulatoryCosts::default();

// Create product automaton with articulatory costs
let product = ProductAutomatonChar::with_articulatory_costs(
    nfa,
    2.0,  // max accumulated cost
    Algorithm::Standard,
    costs,
);

// "bone" now accepted because b→f costs only ~0.1 (voicing)
assert!(product.accepts("bone"));

// "hone" likely rejected because h→f costs ~1.0 (place + manner)
// depending on max_cost threshold
```

**Key insight**: With articulatory costs, the product automaton uses the query character and pattern character at each substitution transition to compute a phonetically-informed cost. This improves ranking quality for phonetic spell correction.

For detailed documentation on articulatory distance computation and configuration, see [Articulatory Distance Guide](articulatory-distance.md).

### 5.5 Visual Walkthrough

Let's trace through a complete example.

```
Query pattern: "(ph|f)one"  (NFA accepts "phone" or "fone")
Max distance: 1
Dictionary contains: "phone", "fone", "bone", "cone"

NFA structure:

   (0)──p──▶(1)──h──▶(2)──o──▶(3)──n──▶(4)──e──▶((5))
    │                 ▲
    └───f──▶(6)───ε───┘

State 0: start
States 1,2: "ph" path
State 6: "f" path (ε-transitions to state 2)
State 2: implicit merge point (both paths converge here)
States 3,4,5: shared "one" suffix (5 is accepting)

---

Searching for "phone" in dictionary:

Step 0: Initial
  ProductState = ({0}, 0)
  Queue: [({0}, 0, "")]

Step 1: See 'p'
  From ({0}, 0):
    Match 'p': NFA 0→1, so ({1}, 0, "p")  ✓
    Insert: ({0}, 1, "p")                  ✓
  Queue: [({1}, 0, "p"), ({0}, 1, "p")]

Step 2: See 'h' (continuing best path)
  From ({1}, 0):
    Match 'h': NFA 1→2, so ({2}, 0, "ph") ✓
  From ({0}, 1):
    Can't match 'h' from state 0, only 'p' or 'f'
    Subst: ({1 or 6}, 2) but d=2 > max, prune ✗

Step 3: See 'o'
  From ({2}, 0):
    Match 'o': NFA 2→3, so ({3}, 0, "pho") ✓

Step 4: See 'n'
  From ({3}, 0):
    Match 'n': NFA 3→4, so ({4}, 0, "phon") ✓

Step 5: See 'e'
  From ({4}, 0):
    Match 'e': NFA 4→5, so ({5}, 0, "phone") ✓
    State 5 is accepting!

Result: "phone" matches with distance 0

---

Searching for "bone" in dictionary:

Step 0: Initial
  ProductState = ({0}, 0)

Step 1: See 'b'
  From ({0}, 0):
    Can't match 'b' (NFA wants 'p' or 'f')
    Subst 'b' for 'p': ({1}, 1)  ✓
    Subst 'b' for 'f': ({6}, 1)  ✓
    Insert 'b': ({0}, 1)          ✓

Step 2: See 'o'
  From ({1}, 1):
    NFA 1 needs 'h', got 'o'
    Subst: ({2}, 2) but d=2 > max, prune ✗
  From ({6}, 1):
    NFA 6 → 2 on ε, then 2→3 on 'o'
    Actually: from state 6, ε→2, then need 'o'
    Match 'o': ({3}, 1) ✓
  From ({0}, 1):
    Insert: ({0}, 2) prune ✗

Step 3: See 'n'
  From ({3}, 1):
    Match 'n': ({4}, 1) ✓

Step 4: See 'e'
  From ({4}, 1):
    Match 'e': ({5}, 1) ✓
    State 5 is accepting!

Result: "bone" matches with distance 1 (substituted b for f)
```

### 5.6 Acceptance Condition

A dictionary word is accepted if:

1. We've consumed all characters in the word
2. The NFA can reach an accepting state
3. Total distance `≤ max_distance`

But wait—what if we've consumed the whole dictionary word but the NFA still needs more characters?

```python
def can_reach_final(nfa_states, current_distance, max_distance):
    """Can we reach NFA accepting state within remaining budget?"""
    remaining = max_distance - current_distance

    states = nfa_states
    for _ in range(remaining + 1):
        if any(nfa.is_final(s) for s in states):
            return True
        states = nfa_advance(states)  # Advance by deletion

    return False
```

Each step toward the accepting state costs 1 (deletion from pattern).

### 5.7 Intersection with Dictionary

The product automaton doesn't search in isolation—it simultaneously traverses the dictionary trie:

```
Dictionary Trie:
        ┌─────────────────┐
        │     (root)      │
        └───────┬─────────┘
         ┌──────┼──────┐
         p      f      b      c
         │      │      │      │
         h      o      o      o
         │      │      │      │
         o      n      n      n
         │      │      │      │
         n      e      e      e
         │      ✓      ✓      ✓
         e
         ✓

BFS Exploration:

Queue: [(root, initial_product_state)]

1. Pop (root, ({0}, 0))
   Children: p, f, b, c

   For 'p': transition product automaton
     Match: ({1}, 0)  → push (trie['p'], ({1}, 0))
     Insert: ({0}, 1) → push (trie['p'], ({0}, 1))

   For 'f': transition product automaton
     Match: ({6}, 0)  → push (trie['f'], ({6}, 0))
     ...

2. Pop (trie['p'], ({1}, 0))
   Children: h

   For 'h':
     Match: ({2}, 0)  → push (trie['p']['h'], ({2}, 0))

... continue BFS ...

At each accepting dictionary node, check if product automaton can accept.
```

### 5.8 Pros and Cons

**Advantages**:
- **No pre-processing**: Raw dictionary, any rules at runtime
- **Full cost decomposition**: Know phonetic cost vs edit distance
- **Maximum flexibility**: Change rules without rebuilding

**Disadvantages**:
- **Slower queries**: Must do composition work at runtime
- **Complex implementation**: Product automaton logic is intricate
- **Larger search space**: Explores more states than other approaches

### 5.9 Complete Pseudocode

```python
class ProductAutomaton:
    def __init__(self, phonetic_nfa, max_distance, algorithm="standard"):
        self.nfa = phonetic_nfa
        self.max_distance = max_distance
        self.algorithm = algorithm  # standard, transposition, merge_and_split

    def initial_state(self):
        """Start state: NFA at start with 0 errors"""
        return ProductState(
            nfa_states=self.nfa.epsilon_closure({self.nfa.start}),
            distance=0
        )

    def is_accepting(self, state):
        """Can we accept from this state?"""
        if state.distance > self.max_distance:
            return False
        return any(self.nfa.is_final(s) for s in state.nfa_states)

    def transition(self, state, c):
        """Generate successor states for dictionary character c"""
        successors = []
        S = state.nfa_states
        d = state.distance

        # 1. Match
        match_states = self.nfa.step(S, c)
        if match_states:
            successors.append(ProductState(match_states, d))

        if d < self.max_distance:
            # 2. Substitution
            subst_states = self.nfa.advance(S)
            if subst_states:
                successors.append(ProductState(subst_states, d + 1))

            # 3. Insertion (extra dict char)
            successors.append(ProductState(S, d + 1))

            # 4. Deletion (missing dict char) - handled in accepts()

            # 5. Transposition (if enabled)
            # 6. Merge (if enabled)
            # 7. Split (if enabled)
            # ... (see full implementation)

        return successors

    def accepts(self, input_string):
        """Does this string match the pattern within distance?"""
        visited = set()
        queue = [(0, self.initial_state())]  # (position, state)

        while queue:
            pos, state = queue.pop(0)

            # Dedup
            key = (pos, tuple(sorted(state.nfa_states)), state.distance)
            if key in visited:
                continue
            visited.add(key)

            # Pruning
            if state.distance > self.max_distance:
                continue

            # End of input
            if pos == len(input_string):
                if self.can_reach_final(state):
                    return True
                continue

            # Process next character
            c = input_string[pos]
            for next_state in self.transition(state, c):
                queue.append((pos + 1, next_state))

            # Deletion: advance NFA without consuming input
            if state.distance < self.max_distance:
                del_states = self.nfa.advance(state.nfa_states)
                if del_states:
                    next_state = ProductState(del_states, state.distance + 1)
                    queue.append((pos, next_state))  # Same position!

        return False


class PhoneticTransducer:
    def __init__(self, dictionary, phonetic_nfa, max_distance):
        self.dictionary = dictionary
        self.product = ProductAutomaton(phonetic_nfa, max_distance)

    def query(self, input_string):
        """Find all dictionary words matching the pattern within distance"""
        results = []

        # BFS through dictionary × product automaton
        queue = [(self.dictionary.root(), "", self.product.initial_state())]

        while queue:
            node, path, state = queue.pop(0)

            # Check if this dictionary word matches
            if node.is_final():
                if self.product.is_accepting(state):
                    results.append(Candidate(
                        term=path,
                        edit_distance=state.distance,
                        phonetic_cost=0.0  # Could track separately
                    ))

            # Explore children
            for c, child_node in node.edges():
                for next_state in self.product.transition(state, c):
                    queue.append((child_node, path + c, next_state))

        return results
```

---

## Chapter 6: Comparative Analysis

### 6.1 When to Use Each Approach

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| Autocomplete with millions of queries/sec | 1 | Speed is critical |
| Spell checker showing "did you mean?" | 3 | Need accurate cost decomposition |
| Search engine with static index | 2 | Pre-compute variants, fast queries |
| Interactive editor with custom rules | 3 | Rules change per user |
| Mobile app with memory constraints | 1 or 3 | Small footprint |
| Batch processing OCR corrections | 2 or 3 | Accuracy over speed |

### 6.2 Complexity Comparison

| Metric | Approach 1 | Approach 2 | Approach 3 |
|--------|-----------|-----------|-----------|
| Build Time | O(D·L·R) | O(D·V·L) | O(D·L) |
| Memory | O(D·L) | O(D·V·L) | O(D·L) |
| Query Time | O(m·R + m·n·L) | O(V_q·m·n·L) | O(\|NFA\|·n·m·L) |
| Flexibility | Low | Medium | High |
| Cost Decomposition | No | Yes | Yes |

Where: D=dictionary size, L=avg word length, R=rules, V=variants/word, m=query length, n=max distance, |NFA|=NFA states

### 6.3 Memory vs Speed Trade-off

```
                         Speed
                           ▲
                           │
     Approach 1 ●──────────┼────────────────────┐
                           │                    │
                           │         Approach 2 │
                           │              ●─────┘
                           │             /
                           │            /
                           │           /
                 ──────────┼──────────/─────────▶ Memory
                           │         /
                           │        /
                           │       /
                           │      ●
                           │ Approach 3
                           │
```

### 6.4 Precision vs Recall Trade-off

**Approach 1** can conflate too many words:
- "knight" and "night" both normalize to "nit"
- Might return wrong suggestions if user meant one specifically

**Approach 2** preserves distinctions:
- Each variant tracked separately
- Can weight phonetic changes

**Approach 3** is most precise:
- Every path through product automaton tracked
- Full cost breakdown available

---

## Chapter 7: Implementation in liblevenshtein-rust

### 7.1 Key Files and Structures

```
liblevenshtein-rust/
├── src/
│   ├── phonetic/
│   │   ├── nfa/
│   │   │   ├── product.rs      ← ProductAutomatonChar (Approach 3)
│   │   │   └── thompson.rs     ← NFA construction
│   │   ├── application.rs      ← apply_rules_seq() (for Approach 1)
│   │   └── rules.rs            ← Zompist phonetic rules
│   └── transducer/
│       ├── phonetic_transducer.rs ← High-level API
│       └── intersection.rs     ← Dict × Automaton composition
```

### 7.2 Using the Current Implementation (Approach 3)

```rust
use liblevenshtein::phonetic::nfa::{compile, NFAChar};
use liblevenshtein::phonetic::regex::parse;
use liblevenshtein::transducer::PhoneticTransducerChar;
use liblevenshtein::dictionary::DoubleArrayTrieChar;

// Build dictionary
let dict = DoubleArrayTrieChar::from_terms(["phone", "fone", "bone", "cone"]);

// Build phonetic NFA for pattern "(ph|f)one"
let pattern = compile(&parse("(ph|f)one").unwrap()).unwrap();

// Create transducer (product automaton internally)
let transducer = PhoneticTransducerChar::new(dict, pattern, 1);

// Query
for candidate in transducer.query("fone") {
    println!("{}: distance {}", candidate.term, candidate.edit_distance);
}
// Output:
// fone: distance 0
// phone: distance 0 (via ph→f in NFA)
// bone: distance 1 (via b→f substitution)
```

### 7.3 What's Needed for Other Approaches

**For Approach 1** (~600 lines of code):
```rust
// Not yet implemented, but would look like:
let normalized_dict = PhoneticNormalizedDictionary::new(
    terms,
    &zompist_rules()
);

// Query normalizes input then does Levenshtein
for candidate in normalized_dict.query("fone", 1) {
    println!("{}", candidate.term);
}
```

**For Approach 2** (~1500 lines of code):
```rust
// Not yet implemented, but would look like:
let expanded_dict = PhoneticExpandedDictionary::new(
    terms,
    &zompist_rules(),
    100  // max variants per term
);

// Query searches all variants
for candidate in expanded_dict.query("fone", 1) {
    println!("{}: phonetic={}, edit={}",
        candidate.term,
        candidate.phonetic_cost,
        candidate.edit_distance);
}
```

---

## Chapter 8: Exercises for the Reader

### Exercise 1: Trace a Query
Given the NFA for pattern `(c|k)at` and dictionary `["cat", "kat", "bat", "hat"]`:
1. Draw the NFA states and transitions
2. Trace the product automaton for query "kat" with max_distance=0
3. Trace for query "bat" with max_distance=1

### Exercise 2: Enumerate Variants
Given rules:
- `ph → f` (cost `0.1`)
- `c → k` (cost `0.1`)
- `tion → shun` (cost `0.2`)

Enumerate all variants of "action" with their costs.

### Exercise 3: Compare Approaches
For a dictionary of 10,000 English words:
1. Estimate memory usage for Approach 1 vs Approach 2
2. Estimate query latency for each approach
3. Which would you choose for a mobile keyboard autocomplete?

### Exercise 4: Extend the Product Automaton
The product automaton in liblevenshtein supports transposition. Extend the pseudocode to handle:
1. Transposition of adjacent characters
2. Merge (two input chars → one pattern transition)
3. Split (one input char → two pattern transitions)

---

## Conclusion

Compositional spelling correction combines the power of phonetic matching with edit distance tolerance. The three approaches offer different trade-offs:

1. **Pre-process Both**: Maximum speed, minimum precision
2. **Pre-process Dictionary**: Good balance, high memory
3. **Product Automaton**: Maximum precision, runtime flexibility

The liblevenshtein-rust library implements Approach 3 via `ProductAutomatonChar`, providing full cost decomposition and runtime rule flexibility. The other approaches can be built on top of existing primitives (`apply_rules_seq()` for normalization, dictionary types for storage).

Understanding these trade-offs helps you choose the right approach for your application's constraints on speed, memory, precision, and flexibility.

---

## References

1. Schulz, Klaus U. & Mihov, Stoyan (2002). "Fast String Correction with Levenshtein Automata". *International Journal on Document Analysis and Recognition (IJDAR)* 5(1), pp. 67–85. DOI: [10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
2. Mihov, Stoyan & Schulz, Klaus U. (2004). "Fast Approximate Search in Large Dictionaries". *Computational Linguistics* 30(4), pp. 451–477. DOI: [10.1162/0891201042544938](https://doi.org/10.1162/0891201042544938)
3. Thompson, Ken (1968). "Programming Techniques: Regular Expression Search Algorithm". *Communications of the ACM* 11(6), pp. 419–422. DOI: [10.1145/363347.363387](https://doi.org/10.1145/363347.363387)
4. Rosenfelder, Mark (Zompist). "Sound Change / English Phonetic Rules". https://www.zompist.com/spell.html (No DOI.)

---

[← Documentation Index](../README.md)

---

## Appendix A: Glossary

**Automaton**: A state machine that processes input and decides accept/reject.

**DFA**: Deterministic Finite Automaton — exactly one transition per state per input.

**Edit Distance**: Minimum character edits (insert/delete/substitute) between strings.

**Epsilon Closure**: All states reachable via epsilon (free) transitions.

**Levenshtein Automaton**: Accepts all strings within edit distance `n` of a query.

**NFA**: Non-deterministic Finite Automaton — can have multiple transitions per input.

**Phonetic Rules**: Transformations based on pronunciation (ph→f).

**Product Automaton**: Composition of two automata; states are pairs from each.

**Thompson Construction**: Algorithm to convert regex to NFA.

**Trie**: Tree data structure for efficient prefix-based string storage.

---

## Appendix B: Full Product Automaton State Diagram

For pattern `(ph|f)one` and max_distance=1:

```
                                         Accepting
                                            │
                                            ▼
({0}, 0) ─p→ ({1}, 0) ─h→ ({2}, 0) ─o→ ({3}, 0) ─n→ ({4}, 0) ─e→ ({5}, 0) ✓
    │            │           │           │           │           │
    │ insert     │ insert    │ insert    │ insert    │ insert    │ insert
    ▼            ▼           ▼           ▼           ▼           ▼
({0}, 1)    ({1}, 1)    ({2}, 1)    ({3}, 1)    ({4}, 1)    ({5}, 1) ✓
    │            │           │           │           │
    │ subst      │ subst     │ subst     │ subst     │ subst
    ▼            ▼           ▼           ▼           ▼
({1|6}, 1) ({2}, 1)   ({3}, 1)   ({4}, 1)   ({5}, 1) ✓

({0}, 0) ─f→ ({6}, 0) ─ε→ ({2}, 0) ─o→ ... (same as above)
    │
    │ insert
    ▼
({0}, 1) ─f→ ({6}, 1) ...

Legend:
  ─x→  : Match transition on character x
  subst: Substitution (advance NFA, +1 distance)
  insert: Insertion (stay in NFA, +1 distance)
  ✓    : Accepting state
```
