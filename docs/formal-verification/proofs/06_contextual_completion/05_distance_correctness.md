# Levenshtein Distance Correctness - Formal Proof Documentation

**Status**: Proof sketch documented; no checked contextual Rocq module exists yet
**Rocq target module**: `rocq/liblevenshtein/ContextualCompletion/Distance.v`
**Date**: 2025-01-21
**Authors**: Formal Verification Team

---

## Overview

This theorem establishes that the naive $`\mathcal{O}(n\cdot m)`$ dynamic programming implementation of Levenshtein distance correctly computes the minimum edit distance between two strings, matching the formal Wagner-Fischer recurrence relation.

### Why This Matters

**User Impact**: This is the **fuzzy matching correctness theorem**. Without it:
- **Wrong rankings**: Completions ordered by incorrect distances
- **Missing suggestions**: Terms incorrectly filtered as "too far"
- **False matches**: Terms incorrectly accepted as "close enough"
- **Broken UX**: Autocomplete feels broken or unpredictable

**Dependencies**: Used by:
- **Theorem 4**: Query fusion relies on accurate distance filtering
- **Completion ranking**: Results sorted by distance
- **Fuzzy search**: `max_distance` threshold must be accurate

### Real-World Example

```rust
let engine = DynamicContextualCompletionEngine::new();
let global = engine.create_root_context(0);

// Finalize symbols
engine.finalize_direct(global, "kitten");
engine.finalize_direct(global, "sitting");
engine.finalize_direct(global, "written");

// Query with distance threshold
let completions = engine.complete(global, "kittn", 2);

// Expected behavior (distance 2 threshold):
// - "kitten" (distance 1: delete 'e') ✓ INCLUDED
// - "sitting" (distance 3: sub k→s, sub i→i, sub t→t, ins t, ins i, del e, sub n→n, ins g) ✗ EXCLUDED
// - "written" (distance 2: sub k→w, sub i→r) ✓ INCLUDED

// If distance function is WRONG:
// - Could include "sitting" (user sees irrelevant suggestions)
// - Could exclude "written" (user misses relevant suggestions)
```

---

## Definitions

```coq
(* Character equality *)
Definition char_eq (c1 c2 : Char) : bool := (* equality test *).

(* Cost of substitution *)
Definition subst_cost (c1 c2 : Char) : nat :=
  if char_eq c1 c2 then 0 else 1.

(* Wagner-Fischer recurrence relation *)
Fixpoint lev_distance (s1 s2 : list Char) : nat :=
  match s1, s2 with
  | [], s2' => length s2'                    (* Base case: empty s1 → insert all of s2 *)
  | s1', [] => length s1'                    (* Base case: empty s2 → delete all of s1 *)
  | c1 :: rest1, c2 :: rest2 =>
      min3
        (lev_distance rest1 s2 + 1)          (* Delete c1 *)
        (lev_distance s1 rest2 + 1)          (* Insert c2 *)
        (lev_distance rest1 rest2 + subst_cost c1 c2)  (* Substitute/keep *)
  end.

(* Matrix-based dynamic programming *)
Definition dp_matrix (s1 s2 : list Char) : Matrix nat :=
  let m := length s1 in
  let n := length s2 in
  let matrix := init_matrix (m+1) (n+1) 0 in
  let matrix := init_row_0 matrix in     (* matrix[i][0] = i *)
  let matrix := init_col_0 matrix in     (* matrix[0][j] = j *)
  fill_matrix matrix s1 s2.

(* Matrix filling (iterative DP) *)
Fixpoint fill_matrix (matrix : Matrix nat) (s1 s2 : list Char) : Matrix nat :=
  for i from 1 to length s1 do
    for j from 1 to length s2 do
      let cost := subst_cost (nth s1 (i-1)) (nth s2 (j-1)) in
      matrix[i][j] := min3
        (matrix[i-1][j] + 1)       (* Deletion *)
        (matrix[i][j-1] + 1)       (* Insertion *)
        (matrix[i-1][j-1] + cost)  (* Substitution *)
    done
  done.
```

---

## Theorem Statement

```coq
Theorem levenshtein_distance_correctness :
  forall (s1 s2 : list Char),
    let naive_result := dp_matrix s1 s2 [length s1, length s2] in
    let recursive_result := lev_distance s1 s2 in
    naive_result = recursive_result /\  (* Implementation matches definition *)
    (* Triangle inequality *)
    (forall s3, lev_distance s1 s3 <= lev_distance s1 s2 + lev_distance s2 s3) /\
    (* Symmetry *)
    lev_distance s1 s2 = lev_distance s2 s1 /\
    (* Identity *)
    lev_distance s1 s1 = 0 /\
    (* Non-negativity *)
    lev_distance s1 s2 >= 0 /\
    (* Upper bound *)
    lev_distance s1 s2 <= max (length s1) (length s2).
```

---

## Proof Sketch

**Strategy**: Prove equivalence between recursive definition and DP matrix by structural induction on strings, then prove metric properties.

**Main Steps**:

### Part 1: DP Matrix Correctness (Equivalence to Recursion)

**Claim**: `dp_matrix s1 s2 [i, j] = lev_distance (take i s1) (take j s2)`

**Proof by Induction** on `i + j`:

**Base Cases**:
1. **i = 0, j = 0**: `dp_matrix[0,0] = 0 = lev_distance [] []` ✓
2. **i = 0, j > 0**: `dp_matrix[0,j] = j = lev_distance [] (take j s2)` (j insertions) ✓
3. **i > 0, j = 0**: `dp_matrix[i,0] = i = lev_distance (take i s1) []` (i deletions) ✓

**Inductive Step** (i > 0, j > 0):

**Assume** (induction hypothesis):
- `dp_matrix[i-1, j] = lev_distance (take (i-1) s1) (take j s2)`
- `dp_matrix[i, j-1] = lev_distance (take i s1) (take (j-1) s2)`
- `dp_matrix[i-1, j-1] = lev_distance (take (i-1) s1) (take (j-1) s2)`

**Show**: `dp_matrix[i, j] = lev_distance (take i s1) (take j s2)`

**By definition** of `dp_matrix`:
```
dp_matrix[i, j] = min3(
  dp_matrix[i-1, j] + 1,           (* Delete s1[i] *)
  dp_matrix[i, j-1] + 1,           (* Insert s2[j] *)
  dp_matrix[i-1, j-1] + cost       (* Substitute s1[i] ↔ s2[j] *)
)
```

**By IH** (substituting induction hypothesis):
```
= min3(
  lev_distance (take (i-1) s1) (take j s2) + 1,
  lev_distance (take i s1) (take (j-1) s2) + 1,
  lev_distance (take (i-1) s1) (take (j-1) s2) + cost
)
```

**By definition** of `lev_distance` (recursive case):
```
= lev_distance (take i s1) (take j s2)
```

**Therefore**: DP matrix computes exact same values as recursive definition. $`\blacksquare`$

### Part 2: Triangle Inequality

**Claim**: `lev_distance s1 s3 <= lev_distance s1 s2 + lev_distance s2 s3`

**Intuition**: Any edit sequence `s1 → s2 → s3` is a valid (possibly suboptimal) edit sequence `s1 → s3`.

**Proof**:
1. Let `e12` be an optimal edit sequence `s1 → s2` (length = `lev_distance s1 s2`)
2. Let `e23` be an optimal edit sequence `s2 → s3` (length = `lev_distance s2 s3`)
3. Concatenate: `e12 ++ e23` is a valid edit sequence `s1 → s3`
4. Length: `|e12 ++ e23| = |e12| + |e23| = lev_distance s1 s2 + lev_distance s2 s3`
5. But `lev_distance s1 s3` is the **minimum** length edit sequence
6. Therefore: `lev_distance s1 s3 <= |e12 ++ e23| = lev_distance s1 s2 + lev_distance s2 s3` $`\blacksquare`$

### Part 3: Symmetry

**Claim**: `lev_distance s1 s2 = lev_distance s2 s1`

**Proof**: Every edit operation has an inverse:
- `insert(c)` ↔ `delete(c)`
- `substitute(c1, c2)` ↔ `substitute(c2, c1)`

For any edit sequence `s1 → s2`, the reversed sequence of inverse operations is `s2 → s1` with the same length. $`\blacksquare`$

### Part 4: Identity

**Claim**: `lev_distance s s = 0`

**Proof by Induction** on `s`:
- **Base case**: `lev_distance [] [] = 0` (by definition)
- **Inductive step**:
  - Assume `lev_distance rest rest = 0`
  - Then `lev_distance (c::rest) (c::rest) = min3(lev_distance rest (c::rest) + 1, lev_distance (c::rest) rest + 1, lev_distance rest rest + 0) = min3(IH+1, IH+1, 0) = 0` $`\blacksquare`$

### Part 5: Non-Negativity

**Claim**: `lev_distance s1 s2 >= 0`

**Proof**: By definition, `lev_distance` returns a `nat` (natural number), which is always $`\ge 0`$. $`\blacksquare`$

### Part 6: Upper Bound

**Claim**: `lev_distance s1 s2 <= max (length s1) (length s2)`

**Proof**: Worst-case strategy - delete all of `s1`, then insert all of `s2`:
- Cost: `length s1 + length s2`
- But we can do better: delete `min(length s1, length s2)` and insert `|length s2 - length s1|`
- This gives: `lev_distance s1 s2 <= max(length s1, length s2)` $`\blacksquare`$

---

## Key Lemmas

**Lemma 1** (Matrix Initialization):
```coq
Lemma matrix_init_correct :
  forall (s1 s2 : list Char) (matrix : Matrix nat),
    matrix = init_row_0 (init_col_0 (init_matrix (length s1 + 1) (length s2 + 1) 0)) ->
    (forall i, i <= length s1 -> matrix[i][0] = i) /\
    (forall j, j <= length s2 -> matrix[0][j] = j).
```

**Lemma 2** (Cell Dependency):
```coq
Lemma cell_depends_on_predecessors :
  forall (matrix : Matrix nat) (i j : nat),
    i > 0 -> j > 0 ->
    matrix[i][j] depends only on {matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]}.
```

**Lemma 3** (Min3 Properties):
```coq
Lemma min3_lower_bound :
  forall a b c : nat,
    min3 a b c <= a /\ min3 a b c <= b /\ min3 a b c <= c.

Lemma min3_greatest_lower_bound :
  forall a b c x : nat,
    x <= a -> x <= b -> x <= c -> x <= min3 a b c.
```

**Lemma 4** (Optimal Substructure):
```coq
Lemma optimal_substructure :
  forall (s1 s2 : list Char) (c1 c2 : Char),
    lev_distance (s1 ++ [c1]) (s2 ++ [c2]) =
    min3
      (lev_distance (s1 ++ [c1]) s2 + 1)
      (lev_distance s1 (s2 ++ [c2]) + 1)
      (lev_distance s1 s2 + subst_cost c1 c2).
```

**Lemma 5** (UTF-8 Character Comparison):
```coq
(* Leverages Rust's char type guarantees *)
Axiom rust_char_eq_decidable :
  forall (c1 c2 : Char), {c1 = c2} + {c1 <> c2}.

Lemma char_eq_correct :
  forall (c1 c2 : Char),
    char_eq c1 c2 = true <-> c1 = c2.
```

---

## Implementation Correspondence

**Source**: `src/contextual/engine.rs:1215-1254`

```rust
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();  // O(n) - count Unicode scalar values
    let len2 = s2.chars().count();  // O(m)

    // Base cases (empty strings)
    if len1 == 0 { return len2; }  // lev_distance [] s2 = length s2
    if len2 == 0 { return len1; }  // lev_distance s1 [] = length s1

    // Initialize DP matrix: (len1+1) × (len2+1)
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize first column: matrix[i][0] = i (deletions)
    for (i, row) in matrix.iter_mut().enumerate() {
        row[0] = i;
    }
    // Initialize first row: matrix[0][j] = j (insertions)
    for (j, cell) in matrix[0].iter_mut().enumerate() {
        *cell = j;
    }

    // Collect characters once (Rust char = Unicode scalar value)
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    // Fill matrix using Wagner-Fischer recurrence
    for i in 1..=len1 {
        for j in 1..=len2 {
            // Substitution cost: 0 if characters match, 1 otherwise
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };

            // Compute minimum of three operations
            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1,      // Deletion
                    matrix[i][j - 1] + 1       // Insertion
                ),
                matrix[i - 1][j - 1] + cost    // Substitution (or keep if cost=0)
            );
        }
    }

    // Return bottom-right cell (final answer)
    matrix[len1][len2]
}
```

**Correspondence**:
- **Base cases** (lines 1219-1224) → `lev_distance [] s2` and `lev_distance s1 []`
- **Matrix initialization** (lines 1226-1233) → `init_row_0` and `init_col_0`
- **Character collection** (lines 1235-1236) → Rust's `char` type = Unicode scalar value
- **DP loop** (lines 1238-1251) → `fill_matrix` with Wagner-Fischer recurrence
- **Min computation** (lines 1246-1249) → `min3` function
- **Cost function** (lines 1240-1244) → `subst_cost c1 c2`
- **Result** (line 1253) → `dp_matrix[len1][len2]`

**Complexity**:
- **Time**: $`\mathcal{O}(n\cdot m)`$ where n = `|s1|`, m = `|s2|`
- **Space**: $`\mathcal{O}(n\cdot m)`$ for matrix (could be optimized to $`\mathcal{O}(\min (n, m)`$) using rolling arrays)

---

## Test Coverage

**Unit Tests**: `src/contextual/engine.rs:1849-1859` (#[cfg(test)])

```rust
#[test]
fn test_levenshtein_distance() {
    type Engine = DynamicContextualCompletionEngine<PathMapDictionary<Vec<ContextId>>>;

    // Base cases
    assert_eq!(Engine::levenshtein_distance("", ""), 0);        // Empty strings
    assert_eq!(Engine::levenshtein_distance("abc", ""), 3);     // Delete all
    assert_eq!(Engine::levenshtein_distance("", "abc"), 3);     // Insert all

    // Identity
    assert_eq!(Engine::levenshtein_distance("abc", "abc"), 0);  // No edits

    // Single substitution
    assert_eq!(Engine::levenshtein_distance("abc", "abd"), 1);  // Substitute c→d

    // Single insertion
    assert_eq!(Engine::levenshtein_distance("abc", "abcd"), 1); // Insert d

    // Classic example (substitutions, insertions, deletions)
    assert_eq!(Engine::levenshtein_distance("kitten", "sitting"), 3);
    // Edits: k→s, e→i, insert g
}
```

**Integration Tests** (rholang-language-server):

- `tests/test_completion.rs` - Fuzzy matching with distance thresholds
- `tests/test_pattern_matching_performance.rs` - Distance calculation performance

**Candidate Property-Based Tests**:

Using `proptest` crate:

```rust
proptest! {
    // Triangle inequality
    #[test]
    fn prop_triangle_inequality(s1: String, s2: String, s3: String) {
        let d12 = Engine::levenshtein_distance(&s1, &s2);
        let d23 = Engine::levenshtein_distance(&s2, &s3);
        let d13 = Engine::levenshtein_distance(&s1, &s3);
        prop_assert!(d13 <= d12 + d23);
    }

    // Symmetry
    #[test]
    fn prop_symmetry(s1: String, s2: String) {
        let d12 = Engine::levenshtein_distance(&s1, &s2);
        let d21 = Engine::levenshtein_distance(&s2, &s1);
        prop_assert_eq!(d12, d21);
    }

    // Identity
    #[test]
    fn prop_identity(s: String) {
        let d = Engine::levenshtein_distance(&s, &s);
        prop_assert_eq!(d, 0);
    }

    // Upper bound
    #[test]
    fn prop_upper_bound(s1: String, s2: String) {
        let d = Engine::levenshtein_distance(&s1, &s2);
        let max_len = s1.chars().count().max(s2.chars().count());
        prop_assert!(d <= max_len);
    }

    // Non-negativity (automatic with usize)
    #[test]
    fn prop_non_negative(s1: String, s2: String) {
        let d = Engine::levenshtein_distance(&s1, &s2);
        prop_assert!(d >= 0);  // Always true for usize
    }
}
```

---

## Related Theorems

**Depends on**:
- Rust's `char` type guarantees (Unicode scalar values)
- Rust's `==` operator for `char` (correct equality)

**Required by**:
- **Theorem 4** (Query Fusion) - Distance filtering in `complete()`
- **Completion ranking** - Sorting results by relevance
- **Fuzzy search** - Approximate string matching

**Independent of**:
- Context tree structure (Theorem 1)
- Draft buffer operations (Theorem 2)
- Checkpoint system (Theorem 3)

---

## Promotion Criteria and Optimization Notes

### Rocq Formalization

**File**: `rocq/liblevenshtein/ContextualCompletion/Distance.v`

**Dependencies**:
- Standard library: `Coq.Arith.Min`, `Coq.Lists.List`
- String representation (list of characters)
- Matrix type definition

**Proof Strategy**:
1. Define `lev_distance` as recursive function
2. Define `dp_matrix` as iterative computation
3. Prove equivalence by strong induction on string lengths
4. Prove metric properties (triangle inequality, symmetry, etc.)
5. Extract certified Rust code (optional)

### Optimization Opportunities (Beyond Correctness)

**Space Optimization** (Rolling Arrays):
```rust
// Current: O(n·m) space
let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

// Optimized: O(min(n, m)) space - only 2 rows needed
let mut prev_row = vec![0; len2 + 1];
let mut curr_row = vec![0; len2 + 1];
```

**Early Termination**:
```rust
// If we only care about "distance ≤ k", can terminate early
if matrix[i].iter().all(|&d| d > max_distance) {
    return max_distance + 1;  // Exceeds threshold
}
```

**Note**: These optimizations do NOT affect correctness, only performance. Theorem 5 proves the **algorithm** is correct, regardless of implementation details.

---

**Last Updated**: 2025-01-21
**Review trigger**: Reconcile this page when a checked contextual Rocq module is added.
