# Formal Verification of Grammar Correction Pipeline

## Overview

This directory contains a **Coq formalization** of the 5-layer grammar correction pipeline designed for the Rholang programming language. The verification establishes correctness, soundness, completeness, and optimality properties for the error correction system.

**Status**: Active grammar proof suite with checked core, layer, composition, and NFA modules. The current `.v` sources have no active `Admitted`, `Axiom`, `Parameter`, `Conjecture`, or `Hypothesis` proof escapes.

**Related Design Document**: [`docs/design/grammar-correction/MAIN_DESIGN.md`](../../design/grammar-correction/MAIN_DESIGN.md)

## Architecture

The verification is organized into three main categories:

### Core Modules (`theories/Core/`)

**1. `Types.v`**
- Foundational type definitions
- Programs, characters, positions, spans
- Scores and probabilities (using `Q` rationals)
- Edit operations (insertion, deletion, substitution, transposition)
- Parse trees and node kinds
- Type system (Rholang types)
- Lattice structures (nodes, edges, paths)
- Correction candidates
- Well-formedness conditions

**2. `Edit.v`**
- Edit operation application
- Levenshtein distance computation
- Edit distance properties:
  - Symmetry: `levenshtein s1 s2 = levenshtein s2 s1`
  - Triangle inequality: `levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3`
  - Zero iff equal: `levenshtein s1 s2 = 0 <-> s1 = s2`
- Optimal edit sequences
- Edit sequence composition
- Weighted edit distance (keyboard, phonetic)

**3. `Lattice.v`**
- Lattice path definitions
- Path validation and completeness
- Path score computation
- Best path (Viterbi algorithm)
- Top-k paths enumeration
- Lattice expansion with error edges
- Lattice composition
- Lattice pruning
- Beam search
- Lattice minimization

**4. `Program.v`**
- Syntactic and semantic validity
- Correction goals and quality metrics
- Correction ordering and optimality
- Correction soundness: `apply_edits original edits = corrected`
- Program equivalence
- Layer results and composition
- Pipeline execution
- Main correctness theorem

### Layer Modules (`theories/Layers/`)

**Layer 1: `Layer1.v`** (Levenshtein Lattice)
- **Configuration**: max edit distance, transposition, phonetic, keyboard
- **Properties**:
  - **Completeness**: Every string within max edit distance is reachable
  - **Soundness**: Every reachable string is within max edit distance
  - **Optimality**: Best path has minimal edit distance
- **Theorems**:
  - `layer1_produces_wf_lattice`: Lattice is well-formed
  - `layer1_completeness`: All strings within distance are included
  - `layer1_soundness`: All paths respect distance bound
  - `layer1_optimality`: Optimal paths exist
- **Performance**: Candidate count bounded by $`\mathcal{O}(n^d \times  \sigma ^d)`$
- **Features**: Transposition support, phonetic similarity, keyboard distance

**Layer 2: `Layer2.v`** (Tree-sitter Parsing)
- **Configuration**: accept partial parses, error node penalty
- **Abstract parsing function** (implemented via Tree-sitter FFI)
- **Properties**:
  - **Soundness**: All accepted programs parse without errors (or with accepted errors)
  - **Progress**: Attempts to parse all candidates
- **Theorems**:
  - `layer2_soundness`: Parse trees are valid
  - `parse_program_deterministic`: Parsing is deterministic

**Layer 3: `Layer3.v`** (Type Checking)
- **Configuration**: strict mode
- **Type checking function**: simplified executable type result over parse trees
- **Properties**: type annotation preserves correction soundness and layer-result validity

**Layer 4: `Layer4.v`** (Semantic Repair)
- **Configuration**: max repairs
- **Proof role**: validity-preserving semantic boundary

**Layer 5: `Layer5.v`** (Process Calculus)
- **Configuration**: deadlock and race-condition checks
- **Proof role**: validity-preserving process-calculus boundary

### Composition Modules (`theories/Composition/`)

**1. `Forward.v`** (Sequential Composition)
- Forward composition of layers
- **Theorem**: `forward_composition_valid` - composition preserves validity

**2. `Backward.v`** (Feedback)
- Backward feedback for layer rescoring
- Improves scores based on later layer results

**3. `Pipeline.v`** (Pipeline Execution)
- Pipeline execution semantics
- **Theorem**: `pipeline_always_produces_result` - termination guarantee

**4. `Correctness.v`** (End-to-End)
- **Main Theorem**: `grammar_correction_correctness`
  - If pipeline produces a correction, it is sound and complete
- **Soundness**: `all_corrections_sound` - all corrections are valid transformations
- **Termination**: `pipeline_terminates_always` - pipeline always terminates
- **Optimality**: `best_correction_optimal` - best correction minimizes edit distance
- **Progress**: `pipeline_makes_progress` - pipeline makes forward progress

## Key Theorems

### Levenshtein Distance (Edit.v)

```coq
Theorem levenshtein_symmetric : forall s1 s2,
  levenshtein s1 s2 = levenshtein s2 s1.

Theorem levenshtein_triangle : forall s1 s2 s3,
  levenshtein s1 s3 <= levenshtein s1 s2 + levenshtein s2 s3.

Theorem levenshtein_zero_iff_eq : forall s1 s2,
  levenshtein s1 s2 = 0 <-> s1 = s2.

Theorem optimal_edit_exists : forall s1 s2,
  exists edits, optimal_edit_sequence s1 s2 edits.
```

### Lattice Properties (Lattice.v)

```coq
Theorem lattice_has_path : forall lat,
  wf_lattice lat ->
  exists path, complete_path lat path = true.

Theorem best_path_achievable : forall lat,
  wf_lattice lat ->
  exists path,
    complete_path lat path = true /\
    path_score lat path == best_path_score lat.

Theorem top_k_paths_sorted : forall lat k,
  let paths := top_k_paths lat k in
  forall i j,
    i < j < length paths ->
    path_score lat (nth i paths []) >= path_score lat (nth j paths []).
```

### Layer 1 Correctness (Layer1.v)

```coq
Theorem layer1_completeness : forall config input output,
  levenshtein input output <= config.(max_edit_distance) ->
  exists path,
    let lat := build_error_lattice config input in
    complete_path lat path = true.

Theorem layer1_soundness : forall config input path,
  let lat := build_error_lattice config input in
  complete_path lat path = true ->
  exists output edits,
    apply_edits input edits = output /\
    edit_distance edits <= config.(max_edit_distance).

Theorem layer1_optimality : forall config input output,
  levenshtein input output <= config.(max_edit_distance) ->
  let lat := build_error_lattice config input in
  exists path edits,
    complete_path lat path = true /\
    apply_edits input edits = output /\
    edit_distance edits = levenshtein input output.
```

### Pipeline Correctness (Program.v, Correctness.v)

```coq
Theorem pipeline_execution_valid : forall p pipe,
  (forall layer, In layer pipe -> valid_layer_result p (layer p)) ->
  valid_layer_result p (execute_pipeline p pipe).

Theorem pipeline_always_terminates : forall p pipe,
  pipeline_terminates p pipe.

Theorem correction_correctness : forall p pipe goal,
  let result := execute_pipeline p pipe in
  match result.(layer_best_correction) with
  | Some corr =>
      correction_sound p corr /\
      correction_complete goal p corr
  | None => True
  end.

Theorem grammar_correction_correctness :
  forall input config1 config2 goal,
    let layer1 := execute_layer1 config1 in
    let layer2 := fun p r => execute_layer2 config2 p r in
    let pipe := [layer1; fun p => layer2 p (layer1 p)] in
    let result := execute_pipeline input pipe in
    match result.(layer_best_correction) with
    | Some corr =>
        correction_sound input corr /\
        correction_complete goal input corr
    | None => True
    end.
```

## Compilation

### Prerequisites

- Coq 8.19+ (or Rocq)
- Standard library with `QArith` (rational numbers)

### Building

```bash
cd docs/verification/grammar
coq_makefile -f _CoqProject -o Makefile
make
```

This compiles all configured `.v` files and produces `.vo` object files. For agent or CI runs on memory-constrained machines, prefer a capped invocation:

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  make -C docs/verification/grammar -j1
```

### Compilation Status

**Current Status**: checked proof suite. Recent capped verification compiled the active grammar core, layer, composition, and NFA slices; targeted escape-hatch scans over `docs/verification/grammar/theories/**/*.v` found no active `Admitted`, `Axiom`, `Parameter`, `Conjecture`, or `Hypothesis`.

The current codebase provides:
- Complete type definitions
- Checked edit-distance, lattice, layer, pipeline, and NFA proof families
- Bounded executable best-path scoring for lattices
- Executable Layer 1 identity candidate construction
- Validity-preserving Layer 3/4/5 boundaries
- NFA path extraction and bounded-cost soundness evidence

## Proof Strategy

### Edit Distance (Edit.v)

**Levenshtein Symmetry**:
- Proof by strong induction on `length s1 + length s2`
- Show that swapping arguments preserves recurrence relation

**Triangle Inequality**:
- Construct composed edit sequence from `s1 -> s2 -> s3`
- Show composition has distance <= sum of individual distances

**Zero Iff Equal**:
- Forward: Induction on strings, showing distance 0 implies character-by-character equality
- Backward: Induction showing identical strings have distance 0

### Lattice Completeness (Lattice.v)

**Path Existence**:
- Construct path by breadth-first search
- Show BFS always finds start-to-end path in connected lattice

**Best Path**:
- Implement Viterbi algorithm constructively
- Prove dynamic programming optimality

### Layer 1 (Layer1.v)

**Completeness**:
- Given string `output` within edit distance `d`
- Construct path in lattice by following optimal edit sequence
- Show each edit corresponds to a lattice edge

**Soundness**:
- Given complete path in lattice
- Extract edit sequence from path
- Show edit distance is bounded by configuration

**Optimality**:
- Use optimal edit sequence existence
- Show lattice contains corresponding path
- Prove path score reflects edit distance

### Pipeline Composition (Program.v)

**Validity Preservation**:
- By induction on pipeline length
- Show each layer preserves soundness
- Use `Forall` to propagate property

**Termination**:
- Trivial: Coq functions are total
- Pipeline is finite recursive structure

## Correspondence with Implementation

The Rocq verification models the architectural contracts used by the Rust implementation and by the WFST/lattice design work:

| Coq Module | Rust Module |
|------------|-------------|
| `Core/Types.v` | correction/lattice data model used by the design docs |
| `Core/Edit.v` | edit operations and distance contracts |
| `Core/Lattice.v` | lattice-path scoring and expansion contracts |
| `Layers/Layer1.v` | Levenshtein lattice layer contract |
| `Layers/Layer2.v` | parser-validity layer contract |
| `Layers/Layer3.v` | type-annotation preservation contract |
| `Composition/Pipeline.v` | pipeline execution and result contract |
| `NFA/*.v` | phonetic/grammar NFA soundness and path-extraction contracts |

The verification focuses on **algorithmic correctness**, not Rust-specific concerns (memory safety, concurrency). Those are handled by Rust's type system.

## Statistics

- **Total `.v` Files**: 26
- **Core Modules**: `Core/*.v`
- **Layer Modules**: `Layers/*.v`
- **Composition Modules**: `Composition/*.v`
- **NFA Modules**: `NFA/*.v`
- **Active Proof Escapes**: 0 found by the current grammar-theories scan

## Maintenance Gates

Before changing this proof suite:

1. Compile the touched dependency slice under `systemd-run --user --scope` with `MemoryMax` and `MemorySwapMax=0`.
2. Run `rg -n "\\bAdmitted\\b|\\badmit\\b|\\bAxiom\\b|\\bParameter\\b|\\bConjecture\\b|\\bHypothesis\\b" docs/verification/grammar/theories -g '*.v'`.
3. Run the stale-marker scan for implementation language in touched docs and proof files.
4. Remove generated `.vo`, `.glob`, `.vok`, `.vos`, `.aux`, `.lia.cache`, and makefile dependency artifacts unless they are intentionally tracked.

## Related Documents

- **Design**: [`docs/design/grammar-correction/MAIN_DESIGN.md`](../../design/grammar-correction/MAIN_DESIGN.md)
- **Design README**: [`docs/design/grammar-correction/README.md`](../../design/grammar-correction/README.md)
- **Phonetic Verification**: [`docs/verification/phonetic/`](../phonetic/README.md)

## References

1. Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals."
2. Schulz, K. U., & Mihov, S. (2002). "Fast string correction with Levenshtein automata."
3. Wagner, R. A., & Fischer, M. J. (1974). "The string-to-string correction problem."
4. Mohri, M. (1997). "Finite-state transducers in language and speech processing."
