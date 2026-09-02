# Crate boundary and pruning duality

## The placement rule

A **cost measure** assigns smaller values to better paths. A **gain measure**
assigns larger values to better paths. liblevenshtein owns a measure when its
dictionary pruning follows from the ordered cost-monoid laws, especially
monotonicity and non-negative extension. A measure crosses into a weighted
finite-state transducer (WFST) when it needs a caller-selectable alternative
operator or gain-valued steps and therefore needs an explicit admissible bound.

![Cross-crate placement and control flow: libdictenstein supplies dictionary edges, liblevenshtein owns structural DFS, duallity owns fzf dynamic-programming state and the corrected upper bound, lling-llang owns the Arctic semiring and WFST traits, and both score paths feed top-k results.](../diagrams/traversal/crate-boundary-prune-duality.svg)

This is a semantic boundary, not a preference about module names.

## Why min-plus distance prunes locally

`CostMonoid` fixes alternative selection to minimum. Let $`c`$ be the
cost accumulated at a dictionary prefix and $`w`$ a lawful non-negative
extension. The positive-order and monotonicity laws give

```math
c \le c \otimes w.
```

Every descendant costs at least its prefix lower bound. If a query budget is
$`\tau`$, rejecting a subtree when $`c>\tau`$ is sound without a
measure-specific heuristic. Unit edit distance, scaled affine-gap distance,
and the elastic lower-bound walkers all use this direction.

The `CostMonoid` interface deliberately does not expose a configurable
$`\oplus`$, Kleene star, or division. Those operations are not needed for
bounded distance dynamic programming and would blur the proof boundary.

## Why sign-flipping fzf does not repair the law

fzf's `FuzzyMatchV2` gives rewards for character matches, word boundaries,
camel-case transitions, and consecutive runs; gaps are penalties. Parallel
alignments choose their maximum score. The natural algebra is

```math
(\mathbb{R}\cup\{-\infty\},\ \max,\ +,\ -\infty,\ 0).
```

Negating a score changes max-plus to min-plus, but it also turns rewards into
negative costs. A negative extension can improve a path, so the inflation law
$`c\le c\otimes w`$ fails. Changing the sign changes notation, not the
proof obligation.

The standard WFST framework separates a graph from its weight algebra
[Mohri 2009](https://doi.org/10.1007/978-3-642-01492-5_6). lling-llang therefore
owns `ArcticWeight`; duallity owns the concrete scorer and WFST adapter; and
liblevenshtein owns only the generic balanced structural DFS.

## The local-alignment correction

Claude's original plan proposed an fzf prefix bound of the form
$`S+(m-j)\beta`$. That formula describes extensions of one active
alignment. fzf is a local matcher, so a descendant may skip the complete
current prefix and start a different alignment later. An active-only formula
can therefore be smaller than a descendant's exact score.

Let $`U_0`$ be the best score of an alignment that has not started, and
let $`U_i`$ bound extensions of the live cell that has matched through
query index $`i`$. A sound bound is

```math
U(p)=\max\!\left(U_0,\ \max_i U_i\right).
```

Every descendant alignment is in exactly one of those families. If the current
top-$`k`$ cutoff is $`\tau_k`$, pruning requires
$`U(p)<\tau_k`$. The strict comparison preserves candidates tied at the
cutoff for the caller's deterministic tie policy.

The unstarted term exposes a real trade-off: without subtree metadata it is the
global maximum and makes most score-based prefix pruning impossible. The
implementation still shares DP columns across common prefixes. A future
reachable-character or maximum-depth summary is allowed only after its
pre-registered benchmark gate passes.

## Why DFS is part of the API contract

The prefix visitor is stack-shaped. `enter(character, depth)` creates state for
one root-to-node prefix, and `leave(character, depth)` restores its parent. A
breadth-first search (BFS) does not have one current root-to-node stack: it must
clone state per queued frontier item or store an immutable state handle in each
item. A depth-first search (DFS) can keep one mutable stack and balance it while
backtracking.

Consequently, the existing breadth-first query iterator remains unchanged.
`SubsequenceQueryIterator` is a separate explicit-stack DFS surface. Treating
BFS and DFS as interchangeable would break both result order and visitor-state
ownership even if their final result sets happened to match.

## End-to-end literate algorithm

```text
procedure PREFIX-SHARED-SCORED-DFS(root, query, visitor)
    stack  := [frame(root)]
    prefix := empty

    while stack is not empty:
        frame := stack.last

        if frame is an unchecked accepting node:
            mark frame checked
            if query is structurally matched and visitor permits prefix:
                emit prefix with visitor's exact score

        else if frame has another edge (character, child):
            if visitor.enter(character, |prefix| + 1):
                append character to prefix
                push frame(child)
            else:
                visitor.leave(character, |prefix| + 1)

        else:
            pop frame
            if frame was entered by character:
                remove the character from prefix
                visitor.leave(character, |prefix| + 1)
```

The two `leave` sites cover rejected subtrees and ordinary backtracking. Early
drop unwinds all remaining frames. Unit tests, property tests, Rocq, Dafny,
Verus, SMT, and TLA+ all encode the balance or no-false-negative invariants.

## Decision table for future measures

| Question | If yes | Placement |
|---|---|---|
| Is the result a bounded distance with fixed minimum selection? | continue | potentially liblevenshtein |
| Are extension steps non-negative and monotone? | continue | prefix lower bounds may be structural |
| Does the measure need max, log-sum, probability sum, star, or division? | stop | lling-llang algebra plus a duallity adapter |
| Does pruning require a measure-specific admissible heuristic? | stop | duallity owns the heuristic and its proof |
| Can traversal remain independent of scores? | yes | keep the structural visitor in liblevenshtein |

## Evidence map

- `src/transducer/subsequence_query.rs`: balanced explicit-stack DFS;
- `duallity/src/fzf_support.rs`: shared exact recurrence and corrected bound;
- `duallity/src/fzf_scorer.rs`: top-$`k`$ visitor;
- `duallity/src/fzf_state_source.rs`: path-sensitive state and telescoping arcs;
- `lling-llang/src/semiring/basic/arctic.rs`: max-plus weight algebra;
- `docs/verification/core/theories/Conformance/FzfUpperBound.v`: core theorem;
- `docs/verification/tla/FzfTrieSearch.tla`: exhaustive finite traversal model.

The score constants and recurrence are differential-tested against fzf's
[implementation](https://github.com/junegunn/fzf/blob/master/src/algo/algo.go)
and [fixture corpus](https://github.com/junegunn/fzf/blob/master/src/algo/algo_test.go).
