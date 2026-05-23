-------------------------------- MODULE Subsumption --------------------------------
(***************************************************************************)
(* TLA+ Specification for Subsumption Relation                              *)
(*                                                                          *)
(* This module specifies and verifies the subsumption relation used to      *)
(* prune redundant states in the Levenshtein automaton. The subsumption     *)
(* relation is a strict partial order that identifies positions that can    *)
(* be safely removed without affecting correctness.                         *)
(*                                                                          *)
(* Key Properties to Verify:                                                *)
(* 1. Irreflexivity: ~Subsumes(p, p) for all p                              *)
(* 2. Antisymmetry: ~(Subsumes(p, q) /\ Subsumes(q, p))                     *)
(* 3. Transitivity: Subsumes(p, q) /\ Subsumes(q, r) => Subsumes(p, r)     *)
(* 4. Correctness: If Subsumes(p, q), q's completions are covered by p     *)
(*                                                                          *)
(* Corresponds to: src/transducer/universal/subsumption.rs                  *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    MAX_POSITION,       \* Maximum word position (n)
    MAX_ERRORS,         \* Maximum error count
    ALGORITHMS          \* Set of algorithm types {Standard, Transposition, MergeSplit}

ASSUME MAX_POSITION >= 1
ASSUME MAX_ERRORS >= 0
ASSUME ALGORITHMS # {}

(***************************************************************************)
(* Type Definitions                                                         *)
(***************************************************************************)

\* Position type - represents a state in the automaton
\* For Standard: (i, e) where i is word position, e is errors
\* For Transposition: (i, e, special) where special indicates T-state
\* For MergeSplit: (i, e, offset) where offset is merge/split offset
Position == [
    i: 0..MAX_POSITION,           \* Word position
    e: 0..MAX_ERRORS,             \* Error count
    special: BOOLEAN,             \* Special position flag (T-state for transposition)
    offset: -1..1,                \* Merge/split offset (-1, 0, 1)
    alg: ALGORITHMS               \* Algorithm type
]

(***************************************************************************)
(* Subsumption Rules by Algorithm                                           *)
(***************************************************************************)

\* Standard Levenshtein subsumption:
\* (i1, e1) subsumes (i2, e2) iff i1 = i2 and e1 < e2
StandardSubsumes(p, q) ==
    /\ p.alg = "Standard"
    /\ q.alg = "Standard"
    /\ p.i = q.i
    /\ p.e < q.e

\* Transposition subsumption:
\* More complex due to T-states (special positions)
\* A T-state at (i, e) can reach (i+1, e) via transposition
\* Normal state (i, e1) subsumes normal state (i, e2) if e1 < e2
\* Normal state (i, e1) subsumes T-state (i, e2) if e1 < e2
\* T-state (i, e1) subsumes T-state (i, e2) if e1 < e2
\* T-state CANNOT subsume normal state (asymmetric)
TranspositionSubsumes(p, q) ==
    /\ p.alg = "Transposition"
    /\ q.alg = "Transposition"
    /\ p.i = q.i
    /\ p.e < q.e
    /\ (p.special = q.special \/ ~p.special)  \* Normal can subsume T, T can subsume T

\* MergeSplit subsumption:
\* Positions have an offset indicating merge/split state
\* (i, e1, o1) subsumes (i, e2, o2) if:
\*   - e1 < e2, or
\*   - e1 = e2 and |o1| < |o2| (closer to aligned position)
MergeSplitSubsumes(p, q) ==
    /\ p.alg = "MergeSplit"
    /\ q.alg = "MergeSplit"
    /\ p.i = q.i
    /\ \/ p.e < q.e
       \/ /\ p.e = q.e
          /\ (IF p.offset < 0 THEN -p.offset ELSE p.offset)
             < (IF q.offset < 0 THEN -q.offset ELSE q.offset)

\* Combined subsumption (dispatch by algorithm)
Subsumes(p, q) ==
    \/ StandardSubsumes(p, q)
    \/ TranspositionSubsumes(p, q)
    \/ MergeSplitSubsumes(p, q)

(***************************************************************************)
(* Partial Order Properties                                                 *)
(***************************************************************************)

\* All positions for model checking
AllPositions == [
    i: 0..MAX_POSITION,
    e: 0..MAX_ERRORS,
    special: BOOLEAN,
    offset: -1..1,
    alg: ALGORITHMS
]

\* Representative finite pruning frontier for the state-machine check.  The
\* universal order properties above still quantify over all model positions.
SamplePositions ==
    {p \in AllPositions :
        /\ p.i \in 0..1
        /\ p.offset = 0
        /\ p.special = FALSE}

\* PROPERTY 1: Irreflexivity
\* No position subsumes itself
Irreflexive ==
    \A p \in AllPositions : ~Subsumes(p, p)

\* PROPERTY 2: Asymmetry (stronger than antisymmetry for strict orders)
\* If p subsumes q, then q does not subsume p
Asymmetric ==
    \A p, q \in AllPositions :
        Subsumes(p, q) => ~Subsumes(q, p)

\* PROPERTY 3: Transitivity
\* If p subsumes q and q subsumes r, then p subsumes r
Transitive ==
    \A p, q, r \in AllPositions :
        (Subsumes(p, q) /\ Subsumes(q, r)) => Subsumes(p, r)

\* Combined: Strict Partial Order
StrictPartialOrder ==
    /\ Irreflexive
    /\ Asymmetric
    /\ Transitive

(***************************************************************************)
(* Correctness Properties                                                   *)
(***************************************************************************)

\* Reachable function: positions reachable from p with remaining budget b
\* (Abstract definition - actual implementation depends on transitions)
\* For model checking, we define a simplified version
CanComplete(p, remaining_length) ==
    \* p can reach a final position (i = word_length) with e <= MAX_ERRORS
    \* if remaining_length + p.i >= MAX_POSITION and p.e <= MAX_ERRORS
    /\ p.e <= MAX_ERRORS
    /\ p.i + remaining_length >= MAX_POSITION

\* Completion Preservation:
\* If p subsumes q, then any completion of q is also a completion of p
\* (with possibly better or equal cost)
CompletionPreservation(word_length) ==
    \A p, q \in AllPositions :
        \A remaining \in 0..word_length :
            (Subsumes(p, q) /\ CanComplete(q, remaining)) =>
            CanComplete(p, remaining)

\* Subsumption preserves antichain minimality
\* After subsumption, the remaining set is an antichain
FormAntichain(positions) ==
    \A p, q \in positions :
        p # q => ~Subsumes(p, q) /\ ~Subsumes(q, p)

(***************************************************************************)
(* State Machine for Testing Subsumption                                    *)
(***************************************************************************)

VARIABLES
    positions,          \* Current set of positions
    removed,            \* Set of removed (subsumed) positions
    iteration           \* Iteration counter

\* Type invariant
TypeInv ==
    /\ positions \subseteq AllPositions
    /\ removed \subseteq AllPositions
    /\ iteration \in Nat
    /\ StrictPartialOrder

\* Initial state with arbitrary set of positions
Init ==
    /\ positions = SamplePositions
    /\ removed = {}
    /\ iteration = 0

\* Remove a subsumed position
RemoveSubsumed ==
    /\ \E p, q \in positions :
        /\ p # q
        /\ Subsumes(p, q)
        /\ positions' = positions \ {q}
        /\ removed' = removed \cup {q}
        /\ iteration' = iteration + 1

\* No more positions can be removed
Done ==
    /\ \A p, q \in positions : p = q \/ (~Subsumes(p, q) /\ ~Subsumes(q, p))
    /\ UNCHANGED <<positions, removed, iteration>>

Next == RemoveSubsumed \/ Done

SubsumptionVars == <<positions, removed, iteration>>

Spec == Init /\ [][Next]_SubsumptionVars /\ WF_SubsumptionVars(RemoveSubsumed)

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

\* After termination, positions form an antichain
EventuallyAntichain ==
    <>FormAntichain(positions)

\* Removed positions were actually subsumed
RemovedWereSubsumed ==
    \A q \in removed :
        \E p \in AllPositions :
            /\ Subsumes(p, q)
            /\ (p \in positions \/ \E p2 \in positions : Subsumes(p2, p))

\* No non-subsumed position is removed
NoFalseRemoval ==
    \A q \in removed :
        \E p \in (positions \cup removed) : p # q /\ Subsumes(p, q)

(***************************************************************************)
(* Theorems                                                                 *)
(***************************************************************************)

THEOREM Irreflexive
THEOREM Asymmetric
THEOREM Transitive
THEOREM Spec => []TypeInv
THEOREM Spec => EventuallyAntichain

(***************************************************************************)
(* Model Checking Configuration                                             *)
(***************************************************************************)

\* For efficient model checking, use small bounds:
\* MAX_POSITION = 3
\* MAX_ERRORS = 2
\* ALGORITHMS = {"Standard", "Transposition", "MergeSplit"}
\*
\* This gives |AllPositions| = 4 * 3 * 2 * 3 * 3 = 216 positions
\* Total state space is manageable for exhaustive checking

================================================================================
