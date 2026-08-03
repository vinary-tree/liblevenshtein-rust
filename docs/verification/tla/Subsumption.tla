------------------------------- MODULE Subsumption -------------------------------
(***************************************************************************)
(* Exact bounded model of src/transducer/position.rs and State::insert.     *)
(*                                                                         *)
(* Subsumption is a coverage relation, not uniformly a strict order:        *)
(* Standard and Transposition are reflexive, while MergeAndSplit uses a     *)
(* strict error inequality. State insertion removes only distinct covered   *)
(* representatives, so StrictDominates is the lifecycle relation.           *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    MAX_POSITION,
    MAX_ERRORS,
    QUERY_LENGTH,
    ALGORITHMS

ASSUME MAX_POSITION >= 1
ASSUME MAX_ERRORS >= 1
ASSUME QUERY_LENGTH \in 0..MAX_POSITION
ASSUME ALGORITHMS = {"Standard", "Transposition", "MergeSplit"}

AbsInt(value) == IF value < 0 THEN -value ELSE value

(***************************************************************************)
(* Position domain. Standard never generates a special representative.      *)
(***************************************************************************)

RawPositions == [
    i: 0..MAX_POSITION,
    e: 0..MAX_ERRORS,
    special: BOOLEAN,
    alg: ALGORITHMS
]

WellFormed(p) == p.alg # "Standard" \/ ~p.special

AllPositions == {p \in RawPositions : WellFormed(p)}

(***************************************************************************)
(* Executable relations, branch-for-branch with Position::subsumes.          *)
(***************************************************************************)

StandardSubsumes(p, q) ==
    /\ p.alg = "Standard"
    /\ q.alg = "Standard"
    /\ p.e <= q.e
    /\ AbsInt(p.i - q.i) <= q.e - p.e

TranspositionSubsumes(p, q) ==
    /\ p.alg = "Transposition"
    /\ q.alg = "Transposition"
    /\ p.e <= q.e
    /\ p.special = q.special
    /\ IF p.special
          THEN p.i = q.i
          ELSE AbsInt(p.i - q.i) <= q.e - p.e

MergeSplitSubsumes(p, q) ==
    /\ p.alg = "MergeSplit"
    /\ q.alg = "MergeSplit"
    /\ p.special = q.special
    /\ p.i <= QUERY_LENGTH
    /\ ~(p.special /\ p.i >= QUERY_LENGTH /\ q.i < QUERY_LENGTH)
    /\ p.e < q.e
    /\ p.i = q.i

Subsumes(p, q) ==
    \/ StandardSubsumes(p, q)
    \/ TranspositionSubsumes(p, q)
    \/ MergeSplitSubsumes(p, q)

StrictDominates(p, q) == p # q /\ Subsumes(p, q)

(***************************************************************************)
(* Algebraic conformance properties.                                        *)
(***************************************************************************)

StandardReflexive ==
    \A p \in AllPositions : p.alg = "Standard" => Subsumes(p, p)

TranspositionReflexive ==
    \A p \in AllPositions : p.alg = "Transposition" => Subsumes(p, p)

MergeSplitIrreflexive ==
    \A p \in AllPositions : p.alg = "MergeSplit" => ~Subsumes(p, p)

CrossAlgorithmSeparated ==
    \A p, q \in AllPositions : p.alg # q.alg => ~Subsumes(p, q)

TranspositionVariantSeparated ==
    \A p, q \in AllPositions :
        /\ p.alg = "Transposition"
        /\ q.alg = "Transposition"
        /\ p.special # q.special
        => ~Subsumes(p, q)

SubsumesAntisymmetric ==
    \A p, q \in AllPositions :
        (Subsumes(p, q) /\ Subsumes(q, p)) => p = q

SubsumesTransitive ==
    \A p, q, r \in AllPositions :
        (Subsumes(p, q) /\ Subsumes(q, r)) => Subsumes(p, r)

StrictIrreflexive ==
    \A p \in AllPositions : ~StrictDominates(p, p)

StrictAsymmetric ==
    \A p, q \in AllPositions :
        StrictDominates(p, q) => ~StrictDominates(q, p)

StrictTransitive ==
    \A p, q, r \in AllPositions :
        (StrictDominates(p, q) /\ StrictDominates(q, r))
        => StrictDominates(p, r)

(***************************************************************************)
(* State insertion lifecycle.                                               *)
(***************************************************************************)

SamplePositions ==
    {p \in AllPositions : p.i \in 0..1 /\ p.e \in 0..1}

FormAntichain(ps) ==
    \A p, q \in ps : p # q => ~StrictDominates(p, q)

VARIABLES positions, removed, iteration

SubsumptionVars == <<positions, removed, iteration>>

TypeInv ==
    /\ positions \subseteq AllPositions
    /\ removed \subseteq AllPositions
    /\ positions \cap removed = {}
    /\ iteration \in Nat

AlgebraicInv ==
    /\ TypeInv
    /\ StandardReflexive
    /\ TranspositionReflexive
    /\ MergeSplitIrreflexive
    /\ CrossAlgorithmSeparated
    /\ TranspositionVariantSeparated
    /\ SubsumesAntisymmetric
    /\ SubsumesTransitive
    /\ StrictIrreflexive
    /\ StrictAsymmetric
    /\ StrictTransitive

Init ==
    /\ positions = SamplePositions
    /\ removed = {}
    /\ iteration = 0

RemoveSubsumed ==
    /\ \E p, q \in positions :
        /\ StrictDominates(p, q)
        /\ positions' = positions \ {q}
        /\ removed' = removed \cup {q}
        /\ iteration' = iteration + 1

Done ==
    /\ FormAntichain(positions)
    /\ UNCHANGED SubsumptionVars

Next == RemoveSubsumed \/ Done

Spec == Init /\ [][Next]_SubsumptionVars /\ WF_SubsumptionVars(RemoveSubsumed)

RemovedHasCurrentCover ==
    \A q \in removed : \E p \in positions : StrictDominates(p, q)

IterationBound == iteration <= Cardinality(SamplePositions)

EventuallyAntichain == <>FormAntichain(positions)

THEOREM Spec => []TypeInv
THEOREM Spec => []AlgebraicInv
THEOREM Spec => []RemovedHasCurrentCover
THEOREM Spec => []IterationBound
THEOREM Spec => EventuallyAntichain

=============================================================================
