-------------------- MODULE TemporalStreamingGenerations --------------------
(***************************************************************************)
(* Fixed-query temporal streaming with current/next generations.           *)
(*                                                                         *)
(* The consumed prefix is a counter, never retained input. Each transition *)
(* builds only the next canonical frontier, commits it, and reclaims the    *)
(* prior generation. The bounded cache is independent of prefix length.    *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS FrontierLimit, CacheLimit, InputLength

Positions == 1..5
CacheCells == 1..4

FrontierAt ==
  (0 :> {1}) @@ (1 :> {1, 2}) @@ (2 :> {2, 3}) @@ (3 :> {3})
  @@ (4 :> {2, 4}) @@ (5 :> {4, 5}) @@ (6 :> {5})

CacheAt ==
  (0 :> {}) @@ (1 :> {1}) @@ (2 :> {1, 2}) @@ (3 :> {2, 3})
  @@ (4 :> {3, 4}) @@ (5 :> {1, 4}) @@ (6 :> {1, 2})

VARIABLES consumed, current, nextGeneration, cache, phase
vars == <<consumed, current, nextGeneration, cache, phase>>

Init ==
  /\ consumed = 0
  /\ current = FrontierAt[0]
  /\ nextGeneration = {}
  /\ cache = CacheAt[0]
  /\ phase = "Ready"

BuildNext ==
  /\ phase = "Ready"
  /\ consumed < InputLength
  /\ nextGeneration' = FrontierAt[consumed + 1]
  /\ cache' = CacheAt[consumed + 1]
  /\ phase' = "Built"
  /\ UNCHANGED <<consumed, current>>

CommitAndReclaim ==
  /\ phase = "Built"
  /\ current' = nextGeneration
  /\ nextGeneration' = {}
  /\ consumed' = consumed + 1
  /\ phase' = "Ready"
  /\ UNCHANGED cache

Finished == phase = "Ready" /\ consumed = InputLength

Next == BuildNext \/ CommitAndReclaim \/ (Finished /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(BuildNext) /\ WF_vars(CommitAndReclaim)

TypeOK ==
  /\ consumed \in 0..InputLength
  /\ current \subseteq Positions
  /\ nextGeneration \subseteq Positions
  /\ cache \subseteq CacheCells
  /\ phase \in {"Ready", "Built"}

FrontiersBounded ==
  /\ Cardinality(current) <= FrontierLimit
  /\ Cardinality(nextGeneration) <= FrontierLimit

CacheBounded == Cardinality(cache) <= CacheLimit

RetainedStateBound ==
  Cardinality(current) + Cardinality(nextGeneration) + Cardinality(cache)
    <= 2 * FrontierLimit + CacheLimit

ReadyHasNoPreviousGeneration == phase = "Ready" => nextGeneration = {}
CurrentMatchesConsumedPrefix == phase = "Ready" => current = FrontierAt[consumed]
EventuallyConsumesInput == <>Finished

THEOREM Spec => []TypeOK
THEOREM Spec => []FrontiersBounded
THEOREM Spec => []CacheBounded
THEOREM Spec => []RetainedStateBound
THEOREM Spec => []ReadyHasNoPreviousGeneration
THEOREM Spec => []CurrentMatchesConsumedPrefix
THEOREM Spec => EventuallyConsumesInput

=============================================================================
