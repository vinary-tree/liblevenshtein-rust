--------------------------- MODULE ClassAPresets ---------------------------
EXTENDS Naturals, TLC

(***************************************************************************)
(* Finite operational model of the three built-in Class-A operation sets.  *)
(* Coordinates are consumed Unicode-scalar counts; cost is unit-scaled.     *)
(***************************************************************************)

CONSTANTS MaxSource, MaxTarget, MaxCost, TotalLimit

Presets == {"Hamming", "Indel", "Skip"}

Match == [dx |-> 1, dy |-> 1, cost |-> 0]
Substitute == [dx |-> 1, dy |-> 1, cost |-> 1]
Insert == [dx |-> 0, dy |-> 1, cost |-> 1]
Delete == [dx |-> 1, dy |-> 0, cost |-> 1]

Operations(preset) ==
  CASE preset = "Hamming" -> {Match, Substitute}
    [] preset = "Indel" -> {Match, Insert, Delete}
    [] preset = "Skip" -> {Match, Delete}

DeclaredTotal(preset) ==
  CASE preset = "Hamming" -> 4
    [] preset = "Indel" -> 4
    [] preset = "Skip" -> 3

VARIABLES preset, source, target, cost, steps
vars == <<preset, source, target, cost, steps>>

Init ==
  /\ preset \in Presets
  /\ source = 0
  /\ target = 0
  /\ cost = 0
  /\ steps = 0

Take(operation) ==
  /\ operation \in Operations(preset)
  /\ source + operation.dx <= MaxSource
  /\ target + operation.dy <= MaxTarget
  /\ cost + operation.cost <= MaxCost
  /\ source' = source + operation.dx
  /\ target' = target + operation.dy
  /\ cost' = cost + operation.cost
  /\ steps' = steps + 1
  /\ UNCHANGED preset

Next == \E operation \in Operations(preset) : Take(operation)
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ preset \in Presets
  /\ source \in 0..MaxSource
  /\ target \in 0..MaxTarget
  /\ cost \in 0..MaxCost
  /\ steps \in Nat

EveryOperationProgresses ==
  \A operation \in Operations(preset) : operation.dx + operation.dy > 0

DeclaredConsumptionBounded == DeclaredTotal(preset) <= TotalLimit

HammingPreservesLength == preset = "Hamming" => source = target

IndelLengthBound == preset = "Indel" =>
  /\ source <= target + cost
  /\ target <= source + cost

BoundedSkipDirection == preset = "Skip" =>
  /\ target <= source
  /\ source = target + cost

CostIsBudgetBounded == cost <= MaxCost

=============================================================================
