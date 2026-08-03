-------------------------- MODULE DamerauStreaming ---------------------------
EXTENDS Naturals, TLC

(***************************************************************************)
(* Finite operational model of one Lowrance--Wagner streaming macro.       *)
(* Every Pending step consumes one dictionary unit; there is no epsilon     *)
(* action. Entry prepays delta, Extend adds one, and Resolve adds no cost.   *)
(***************************************************************************)

CONSTANT K

VARIABLES stage, delta, errors, between, queryIndex, consumed
vars == <<stage, delta, errors, between, queryIndex, consumed>>

Init ==
    /\ stage = "Normal"
    /\ delta = 0
    /\ errors = 0
    /\ between = 0
    /\ queryIndex = 0
    /\ consumed = 0

Entry(d) ==
    /\ stage = "Normal"
    /\ d \in 1..K
    /\ stage' = "Pending"
    /\ delta' = d
    /\ errors' = d
    /\ between' = 0
    /\ queryIndex' = queryIndex
    /\ consumed' = consumed + 1

Extend ==
    /\ stage = "Pending"
    /\ errors < K
    /\ stage' = "Pending"
    /\ delta' = delta
    /\ errors' = errors + 1
    /\ between' = between + 1
    /\ queryIndex' = queryIndex
    /\ consumed' = consumed + 1

Resolve ==
    /\ stage = "Pending"
    /\ stage' = "Done"
    /\ delta' = delta
    /\ errors' = errors
    /\ between' = between
    /\ queryIndex' = queryIndex + delta + 1
    /\ consumed' = consumed + 1

Next == (\E d \in 1..K : Entry(d)) \/ Extend \/ Resolve
Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ stage \in {"Normal", "Pending", "Done"}
    /\ delta \in 0..K
    /\ errors \in 0..K
    /\ between \in 0..K
    /\ queryIndex \in Nat
    /\ consumed \in Nat

PendingCharge == stage = "Pending" => errors = delta + between
PendingDeltaValid == stage = "Pending" => delta \in 1..K
ResolvedMacroCost == stage = "Done" => errors = delta + between
ResolvedEndpoint == stage = "Done" => queryIndex = delta + 1
PendingAlwaysConsumes == stage = "Pending" => consumed >= 1

=============================================================================
