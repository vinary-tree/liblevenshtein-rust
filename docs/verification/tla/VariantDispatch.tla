--------------------------- MODULE VariantDispatch ---------------------------
EXTENDS Naturals, Sequences, TLC

(***************************************************************************)
(* Exact finite model of Phase 5's dispatch placement. `legacy` chooses the *)
(* leaf for every position; `generic` chooses it once and reuses it.         *)
(***************************************************************************)

Algorithms == {"Standard", "Osa", "MergeSplit"}
Positions == 0..2

Select(a) ==
    CASE a = "Standard" -> 0
      [] a = "Osa" -> 1
      [] OTHER -> 2

Result(v, p) == 10 * v + p

VARIABLES algorithm, selected, cursor, generic, legacy
vars == <<algorithm, selected, cursor, generic, legacy>>

Init ==
    /\ algorithm \in Algorithms
    /\ selected = Select(algorithm)
    /\ cursor = 0
    /\ generic = <<>>
    /\ legacy = <<>>

Step ==
    /\ cursor < 3
    /\ generic' = Append(generic, Result(selected, cursor))
    /\ legacy' = Append(legacy, Result(Select(algorithm), cursor))
    /\ cursor' = cursor + 1
    /\ UNCHANGED <<algorithm, selected>>

Next == Step
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

TypeOK ==
    /\ algorithm \in Algorithms
    /\ selected \in 0..2
    /\ cursor \in 0..3
    /\ generic \in Seq(Nat)
    /\ legacy \in Seq(Nat)

SelectionStable == selected = Select(algorithm)
DispatchEquivalent == generic = legacy
ProcessedExactlyOnce == Len(generic) = cursor /\ Len(legacy) = cursor

=============================================================================
