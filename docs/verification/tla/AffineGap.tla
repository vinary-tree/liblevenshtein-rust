------------------------------ MODULE AffineGap -----------------------------
EXTENDS Integers, Naturals, TLC

(***************************************************************************)
(* Finite transition model of B-4 affine dominance. Two representatives    *)
(* begin at the same query index under B-4, take identical actions, and     *)
(* must preserve cost dominance.                                            *)
(***************************************************************************)

CONSTANTS GapOpen, GapExtend, Substitution, MaxCost, StepLimit

Layers == {"M", "Q", "D"}
Actions == {"Diagonal", "QueryGap", "DictGap"}

GapStep(incoming, target) ==
    IF incoming = target THEN GapExtend ELSE GapOpen + GapExtend

StepCost(incoming, action) ==
    CASE action = "Diagonal" -> Substitution
      [] action = "QueryGap" -> GapStep(incoming, "Q")
      [] action = "DictGap" -> GapStep(incoming, "D")

TargetLayer(action) ==
    CASE action = "Diagonal" -> "M"
      [] action = "QueryGap" -> "Q"
      [] action = "DictGap" -> "D"

Precedes(left, right) == left = right \/ right = "M"
B4(leftCost, leftLayer, rightCost, rightLayer) ==
    (Precedes(leftLayer, rightLayer) /\ leftCost <= rightCost)
    \/ leftCost + GapOpen <= rightCost

VARIABLES leftCost, leftLayer, rightCost, rightLayer, steps
vars == <<leftCost, leftLayer, rightCost, rightLayer, steps>>

Init ==
    /\ leftLayer \in Layers
    /\ rightLayer \in Layers
    /\ leftCost \in 0..MaxCost
    /\ rightCost \in 0..MaxCost
    /\ B4(leftCost, leftLayer, rightCost, rightLayer)
    /\ steps = 0

Take(action) ==
    /\ action \in Actions
    /\ steps < StepLimit
    /\ leftCost' = leftCost + StepCost(leftLayer, action)
    /\ rightCost' = rightCost + StepCost(rightLayer, action)
    /\ leftLayer' = TargetLayer(action)
    /\ rightLayer' = TargetLayer(action)
    /\ steps' = steps + 1

Next == \E action \in Actions : Take(action)
Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ leftLayer \in Layers
    /\ rightLayer \in Layers
    /\ leftCost \in Nat
    /\ rightCost \in Nat
    /\ steps \in 0..StepLimit

DominancePreserved == leftCost <= rightCost
CanonicalLayersAgreeAfterStep == steps > 0 => leftLayer = rightLayer
B4Preserved == B4(leftCost, leftLayer, rightCost, rightLayer)

B2SeparatingActions ==
    steps >= 0 /\ (GapOpen > 0 =>
      /\ StepCost("Q", "QueryGap") < StepCost("D", "QueryGap")
      /\ StepCost("D", "DictGap") < StepCost("Q", "DictGap"))

OperationWindowSound ==
    steps >= 0 /\ \A cost \in 0..MaxCost :
      \A operations \in 0..MaxCost :
        cost + operations * GapExtend <= MaxCost
        => operations < (MaxCost - cost) \div GapExtend + 1

=============================================================================
