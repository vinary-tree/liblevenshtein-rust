// Dafny model of exact-scaled affine-gap arithmetic and B-4/B-5 subsumption.

datatype Layer = Match | QueryGap | DictGap
datatype Action = Diagonal(substitution: nat) | OpenQuery | OpenDict

function GapStep(incoming: Layer, target: Layer, open: nat, extend: nat): nat
{
  if incoming == target then extend else open + extend
}

function FirstStep(incoming: Layer, action: Action, open: nat, extend: nat): nat
{
  match action
  case Diagonal(substitution) => substitution
  case OpenQuery => GapStep(incoming, QueryGap, open, extend)
  case OpenDict => GapStep(incoming, DictGap, open, extend)
}

predicate LayerPrecedes(left: Layer, right: Layer)
{
  left == right || right == Match
}

predicate B4(leftCost: nat, leftLayer: Layer,
             rightCost: nat, rightLayer: Layer, open: nat)
{
  (LayerPrecedes(leftLayer, rightLayer) && leftCost <= rightCost)
  || leftCost + open <= rightCost
}

function QueryGapOpenCharge(incoming: Layer, open: nat): nat
{
  if incoming == QueryGap then 0 else open
}

function QueryGapRunCost(leftCost: nat, incoming: Layer, skipped: nat,
                         open: nat, extend: nat): nat
{
  leftCost + QueryGapOpenCharge(incoming, open) + skipped * extend
}

function RightRealignmentCharge(right: Layer, open: nat): nat
{
  if right == DictGap then open else 0
}

predicate B5Forward(leftCost: nat, leftLayer: Layer,
                    rightCost: nat, rightLayer: Layer, skipped: nat,
                    open: nat, extend: nat)
{
  skipped > 0
  && QueryGapRunCost(leftCost, leftLayer, skipped, open, extend)
       + RightRealignmentCharge(rightLayer, open) <= rightCost
}

lemma B1QueryGapPrecedesMatch(action: Action, open: nat, extend: nat)
  ensures FirstStep(QueryGap, action, open, extend)
       <= FirstStep(Match, action, open, extend)
{
  match action
  case Diagonal(_) =>
  case OpenQuery =>
  case OpenDict =>
}

lemma B1DictGapPrecedesMatch(action: Action, open: nat, extend: nat)
  ensures FirstStep(DictGap, action, open, extend)
       <= FirstStep(Match, action, open, extend)
{
  match action
  case Diagonal(_) =>
  case OpenQuery =>
  case OpenDict =>
}

lemma B2GapLayersAreIncomparable(open: nat, extend: nat)
  requires open > 0
  ensures FirstStep(QueryGap, OpenQuery, open, extend)
        < FirstStep(DictGap, OpenQuery, open, extend)
  ensures FirstStep(DictGap, OpenDict, open, extend)
        < FirstStep(QueryGap, OpenDict, open, extend)
{
}

lemma B3UniformSwitchPenalty(left: Layer, right: Layer, action: Action,
                             open: nat, extend: nat)
  ensures FirstStep(left, action, open, extend)
       <= FirstStep(right, action, open, extend) + open
{
  match left {
    case Match =>
      match right {
        case Match =>
        case QueryGap =>
        case DictGap =>
      }
    case QueryGap =>
      match right {
        case Match =>
        case QueryGap =>
        case DictGap =>
      }
    case DictGap =>
      match right {
        case Match =>
        case QueryGap =>
        case DictGap =>
      }
  }
}

lemma LayerPreorderStep(left: Layer, right: Layer, action: Action,
                        open: nat, extend: nat)
  requires LayerPrecedes(left, right)
  ensures FirstStep(left, action, open, extend)
       <= FirstStep(right, action, open, extend)
{
  if left != right {
    assert right == Match;
    match left
    case Match =>
    case QueryGap => B1QueryGapPrecedesMatch(action, open, extend);
    case DictGap => B1DictGapPrecedesMatch(action, open, extend);
  }
}

lemma B4PreservesEveryCommonStep(leftCost: nat, leftLayer: Layer,
                                 rightCost: nat, rightLayer: Layer,
                                 action: Action, open: nat, extend: nat)
  requires B4(leftCost, leftLayer, rightCost, rightLayer, open)
  ensures leftCost + FirstStep(leftLayer, action, open, extend)
       <= rightCost + FirstStep(rightLayer, action, open, extend)
{
  if LayerPrecedes(leftLayer, rightLayer) && leftCost <= rightCost {
    LayerPreorderStep(leftLayer, rightLayer, action, open, extend);
  } else {
    B3UniformSwitchPenalty(leftLayer, rightLayer, action, open, extend);
  }
}

lemma B5ForwardReachesB4(leftCost: nat, leftLayer: Layer,
                         rightCost: nat, rightLayer: Layer, skipped: nat,
                         open: nat, extend: nat)
  requires B5Forward(leftCost, leftLayer, rightCost, rightLayer,
                     skipped, open, extend)
  ensures B4(QueryGapRunCost(leftCost, leftLayer, skipped, open, extend),
             QueryGap, rightCost, rightLayer, open)
{
  match rightLayer
  case Match =>
  case QueryGap =>
  case DictGap =>
}

lemma B5ForwardPreservesEveryCommonStep(
    leftCost: nat, leftLayer: Layer,
    rightCost: nat, rightLayer: Layer, skipped: nat,
    action: Action, open: nat, extend: nat)
  requires B5Forward(leftCost, leftLayer, rightCost, rightLayer,
                     skipped, open, extend)
  ensures QueryGapRunCost(leftCost, leftLayer, skipped, open, extend)
            + FirstStep(QueryGap, action, open, extend)
       <= rightCost + FirstStep(rightLayer, action, open, extend)
{
  B5ForwardReachesB4(leftCost, leftLayer, rightCost, rightLayer,
                     skipped, open, extend);
  B4PreservesEveryCommonStep(
    QueryGapRunCost(leftCost, leftLayer, skipped, open, extend), QueryGap,
    rightCost, rightLayer, action, open, extend);
}

function EpsilonQueryGapCost(incoming: Layer, count: nat,
                             open: nat, extend: nat): nat
  decreases count
{
  if count == 0 then 0
  else GapStep(incoming, QueryGap, open, extend)
       + EpsilonQueryGapCost(QueryGap, count - 1, open, extend)
}

lemma EpsilonQueryGapFromOpen(count: nat, open: nat, extend: nat)
  ensures EpsilonQueryGapCost(QueryGap, count, open, extend) == count * extend
  decreases count
{
  if count > 0 {
    EpsilonQueryGapFromOpen(count - 1, open, extend);
  }
}

lemma FusedQueryGapCostRefinesEpsilonChain(
    leftCost: nat, incoming: Layer, skipped: nat, action: Action,
    open: nat, extend: nat)
  requires skipped > 0
  ensures QueryGapRunCost(leftCost, incoming, skipped, open, extend)
            + FirstStep(QueryGap, action, open, extend)
       == leftCost + EpsilonQueryGapCost(incoming, skipped, open, extend)
            + FirstStep(QueryGap, action, open, extend)
{
  EpsilonQueryGapFromOpen(skipped - 1, open, extend);
  match incoming
  case Match =>
  case QueryGap =>
  case DictGap =>
}

function FinishCost(cost: nat, incoming: Layer, remaining: nat,
                    open: nat, extend: nat): nat
{
  cost + remaining * extend
    + (if remaining == 0 || incoming == QueryGap then 0 else open)
}

lemma TrailingQueryGapDoesNotReopen(cost: nat, remaining: nat,
                                    open: nat, extend: nat)
  ensures FinishCost(cost, QueryGap, remaining, open, extend)
       == cost + remaining * extend
{
}

lemma TrailingRunOpensOnceFromMatch(cost: nat, remaining: nat,
                                    open: nat, extend: nat)
  requires remaining > 0
  ensures FinishCost(cost, Match, remaining, open, extend)
       == cost + open + remaining * extend
{
}

function OperationWindow(maximum: nat, cost: nat, extend: nat): nat
  requires cost <= maximum
  requires extend > 0
{
  (maximum - cost) / extend + 1
}

lemma OperationWindowBoundsAffordableRun(maximum: nat, cost: nat,
                                         extend: nat, operations: nat)
  requires cost <= maximum
  requires extend > 0
  requires cost + operations * extend <= maximum
  ensures operations < OperationWindow(maximum, cost, extend)
{
}

lemma CheckedAdditionStaysInBudget(cost: nat, increment: nat, maximum: nat)
  requires cost <= maximum
  requires increment <= maximum - cost
  ensures cost + increment <= maximum
{
}
