// Dafny obligations for Phase-9 downstream query surfaces.

function SubstitutionCost(left: nat, right: nat): nat
{
  if left == right then 0 else 1
}

function EraseBracketKind(kinds: nat, token: nat): nat
{
  if token < kinds then 0 else 1
}

function ProjectedSubstitutionCost(kinds: nat, left: nat, right: nat): nat
{
  SubstitutionCost(EraseBracketKind(kinds, left), EraseBracketKind(kinds, right))
}

lemma KindErasureIsNonExpansive(kinds: nat, left: nat, right: nat)
  ensures ProjectedSubstitutionCost(kinds, left, right)
       <= SubstitutionCost(left, right)
{
}

function AlignmentCost(alignment: seq<(nat, nat)>): nat
  decreases |alignment|
{
  if |alignment| == 0 then 0
  else SubstitutionCost(alignment[0].0, alignment[0].1)
       + AlignmentCost(alignment[1..])
}

function ProjectedAlignmentCost(kinds: nat, alignment: seq<(nat, nat)>): nat
  decreases |alignment|
{
  if |alignment| == 0 then 0
  else ProjectedSubstitutionCost(kinds, alignment[0].0, alignment[0].1)
       + ProjectedAlignmentCost(kinds, alignment[1..])
}

lemma KindErasurePreservesAlignmentLowerBound(
    kinds: nat, alignment: seq<(nat, nat)>)
  ensures ProjectedAlignmentCost(kinds, alignment) <= AlignmentCost(alignment)
  decreases |alignment|
{
  if |alignment| > 0 {
    KindErasureIsNonExpansive(kinds, alignment[0].0, alignment[0].1);
    KindErasurePreservesAlignmentLowerBound(kinds, alignment[1..]);
  }
}

function BracketStateCount(kinds: nat, depth: nat): nat
  decreases depth
{
  if depth == 0 then 1
  else 1 + kinds * BracketStateCount(kinds, depth - 1)
}

lemma ZeroKindStateCountIsOne(depth: nat)
  ensures BracketStateCount(0, depth) == 1
  decreases depth
{
  if depth > 0 {
    ZeroKindStateCountIsOne(depth - 1);
  }
}

lemma BracketStateCountIsDepthMonotone(kinds: nat, depth: nat)
  ensures BracketStateCount(kinds, depth)
       <= BracketStateCount(kinds, depth + 1)
{
  if kinds == 0 {
    ZeroKindStateCountIsOne(depth);
    ZeroKindStateCountIsOne(depth + 1);
  }
}

lemma ThreeKindsDepthTenExceedsPublicGuard()
  ensures BracketStateCount(3, 10) == 88573
  ensures BracketStateCount(3, 10) > 4096
{
}

predicate VisitorBalanced(enters: nat, leaves: nat)
{
  enters == leaves
}

lemma RejectedEnterStillReceivesLeave(enters: nat, leaves: nat)
  requires VisitorBalanced(enters, leaves)
  ensures VisitorBalanced(enters + 1, leaves + 1)
{
}

predicate MatchModeAccepts(minimum: nat, maximum: nat, distance: nat)
{
  minimum <= distance <= maximum
}

lemma ExactMatchModeAcceptsOnlyItsDistance(exact: nat, distance: nat)
  ensures MatchModeAccepts(exact, exact, distance) <==> distance == exact
{
}

lemma RangeMatchModeRespectsAutomatonBudget(
    minimum: nat, maximum: nat, distance: nat)
  requires MatchModeAccepts(minimum, maximum, distance)
  ensures distance <= maximum
{
}

lemma UnwindingActiveDfsFramesRestoresBalance(
    enters: nat, leaves: nat, depth: nat)
  requires enters == leaves + depth
  ensures enters == (leaves + depth)
{
}

predicate RankedBefore(
    leftDistance: nat, leftConfidence: nat, leftTerm: nat,
    rightDistance: nat, rightConfidence: nat, rightTerm: nat)
{
  leftDistance < rightDistance
  || (leftDistance == rightDistance
      && (leftConfidence > rightConfidence
          || (leftConfidence == rightConfidence && leftTerm <= rightTerm)))
}

lemma RankedOrderIsAntisymmetric(
    ld: nat, lc: nat, lt: nat, rd: nat, rc: nat, rt: nat)
  requires RankedBefore(ld, lc, lt, rd, rc, rt)
  requires RankedBefore(rd, rc, rt, ld, lc, lt)
  ensures ld == rd && lc == rc && lt == rt
{
}

function IndexOffset(left: nat, right: nat): nat
{
  if left <= right then right - left else left - right
}

predicate ContextualRealignmentSafe(
    left: nat, right: nat, minimum: nat, slack: nat)
{
  IndexOffset(left, right) * minimum <= slack
}

lemma ContextualGuardIsSymmetric(
    left: nat, right: nat, minimum: nat, slack: nat)
  ensures ContextualRealignmentSafe(left, right, minimum, slack)
       <==> ContextualRealignmentSafe(right, left, minimum, slack)
{
}

lemma ZeroSlackForbidsDistinctPositions(left: nat, right: nat, minimum: nat)
  requires minimum > 0
  requires ContextualRealignmentSafe(left, right, minimum, 0)
  ensures left == right
{
}
