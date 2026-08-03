// Dafny model of Class-A metric, band, subsequence, and validation invariants.

function Mismatch(left: int, right: int): nat
{
  if left == right then 0 else 1
}

lemma HammingIdentity(value: int)
  ensures Mismatch(value, value) == 0
{
}

lemma HammingSymmetry(left: int, right: int)
  ensures Mismatch(left, right) == Mismatch(right, left)
{
}

lemma HammingCoordinateTriangle(left: int, middle: int, right: int)
  ensures Mismatch(left, right)
       <= Mismatch(left, middle) + Mismatch(middle, right)
{
  if left != right {
    if left == middle {
      assert middle != right;
    }
  }
}

lemma HammingPrefixTriangle(leftRight: nat, leftMiddle: nat, middleRight: nat,
                            cellLeftRight: nat, cellLeftMiddle: nat,
                            cellMiddleRight: nat)
  requires leftRight <= leftMiddle + middleRight
  requires cellLeftRight <= cellLeftMiddle + cellMiddleRight
  ensures leftRight + cellLeftRight
       <= (leftMiddle + cellLeftMiddle) + (middleRight + cellMiddleRight)
{
}

lemma IndelReversePreservesCost(kept: nat, inserted: nat, deleted: nat)
  ensures inserted + deleted == deleted + inserted
{
}

lemma IndelLengthBounds(kept: nat, inserted: nat, deleted: nat)
  ensures kept + deleted <= kept + inserted + inserted + deleted
  ensures kept + inserted <= kept + deleted + inserted + deleted
{
}

lemma IndelParity(kept: nat, inserted: nat, deleted: nat)
  ensures ((kept + deleted) + (kept + inserted)) % 2
       == (inserted + deleted) % 2
{
}

lemma IndelComposition(firstInserted: nat, firstDeleted: nat,
                       secondInserted: nat, secondDeleted: nat)
  ensures (firstInserted + secondInserted) + (firstDeleted + secondDeleted)
       == (firstInserted + firstDeleted) + (secondInserted + secondDeleted)
{
}

lemma BoundedSkipExactCost(matched: nat, skipped: nat)
  ensures matched <= matched + skipped
  ensures (matched + skipped) - matched == skipped
{
}

lemma ProgressingOperationAdvances(source: nat, target: nat)
  requires source + target > 0
  ensures source > 0 || target > 0
{
}

lemma ValidatedPrefixIsBounded(prefix: nat, suffix: nat, limit: nat)
  requires prefix + suffix <= limit
  ensures prefix <= limit
{
}

lemma CheckedAggregateUpdate(aggregate: nat, source: nat,
                             target: nat, limit: nat)
  requires aggregate <= limit
  requires source + target <= limit - aggregate
  ensures aggregate + source + target <= limit
{
}

lemma AffordableEmptySideBoundary(length: nat, budget: nat)
  requires length <= budget
  ensures length <= budget
{
}

lemma BandedCellMustBeNearDiagonal(row: nat, column: nat,
                                   prefixCost: nat, budget: nat)
  requires prefixCost <= budget
  requires row <= column + prefixCost
  requires column <= row + prefixCost
  ensures row <= column + budget
  ensures column <= row + budget
{
}
