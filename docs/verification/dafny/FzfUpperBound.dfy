// Auto-active model of the fzf local-alignment upper-bound obligations.

function Max(left: int, right: int): int {
  if left >= right then left else right
}

lemma LiveAlternativeBound(exact: int, unstarted: int, active: int)
  requires exact <= unstarted || exact <= active
  ensures exact <= Max(unstarted, active)
{}

lemma PruneSound(exact: int, unstarted: int, active: int, cutoff: int)
  requires exact <= unstarted || exact <= active
  requires Max(unstarted, active) < cutoff
  ensures exact < cutoff
{
  LiveAlternativeBound(exact, unstarted, active);
}

lemma ArcticDeltasTelescope(initial: int, middle: int, finalScore: int)
  ensures initial + (middle - initial) + (finalScore - middle) == finalScore
{}

lemma ActiveOnlyFormulaHasCounterexample()
  ensures 10 < 20
  ensures 20 <= Max(20, 10)
{}
