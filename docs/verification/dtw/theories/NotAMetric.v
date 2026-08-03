(** An executable triangle-inequality counterexample for band-one DTW.

    With squared local deviations, the only lawful couplings give squared
    costs 1 from [0] to [1], 0 from [1] to [1;1], and 2 from [0] to [1;1].
    Taking the public non-negative square root therefore yields
    [sqrt 2 > 1 + 0].  No metric-tree implementation may assume otherwise. *)

From Stdlib Require Import Reals Lra.

Open Scope R_scope.

Definition sq (x : R) : R := x * x.
Definition singleton_cost (x y : R) : R := sq (x - y).
Definition singleton_pair_cost (x y1 y2 : R) : R :=
  sq (x - y1) + sq (x - y2).

Example band_one_dtw_triangle_counterexample :
  sqrt (singleton_pair_cost 0 1 1)
  > sqrt (singleton_cost 0 1) + sqrt (singleton_pair_cost 1 1 1).
Proof.
  replace (singleton_pair_cost 0 1 1) with 2 by
    (unfold singleton_pair_cost, sq; ring).
  replace (singleton_cost 0 1) with 1 by
    (unfold singleton_cost, sq; ring).
  replace (singleton_pair_cost 1 1 1) with 0 by
    (unfold singleton_pair_cost, sq; ring).
  rewrite sqrt_1, sqrt_0, Rplus_0_r.
  assert (Hsqrt_nonnegative : 0 <= sqrt 2) by apply sqrt_pos.
  pose proof (Rsqr_sqrt 2 ltac:(lra)) as Hsqrt_square.
  unfold Rsqr in Hsqrt_square.
  nra.
Qed.
