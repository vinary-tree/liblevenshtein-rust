(** * Restricted Damerau Triangle Counterexample

    The executable recurrence in Core.DamerauLevDistanceDef is the restricted
    adjacent-transposition, or optimal-string-alignment, recurrence. That model
    is useful, but it is not an unconditional metric: a substring cannot be
    edited more than once along a single optimal path, so the triangle
    inequality can fail.

    Reference: Boytsov, L. (2011), "Indexing methods for approximate dictionary
    searching: Comparative analysis", ACM Journal of Experimental Algorithmics
    16, Article 1.1, Section 2.2.1 distinguishes unrestricted Damerau-
    Levenshtein distance from the restricted/optimal-string-alignment variant.
    DOI: 10.1145/1963190.1963191.
*)

From Stdlib Require Import List Arith Ascii Lia.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.DamerauLevDistanceDef.

Definition dl_ce_a : Char := Ascii.ascii_of_nat 97.
Definition dl_ce_b : Char := Ascii.ascii_of_nat 98.
Definition dl_ce_c : Char := Ascii.ascii_of_nat 99.

Lemma restricted_damerau_counterexample_values :
  damerau_lev_distance [dl_ce_a; dl_ce_b] [dl_ce_b; dl_ce_c; dl_ce_a] = 3 /\
  damerau_lev_distance [dl_ce_a; dl_ce_b] [dl_ce_b; dl_ce_a] = 1 /\
  damerau_lev_distance [dl_ce_b; dl_ce_a] [dl_ce_b; dl_ce_c; dl_ce_a] = 1.
Proof.
  vm_compute. repeat split; reflexivity.
Qed.

Theorem restricted_damerau_triangle_counterexample :
  damerau_lev_distance [dl_ce_a; dl_ce_b] [dl_ce_b; dl_ce_c; dl_ce_a] >
  damerau_lev_distance [dl_ce_a; dl_ce_b] [dl_ce_b; dl_ce_a] +
  damerau_lev_distance [dl_ce_b; dl_ce_a] [dl_ce_b; dl_ce_c; dl_ce_a].
Proof.
  vm_compute. lia.
Qed.

Theorem restricted_damerau_not_unconditional_triangle :
  ~ (forall A B C : list Char,
      damerau_lev_distance A C <=
      damerau_lev_distance A B + damerau_lev_distance B C).
Proof.
  intro Htriangle.
  pose proof (Htriangle
    [dl_ce_a; dl_ce_b]
    [dl_ce_b; dl_ce_a]
    [dl_ce_b; dl_ce_c; dl_ce_a]) as Hbad.
  vm_compute in Hbad. lia.
Qed.
