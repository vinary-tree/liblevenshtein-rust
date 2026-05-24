(** * Restricted Damerau-Levenshtein Distance Interface

    This module re-exports the Damerau-Levenshtein distance function and all
    basic lemmas from DamerauLevDistanceDef.

    Part of: Liblevenshtein.Core

    This module provides the interface for the executable restricted adjacent-
    transposition recurrence, including:
    - The distance function (from DamerauLevDistanceDef)
    - Basic lemmas: empty cases, single char, symmetry, etc. (from DamerauLevDistanceDef)
    - A formal counterexample showing that the unrestricted triangle inequality
      is false for this recurrence.

    Module dependency structure (no cycles):
    - DamerauLevDistanceDef: defines distance function + basic lemmas
    - DamerauComposition: records the restricted-model triangle counterexample
    - DamerauLevDistance (this module): imports DamerauLevDistanceDef and the
      counterexample theorem
*)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia Wf_nat.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.

(** Re-export all definitions and lemmas from DamerauLevDistanceDef *)
From Liblevenshtein.Core Require Export Core.DamerauLevDistanceDef.

(** Import the restricted-model counterexample from DamerauComposition. *)
From Liblevenshtein.Core Require Import Composition.DamerauComposition.

(** * Restricted-Model Triangle Failure

    The local recurrence is the restricted/optimal-string-alignment variant,
    not unrestricted Damerau-Levenshtein distance. The theorem below keeps that
    limitation visible at the public interface.
*)
Theorem damerau_lev_not_unconditional_triangle :
  ~ (forall A B C : list Char,
      damerau_lev_distance A C <=
      damerau_lev_distance A B + damerau_lev_distance B C).
Proof.
  apply restricted_damerau_not_unconditional_triangle.
Qed.
