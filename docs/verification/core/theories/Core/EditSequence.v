(** * Edit Sequence Infrastructure for Triangle Inequality Proofs

    This module defines edit operations, their application to strings, and proves
    key composition lemmas needed for triangle inequality proofs.

    Part of: Liblevenshtein.Core

    The key insight for proving d(A,C) <= d(A,B) + d(B,C):
    1. Edit sequences can be composed: seq_AB from A→B and seq_BC from B→C
       give seq_AC from A→C.
    2. The cost of the composed sequence is at most the sum of the individual costs.
    3. Since d(A,C) is the minimum cost of any sequence A→C, we have
       d(A,C) <= cost(seq_AC) <= cost(seq_AB) + cost(seq_BC) = d(A,B) + d(B,C).

    References:
    - Wagner & Fischer (1974), "The String-to-String Correction Problem"
    - Damerau (1964), "A technique for computer detection and correction..."
    - This is a well-known result in edit distance theory
*)

From Stdlib Require Import String List Arith Ascii Bool Nat Lia.
From Stdlib Require Import Wf_nat.
Import ListNotations.

From Liblevenshtein.Core Require Import Core.Definitions.
From Liblevenshtein.Core Require Import Core.LevDistance.
From Liblevenshtein.Core Require Import Core.DamerauLevDistanceDef.
From Liblevenshtein.Core Require Import Core.MergeSplitDistance.
From Liblevenshtein.Core Require Import Triangle.TriangleInequality.

(** * Edit Operation Types *)

(** Standard Levenshtein edit operations *)
Inductive LevEditOp : Type :=
  | LEOMatch (c : Char)        (* Match character, advance both positions *)
  | LEOInsert (c : Char)       (* Insert c into target, advance target position *)
  | LEODelete (c : Char)       (* Delete c from source, advance source position *)
  | LEOSubstitute (c1 c2 : Char). (* Replace c1 with c2, advance both positions *)

(** Damerau-Levenshtein edit operations (include transposition) *)
Inductive DLEditOp : Type :=
  | DLMatch (c : Char)
  | DLInsert (c : Char)
  | DLDelete (c : Char)
  | DLSubstitute (c1 c2 : Char)
  | DLTranspose (c1 c2 : Char). (* Transpose adjacent chars c1, c2 -> c2, c1 *)

(** Merge-Split edit operations *)
Inductive MSEditOp : Type :=
  | MSOMatch (c : Char)
  | MSOInsert (c : Char)
  | MSODelete (c : Char)
  | MSOSubstitute (c1 c2 : Char)
  | MSOMerge (c1 c2 c : Char)    (* Merge two chars c1, c2 into c *)
  | MSOSplit (c c1 c2 : Char).   (* Split char c into c1, c2 *)

(** Edit sequence types *)
Definition LevEditSeq := list LevEditOp.
Definition DLEditSeq := list DLEditOp.
Definition MSEditSeq := list MSEditOp.

(** * Edit Operation Costs *)

(** Cost of a Levenshtein edit operation *)
Definition lev_op_cost (op : LevEditOp) : nat :=
  match op with
  | LEOMatch _ => 0
  | LEOInsert _ => 1
  | LEODelete _ => 1
  | LEOSubstitute c1 c2 => if char_eq c1 c2 then 0 else 1
  end.

(** Cost of a Damerau-Levenshtein edit operation *)
Definition dl_op_cost (op : DLEditOp) : nat :=
  match op with
  | DLMatch _ => 0
  | DLInsert _ => 1
  | DLDelete _ => 1
  | DLSubstitute c1 c2 => if char_eq c1 c2 then 0 else 1
  | DLTranspose _ _ => 1
  end.

(** Cost of a Merge-Split edit operation *)
Definition ms_op_cost (op : MSEditOp) : nat :=
  match op with
  | MSOMatch _ => 0
  | MSOInsert _ => 1
  | MSODelete _ => 1
  | MSOSubstitute c1 c2 => if char_eq c1 c2 then 0 else 1
  | MSOMerge c1 c2 c => merge_cost c1 c2 c
  | MSOSplit c c1 c2 => split_cost c c1 c2
  end.

(** Sequence costs *)
Definition lev_seq_cost (seq : LevEditSeq) : nat :=
  fold_left (fun acc op => acc + lev_op_cost op) seq 0.

Definition dl_seq_cost (seq : DLEditSeq) : nat :=
  fold_left (fun acc op => acc + dl_op_cost op) seq 0.

Definition ms_seq_cost (seq : MSEditSeq) : nat :=
  fold_left (fun acc op => acc + ms_op_cost op) seq 0.

(** * Apply Edit Operations *)

(** Apply a Levenshtein edit sequence to transform source to target.
    Returns Some target if valid, None if invalid. *)
Fixpoint apply_lev_seq (source : list Char) (seq : LevEditSeq) : option (list Char) :=
  match seq with
  | [] => Some source  (* Empty sequence: result is source *)
  | op :: rest =>
    match op, source with
    | LEOMatch c, c' :: source' =>
      if char_eq c c' then apply_lev_seq source' rest else None
    | LEOInsert c, _ =>
      match apply_lev_seq source rest with
      | Some target' => Some (c :: target')
      | None => None
      end
    | LEODelete c, c' :: source' =>
      if char_eq c c' then apply_lev_seq source' rest else None
    | LEOSubstitute c1 c2, c' :: source' =>
      if char_eq c1 c' then
        match apply_lev_seq source' rest with
        | Some target' => Some (c2 :: target')
        | None => None
        end
      else None
    | LEOMatch _, [] => None
    | LEODelete _, [] => None
    | LEOSubstitute _ _, [] => None
    end
  end.

(** Apply a Damerau-Levenshtein edit sequence *)
Fixpoint apply_dl_seq (source : list Char) (seq : DLEditSeq) : option (list Char) :=
  match seq with
  | [] => Some source
  | op :: rest =>
    match op, source with
    | DLMatch c, c' :: source' =>
      if char_eq c c' then apply_dl_seq source' rest else None
    | DLInsert c, _ =>
      match apply_dl_seq source rest with
      | Some target' => Some (c :: target')
      | None => None
      end
    | DLDelete c, c' :: source' =>
      if char_eq c c' then apply_dl_seq source' rest else None
    | DLSubstitute c1 c2, c' :: source' =>
      if char_eq c1 c' then
        match apply_dl_seq source' rest with
        | Some target' => Some (c2 :: target')
        | None => None
        end
      else None
    | DLTranspose c1 c2, c1' :: c2' :: source' =>
      if andb (char_eq c1 c1') (char_eq c2 c2') then
        match apply_dl_seq source' rest with
        | Some target' => Some (c2 :: c1 :: target')  (* Swap output *)
        | None => None
        end
      else None
    | DLMatch _, [] => None
    | DLDelete _, [] => None
    | DLSubstitute _ _, [] => None
    | DLTranspose _ _, [] => None
    | DLTranspose _ _, [_] => None
    end
  end.

(** * Valid Edit Sequences *)

(** A valid Levenshtein edit sequence transforms source to target *)
Definition valid_lev_seq (source target : list Char) (seq : LevEditSeq) : Prop :=
  apply_lev_seq source seq = Some target.

(** A valid Damerau-Levenshtein edit sequence transforms source to target *)
Definition valid_dl_seq (source target : list Char) (seq : DLEditSeq) : Prop :=
  apply_dl_seq source seq = Some target.

(** * Sequence Concatenation Properties *)

(** Helper: fold_left cost shift lemma *)
Lemma fold_left_lev_cost_shift :
  forall seq init,
    fold_left (fun acc op => acc + lev_op_cost op) seq init =
    init + fold_left (fun acc op => acc + lev_op_cost op) seq 0.
Proof.
  induction seq as [| op rest IH]; intros init.
  - simpl. lia.
  - simpl. rewrite IH. rewrite (IH (lev_op_cost op)). lia.
Qed.

Lemma fold_left_dl_cost_shift :
  forall seq init,
    fold_left (fun acc op => acc + dl_op_cost op) seq init =
    init + fold_left (fun acc op => acc + dl_op_cost op) seq 0.
Proof.
  induction seq as [| op rest IH]; intros init.
  - simpl. lia.
  - simpl. rewrite IH. rewrite (IH (dl_op_cost op)). lia.
Qed.

Lemma fold_left_ms_cost_shift :
  forall seq init,
    fold_left (fun acc op => acc + ms_op_cost op) seq init =
    init + fold_left (fun acc op => acc + ms_op_cost op) seq 0.
Proof.
  induction seq as [| op rest IH]; intros init.
  - simpl. lia.
  - simpl. rewrite IH. rewrite (IH (ms_op_cost op)). lia.
Qed.

(** Cost of concatenated sequences equals sum of costs *)
Lemma lev_seq_cost_app : forall seq1 seq2,
  lev_seq_cost (seq1 ++ seq2) = lev_seq_cost seq1 + lev_seq_cost seq2.
Proof.
  intros seq1 seq2.
  unfold lev_seq_cost.
  rewrite fold_left_app.
  rewrite fold_left_lev_cost_shift.
  reflexivity.
Qed.

Lemma dl_seq_cost_app : forall seq1 seq2,
  dl_seq_cost (seq1 ++ seq2) = dl_seq_cost seq1 + dl_seq_cost seq2.
Proof.
  intros seq1 seq2.
  unfold dl_seq_cost.
  rewrite fold_left_app.
  rewrite fold_left_dl_cost_shift.
  reflexivity.
Qed.

Lemma ms_seq_cost_app : forall seq1 seq2,
  ms_seq_cost (seq1 ++ seq2) = ms_seq_cost seq1 + ms_seq_cost seq2.
Proof.
  intros seq1 seq2.
  unfold ms_seq_cost.
  rewrite fold_left_app.
  rewrite fold_left_ms_cost_shift.
  reflexivity.
Qed.

(** * Triangle Inequality Proofs *)

(** Theorem: Levenshtein distance satisfies triangle inequality.
    Proof: reuse the trace-composition theorem from Triangle/TriangleInequality.v. *)
Theorem lev_triangle_inequality : forall A B C,
  lev_distance A C <= lev_distance A B + lev_distance B C.
Proof.
  intros A B C.
  apply lev_distance_triangle_inequality.
Qed.

(** The local Damerau recurrence in Core.DamerauLevDistanceDef is the restricted
    adjacent-transposition recurrence. Unlike unrestricted Damerau-Levenshtein
    distance, this executable model is not an unconditional metric; see
    Composition.DamerauComposition for the formal counterexample. *)
