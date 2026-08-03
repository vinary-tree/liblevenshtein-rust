(** * Rust Subsumption and Insertion Conformance

    This trusted, assumption-free model mirrors the branch structure in:

    - [src/transducer/position.rs]
    - [src/transducer/position_f64.rs] for variant separation
    - [src/transducer/state.rs]
    - [src/transducer/state_f64.rs]

    The larger Automaton proof tree intentionally remains a partial research
    model. This module is the focused conformance boundary for the executable
    Phase-0 fixes: normal and transposition-in-progress positions have distinct
    continuation languages, and insertion reports retention directly rather
    than inferring it from a list-length change.
*)

From Stdlib Require Import Arith Bool List Nat Lia.
Import ListNotations.

(** A direct mathematical copy of the legacy Rust position representation. *)
Record RustPosition : Type := mkRustPosition {
  rust_index : nat;
  rust_errors : nat;
  rust_special : bool
}.

Definition rust_normal (i e : nat) : RustPosition :=
  mkRustPosition i e false.

Definition rust_pending (i e : nat) : RustPosition :=
  mkRustPosition i e true.

Definition rust_position_eqb (left right : RustPosition) : bool :=
  (rust_index left =? rust_index right) &&
  (rust_errors left =? rust_errors right) &&
  Bool.eqb (rust_special left) (rust_special right).

Lemma rust_position_eqb_eq : forall left right,
  rust_position_eqb left right = true <-> left = right.
Proof.
  intros [i e s] [j f t]. unfold rust_position_eqb. simpl.
  rewrite !andb_true_iff, !Nat.eqb_eq, Bool.eqb_true_iff.
  split.
  - intros [[Hi He] Hs]. subst. reflexivity.
  - intros H. injection H as Hi He Hs. subst. auto.
Qed.

Inductive RustAlgorithm : Type :=
  | RustStandard
  | RustTransposition
  | RustMergeSplit.

Definition rust_abs_diff (left right : nat) : nat :=
  if left <=? right then right - left else left - right.

Definition rust_standard_subsumes (left right : RustPosition) : bool :=
  (rust_errors left <=? rust_errors right) &&
  (rust_abs_diff (rust_index left) (rust_index right)
    <=? rust_errors right - rust_errors left).

(** Exact branch order of [Position::subsumes] for OSA. *)
Definition rust_transposition_subsumes (left right : RustPosition) : bool :=
  if negb (rust_errors left <=? rust_errors right) then false
  else if rust_special left then
    if rust_special right then rust_index left =? rust_index right
    else false
  else if rust_special right then false
  else rust_abs_diff (rust_index left) (rust_index right)
       <=? rust_errors right - rust_errors left.

(** Exact branch order of [Position::subsumes] for Merge-and-Split. *)
Definition rust_merge_split_subsumes
    (query_length : nat) (left right : RustPosition) : bool :=
  if negb (rust_errors left <=? rust_errors right) then false
  else if negb (Bool.eqb (rust_special left) (rust_special right)) then false
  else if query_length <? rust_index left then false
  else if rust_special left &&
          (query_length <=? rust_index left) &&
          (rust_index right <? query_length) then false
  else if negb (rust_errors left <? rust_errors right) then false
  else rust_index left =? rust_index right.

Definition rust_subsumes
    (algorithm : RustAlgorithm)
    (query_length : nat)
    (left right : RustPosition) : bool :=
  match algorithm with
  | RustStandard => rust_standard_subsumes left right
  | RustTransposition => rust_transposition_subsumes left right
  | RustMergeSplit => rust_merge_split_subsumes query_length left right
  end.

(** ** D2: variant separation *)

Theorem pending_never_subsumes_normal : forall i e j f,
  rust_transposition_subsumes (rust_pending i e) (rust_normal j f) = false.
Proof.
  intros i e j f. unfold rust_transposition_subsumes, rust_pending, rust_normal.
  simpl. destruct (e <=? f); reflexivity.
Qed.

Theorem normal_never_subsumes_pending : forall i e j f,
  rust_transposition_subsumes (rust_normal i e) (rust_pending j f) = false.
Proof.
  intros i e j f. unfold rust_transposition_subsumes, rust_pending, rust_normal.
  simpl. destruct (e <=? f); reflexivity.
Qed.

Theorem pending_pending_exact : forall i e j f,
  rust_transposition_subsumes (rust_pending i e) (rust_pending j f) =
  ((e <=? f) && (i =? j)).
Proof.
  intros i e j f. unfold rust_transposition_subsumes, rust_pending. simpl.
  destruct (e <=? f); reflexivity.
Qed.

Theorem normal_normal_matches_standard : forall i e j f,
  rust_transposition_subsumes (rust_normal i e) (rust_normal j f) =
  rust_standard_subsumes (rust_normal i e) (rust_normal j f).
Proof.
  intros i e j f.
  unfold rust_transposition_subsumes, rust_standard_subsumes, rust_normal.
  simpl. destruct (e <=? f); reflexivity.
Qed.

(** The realignment bound used by Standard and normal-normal OSA pruning is
    transitive. This is the arithmetic core needed for antichain coverage. *)
Lemma rust_abs_diff_triangle : forall a b c,
  rust_abs_diff a c <= rust_abs_diff a b + rust_abs_diff b c.
Proof.
  intros a b c. unfold rust_abs_diff.
  destruct (a <=? b) eqn:Hab, (b <=? c) eqn:Hbc, (a <=? c) eqn:Hac;
  apply Nat.leb_le in Hab || apply Nat.leb_gt in Hab;
  apply Nat.leb_le in Hbc || apply Nat.leb_gt in Hbc;
  apply Nat.leb_le in Hac || apply Nat.leb_gt in Hac;
  lia.
Qed.

Theorem rust_standard_subsumes_transitive : forall p q r,
  rust_standard_subsumes p q = true ->
  rust_standard_subsumes q r = true ->
  rust_standard_subsumes p r = true.
Proof.
  intros [i e s] [j f t] [k g u].
  unfold rust_standard_subsumes. simpl.
  rewrite !andb_true_iff, !Nat.leb_le.
  intros [Hef Hij] [Hfg Hjk].
  split; [lia|].
  pose proof (rust_abs_diff_triangle i j k). lia.
Qed.

(** ** D3: retention is not a length delta *)

Definition rust_covered
    (algorithm : RustAlgorithm)
    (query_length : nat)
    (position : RustPosition)
    (state : list RustPosition) : bool :=
  existsb
    (fun existing =>
       rust_position_eqb existing position ||
       rust_subsumes algorithm query_length existing position)
    state.

Definition rust_insert_model
    (algorithm : RustAlgorithm)
    (query_length : nat)
    (position : RustPosition)
    (state : list RustPosition) : bool * list RustPosition :=
  if rust_covered algorithm query_length position state then (false, state)
  else
    (true,
     position ::
       filter
         (fun existing =>
            negb (rust_subsumes algorithm query_length position existing))
         state).

Theorem insert_true_retains_position : forall algorithm query_length position state,
  fst (rust_insert_model algorithm query_length position state) = true ->
  In position (snd (rust_insert_model algorithm query_length position state)).
Proof.
  intros algorithm query_length position state.
  unfold rust_insert_model.
  destruct (rust_covered algorithm query_length position state);
    simpl; intros H; try discriminate; auto.
Qed.

Theorem insert_false_has_existing_cover : forall algorithm query_length position state,
  fst (rust_insert_model algorithm query_length position state) = false ->
  exists existing,
    In existing state /\
    (existing = position \/
     rust_subsumes algorithm query_length existing position = true).
Proof.
  intros algorithm query_length position state.
  unfold rust_insert_model.
  destruct (rust_covered algorithm query_length position state) eqn:Hcovered;
    simpl; intros Hresult; try discriminate.
  unfold rust_covered in Hcovered.
  apply existsb_exists in Hcovered.
  destruct Hcovered as [existing [Hin Hcovered]].
  apply orb_true_iff in Hcovered.
  destruct Hcovered as [Heq | Hsub].
  - apply rust_position_eqb_eq in Heq. exists existing. auto.
  - exists existing. auto.
Qed.

(** A newly retained representative can remove two existing representatives,
    shrinking the state. Therefore a length comparison cannot witness whether
    the closure worklist must process the new position. *)
Example retained_insert_can_shrink_state :
  rust_insert_model RustStandard 3 (rust_normal 1 0)
    [rust_normal 0 1; rust_normal 2 1] =
  (true, [rust_normal 1 0]).
Proof. reflexivity. Qed.

Example retained_insert_shrink_breaks_length_heuristic :
  let before := [rust_normal 0 1; rust_normal 2 1] in
  let result := rust_insert_model RustStandard 3 (rust_normal 1 0) before in
  fst result = true /\ length (snd result) < length before.
Proof. simpl. auto. Qed.
