(** * Downstream query-surface invariants

    This assumption-free model captures the reusable mathematical obligations
    behind the Phase-9 DFS prefix visitor, subsequence traversal, ranked values,
    bracket-language projection, and contextual-cost realignment guard.  The
    executable counterparts live in [tests/proptest_phase9_downstream.rs].
*)

From Stdlib Require Import Arith Bool Lia List PeanoNat.
Import ListNotations.

(** A subsequence may skip dictionary units but must retain query order. *)
Inductive subsequence {A : Type} : list A -> list A -> Prop :=
  | subsequence_nil : forall ys, subsequence [] ys
  | subsequence_keep : forall x xs ys,
      subsequence xs ys -> subsequence (x :: xs) (x :: ys)
  | subsequence_skip : forall x xs ys,
      subsequence xs ys -> subsequence xs (x :: ys).

Theorem subsequence_reflexive : forall (A : Type) (xs : list A),
  subsequence xs xs.
Proof.
  intros A xs; induction xs as [|x xs IH].
  - constructor.
  - constructor; exact IH.
Qed.

Theorem subsequence_survives_dictionary_suffix :
  forall (A : Type) (query dictionary suffix : list A),
    subsequence query dictionary ->
    subsequence query (dictionary ++ suffix).
Proof.
  intros A query dictionary suffix H; induction H; simpl.
  - constructor.
  - constructor; exact IHsubsequence.
  - apply subsequence_skip; exact IHsubsequence.
Qed.

(** Structural prefix reachability and terminal membership are intentionally
    separate predicates.  A prefix can reach an admitted term without itself
    being admitted as a result. *)
Definition prefix {A : Type} (candidate term : list A) : Prop :=
  exists suffix, term = candidate ++ suffix.

Theorem every_term_is_its_own_reachable_prefix :
  forall (A : Type) (term : list A), prefix term term.
Proof.
  intros A term; exists []; now rewrite app_nil_r.
Qed.

Example reachable_prefix_need_not_be_a_terminal :
  prefix ([] : list nat) [0] /\ ~ In [] [[0]].
Proof.
  split.
  - exists [0]; reflexivity.
  - simpl; intros [H | H]; [discriminate | contradiction].
Qed.

(** Kind erasure maps every opening token to zero and every closing token to
    one.  It can turn a substitution into a match, but never the reverse. *)
Definition substitution_cost (left right : nat) : nat :=
  if Nat.eq_dec left right then 0 else 1.

Definition erase_bracket_kind (kinds token : nat) : nat :=
  if token <? kinds then 0 else 1.

Definition projected_substitution_cost
    (kinds left right : nat) : nat :=
  substitution_cost (erase_bracket_kind kinds left)
                    (erase_bracket_kind kinds right).

Lemma projected_substitution_nonexpansive : forall kinds left right,
  projected_substitution_cost kinds left right <=
  substitution_cost left right.
Proof.
  intros kinds left right; unfold projected_substitution_cost, substitution_cost.
  destruct (Nat.eq_dec left right) as [Heq | Hneq].
  - subst; destruct (Nat.eq_dec (erase_bracket_kind kinds right)
                                (erase_bracket_kind kinds right)) as [_ | H].
    + lia.
    + exfalso; apply H; reflexivity.
  - destruct (Nat.eq_dec (erase_bracket_kind kinds left)
                         (erase_bracket_kind kinds right)); lia.
Qed.

Fixpoint raw_alignment_cost (alignment : list (nat * nat)) : nat :=
  match alignment with
  | [] => 0
  | (x, y) :: rest =>
      substitution_cost x y + raw_alignment_cost rest
  end.

Fixpoint projected_alignment_cost
    (kinds : nat) (alignment : list (nat * nat)) : nat :=
  match alignment with
  | [] => 0
  | (x, y) :: rest =>
      projected_substitution_cost kinds x y
      + projected_alignment_cost kinds rest
  end.

Theorem kind_erasure_never_increases_alignment_cost :
  forall kinds alignment,
    projected_alignment_cost kinds alignment <= raw_alignment_cost alignment.
Proof.
  intros kinds alignment; induction alignment as [|[left right] rest IH]; simpl.
  - lia.
  - pose proof (projected_substitution_nonexpansive kinds left right); lia.
Qed.

(** A bounded kind-sensitive Dyck DFA stores every stack word up to the depth
    cap, hence the geometric state-count recurrence. *)
Fixpoint bracket_stack_state_count (kinds depth : nat) : nat :=
  match depth with
  | 0 => 1
  | S previous =>
      bracket_stack_state_count kinds previous + kinds ^ (S previous)
  end.

Lemma bracket_state_count_grows_by_next_level : forall kinds depth,
  bracket_stack_state_count kinds (S depth) =
  bracket_stack_state_count kinds depth + kinds ^ (S depth).
Proof. reflexivity. Qed.

Lemma bracket_state_count_is_depth_monotone : forall kinds depth,
  bracket_stack_state_count kinds depth <=
  bracket_stack_state_count kinds (S depth).
Proof. intros; simpl; lia. Qed.

Example three_kinds_depth_ten_needs_88573_states :
  bracket_stack_state_count 3 10 = 88573.
Proof. vm_compute; reflexivity. Qed.

Example three_kinds_depth_ten_exceeds_public_guard :
  Nat.leb (bracket_stack_state_count 3 10) 4096 = false.
Proof. vm_compute; reflexivity. Qed.

(** Every entered DFS edge receives exactly one leave callback, including a
    rejected edge. *)
Definition visitor_balanced (enters leaves : nat) : Prop := enters = leaves.

Definition active_frame_count (enters leaves : nat) : nat := enters - leaves.

Fixpoint leaves_after_unwind (leaves depth : nat) : nat :=
  match depth with
  | 0 => leaves
  | S remaining => S (leaves_after_unwind leaves remaining)
  end.

Lemma leaves_after_unwind_adds_depth : forall leaves depth,
  leaves_after_unwind leaves depth = leaves + depth.
Proof.
  intros leaves depth; induction depth as [|depth IH]; simpl; lia.
Qed.

Lemma enter_then_leave_preserves_balance : forall enters leaves,
  visitor_balanced enters leaves ->
  visitor_balanced (S enters) (S leaves).
Proof. unfold visitor_balanced; intros; now f_equal. Qed.

Theorem unwinding_active_dfs_frames_restores_balance :
  forall enters leaves depth,
    leaves <= enters ->
    active_frame_count enters leaves = depth ->
    visitor_balanced enters (leaves_after_unwind leaves depth).
Proof.
  unfold active_frame_count, visitor_balanced.
  intros enters leaves depth Horder Hdepth.
  rewrite leaves_after_unwind_adds_depth.
  lia.
Qed.

(** BFS and DFS may enumerate in different orders.  No-pruning conformance is
    equality of result membership, never equality of output sequences. *)
Definition same_result_set (left right : list nat) : Prop :=
  forall candidate, In candidate left <-> In candidate right.

Theorem filtering_preserves_same_result_set :
  forall (keep : nat -> bool) left right,
    same_result_set left right ->
    same_result_set (filter keep left) (filter keep right).
Proof.
  unfold same_result_set; intros keep left right H candidate.
  repeat rewrite filter_In. rewrite H. tauto.
Qed.

(** MatchMode applies inclusive bounds to completed candidates.  Only the
    maximum is an automaton budget; the minimum is a terminal predicate. *)
Definition match_mode_accepts (minimum maximum distance : nat) : Prop :=
  minimum <= distance /\ distance <= maximum.

Theorem exact_match_mode_accepts_only_its_distance : forall exact distance,
  match_mode_accepts exact exact distance <-> distance = exact.
Proof. unfold match_mode_accepts; intros; lia. Qed.

Theorem range_match_mode_respects_automaton_budget :
  forall minimum maximum distance,
    match_mode_accepts minimum maximum distance -> distance <= maximum.
Proof. unfold match_mode_accepts; intros; lia. Qed.

(** Ranking is distance ascending, confidence descending, then lexical key
    ascending.  Mutual precedence therefore implies identical keys. *)
Definition ranked_before
    (left_distance left_confidence left_term
     right_distance right_confidence right_term : nat) : Prop :=
  left_distance < right_distance \/
  (left_distance = right_distance /\
   (left_confidence > right_confidence \/
    (left_confidence = right_confidence /\ left_term <= right_term))).

Theorem ranked_precedence_is_antisymmetric :
  forall ld lc lt rd rc rt,
    ranked_before ld lc lt rd rc rt ->
    ranked_before rd rc rt ld lc lt ->
    ld = rd /\ lc = rc /\ lt = rt.
Proof. unfold ranked_before; intros; lia. Qed.

(** The contextual iterator may realign two DP positions only when their index
    displacement, charged at the declared minimum non-zero edit cost, fits in
    the available slack. *)
Definition index_offset (left right : nat) : nat :=
  (left - right) + (right - left).

Definition contextual_realignment_safe
    (left right minimum slack : nat) : Prop :=
  index_offset left right * minimum <= slack.

Lemma index_offset_is_symmetric : forall left right,
  index_offset left right = index_offset right left.
Proof. unfold index_offset; intros; lia. Qed.

Theorem contextual_realignment_guard_is_symmetric : forall left right minimum slack,
  contextual_realignment_safe left right minimum slack <->
  contextual_realignment_safe right left minimum slack.
Proof.
  unfold contextual_realignment_safe; intros.
  rewrite index_offset_is_symmetric; tauto.
Qed.

Theorem zero_slack_forbids_distinct_positions : forall left right minimum,
  minimum > 0 ->
  contextual_realignment_safe left right minimum 0 ->
  left = right.
Proof.
  unfold contextual_realignment_safe, index_offset; intros.
  destruct (Nat.eq_dec left right); [assumption |].
  assert ((left - right) + (right - left) > 0) by lia.
  nia.
Qed.
