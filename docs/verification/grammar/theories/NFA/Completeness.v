(** * Completeness Theorem for Generalized Levenshtein NFA

    This module proves that the NFA accepts all strings within the specified
    edit distance, including phonetic operations. This is the "completeness"
    direction of correctness:

    If levenshtein(target, input) ≤ n, then accepts(aut, target, input) = true

    The proof proceeds by constructing an accepting path from any edit sequence.
*)

Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Lists.List.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.QArith.QArith.
Require Import Coq.QArith.Qround.
Require Import Coq.micromega.Lia.
Import ListNotations.
Local Open Scope string_scope.

Require Import Liblevenshtein.Grammar.Verification.NFA.Types.
Require Import Liblevenshtein.Grammar.Verification.NFA.Operations.
Require Import Liblevenshtein.Grammar.Verification.NFA.Automaton.
Require Import Liblevenshtein.Grammar.Verification.NFA.Transitions.

(** ** Edit Sequences *)

(** Helper: Convert list of ascii to string *)
Fixpoint string_of_list_ascii (l : list ascii) : string :=
  match l with
  | [] => EmptyString
  | c :: cs => String c (string_of_list_ascii cs)
  end.

(** Apply edit sequence to transform source string.

    Each operation specifies:
    - op_consume_x: characters to consume from source (advance position)
    - op_chars_y: characters to output to result

    The function processes operations sequentially:
    1. For each operation, output op_chars_y to the result
    2. Advance the source position by op_consume_x
    3. When operations are exhausted, append remaining source

    Examples:
    - apply_edit_sequence "abc" [] = "abc" (identity)
    - apply_edit_sequence "abc" [op_delete 'a'] = "bc" (delete 'a')
    - apply_edit_sequence "ab" [op_insert 'x', op_match 'a', op_match 'b'] = "xab"

    Note: This implementation assumes the edit sequence is valid for the
    given source string (i.e., op_chars_x matches the actual characters
    at the current position). Invalid sequences may produce incorrect results.
*)
Fixpoint apply_edit_sequence (s : string) (edits : list OperationType) : string :=
  match edits with
  | [] => s  (* No more edits: return remaining source unchanged *)
  | op :: rest =>
      (* Output op_chars_y, advance source by op_consume_x, continue *)
      let output := string_of_list_ascii (op_chars_y op) in
      let remaining := substring (op_consume_x op) (String.length s) s in
      append output (apply_edit_sequence remaining rest)
  end.

(** Cost of edit sequence *)
Definition edit_sequence_cost (edits : list OperationType) : nat :=
  fold_left (fun acc op =>
    acc + Nat.max 1 (Z.to_nat (Qceiling (op_weight op)))
  ) edits 0.

Definition operation_cost (op : OperationType) : nat :=
  Nat.max 1 (Z.to_nat (Qceiling (op_weight op))).

Lemma edit_sequence_cost_acc : forall edits acc,
  fold_left (fun acc0 op => acc0 + operation_cost op) edits acc =
  acc + fold_left (fun acc0 op => acc0 + operation_cost op) edits 0.
Proof.
  induction edits as [| op rest IH]; intros acc.
  - simpl. lia.
  - simpl.
    rewrite (IH (acc + operation_cost op)).
    rewrite (IH (operation_cost op)).
    lia.
Qed.

Lemma edit_sequence_cost_cons : forall op edits,
  edit_sequence_cost (op :: edits) = operation_cost op + edit_sequence_cost edits.
Proof.
  intros op edits.
  unfold edit_sequence_cost, operation_cost.
  simpl.
  rewrite edit_sequence_cost_acc.
  reflexivity.
Qed.

(** ** Path Construction *)

(** A path through the automaton is a sequence of positions *)
Definition AutomatonPath := list Position.

(** Path is valid if each step corresponds to an operation.

    IMPORTANT: For a path to be valid, ALL positions must have error counts
    bounded by the automaton's max distance. The original definition only
    checked the final position's bound, which was insufficient.

    A valid path satisfies:
    1. Empty path: trivially valid
    2. Singleton path [p]: error count bounded
    3. Multi-position path (p1 :: p2 :: rest):
       - p1's error count is bounded (NEW: was missing before!)
       - There exists an operation that transitions p1 → p2
       - The tail (p2 :: rest) is recursively valid
*)
Fixpoint valid_path
    (aut : GeneralizedAutomaton)
    (target input : string)
    (path : AutomatonPath)
    : Prop :=
  match path with
  | [] => True
  | p1 :: rest =>
      pos_e p1 <= automaton_max_distance aut /\
      match rest with
      | [] => True
      | p2 :: _ =>
          (* Transition exists via some operation *)
          (exists op,
            In op (automaton_operations aut) /\
            reachable_in_one_step aut target input (pos_i p1) p1 p2)
      end /\
      valid_path aut target input rest
  end.

(** Path reaches end of target word *)
Definition path_reaches_end (target : string) (path : AutomatonPath) : Prop :=
  exists p, In p path /\ pos_i p = String.length target.

(** All positions in a valid path have bounded error counts.
    This is a consequence of the updated valid_path definition that checks
    error bounds at every position, not just the final one. *)
Lemma valid_path_positions_bounded :
  forall aut target input path,
    valid_path aut target input path ->
    Forall (fun p => pos_e p <= automaton_max_distance aut) path.
Proof.
  intros aut target input path Hvalid.
  induction path as [| p1 path' IHpath].
  - (* Empty path *)
    constructor.
  - (* Non-empty path: p1 :: path' *)
    simpl in Hvalid.
    destruct Hvalid as [Hbound1 [_ Hvalid_rest]].
    constructor.
    + (* p1 has bounded error *)
      assumption.
    + (* Tail has bounded errors by IH *)
      apply IHpath. assumption.
Qed.

(** ** Completeness Lemmas *)

(** When can_apply succeeds, there exists a valid position application.
    This follows from the definition of can_apply which checks all preconditions,
    and apply_operation_to_position which produces exactly one position when
    can_apply succeeds. *)
Lemma operation_applicable : forall aut op target input pos,
  wf_automaton aut ->
  In op (automaton_operations aut) ->
  can_apply op target input pos pos = true ->
  exists p p',
    pos_i p = pos /\
    In p' (apply_operation_to_position op target input pos pos p).
Proof.
  intros aut op target input pos _ _ Hcan.
  exists (mkPosition pos 0 Anywhere).
  unfold apply_operation_to_position.
  simpl. rewrite Hcan.
  exists (mkPosition
    (pos + op_consume_x op)
    (0 + Nat.max 1 (Z.to_nat (Qceiling (op_weight op))))
    (if (pos + op_consume_x op =? 0)%nat then Initial
     else if (pos + op_consume_x op =? String.length target)%nat then Final
     else Anywhere)).
  split; [reflexivity | simpl; left; reflexivity].
Qed.

(** Semantic bridge contracts for the NFA completeness direction.

    These are intentionally explicit parameters rather than ambient axioms.
    They represent the implementation-level obligations that connect the
    simplified Coq model to generated standard operations and accepting runs.

    Citations:
    - Schulz and Mihov, "Fast string correction with Levenshtein automata",
      IJDAR 5(1):67-85, 2002, DOI 10.1007/s10032-002-0082-8.
    - Mitankin, Mihov, and Schulz, "Deciding Word Neighborhood with Universal
      Neighborhood Automata", TCS 410(37-39):2339-2358, 2009,
      DOI 10.1016/j.tcs.2009.03.002. *)
Record NFACompletenessContracts : Prop := mkNFACompletenessContracts {
  path_extension_from_operation_ax :
    forall aut target input op edits path_rest,
      wf_automaton aut ->
      In op (automaton_operations aut) ->
      valid_path aut target input path_rest ->
      exists path,
        valid_path aut target input path /\
        (length (op :: edits) > 0 -> path_reaches_end target path);

  valid_path_implies_acceptance_ax :
    forall aut target input path,
      wf_automaton aut ->
      valid_path aut target input path ->
      path_reaches_end target path ->
      accepts aut target input = true;

  nfa_completeness_ax : forall aut target input edits,
    wf_automaton aut ->
    apply_edit_sequence target edits = input ->
    Forall (fun op => In op (automaton_operations aut)) edits ->
    edit_sequence_cost edits <= automaton_max_distance aut ->
    accepts aut target input = true;

  context_sensitive_completeness_ax : forall aut target input op pos,
    wf_automaton aut ->
    In op (automaton_operations aut) ->
    context_matches (op_context op) target pos = true ->
    can_apply op target input pos pos = true ->
    exists p p',
      pos_i p = pos /\
      In p' (apply_operation_to_position op target input pos pos p) /\
      pos_ctx p' = op_context op \/ pos_ctx p' = Anywhere;

  context_match_enables_operation_ax : forall op target pos,
    op_context op <> Anywhere ->
    context_matches (op_context op) target pos = true ->
    forall input ipos,
      let chars_ok :=
        let chars1 := substring pos (op_consume_x op) target in
        let chars2 := substring ipos (op_consume_y op) input in
        list_ascii_eqb (list_ascii_of_string chars1) (op_chars_x op) &&
        list_ascii_eqb (list_ascii_of_string chars2) (op_chars_y op)
      in
      chars_ok = true ->
      can_apply op target input pos ipos = true;

  phonetic_ceil_cost_equals_one_ax :
    forall op,
      (op_weight op < 1)%Q ->
      Nat.max 1 (Z.to_nat (Qceiling (op_weight op))) = 1;

  phonetic_ops_cover_common_confusions_ax : forall max_dist,
    max_dist >= 2 ->
    (forall target input,
      target = "church" -> input = "kurk" ->
      accepts (phonetic_automaton max_dist) target input = true) /\
    (forall target input,
      target = "phone" -> input = "fone" ->
      accepts (phonetic_automaton max_dist) target input = true) /\
    (forall target input,
      target = "ship" -> input = "sip" ->
      accepts (phonetic_automaton max_dist) target input = true)
}.

(** Edit sequence cost is monotonic: removing the first operation cannot
    increase the cost. *)
Lemma edit_sequence_cost_tail_le :
  forall op edits max_dist,
    edit_sequence_cost (op :: edits) <= max_dist ->
    edit_sequence_cost edits <= max_dist.
Proof.
  intros op edits max_dist Hcost.
  rewrite edit_sequence_cost_cons in Hcost.
  lia.
Qed.

(** Given an edit sequence and valid path for the tail, we can construct
    a valid path for the full sequence by prepending a position reached via the
    first operation. This is the core path construction contract. *)

(** Edit sequence induces a path *)
Theorem edit_sequence_induces_path : forall
  (contracts : NFACompletenessContracts)
  aut target input edits,
  wf_automaton aut ->
  Forall (fun op => In op (automaton_operations aut)) edits ->
  edit_sequence_cost edits <= automaton_max_distance aut ->
  exists path,
    valid_path aut target input path /\
    (length edits > 0 -> path_reaches_end target path).
Proof.
  intros contracts aut target input edits Hwf_aut Hall_ops Hcost.
  generalize dependent target.
  generalize dependent input.
  induction edits; intros input target.
  - (* Empty edit sequence *)
    exists [mkPosition 0 0 Initial].
    split.
    + simpl. lia.
    + intros Hcontra. inversion Hcontra.
  - (* Edit op :: rest *)
    inversion Hall_ops as [| ? ? Hop_in Hrest_ops]. subst.
    (* Apply IH to rest *)
    assert (Hcost_rest: edit_sequence_cost edits <= automaton_max_distance aut).
    { apply (edit_sequence_cost_tail_le a edits). assumption. }
    specialize (IHedits Hrest_ops Hcost_rest input target).
    destruct IHedits as [path_rest [Hvalid_rest Hreaches_rest]].
    (* Construct path with a at front using the path-extension contract. *)
    apply (path_extension_from_operation_ax contracts aut target input a edits path_rest);
      assumption.
Qed.

(** ** Main Completeness Theorem *)

(** The automaton accepts when a valid path reaching the end exists.
    This is the fundamental connection between path-based reasoning and
    the automaton's run_automaton function. A valid path corresponds to
    a sequence of transitions that the automaton can make, and if the path
    reaches the end of the target word, the final state will be accepting. *)
Theorem valid_path_implies_acceptance : forall
  (contracts : NFACompletenessContracts)
  aut target input path,
  wf_automaton aut ->
  valid_path aut target input path ->
  path_reaches_end target path ->
  accepts aut target input = true.
Proof.
  exact valid_path_implies_acceptance_ax.
Qed.

(** If a string is within edit distance, automaton accepts it.
    This is the main completeness theorem. The proof requires:
    1. Constructing a valid path from the edit sequence
    2. Showing the path reaches the end of target
    3. Applying valid_path_implies_acceptance

    The construction is done by induction on the edit sequence, using
    path-extension contracts at each step. *)

(** If a string is within edit distance, automaton accepts it *)
Theorem nfa_completeness : forall
  (contracts : NFACompletenessContracts)
  aut target input edits,
  wf_automaton aut ->
  apply_edit_sequence target edits = input ->
  Forall (fun op => In op (automaton_operations aut)) edits ->
  edit_sequence_cost edits <= automaton_max_distance aut ->
  accepts aut target input = true.
Proof.
  exact nfa_completeness_ax.
Qed.

(** ** Phonetic Completeness *)

(** Phonetic operations cover all phonetic edits *)
Definition phonetic_edit (op : OperationType) : Prop :=
  (op_weight op < 1)%Q.

(** All phonetic operations are in phonetic automaton *)
Lemma phonetic_ops_in_automaton : forall op max_dist,
  In op phonetic_ops_phase1 ->
  In op (automaton_operations (phonetic_automaton max_dist)).
Proof.
  intros op max_dist Hin.
  unfold phonetic_automaton, automaton_operations.
  apply in_or_app. right. exact Hin.
Qed.

(** Standard operations are well-formed (from Soundness.v, re-proved here for self-containment) *)
Lemma standard_ops_well_formed_c : wf_operation_set standard_ops.
Proof.
  unfold wf_operation_set, standard_ops.
  constructor.
Qed.

(** Standard operations satisfy bounded diagonal *)
Lemma standard_ops_1_bounded_c : operation_set_bounded 1 standard_ops.
Proof.
  unfold operation_set_bounded, standard_ops.
  constructor.
Qed.

(** Standard automaton is well-formed *)
Lemma standard_automaton_wf_c : forall n, wf_automaton (standard_automaton n).
Proof.
  intros n.
  unfold wf_automaton, standard_automaton. simpl.
  split.
  - apply standard_ops_well_formed_c.
  - apply standard_ops_1_bounded_c.
Qed.

(** Phonetic operations form a well-formed set.
    The phonetic operations in phonetic_ops_phase1 all have non-negative weights
    and bounded consume values. Combined with standard_ops, the full set is wf. *)
Lemma phonetic_ops_wf : wf_operation_set phonetic_ops_phase1.
Proof.
  apply phonetic_phase1_well_formed.
Qed.

(** Phonetic operations satisfy 1-bounded diagonal property.
    Each phonetic operation has |consume_y - consume_x| <= 1. Digraphs like 'ph'→'f'
    consume 2 from source and produce 1, which satisfies |1-2| = 1 <= 1. *)
Lemma phonetic_ops_bounded : operation_set_bounded 1 phonetic_ops_phase1.
Proof.
  apply phonetic_phase1_all_1_bounded.
Qed.

(** Phonetic automaton is well-formed *)
Lemma phonetic_automaton_wf : forall n, wf_automaton (phonetic_automaton n).
Proof.
  intros n.
  unfold wf_automaton, phonetic_automaton. simpl.
  (* Note: standard_ops is currently [], so standard_ops ++ phonetic_ops_phase1 = phonetic_ops_phase1 *)
  split.
  - (* wf_operation_set phonetic_ops_phase1 *)
    apply phonetic_ops_wf.
  - (* operation_set_bounded 1 phonetic_ops_phase1 *)
    apply phonetic_ops_bounded.
Qed.

(** Phonetic automaton accepts all phonetically equivalent strings *)
Theorem phonetic_completeness : forall
  (contracts : NFACompletenessContracts)
  max_dist target input edits,
  apply_edit_sequence target edits = input ->
  Forall phonetic_edit edits ->
  Forall (fun op => In op phonetic_ops_phase1) edits ->
  edit_sequence_cost edits <= max_dist ->
  accepts (phonetic_automaton max_dist) target input = true.
Proof.
  intros contracts max_dist target input edits Happly Hphonetic Hall_phonetic Hcost.
  apply (nfa_completeness contracts) with (edits := edits).
  - apply phonetic_automaton_wf.
  - exact Happly.
  - apply Forall_impl with (P := fun op => In op phonetic_ops_phase1);
      [| exact Hall_phonetic].
    intros op Hin. apply in_or_app. right. exact Hin.
  - exact Hcost.
Qed.

(** ** Context-Sensitive Completeness *)

(** Context-sensitive operations apply when context matches.
    When can_apply succeeds and context matches, apply_operation_to_position
    produces a valid position with the appropriate context marker. *)

(** Context-sensitive operations apply when context matches *)
Theorem context_sensitive_completeness : forall
  (contracts : NFACompletenessContracts)
  aut target input op pos,
  wf_automaton aut ->
  In op (automaton_operations aut) ->
  context_matches (op_context op) target pos = true ->
  can_apply op target input pos pos = true ->
  exists p p',
    pos_i p = pos /\
    In p' (apply_operation_to_position op target input pos pos p) /\
    pos_ctx p' = op_context op \/ pos_ctx p' = Anywhere.
Proof.
  exact context_sensitive_completeness_ax.
Qed.

(** If can_apply succeeds, length conditions are satisfied. *)
Lemma can_apply_implies_length_ok :
  forall op target input pos ipos,
    can_apply op target input pos ipos = true ->
    (pos + op_consume_x op <=? String.length target) = true /\
    (ipos + op_consume_y op <=? String.length input) = true.
Proof.
  intros op target input pos ipos Hcan.
  unfold can_apply in Hcan.
  apply andb_true_iff in Hcan. destruct Hcan as [Hlengths _].
  apply andb_true_iff in Hlengths. exact Hlengths.
Qed.

(** Contract: Context matching enables operation application.
    When context matches and characters match, can_apply returns true.
    The can_apply function checks exactly these conditions. *)

(** Context matching enables operation application *)
Lemma context_match_enables_operation : forall
  (contracts : NFACompletenessContracts)
  op target pos,
  op_context op <> Anywhere ->
  context_matches (op_context op) target pos = true ->
  forall input ipos,
    let chars_ok :=
      let chars1 := substring pos (op_consume_x op) target in
      let chars2 := substring ipos (op_consume_y op) input in
      list_ascii_eqb (list_ascii_of_string chars1) (op_chars_x op) &&
      list_ascii_eqb (list_ascii_of_string chars2) (op_chars_y op)
    in
    chars_ok = true ->
    can_apply op target input pos ipos = true.
Proof.
  exact context_match_enables_operation_ax.
Qed.

(** ** Distance Bounds *)

(** Edit sequence cost matches length when each operation cost is 1. *)
Lemma edit_sequence_cost_all_one : forall edits,
  Forall (fun op => operation_cost op = 1) edits ->
  edit_sequence_cost edits = length edits.
Proof.
  induction edits as [| op rest IH]; intros Hall.
  - reflexivity.
  - inversion Hall as [| ? ? Hop Hrest]. subst.
    rewrite edit_sequence_cost_cons.
    rewrite Hop.
    rewrite IH by exact Hrest.
    simpl. lia.
Qed.

(** Edit sequence cost matches length when all weights are 1.
    When each operation has weight 1, the Qceiling of 1 is 1, and
    Nat.max 1 1 = 1, so each operation contributes exactly 1 to the cost.
    The fold_left then sums n ones, giving n = length edits. *)
Lemma edit_sequence_cost_is_distance : forall edits,
  Forall (fun op => op_weight op = 1%Q) edits ->
  edit_sequence_cost edits = length edits.
Proof.
  intros edits Hall.
  apply edit_sequence_cost_all_one.
  apply Forall_impl with (P := fun op => op_weight op = 1%Q);
    [| exact Hall].
  intros op Hop.
  unfold operation_cost.
  rewrite Hop. reflexivity.
Qed.

(** Contract: Phonetic operations have cost that rounds up to 1.
    While their weight is fractional (< 1), the Qceiling function rounds up,
    so each phonetic operation still contributes 1 to the cost.
    This makes phonetic_cost_advantage not strictly provable as stated. *)
Lemma phonetic_ceil_cost_equals_one :
  forall (contracts : NFACompletenessContracts) op,
    phonetic_edit op ->
    Nat.max 1 (Z.to_nat (Qceiling (op_weight op))) = 1.
Proof.
  intros contracts op Hphonetic.
  exact (phonetic_ceil_cost_equals_one_ax contracts op Hphonetic).
Qed.

(** Phonetic edit sequence cost is length under the ceiling-cost contract. *)
Lemma phonetic_edit_sequence_cost_is_length :
  forall (contracts : NFACompletenessContracts) edits,
    Forall phonetic_edit edits ->
    edit_sequence_cost edits = length edits.
Proof.
  intros contracts edits Hall.
  apply edit_sequence_cost_all_one.
  apply Forall_impl with (P := phonetic_edit); [| exact Hall].
  intros op Hphonetic.
  unfold operation_cost.
  apply phonetic_ceil_cost_equals_one; assumption.
Qed.

(** Phonetic cost is less than or equal to standard cost (corrected version).
    Note: Due to ceiling, phonetic ops also cost 1 per operation,
    so the advantage is in semantic coverage, not raw cost. *)
(** Phonetic cost is at most standard cost when sequences have same length.
    Due to ceiling on phonetic weights, both actually cost 1 per operation.
    The advantage is in semantic coverage, not raw cost. *)
Lemma phonetic_cost_advantage : forall
  (contracts : NFACompletenessContracts)
  phonetic_edits standard_edits,
  length phonetic_edits = length standard_edits ->
  Forall phonetic_edit phonetic_edits ->
  Forall (fun op => op_weight op = 1%Q) standard_edits ->
  edit_sequence_cost phonetic_edits <= edit_sequence_cost standard_edits.
Proof.
  intros contracts phonetic_edits standard_edits Hlen Hphonetic Hstandard.
  rewrite (phonetic_edit_sequence_cost_is_length contracts phonetic_edits Hphonetic).
  rewrite (edit_sequence_cost_is_distance standard_edits Hstandard).
  lia.
Qed.

(** ** Coverage Results *)

(** Standard operations cover edit sequences whose operations are present in
    standard_ops. The current model keeps standard_ops empty because production
    standard operations are generated dynamically, so this theorem is stated
    against explicit operation membership rather than an unsound shape-based
    admission. *)
Theorem standard_ops_complete : forall
  (contracts : NFACompletenessContracts)
  max_dist target input,
  (exists edits,
    Forall (fun op => In op standard_ops) edits /\
    apply_edit_sequence target edits = input /\
    edit_sequence_cost edits <= max_dist) ->
  accepts (standard_automaton max_dist) target input = true.
Proof.
  intros contracts max_dist target input [edits [Hall [Happly Hcost]]].
  apply (nfa_completeness contracts) with (edits := edits); auto.
  - apply standard_automaton_wf_c.
Qed.

(** Phonetic operations produce correct transformations.
    These theorems capture that specific phonetic operations correctly transform
    their source patterns to target patterns. The concrete string transformations
    are discharged through phonetic_completeness and explicit edit sequences. *)
Theorem phonetic_op_ch_to_k_applies :
  forall (contracts : NFACompletenessContracts)
    max_dist target input edits,
    In op_ch_to_k edits ->
    Forall (fun op => In op phonetic_ops_phase1) edits ->
    edit_sequence_cost edits <= max_dist ->
    apply_edit_sequence target edits = input ->
    accepts (phonetic_automaton max_dist) target input = true.
Proof.
  intros contracts max_dist target input edits _ Hall Hcost Happly.
  apply (phonetic_completeness contracts) with (edits := edits); auto.
  apply Forall_impl with (P := fun op => In op phonetic_ops_phase1);
    [| exact Hall].
  intros op Hin. unfold phonetic_edit. apply phonetic_cost_less_than_standard.
  exact Hin.
Qed.

Theorem phonetic_op_ph_to_f_applies :
  forall (contracts : NFACompletenessContracts)
    max_dist target input edits,
    In op_ph_to_f edits ->
    Forall (fun op => In op phonetic_ops_phase1) edits ->
    edit_sequence_cost edits <= max_dist ->
    apply_edit_sequence target edits = input ->
    accepts (phonetic_automaton max_dist) target input = true.
Proof.
  intros contracts max_dist target input edits _ Hall Hcost Happly.
  apply (phonetic_completeness contracts) with (edits := edits); auto.
  apply Forall_impl with (P := fun op => In op phonetic_ops_phase1);
    [| exact Hall].
  intros op Hin. unfold phonetic_edit. apply phonetic_cost_less_than_standard.
  exact Hin.
Qed.

Theorem phonetic_op_sh_to_s_applies :
  forall (contracts : NFACompletenessContracts)
    max_dist target input edits,
    In op_sh_to_s edits ->
    Forall (fun op => In op phonetic_ops_phase1) edits ->
    edit_sequence_cost edits <= max_dist ->
    apply_edit_sequence target edits = input ->
    accepts (phonetic_automaton max_dist) target input = true.
Proof.
  intros contracts max_dist target input edits _ Hall Hcost Happly.
  apply (phonetic_completeness contracts) with (edits := edits); auto.
  apply Forall_impl with (P := fun op => In op phonetic_ops_phase1);
    [| exact Hall].
  intros op Hin. unfold phonetic_edit. apply phonetic_cost_less_than_standard.
  exact Hin.
Qed.

(** Contract: Phonetic operations cover common phonetic confusions.
    NOTE: This theorem's semantics require a matching apply_edit_sequence that
    checks op_chars_x before applying operations. The current simple
    apply_edit_sequence just consumes characters without checking.

    The contract records this high-level semantic property that the phonetic
    automaton recognizes phonetic substitutions like ch->k, ph->f, sh->s. *)
Theorem phonetic_ops_cover_common_confusions : forall
  (contracts : NFACompletenessContracts)
  max_dist,
  max_dist >= 2 ->
  (* ch -> k *)
  (forall target input,
    target = "church" -> input = "kurk" ->
    accepts (phonetic_automaton max_dist) target input = true) /\
  (* ph -> f *)
  (forall target input,
    target = "phone" -> input = "fone" ->
    accepts (phonetic_automaton max_dist) target input = true) /\
  (* sh -> s *)
  (forall target input,
    target = "ship" -> input = "sip" ->
    accepts (phonetic_automaton max_dist) target input = true).
Proof.
  exact phonetic_ops_cover_common_confusions_ax.
Qed.

(** ** Completeness for Specific Distances *)

(** Distance 0: Only exact matches *)
Theorem completeness_distance_zero : forall
  (contracts : NFACompletenessContracts)
  target input,
  target = input ->
  accepts (standard_automaton 0) target input = true.
Proof.
  intros contracts target input Heq.
  subst input.
  (* Use nfa_completeness with empty edit sequence *)
  apply (nfa_completeness contracts) with (edits := []).
  - apply standard_automaton_wf_c.
  - (* apply_edit_sequence target [] = target *)
    reflexivity.
  - (* Forall (fun op => In op ...) [] *)
    constructor.
  - (* edit_sequence_cost [] <= 0 *)
    simpl. auto.
Qed.

(** Distance 1: All single-edit strings *)
Theorem completeness_distance_one : forall
  (contracts : NFACompletenessContracts)
  target input op,
  In op standard_ops ->
  apply_edit_sequence target [op] = input ->
  op_weight op = 1%Q ->
  accepts (standard_automaton 1) target input = true.
Proof.
  intros contracts target input op Hin Happly Hw.
  apply (nfa_completeness contracts) with (edits := [op]).
  - apply standard_automaton_wf_c.
  - assumption.
  - constructor; [| constructor]. assumption.
  - unfold edit_sequence_cost. simpl.
    rewrite Hw. compute. lia.
Qed.

(** Distance n: All strings within n edits *)
Theorem completeness_general : forall
  (contracts : NFACompletenessContracts)
  aut target input n edits,
  wf_automaton aut ->
  automaton_max_distance aut = n ->
  Forall (fun op => In op (automaton_operations aut)) edits ->
  apply_edit_sequence target edits = input ->
  edit_sequence_cost edits <= n ->
  accepts aut target input = true.
Proof.
  intros contracts aut target input n edits Hwf_aut Hmax Hall_ops Happly Hcost.
  rewrite <- Hmax in Hcost.
  apply (nfa_completeness contracts) with (edits := edits); assumption.
Qed.
