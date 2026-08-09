(** * CursorSnapshotSemantics: the consumer-side snapshot boundary law

    Family FV obligation #4 (wave W3), the consumer half of the query-start
    snapshot boundary. A liblevenshtein cursor captures a dictionary revision
    at query start (`QueryCursor`, src/bindings.rs) and then lazily enumerates
    the matches visible in THAT captured revision, forwarding any provider
    fault on the current-or-next advance. This module proves the consumer-side
    laws over an abstract captured snapshot, importing the producer's
    immutability guarantee as a premise (per the family's assumption-import
    discipline — the producer home is libdictenstein
    `formal-verification/tla+/AbiProducerSnapshot.tla`, obligation #10, which
    proves a captured revision is immutable; here that is the STRUCTURAL fact
    that the cursor enumerates a fixed list).

    Laws (registry: docs/verification/ABI_INVARIANTS.tsv):
      VT-SNAP-1  emitted ⊆ captured: every match a cursor yields is a member
                 of the captured revision (no fabricated matches);
      VT-SNAP-2  completeness (fault-free run): a cursor drained without a
                 fault yields exactly the captured revision, once each;
      VT-SNAP-3  fault forwarding: once a provider fault is latched, the next
                 advance surfaces it and the enumeration never fabricates a
                 further match past it (the fault-channel `record_fault` /
                 `take_fault` contract, src/bindings.rs).

    The lifetime half (a cursor may outlive the transducer and the dictionary
    handle) is a corollary of the interop lifecycle model
    (docs/verification/tla/AbiResourceLifecycle.tla, VT-LIFE): the cursor owns
    a retain on the captured snapshot, so the captured list stays valid — that
    list is exactly the fixed [captured] this model enumerates.

    Executable mirrors: tests/{query_start,binding,ffi_resource}_snapshot_
    semantics.rs (the captured-revision-vs-oracle checks) and
    tests/ffi_provider_fault_injection.rs (fault forwarding).
*)

From Coq Require Import List Arith Bool.
Import ListNotations.

Section Cursor.

(** A match is an opaque element of the captured revision. *)
Variable Match : Type.

(** The cursor state: the still-unemitted tail of the captured revision, and
    an optional latched fault (modeled as a unit — its identity is irrelevant
    to the laws; what matters is presence). *)
Record Cursor := mkCursor {
  remaining : list Match;
  faulted : bool;
}.

(** The outcome of one advance. *)
Inductive Step :=
  | Yielded (m : Match) (next : Cursor)
  | Exhausted
  | Faulted (next : Cursor).

(** One advance. A latched fault is surfaced FIRST (current-or-next-advance
    forwarding) and clears; only then does the enumeration proceed. This
    mirrors `next_match`, which returns the fault before and after producing a
    match and never fabricates one while a fault is pending. *)
Definition advance (c : Cursor) : Step :=
  if faulted c then
    Faulted (mkCursor (remaining c) false)
  else
    match remaining c with
    | [] => Exhausted
    | m :: rest => Yielded m (mkCursor rest false)
    end.

(** The initial cursor over a captured revision. *)
Definition start (captured : list Match) : Cursor :=
  mkCursor captured false.

(** Latch a provider fault (idempotent: only the first sticks, exactly like
    `record_fault`'s `if fault.is_none()`). *)
Definition record_fault (c : Cursor) : Cursor :=
  mkCursor (remaining c) true.

(** Drain a fault-free cursor to the list of matches it yields, fuelled by the
    remaining length. *)
Fixpoint drain (fuel : nat) (c : Cursor) : list Match :=
  match fuel with
  | 0 => []
  | S fuel' =>
      match advance c with
      | Yielded m next => m :: drain fuel' next
      | Exhausted => []
      | Faulted _ => []
      end
  end.

Definition drain_all (c : Cursor) : list Match :=
  drain (length (remaining c)) c.

(** ** VT-SNAP-1: emitted ⊆ captured *)

Lemma drain_is_prefix_of_remaining :
  forall fuel c,
    faulted c = false ->
    exists rest, remaining c = drain fuel c ++ rest.
Proof.
  induction fuel as [| fuel IH]; intros c Hclean; simpl.
  - exists (remaining c). reflexivity.
  - unfold advance. rewrite Hclean.
    destruct (remaining c) as [| m rest] eqn:Hrem.
    + exists []. rewrite app_nil_r. reflexivity.
    + destruct (IH (mkCursor rest false) eq_refl) as [tail Htail].
      simpl in Htail. exists tail. simpl. rewrite <- Htail. reflexivity.
Qed.

Theorem emitted_subset_captured :
  forall captured m,
    In m (drain_all (start captured)) -> In m captured.
Proof.
  intros captured m Hin.
  unfold drain_all, start in Hin. simpl in Hin.
  destruct (drain_is_prefix_of_remaining (length captured) (start captured) eq_refl)
    as [rest Hrest].
  unfold start in Hrest. simpl in Hrest.
  rewrite Hrest. apply in_or_app. left. exact Hin.
Qed.

(** ** VT-SNAP-2: completeness of a fault-free run *)

Theorem fault_free_run_is_exactly_captured :
  forall captured,
    drain_all (start captured) = captured.
Proof.
  intros captured. unfold drain_all, start. simpl.
  assert (General : forall items,
    drain (length items) (mkCursor items false) = items).
  { induction items as [| x xs IH]; simpl.
    - reflexivity.
    - unfold advance. simpl. rewrite IH. reflexivity. }
  apply General.
Qed.

(** ** VT-SNAP-3: fault forwarding *)

(** A latched fault is surfaced by the very next advance. *)
Theorem latched_fault_surfaces_next :
  forall c,
    exists next, advance (record_fault c) = Faulted next.
Proof.
  intro c. unfold advance, record_fault. simpl. eauto.
Qed.

(** No match is fabricated while a fault is pending: an advance on a faulted
    cursor never yields. *)
Theorem no_match_past_a_fault :
  forall c m next,
    faulted c = true -> advance c <> Yielded m next.
Proof.
  intros c m next Hfault. unfold advance. rewrite Hfault. discriminate.
Qed.

(** Draining a faulted cursor yields nothing — the enumeration halts at the
    fault rather than resuming with captured items. *)
Theorem drain_halts_on_fault :
  forall fuel c,
    faulted c = true -> drain fuel c = [].
Proof.
  intros fuel c Hfault. destruct fuel; simpl; [reflexivity |].
  unfold advance. rewrite Hfault. reflexivity.
Qed.

(** record_fault is idempotent: a second fault does not change the latched
    state (matching `if fault.is_none()`). *)
Theorem record_fault_idempotent :
  forall c, record_fault (record_fault c) = record_fault c.
Proof.
  intro c. unfold record_fault. reflexivity.
Qed.

End Cursor.
