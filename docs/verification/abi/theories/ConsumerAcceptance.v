(** * ConsumerAcceptance: the single paging-acceptance predicate

    Family FV obligation #5, consumer half (wave W3). Finding LLEV-B8 / F3
    recorded that the three consumers each validated a `node_edges` /
    `state_arcs` reply with a slightly different ad-hoc predicate (llev
    checked `total < start + written`; lling added `offset > total` and an
    in-loop progress check; duallity used saturating adds). This module
    defines ONE predicate, [accepts], proves it accepts exactly the
    law-abiding replies of [PagingLaws], and gives the rejection lemmas the
    adversarial provider fixtures mirror — so all three consumers can align
    to a single proven check.

    A reply is the raw triple a provider writes: `(written, out_total,
    page_len)` for a call at `(start, capacity)` over a snapshot whose true
    item list is `items`. The consumer cannot see `items`; it sees only the
    triple and its own `(start, capacity)`. [accepts] is the strongest check
    expressible from what the consumer sees that no honest provider fails.

    Registry: docs/verification/ABI_INVARIANTS.tsv VT-PAGE-1..6 (this file is
    the acceptance-soundness/completeness home; adversarial fixtures mirror
    the reject_* lemmas). Executable mirror: the harmonized `expanded_edges`
    check in src/bindings.rs plus the sibling consumers, each pinned by its
    repo's paging proptest.
*)

From Coq Require Import List Arith Lia.
From Liblevenshtein.Abi Require Import PagingLaws.
Import ListNotations.

Section Acceptance.

Variable Item : Type.

(** What the consumer observes about one paged reply. *)
Record Reply := mkReply {
  reply_written : nat;   (* out_written *)
  reply_total : nat;     (* out_total *)
  reply_page_len : nat   (* items actually delivered in the page buffer *)
}.

(** The honest reply a conformant provider produces for [items] at
    [(start, capacity)]. *)
Definition honest_reply (items : list Item) (start capacity : nat) : Reply :=
  {| reply_written := written Item items start capacity;
     reply_total := total Item items;
     reply_page_len := length (page Item items start capacity) |}.

(** The acceptance predicate, from the consumer's viewpoint. It must hold for
    every honest reply and reject the misbehaviors the fixtures inject:
      - the page buffer holds exactly [written] items;
      - [written] never exceeds the requested [capacity];
      - [written] never exceeds what remains past [start] in [total];
      - progress: an empty page is legal only when nothing remains
        (start >= total) or no capacity was offered (capacity = 0);
      - the claimed total is bounded — a consumer that preallocates on
        [total] must not be driven to allocate more than the snapshot could
        possibly hold, so [total] may not exceed a caller-supplied ceiling
        [max_total] (the number of items the consumer has budget for). This
        is the claimed-total sanity bound that closes the preallocation
        finding. *)
Definition accepts
    (start capacity max_total : nat) (reply : Reply) : Prop :=
  reply_page_len reply = reply_written reply /\
  reply_written reply <= capacity /\
  reply_written reply <= reply_total reply - start /\
  (reply_written reply = 0 -> capacity = 0 \/ reply_total reply <= start) /\
  reply_total reply <= max_total.

(** ** Soundness: every honest reply is accepted (given a true ceiling). *)

Theorem accepts_honest :
  forall items start capacity max_total,
    total Item items <= max_total ->
    accepts start capacity max_total (honest_reply items start capacity).
Proof.
  intros items start capacity max_total Hceiling.
  unfold accepts, honest_reply. simpl.
  repeat split.
  - apply written_le_capacity.
  - apply written_le_remaining.
  - intro Hzero.
    apply (written_zero_iff_exhausted_or_no_capacity Item items start capacity).
    exact Hzero.
  - exact Hceiling.
Qed.

(** ** Completeness of rejection: each injected misbehavior is rejected.

    These are exactly the fixtures the adversarial providers in the three
    repos emit; the lemma names track the fixture names. *)

(** written > capacity (over-fill). *)
Theorem rejects_overfill :
  forall start capacity max_total reply,
    reply_written reply > capacity ->
    ~ accepts start capacity max_total reply.
Proof.
  intros start capacity max_total reply Hover [_ [Hle _]]. lia.
Qed.

(** written > total - start (past the end). *)
Theorem rejects_past_end :
  forall start capacity max_total reply,
    reply_written reply > reply_total reply - start ->
    ~ accepts start capacity max_total reply.
Proof.
  intros start capacity max_total reply Hpast [_ [_ [Hle _]]]. lia.
Qed.

(** page buffer length disagrees with the written count. *)
Theorem rejects_page_len_mismatch :
  forall start capacity max_total reply,
    reply_page_len reply <> reply_written reply ->
    ~ accepts start capacity max_total reply.
Proof.
  intros start capacity max_total reply Hne [Heq _]. contradiction.
Qed.

(** empty page while items remain and capacity was offered (no progress). *)
Theorem rejects_stalled_progress :
  forall start capacity max_total reply,
    reply_written reply = 0 ->
    capacity <> 0 ->
    reply_total reply > start ->
    ~ accepts start capacity max_total reply.
Proof.
  intros start capacity max_total reply Hzero Hcap Hremain
         [_ [_ [_ [Hprogress _]]]].
  destruct (Hprogress Hzero) as [Hc | Ht]; lia.
Qed.

(** inflated total beyond the consumer's ceiling (the preallocation-abort
    vector, LLEV-B8 / the expanded_edges claimed-total finding). *)
Theorem rejects_inflated_total :
  forall start capacity max_total reply,
    reply_total reply > max_total ->
    ~ accepts start capacity max_total reply.
Proof.
  intros start capacity max_total reply Hinflate [_ [_ [_ [_ Hle]]]]. lia.
Qed.

(** ** Decidability: the predicate is a runnable check.

    This is what the Rust consumers evaluate; [accepts_reflect] certifies the
    boolean the implementation runs equals the proposition proved above. *)

Definition accepts_dec
    (start capacity max_total : nat) (reply : Reply) : bool :=
  Nat.eqb (reply_page_len reply) (reply_written reply) &&
  Nat.leb (reply_written reply) capacity &&
  Nat.leb (reply_written reply) (reply_total reply - start) &&
  (implb (Nat.eqb (reply_written reply) 0)
         (Nat.eqb capacity 0 || Nat.leb (reply_total reply) start)) &&
  Nat.leb (reply_total reply) max_total.

Theorem accepts_reflect :
  forall start capacity max_total reply,
    accepts_dec start capacity max_total reply = true <->
    accepts start capacity max_total reply.
Proof.
  intros start capacity max_total reply.
  unfold accepts_dec, accepts.
  rewrite !Bool.andb_true_iff.
  rewrite Nat.eqb_eq, !Nat.leb_le.
  split.
  - intros [[[[Hlen Hcap] Hrem] Hprog] Htot].
    repeat split; try assumption.
    + intro Hzero.
      rewrite Hzero in Hprog. simpl in Hprog.
      apply Bool.orb_true_iff in Hprog as [Hc | Ht].
      * left. apply Nat.eqb_eq. exact Hc.
      * right. apply Nat.leb_le. exact Ht.
  - intros [Hlen [Hcap [Hrem [Hprog Htot]]]].
    repeat split; try assumption.
    + destruct (Nat.eqb (reply_written reply) 0) eqn:Hzb; simpl; [| reflexivity].
      apply Nat.eqb_eq in Hzb.
      destruct (Hprog Hzb) as [Hc | Ht].
      * rewrite (proj2 (Nat.eqb_eq _ _) Hc). reflexivity.
      * rewrite (proj2 (Nat.leb_le _ _) Ht). apply Bool.orb_true_r.
Qed.

End Acceptance.
