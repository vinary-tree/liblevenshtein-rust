(** * PagingLaws: the interop node/arc paging contract

    Family FV obligation #5 (wave W3), the interop-level home. Both dictionary
    edge enumeration (`node_edges`) and WFST arc enumeration (`state_arcs`)
    expose the same paged shape across the ABI: given a full list and a
    (start, capacity) window, a provider fills at most `capacity` items from
    `start`, reports the FULL length as `out_total`, and returns how many it
    wrote in `out_written`. This module fixes that shape as

        page(items, start, capacity) = firstn capacity (skipn start items)

    and proves the six laws every conformant provider satisfies and every
    consumer may rely on. `ConsumerAcceptance.v` builds the single acceptance
    predicate on top of these; the producer side is refined per-repo
    (libdictenstein `AbiPagingProducerSpec.v` already discharges the
    dictionary case).

    Laws (registry: docs/verification/ABI_INVARIANTS.tsv, VT-PAGE-1..6):
      1 [VT-PAGE-1] written <= capacity;
      2 [VT-PAGE-2] written <= total - start (never past the end);
      3 [VT-PAGE-3] a full-capacity page exists exactly while start+capacity
        <= total;
      4 [VT-PAGE-4] start >= total yields the empty page (End);
      5 [VT-PAGE-5] out_total is independent of the paging window;
      6 [VT-PAGE-6] paging from 0 by capacity concatenates losslessly to the
        whole list, and the k-th page is the k-th consumer call.

    Executable mirrors: libdictenstein tests/ffi_resource_paging_proptest.rs
    and lling-llang tests/ffi_roundtrip_proptest.rs (arc paging).
*)

From Coq Require Import List Arith Lia.
Import ListNotations.

Section Paging.

Variable Item : Type.

Definition page (items : list Item) (start capacity : nat) : list Item :=
  firstn capacity (skipn start items).

Definition total (items : list Item) : nat := length items.

Definition written (items : list Item) (start capacity : nat) : nat :=
  length (page items start capacity).

(** ** VT-PAGE-1: a page never exceeds its capacity *)

Theorem written_le_capacity :
  forall items start capacity, written items start capacity <= capacity.
Proof.
  intros. unfold written, page. apply firstn_le_length.
Qed.

(** ** VT-PAGE-2: a page never runs past the end of the list *)

Theorem written_le_remaining :
  forall items start capacity,
    written items start capacity <= total items - start.
Proof.
  intros. unfold written, page, total.
  rewrite length_firstn, length_skipn. lia.
Qed.

(** ** VT-PAGE-3: full pages exist exactly while the window fits *)

Theorem full_page_iff_fits :
  forall items start capacity,
    written items start capacity = capacity <->
    capacity <= total items - start.
Proof.
  intros items start capacity. unfold written, page, total.
  rewrite length_firstn, length_skipn. lia.
Qed.

(** ** VT-PAGE-4: past-the-end starts yield the empty page (the End signal) *)

Theorem page_after_end_empty :
  forall items start capacity,
    total items <= start -> page items start capacity = [].
Proof.
  intros items start capacity Hle. unfold page, total in *.
  rewrite skipn_all2 by lia. apply firstn_nil.
Qed.

Theorem written_zero_iff_exhausted_or_no_capacity :
  forall items start capacity,
    written items start capacity = 0 <->
    (capacity = 0 \/ total items <= start).
Proof.
  intros items start capacity. unfold written, page, total.
  rewrite length_firstn, length_skipn. lia.
Qed.

(** ** VT-PAGE-5: the reported total does not depend on the window *)

Definition paged_call (items : list Item) (start capacity : nat)
  : list Item * nat :=
  (page items start capacity, total items).

Theorem total_window_independent :
  forall items s1 c1 s2 c2,
    snd (paged_call items s1 c1) = snd (paged_call items s2 c2).
Proof.
  reflexivity.
Qed.

(** ** VT-PAGE-6: lossless decomposition, and page k = consumer call k *)

Fixpoint pages_fuel (fuel capacity : nat) (items : list Item)
  : list (list Item) :=
  match fuel with
  | 0 => []
  | S fuel' =>
      match items with
      | [] => []
      | _ => firstn capacity items :: pages_fuel fuel' capacity (skipn capacity items)
      end
  end.

Definition pages (capacity : nat) (items : list Item) : list (list Item) :=
  pages_fuel (length items) capacity items.

Lemma pages_fuel_concat :
  forall fuel capacity items,
    1 <= capacity ->
    length items <= fuel ->
    concat (pages_fuel fuel capacity items) = items.
Proof.
  induction fuel as [| fuel IH]; simpl; intros capacity items Hcap Hlen.
  - destruct items; [reflexivity | simpl in Hlen; lia].
  - destruct items as [| head rest]; [reflexivity |].
    simpl in Hlen |- *.
    rewrite IH.
    + apply firstn_skipn.
    + exact Hcap.
    + rewrite length_skipn.
      destruct capacity as [| c]; [lia | simpl; lia].
Qed.

Theorem pages_concat_exact :
  forall capacity items,
    1 <= capacity -> concat (pages capacity items) = items.
Proof.
  intros. unfold pages. apply pages_fuel_concat; auto.
Qed.

Lemma pages_fuel_all_bounded :
  forall fuel capacity items,
    Forall (fun p : list Item => length p <= capacity)
           (pages_fuel fuel capacity items).
Proof.
  induction fuel as [| fuel IH]; simpl; intros capacity items.
  - constructor.
  - destruct items as [| head rest]; [constructor |].
    constructor; [apply firstn_le_length | apply IH].
Qed.

Theorem pages_all_bounded :
  forall capacity items,
    Forall (fun p => length p <= capacity) (pages capacity items).
Proof.
  intros. unfold pages. apply pages_fuel_all_bounded.
Qed.

Theorem nth_page_is_paged_call :
  forall capacity items k,
    1 <= capacity ->
    k * capacity < length items ->
    nth_error (pages capacity items) k
    = Some (page items (k * capacity) capacity).
Proof.
  intros capacity items k Hcap.
  assert (General :
    forall fuel items k,
      length items <= fuel ->
      k * capacity < length items ->
      nth_error (pages_fuel fuel capacity items) k
      = Some (page items (k * capacity) capacity)).
  { induction fuel as [| fuel IH]; simpl; intros xs k' Hfuel Hin.
    - lia.
    - destruct xs as [| head rest]; [simpl in Hin; lia |].
      simpl in Hfuel, Hin.
      destruct k' as [| k''].
      + simpl. unfold page. simpl. reflexivity.
      + simpl. rewrite IH.
        * unfold page. rewrite skipn_skipn.
          replace (S k'' * capacity) with (k'' * capacity + capacity) by lia.
          rewrite Nat.add_comm. reflexivity.
        * rewrite length_skipn.
          destruct capacity as [| c]; [lia | simpl; lia].
        * rewrite length_skipn.
          assert (Hstep : S k'' * capacity = capacity + k'' * capacity) by lia.
          destruct capacity as [| c]; [lia | simpl in Hstep |- *; lia]. }
  intros Hin. unfold pages. apply General; auto.
Qed.

End Paging.
