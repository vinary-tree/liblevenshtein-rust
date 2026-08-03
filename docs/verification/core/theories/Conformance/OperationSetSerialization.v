(** * OperationSet binary-persistence invariants

    This assumption-free model isolates the semantic and persistence
    obligations of [operation_set_binary.rs], [operation_set_protobuf.rs], and
    [operation_set_gzip.rs]. Operation applicability is explicit and independent
    of a diagnostic name. Bincode requires exact envelope consumption;
    protobuf requires a supported schema and pre-allocation scan; optional gzip
    requires one complete bounded member. Every path ends in semantic validation.
*)

From Stdlib Require Import Arith Bool Lia List PeanoNat.
Import ListNotations.

Inductive applicability : Type :=
  | apply_any
  | apply_equal
  | apply_adjacent_transpose
  | apply_listed (pairs : list (list nat * list nat)).

Record operation : Type := {
  operation_name : list nat;
  operation_input_consumption : nat;
  operation_output_consumption : nat;
  operation_cost : nat;
  operation_applicability : applicability
}.

Definition applicability_accepts
    (rule : applicability) (source target : list nat) : Prop :=
  match rule with
  | apply_any => True
  | apply_equal => source = target
  | apply_adjacent_transpose =>
      exists left right,
        source = [left; right] /\ target = [right; left]
  | apply_listed pairs => In (source, target) pairs
  end.

Definition operation_accepts
    (op : operation) (source target : list nat) : Prop :=
  applicability_accepts (operation_applicability op) source target.

Definition rename_operation (name : list nat) (op : operation) : operation :=
  {| operation_name := name;
     operation_input_consumption := operation_input_consumption op;
     operation_output_consumption := operation_output_consumption op;
     operation_cost := operation_cost op;
     operation_applicability := operation_applicability op |}.

Theorem diagnostic_rename_is_semantics_preserving :
  forall op name source target,
    operation_accepts (rename_operation name op) source target <->
    operation_accepts op source target.
Proof. reflexivity. Qed.

Theorem zero_cost_listed_operation_does_not_become_equality :
  forall name input_count output_count pairs source target,
    operation_accepts
      {| operation_name := name;
         operation_input_consumption := input_count;
         operation_output_consumption := output_count;
         operation_cost := 0;
         operation_applicability := apply_listed pairs |}
      source target <->
    In (source, target) pairs.
Proof. reflexivity. Qed.

Record binary_limits : Type := {
  maximum_payload_bytes : nat;
  maximum_operations : nat;
  maximum_name_bytes : nat;
  maximum_pairs : nat;
  maximum_pair_text_bytes : nat
}.

Record binary_envelope : Type := {
  envelope_magic : list nat;
  envelope_version : nat;
  envelope_flags : nat;
  declared_payload_bytes : nat;
  available_payload_bytes : nat;
  consumed_payload_bytes : nat;
  decoded_operations : nat;
  longest_name_bytes : nat;
  decoded_pairs : nat;
  decoded_pair_text_bytes : nat;
  semantic_validation_passed : bool
}.

Definition operation_set_magic : list nat := [76; 76; 69; 86; 79; 80; 83; 0].

Definition accepts_envelope
    (limits : binary_limits) (envelope : binary_envelope) : Prop :=
  envelope_magic envelope = operation_set_magic /\
  envelope_version envelope = 1 /\
  envelope_flags envelope = 0 /\
  declared_payload_bytes envelope = available_payload_bytes envelope /\
  consumed_payload_bytes envelope = declared_payload_bytes envelope /\
  declared_payload_bytes envelope <= maximum_payload_bytes limits /\
  decoded_operations envelope <= maximum_operations limits /\
  longest_name_bytes envelope <= maximum_name_bytes limits /\
  decoded_pairs envelope <= maximum_pairs limits /\
  decoded_pair_text_bytes envelope <= maximum_pair_text_bytes limits /\
  semantic_validation_passed envelope = true.

Theorem accepted_envelope_is_exact_and_bounded :
  forall limits envelope,
    accepts_envelope limits envelope ->
    envelope_magic envelope = operation_set_magic /\
    envelope_version envelope = 1 /\
    envelope_flags envelope = 0 /\
    consumed_payload_bytes envelope = available_payload_bytes envelope /\
    declared_payload_bytes envelope <= maximum_payload_bytes limits /\
    decoded_operations envelope <= maximum_operations limits /\
    decoded_pairs envelope <= maximum_pairs limits /\
    semantic_validation_passed envelope = true.
Proof.
  unfold accepts_envelope; intros limits envelope H.
  destruct H as [Hmagic [Hversion [Hflags [Hdeclared [Hconsumed
    [Hpayload [Hops [Hname [Hpairs [Htext Hsemantic]]]]]]]]]].
  repeat split; try assumption; lia.
Qed.

Theorem trailing_payload_bytes_are_rejected :
  forall limits envelope,
    available_payload_bytes envelope > declared_payload_bytes envelope ->
    ~ accepts_envelope limits envelope.
Proof. unfold accepts_envelope; intros; lia. Qed.

Theorem partial_payload_consumption_is_rejected :
  forall limits envelope,
    consumed_payload_bytes envelope < declared_payload_bytes envelope ->
    ~ accepts_envelope limits envelope.
Proof. unfold accepts_envelope; intros; lia. Qed.

Theorem oversized_operation_count_is_rejected :
  forall limits envelope,
    decoded_operations envelope > maximum_operations limits ->
    ~ accepts_envelope limits envelope.
Proof. unfold accepts_envelope; intros; lia. Qed.

Theorem invalid_semantics_are_rejected :
  forall limits envelope,
    semantic_validation_passed envelope = false ->
    ~ accepts_envelope limits envelope.
Proof.
  unfold accepts_envelope; intros limits envelope Hfalse Haccepts.
  destruct Haccepts as [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ Htrue]]]]]]]]]].
  congruence.
Qed.

(** ** Portable protobuf preflight

    Protobuf permits unknown fields, so its acceptance contract is not byte
    identity with one canonical input.  The executable decoder first performs
    a non-allocating scan of known allocation-bearing fields.  Prost is called
    only after every count is within policy, then semantic validation runs on
    the decoded object.
*)

Record protobuf_preflight : Type := {
  protobuf_wire_well_formed : bool;
  protobuf_supported_format : bool;
  protobuf_payload_bytes : nat;
  protobuf_operations : nat;
  protobuf_largest_name_bytes : nat;
  protobuf_largest_operation_pairs : nat;
  protobuf_total_pairs : nat;
  protobuf_pair_text_bytes : nat;
  protobuf_semantic_validation_passed : bool
}.

Definition accepts_protobuf
    (limits : binary_limits) (scan : protobuf_preflight) : Prop :=
  protobuf_wire_well_formed scan = true /\
  protobuf_supported_format scan = true /\
  protobuf_payload_bytes scan <= maximum_payload_bytes limits /\
  protobuf_operations scan <= maximum_operations limits /\
  protobuf_largest_name_bytes scan <= maximum_name_bytes limits /\
  protobuf_largest_operation_pairs scan <= maximum_pairs limits /\
  protobuf_total_pairs scan <= maximum_pairs limits /\
  protobuf_pair_text_bytes scan <= maximum_pair_text_bytes limits /\
  protobuf_semantic_validation_passed scan = true.

Theorem accepted_protobuf_is_preflight_bounded :
  forall limits scan,
    accepts_protobuf limits scan ->
    protobuf_wire_well_formed scan = true /\
    protobuf_supported_format scan = true /\
    protobuf_payload_bytes scan <= maximum_payload_bytes limits /\
    protobuf_operations scan <= maximum_operations limits /\
    protobuf_largest_operation_pairs scan <= maximum_pairs limits /\
    protobuf_total_pairs scan <= maximum_pairs limits /\
    protobuf_pair_text_bytes scan <= maximum_pair_text_bytes limits /\
    protobuf_semantic_validation_passed scan = true.
Proof.
  unfold accepts_protobuf; intros limits scan H.
  destruct H as [Hwire [Hformat [Hpayload [Hoperations [Hname
    [Hper_operation [Htotal [Htext Hsemantic]]]]]]]].
  repeat split; assumption.
Qed.

Theorem protobuf_over_limit_never_reaches_prost :
  forall limits scan,
    protobuf_operations scan > maximum_operations limits \/
    protobuf_largest_operation_pairs scan > maximum_pairs limits \/
    protobuf_total_pairs scan > maximum_pairs limits \/
    protobuf_pair_text_bytes scan > maximum_pair_text_bytes limits ->
    ~ accepts_protobuf limits scan.
Proof. unfold accepts_protobuf; intros; lia. Qed.

Theorem unsupported_protobuf_version_is_rejected :
  forall limits scan,
    protobuf_supported_format scan = false ->
    ~ accepts_protobuf limits scan.
Proof.
  unfold accepts_protobuf; intros limits scan Hfalse Haccepts.
  destruct Haccepts as [_ [Htrue _]].
  congruence.
Qed.

(** Weight storage is a fixed-width bit-vector field.  At the abstract wire
    boundary, encoding and decoding are inverse identities on all 64 bits. *)
Definition encode_weight_bits (bits : nat) : nat := bits.
Definition decode_weight_bits (bits : nat) : nat := bits.

Theorem protobuf_weight_bits_round_trip_exactly :
  forall bits,
    decode_weight_bits (encode_weight_bits bits) = bits.
Proof. reflexivity. Qed.

(** ** Optional single-member gzip wrapper *)

Record gzip_member : Type := {
  gzip_checksum_valid : bool;
  gzip_compressed_bytes : nat;
  gzip_consumed_compressed_bytes : nat;
  gzip_decompressed_bytes : nat
}.

Definition accepts_gzip
    (compressed_limit decompressed_limit supplied_bytes : nat)
    (member : gzip_member) (inner_accepts : bool) : Prop :=
  gzip_checksum_valid member = true /\
  gzip_compressed_bytes member <= compressed_limit /\
  gzip_decompressed_bytes member <= decompressed_limit /\
  gzip_consumed_compressed_bytes member = supplied_bytes /\
  inner_accepts = true.

Theorem accepted_gzip_is_single_bounded_and_inner_valid :
  forall compressed_limit decompressed_limit supplied_bytes member inner,
    accepts_gzip compressed_limit decompressed_limit supplied_bytes member inner ->
    gzip_checksum_valid member = true /\
    gzip_compressed_bytes member <= compressed_limit /\
    gzip_decompressed_bytes member <= decompressed_limit /\
    gzip_consumed_compressed_bytes member = supplied_bytes /\
    inner = true.
Proof. unfold accepts_gzip; auto. Qed.

Theorem trailing_compressed_data_is_rejected :
  forall compressed_limit decompressed_limit supplied_bytes member inner,
    gzip_consumed_compressed_bytes member < supplied_bytes ->
    ~ accepts_gzip compressed_limit decompressed_limit supplied_bytes member inner.
Proof. unfold accepts_gzip; intros; lia. Qed.

Theorem decompression_over_limit_is_rejected :
  forall compressed_limit decompressed_limit supplied_bytes member inner,
    gzip_decompressed_bytes member > decompressed_limit ->
    ~ accepts_gzip compressed_limit decompressed_limit supplied_bytes member inner.
Proof. unfold accepts_gzip; intros; lia. Qed.
