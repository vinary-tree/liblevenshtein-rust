(** * Executable byte-parser refinement for OperationSet persistence

    This file models the untrusted byte boundary used by
    [operation_set_binary.rs], [operation_set_protobuf.rs], and
    [operation_set_gzip.rs].  Unlike the higher-level acceptance predicates in
    [OperationSetSerialization], the functions below consume concrete byte
    lists.  They decode the bincode envelope's little-endian header and the
    protobuf preflight scanner's varints, keys, wire values, nested messages,
    and allocation-bearing counters.

    DEFLATE is intentionally not reimplemented here.  The gzip adapter starts
    from an observation supplied by [flate2] and verifies only the crate-owned
    boundary: gzip identity, complete compressed-input consumption, bounded
    output, and successful inner binary decoding. *)

From Stdlib Require Import Arith Bool Lia List PeanoNat.
Require Import Liblevenshtein.Core.Conformance.OperationSetSerialization.
Import ListNotations.

Definition byte_well_formed (value : nat) : Prop := value < 256.
Definition bytes_well_formed (bytes : list nat) : Prop :=
  Forall byte_well_formed bytes.

Definition bytes_well_formedb (bytes : list nat) : bool :=
  forallb (fun value => value <? 256) bytes.

Lemma bytes_well_formedb_reflects_bytes : forall bytes,
  bytes_well_formedb bytes = true -> bytes_well_formed bytes.
Proof.
  intros bytes Hbytes; unfold bytes_well_formedb in Hbytes.
  rewrite forallb_forall in Hbytes.
  apply Forall_forall; intros value Hin.
  apply Nat.ltb_lt, Hbytes; assumption.
Qed.

Fixpoint little_endian_value (bytes : list nat) : nat :=
  match bytes with
  | [] => 0
  | byte :: rest => byte + 256 * little_endian_value rest
  end.

Definition parser_operation_set_magic : list nat := [76; 76; 69; 86; 79; 80; 83; 0].
Definition bincode_header_bytes : nat := 20.

Record parsed_bincode_header : Type := {
  parsed_bincode_magic : list nat;
  parsed_bincode_version : nat;
  parsed_bincode_flags : nat;
  parsed_bincode_declared : nat;
  parsed_bincode_payload : list nat
}.

Definition parse_bincode_header (bytes : list nat)
    : option parsed_bincode_header :=
  if bincode_header_bytes <=? length bytes then
    Some
      {| parsed_bincode_magic := firstn 8 bytes;
         parsed_bincode_version := little_endian_value (firstn 2 (skipn 8 bytes));
         parsed_bincode_flags := little_endian_value (firstn 2 (skipn 10 bytes));
         parsed_bincode_declared := little_endian_value (firstn 8 (skipn 12 bytes));
         parsed_bincode_payload := skipn bincode_header_bytes bytes |}
  else None.

Definition bincode_byte_preflight
    (payload_limit : nat) (bytes : list nat) : bool :=
  if bytes_well_formedb bytes then
    match parse_bincode_header bytes with
    | None => false
    | Some header =>
        if list_eq_dec Nat.eq_dec
             (parsed_bincode_magic header) parser_operation_set_magic then
          (parsed_bincode_version header =? 1) &&
          (parsed_bincode_flags header =? 0) &&
          (parsed_bincode_declared header =?
            length (parsed_bincode_payload header)) &&
          (parsed_bincode_declared header <=? payload_limit)
        else false
    end
  else false.

Lemma parse_bincode_header_is_a_real_prefix : forall bytes header,
  parse_bincode_header bytes = Some header ->
  bincode_header_bytes <= length bytes /\
  parsed_bincode_magic header = firstn 8 bytes /\
  parsed_bincode_version header =
    little_endian_value (firstn 2 (skipn 8 bytes)) /\
  parsed_bincode_flags header =
    little_endian_value (firstn 2 (skipn 10 bytes)) /\
  parsed_bincode_declared header =
    little_endian_value (firstn 8 (skipn 12 bytes)) /\
  parsed_bincode_payload header = skipn bincode_header_bytes bytes.
Proof.
  intros bytes header Hparse; unfold parse_bincode_header in Hparse.
  destruct (bincode_header_bytes <=? length bytes) eqn:Hlength;
    try discriminate.
  inversion Hparse; subst; simpl.
  apply Nat.leb_le in Hlength; repeat split; reflexivity || assumption.
Qed.

Theorem accepted_bincode_bytes_have_the_exact_runtime_envelope :
  forall payload_limit bytes,
    bincode_byte_preflight payload_limit bytes = true ->
    exists header,
      bytes_well_formed bytes /\
      parse_bincode_header bytes = Some header /\
      parsed_bincode_magic header = parser_operation_set_magic /\
      parsed_bincode_version header = 1 /\
      parsed_bincode_flags header = 0 /\
      parsed_bincode_declared header = length (parsed_bincode_payload header) /\
      parsed_bincode_declared header <= payload_limit /\
      length bytes = bincode_header_bytes + parsed_bincode_declared header.
Proof.
  intros payload_limit bytes Haccept.
  unfold bincode_byte_preflight in Haccept.
  destruct (bytes_well_formedb bytes) eqn:Hbytes; try discriminate.
  destruct (parse_bincode_header bytes) as [header|] eqn:Hparse;
    try discriminate.
  destruct (list_eq_dec Nat.eq_dec
    (parsed_bincode_magic header) parser_operation_set_magic) as [Hmagic|Hmagic];
    try discriminate.
  apply andb_true_iff in Haccept as [Haccept Hlimit].
  apply andb_true_iff in Haccept as [Haccept Hdeclared].
  apply andb_true_iff in Haccept as [Hversion Hflags].
  apply Nat.eqb_eq in Hversion, Hflags, Hdeclared.
  apply Nat.leb_le in Hlimit.
  assert (Hwell_formed : bytes_well_formed bytes).
  { apply bytes_well_formedb_reflects_bytes; exact Hbytes. }
  destruct (parse_bincode_header_is_a_real_prefix bytes header Hparse)
    as [Hheader [_ [_ [_ [_ Hpayload]]]]].
  assert (Hexact_length :
    length bytes = bincode_header_bytes + parsed_bincode_declared header).
  { rewrite Hpayload in Hdeclared.
    rewrite length_skipn in Hdeclared; lia. }
  exists header; repeat split; try assumption.
Qed.

(** ** Protobuf wire cursor *)

Definition maximum_u64 : nat := 2 ^ 64 - 1.
Definition maximum_protobuf_tag : nat := 2 ^ 29 - 1.

Record varint_result : Type := {
  parsed_varint_value : nat;
  parsed_varint_consumed : nat;
  parsed_varint_rest : list nat
}.

Fixpoint parse_varint_aux
    (fuel place : nat) (bytes : list nat) : option varint_result :=
  match fuel, bytes with
  | 0, _ => None
  | _, [] => None
  | S remaining_fuel, byte :: rest =>
      if byte <? 256 then
        if byte <? 128 then
          Some
            {| parsed_varint_value := byte * place;
               parsed_varint_consumed := 1;
               parsed_varint_rest := rest |}
        else
          match parse_varint_aux remaining_fuel (place * 128) rest with
          | None => None
          | Some suffix =>
              Some
                {| parsed_varint_value :=
                     (byte mod 128) * place + parsed_varint_value suffix;
                   parsed_varint_consumed := S (parsed_varint_consumed suffix);
                   parsed_varint_rest := parsed_varint_rest suffix |}
          end
      else None
  end.

Definition parse_varint (bytes : list nat) : option varint_result :=
  match parse_varint_aux 10 1 bytes with
  | Some result =>
      if parsed_varint_value result <=? maximum_u64 then Some result else None
  | None => None
  end.

Lemma parse_varint_aux_consumes_a_prefix : forall fuel place bytes result,
  parse_varint_aux fuel place bytes = Some result ->
  length bytes = parsed_varint_consumed result + length (parsed_varint_rest result) /\
  1 <= parsed_varint_consumed result /\
  parsed_varint_consumed result <= fuel.
Proof.
  induction fuel as [|fuel IH]; intros place bytes result Hparse;
    destruct bytes as [|byte rest]; simpl in Hparse; try discriminate.
  destruct (byte <? 256) eqn:Hbyte; try discriminate.
  destruct (byte <? 128) eqn:Hterminal.
  - inversion Hparse; subst; simpl; lia.
  - destruct (parse_varint_aux fuel (place * 128) rest)
      as [suffix|] eqn:Hsuffix; try discriminate.
    inversion Hparse; subst; simpl.
    destruct (IH (place * 128) rest suffix Hsuffix)
      as [Hlength [Hpositive Hfuel]].
    lia.
Qed.

Theorem parsed_varint_is_bounded_and_consumes_one_prefix :
  forall bytes result,
    parse_varint bytes = Some result ->
    length bytes = parsed_varint_consumed result + length (parsed_varint_rest result) /\
    1 <= parsed_varint_consumed result <= 10 /\
    parsed_varint_value result <= maximum_u64.
Proof.
  intros bytes result Hparse; unfold parse_varint in Hparse.
  destruct (parse_varint_aux 10 1 bytes) as [raw|] eqn:Hraw;
    try discriminate.
  destruct (parsed_varint_value raw <=? maximum_u64) eqn:Hbound;
    try discriminate.
  inversion Hparse; subst.
  destruct (parse_varint_aux_consumes_a_prefix 10 1 bytes result Hraw)
    as [Hlength [Hpositive Hfuel]].
  apply Nat.leb_le in Hbound; repeat split; assumption.
Qed.

Record delimited_result : Type := {
  parsed_delimited_prefix_bytes : nat;
  parsed_delimited_payload : list nat;
  parsed_delimited_rest : list nat
}.

Definition parse_length_delimited (bytes : list nat)
    : option delimited_result :=
  match parse_varint bytes with
  | None => None
  | Some length_prefix =>
      let declared := parsed_varint_value length_prefix in
      let body := parsed_varint_rest length_prefix in
      if declared <=? length body then
        Some
          {| parsed_delimited_prefix_bytes :=
               parsed_varint_consumed length_prefix;
             parsed_delimited_payload := firstn declared body;
             parsed_delimited_rest := skipn declared body |}
      else None
  end.

Theorem parsed_length_delimited_field_consumes_exactly_its_prefix_and_body :
  forall bytes result,
    parse_length_delimited bytes = Some result ->
    length bytes =
      parsed_delimited_prefix_bytes result +
      length (parsed_delimited_payload result) +
      length (parsed_delimited_rest result).
Proof.
  intros bytes result Hparse; unfold parse_length_delimited in Hparse.
  destruct (parse_varint bytes) as [prefix|] eqn:Hprefix; try discriminate.
  destruct (parsed_varint_value prefix <=?
    length (parsed_varint_rest prefix)) eqn:Hfits; try discriminate.
  inversion Hparse; subst; simpl.
  destruct (parsed_varint_is_bounded_and_consumes_one_prefix
    bytes prefix Hprefix) as [Hbytes _].
  apply Nat.leb_le in Hfits.
  rewrite length_firstn, length_skipn.
  rewrite Nat.min_l by assumption; lia.
Qed.

Inductive wire_value : Type :=
  | wire_varint (value : nat)
  | wire_fixed64 (bytes : list nat)
  | wire_bytes (bytes : list nat)
  | wire_fixed32 (bytes : list nat).

Record parsed_wire_value : Type := {
  parsed_wire_contents : wire_value;
  parsed_wire_rest : list nat
}.

Definition parse_wire_value (wire_type : nat) (bytes : list nat)
    : option parsed_wire_value :=
  match wire_type with
  | 0 =>
      match parse_varint bytes with
      | Some result =>
          Some {| parsed_wire_contents := wire_varint (parsed_varint_value result);
                  parsed_wire_rest := parsed_varint_rest result |}
      | None => None
      end
  | 1 =>
      if 8 <=? length bytes then
        Some {| parsed_wire_contents := wire_fixed64 (firstn 8 bytes);
                parsed_wire_rest := skipn 8 bytes |}
      else None
  | 2 =>
      match parse_length_delimited bytes with
      | Some result =>
          Some {| parsed_wire_contents := wire_bytes (parsed_delimited_payload result);
                  parsed_wire_rest := parsed_delimited_rest result |}
      | None => None
      end
  | 5 =>
      if 4 <=? length bytes then
        Some {| parsed_wire_contents := wire_fixed32 (firstn 4 bytes);
                parsed_wire_rest := skipn 4 bytes |}
      else None
  | _ => None
  end.

Record parsed_key : Type := {
  parsed_field_tag : nat;
  parsed_wire_type : nat;
  parsed_key_rest : list nat
}.

Definition parse_key (bytes : list nat) : option parsed_key :=
  match parse_varint bytes with
  | None => None
  | Some result =>
      let tag := parsed_varint_value result / 8 in
      let wire_type := parsed_varint_value result mod 8 in
      if (1 <=? tag) && (tag <=? maximum_protobuf_tag) then
        Some {| parsed_field_tag := tag;
                parsed_wire_type := wire_type;
                parsed_key_rest := parsed_varint_rest result |}
      else None
  end.

Record parsed_field : Type := {
  field_tag : nat;
  field_contents : wire_value
}.

Fixpoint parse_fields (fuel : nat) (bytes : list nat)
    : option (list parsed_field) :=
  match bytes with
  | [] => Some []
  | _ =>
      match fuel with
      | 0 => None
      | S remaining_fuel =>
          match parse_key bytes with
          | None => None
          | Some key =>
              match parse_wire_value
                (parsed_wire_type key) (parsed_key_rest key) with
              | None => None
              | Some value =>
                  match parse_fields remaining_fuel (parsed_wire_rest value) with
                  | None => None
                  | Some suffix =>
                      Some
                        ({| field_tag := parsed_field_tag key;
                            field_contents := parsed_wire_contents value |} :: suffix)
                  end
              end
          end
      end
  end.

Record protobuf_counts : Type := {
  counted_operations : nat;
  counted_largest_name : nat;
  counted_largest_operation_pairs : nat;
  counted_total_pairs : nat;
  counted_text_bytes : nat
}.

Definition empty_protobuf_counts : protobuf_counts :=
  {| counted_operations := 0;
     counted_largest_name := 0;
     counted_largest_operation_pairs := 0;
     counted_total_pairs := 0;
     counted_text_bytes := 0 |}.

Definition with_text_bytes (counts : protobuf_counts) (additional : nat)
    : protobuf_counts :=
  {| counted_operations := counted_operations counts;
     counted_largest_name := counted_largest_name counts;
     counted_largest_operation_pairs := counted_largest_operation_pairs counts;
     counted_total_pairs := counted_total_pairs counts;
     counted_text_bytes := counted_text_bytes counts + additional |}.

Fixpoint scan_string_pair_fields
    (fields : list parsed_field) (counts : protobuf_counts)
    : option protobuf_counts :=
  match fields with
  | [] => Some counts
  | field :: rest =>
      if (field_tag field =? 1) || (field_tag field =? 2) then
        match field_contents field with
        | wire_bytes text =>
            scan_string_pair_fields rest (with_text_bytes counts (length text))
        | _ => None
        end
      else scan_string_pair_fields rest counts
  end.

Fixpoint scan_pair_fields
    (fields : list parsed_field) (counts : protobuf_counts)
    : option protobuf_counts :=
  match fields with
  | [] => Some counts
  | field :: rest =>
      if field_tag field =? 2 then
        match field_contents field with
        | wire_bytes payload =>
            match parse_fields (S (length payload)) payload with
            | Some nested =>
                match scan_string_pair_fields nested counts with
                | Some next => scan_pair_fields rest next
                | None => None
                end
            | None => None
            end
        | _ => None
        end
      else scan_pair_fields rest counts
  end.

Definition with_pair
    (counts : protobuf_counts) (operation_pairs : nat) : protobuf_counts :=
  {| counted_operations := counted_operations counts;
     counted_largest_name := counted_largest_name counts;
     counted_largest_operation_pairs :=
       Nat.max (counted_largest_operation_pairs counts) operation_pairs;
     counted_total_pairs := S (counted_total_pairs counts);
     counted_text_bytes := counted_text_bytes counts |}.

Definition with_name (counts : protobuf_counts) (name_bytes : nat)
    : protobuf_counts :=
  {| counted_operations := counted_operations counts;
     counted_largest_name := Nat.max (counted_largest_name counts) name_bytes;
     counted_largest_operation_pairs := counted_largest_operation_pairs counts;
     counted_total_pairs := counted_total_pairs counts;
     counted_text_bytes := counted_text_bytes counts |}.

Fixpoint scan_operation_fields
    (fields : list parsed_field) (operation_pairs : nat)
    (counts : protobuf_counts) : option protobuf_counts :=
  match fields with
  | [] =>
      Some
        {| counted_operations := counted_operations counts;
           counted_largest_name := counted_largest_name counts;
           counted_largest_operation_pairs :=
             Nat.max (counted_largest_operation_pairs counts) operation_pairs;
           counted_total_pairs := counted_total_pairs counts;
           counted_text_bytes := counted_text_bytes counts |}
  | field :: rest =>
      if field_tag field =? 5 then
        match field_contents field with
        | wire_bytes payload =>
            let next_pairs := S operation_pairs in
            let next_counts := with_pair counts next_pairs in
            match parse_fields (S (length payload)) payload with
            | Some nested =>
                match scan_pair_fields nested next_counts with
                | Some scanned => scan_operation_fields rest next_pairs scanned
                | None => None
                end
            | None => None
            end
        | _ => None
        end
      else if field_tag field =? 6 then
        match field_contents field with
        | wire_bytes name =>
            scan_operation_fields rest operation_pairs
              (with_name counts (length name))
        | _ => None
        end
      else scan_operation_fields rest operation_pairs counts
  end.

Definition with_operation (counts : protobuf_counts) : protobuf_counts :=
  {| counted_operations := S (counted_operations counts);
     counted_largest_name := counted_largest_name counts;
     counted_largest_operation_pairs := counted_largest_operation_pairs counts;
     counted_total_pairs := counted_total_pairs counts;
     counted_text_bytes := counted_text_bytes counts |}.

Fixpoint scan_operation_set_fields
    (fields : list parsed_field) (counts : protobuf_counts)
    : option protobuf_counts :=
  match fields with
  | [] => Some counts
  | field :: rest =>
      if field_tag field =? 1 then
        match field_contents field with
        | wire_bytes payload =>
            match parse_fields (S (length payload)) payload with
            | Some nested =>
                match scan_operation_fields nested 0 (with_operation counts) with
                | Some scanned => scan_operation_set_fields rest scanned
                | None => None
                end
            | None => None
            end
        | _ => None
        end
      else scan_operation_set_fields rest counts
  end.

Fixpoint scan_container_fields
    (fields : list parsed_field) (counts : protobuf_counts)
    : option protobuf_counts :=
  match fields with
  | [] => Some counts
  | field :: rest =>
      if field_tag field =? 1 then
        match field_contents field with
        | wire_bytes payload =>
            match parse_fields (S (length payload)) payload with
            | Some nested =>
                match scan_operation_set_fields nested counts with
                | Some scanned => scan_container_fields rest scanned
                | None => None
                end
            | None => None
            end
        | _ => None
        end
      else scan_container_fields rest counts
  end.

Record parser_limits : Type := {
  parser_max_payload : nat;
  parser_max_operations : nat;
  parser_max_name : nat;
  parser_max_pairs_per_operation : nat;
  parser_max_total_pairs : nat;
  parser_max_text : nat
}.

Definition protobuf_byte_preflight
    (limits : parser_limits) (bytes : list nat) : option protobuf_counts :=
  if bytes_well_formedb bytes then
  if length bytes <=? parser_max_payload limits then
    match parse_fields (S (length bytes)) bytes with
    | None => None
    | Some fields =>
        match scan_container_fields fields empty_protobuf_counts with
        | None => None
        | Some counts =>
            if (counted_operations counts <=? parser_max_operations limits) &&
               (counted_largest_name counts <=? parser_max_name limits) &&
               (counted_largest_operation_pairs counts <=?
                  parser_max_pairs_per_operation limits) &&
               (counted_total_pairs counts <=? parser_max_total_pairs limits) &&
               (counted_text_bytes counts <=? parser_max_text limits)
            then Some counts
            else None
        end
    end
  else None
  else None.

Theorem accepted_protobuf_bytes_are_wire_parsed_before_allocation :
  forall limits bytes counts,
    protobuf_byte_preflight limits bytes = Some counts ->
    exists fields,
      bytes_well_formed bytes /\
      parse_fields (S (length bytes)) bytes = Some fields /\
      scan_container_fields fields empty_protobuf_counts = Some counts /\
      length bytes <= parser_max_payload limits /\
      counted_operations counts <= parser_max_operations limits /\
      counted_largest_name counts <= parser_max_name limits /\
      counted_largest_operation_pairs counts <=
        parser_max_pairs_per_operation limits /\
      counted_total_pairs counts <= parser_max_total_pairs limits /\
      counted_text_bytes counts <= parser_max_text limits.
Proof.
  intros limits bytes counts Haccept; unfold protobuf_byte_preflight in Haccept.
  destruct (bytes_well_formedb bytes) eqn:Hbytes; try discriminate.
  destruct (length bytes <=? parser_max_payload limits) eqn:Hpayload;
    try discriminate.
  destruct (parse_fields (S (length bytes)) bytes) as [fields|] eqn:Hfields;
    try discriminate.
  destruct (scan_container_fields fields empty_protobuf_counts)
    as [scanned|] eqn:Hscan; try discriminate.
  destruct
    ((counted_operations scanned <=? parser_max_operations limits) &&
     (counted_largest_name scanned <=? parser_max_name limits) &&
     (counted_largest_operation_pairs scanned <=?
        parser_max_pairs_per_operation limits) &&
     (counted_total_pairs scanned <=? parser_max_total_pairs limits) &&
     (counted_text_bytes scanned <=? parser_max_text limits))
    eqn:Hbounds; try discriminate.
  inversion Haccept; subst counts.
  apply andb_true_iff in Hbounds as [Hbounds Htext].
  apply andb_true_iff in Hbounds as [Hbounds Htotal].
  apply andb_true_iff in Hbounds as [Hbounds Hper_operation].
  apply andb_true_iff in Hbounds as [Hoperations Hname].
  apply Nat.leb_le in Hpayload, Hoperations, Hname, Hper_operation, Htotal, Htext.
  exists fields; repeat split; try assumption.
  apply bytes_well_formedb_reflects_bytes; exact Hbytes.
Qed.

(** ** Refinement to the semantic admission model

    These theorems compose the concrete byte parsers with the already verified
    post-decode validation predicates.  Bincode and protobuf still require the
    semantic decoder's own success evidence; byte preflight cannot manufacture
    a valid operation model from a merely well-formed wire stream. *)

Theorem concrete_bincode_and_validated_payload_refine_abstract_admission :
  forall limits bytes decoded_operations longest_name decoded_pairs pair_text,
    bincode_byte_preflight (maximum_payload_bytes limits) bytes = true ->
    decoded_operations <= maximum_operations limits ->
    longest_name <= maximum_name_bytes limits ->
    decoded_pairs <= maximum_pairs limits ->
    pair_text <= maximum_pair_text_bytes limits ->
    exists envelope,
      accepts_envelope limits envelope /\
      envelope_magic envelope = parser_operation_set_magic /\
      envelope_version envelope = 1 /\
      envelope_flags envelope = 0 /\
      declared_payload_bytes envelope = available_payload_bytes envelope.
Proof.
  intros limits bytes decoded_operations longest_name decoded_pairs pair_text
    Hbytes Hoperations Hname Hpairs Htext.
  destruct (accepted_bincode_bytes_have_the_exact_runtime_envelope
    (maximum_payload_bytes limits) bytes Hbytes)
    as [header [Hwell_formed [Hparse [Hmagic [Hversion [Hflags
      [Hdeclared [Hpayload_limit Hlength]]]]]]]].
  exists
    {| envelope_magic := parsed_bincode_magic header;
       envelope_version := parsed_bincode_version header;
       envelope_flags := parsed_bincode_flags header;
       declared_payload_bytes := parsed_bincode_declared header;
       available_payload_bytes := length (parsed_bincode_payload header);
       consumed_payload_bytes := length (parsed_bincode_payload header);
       decoded_operations := decoded_operations;
       longest_name_bytes := longest_name;
       decoded_pairs := decoded_pairs;
       decoded_pair_text_bytes := pair_text;
       semantic_validation_passed := true |}.
  split.
  - unfold accepts_envelope; simpl; repeat split; try assumption;
      try (symmetry; assumption).
  - simpl; repeat split; assumption.
Qed.

Definition parser_limits_from_binary_limits (limits : binary_limits)
    : parser_limits :=
  {| parser_max_payload := maximum_payload_bytes limits;
     parser_max_operations := maximum_operations limits;
     parser_max_name := maximum_name_bytes limits;
     parser_max_pairs_per_operation := maximum_pairs limits;
     parser_max_total_pairs := maximum_pairs limits;
     parser_max_text := maximum_pair_text_bytes limits |}.

Theorem concrete_protobuf_and_validated_message_refine_abstract_admission :
  forall limits bytes counts supported semantic,
    protobuf_byte_preflight (parser_limits_from_binary_limits limits) bytes =
      Some counts ->
    supported = true ->
    semantic = true ->
    accepts_protobuf limits
      {| protobuf_wire_well_formed := true;
         protobuf_supported_format := supported;
         protobuf_payload_bytes := length bytes;
         protobuf_operations := counted_operations counts;
         protobuf_largest_name_bytes := counted_largest_name counts;
         protobuf_largest_operation_pairs := counted_largest_operation_pairs counts;
         protobuf_total_pairs := counted_total_pairs counts;
         protobuf_pair_text_bytes := counted_text_bytes counts;
         protobuf_semantic_validation_passed := semantic |}.
Proof.
  intros limits bytes counts supported semantic Hpreflight Hsupported Hsemantic.
  destruct (accepted_protobuf_bytes_are_wire_parsed_before_allocation
    (parser_limits_from_binary_limits limits) bytes counts Hpreflight)
    as [fields [Hwell_formed [Hparse [Hscan [Hpayload [Hoperations [Hname
      [Hper_operation [Htotal Htext]]]]]]]]].
  unfold accepts_protobuf; simpl in *.
  subst; repeat split; assumption.
Qed.

(** ** Explicit gzip/flate2 boundary *)

Record gzip_decompressor_observation : Type := {
  observed_gzip_consumed : nat;
  observed_gzip_output : list nat;
  observed_gzip_checksum_valid : bool
}.

Definition gzip_magic_valid (input : list nat) : bool :=
  match input with
  | first :: second :: _ => (first =? 31) && (second =? 139)
  | _ => false
  end.

Definition gzip_adapter_preflight
    (compressed_limit decompressed_limit : nat)
    (input : list nat) (observation : gzip_decompressor_observation)
    (inner_decoder_accepts : bool) : bool :=
  bytes_well_formedb input &&
  gzip_magic_valid input &&
  (length input <=? compressed_limit) &&
  (observed_gzip_consumed observation =? length input) &&
  (length (observed_gzip_output observation) <=? decompressed_limit) &&
  observed_gzip_checksum_valid observation &&
  inner_decoder_accepts.

Theorem accepted_gzip_adapter_observation_is_complete_bounded_and_inner_valid :
  forall compressed_limit decompressed_limit input observation inner,
    gzip_adapter_preflight compressed_limit decompressed_limit
      input observation inner = true ->
    bytes_well_formed input /\
    firstn 2 input = [31; 139] /\
    length input <= compressed_limit /\
    observed_gzip_consumed observation = length input /\
    length (observed_gzip_output observation) <= decompressed_limit /\
    observed_gzip_checksum_valid observation = true /\
    inner = true.
Proof.
  intros compressed_limit decompressed_limit input observation inner Haccept.
  apply andb_true_iff in Haccept as [Haccept Hinner].
  apply andb_true_iff in Haccept as [Haccept Hchecksum].
  apply andb_true_iff in Haccept as [Haccept Houtput].
  apply andb_true_iff in Haccept as [Haccept Hconsumed].
  apply andb_true_iff in Haccept as [Haccept Hcompressed].
  apply andb_true_iff in Haccept as [Hbytes Hmagic].
  apply Nat.leb_le in Hcompressed, Houtput.
  apply Nat.eqb_eq in Hconsumed.
  destruct input as [|first [|second rest]]; simpl in Hmagic; try discriminate.
  apply andb_true_iff in Hmagic as [Hfirst Hsecond].
  apply Nat.eqb_eq in Hfirst, Hsecond; subst.
  repeat split; try assumption; try reflexivity.
  apply bytes_well_formedb_reflects_bytes; exact Hbytes.
Qed.

Theorem concrete_gzip_adapter_refines_abstract_admission :
  forall compressed_limit decompressed_limit input observation inner,
    gzip_adapter_preflight compressed_limit decompressed_limit
      input observation inner = true ->
    accepts_gzip compressed_limit decompressed_limit (length input)
      {| gzip_checksum_valid := observed_gzip_checksum_valid observation;
         gzip_compressed_bytes := length input;
         gzip_consumed_compressed_bytes := observed_gzip_consumed observation;
         gzip_decompressed_bytes := length (observed_gzip_output observation) |}
      inner.
Proof.
  intros compressed_limit decompressed_limit input observation inner Haccept.
  destruct (accepted_gzip_adapter_observation_is_complete_bounded_and_inner_valid
    compressed_limit decompressed_limit input observation inner Haccept)
    as [Hwell_formed [Hmagic [Hcompressed [Hconsumed [Houtput
      [Hchecksum Hinner]]]]]].
  unfold accepts_gzip; simpl; repeat split; assumption.
Qed.
