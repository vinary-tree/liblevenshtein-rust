type algorithm = Standard | Transposition | Merge_and_split | Damerau_levenshtein
type query_order = Traversal | Distance_then_term
type term = Text of string | Tokens of int64 array
type match_result = { term : term; distance : int; id : int64 option }
type transducer
type cursor
type phonetic_pattern
type phonetic_rules

external raw_transducer : Vinary_tree_interop.resource -> algorithm -> transducer
  = "ocaml_llev_transducer"
external close_transducer : transducer -> unit = "ocaml_llev_transducer_close"
external raw_query : transducer -> string -> int -> query_order -> cursor
  = "ocaml_llev_query"
external raw_query_bytes : transducer -> bytes -> int -> query_order -> cursor
  = "ocaml_llev_query_bytes"
external raw_query_u64 : transducer -> int64 array -> int -> query_order -> cursor
  = "ocaml_llev_query_u64"
external query_pattern : transducer -> phonetic_pattern -> maximum_distance:int -> cursor
  = "ocaml_llev_query_pattern"
external cursor_close : cursor -> unit = "ocaml_llev_cursor_close"
external next : cursor -> match_result option = "ocaml_llev_cursor_next"
external raw_next_batch : cursor -> int -> match_result array option
  = "ocaml_llev_cursor_next_batch"
external regex_pattern : string -> phonetic_pattern = "ocaml_llev_regex_pattern"
external llre_pattern : string -> phonetic_pattern = "ocaml_llev_llre_pattern"
external pattern_matches : phonetic_pattern -> string -> bool = "ocaml_llev_pattern_matches"
external pattern_size : phonetic_pattern -> int * int = "ocaml_llev_pattern_size"
external close_pattern : phonetic_pattern -> unit = "ocaml_llev_pattern_close"
external phonetic_rules : string -> phonetic_rules = "ocaml_llev_phonetic_rules"
external rules_length : phonetic_rules -> int = "ocaml_llev_rules_length"
external apply_rules : phonetic_rules -> string -> string = "ocaml_llev_apply_rules"
external close_rules : phonetic_rules -> unit = "ocaml_llev_rules_close"
external distance : string -> string -> int = "ocaml_llev_distance"
external distance_threshold : string -> string -> int -> int
  = "ocaml_llev_distance_threshold"
external damerau_distance : string -> string -> int = "ocaml_llev_damerau_distance"
external damerau_distance_threshold : string -> string -> int -> int
  = "ocaml_llev_damerau_distance_threshold"
external true_damerau_distance : string -> string -> int
  = "ocaml_llev_true_damerau_distance"
external true_damerau_distance_threshold : string -> string -> int -> int
  = "ocaml_llev_true_damerau_distance_threshold"

let transducer ?(algorithm = Standard) resource = raw_transducer resource algorithm
let query ?(order = Traversal) value text ~maximum_distance =
  raw_query value text maximum_distance order
let query_bytes ?(order = Traversal) value text ~maximum_distance =
  raw_query_bytes value text maximum_distance order
let query_u64 ?(order = Traversal) value tokens ~maximum_distance =
  raw_query_u64 value tokens maximum_distance order
let next_batch ?(maximum = 256) cursor = raw_next_batch cursor maximum

let rec to_seq cursor () =
  match next cursor with
  | None -> Seq.Nil
  | Some value -> Seq.Cons (value, to_seq cursor)

let fold_batches ?(maximum = 256) cursor initial reducer =
  let rec loop state =
    match next_batch ~maximum cursor with
    | None -> state
    | Some batch -> loop (reducer state batch)
  in
  loop initial
