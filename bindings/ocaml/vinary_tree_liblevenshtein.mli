type algorithm =
  | Standard
  | Transposition
  | Merge_and_split
  | Damerau_levenshtein

type query_order = Traversal | Distance_then_term
type term = Text of string | Tokens of int64 array
type match_result = { term : term; distance : int; id : int64 option }
type transducer
type query_cache
type query_cache_stats = {
  requests : int64;
  hits : int64;
  misses : int64;
  admissions : int64;
  rejections : int64;
  evictions : int64;
  resident_entries : int;
  resident_weight : int;
}
type cursor
type phonetic_pattern
type phonetic_rules

val transducer :
  ?algorithm:algorithm -> Vinary_tree_interop.resource -> transducer
val close_transducer : transducer -> unit
val query_cache :
  ?maximum_entries:int -> ?maximum_weight:int -> transducer -> query_cache
val close_query_cache : query_cache -> unit
val clear_query_cache : query_cache -> unit
val reset_query_cache_stats : query_cache -> unit
val query_cache_stats : query_cache -> query_cache_stats
val cached_query :
  ?order:query_order -> query_cache -> string -> maximum_distance:int -> cursor
val cached_query_bytes :
  ?order:query_order -> query_cache -> bytes -> maximum_distance:int -> cursor
val cached_query_u64 :
  ?order:query_order -> query_cache -> int64 array -> maximum_distance:int -> cursor
val query :
  ?order:query_order -> transducer -> string -> maximum_distance:int -> cursor
val query_bytes :
  ?order:query_order -> transducer -> bytes -> maximum_distance:int -> cursor
val query_u64 :
  ?order:query_order -> transducer -> int64 array -> maximum_distance:int -> cursor
val query_pattern :
  transducer -> phonetic_pattern -> maximum_distance:int -> cursor
val cursor_close : cursor -> unit
val next : cursor -> match_result option
val next_batch : ?maximum:int -> cursor -> match_result array option
val to_seq : cursor -> match_result Seq.t
val fold_batches :
  ?maximum:int -> cursor -> 'state -> ('state -> match_result array -> 'state) -> 'state

val regex_pattern : string -> phonetic_pattern
val llre_pattern : string -> phonetic_pattern
val pattern_matches : phonetic_pattern -> string -> bool
val pattern_size : phonetic_pattern -> int * int
val close_pattern : phonetic_pattern -> unit
val phonetic_rules : string -> phonetic_rules
val rules_length : phonetic_rules -> int
val apply_rules : phonetic_rules -> string -> string
val close_rules : phonetic_rules -> unit

val distance : string -> string -> int
val distance_threshold : string -> string -> int -> int
val damerau_distance : string -> string -> int
val damerau_distance_threshold : string -> string -> int -> int
val true_damerau_distance : string -> string -> int
val true_damerau_distance_threshold : string -> string -> int -> int
