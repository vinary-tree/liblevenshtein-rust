(** Snapshot-consistent fuzzy search over a retained Vinary Tree dictionary.
    A transducer retains a [Vinary_tree_interop.resource]. Each query opens a
    one-shot cursor over the dictionary revision visible at query start.
    Explicitly close native handles; finalizers are only a fallback. *)

(** Select standard Levenshtein edits, optimal-string-alignment transpositions,
    merge/split edits, or unrestricted Damerau-Levenshtein transpositions.
    Optimal string alignment is not a metric; unrestricted Damerau is. *)
type algorithm =
  | Standard
  | Transposition
  | Merge_and_split
  | Damerau_levenshtein

(** [Traversal] follows the dictionary walk; [Distance_then_term] sorts by
    edit distance and then term. *)
type query_order = Traversal | Distance_then_term

(** [Text] holds UTF-8 text or raw bytes according to the dictionary domain.
    [Tokens] holds unsigned-64 unit bit patterns in OCaml [int64] elements. *)
type term = Text of string | Tokens of int64 array

(** Owned match. [id = None] means present without a value, not absent. *)
type match_result = { term : term; distance : int; id : int64 option }

(** Owned query engine retaining its dictionary provider. *)
type transducer

(** Bounded, revision-aware query-result cache. *)
type query_cache

(** Cache counters and resident sizes. Counter reset does not evict entries. *)
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

(** One-shot, single-consumer query traversal. *)
type cursor

(** Compiled phonetic matcher; release with {!close_pattern}. *)
type phonetic_pattern

(** Parsed or built-in rewrite rules; release with {!close_rules}. *)
type phonetic_rules

(** Retain [dictionary] in constant time. [Standard] is the default algorithm. *)
val transducer :
  ?algorithm:algorithm -> Vinary_tree_interop.resource -> transducer

(** Release the transducer; already opened cursors retain their snapshots. *)
val close_transducer : transducer -> unit

(** Bound cached results by count and weight. Defaults: 1024 entries and
    64 MiB. Eviction loses only acceleration, never query correctness. *)
val query_cache :
  ?maximum_entries:int -> ?maximum_weight:int -> transducer -> query_cache

(** Release the owned cache. *)
val close_query_cache : query_cache -> unit

(** Evict entries while retaining counters. *)
val clear_query_cache : query_cache -> unit

(** Reset counters while retaining entries. *)
val reset_query_cache_stats : query_cache -> unit

(** Read cache counters and resident sizes. *)
val query_cache_stats : query_cache -> query_cache_stats

(** Open a cached UTF-8 query with an inclusive edit-distance bound. *)
val cached_query :
  ?order:query_order -> query_cache -> string -> maximum_distance:int -> cursor

(** Open a cached raw-byte query, preserving zero and non-UTF-8 bytes. *)
val cached_query_bytes :
  ?order:query_order -> query_cache -> bytes -> maximum_distance:int -> cursor

(** Open a cached unsigned-64-token query. *)
val cached_query_u64 :
  ?order:query_order -> query_cache -> int64 array -> maximum_distance:int -> cursor

(** Query UTF-8 text against one immutable dictionary revision. *)
val query :
  ?order:query_order -> transducer -> string -> maximum_distance:int -> cursor

(** Query arbitrary bytes against a byte-domain dictionary. *)
val query_bytes :
  ?order:query_order -> transducer -> bytes -> maximum_distance:int -> cursor

(** Query token arrays against an unsigned-64-domain dictionary. *)
val query_u64 :
  ?order:query_order -> transducer -> int64 array -> maximum_distance:int -> cursor

(** Query a compiled phonetic pattern. *)
val query_pattern :
  transducer -> phonetic_pattern -> maximum_distance:int -> cursor

(** Release a cursor after exhaustion, early return, or exception. *)
val cursor_close : cursor -> unit

(** Pull one owned match, or [None] at exhaustion. *)
val next : cursor -> match_result option

(** Pull up to [maximum] owned matches per call (default 256), or [None]
    at exhaustion. The returned array has no native lease. *)
val next_batch : ?maximum:int -> cursor -> match_result array option

(** Adapt [cursor] to a lazy sequence. The sequence does not own the cursor:
    scope consumption with [Fun.protect] and {!cursor_close}. *)
val to_seq : cursor -> match_result Seq.t

(** Fold bounded batches (default 256). This does not close the cursor. *)
val fold_batches :
  ?maximum:int -> cursor -> 'state -> ('state -> match_result array -> 'state) -> 'state

(** Compile a regular-expression pattern. *)
val regex_pattern : string -> phonetic_pattern

(** Compile a LLRE phonetic pattern. *)
val llre_pattern : string -> phonetic_pattern

(** Test whether a compiled pattern matches a string. *)
val pattern_matches : phonetic_pattern -> string -> bool

(** Return the pattern's state and transition counts. *)
val pattern_size : phonetic_pattern -> int * int

(** Release a compiled pattern. *)
val close_pattern : phonetic_pattern -> unit

(** Parse rules or select a built-in set such as ["english-orthography"] or
    ["english-phonetic"]. *)
val phonetic_rules : string -> phonetic_rules

(** Number of rules in a set. *)
val rules_length : phonetic_rules -> int

(** Apply rules, returning an owned OCaml string. *)
val apply_rules : phonetic_rules -> string -> string

(** Release a rule set. *)
val close_rules : phonetic_rules -> unit

(** Exact standard Levenshtein distance between UTF-8 strings. *)
val distance : string -> string -> int

(** Thresholded standard Levenshtein distance; see the native ABI contract
    for the result when the exact distance exceeds the bound. *)
val distance_threshold : string -> string -> int -> int

(** Exact optimal-string-alignment distance. *)
val damerau_distance : string -> string -> int

(** Thresholded optimal-string-alignment distance. *)
val damerau_distance_threshold : string -> string -> int -> int

(** Exact unrestricted Damerau-Levenshtein distance. *)
val true_damerau_distance : string -> string -> int

(** Thresholded unrestricted Damerau-Levenshtein distance. *)
val true_damerau_distance_threshold : string -> string -> int -> int
