(* C9 leak-discipline suite for the OCaml facade, measured with Gc.stat.

   A >=10,000-cycle create/use/free loop over dictionaries, transducers, query
   cursors, phonetic patterns, and rule sets must reach a heap steady state:
   after a full major collection the live word count must not drift upward. The
   facade frees native handles at close; Gc confirms the OCaml wrappers are
   reclaimed rather than accumulating. *)

module Dict = Vinary_tree_libdictenstein
module Lev = Vinary_tree_liblevenshtein

let cycles = 10_000
let warmup = 2_000
(* Generous ceiling; a per-cycle leak would accrue far more over 10k cycles. *)
let max_growth_words = 2_000_000

let settled_live_words () =
  Gc.full_major ();
  (Gc.stat ()).Gc.live_words

let assert_steady label cycle =
  for _ = 1 to warmup do
    cycle ()
  done;
  let base = settled_live_words () in
  for _ = 1 to cycles do
    cycle ()
  done;
  let growth = settled_live_words () - base in
  if growth > max_growth_words then
    failwith (Printf.sprintf "%s: heap grew %d words over %d cycles" label growth cycles)

let transducer_cycle () =
  let dictionary = Dict.dynamic_dawg () in
  ignore (Dict.put dictionary "cat" (Some 1L));
  ignore (Dict.put dictionary "cot" (Some 2L));
  ignore (Dict.put dictionary "cut" (Some 3L));
  ignore (Dict.put dictionary "scat" None);
  let automaton = Lev.transducer (Dict.resource dictionary) in
  let cursor = Lev.query automaton "cat" ~maximum_distance:2 in
  ignore (List.of_seq (Lev.to_seq cursor));
  Lev.cursor_close cursor;
  Dict.close dictionary;
  Lev.close_transducer automaton

let phonetic_cycle () =
  let pattern = Lev.regex_pattern "c[ao]t" in
  ignore (Lev.pattern_matches pattern "cat");
  Lev.close_pattern pattern;
  let rules = Lev.phonetic_rules "english-orthography" in
  ignore (Lev.apply_rules rules "phone");
  Lev.close_rules rules

let () =
  assert_steady "transducer" transducer_cycle;
  assert_steady "phonetic" phonetic_cycle;
  print_endline "OCaml leak tests passed"
