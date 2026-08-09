(* C8 native property-based tests for the OCaml facade (qcheck).

   Every property is checked against an in-language brute-force Levenshtein
   oracle:
     (a) distance symmetry d(a,b)=d(b,a), identity d(a,a)=0, and threshold
         consistency dThreshold(a,b,k) = (if d<=k then d else sentinel) across
         the Levenshtein/Damerau/true-Damerau variants, plus standard oracle
         agreement. The over-bound sentinel is derived at runtime rather than
         hard-coded, so the test is robust to its representation as an OCaml int.
     (b) a query's result set at distance k equals {t in dict : lev(query,t)<=k}
         over a random libdictenstein DynamicDawg dictionary, with exact
         distances and value round-trips.
     (c) u64 value round-trips, with 0 and boundary patterns pinned.

   qcheck runs from a fixed Random.State seed, so a failing run reproduces. An
   ASCII alphabet keeps byte = scalar for the oracle; scalar-vs-byte counting is
   pinned by the snapshot suite. Distance operands may be empty: the facade
   marshals an empty string as a null pointer, and as of the LLEV-B18 fix the
   native distance entry points accept NULL+0 as the empty string. *)

module Dict = Vinary_tree_libdictenstein
module Lev = Vinary_tree_liblevenshtein

let alphabet = [ 'a'; 'b'; 'c'; 'd' ]

let term_gen = QCheck.Gen.(string_size ~gen:(oneof_list alphabet) (int_range 0 6))

let term = QCheck.make ~print:(Printf.sprintf "%S") term_gen

(* Derive the over-bound sentinel from a known over-threshold pair. *)
let sentinel = Lev.distance_threshold "aaaa" "bbbb" 0

let oracle a b =
  let la = String.length a and lb = String.length b in
  if a = b then 0
  else if la = 0 then lb
  else if lb = 0 then la
  else begin
    let previous = Array.init (lb + 1) (fun j -> j) in
    let current = Array.make (lb + 1) 0 in
    for i = 1 to la do
      current.(0) <- i;
      for j = 1 to lb do
        let cost = if a.[i - 1] = b.[j - 1] then 0 else 1 in
        current.(j) <-
          min (min (previous.(j) + 1) (current.(j - 1) + 1)) (previous.(j - 1) + cost)
      done;
      Array.blit current 0 previous 0 (lb + 1)
    done;
    previous.(lb)
  end

let text_of_match result =
  match result.Lev.term with
  | Lev.Text text -> text
  | Lev.Tokens _ -> failwith "expected a text match"

let run_query entries query k =
  let dictionary = Dict.dynamic_dawg () in
  List.iter (fun (t, v) -> ignore (Dict.put dictionary t v)) entries;
  let automaton = Lev.transducer (Dict.resource dictionary) in
  let cursor = Lev.query automaton query ~maximum_distance:k in
  let matches = List.of_seq (Lev.to_seq cursor) in
  Lev.cursor_close cursor;
  Dict.close dictionary;
  Lev.close_transducer automaton;
  List.map (fun r -> (text_of_match r, (r.Lev.distance, r.Lev.id))) matches

let distance_property name dist thr =
  QCheck.Test.make ~count:400 ~name (QCheck.triple term term (QCheck.int_range 0 3))
    (fun (a, b, k) ->
      dist a b = dist b a
      && dist a a = 0
      &&
      let full = dist a b in
      let bounded = thr a b k in
      if full <= k then bounded = full else bounded = sentinel)

let oracle_property =
  QCheck.Test.make ~count:400 ~name:"standard oracle agreement" (QCheck.pair term term)
    (fun (a, b) -> Lev.distance a b = oracle a b)

let value_gen =
  QCheck.Gen.(oneof_weighted [ (1, return None); (3, map (fun i -> Some (Int64.of_int i)) (int_bound 1_000_000)) ])

let dictionary_arb =
  QCheck.make
    QCheck.Gen.(list_size (int_range 0 8) (pair term_gen value_gen))

let query_property =
  QCheck.Test.make ~count:150 ~name:"query result set equals oracle"
    (QCheck.triple dictionary_arb term (QCheck.int_range 0 3))
    (fun (raw_entries, query, k) ->
      (* Deduplicate terms (later bindings win), mirroring dictionary insertion. *)
      let table = Hashtbl.create 16 in
      List.iter (fun (t, v) -> Hashtbl.replace table t v) raw_entries;
      let entries = Hashtbl.fold (fun t v acc -> (t, v) :: acc) table [] in
      let got = run_query entries query k in
      let expected = List.filter (fun (t, _) -> oracle query t <= k) entries in
      let got_terms = List.sort compare (List.map fst got) in
      let expected_terms = List.sort compare (List.map fst expected) in
      got_terms = expected_terms
      && List.for_all
           (fun (t, (d, id)) -> d = oracle query t && List.assoc t entries = id)
           got)

let () =
  let rand = Random.State.make [| 20260809 |] in
  let check t = QCheck.Test.check_exn ~rand t in
  check (distance_property "levenshtein" Lev.distance Lev.distance_threshold);
  check (distance_property "damerau" Lev.damerau_distance Lev.damerau_distance_threshold);
  check (distance_property "true_damerau" Lev.true_damerau_distance Lev.true_damerau_distance_threshold);
  check oracle_property;
  check query_property;
  (* (c) u64 value round-trip: 0, 1, and boundary bit patterns. *)
  List.iter
    (fun value ->
      match run_query [ ("term", Some value) ] "term" 0 with
      | [ ("term", (0, id)) ] when id = Some value -> ()
      | _ -> failwith (Printf.sprintf "u64 round-trip failed for %Ld" value))
    [ 0L; 1L; Int64.max_int; Int64.min_int; -1L ];
  print_endline "OCaml property tests passed"
