module Dict = Vinary_tree_libdictenstein
module Lev = Vinary_tree_liblevenshtein

let collect cursor =
  let values = List.of_seq (Lev.to_seq cursor) in
  Lev.cursor_close cursor;
  values

let text_of_match result =
  match result.Lev.term with
  | Lev.Text text -> text
  | Lev.Tokens _ -> failwith "expected a text match"

let remove_if_present path =
  if Sys.file_exists path then Sys.remove path

let scratch_directory =
  match Sys.getenv_opt "VINARY_TREE_TEST_TMPDIR" with
  | Some path -> path
  | None ->
      invalid_arg
        "VINARY_TREE_TEST_TMPDIR must name a writable non-tmpfs test directory"

let temporary_path suffix =
  let path = Filename.concat scratch_directory ("liblevenshtein-ocaml" ^ suffix) in
  List.iter remove_if_present [path; path ^ ".wal"; path ^ ".wlock"];
  path

let () =
  let algorithms : Lev.algorithm list =
    [Lev.Standard; Lev.Transposition; Lev.Merge_and_split;
     Lev.Damerau_levenshtein]
  in
  let orders : Lev.query_order list = [Lev.Traversal; Lev.Distance_then_term] in
  assert (List.length algorithms = 4 && List.length orders = 2);

  for trace = 0 to 63 do
    let words = Dict.dynamic_dawg () in
    let entries =
      Array.init 16 (fun index ->
        (Printf.sprintf "t%d-%d" trace index, Some (Int64.of_int index)))
    in
    assert (Dict.put_many words entries = 16);
    let automaton = Lev.transducer (Dict.resource words) in
    let expected =
      collect (Lev.query ~order:Lev.Distance_then_term automaton ""
                 ~maximum_distance:64)
    in
    let cursor =
      Lev.query ~order:Lev.Distance_then_term automaton "" ~maximum_distance:64
    in
    let first = Option.get (Lev.next cursor) in
    assert (Dict.remove words (Printf.sprintf "t%d-1" trace));
    ignore (Dict.put words (Printf.sprintf "t%d-2" trace) (Some 999L));
    let second = Option.get (Lev.next cursor) in
    Dict.clear words;
    ignore (Dict.compact words);
    ignore (Dict.put words (Printf.sprintf "after-%d" trace) (Some 1000L));
    let remainder = List.of_seq (Lev.to_seq cursor) in
    Lev.cursor_close cursor;
    assert (first :: second :: remainder = expected);
    Dict.close words;
    let fresh = collect (Lev.query automaton "" ~maximum_distance:64) in
    assert (List.map text_of_match fresh = [Printf.sprintf "after-%d" trace]);
    Lev.close_transducer automaton
  done;

  let dat = Dict.double_array_trie [|("café", Some 7L); ("caff", None)|] in
  assert ((Dict.get dat "café").value = Some 7L);
  assert ((Dict.get dat "caff").found);
  Dict.close dat;

  let suffixes = Dict.scdawg () in
  ignore (Dict.put suffixes "cat" (Some 1L));
  ignore (Dict.put suffixes "cot" (Some 2L));
  assert (Dict.contains_substring suffixes "ot");
  assert (Dict.substring_frequency suffixes "t" = 2);
  Dict.close suffixes;

  let token_dictionary = Dict.dynamic_dawg ~domain:Vinary_tree_interop.U64 () in
  assert (Dict.put_u64 token_dictionary [|1L; 2L; 3L|] (Some 12L));
  let token_automaton = Lev.transducer (Dict.resource token_dictionary) in
  let token_cursor =
    Lev.query_u64 token_automaton [|1L; 2L; 4L|] ~maximum_distance:1
  in
  let token_match = Option.get (Lev.next token_cursor) in
  (match token_match.term with
   | Lev.Tokens tokens -> assert (tokens = [|1L; 2L; 3L|])
   | Lev.Text _ -> assert false);
  assert (token_match.distance = 1 && token_match.id = Some 12L);
  Lev.cursor_close token_cursor;
  Dict.close token_dictionary;
  Lev.close_transducer token_automaton;

  let byte_dictionary =
    Dict.dynamic_dawg ~domain:Vinary_tree_interop.Byte ()
  in
  let byte_text = "\x63\x00\xff" in
  assert (Dict.put byte_dictionary byte_text (Some 13L));
  let byte_automaton = Lev.transducer (Dict.resource byte_dictionary) in
  let byte_cursor =
    Lev.query_bytes byte_automaton (Bytes.of_string byte_text)
      ~maximum_distance:0
  in
  let byte_batch = Option.get (Lev.next_batch ~maximum:1 byte_cursor) in
  assert (Array.length byte_batch = 1);
  (match byte_batch.(0).term with
   | Lev.Text text -> assert (text = byte_text)
   | Lev.Tokens _ -> assert false);
  assert (byte_batch.(0).distance = 0 && byte_batch.(0).id = Some 13L);
  assert (Lev.next_batch ~maximum:1 byte_cursor = None);
  Lev.cursor_close byte_cursor;
  Dict.close byte_dictionary;
  Lev.close_transducer byte_automaton;

  let persistent_path = temporary_path ".artrie" in
  let persistent = Dict.create_persistent_artrie persistent_path in
  ignore (Dict.put persistent "durable" (Some 17L));
  Dict.checkpoint persistent;
  Dict.close persistent;
  let reopened = Dict.open_persistent_artrie persistent_path in
  assert ((Dict.get reopened "durable").value = Some 17L);
  Dict.close reopened;

  let vocabulary_path = temporary_path ".vocab" in
  let vocabulary = Dict.create_persistent_vocabulary vocabulary_path in
  ignore (Dict.put vocabulary "alpha" (Some 0L));
  Dict.checkpoint vocabulary;
  assert (Dict.term vocabulary 0L = Some "alpha");
  Dict.close vocabulary;
  let reopened_vocabulary = Dict.open_persistent_vocabulary vocabulary_path in
  assert ((Dict.get reopened_vocabulary "alpha").value = Some 0L);
  Dict.close reopened_vocabulary;

  let pattern = Lev.regex_pattern "c[ao]t" in
  assert (Lev.pattern_matches pattern "cat");
  assert (not (Lev.pattern_matches pattern "cut"));
  let states, transitions = Lev.pattern_size pattern in
  assert (states > 0 && transitions > 0);
  Lev.close_pattern pattern;
  let rules = Lev.phonetic_rules "english-orthography" in
  assert (Lev.rules_length rules > 0);
  ignore (Lev.apply_rules rules "phone");
  Lev.close_rules rules;
  assert (Lev.distance "kitten" "sitting" = 3);
  assert (Lev.distance_threshold "kitten" "sitting" 3 = 3);
  assert (Lev.damerau_distance "ca" "ac" = 1);
  assert (Lev.damerau_distance_threshold "ca" "ac" 1 = 1);
  assert (Lev.true_damerau_distance "ca" "ac" = 1);
  assert (Lev.true_damerau_distance_threshold "ca" "ac" 1 = 1);

  (* C5 (reduced): the pull iterator (to_seq) and the push reducer (fold_batches)
     drain the same query to the same multiset of matched terms. *)
  let equality_dictionary = Dict.dynamic_dawg () in
  ignore (Dict.put equality_dictionary "cat" (Some 1L));
  ignore (Dict.put equality_dictionary "cot" (Some 2L));
  ignore (Dict.put equality_dictionary "cut" (Some 3L));
  let equality_automaton = Lev.transducer (Dict.resource equality_dictionary) in
  List.iter
    (fun algorithm ->
       let selected =
         Lev.transducer ~algorithm (Dict.resource equality_dictionary)
       in
       let exact =
         collect
           (Lev.query ~order:Lev.Traversal selected "cat" ~maximum_distance:0)
       in
       assert (List.map text_of_match exact = ["cat"]);
       Lev.close_transducer selected)
    algorithms;
  let by_iterator =
    List.sort compare
      (List.map (fun result -> result.Lev.term)
         (collect (Lev.query equality_automaton "cat" ~maximum_distance:1)))
  in
  let reducer_cursor = Lev.query equality_automaton "cat" ~maximum_distance:1 in
  let by_reducer =
    List.sort compare
      (Lev.fold_batches reducer_cursor []
         (fun state batch ->
            Array.fold_left (fun acc result -> result.Lev.term :: acc) state batch))
  in
  Lev.cursor_close reducer_cursor;
  assert (by_iterator = by_reducer);
  assert (List.length by_iterator = 3);
  let llre = Lev.llre_pattern "@name \"Greeting\"\n^hello$" in
  assert (Lev.pattern_matches llre "hello");
  assert (not (Lev.pattern_matches llre "world"));
  Lev.close_pattern llre;
  let product_pattern = Lev.regex_pattern "c[ao]t" in
  let product_matches =
    collect
      (Lev.query_pattern equality_automaton product_pattern ~maximum_distance:0)
  in
  assert
    (List.sort compare (List.map text_of_match product_matches) = ["cat"; "cot"]);
  Lev.close_pattern product_pattern;
  Dict.close equality_dictionary;
  Lev.close_transducer equality_automaton;

  (* C6 (reduced): distances decode UTF-8 to Unicode scalar values, so a
     multi-byte character is one edit, not one edit per byte. *)
  assert (Lev.distance "café" "cafe" = 1);
  assert (Lev.distance "🦀" "x" = 1);
  assert (Lev.distance "é" "e" = 1);
  assert (Lev.distance_threshold "café" "cafe" 1 = 1);

  (* C2/C3 (reduced): close is idempotent and a closed pattern raises
     Invalid_argument on use rather than dereferencing a freed automaton. *)
  let lifecycle = Lev.regex_pattern "c[ao]t" in
  assert (Lev.pattern_matches lifecycle "cat");
  Lev.close_pattern lifecycle;
  Lev.close_pattern lifecycle;
  (match Lev.pattern_matches lifecycle "cat" with
   | (_ : bool) -> assert false
   | exception Invalid_argument _ -> ());

  List.iter remove_if_present
    [ persistent_path; persistent_path ^ ".wal"; persistent_path ^ ".wlock";
      vocabulary_path; vocabulary_path ^ ".wal"; vocabulary_path ^ ".wlock" ];
  print_endline "OCaml binding snapshot integration passed"
