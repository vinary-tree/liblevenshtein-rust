(* OCaml harness for the cross-language benchmark program.

   Implements harnesses/common/PROTOCOL.md over the ctypes-style C stub
   facades (vinary-tree-liblevenshtein + vinary-tree-libdictenstein dune
   packages installed in the opam switch). The runner provides
   LD_LIBRARY_PATH pointing at the RELEASE cdylibs.

   Fairness notes (PROTOCOL.md §10): stock OCaml 5 runtime, no flambda
   flags; monotonic time via a harness-local clock_gettime(CLOCK_MONOTONIC)
   C stub because OCaml 5's unix library exposes no Unix.clock_gettime. *)

module Dict = Vinary_tree_libdictenstein
module Lev = Vinary_tree_liblevenshtein

external now_ns : unit -> int64 = "bench_now_ns"

let wall_cap_seconds = 300.0

let sample_definition =
  "one full pass over the query set; every cursor fully drained and "
  ^ "(term, distance) materialized"

let base_notes =
  [ "monotonic clock via harness-local clock_gettime(CLOCK_MONOTONIC) C \
     stub (OCaml 5 unix exposes no Unix.clock_gettime)" ]

let die message =
  Printf.eprintf "bench-cross-ocaml: %s\n%!" message;
  exit 2

(* ------------------------------------------------------------------ *)
(* Checksum primitives (PROTOCOL.md §8) — Int64 ops wrap by definition *)
(* ------------------------------------------------------------------ *)

let fnv_offset = 0xcbf29ce484222325L
let fnv_prime = 0x100000001b3L

let fnv_update hash byte =
  Int64.mul (Int64.logxor hash (Int64.of_int byte)) fnv_prime

(* entry(term, distance) over utf8(term) ‖ 0x00 ‖ LE64(distance).
   OCaml strings are byte strings, so Char.code iterates UTF-8 bytes. *)
let entry_hash term distance =
  let hash = ref fnv_offset in
  String.iter (fun c -> hash := fnv_update !hash (Char.code c)) term;
  hash := fnv_update !hash 0x00;
  let remaining = ref (Int64.of_int distance) in
  for _ = 0 to 7 do
    hash := fnv_update !hash (Int64.to_int (Int64.logand !remaining 0xFFL));
    remaining := Int64.shift_right_logical !remaining 8
  done;
  !hash

(* 16 lowercase hex digits; %Lx formats the Int64 bit pattern unsigned. *)
let checksum_hex value = Printf.sprintf "%016Lx" value

let self_test () =
  let expect actual wanted label =
    if not (Int64.equal actual wanted) then
      die
        (Printf.sprintf "checksum self-test failed for %s: got %Lx, want %Lx"
           label actual wanted)
  in
  let fnv_string s =
    String.fold_left (fun h c -> fnv_update h (Char.code c)) fnv_offset s
  in
  expect (fnv_string "") 0xcbf29ce484222325L "fnv1a64(\"\")";
  expect (fnv_string "a") 0xaf63dc4c8601ec8cL "fnv1a64(\"a\")";
  expect (entry_hash "cat" 1) 0x9697fa3e50464bc4L "entry(cat,1)";
  expect (entry_hash "cat" 0) 0xb592c1475b3595e5L "entry(cat,0)";
  expect (entry_hash "cot" 1) 0xb8acc5d3816bcdeaL "entry(cot,1)";
  expect
    (Int64.add (entry_hash "cat" 0) (entry_hash "cot" 1))
    0x6e3f871adca163cfL "checksum{2}";
  expect 0L 0L "checksum{}";
  if String.compare (checksum_hex (-1L)) "ffffffffffffffff" <> 0 then
    die "hex serialization of high-bit checksums is not unsigned"

(* ------------------------------------------------------------------ *)
(* CLI contract (PROTOCOL.md §1)                                       *)
(* ------------------------------------------------------------------ *)

type args = {
  mutable mode : string;
  mutable algorithm : string option;
  mutable max_distance : int;
  mutable dictionary : string option;
  mutable queries : string option;
  mutable backend : string;
  mutable out : string option;
  mutable samples : int;
  mutable warmup_seconds : float;
  mutable gate_limit : int;
  mutable reps : int;
  mutable cells : string option;
}

let parse_int flag value =
  match int_of_string_opt value with
  | Some parsed -> parsed
  | None -> die (Printf.sprintf "%s expects an integer, got %S" flag value)

let parse_float flag value =
  match float_of_string_opt value with
  | Some parsed -> parsed
  | None -> die (Printf.sprintf "%s expects a number, got %S" flag value)

let parse_args () =
  let args =
    {
      mode = "";
      algorithm = None;
      max_distance = -1;
      dictionary = None;
      queries = None;
      backend = "";
      out = None;
      samples = 30;
      warmup_seconds = 3.0;
      gate_limit = 200;
      reps = 10;
      cells = None;
    }
  in
  let argv = Sys.argv in
  let total = Array.length argv in
  let index = ref 1 in
  while !index + 1 < total do
    let flag = argv.(!index) in
    let value = argv.(!index + 1) in
    (match flag with
     | "--mode" -> args.mode <- value
     | "--algorithm" -> args.algorithm <- Some value
     | "--max-distance" -> args.max_distance <- parse_int flag value
     | "--dictionary" -> args.dictionary <- Some value
     | "--queries" -> args.queries <- Some value
     | "--backend" -> args.backend <- value
     | "--out" -> args.out <- Some value
     | "--samples" -> args.samples <- parse_int flag value
     | "--warmup-seconds" -> args.warmup_seconds <- parse_float flag value
     | "--gate-limit" -> args.gate_limit <- parse_int flag value
     | "--reps" -> args.reps <- parse_int flag value
     | "--cells" -> args.cells <- Some value
     | other -> die ("unknown flag: " ^ other));
    index := !index + 2
  done;
  if !index < total then die ("dangling argument: " ^ argv.(!index));
  if args.mode = "" || args.dictionary = None || args.backend = "" then
    die "--mode, --dictionary, --backend are required";
  args

(* ------------------------------------------------------------------ *)
(* Input loading (PROTOCOL.md §3)                                      *)
(* ------------------------------------------------------------------ *)

let read_lines path =
  let content =
    match In_channel.with_open_bin path In_channel.input_all with
    | content -> content
    | exception Sys_error message -> die message
  in
  let lines =
    List.filter (fun line -> line <> "") (String.split_on_char '\n' content)
  in
  match lines with
  | [] -> die (path ^ " contains no lines")
  | _ -> Array.of_list lines

(* OCaml String.compare is byte-lexicographic: exactly the §3 invariant. *)
let assert_strictly_sorted lines path =
  for i = 0 to Array.length lines - 2 do
    if String.compare lines.(i) lines.(i + 1) >= 0 then
      die
        (Printf.sprintf "%s is not strictly byte-sorted at line %d: %S >= %S"
           path (i + 1) lines.(i) lines.(i + 1))
  done

(* ------------------------------------------------------------------ *)
(* Dictionary, transducer, and the pass (PROTOCOL.md §4–5)             *)
(* ------------------------------------------------------------------ *)

type side = {
  prepared : (string * int64 option) array; (* entry prep happens once *)
  mutable dictionary : Dict.t option;
  mutable transducer : Lev.transducer option;
}

let build_dictionary side backend =
  match backend with
  | "dynamic_dawg" ->
    let dawg = Dict.dynamic_dawg () in
    let inserted = Dict.put_many dawg side.prepared in
    if inserted <> Array.length side.prepared then
      die
        (Printf.sprintf "batch insert count mismatch: %d != %d" inserted
           (Array.length side.prepared));
    side.dictionary <- Some dawg
  | other ->
    die ("unsupported backend for the OCaml target (dynamic_dawg only): "
         ^ other)

let free_dictionary side =
  (match side.transducer with
   | Some automaton ->
     Lev.close_transducer automaton;
     side.transducer <- None
   | None -> ());
  match side.dictionary with
  | Some dictionary ->
    Dict.close dictionary;
    side.dictionary <- None
  | None -> ()

let algorithm_of_name name =
  match name with
  | "standard" -> Lev.Standard
  | "transposition" -> Lev.Transposition
  | "merge_and_split" -> Lev.Merge_and_split
  | "damerau_levenshtein" -> Lev.Damerau_levenshtein
  | other -> die ("unknown algorithm: " ^ other)

let create_transducer side algorithm =
  (match side.transducer with
   | Some previous -> Lev.close_transducer previous
   | None -> ());
  match side.dictionary with
  | None -> die "dictionary must be built before the transducer"
  | Some dictionary ->
    side.transducer <-
      Some
        (Lev.transducer ~algorithm:(algorithm_of_name algorithm)
           (Dict.resource dictionary))

(* One full pass (§5): drain every cursor through the facade's batched
   reducer (256 matches per crossing — the declared batch_size), summing
   the O(1) triple; the FNV checksum only in untimed gate contexts.
   String.length is the UTF-8 byte length by construction in OCaml. *)
let full_pass side queries max_distance with_checksum =
  let automaton =
    match side.transducer with
    | Some automaton -> automaton
    | None -> die "create_transducer must run before full_pass"
  in
  let matches = ref 0 in
  let term_bytes = ref 0 in
  let distance_sum = ref 0 in
  let checksum = ref 0L in
  Array.iter
    (fun query ->
       let cursor = Lev.query automaton query ~maximum_distance:max_distance in
       Lev.fold_batches cursor () (fun () batch ->
         Array.iter
           (fun result ->
              match result.Lev.term with
              | Lev.Text text ->
                incr matches;
                term_bytes := !term_bytes + String.length text;
                distance_sum := !distance_sum + result.Lev.distance;
                if with_checksum then
                  checksum :=
                    Int64.add !checksum (entry_hash text result.Lev.distance)
              | Lev.Tokens _ ->
                die "unexpected token-domain match for a text query")
           batch);
       Lev.cursor_close cursor)
    queries;
  (!matches, !term_bytes, !distance_sum, !checksum)

let triple_equals (m1, b1, d1) (m2, b2, d2) = m1 = m2 && b1 = b2 && d1 = d2

(* ------------------------------------------------------------------ *)
(* Result JSON (PROTOCOL.md §11 — runner post-fills run_id, sha256s,   *)
(* cell_snapshot, environment_ref)                                     *)
(* ------------------------------------------------------------------ *)

let escape_json value =
  let buffer = Buffer.create (String.length value + 8) in
  String.iter
    (fun c ->
       match c with
       | '"' -> Buffer.add_string buffer "\\\""
       | '\\' -> Buffer.add_string buffer "\\\\"
       | '\n' -> Buffer.add_string buffer "\\n"
       | '\r' -> Buffer.add_string buffer "\\r"
       | '\t' -> Buffer.add_string buffer "\\t"
       | c when Char.code c < 0x20 ->
         Buffer.add_string buffer (Printf.sprintf "\\u%04x" (Char.code c))
       | c -> Buffer.add_char buffer c)
    value;
  Buffer.contents buffer

let rec mkdir_p path =
  if path <> "" && path <> "/" && path <> "." && not (Sys.file_exists path)
  then begin
    mkdir_p (Filename.dirname path);
    match Unix.mkdir path 0o755 with
    | () -> ()
    | exception Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  end

let timestamp_utc () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ" (tm.Unix.tm_year + 1900)
    (tm.Unix.tm_mon + 1) tm.Unix.tm_mday tm.Unix.tm_hour tm.Unix.tm_min
    tm.Unix.tm_sec

let queryset_of_path path =
  Filename.remove_extension (Filename.basename path)

let join_int64s values =
  String.concat ", " (List.map Int64.to_string (Array.to_list values))

let write_result ~out ~(args : args) ~mode ~algorithm ~max_distance
    ~queries_path ~query_count ~term_count ~construct_ns ~warmup_passes
    ~samples_ns ~triple:(matches, term_bytes, distance_sum) ~checksum
    ~construct_times ~status ~notes =
  let buffer = Buffer.create 4096 in
  let add = Buffer.add_string buffer in
  add "{\n";
  add "  \"schema_version\": \"1.0.0\",\n";
  add "  \"suite\": \"cross-language-v1\",\n";
  add (Printf.sprintf "  \"timestamp_utc\": \"%s\",\n" (timestamp_utc ()));
  add "  \"target\": {\n";
  add "    \"language\": \"ocaml\",\n";
  add "    \"implementation\": \"vinary-tree\",\n";
  add "    \"backend\": \"ctypes-stubs\",\n";
  add
    (Printf.sprintf "    \"runtime_version\": \"OCaml %s\",\n"
       (escape_json Sys.ocaml_version));
  add "    \"library_version\": \"0.10.0\",\n";
  add
    "    \"artifact\": { \"kind\": \"local-build\", \"id\": \
     \"vinary-tree-liblevenshtein@0.10.0 (opam switch)\" }\n";
  add "  },\n";
  add "  \"dictionary\": {\n";
  add
    (Printf.sprintf "    \"file\": \"%s\",\n"
       (escape_json (Option.value args.dictionary ~default:"")));
  add (Printf.sprintf "    \"term_count\": %d,\n" term_count);
  add "    \"structure\": \"dynamic_dawg\",\n";
  add "    \"unit_domain\": \"unicode_scalar\"";
  (match construct_ns with
   | Some value -> add (Printf.sprintf ",\n    \"construct_ns\": %Ld\n" value)
   | None -> add "\n");
  add "  },\n";
  add "  \"workload\": {\n";
  add
    (Printf.sprintf "    \"queryset\": \"%s\",\n"
       (escape_json (queryset_of_path queries_path)));
  add
    (Printf.sprintf "    \"file\": \"%s\",\n" (escape_json queries_path));
  add (Printf.sprintf "    \"query_count\": %d\n" query_count);
  add "  },\n";
  add (Printf.sprintf "  \"algorithm\": \"%s\",\n" algorithm);
  add (Printf.sprintf "  \"max_distance\": %d,\n" max_distance);
  add
    (Printf.sprintf "  \"mode\": \"%s\",\n"
       (match mode with "memory-child" -> "memory" | other -> other));
  add "  \"protocol\": {\n";
  add "    \"timer\": \"monotonic\",\n";
  add "    \"harness\": \"self-timed\",\n";
  add
    (Printf.sprintf "    \"warmup_seconds_min\": %g,\n" args.warmup_seconds);
  add (Printf.sprintf "    \"warmup_passes\": %d,\n" warmup_passes);
  add
    (Printf.sprintf "    \"samples_requested\": %d,\n"
       (match mode with
        | "construct" -> args.reps
        | "query" -> args.samples
        | _ -> 0));
  add
    (Printf.sprintf "    \"sample_definition\": \"%s\",\n"
       (escape_json sample_definition));
  add "    \"batch_size\": 256,\n";
  add
    (Printf.sprintf "    \"wall_cap_seconds\": %d\n"
       (int_of_float wall_cap_seconds));
  add "  },\n";
  (match construct_times with
   | Some times ->
     add "  \"construct\": {\n";
     add (Printf.sprintf "    \"reps\": %d,\n" (Array.length times));
     add (Printf.sprintf "    \"times_ns\": [%s],\n" (join_int64s times));
     add (Printf.sprintf "    \"term_count\": %d\n" term_count);
     add "  },\n"
   | None ->
     add "  \"measurements\": {\n";
     add
       (Printf.sprintf "    \"samples_ns\": [%s],\n" (join_int64s samples_ns));
     add
       (Printf.sprintf "    \"sample_count\": %d,\n" (Array.length samples_ns));
     add (Printf.sprintf "    \"matches_per_pass\": %d,\n" matches);
     add (Printf.sprintf "    \"term_bytes_per_pass\": %d,\n" term_bytes);
     add (Printf.sprintf "    \"distance_sum_per_pass\": %d,\n" distance_sum);
     add
       (Printf.sprintf "    \"checksum_hex\": \"%s\"\n" (checksum_hex checksum));
     add "  },\n");
  add (Printf.sprintf "  \"status\": \"%s\",\n" status);
  add "  \"notes\": [";
  List.iteri
    (fun i note ->
       if i > 0 then add ", ";
       add (Printf.sprintf "\"%s\"" (escape_json note)))
    notes;
  add "]\n}\n";
  mkdir_p (Filename.dirname out);
  Out_channel.with_open_bin out (fun channel ->
    Out_channel.output_string channel (Buffer.contents buffer))

(* ------------------------------------------------------------------ *)
(* Modes (PROTOCOL.md §6) and the batch driver (§7)                    *)
(* ------------------------------------------------------------------ *)

let run_construct args side term_count =
  let out =
    match args.out with
    | Some out -> out
    | None -> die "--out is required for construct mode"
  in
  build_dictionary side args.backend;
  (* warmup build *)
  free_dictionary side;
  let times = Array.make (max args.reps 1) 0L in
  for rep = 0 to args.reps - 1 do
    let started = now_ns () in
    build_dictionary side args.backend;
    times.(rep) <- Int64.sub (now_ns ()) started;
    free_dictionary side
  done;
  write_result ~out ~args ~mode:"construct" ~algorithm:"standard"
    ~max_distance:1
    ~queries_path:(Option.value args.queries
                     ~default:"workload/queries/hits.txt")
    ~query_count:1 ~term_count ~construct_ns:None ~warmup_passes:1
    ~samples_ns:[||] ~triple:(0, 0, 0) ~checksum:0L
    ~construct_times:(Some times) ~status:"ok"
    ~notes:
      (base_notes
       @ [ "construct mode: timed region is the build from the pre-sorted \
            in-memory list only" ])

let run_query_cell args side queries algorithm max_distance queries_path out
    term_count construct_ns =
  let gate = full_pass side queries max_distance true in
  let gate_triple = match gate with m, b, d, _ -> (m, b, d) in
  let gate_checksum = match gate with _, _, _, c -> c in
  let warm_start = now_ns () in
  let warmup_budget = Int64.of_float (args.warmup_seconds *. 1e9) in
  let warmup_passes = ref 0 in
  let last_pass_ns = ref 0L in
  while
    Int64.compare (Int64.sub (now_ns ()) warm_start) warmup_budget < 0
    || !warmup_passes < 2
  do
    let started = now_ns () in
    let m, b, d, _ = full_pass side queries max_distance false in
    last_pass_ns := Int64.sub (now_ns ()) started;
    if not (triple_equals (m, b, d) gate_triple) then
      die "nondeterministic result during warmup";
    incr warmup_passes
  done;
  let last_pass_seconds = Int64.to_float !last_pass_ns /. 1e9 in
  let sample_count = ref args.samples in
  let status = ref "ok" in
  let notes = ref base_notes in
  if float_of_int !sample_count *. last_pass_seconds > wall_cap_seconds then begin
    let reduced =
      max 10 (int_of_float (wall_cap_seconds /. last_pass_seconds))
    in
    notes :=
      !notes
      @ [ Printf.sprintf
            "samples reduced from %d to %d by the %.0fs wall cap (estimated \
             pass %.3fs)"
            !sample_count reduced wall_cap_seconds last_pass_seconds ];
    sample_count := reduced;
    status := "degraded"
  end;
  let samples_ns = Array.make !sample_count 0L in
  for i = 0 to !sample_count - 1 do
    let started = now_ns () in
    let m, b, d, _ = full_pass side queries max_distance false in
    samples_ns.(i) <- Int64.sub (now_ns ()) started;
    if not (triple_equals (m, b, d) gate_triple) then
      die "nondeterministic result during measurement"
  done;
  write_result ~out ~args ~mode:"query" ~algorithm ~max_distance ~queries_path
    ~query_count:(Array.length queries) ~term_count
    ~construct_ns:(Some construct_ns) ~warmup_passes:!warmup_passes
    ~samples_ns ~triple:gate_triple ~checksum:gate_checksum
    ~construct_times:None ~status:!status ~notes:!notes

let run_one args side algorithm max_distance queries_path out term_count
    construct_ns =
  create_transducer side algorithm;
  let queries = read_lines queries_path in
  match args.mode with
  | "verify" ->
    let limit = min args.gate_limit (Array.length queries) in
    let subset = Array.sub queries 0 limit in
    let m, b, d, checksum = full_pass side subset max_distance true in
    write_result ~out ~args ~mode:"verify" ~algorithm ~max_distance
      ~queries_path ~query_count:limit ~term_count
      ~construct_ns:(Some construct_ns) ~warmup_passes:0 ~samples_ns:[||]
      ~triple:(m, b, d) ~checksum ~construct_times:None ~status:"ok"
      ~notes:base_notes
  | "memory-child" ->
    let m, b, d, checksum = full_pass side queries max_distance true in
    write_result ~out ~args ~mode:"memory-child" ~algorithm ~max_distance
      ~queries_path ~query_count:(Array.length queries) ~term_count
      ~construct_ns:(Some construct_ns) ~warmup_passes:0 ~samples_ns:[||]
      ~triple:(m, b, d) ~checksum ~construct_times:None ~status:"ok"
      ~notes:base_notes
  | "query" ->
    run_query_cell args side queries algorithm max_distance queries_path out
      term_count construct_ns
  | other -> die ("unknown mode: " ^ other)

let run_cells args side cells_path term_count construct_ns =
  let rows =
    Array.to_list (read_lines cells_path)
    |> List.filter (fun line ->
      let trimmed = String.trim line in
      trimmed <> "" && not (String.length trimmed > 0 && trimmed.[0] = '#'))
  in
  List.iter
    (fun line ->
       match String.split_on_char '\t' line with
       | [ algorithm; distance; queries_path; out ] ->
         run_one args side algorithm
           (parse_int "--cells max_distance" distance)
           queries_path out term_count construct_ns
       | _ -> die ("cells row needs 4 fields: " ^ line))
    rows

let () =
  self_test ();
  let args = parse_args () in
  let dictionary_path = Option.get args.dictionary in
  let terms = read_lines dictionary_path in
  assert_strictly_sorted terms dictionary_path;
  let side =
    {
      prepared = Array.map (fun term -> (term, None)) terms;
      dictionary = None;
      transducer = None;
    }
  in
  let term_count = Array.length terms in
  match args.mode with
  | "construct" -> run_construct args side term_count
  | "query" | "verify" | "memory-child" -> begin
    let build_start = now_ns () in
    build_dictionary side args.backend;
    let construct_ns = Int64.sub (now_ns ()) build_start in
    match args.cells with
    | Some cells_path ->
      run_cells args side cells_path term_count construct_ns
    | None ->
      let algorithm =
        match args.algorithm with
        | Some algorithm -> algorithm
        | None -> die "--algorithm is required"
      in
      let queries_path =
        match args.queries with
        | Some queries -> queries
        | None -> die "--queries is required"
      in
      let out =
        match args.out with
        | Some out -> out
        | None -> die "--out is required"
      in
      if args.max_distance < 0 then die "--max-distance is required";
      run_one args side algorithm args.max_distance queries_path out
        term_count construct_ns
  end
  | other -> die ("unknown mode: " ^ other)
