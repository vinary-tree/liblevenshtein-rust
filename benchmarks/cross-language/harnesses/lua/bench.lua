#!/usr/bin/env lua5.4
-- Lua 5.4 harness for the cross-language benchmark program.
--
-- Implements harnesses/common/PROTOCOL.md over the modular C-module facades
-- (vinary_tree.libdictenstein + vinary_tree.liblevenshtein) built by
-- build.sh against the RELEASE cdylibs. The monotonic clock comes from the
-- harness-local bench_clock.so shim (section 9): os.clock() is CPU time and
-- is never used.
--
-- Checksum arithmetic (section 8): Lua 5.4 integers are native 64-bit
-- two's-complement and wrap on overflow, so +, *, ~ and & give exactly the
-- required mod-2^64 behavior; only equality is ever compared. Hex literals
-- above 2^63-1 wrap to the identical bit pattern. Serialization formats the
-- two 32-bit halves separately because string.format("%x") rejects the
-- negative integers that represent bit patterns above 2^63-1.

local clock = require("bench_clock")
local dictionary_module = require("vinary_tree.libdictenstein")
local levenshtein = require("vinary_tree.liblevenshtein")

local now_ns = clock.now_ns
local string_byte = string.byte
local string_format = string.format

local FNV_OFFSET = 0xcbf29ce484222325
local FNV_PRIME = 0x100000001b3
local BATCH_SIZE = 256
local WALL_CAP_SECONDS = 300.0
local SAMPLE_DEFINITION = "one full pass over the query set; every cursor fully drained and "
  .. "(term, distance) materialized"

local ALGORITHMS = {
  standard = "standard",
  transposition = "transposition",
  merge_and_split = "merge-and-split",
  damerau_levenshtein = "damerau-levenshtein",
}

local function die(message)
  io.stderr:write("bench-cross-lua: " .. message .. "\n")
  os.exit(2)
end

-- Lua string order (<) routes through strcoll and is locale-sensitive; pin
-- the collate category to "C" so the section 3 sortedness assertion compares
-- raw bytes exactly.
os.setlocale("C", "collate")

--------------------------------------------------------------------------
-- Checksum primitives (PROTOCOL.md section 8) + self-test (section 2)
--------------------------------------------------------------------------

local function fnv1a64(text)
  local hash = FNV_OFFSET
  for index = 1, #text do
    hash = (hash ~ string_byte(text, index)) * FNV_PRIME
  end
  return hash
end

local function entry_hash(term, distance)
  local hash = FNV_OFFSET
  for index = 1, #term do
    hash = (hash ~ string_byte(term, index)) * FNV_PRIME
  end
  hash = hash * FNV_PRIME -- separator: XOR with 0x00 is identity
  for shift = 0, 56, 8 do -- LE64(distance)
    hash = (hash ~ ((distance >> shift) & 0xff)) * FNV_PRIME
  end
  return hash
end

local function hex16(value)
  -- Two non-negative 32-bit halves: >> is a logical shift in Lua 5.4, so the
  -- high half is already in [0, 2^32) even when the integer is negative.
  return string_format("%08x%08x", (value >> 32) & 0xffffffff, value & 0xffffffff)
end

local function self_test()
  local function expect(actual, wanted, label)
    if actual ~= wanted then
      die(string_format("checksum self-test failed for %s: got %s, want %s",
                        label, hex16(actual), hex16(wanted)))
    end
  end
  expect(fnv1a64(""), 0xcbf29ce484222325, 'fnv1a64("")')
  expect(fnv1a64("a"), 0xaf63dc4c8601ec8c, 'fnv1a64("a")')
  expect(entry_hash("cat", 1), 0x9697fa3e50464bc4, "entry(cat,1)")
  expect(entry_hash("cat", 0), 0xb592c1475b3595e5, "entry(cat,0)")
  expect(entry_hash("cot", 1), 0xb8acc5d3816bcdea, "entry(cot,1)")
  expect(entry_hash("cat", 0) + entry_hash("cot", 1), 0x6e3f871adca163cf, "checksum{2}")
  expect(0, 0x0000000000000000, "checksum{}")
end

--------------------------------------------------------------------------
-- CLI (PROTOCOL.md section 1)
--------------------------------------------------------------------------

local function parse_integer(flag, value)
  local parsed = math.tointeger(tonumber(value))
  if parsed == nil then
    die(flag .. " expects an integer, got: " .. tostring(value))
  end
  return parsed
end

local function parse_args(argv)
  local args = {
    mode = nil, algorithm = nil, max_distance = -1, dictionary = nil,
    queries = nil, backend = nil, out = nil, samples = 30,
    warmup_seconds = 3.0, gate_limit = 200, reps = 10, cells = nil,
  }
  local index = 1
  while index <= #argv do
    local flag = argv[index]
    local value = argv[index + 1]
    if value == nil then die("flag requires a value: " .. flag) end
    if flag == "--mode" then args.mode = value
    elseif flag == "--algorithm" then args.algorithm = value
    elseif flag == "--max-distance" then args.max_distance = parse_integer(flag, value)
    elseif flag == "--dictionary" then args.dictionary = value
    elseif flag == "--queries" then args.queries = value
    elseif flag == "--backend" then args.backend = value
    elseif flag == "--out" then args.out = value
    elseif flag == "--samples" then args.samples = parse_integer(flag, value)
    elseif flag == "--warmup-seconds" then
      local parsed = tonumber(value)
      if parsed == nil then die("--warmup-seconds expects a number, got: " .. tostring(value)) end
      args.warmup_seconds = parsed
    elseif flag == "--gate-limit" then args.gate_limit = parse_integer(flag, value)
    elseif flag == "--reps" then args.reps = parse_integer(flag, value)
    elseif flag == "--cells" then args.cells = value
    else die("unknown flag: " .. flag)
    end
    index = index + 2
  end
  if args.mode == nil or args.dictionary == nil or args.backend == nil then
    die("--mode, --dictionary, --backend are required")
  end
  return args
end

--------------------------------------------------------------------------
-- Input loading (PROTOCOL.md section 3)
--------------------------------------------------------------------------

local function read_lines(path)
  local handle = io.open(path, "rb")
  if handle == nil then die("cannot open " .. path) end
  local data = handle:read("a")
  handle:close()
  if data == nil then die("cannot read " .. path) end
  -- Stock Lua 5.4 has no table-capacity preallocation primitive; sequential
  -- integer assignment below is the idiomatic single-growth-path equivalent,
  -- and it happens only here, outside every timed region.
  local lines = {}
  local count = 0
  local length = #data
  local start = 1
  while start <= length do
    local newline = string.find(data, "\n", start, true)
    local finish = (newline or (length + 1)) - 1
    if finish >= start then
      count = count + 1
      lines[count] = string.sub(data, start, finish)
    end
    if newline == nil then break end
    start = newline + 1
  end
  if count == 0 then die(path .. " contains no lines") end
  return lines
end

local function assert_strictly_sorted(lines, path)
  for index = 1, #lines - 1 do
    -- Byte order: < is strcoll under LC_COLLATE=C (pinned above) = memcmp.
    if not (lines[index] < lines[index + 1]) then
      die(string_format("%s is not strictly byte-sorted at line %d: %q >= %q",
                        path, index, lines[index], lines[index + 1]))
    end
  end
end

--------------------------------------------------------------------------
-- Dictionary + transducer side (PROTOCOL.md section 4)
--------------------------------------------------------------------------

local side = {
  dictionary = nil,
  transducer = nil,
  prepared_entries = nil,
}

function side.prepare_entries(terms)
  if side.prepared_entries == nil then
    local prepared = {}
    for index = 1, #terms do prepared[index] = { terms[index] } end
    side.prepared_entries = prepared
  end
  return side.prepared_entries
end

function side.build_dictionary(terms, backend)
  if backend == "dynamic_dawg" then
    local words = dictionary_module.dynamic_dawg("unicode")
    -- The Lua facade exposes no batch insert (unlike PutAll/put_all
    -- elsewhere); this put() loop IS the facade's bulk path and is recorded
    -- in the cell notes as PROTOCOL.md section 4 requires.
    for index = 1, #terms do
      words:put(terms[index])
    end
    if words:len() ~= #terms then
      die(string_format("dynamic_dawg term count mismatch: %d != %d", words:len(), #terms))
    end
    side.dictionary = words
  elseif backend == "double_array_trie" then
    local trie = dictionary_module.double_array_trie(side.prepare_entries(terms), "unicode")
    if trie:len() ~= #terms then
      die(string_format("double_array_trie term count mismatch: %d != %d", trie:len(), #terms))
    end
    side.dictionary = trie
  else
    die("unknown backend: " .. backend)
  end
end

function side.free_dictionary()
  if side.transducer ~= nil then
    side.transducer:close()
    side.transducer = nil
  end
  if side.dictionary ~= nil then
    side.dictionary:close()
    side.dictionary = nil
  end
end

function side.create_transducer(algorithm)
  if side.transducer ~= nil then
    side.transducer:close()
    side.transducer = nil
  end
  local mapped = ALGORITHMS[algorithm]
  if mapped == nil then die("unknown algorithm: " .. algorithm) end
  side.transducer = levenshtein.transducer(side.dictionary, mapped)
end

--------------------------------------------------------------------------
-- Passes (PROTOCOL.md section 5)
--------------------------------------------------------------------------

-- One full pass: every cursor fully drained via the facade's iterator
-- (__call -> cursor:next(), 256-match native batches underneath), (term,
-- distance) materialized per match, cursor closed. Timed passes accumulate
-- only the O(1) triple; the checksum is computed exclusively in untimed
-- gate/verify contexts.
local function full_pass(queries, limit, max_distance, with_checksum)
  local matches = 0
  local term_bytes = 0
  local distance_sum = 0
  local checksum = 0
  local automaton = side.transducer
  for index = 1, limit do
    local cursor = automaton:query(queries[index], max_distance)
    for match in cursor do
      local term = match.term
      matches = matches + 1
      term_bytes = term_bytes + #term -- UTF-8 byte length
      distance_sum = distance_sum + match.distance
      if with_checksum then
        checksum = checksum + entry_hash(term, match.distance)
      end
    end
    cursor:close()
  end
  return matches, term_bytes, distance_sum, checksum
end

--------------------------------------------------------------------------
-- Result JSON (PROTOCOL.md section 11 — runner post-fills run_id, sha256s,
-- cell_snapshot, environment_ref, and the memory object)
--------------------------------------------------------------------------

local function json_escape(text)
  return (text:gsub('[%c"\\]', function(character)
    if character == '"' then return '\\"' end
    if character == "\\" then return "\\\\" end
    if character == "\n" then return "\\n" end
    if character == "\r" then return "\\r" end
    if character == "\t" then return "\\t" end
    return string_format("\\u%04x", string_byte(character))
  end))
end

local function json_number(value)
  if math.type(value) == "integer" then
    return string_format("%d", value)
  end
  return tostring(value)
end

local function queryset_stem(path)
  local base = path:match("([^/]+)$") or path
  return (base:gsub("%.txt$", ""))
end

local function ensure_parent_directory(path)
  local directory = path:match("^(.*)/[^/]*$")
  if directory ~= nil and directory ~= "" then
    os.execute("mkdir -p '" .. directory:gsub("'", "'\\''") .. "'")
  end
end

local function render_result(args, mode, algorithm, max_distance, queries_path, query_count,
                             term_count, backend, construct_ns, warmup_passes, samples_ns,
                             triple, checksum, construct_times, status, notes)
  local samples_requested = 0
  if mode == "construct" then
    samples_requested = args.reps
  elseif mode == "query" then
    samples_requested = args.samples
  end
  local pieces = {}
  local function emit(text) pieces[#pieces + 1] = text end
  emit('{\n')
  emit('  "schema_version": "1.0.0",\n')
  emit('  "suite": "cross-language-v1",\n')
  emit('  "timestamp_utc": "' .. os.date("!%Y-%m-%dT%H:%M:%SZ") .. '",\n')
  emit('  "target": {\n')
  emit('    "language": "lua",\n')
  emit('    "implementation": "vinary-tree",\n')
  emit('    "backend": "lua-cmodule",\n')
  emit('    "runtime_version": "' .. json_escape(_VERSION) .. '",\n')
  emit('    "library_version": "0.10.0",\n')
  emit('    "artifact": { "kind": "local-build", "id": "liblevenshtein@0.10.0" }\n')
  emit('  },\n')
  emit('  "dictionary": {\n')
  emit('    "file": "' .. json_escape(args.dictionary) .. '",\n')
  emit('    "term_count": ' .. json_number(term_count) .. ',\n')
  emit('    "structure": "' .. backend .. '",\n')
  emit('    "unit_domain": "unicode_scalar"')
  if construct_ns ~= nil then
    emit(',\n    "construct_ns": ' .. json_number(construct_ns) .. '\n')
  else
    emit('\n')
  end
  emit('  },\n')
  emit('  "workload": {\n')
  emit('    "queryset": "' .. json_escape(queryset_stem(queries_path)) .. '",\n')
  emit('    "file": "' .. json_escape(queries_path) .. '",\n')
  emit('    "query_count": ' .. json_number(query_count) .. '\n')
  emit('  },\n')
  emit('  "algorithm": "' .. algorithm .. '",\n')
  emit('  "max_distance": ' .. json_number(max_distance) .. ',\n')
  emit('  "mode": "' .. (mode == "memory-child" and "memory" or mode) .. '",\n')
  emit('  "protocol": {\n')
  emit('    "timer": "monotonic",\n')
  emit('    "harness": "self-timed",\n')
  emit('    "warmup_seconds_min": ' .. json_number(args.warmup_seconds) .. ',\n')
  emit('    "warmup_passes": ' .. json_number(warmup_passes) .. ',\n')
  emit('    "samples_requested": ' .. json_number(samples_requested) .. ',\n')
  emit('    "sample_definition": "' .. json_escape(SAMPLE_DEFINITION) .. '",\n')
  emit('    "batch_size": ' .. json_number(BATCH_SIZE) .. ',\n')
  emit('    "wall_cap_seconds": ' .. json_number(math.tointeger(WALL_CAP_SECONDS)) .. '\n')
  emit('  },\n')
  if construct_times ~= nil then
    emit('  "construct": {\n')
    emit('    "reps": ' .. json_number(#construct_times) .. ',\n')
    local rendered = {}
    for index = 1, #construct_times do rendered[index] = json_number(construct_times[index]) end
    emit('    "times_ns": [' .. table.concat(rendered, ", ") .. '],\n')
    emit('    "term_count": ' .. json_number(term_count) .. '\n')
    emit('  },\n')
  else
    emit('  "measurements": {\n')
    local rendered = {}
    for index = 1, #samples_ns do rendered[index] = json_number(samples_ns[index]) end
    emit('    "samples_ns": [' .. table.concat(rendered, ", ") .. '],\n')
    emit('    "sample_count": ' .. json_number(#samples_ns) .. ',\n')
    emit('    "matches_per_pass": ' .. json_number(triple[1]) .. ',\n')
    emit('    "term_bytes_per_pass": ' .. json_number(triple[2]) .. ',\n')
    emit('    "distance_sum_per_pass": ' .. json_number(triple[3]) .. ',\n')
    emit('    "checksum_hex": "' .. hex16(checksum) .. '"\n')
    emit('  },\n')
  end
  emit('  "status": "' .. status .. '",\n')
  local rendered_notes = {}
  for index = 1, #notes do
    rendered_notes[index] = '"' .. json_escape(notes[index]) .. '"'
  end
  emit('  "notes": [' .. table.concat(rendered_notes, ", ") .. ']\n')
  emit('}\n')
  return table.concat(pieces)
end

local function write_result(out_path, rendered)
  ensure_parent_directory(out_path)
  local handle = io.open(out_path, "wb")
  if handle == nil then die("cannot write " .. out_path) end
  handle:write(rendered)
  handle:close()
end

--------------------------------------------------------------------------
-- Modes (PROTOCOL.md sections 6-7)
--------------------------------------------------------------------------

local function run_query_cell(args, queries, algorithm, max_distance, queries_path, out_path,
                              term_count, construct_ns, base_notes)
  -- Untimed gate pass: checksum + reference triple.
  local gate_matches, gate_bytes, gate_distance, gate_checksum =
    full_pass(queries, #queries, max_distance, true)

  local warm_start = now_ns()
  local warmup_budget_ns = math.floor(args.warmup_seconds * 1e9 + 0.5)
  local warmup_passes = 0
  local last_pass_ns = 0
  while (now_ns() - warm_start) < warmup_budget_ns or warmup_passes < 2 do
    local t0 = now_ns()
    local matches, term_bytes, distance_sum = full_pass(queries, #queries, max_distance, false)
    last_pass_ns = now_ns() - t0
    if matches ~= gate_matches or term_bytes ~= gate_bytes or distance_sum ~= gate_distance then
      die("nondeterministic result during warmup")
    end
    warmup_passes = warmup_passes + 1
  end

  local sample_count = args.samples
  local status = "ok"
  local notes = {}
  for index = 1, #base_notes do notes[index] = base_notes[index] end
  local last_pass_seconds = last_pass_ns / 1e9
  if sample_count * last_pass_seconds > WALL_CAP_SECONDS then
    local reduced = math.floor(WALL_CAP_SECONDS / last_pass_seconds)
    if reduced < 10 then reduced = 10 end
    notes[#notes + 1] = string_format(
      "samples reduced from %d to %d by the %ds wall cap (estimated pass %.3fs)",
      sample_count, reduced, math.tointeger(WALL_CAP_SECONDS), last_pass_seconds)
    sample_count = reduced
    status = "degraded"
  end

  local samples_ns = {}
  for index = 1, sample_count do
    local t0 = now_ns()
    local matches, term_bytes, distance_sum = full_pass(queries, #queries, max_distance, false)
    samples_ns[index] = now_ns() - t0
    if matches ~= gate_matches or term_bytes ~= gate_bytes or distance_sum ~= gate_distance then
      die("nondeterministic result during measurement")
    end
  end

  write_result(out_path, render_result(args, "query", algorithm, max_distance, queries_path,
                                       #queries, term_count, args.backend, construct_ns,
                                       warmup_passes, samples_ns,
                                       { gate_matches, gate_bytes, gate_distance },
                                       gate_checksum, nil, status, notes))
end

local function main()
  self_test()
  local args = parse_args(arg)

  local terms = read_lines(args.dictionary)
  assert_strictly_sorted(terms, args.dictionary)

  local base_notes = {
    "lua-cmodule facade (bindings/lua) compiled by build.sh against the release cdylibs",
    "dynamic_dawg construction uses the facade's put() loop (no batch API in the Lua facade)",
    "monotonic clock via harness-local bench_clock.so shim (clock_gettime(CLOCK_MONOTONIC))",
  }

  if args.mode == "construct" then
    if args.out == nil then die("--out is required for construct mode") end
    side.build_dictionary(terms, args.backend) -- warmup build (section 6.2)
    side.free_dictionary()
    local times = {}
    for rep = 1, args.reps do
      local t0 = now_ns()
      side.build_dictionary(terms, args.backend)
      times[rep] = now_ns() - t0
      side.free_dictionary()
    end
    local notes = {}
    for index = 1, #base_notes do notes[index] = base_notes[index] end
    notes[#notes + 1] =
      "construct mode: timed region is the build from the pre-sorted in-memory list only"
    write_result(args.out, render_result(args, "construct", "standard", 1,
                                         args.queries or "workload/queries/hits.txt", 1,
                                         #terms, args.backend, nil, 1, {}, { 0, 0, 0 }, 0,
                                         times, "ok", notes))
    return
  end

  local build_start = now_ns()
  side.build_dictionary(terms, args.backend)
  local construct_ns = now_ns() - build_start

  local function run_one(algorithm, max_distance, queries_path, out_path)
    side.create_transducer(algorithm)
    local queries = read_lines(queries_path)
    if args.mode == "verify" then
      local limit = math.min(args.gate_limit, #queries)
      local matches, term_bytes, distance_sum, checksum =
        full_pass(queries, limit, max_distance, true)
      write_result(out_path, render_result(args, "verify", algorithm, max_distance,
                                           queries_path, limit, #terms, args.backend,
                                           construct_ns, 0, {},
                                           { matches, term_bytes, distance_sum }, checksum,
                                           nil, "ok", base_notes))
    elseif args.mode == "memory-child" then
      local matches, term_bytes, distance_sum, checksum =
        full_pass(queries, #queries, max_distance, true)
      write_result(out_path, render_result(args, "memory-child", algorithm, max_distance,
                                           queries_path, #queries, #terms, args.backend,
                                           construct_ns, 0, {},
                                           { matches, term_bytes, distance_sum }, checksum,
                                           nil, "ok", base_notes))
    elseif args.mode == "query" then
      run_query_cell(args, queries, algorithm, max_distance, queries_path, out_path,
                     #terms, construct_ns, base_notes)
    else
      die("unknown mode: " .. args.mode)
    end
  end

  if args.cells ~= nil then
    local handle = io.open(args.cells, "rb")
    if handle == nil then die("cannot open cells file " .. args.cells) end
    local content = handle:read("a")
    handle:close()
    for line in content:gmatch("[^\n]+") do
      line = line:gsub("\r$", "")
      if line ~= "" and line:sub(1, 1) ~= "#" then
        local fields = {}
        for field in line:gmatch("[^\t]+") do fields[#fields + 1] = field end
        if #fields ~= 4 then die("cells row needs 4 tab-separated fields: " .. line) end
        run_one(fields[1], parse_integer("cells max_distance", fields[2]), fields[3], fields[4])
      end
    end
  else
    if args.algorithm == nil or args.max_distance < 0 or args.queries == nil
        or args.out == nil then
      die("--algorithm, --max-distance, --queries, --out are required")
    end
    run_one(args.algorithm, args.max_distance, args.queries, args.out)
  end
end

main()
