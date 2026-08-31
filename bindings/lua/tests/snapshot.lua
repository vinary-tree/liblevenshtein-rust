local dictionary = require("vinary_tree.libdictenstein")
local levenshtein = require("vinary_tree.liblevenshtein")

assert(levenshtein.status.ok == 0)
assert(levenshtein.status.end_of_stream == 1)
assert(levenshtein.status.domain_mismatch == 12)
local algorithms = {
  levenshtein.algorithm.standard,
  levenshtein.algorithm.transposition,
  levenshtein.algorithm.merge_and_split,
  levenshtein.algorithm.damerau_levenshtein,
}
local orders = {levenshtein.order.traversal, levenshtein.order.distance_then_term}
assert(#algorithms == 4 and #orders == 2)

local function collect(cursor)
  local output = {}
  for match in cursor do output[#output + 1] = {match.term, match.distance, match.id} end
  cursor:close()
  return output
end

for trace = 1, 64 do
  local words <close> = dictionary.dynamic_dawg("unicode")
  for index = 1, 16 do words:put("t" .. trace .. "-" .. index, index) end
  local automaton <close> = levenshtein.transducer(words)
  local expected = collect(automaton:query("", 64, "distance-then-term"))
  local cursor <close> = automaton:query("", 64, "distance-then-term")
  local actual = {cursor:next()}
  words:remove("t" .. trace .. "-1")
  words:put("t" .. trace .. "-2", 999)
  actual[#actual + 1] = cursor:next()
  words:clear()
  words:compact()
  words:put("after-" .. trace, 1000)
  for match in cursor do actual[#actual + 1] = match end
  assert(#actual == #expected)
  for index = 1, #actual do
    assert(actual[index].term == expected[index][1])
    assert(actual[index].distance == expected[index][2])
    assert(actual[index].id == expected[index][3])
  end
  words:close()
  local fresh = collect(automaton:query("", 64))
  assert(#fresh == 1 and fresh[1][1] == "after-" .. trace)
end

do
  local words <close> = dictionary.dynamic_dawg("unicode")
  words:put("cat", 1)
  words:put("cot", 2)
  local automaton <close> = levenshtein.transducer(words)
  local cache <close> = levenshtein.query_cache(automaton, 8, 1024 * 1024)
  local cold = collect(cache:query("cut", 1))
  local hit = collect(cache:query("cut", 1))
  assert(#cold == #hit)
  for index = 1, #cold do
    assert(cold[index][1] == hit[index][1] and cold[index][2] == hit[index][2])
  end
  local stats = cache:stats()
  assert(stats.requests == 2 and stats.hits == 1 and stats.misses == 1)
  assert(stats.resident_entries == 1 and stats.resident_weight > 0)
  assert(cache:reset_stats():stats().requests == 0)
  assert(cache:stats().resident_entries == 1)
  assert(cache:clear():stats().resident_entries == 0)
end

local dat <close> = dictionary.double_array_trie({{"café", 7}, {"caff"}}, "unicode")
assert(dat:get("café").value == 7)
assert(dat:get("caff").found and dat:get("caff").value == nil)

local suffixes <close> = dictionary.scdawg("unicode")
suffixes:put("cat", 1)
suffixes:put("cot", 2)
assert(suffixes:contains_substring("ot"))
assert(suffixes:frequency("t") == 2)
for _, algorithm in ipairs(algorithms) do
  local selected <close> = levenshtein.transducer(suffixes, algorithm)
  local exact = collect(selected:query("cat", 0, levenshtein.order.traversal))
  assert(#exact == 1 and exact[1][1] == "cat")
end

local bytes <close> = dictionary.dynamic_dawg("byte")
bytes:put("\0\255cat", 9)
local byte_automaton <close> = levenshtein.transducer(bytes)
assert(byte_automaton:domain() == "byte")
local byte_cursor <close> = byte_automaton:query_bytes("\0\255cot", 1)
local byte_batch = byte_cursor:next_batch(1)
assert(#byte_batch == 1 and byte_cursor:next_batch(1) == nil)
byte_cursor:close()
local byte_matches = {{byte_batch[1].term, byte_batch[1].distance, byte_batch[1].id}}
assert(#byte_matches == 1 and byte_matches[1][1] == "\0\255cat")
local reduction_cursor <close> = byte_automaton:query_bytes("\0\255cot", 1)
local reduced = reduction_cursor:reduce_batches(
  0,
  function(count, batch) return count + #batch end,
  1
)
assert(reduced == 1)
reduction_cursor:close()

local tokens <close> = dictionary.dynamic_dawg("u64")
assert(tokens:put_u64({1, 2, 3}, 12))
assert(tokens:contains_u64({1, 2, 3}))
assert(tokens:get_u64({1, 2, 3}).value == 12)
local token_automaton <close> = levenshtein.transducer(tokens)
assert(token_automaton:domain() == "u64")
local token_cursor <close> = token_automaton:query_u64({1, 2, 4}, 1)
local token_match = token_cursor:next()
assert(token_match.distance == 1 and token_match.term[3] == 3 and token_match.id == 12)
assert(tokens:remove_u64({1, 2, 3}))

local scratch = assert(os.getenv("VINARY_TREE_TEST_TMPDIR"),
  "VINARY_TREE_TEST_TMPDIR must name a writable non-tmpfs test directory")
local separator = package.config:sub(1, 1)
local function remove_persistence_artifacts(path)
  os.remove(path)
  os.remove(path .. ".wal")
  os.remove(path .. ".wlock")
end
local persistent_path = scratch .. separator .. "liblevenshtein-lua.artrie"
remove_persistence_artifacts(persistent_path)
do
  local persistent <close> = dictionary.create_persistent_artrie(persistent_path, "unicode")
  persistent:put("durable", 17)
  persistent:checkpoint()
end
do
  local persistent <close> = dictionary.open_persistent_artrie(persistent_path, "unicode")
  assert(persistent:get("durable").value == 17)
end

local vocabulary_path = scratch .. separator .. "liblevenshtein-lua.vocab"
remove_persistence_artifacts(vocabulary_path)
do
  local vocabulary <close> = dictionary.create_persistent_vocabulary(vocabulary_path)
  vocabulary:put("alpha", 0)
  vocabulary:checkpoint()
  assert(vocabulary:term(0) == "alpha")
end
do
  local vocabulary <close> = dictionary.open_persistent_vocabulary(vocabulary_path)
  assert(vocabulary:get("alpha").value == 0)
  assert(vocabulary:term(0) == "alpha")
end

local pattern <close> = levenshtein.phonetic_pattern("c[ao]t")
assert(pattern:matches("cat"))
assert(not pattern:matches("cut"))
local states, transitions = pattern:size()
assert(states > 0 and transitions > 0)
local llre <close> = levenshtein.llre_pattern("@name \"Greeting\"\n^hello$")
assert(llre:matches("hello") and not llre:matches("world"))
local product_automaton <close> = levenshtein.transducer(suffixes)
local product_pattern <close> = levenshtein.phonetic_pattern("c[ao]t")
local product_matches = collect(product_automaton:query_pattern(product_pattern, 0))
assert(#product_matches == 2)
local rules <close> = levenshtein.phonetic_rules(
  levenshtein.phonetic_rule_set_kind.english_orthography
)
assert(rules:len() > 0 and type(rules:apply("phone")) == "string")
local phonetic_rules <close> = levenshtein.phonetic_rules(
  levenshtein.phonetic_rule_set_kind.english_phonetic
)
assert(phonetic_rules:len() > 0)
assert(levenshtein.distance("kitten", "sitting") == 3)
assert(levenshtein.distance_threshold("kitten", "sitting", 3) == 3)
assert(levenshtein.damerau_distance("ca", "ac") == 1)
assert(levenshtein.damerau_distance_threshold("ca", "ac", 1) == 1)
assert(levenshtein.true_damerau_distance("ca", "ac") == 1)
assert(levenshtein.true_damerau_distance_threshold("ca", "ac", 1) == 1)
-- C6: distances count Unicode scalars, not bytes/UTF-16 code units.
assert(levenshtein.distance("café", "cafe") == 1)
assert(levenshtein.distance("🦀", "x") == 1)
assert(levenshtein.distance("é", "e") == 1)
-- C6: the exceeded-bound native sentinel (usize::MAX - 1) wraps to -2 under the
-- size_t -> lua_Integer conversion; "ca" -> "abc" separates OSA 3 from true 2.
assert(levenshtein.distance_threshold("kitten", "sitting", 2) == -2)
assert(levenshtein.damerau_distance_threshold("ca", "abc", 2) == -2)
assert(levenshtein.true_damerau_distance_threshold("ca", "abc", 2) == 2)
remove_persistence_artifacts(persistent_path)
remove_persistence_artifacts(vocabulary_path)
print("Lua binding snapshot integration passed")
