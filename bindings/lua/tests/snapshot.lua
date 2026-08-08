local dictionary = require("vinary_tree.libdictenstein")
local levenshtein = require("vinary_tree.liblevenshtein")

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

local dat <close> = dictionary.double_array_trie({{"café", 7}, {"caff"}}, "unicode")
assert(dat:get("café").value == 7)
assert(dat:get("caff").found and dat:get("caff").value == nil)

local suffixes <close> = dictionary.scdawg("unicode")
suffixes:put("cat", 1)
suffixes:put("cot", 2)
assert(suffixes:contains_substring("ot"))
assert(suffixes:frequency("t") == 2)

local bytes <close> = dictionary.dynamic_dawg("byte")
bytes:put("\0\255cat", 9)
local byte_automaton <close> = levenshtein.transducer(bytes)
assert(byte_automaton:domain() == "byte")
local byte_matches = collect(byte_automaton:query_bytes("\0\255cot", 1))
assert(#byte_matches == 1 and byte_matches[1][1] == "\0\255cat")

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

local persistent_path = os.tmpname()
os.remove(persistent_path)
do
  local persistent <close> = dictionary.create_persistent_artrie(persistent_path, "unicode")
  persistent:put("durable", 17)
  persistent:checkpoint()
end
do
  local persistent <close> = dictionary.open_persistent_artrie(persistent_path, "unicode")
  assert(persistent:get("durable").value == 17)
end

local vocabulary_path = os.tmpname()
os.remove(vocabulary_path)
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
local rules <close> = levenshtein.phonetic_rules("english-orthography")
assert(rules:len() > 0 and type(rules:apply("phone")) == "string")
assert(levenshtein.distance("kitten", "sitting") == 3)
assert(levenshtein.distance_threshold("kitten", "sitting", 3) == 3)
assert(levenshtein.damerau_distance("ca", "ac") == 1)
assert(levenshtein.damerau_distance_threshold("ca", "ac", 1) == 1)
assert(levenshtein.true_damerau_distance("ca", "ac") == 1)
assert(levenshtein.true_damerau_distance_threshold("ca", "ac", 1) == 1)
os.remove(persistent_path)
os.remove(persistent_path .. ".wal")
os.remove(vocabulary_path)
os.remove(vocabulary_path .. ".wal")
print("Lua binding snapshot integration passed")
