-- Lua 5.4: the <close> variables deterministically release native resources.
local dictionaries = require("vinary_tree.libdictenstein")
local levenshtein = require("vinary_tree.liblevenshtein")

local words <close> = dictionaries.dynamic_dawg("unicode")
words:put("cat", 10)
words:put("cot", 20)
words:put("cut") -- Membership without a mapped value.

local search <close> = levenshtein.transducer(
  words, levenshtein.algorithm.standard
)
local results <close> = search:query(
  "cet", 1, levenshtein.order.distance_then_term
)
local found = {}
for match in results do
  -- match.term is a Lua string; match.id is absent for a valueless term.
  found[match.term] = { distance = match.distance, id = match.id }
end
assert(found.cat.distance == 1 and found.cat.id == 10)
assert(found.cot.distance == 1 and found.cot.id == 20)
assert(found.cut.distance == 1 and found.cut.id == nil)

local cache <close> = levenshtein.query_cache(search, 32, 1024 * 1024)
local first <close> = cache:query("cet", 1)
assert(first:next() ~= nil)
local second <close> = cache:query("cet", 1)
assert(second:next() ~= nil)
assert(cache:stats().hits == 1)
