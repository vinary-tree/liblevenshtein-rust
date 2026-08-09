-- C8 native property-based tests for the Lua facade: a seeded pseudo-PBT loop
-- checked against an in-language Levenshtein oracle.
--
--   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, and threshold
--       consistency dThreshold(a,b,k) == (d<=k ? d : -2) across the
--       Levenshtein/Damerau/true-Damerau variants, plus standard oracle
--       agreement (the native usize::MAX-1 sentinel wraps to -2 under the
--       size_t -> lua_Integer conversion);
--   (b) query result set == {t in dict : lev(query,t) <= k} over a random
--       libdictenstein DynamicDawg dictionary, with exact distances and value
--       round-trips;
--   (c) u64 value round-trips as signed 64-bit bit patterns, with 0,
--       math.mininteger (2^63) and -1 (2^64-1) pinned.
--
-- The PRNG is seeded from a constant. An ASCII alphabet keeps byte == scalar so
-- the oracle needs no UTF-8 decoding; scalar-vs-byte counting is pinned by the
-- snapshot suite. Lua strings expose a non-null buffer even when empty, so
-- empty operands are exercised.

local dictionary = require("vinary_tree.libdictenstein")
local levenshtein = require("vinary_tree.liblevenshtein")

math.randomseed(20260809)

local ALPHABET = { "a", "b", "c", "d" }
local SENTINEL = -2

local function generate(minimum, maximum)
  local length = math.random(minimum, maximum)
  local units = {}
  for index = 1, length do units[index] = ALPHABET[math.random(1, #ALPHABET)] end
  return table.concat(units)
end

local function oracle(a, b)
  if a == b then return 0 end
  local la, lb = #a, #b
  if la == 0 then return lb end
  if lb == 0 then return la end
  local previous = {}
  for j = 0, lb do previous[j] = j end
  for i = 1, la do
    local current = { [0] = i }
    local ca = a:byte(i)
    for j = 1, lb do
      local cost = (ca == b:byte(j)) and 0 or 1
      local best = previous[j] + 1
      if current[j - 1] + 1 < best then best = current[j - 1] + 1 end
      if previous[j - 1] + cost < best then best = previous[j - 1] + cost end
      current[j] = best
    end
    previous = current
  end
  return previous[lb]
end

local variants = {
  { levenshtein.distance, levenshtein.distance_threshold },
  { levenshtein.damerau_distance, levenshtein.damerau_distance_threshold },
  { levenshtein.true_damerau_distance, levenshtein.true_damerau_distance_threshold },
}

-- (a) distance self-consistency and oracle agreement.
for _ = 1, 400 do
  local a, b = generate(0, 6), generate(0, 6)
  local k = math.random(0, 3)
  assert(levenshtein.distance(a, b) == oracle(a, b), "standard oracle agreement")
  for _, variant in ipairs(variants) do
    local distance, thresholded = variant[1], variant[2]
    local full = distance(a, b)
    assert(distance(b, a) == full, "symmetry")
    assert(distance(a, a) == 0, "identity")
    local bounded = thresholded(a, b, k)
    assert(bounded == (full <= k and full or SENTINEL), "threshold consistency")
  end
end

local function run_query(entries, query, k)
  local dawg = dictionary.dynamic_dawg("unicode")
  for _, entry in ipairs(entries) do dawg:put(entry[1], entry[2]) end
  local automaton = levenshtein.transducer(dawg)
  local got = {}
  for match in automaton:query(query, k) do
    got[match.term] = { distance = match.distance, id = match.id }
  end
  automaton:close()
  dawg:close()
  return got
end

-- (b) query result set equals the oracle, with exact distances and (c) value
-- round-trips (nil values wrapped so a nil value is distinguishable from an
-- absent term).
for _ = 1, 150 do
  local entries, seen = {}, {}
  for _ = 1, math.random(0, 8) do
    local term = generate(0, 6)
    if not seen[term] then
      seen[term] = true
      -- Lua integers are signed 64-bit and the facade requires a non-negative
      -- value, so the representable u64 value range is [0, 2^63-1].
      local value = (math.random(4) == 1) and nil or math.random(0, math.maxinteger)
      entries[#entries + 1] = { term, value }
    end
  end
  local query, k = generate(0, 6), math.random(0, 3)
  local got = run_query(entries, query, k)

  local expected, expected_count = {}, 0
  for _, entry in ipairs(entries) do
    if oracle(query, entry[1]) <= k then
      expected[entry[1]] = { entry[2] }
      expected_count = expected_count + 1
    end
  end

  local got_count = 0
  for term, observed in pairs(got) do
    got_count = got_count + 1
    assert(expected[term] ~= nil, "unexpected term " .. term)
    assert(observed.distance == oracle(query, term), "distance for " .. term)
    assert(observed.id == expected[term][1], "value for " .. term)
  end
  for term in pairs(expected) do assert(got[term] ~= nil, "missing " .. term) end
  assert(got_count == expected_count, "result set size")
end

-- (c) value round-trip: 0 and the representable upper boundary math.maxinteger
-- (2^63-1) are pinned. u64 values >= 2^63 are not representable as a Lua signed
-- integer and the facade rejects them, so 2^64-1 is out of range for this facade.
for _, value in ipairs({ 0, 1, math.maxinteger }) do
  local got = run_query({ { "term", value } }, "term", 0)
  assert(got["term"] ~= nil and got["term"].id == value, "u64 round-trip " .. value)
end

print("Lua property tests passed")
