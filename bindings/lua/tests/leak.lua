-- C9 leak-discipline suite for the Lua facade, measured with collectgarbage.
-- A >=10,000-cycle create/use/free loop over dictionaries, transducers, query
-- cursors, phonetic patterns, and rule sets must reach a steady state: after a
-- full collection the Lua heap (collectgarbage("count"), in KB) must not drift
-- upward. Native handles are freed at close(); collectgarbage confirms the Lua
-- wrappers are reclaimed rather than accumulating.

local dictionary = require("vinary_tree.libdictenstein")
local levenshtein = require("vinary_tree.liblevenshtein")

local CYCLES = 10000
local WARMUP = 2000
local MAX_GROWTH_KB = 1024 -- 1 MiB; a real per-cycle leak would blow past this

local function settled_kb()
  collectgarbage("collect")
  collectgarbage("collect")
  return collectgarbage("count")
end

local function assert_steady(label, cycle)
  for _ = 1, WARMUP do cycle() end
  local base = settled_kb()
  for _ = 1, CYCLES do cycle() end
  local growth = settled_kb() - base
  assert(growth < MAX_GROWTH_KB, label .. ": Lua heap grew " .. growth .. " KB over " .. CYCLES .. " cycles")
end

assert_steady("transducer", function()
  local dawg = dictionary.dynamic_dawg("unicode")
  dawg:put("cat", 1)
  dawg:put("cot", 2)
  dawg:put("cut", 3)
  dawg:put("scat", 4)
  local automaton = levenshtein.transducer(dawg)
  local cursor = automaton:query("cat", 2)
  for _ in cursor do end
  cursor:close()
  automaton:close()
  dawg:close()
end)

assert_steady("phonetic", function()
  local pattern = levenshtein.phonetic_pattern("c[ao]t")
  pattern:matches("cat")
  pattern:close()
  local rules = levenshtein.phonetic_rules("english-orthography")
  rules:apply("phone")
  rules:close()
end)

print("Lua leak loop completed " .. CYCLES .. " cycles")
