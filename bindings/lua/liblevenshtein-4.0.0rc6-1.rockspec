package = "liblevenshtein"
version = "4.0.0rc6-1"
source = { url = "git+https://github.com/vinary-tree/liblevenshtein-rust.git", tag = "v4.0.0-rc.6" }
description = {
  summary = "Fast spelling correction and fuzzy search with Levenshtein automata",
  detailed = [[
Search libdictenstein dictionaries with standard Levenshtein, transposition,
merge-and-split, or unrestricted Damerau automata. Stream snapshot-consistent
matches, reuse bounded query caches, compute edit distances, and compile
phonetic patterns and rules. Requires matching native SDKs and Lua 5.4.

Lua quickstart and complete versioned API reference:
https://vinary-tree.github.io/liblevenshtein-rust/4.0.0-rc.6/lua/
]],
  homepage = "https://vinary-tree.github.io/liblevenshtein-rust/4.0.0-rc.6/lua/",
  license = "Apache-2.0"
}
dependencies = { "lua >= 5.4", "libdictenstein == 4.0.0rc6-1" }
external_dependencies = {
  LIBLEVENSHTEIN = { header = "liblevenshtein.h", library = "liblevenshtein" }
}
build = {
  type = "builtin",
  modules = {
    ["vinary_tree.liblevenshtein"] = {
      sources = { "bindings/lua/src/liblevenshtein_lua.c" },
      incdirs = { "$(LIBLEVENSHTEIN_INCDIR)", "bindings/lua/include" },
      libraries = { "liblevenshtein" },
      libdirs = { "$(LIBLEVENSHTEIN_LIBDIR)" }
    }
  }
}
