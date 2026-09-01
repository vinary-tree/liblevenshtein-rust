package = "liblevenshtein"
version = "4.0.0rc6-1"
source = { url = "git+https://github.com/vinary-tree/liblevenshtein-rust.git", tag = "v4.0.0-rc.6" }
description = { summary = "Fast spelling correction and fuzzy search with Levenshtein automata", license = "Apache-2.0" }
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
