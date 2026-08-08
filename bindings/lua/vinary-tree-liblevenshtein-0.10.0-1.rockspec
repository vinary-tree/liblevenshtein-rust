package = "vinary-tree-liblevenshtein"
version = "0.10.0-1"
source = { url = "git+https://github.com/vinary-tree/liblevenshtein-rust.git", tag = "v0.10.0" }
description = { summary = "Streaming Lua bindings for liblevenshtein", license = "Apache-2.0" }
dependencies = { "lua >= 5.4", "vinary-tree-libdictenstein >= 0.2.1" }
build = {
  type = "builtin",
  modules = {
    ["vinary_tree.liblevenshtein"] = {
      sources = { "bindings/lua/src/liblevenshtein_lua.c" },
      incdirs = { "include", "vinary-tree-interop/include", "vinary-tree-interop/bindings/lua" },
      libraries = { "liblevenshtein" },
      libdirs = { "target/release" }
    }
  }
}
