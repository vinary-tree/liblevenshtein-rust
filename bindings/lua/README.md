# liblevenshtein Lua binding

The Lua 5.4+ module consumes `vinary-tree.dictionary.v1` userdata created by
the separate libdictenstein rock. Cursors are callable generic-for iterators,
use leased native batches, and implement `__close` plus `__gc`. The module is
published as `vinary-tree-liblevenshtein` on LuaRocks.
