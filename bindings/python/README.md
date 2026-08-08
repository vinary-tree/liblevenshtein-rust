# Python binding

The Python 3.10+ adapter consumes a `vinary_tree_interop.DictionaryResource`
produced by libdictenstein or a host provider and pulls results in cursor-owned
batches. It does not construct or mutate dictionaries. `ctypes.CDLL` releases
the GIL during first-party native search calls.

Published wheels contain the appropriate liblevenshtein native library under
`liblevenshtein/native/` and depend on `vinary-tree-interop` for shared resource
types, so no separately installed system library is needed. Source-tree
development may set `LIBLEVENSHTEIN_LIBRARY` to an explicit build.

```python
from liblevenshtein import Transducer

# `dictionary` comes from vinary-tree-libdictenstein or a
# vinary_tree_interop.UnicodeDictionaryResource host provider.
with Transducer(dictionary) as automaton:
    with automaton.query("cat", 1) as matches:
        for match in matches:
            print(match.term, match.distance, match.id)
```

The safe iterator materializes at most one reusable batch. `QueryCursor.reduce`
is the expert path: its borrowed buffer views are valid only during the reducer
callback and avoid allocating one Python match object per result. Both paths
retain the exact dictionary revision visible when `query()` was called.
