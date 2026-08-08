# `@vinary-tree/liblevenshtein`

This package is the project-local JavaScript, TypeScript, and ClojureScript
facade for liblevenshtein. It does not contain a second copy of the runtime.
Every entry point re-exports the liblevenshtein namespace from the exact same
version of `@vinary-tree/vinary-tree`, so objects created by
`@vinary-tree/libdictenstein` can be passed directly to `transducer()`.

```ts
import { dynamicDawg } from "@vinary-tree/libdictenstein";
import { transducer } from "@vinary-tree/liblevenshtein";

using dictionary = dynamicDawg(["cat", "cot", "cut"]);
using automaton = transducer(dictionary, "standard");
using cursor = automaton.query("cat", 1);
for (const match of cursor) console.log(match);
```

Node selects the native N-API runtime by default. Use
`@vinary-tree/liblevenshtein/wasm` for the browser/static WebAssembly runtime or
`@vinary-tree/liblevenshtein/wasi` for the Node/WASI runtime. The WASI facade is
where preopened-directory persistent dictionaries are exposed as portable
support lands; browser persistence remains unsupported.

The ClojureScript facade mirrors the Clojure names: `transducer`, `query`,
`reduce-batches`, `phonetic-pattern`, `phonetic-rules`, `rewrite`, and `close!`.
