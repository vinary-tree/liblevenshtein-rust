# Clojure binding

This is the idiomatic Clojure facade over the Java 22 Foreign Function &
Memory binding. It is published to Clojars as:

```clojure
[io.vinarytree/liblevenshtein-clojure "0.10.0"]
```

Tools.deps users use the same coordinate:

```clojure
io.vinarytree/liblevenshtein-clojure {:mvn/version "0.10.0"}
```

The transitive JVM artifact embeds native libraries for Linux x86_64/aarch64,
macOS aarch64, and Windows x86_64, so no system installation is required.
Enable native access with `-J--enable-native-access=ALL-UNNAMED`.

```clojure
(require '[vinary-tree.liblevenshtein :as llev])

;; `dictionary` is a DictionaryResource returned by libdictenstein.
(with-open [automaton (llev/transducer dictionary {:algorithm :standard})]
  (with-open [matches (llev/query automaton "cut" 1
                                  {:order :distance-then-term})]
    (reduce (fn [count match] (inc count)) 0 matches)))
```

Result cursors are one-shot `Seqable`, `Iterable`, `IReduceInit`, and
`AutoCloseable` values. Reduction is incremental and closes on completion or
`reduced`; sequence traversal closes at EOF. Use `with-open` whenever a lazy
sequence might not be consumed completely. `reduce-batches` is the expert
callback-scoped, zero-copy path. Dictionary construction and CRUD intentionally
remain in the libdictenstein Clojure facade.
