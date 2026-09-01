(ns bench.harness
  "ClojureScript harness for the cross-language benchmark program.

  Implements harnesses/common/PROTOCOL.md over the project-owned CLJS
  facades (vinary-tree.liblevenshtein / vinary-tree.libdictenstein) riding
  the shared N-API runtime (@vinary-tree/javascript-runtime). Runs on Node
  (:target :nodejs, :optimizations :simple); the runner pins Node cpusets.

  Fairness notes (PROTOCOL.md §10): Node without V8 flag overrides; 64-bit
  checksum arithmetic uses BigInt masked to 64 bits after every update
  (§8); process.hrtime.bigint() is the pinned §9 monotonic source.

  Facade quirks recorded in the cell notes: the CLJS libdictenstein facade
  exposes no batch insert (put-all! is a put! loop), and native matches
  carry the term as a {domain, value} record, so the term string is
  match.term.value."
  (:require ["fs" :as fs]
            ["path" :as node-path]
            [vinary-tree.libdictenstein :as dict]
            [vinary-tree.liblevenshtein :as lev]))

(def ^:private wall-cap-seconds 300)

(def ^:private sample-definition
  (str "one full pass over the query set; every cursor fully drained and "
       "(term, distance) materialized"))

(def ^:private base-notes
  #js ["cljs facade over the shared N-API runtime"
       "dynamic_dawg construction uses the facade's put! loop (no batch API in the CLJS facade)"])

(defn- die [message]
  (.write js/process.stderr (str "bench-cross-clojurescript: " message "\n"))
  (.exit js/process 2))

;; ---------------------------------------------------------------------
;; Checksum primitives (PROTOCOL.md §8) — BigInt masked to 64 bits.
;; The js* forms below are the direct BigInt operator translations of
;; bench.mjs's fnvUpdate/entry helpers (CLJS numeric ops are not BigInt
;; aware; bit-and would truncate to 32 bits).
;; ---------------------------------------------------------------------

(def ^:private mask64 (js* "0xFFFFFFFFFFFFFFFFn"))
(def ^:private fnv-offset (js* "0xcbf29ce484222325n"))
(def ^:private fnv-prime (js* "0x100000001b3n"))
(def ^:private zero64 (js* "0n"))

(defn- fnv-update [hash byte]
  (js* "(((~{} ^ ~{}) * ~{}) & ~{})" hash byte fnv-prime mask64))

(defn- u64-add [left right]
  (js* "((~{} + ~{}) & ~{})" left right mask64))

(defn- entry-bytes [buffer distance]
  (let [length (.-length buffer)]
    (loop [i 0
           hash fnv-offset]
      (if (< i length)
        (recur (inc i) (fnv-update hash (js/BigInt (aget buffer i))))
        (loop [k 0
               hash (fnv-update hash zero64)
               remaining (js/BigInt distance)]
          (if (< k 8)
            (recur (inc k)
                   (fnv-update hash (js* "(~{} & 0xffn)" remaining))
                   (js* "(~{} >> 8n)" remaining))
            hash))))))

(defn- entry-ascii
  "entry(term, distance) over utf8(term) ‖ 0x00 ‖ LE64(distance); ASCII fast
  path with a real UTF-8 fallback the moment a non-ASCII char appears."
  [term distance]
  (let [length (.-length term)]
    (loop [i 0
           hash fnv-offset]
      (if (< i length)
        (let [code (.charCodeAt term i)]
          (if (> code 0x7f)
            (entry-bytes (js/Buffer.from term "utf8") distance)
            (recur (inc i) (fnv-update hash (js/BigInt code)))))
        (loop [k 0
               hash (fnv-update hash zero64)
               remaining (js/BigInt distance)]
          (if (< k 8)
            (recur (inc k)
                   (fnv-update hash (js* "(~{} & 0xffn)" remaining))
                   (js* "(~{} >> 8n)" remaining))
            hash))))))

(defn- checksum-hex [value]
  (.padStart (.toString value 16) 16 "0"))

(defn- self-test []
  (let [expect (fn [actual wanted label]
                 (when-not (identical? actual (js* "BigInt(~{})" wanted))
                   (die (str "checksum self-test failed for " label ": got "
                             (checksum-hex actual) ", want "
                             (checksum-hex (js* "BigInt(~{})" wanted))))))]
    (expect fnv-offset "0xcbf29ce484222325" "fnv1a64(\"\")")
    (expect (fnv-update fnv-offset (js/BigInt 0x61))
            "0xaf63dc4c8601ec8c" "fnv1a64(\"a\")")
    (expect (entry-ascii "cat" 1) "0x9697fa3e50464bc4" "entry(cat,1)")
    (expect (entry-ascii "cat" 0) "0xb592c1475b3595e5" "entry(cat,0)")
    (expect (entry-ascii "cot" 1) "0xb8acc5d3816bcdea" "entry(cot,1)")
    (expect (u64-add (entry-ascii "cat" 0) (entry-ascii "cot" 1))
            "0x6e3f871adca163cf" "checksum{2}")
    (expect zero64 "0x0" "checksum{}")
    (when-not (= (checksum-hex (js* "0x80011abde8767aban"))
                 "80011abde8767aba")
      (die "hex serialization of high-bit checksums is broken"))))

;; ---------------------------------------------------------------------
;; Monotonic clock (PROTOCOL.md §9: process.hrtime.bigint())
;; ---------------------------------------------------------------------

(defn- now-ns [] (js/process.hrtime.bigint))

(defn- ns-diff-number [start end]
  (js/Number (js* "(~{} - ~{})" end start)))

;; ---------------------------------------------------------------------
;; CLI contract (PROTOCOL.md §1)
;; ---------------------------------------------------------------------

(defn- parse-int [flag value]
  (let [parsed (js/Number value)]
    (if (js/Number.isSafeInteger parsed)
      parsed
      (die (str flag " expects an integer, got " value)))))

(defn- parse-number [flag value]
  (let [parsed (js/Number value)]
    (if (js/Number.isFinite parsed)
      parsed
      (die (str flag " expects a number, got " value)))))

(defn- parse-args [argv]
  (loop [args {:mode nil :algorithm nil :max-distance -1 :dictionary nil
               :queries nil :backend nil :out nil :samples 30
               :warmup-seconds 3 :gate-limit 200 :reps 10 :cells nil}
         remaining argv]
    (cond
      (empty? remaining)
      (if (or (nil? (:mode args)) (nil? (:dictionary args))
              (nil? (:backend args)))
        (die "--mode, --dictionary, --backend are required")
        args)

      (nil? (second remaining))
      (die (str "dangling argument: " (first remaining)))

      :else
      (let [[flag value & rest-args] remaining
            args (case flag
                   "--mode" (assoc args :mode value)
                   "--algorithm" (assoc args :algorithm value)
                   "--max-distance" (assoc args :max-distance
                                           (parse-int flag value))
                   "--dictionary" (assoc args :dictionary value)
                   "--queries" (assoc args :queries value)
                   "--backend" (assoc args :backend value)
                   "--out" (assoc args :out value)
                   "--samples" (assoc args :samples (parse-int flag value))
                   "--warmup-seconds" (assoc args :warmup-seconds
                                             (parse-number flag value))
                   "--gate-limit" (assoc args :gate-limit
                                         (parse-int flag value))
                   "--reps" (assoc args :reps (parse-int flag value))
                   "--cells" (assoc args :cells value)
                   (die (str "unknown flag: " flag)))]
        (recur args rest-args)))))

;; ---------------------------------------------------------------------
;; Input loading (PROTOCOL.md §3)
;; ---------------------------------------------------------------------

(defn- read-lines
  "Non-empty lines as a preallocated JS array."
  [path]
  (let [raw (try
              (fs/readFileSync path "utf8")
              (catch :default e
                (die (str "cannot read " path ": " (.-message e)))))
        lines #js []]
    (loop [start 0
           index 0]
      (when (<= index (.-length raw))
        (if (or (identical? index (.-length raw))
                (identical? (.charAt raw index) "\n"))
          (do (when (> index start)
                (.push lines (.slice raw start index)))
              (recur (inc index) (inc index)))
          (recur start (inc index)))))
    (when (zero? (.-length lines))
      (die (str path " contains no lines")))
    lines))

(defn- assert-strictly-sorted
  "ASCII workload: JS string < over ASCII equals strict byte order."
  [lines path]
  (let [total (.-length lines)]
    (loop [i 0]
      (when (< (inc i) total)
        (when-not (js* "(~{} < ~{})" (aget lines i) (aget lines (inc i)))
          (die (str path " is not strictly byte-sorted at line " (inc i)
                    ": \"" (aget lines i) "\" >= \"" (aget lines (inc i))
                    "\"")))
        (recur (inc i))))))

;; ---------------------------------------------------------------------
;; Dictionary, transducer, and the pass (PROTOCOL.md §4–5)
;; ---------------------------------------------------------------------

(def ^:private algorithms
  {"standard" :standard
   "transposition" :transposition
   "merge_and_split" :merge-and-split
   "damerau_levenshtein" :damerau-levenshtein})

(defn- build-dictionary
  "The CLJS facade's bulk path IS the put! loop (recorded in notes)."
  [terms backend]
  (if (= backend "dynamic_dawg")
    (let [dawg (dict/dynamic-dawg)
          total (.-length terms)]
      (loop [i 0
             inserted 0]
        (if (< i total)
          (recur (inc i)
                 (if (dict/put! dawg (aget terms i) nil)
                   (inc inserted)
                   inserted))
          (when-not (identical? inserted total)
            (die (str "insert count mismatch: " inserted " != " total)))))
      dawg)
    (die (str "unsupported backend for the ClojureScript target "
              "(dynamic_dawg only): " backend))))

(defn- full-pass
  "One full pass (§5): drain every cursor via the facade's reduce-batches
  (256-match crossings — the declared batch_size; it closes the cursor),
  summing the O(1) triple; FNV checksum only in untimed gate contexts.
  ASCII workload: JS string .length == UTF-8 byte length (gate-asserted)."
  [automaton queries max-distance with-checksum?]
  (let [totals #js [0 0 0 zero64]
        total-queries (.-length queries)]
    (loop [qi 0]
      (when (< qi total-queries)
        (let [cursor (lev/query automaton (aget queries qi) max-distance)]
          (lev/reduce-batches
           (fn [acc batch]
             (let [batch-length (.-length batch)]
               (loop [i 0]
                 (when (< i batch-length)
                   (let [matched (aget batch i)
                         term (.. matched -term -value)
                         distance (.-distance matched)]
                     (aset acc 0 (inc (aget acc 0)))
                     (aset acc 1 (+ (aget acc 1) (.-length term)))
                     (aset acc 2 (+ (aget acc 2) distance))
                     (when with-checksum?
                       (aset acc 3 (u64-add (aget acc 3)
                                            (entry-ascii term distance)))))
                   (recur (inc i)))))
             acc)
           totals cursor))
        (recur (inc qi))))
    totals))

(defn- triple-equals [a b]
  (and (identical? (aget a 0) (aget b 0))
       (identical? (aget a 1) (aget b 1))
       (identical? (aget a 2) (aget b 2))))

;; ---------------------------------------------------------------------
;; Result JSON (PROTOCOL.md §11 — runner post-fills run_id, sha256s,
;; cell_snapshot, environment_ref)
;; ---------------------------------------------------------------------

(defn- timestamp-utc []
  (.replace (.toISOString (js/Date.)) (js/RegExp. "\\.\\d{3}Z$") "Z"))

(defn- queryset-of [queries-path]
  (.replace (node-path/basename queries-path) (js/RegExp. "\\.txt$") ""))

(defn- write-result
  [{:keys [out args mode algorithm max-distance queries-path query-count
           term-count construct-ns warmup-passes samples-ns triple checksum
           construct-times status notes]}]
  (let [dictionary #js {"file" (:dictionary args)
                        "term_count" term-count
                        "structure" "dynamic_dawg"
                        "unit_domain" "unicode_scalar"}
        _ (when (some? construct-ns)
            (aset dictionary "construct_ns" construct-ns))
        result #js {"schema_version" "1.0.0"
                    "suite" "cross-language-v1"
                    "timestamp_utc" (timestamp-utc)
                    "target"
                    #js {"language" "clojurescript"
                         "implementation" "vinary-tree"
                         "backend" "cljs-napi"
                         "runtime_version"
                         (str "ClojureScript " *clojurescript-version*
                              " / node " (.-version js/process))
                         "library_version" "0.10.0"
                         "artifact"
                         #js {"kind" "local-build"
                              "id" "@vinary-tree/liblevenshtein@0.10.0 (cljs, N-API runtime)"}}
                    "dictionary" dictionary
                    "workload" #js {"queryset" (queryset-of queries-path)
                                    "file" queries-path
                                    "query_count" query-count}
                    "algorithm" algorithm
                    "max_distance" max-distance
                    "mode" (if (= mode "memory-child") "memory" mode)
                    "protocol"
                    #js {"timer" "monotonic"
                         "harness" "self-timed"
                         "warmup_seconds_min" (:warmup-seconds args)
                         "warmup_passes" warmup-passes
                         "samples_requested" (case mode
                                               "construct" (:reps args)
                                               "query" (:samples args)
                                               0)
                         "sample_definition" sample-definition
                         "batch_size" 256
                         "wall_cap_seconds" wall-cap-seconds}
                    "status" status
                    "notes" notes}]
    (if (some? construct-times)
      (aset result "construct"
            #js {"reps" (.-length construct-times)
                 "times_ns" construct-times
                 "term_count" term-count})
      (aset result "measurements"
            #js {"samples_ns" (or samples-ns #js [])
                 "sample_count" (if (some? samples-ns)
                                  (.-length samples-ns)
                                  0)
                 "matches_per_pass" (aget triple 0)
                 "term_bytes_per_pass" (aget triple 1)
                 "distance_sum_per_pass" (aget triple 2)
                 "checksum_hex" (checksum-hex checksum)}))
    (fs/mkdirSync (node-path/dirname out) #js {:recursive true})
    (fs/writeFileSync out (str (js/JSON.stringify result nil 2) "\n"))))

;; ---------------------------------------------------------------------
;; Modes (PROTOCOL.md §6) and the batch driver (§7)
;; ---------------------------------------------------------------------

(defn- run-construct [args terms]
  (when-not (:out args) (die "--out is required for construct mode"))
  (let [warmup (build-dictionary terms (:backend args))]
    (dict/close! warmup)
    (let [reps (max (:reps args) 1)
          times (js/Array. reps)]
      (loop [r 0]
        (when (< r reps)
          (let [started (now-ns)
                dawg (build-dictionary terms (:backend args))]
            (aset times r (ns-diff-number started (now-ns)))
            (dict/close! dawg))
          (recur (inc r))))
      (write-result
       {:out (:out args)
        :args args
        :mode "construct"
        :algorithm "standard"
        :max-distance 1
        :queries-path (or (:queries args) "workload/queries/hits.txt")
        :query-count 1
        :term-count (.-length terms)
        :construct-ns nil
        :warmup-passes 1
        :samples-ns nil
        :triple nil
        :checksum zero64
        :construct-times times
        :status "ok"
        :notes (.concat base-notes
                        #js ["construct mode: timed region is the build from the pre-sorted in-memory list only"])}))))

(defn- run-query-cell
  [args automaton queries algorithm max-distance queries-path out term-count
   construct-ns]
  (let [gate (full-pass automaton queries max-distance true)
        warm-start (now-ns)
        warmup-budget (* (:warmup-seconds args) 1e9)]
    (loop [passes 0
           last-pass-ns 0]
      (if (or (< (ns-diff-number warm-start (now-ns)) warmup-budget)
              (< passes 2))
        (let [t0 (now-ns)
              triple (full-pass automaton queries max-distance false)
              elapsed (ns-diff-number t0 (now-ns))]
          (when-not (triple-equals triple gate)
            (die "nondeterministic result during warmup"))
          (recur (inc passes) elapsed))
        (let [last-pass-seconds (/ last-pass-ns 1e9)
              requested (:samples args)
              capped? (> (* requested last-pass-seconds) wall-cap-seconds)
              sample-count (if capped?
                             (max 10 (js/Math.floor
                                      (/ wall-cap-seconds last-pass-seconds)))
                             requested)
              status (if capped? "degraded" "ok")
              notes (if capped?
                      (.concat base-notes
                               #js [(str "samples reduced from " requested
                                         " to " sample-count " by the "
                                         wall-cap-seconds
                                         "s wall cap (estimated pass "
                                         (.toFixed last-pass-seconds 3)
                                         "s)")])
                      base-notes)
              samples (js/Array. sample-count)]
          (loop [i 0]
            (when (< i sample-count)
              (let [t0 (now-ns)
                    triple (full-pass automaton queries max-distance false)]
                (aset samples i (ns-diff-number t0 (now-ns)))
                (when-not (triple-equals triple gate)
                  (die "nondeterministic result during measurement")))
              (recur (inc i))))
          (write-result {:out out
                         :args args
                         :mode "query"
                         :algorithm algorithm
                         :max-distance max-distance
                         :queries-path queries-path
                         :query-count (.-length queries)
                         :term-count term-count
                         :construct-ns construct-ns
                         :warmup-passes passes
                         :samples-ns samples
                         :triple gate
                         :checksum (aget gate 3)
                         :construct-times nil
                         :status status
                         :notes notes}))))))

(defn- run-one
  [args dawg previous-automaton algorithm max-distance queries-path out
   term-count construct-ns]
  (when (some? previous-automaton) (lev/close! previous-automaton))
  (let [algorithm-key (or (get algorithms algorithm)
                          (die (str "unknown algorithm: " algorithm)))
        automaton (lev/transducer dawg {:algorithm algorithm-key})
        query-lines (read-lines queries-path)]
    (case (:mode args)
      "verify"
      (let [limit (min (:gate-limit args) (.-length query-lines))
            subset (.slice query-lines 0 limit)
            gate (full-pass automaton subset max-distance true)]
        (write-result {:out out :args args :mode "verify"
                       :algorithm algorithm :max-distance max-distance
                       :queries-path queries-path :query-count limit
                       :term-count term-count :construct-ns construct-ns
                       :warmup-passes 0 :samples-ns #js []
                       :triple gate :checksum (aget gate 3)
                       :construct-times nil :status "ok" :notes base-notes}))

      "memory-child"
      (let [gate (full-pass automaton query-lines max-distance true)]
        (write-result {:out out :args args :mode "memory-child"
                       :algorithm algorithm :max-distance max-distance
                       :queries-path queries-path
                       :query-count (.-length query-lines)
                       :term-count term-count :construct-ns construct-ns
                       :warmup-passes 0 :samples-ns #js []
                       :triple gate :checksum (aget gate 3)
                       :construct-times nil :status "ok" :notes base-notes}))

      "query"
      (run-query-cell args automaton query-lines algorithm max-distance
                      queries-path out term-count construct-ns)

      (die (str "unknown mode: " (:mode args))))
    automaton))

(defn- run-cells [args dawg cells-path term-count construct-ns]
  (let [rows (->> (array-seq (read-lines cells-path))
                  (map #(.trim %))
                  (remove #(zero? (.-length %)))
                  (remove #(.startsWith % "#")))]
    (loop [remaining rows
           automaton nil]
      (if (empty? remaining)
        (when (some? automaton) (lev/close! automaton))
        (let [row (first remaining)
              fields (.split row "\t")]
          (when-not (identical? (.-length fields) 4)
            (die (str "cells row needs 4 fields: " row)))
          (recur (rest remaining)
                 (run-one args dawg automaton (aget fields 0)
                          (parse-int "--cells max_distance" (aget fields 1))
                          (aget fields 2) (aget fields 3) term-count
                          construct-ns)))))))

(defn -main [& argv]
  (self-test)
  (let [args (parse-args (vec argv))
        terms (read-lines (:dictionary args))
        term-count (.-length terms)]
    (assert-strictly-sorted terms (:dictionary args))
    (case (:mode args)
      "construct"
      (run-construct args terms)

      ("query" "verify" "memory-child")
      (let [build-start (now-ns)
            dawg (build-dictionary terms (:backend args))
            construct-ns (ns-diff-number build-start (now-ns))]
        (if-let [cells-path (:cells args)]
          (run-cells args dawg cells-path term-count construct-ns)
          (do
            (when-not (:algorithm args) (die "--algorithm is required"))
            (when-not (:queries args) (die "--queries is required"))
            (when-not (:out args) (die "--out is required"))
            (when (neg? (:max-distance args))
              (die "--max-distance is required"))
            (let [automaton (run-one args dawg nil (:algorithm args)
                                     (:max-distance args) (:queries args)
                                     (:out args) term-count construct-ns)]
              (lev/close! automaton))))
        (dict/close! dawg))

      (die (str "unknown mode: " (:mode args))))))

(set! *main-cli-fn* -main)
