(ns bench-liblevenshtein.core
  "Clojure harness for the cross-language benchmark program.

  Implements harnesses/common/PROTOCOL.md over the idiomatic Clojure facades
  (vinary-tree.liblevenshtein + vinary-tree.libdictenstein) riding the JVM
  FFM bindings. Fairness (PROTOCOL.md §10): JDK 26, -Xms2g -Xmx2g, default
  G1 (runner-provided), direct linking on (recorded in notes); the timed
  pass consumes the facade's reducible cursor — the persistent-map match
  path a migrating Clojure user writes."
  (:require [vinary-tree.libdictenstein :as dict]
            [vinary-tree.liblevenshtein :as lev])
  (:import (java.nio.charset StandardCharsets)
           (java.nio.file Files Path)
           (java.time Instant ZoneOffset)
           (java.time.format DateTimeFormatter)
           (java.util Arrays Locale))
  (:gen-class))

(set! *warn-on-reflection* true)

(def ^:private wall-cap-seconds 300.0)

(def ^:private sample-definition
  (str "one full pass over the query set; every cursor fully drained and "
       "(term, distance) materialized"))

(def ^:private base-notes
  ["idiomatic reducible facade over the JVM FFM bindings; direct linking on (-Dclojure.compiler.direct-linking=true)"])

(defn- die [^String message]
  (binding [*out* *err*]
    (println (str "bench-cross-clojure: " message)))
  (System/exit 2))

;; ---------------------------------------------------------------------
;; Checksum primitives (PROTOCOL.md §8) — signed long wrapping ops give
;; bit-identical results; hex serialization uses the unsigned form.
;; ---------------------------------------------------------------------

(def ^:private ^:const fnv-prime 0x100000001b3)
(def ^:private fnv-offset (Long/parseUnsignedLong "cbf29ce484222325" 16))

(defn- entry-bytes ^long [^bytes term-utf8 ^long distance]
  (let [n (alength term-utf8)]
    (loop [i 0
           h (long fnv-offset)]
      (if (< i n)
        (recur (unchecked-inc i)
               (unchecked-multiply
                (bit-xor h (bit-and (long (aget term-utf8 i)) 0xff))
                fnv-prime))
        (loop [k 0
               h (unchecked-multiply h fnv-prime) ; XOR with 0x00 separator
               d distance]
          (if (< k 8)
            (recur (unchecked-inc k)
                   (unchecked-multiply (bit-xor h (bit-and d 0xff)) fnv-prime)
                   (unsigned-bit-shift-right d 8))
            h))))))

(defn- entry-ascii
  "entry(term, distance) over utf8(term) ‖ 0x00 ‖ LE64(distance); ASCII fast
  path with a real UTF-8 fallback the moment a non-ASCII char appears."
  ^long [^String term ^long distance]
  (let [n (.length term)]
    (loop [i 0
           h (long fnv-offset)]
      (if (< i n)
        (let [c (long (int (.charAt term i)))]
          (if (> c 0x7f)
            (entry-bytes (.getBytes term StandardCharsets/UTF_8) distance)
            (recur (unchecked-inc i)
                   (unchecked-multiply (bit-xor h c) fnv-prime))))
        (loop [k 0
               h (unchecked-multiply h fnv-prime)
               d distance]
          (if (< k 8)
            (recur (unchecked-inc k)
                   (unchecked-multiply (bit-xor h (bit-and d 0xff)) fnv-prime)
                   (unsigned-bit-shift-right d 8))
            h))))))

(defn- checksum-hex ^String [^long value]
  (String/format Locale/ROOT "%016x" (object-array [value])))

(defn- self-test []
  (let [expect (fn [^long actual ^String wanted ^String label]
                 (when-not (= actual (Long/parseUnsignedLong wanted 16))
                   (die (str "checksum self-test failed for " label ": got "
                             (checksum-hex actual) ", want " wanted))))]
    ;; checksum{} == 0: the wrapping-sum accumulator of an empty multiset.
    (expect 0 "0000000000000000" "checksum{}")
    ;; fnv1a64 over raw bytes: the two published FNV vectors through the
    ;; same update loop entry-bytes uses for the term prefix.
    (let [fnv (fn ^long [^bytes data]
                (let [n (alength data)]
                  (loop [i 0 h (long fnv-offset)]
                    (if (< i n)
                      (recur (unchecked-inc i)
                             (unchecked-multiply
                              (bit-xor h (bit-and (long (aget data i)) 0xff))
                              fnv-prime))
                      h))))]
      (when-not (= (fnv (byte-array 0)) fnv-offset)
        (die "checksum self-test failed for fnv1a64(\"\")"))
      (when-not (= (fnv (.getBytes "a" StandardCharsets/UTF_8))
                   (Long/parseUnsignedLong "af63dc4c8601ec8c" 16))
        (die "checksum self-test failed for fnv1a64(\"a\")")))
    (expect (entry-ascii "cat" 1) "9697fa3e50464bc4" "entry(cat,1)")
    (expect (entry-ascii "cat" 0) "b592c1475b3595e5" "entry(cat,0)")
    (expect (entry-ascii "cot" 1) "b8acc5d3816bcdea" "entry(cot,1)")
    (expect (unchecked-add (entry-ascii "cat" 0) (entry-ascii "cot" 1))
            "6e3f871adca163cf" "checksum{2}")
    (when-not (= "ffffffffffffffff" (checksum-hex -1))
      (die "hex serialization of high-bit checksums is not unsigned"))))

;; ---------------------------------------------------------------------
;; CLI contract (PROTOCOL.md §1)
;; ---------------------------------------------------------------------

(def ^:private default-args
  {:mode nil :algorithm nil :max-distance -1 :dictionary nil :queries nil
   :backend nil :out nil :samples 30 :warmup-seconds 3.0 :gate-limit 200
   :reps 10 :cells nil})

(defn- parse-long-arg ^long [^String flag ^String value]
  (try
    (Long/parseLong value)
    (catch NumberFormatException _
      (die (str flag " expects an integer, got " value))
      0)))

(defn- parse-double-arg ^double [^String flag ^String value]
  (try
    (Double/parseDouble value)
    (catch NumberFormatException _
      (die (str flag " expects a number, got " value))
      0.0)))

(defn- parse-args [argv]
  (loop [args default-args
         remaining argv]
    (cond
      (empty? remaining)
      (if (or (nil? (:mode args)) (nil? (:dictionary args)) (nil? (:backend args)))
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
                                           (parse-long-arg flag value))
                   "--dictionary" (assoc args :dictionary value)
                   "--queries" (assoc args :queries value)
                   "--backend" (assoc args :backend value)
                   "--out" (assoc args :out value)
                   "--samples" (assoc args :samples (parse-long-arg flag value))
                   "--warmup-seconds" (assoc args :warmup-seconds
                                             (parse-double-arg flag value))
                   "--gate-limit" (assoc args :gate-limit
                                         (parse-long-arg flag value))
                   "--reps" (assoc args :reps (parse-long-arg flag value))
                   "--cells" (assoc args :cells value)
                   (die (str "unknown flag: " flag)))]
        (recur args rest-args)))))

;; ---------------------------------------------------------------------
;; Input loading (PROTOCOL.md §3)
;; ---------------------------------------------------------------------

(defn- read-lines
  "Non-empty lines of a file as a vector (preallocated via transient)."
  [^String path]
  (let [raw (try
              (String. (Files/readAllBytes (Path/of path (make-array String 0)))
                       StandardCharsets/UTF_8)
              (catch java.io.IOException e
                (die (str "cannot read " path ": " (.getMessage e)))
                ""))
        length (.length ^String raw)]
    (loop [start 0
           index 0
           lines (transient [])]
      (if (> index length)
        (let [result (persistent! lines)]
          (if (zero? (count result))
            (die (str path " contains no lines"))
            result))
        (if (or (= index length) (= (.charAt ^String raw index) \newline))
          (recur (inc index) (inc index)
                 (if (> index start)
                   (conj! lines (.substring ^String raw start index))
                   lines))
          (recur start (inc index) lines))))))

(defn- assert-strictly-sorted
  "Strict byte-ascending order over UTF-8 bytes (Arrays/compareUnsigned)."
  [lines ^String path]
  (loop [i 0
         ^bytes previous (.getBytes ^String (nth lines 0) StandardCharsets/UTF_8)]
    (when (< (inc i) (count lines))
      (let [^bytes current (.getBytes ^String (nth lines (inc i))
                                      StandardCharsets/UTF_8)]
        (when (>= (Arrays/compareUnsigned previous current) 0)
          (die (str path " is not strictly byte-sorted at line " (inc i) ": \""
                    (nth lines i) "\" >= \"" (nth lines (inc i)) "\"")))
        (recur (inc i) current)))))

;; ---------------------------------------------------------------------
;; Dictionary, transducer, and the pass (PROTOCOL.md §4–5)
;; ---------------------------------------------------------------------

(def ^:private algorithms
  {"standard" :standard
   "transposition" :transposition
   "merge_and_split" :merge-and-split
   "damerau_levenshtein" :damerau-levenshtein})

(defn- build-dictionary
  "One facade batch call (put-all! -> DynamicDawg.putAllStrings)."
  [prepared-entries ^String backend ^long term-count]
  (if (= backend "dynamic_dawg")
    (let [dawg (dict/dynamic-dawg)
          inserted (long (dict/put-all! dawg prepared-entries))]
      (when-not (= inserted term-count)
        (die (str "batch insert count mismatch: " inserted " != " term-count)))
      dawg)
    (die (str "unsupported backend for the Clojure target (dynamic_dawg only): "
              backend))))

(defn- full-pass
  "One full pass (§5): reduce the facade's one-shot reducible cursor per
  query (IReduceInit closes the cursor at exhaustion), summing the O(1)
  triple into a preallocated long[4]; FNV checksum only in gate contexts.
  ASCII workload: String.length == UTF-8 byte length (gate-asserted)."
  ^longs [automaton ^objects queries ^long max-distance with-checksum?]
  (let [totals (long-array 4)
        n (alength queries)]
    (loop [qi 0]
      (when (< qi n)
        (let [cursor (lev/query automaton (aget queries qi) max-distance)]
          (reduce
           (fn [_ match]
             (let [^String term (:term match)
                   distance (long (:distance match))]
               (aset totals 0 (unchecked-inc (aget totals 0)))
               (aset totals 1 (unchecked-add (aget totals 1)
                                             (long (.length term))))
               (aset totals 2 (unchecked-add (aget totals 2) distance))
               (when with-checksum?
                 (aset totals 3 (unchecked-add (aget totals 3)
                                               (entry-ascii term distance))))
               nil))
           nil cursor))
        (recur (unchecked-inc qi))))
    totals))

(defn- triple-equals [^longs a ^longs b]
  (and (= (aget a 0) (aget b 0))
       (= (aget a 1) (aget b 1))
       (= (aget a 2) (aget b 2))))

;; ---------------------------------------------------------------------
;; Result JSON (PROTOCOL.md §11 — runner post-fills run_id, sha256s,
;; cell_snapshot, environment_ref)
;; ---------------------------------------------------------------------

(defn- escape-json ^String [^String value]
  (let [builder (StringBuilder. (+ (.length value) 8))]
    (dotimes [i (.length value)]
      (let [c (.charAt value i)]
        (case c
          \" (.append builder "\\\"")
          \\ (.append builder "\\\\")
          \newline (.append builder "\\n")
          \return (.append builder "\\r")
          \tab (.append builder "\\t")
          (if (< (int c) 0x20)
            (.append builder (String/format Locale/ROOT "\\u%04x"
                                            (object-array [(int c)])))
            (.append builder c)))))
    (.toString builder)))

(defn- timestamp-utc ^String []
  (.format (.withZone (DateTimeFormatter/ofPattern "yyyy-MM-dd'T'HH:mm:ss'Z'"
                                                   Locale/ROOT)
                      ZoneOffset/UTC)
           (Instant/now)))

(defn- queryset-of ^String [^String queries-path]
  (let [base (.getFileName (Path/of queries-path (make-array String 0)))]
    (.replaceFirst (str base) "\\.txt$" "")))

(defn- join-longs ^String [values]
  (let [builder (StringBuilder.)]
    (doseq [[i v] (map-indexed vector values)]
      (when (pos? (long i)) (.append builder ", "))
      (.append builder (str v)))
    (.toString builder)))

(defn- write-result
  [{:keys [^String out args mode algorithm max-distance queries-path
           query-count term-count construct-ns warmup-passes samples-ns
           triple checksum construct-times status notes]}]
  (let [^longs triple (or triple (long-array 3))
        builder (StringBuilder. 4096)
        add (fn [^String s] (.append builder s))]
    (add "{\n")
    (add "  \"schema_version\": \"1.0.0\",\n")
    (add "  \"suite\": \"cross-language-v1\",\n")
    (add (str "  \"timestamp_utc\": \"" (timestamp-utc) "\",\n"))
    (add "  \"target\": {\n")
    (add "    \"language\": \"clojure\",\n")
    (add "    \"implementation\": \"vinary-tree\",\n")
    (add "    \"backend\": \"jvm-ffm-clojure\",\n")
    (add (str "    \"runtime_version\": \""
              (escape-json (str "Clojure " (clojure-version) " / "
                                (System/getProperty "java.vm.name") " "
                                (System/getProperty "java.version")))
              "\",\n"))
    (add "    \"library_version\": \"0.10.0\",\n")
    (add (str "    \"artifact\": { \"kind\": \"local-build\", \"id\": "
              "\"io.vinarytree:liblevenshtein-clojure:0.10.0\" }\n"))
    (add "  },\n")
    (add "  \"dictionary\": {\n")
    (add (str "    \"file\": \"" (escape-json (:dictionary args)) "\",\n"))
    (add (str "    \"term_count\": " term-count ",\n"))
    (add "    \"structure\": \"dynamic_dawg\",\n")
    (add "    \"unit_domain\": \"unicode_scalar\"")
    (if construct-ns
      (add (str ",\n    \"construct_ns\": " construct-ns "\n"))
      (add "\n"))
    (add "  },\n")
    (add "  \"workload\": {\n")
    (add (str "    \"queryset\": \"" (escape-json (queryset-of queries-path))
              "\",\n"))
    (add (str "    \"file\": \"" (escape-json queries-path) "\",\n"))
    (add (str "    \"query_count\": " query-count "\n"))
    (add "  },\n")
    (add (str "  \"algorithm\": \"" algorithm "\",\n"))
    (add (str "  \"max_distance\": " max-distance ",\n"))
    (add (str "  \"mode\": \"" (if (= mode "memory-child") "memory" mode)
              "\",\n"))
    (add "  \"protocol\": {\n")
    (add "    \"timer\": \"monotonic\",\n")
    (add "    \"harness\": \"self-timed\",\n")
    (add (str "    \"warmup_seconds_min\": " (:warmup-seconds args) ",\n"))
    (add (str "    \"warmup_passes\": " warmup-passes ",\n"))
    (add (str "    \"samples_requested\": "
              (case mode
                "construct" (:reps args)
                "query" (:samples args)
                0)
              ",\n"))
    (add (str "    \"sample_definition\": \"" (escape-json sample-definition)
              "\",\n"))
    (add "    \"batch_size\": 256,\n")
    (add (str "    \"wall_cap_seconds\": " (long wall-cap-seconds) "\n"))
    (add "  },\n")
    (if construct-times
      (do
        (add "  \"construct\": {\n")
        (add (str "    \"reps\": " (count construct-times) ",\n"))
        (add (str "    \"times_ns\": [" (join-longs construct-times) "],\n"))
        (add (str "    \"term_count\": " term-count "\n"))
        (add "  },\n"))
      (do
        (add "  \"measurements\": {\n")
        (add (str "    \"samples_ns\": [" (join-longs (or samples-ns [])) "],\n"))
        (add (str "    \"sample_count\": " (count (or samples-ns [])) ",\n"))
        (add (str "    \"matches_per_pass\": " (aget triple 0) ",\n"))
        (add (str "    \"term_bytes_per_pass\": " (aget triple 1) ",\n"))
        (add (str "    \"distance_sum_per_pass\": " (aget triple 2) ",\n"))
        (add (str "    \"checksum_hex\": \"" (checksum-hex (long (or checksum 0)))
                  "\"\n"))
        (add "  },\n")))
    (add (str "  \"status\": \"" status "\",\n"))
    (add "  \"notes\": [")
    (doseq [[i note] (map-indexed vector notes)]
      (when (pos? (long i)) (add ", "))
      (add (str "\"" (escape-json note) "\"")))
    (add "]\n}\n")
    (let [out-path (Path/of out (make-array String 0))]
      (when-let [parent (.getParent out-path)]
        (Files/createDirectories parent (make-array java.nio.file.attribute.FileAttribute 0)))
      (spit out (.toString builder) :encoding "UTF-8"))))

;; ---------------------------------------------------------------------
;; Modes (PROTOCOL.md §6) and the batch driver (§7)
;; ---------------------------------------------------------------------

(defn- run-construct [args prepared-entries ^long term-count]
  (when-not (:out args) (die "--out is required for construct mode"))
  (let [backend (:backend args)
        reps (long (:reps args))
        warmup (build-dictionary prepared-entries backend term-count)]
    (dict/close! warmup)
    (let [times (long-array (max reps 1))]
      (dotimes [r reps]
        (let [started (System/nanoTime)
              dawg (build-dictionary prepared-entries backend term-count)]
          (aset times r (unchecked-subtract (System/nanoTime) started))
          (dict/close! dawg)))
      (write-result {:out (:out args)
                     :args args
                     :mode "construct"
                     :algorithm "standard"
                     :max-distance 1
                     :queries-path (or (:queries args)
                                       "workload/queries/hits.txt")
                     :query-count 1
                     :term-count term-count
                     :construct-ns nil
                     :warmup-passes 1
                     :samples-ns nil
                     :triple nil
                     :checksum 0
                     :construct-times (vec times)
                     :status "ok"
                     :notes (conj base-notes
                                  (str "construct mode: timed region is the "
                                       "build from the pre-sorted in-memory "
                                       "list only"))}))))

(defn- run-query-cell
  [args automaton ^objects queries ^String algorithm max-distance
   ^String queries-path ^String out term-count construct-ns]
  (let [max-distance (long max-distance)
        gate (full-pass automaton queries max-distance true)
        warm-start (System/nanoTime)
        warmup-budget (long (* (double (:warmup-seconds args)) 1e9))]
    (loop [passes 0
           last-pass-ns 0]
      (if (or (< (unchecked-subtract (System/nanoTime) warm-start)
                 warmup-budget)
              (< passes 2))
        (let [t0 (System/nanoTime)
              triple (full-pass automaton queries max-distance false)
              elapsed (unchecked-subtract (System/nanoTime) t0)]
          (when-not (triple-equals triple gate)
            (die "nondeterministic result during warmup"))
          (recur (inc passes) elapsed))
        (let [last-pass-seconds (/ (double last-pass-ns) 1e9)
              requested (long (:samples args))
              capped? (> (* (double requested) last-pass-seconds)
                         wall-cap-seconds)
              sample-count (if capped?
                             (max 10 (long (/ wall-cap-seconds
                                              last-pass-seconds)))
                             requested)
              status (if capped? "degraded" "ok")
              notes (if capped?
                      (conj base-notes
                            (String/format
                             Locale/ROOT
                             "samples reduced from %d to %d by the %.0fs wall cap (estimated pass %.3fs)"
                             (object-array [requested sample-count
                                            wall-cap-seconds
                                            last-pass-seconds])))
                      base-notes)
              samples (long-array sample-count)]
          (dotimes [i sample-count]
            (let [t0 (System/nanoTime)
                  triple (full-pass automaton queries max-distance false)]
              (aset samples i (unchecked-subtract (System/nanoTime) t0))
              (when-not (triple-equals triple gate)
                (die "nondeterministic result during measurement"))))
          (write-result {:out out
                         :args args
                         :mode "query"
                         :algorithm algorithm
                         :max-distance max-distance
                         :queries-path queries-path
                         :query-count (alength queries)
                         :term-count term-count
                         :construct-ns construct-ns
                         :warmup-passes passes
                         :samples-ns (vec samples)
                         :triple gate
                         :checksum (aget gate 3)
                         :construct-times nil
                         :status status
                         :notes notes}))))))

(defn- run-one
  [args dawg previous-automaton ^String algorithm max-distance
   ^String queries-path ^String out term-count construct-ns]
  (when previous-automaton (lev/close! previous-automaton))
  (let [max-distance (long max-distance)
        algorithm-key (or (get algorithms algorithm)
                          (die (str "unknown algorithm: " algorithm)))
        automaton (lev/transducer dawg {:algorithm algorithm-key})
        query-lines (read-lines queries-path)]
    (case (:mode args)
      "verify"
      (let [limit (min (long (:gate-limit args)) (count query-lines))
            subset (object-array (subvec query-lines 0 limit))
            gate (full-pass automaton subset max-distance true)]
        (write-result {:out out :args args :mode "verify"
                       :algorithm algorithm :max-distance max-distance
                       :queries-path queries-path :query-count limit
                       :term-count term-count :construct-ns construct-ns
                       :warmup-passes 0 :samples-ns []
                       :triple gate :checksum (aget gate 3)
                       :construct-times nil :status "ok" :notes base-notes}))

      "memory-child"
      (let [queries (object-array query-lines)
            gate (full-pass automaton queries max-distance true)]
        (write-result {:out out :args args :mode "memory-child"
                       :algorithm algorithm :max-distance max-distance
                       :queries-path queries-path
                       :query-count (count query-lines)
                       :term-count term-count :construct-ns construct-ns
                       :warmup-passes 0 :samples-ns []
                       :triple gate :checksum (aget gate 3)
                       :construct-times nil :status "ok" :notes base-notes}))

      "query"
      (run-query-cell args automaton (object-array query-lines) algorithm
                      max-distance queries-path out term-count construct-ns)

      (die (str "unknown mode: " (:mode args))))
    automaton))

(defn- run-cells [args dawg ^String cells-path term-count construct-ns]
  (let [rows (into []
                   (comp (map #(.trim ^String %))
                         (remove #(.isEmpty ^String %))
                         (remove #(.startsWith ^String % "#")))
                   (read-lines cells-path))]
    (loop [remaining rows
           automaton nil]
      (if (empty? remaining)
        (when automaton (lev/close! automaton))
        (let [^String row (first remaining)
              fields (.split row "\t")]
          (when-not (= 4 (alength fields))
            (die (str "cells row needs 4 fields: " row)))
          (recur (rest remaining)
                 (run-one args dawg automaton (aget fields 0)
                          (parse-long-arg "--cells max_distance"
                                          (aget fields 1))
                          (aget fields 2) (aget fields 3) term-count
                          construct-ns)))))))

(defn -main [& argv]
  (self-test)
  (let [args (parse-args (vec argv))
        terms (read-lines (:dictionary args))
        term-count (count terms)]
    (assert-strictly-sorted terms (:dictionary args))
    ;; Entry preparation happens once and is NOT part of any timed build:
    ;; the timed operation is the facade's one batch call (put-all!).
    (let [prepared-entries (mapv (fn [term] [term nil]) terms)]
      (case (:mode args)
        "construct"
        (run-construct args prepared-entries term-count)

        ("query" "verify" "memory-child")
        (let [build-start (System/nanoTime)
              dawg (build-dictionary prepared-entries (:backend args)
                                     term-count)
              construct-ns (unchecked-subtract (System/nanoTime) build-start)]
          (if-let [cells-path (:cells args)]
            (run-cells args dawg cells-path term-count construct-ns)
            (do
              (when-not (:algorithm args) (die "--algorithm is required"))
              (when-not (:queries args) (die "--queries is required"))
              (when-not (:out args) (die "--out is required"))
              (when (neg? (long (:max-distance args)))
                (die "--max-distance is required"))
              (let [automaton (run-one args dawg nil (:algorithm args)
                                       (long (:max-distance args))
                                       (:queries args) (:out args)
                                       term-count construct-ns)]
                (lev/close! automaton))))
          (dict/close! dawg))

        (die (str "unknown mode: " (:mode args))))))
  (System/exit 0))
