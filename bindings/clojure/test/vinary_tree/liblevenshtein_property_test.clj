(ns vinary-tree.liblevenshtein-property-test
  "C8 native property-based tests for the Clojure facade (test.check).

  Every property is checked against an in-language brute-force Levenshtein
  oracle. The Clojure facade is transducer-only (it exposes no standalone
  distance entry point), so:
    (a) symmetry/identity are realized through a singleton-dictionary query whose
        bound max(|a|,|b|) provably admits the peer;
    (b) a query's result set at distance k equals {t in dict : lev(query,t)<=k}
        over a random dictionary, with exact distances and value round-trips;
    (c) u64 value round-trips, with 0 and boundary long bit patterns pinned.

  test.check runs from a fixed seed via defspec, so a failing run reproduces."
  (:require [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.string :as str]
            [vinary-tree.liblevenshtein :as llev])
  (:import
   (io.vinarytree.interop UnicodeDictionaryResource UnicodeDictionarySnapshot UnicodeDictionarySnapshot$Edge)
   (java.util OptionalLong)
   (java.util.function Supplier)))

(defn- snapshot [entries]
  (let [nodes (atom [{:edges (sorted-map) :final false :value nil}])]
    (doseq [[term id] entries]
      (let [terminal
            (reduce
             (fn [node scalar]
               (if-let [child (get-in @nodes [node :edges scalar])]
                 child
                 (let [child (count @nodes)]
                   (swap! nodes
                          #(-> %
                               (assoc-in [node :edges scalar] child)
                               (conj {:edges (sorted-map) :final false :value nil})))
                   child)))
             0
             (.toArray (.codePoints ^String term)))]
        (swap! nodes #(-> %
                          (assoc-in [terminal :final] true)
                          (assoc-in [terminal :value] id)))))
    (let [revision @nodes]
      (reify UnicodeDictionarySnapshot
        (root [_] 0)
        (size [_] (OptionalLong/of (count entries)))
        (isFinal [_ node] (boolean (get-in revision [(int node) :final])))
        (value [_ node]
          (if-some [id (get-in revision [(int node) :value])]
            (OptionalLong/of (long id))
            (OptionalLong/empty)))
        (edges [_ node]
          (mapv (fn [[scalar child]]
                  (UnicodeDictionarySnapshot$Edge. (int scalar) (long child)))
                (get-in revision [(int node) :edges])))))))

(defn- levenshtein [a b]
  (let [a (vec a) b (vec b) la (count a) lb (count b)]
    (cond
      (= a b) 0
      (zero? la) lb
      (zero? lb) la
      :else
      (peek
       (reduce
        (fn [previous i]
          (reduce
           (fn [current j]
             (let [cost (if (= (a (dec i)) (b (dec j))) 0 1)]
               (conj current (min (inc (previous j)) (inc (peek current)) (+ cost (previous (dec j)))))))
           [i]
           (range 1 (inc lb))))
        (vec (range (inc lb)))
        (range 1 (inc la)))))))

(defn- run-query [entries query k]
  (with-open [dictionary (UnicodeDictionaryResource. (reify Supplier (get [_] (snapshot entries))))
              automaton (llev/transducer dictionary)]
    (into {} (map (fn [m] [(:term m) [(:distance m) (:id m)]])) (llev/query automaton query k))))

(defn- transducer-distance [a b]
  (let [bound (max (count a) (count b))
        got (run-query {b 0} a bound)]
    (first (get got b))))

;; The Clojure facade presents a u64 id as an UNSIGNED value (a BigInt once it
;; exceeds Long/MAX_VALUE), whereas the interop snapshot's value channel accepts
;; a signed long. Generated dictionary values are therefore kept in
;; [0, Long/MAX_VALUE], where the stored signed long equals its unsigned
;; readback; the full unsigned range (including boundary patterns) is exercised
;; by the round-trip test below via `unsigned`.
(def ^:private alphabet [\a \b \c \é])
(def ^:private gen-term (gen/fmap str/join (gen/vector (gen/elements alphabet) 0 6)))
(def ^:private gen-value (gen/one-of [(gen/return nil) (gen/choose 0 Long/MAX_VALUE)]))
(def ^:private gen-dictionary (gen/fmap #(into {} %) (gen/vector (gen/tuple gen-term gen-value) 0 8)))
(def ^:private gen-k (gen/choose 0 3))

;; (a) symmetry and identity, realized through the transducer.
(defspec distance-symmetry-and-identity 200
  (prop/for-all [a gen-term b gen-term]
    (and (= (transducer-distance a b) (transducer-distance b a) (levenshtein a b))
         (zero? (transducer-distance a a)))))

;; (b) result set equals the oracle, with exact distances and (c) value round-trips.
(defspec query-result-set-equals-oracle 150
  (prop/for-all [entries gen-dictionary query gen-term k gen-k]
    (let [got (run-query entries query k)
          expected (into {} (filter (fn [[t _]] (<= (levenshtein query t) k)) entries))]
      (and (= (set (keys got)) (set (keys expected)))
           (every? (fn [[t [d id]]]
                     (and (= d (levenshtein query t)) (= id (get expected t))))
                   got)))))

;; (c) u64 value round-trip with boundary bit patterns pinned. The snapshot
;; stores a signed long; the facade returns its unsigned interpretation.
(defn- unsigned [signed-long]
  (if (neg? signed-long) (+ (bigint signed-long) 18446744073709551616N) signed-long))

(deftest u64-value-round-trip
  (doseq [value [0 1 Long/MAX_VALUE Long/MIN_VALUE -1]]
    (let [[distance id] (get (run-query {"term" value} "term" 0) "term")]
      (is (and (zero? distance) (== id (unsigned value)))
          (str "value " value " (u64 " (unsigned value) ") did not round-trip; got " id)))))
