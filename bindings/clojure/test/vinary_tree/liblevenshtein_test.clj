(ns vinary-tree.liblevenshtein-test
  (:require [clojure.test :refer [deftest is testing]]
            [vinary-tree.liblevenshtein :as llev])
  (:import
   (io.vinarytree.interop
    UnicodeDictionaryResource UnicodeDictionarySnapshot
    UnicodeDictionarySnapshot$Edge)
   (io.vinarytree.liblevenshtein NativeException Status Transducer)
   (java.util ArrayList OptionalLong)
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

(defn- query-with [dictionary algorithm term max-distance order]
  (with-open [automaton (llev/transducer dictionary {:algorithm algorithm})
              matches (llev/query automaton term max-distance {:order order})]
    (is (= "vinary_tree.liblevenshtein.ResultCursor"
           (.getName (class matches))))
    (into [] matches)))

(deftest project-owned-phonetic-facade
  (testing "patterns and rules remain liblevenshtein resources"
    (with-open [pattern (llev/phonetic-pattern "c(at|ot)")]
      (is (.matches pattern "cat"))
      (is (.matches pattern "cot")))))

(deftest dictionary-construction-is-not-reexported
  (is (nil? (ns-resolve 'vinary-tree.liblevenshtein 'string-index)))
  (is (nil? (ns-resolve 'vinary-tree.liblevenshtein 'persistent-create))))

(deftest one-long-lived-reducible-keeps-query-start-snapshot
  (let [current (atom (snapshot {"cat" 1 "cot" 2 "cut" 3 "scat" nil}))
        capture (reify Supplier (get [_] @current))]
    (with-open [dictionary (UnicodeDictionaryResource. capture)
                automaton (llev/transducer dictionary)]
      (let [frozen (sort-by :term (into [] (llev/query automaton "cat" 2)))
            cursor (llev/query automaton "cat" 2)
            iterator (.iterator ^Iterable cursor)
            first (.next iterator)]
        ;; Publish insert/remove/update and compact/checkpoint-equivalent
        ;; immutable revisions after partial consumption.
        (reset! current (snapshot {"cat" 1 "cit" 5 "cut" 30 "scat" nil}))
        (reset! current (snapshot {"cat" 1 "cit" 5 "cut" 30 "scat" nil}))
        (is (= frozen (sort-by :term (into [first] (iterator-seq iterator)))))
        (is (not= frozen
                  (sort-by :term (into [] (llev/query automaton "cat" 2)))))))))

(deftest reducer-and-iterator-drain-to-the-same-matches
  (let [current (atom (snapshot {"cat" 1 "cot" 2 "cut" 3}))
        capture (reify Supplier (get [_] @current))]
    (with-open [dictionary (UnicodeDictionaryResource. capture)
                automaton (llev/transducer dictionary)]
      ;; C5: the reducible cursor (IReduceInit/Iterable) materializes the same
      ;; terms that the zero-copy borrowed-batch reducer (reduce-batches over
      ;; forEachBatch) yields, so both consumption paths agree.
      (let [by-iterator (sort (mapv :term (into [] (llev/query automaton "cat" 1))))
            collected (ArrayList.)]
        (llev/reduce-batches
         (.query ^Transducer automaton "cat" (long 1))
         (fn [batch]
           (dotimes [index (.size batch)]
             (.add collected (.utf8 (.get batch index))))))
        (is (= by-iterator (sort (vec collected))))
        (is (= 3 (count by-iterator)))))))

(deftest algorithms-and-orders-have-distinguishing-semantics
  (let [current (snapshot {"ab" 1 "c" 2 "abc" 3
                           "bat" 4 "cat" 5 "cats" 6})
        capture (reify Supplier (get [_] current))]
    (with-open [dictionary (UnicodeDictionaryResource. capture)]
      (let [standard (query-with dictionary :standard "ba" 1 :traversal)
            transposed (query-with dictionary :transposition "ba" 1 :traversal)
            merged (query-with dictionary :merge-and-split "ab" 1 :traversal)
            damerau (query-with dictionary :damerau-levenshtein "ca" 2 :traversal)
            traversal (query-with dictionary :standard "cat" 1 :traversal)
            ranked (query-with dictionary :standard "cat" 1 :distance-then-term)]
        (is (not-any? #(= "ab" (:term %)) standard))
        (is (some #(= {:term "ab" :distance 1}
                      (select-keys % [:term :distance]))
                  transposed))
        (is (some #(= {:term "c" :distance 1}
                      (select-keys % [:term :distance]))
                  merged))
        (is (some #(= {:term "abc" :distance 2}
                      (select-keys % [:term :distance]))
                  damerau))
        (is (= [["bat" 1] ["cat" 0] ["cats" 1]]
               (mapv (juxt :term :distance) traversal)))
        (is (= [["cat" 0] ["bat" 1] ["cats" 1]]
               (mapv (juxt :term :distance) ranked)))))))

(deftest phonetic-rules-and-llre-pattern-are-reusable-resources
  (with-open [pattern (llev/llre-pattern "@name \"Greeting\"\n^hello$")]
    (is (.matches pattern "hello"))
    (is (not (.matches pattern "world"))))
  (with-open [parsed (llev/phonetic-rules "ph -> f\ngh ->\n")]
    (is (= 2 (.size parsed)))
    (is (= "f" (llev/rewrite parsed "phgh"))))
  (doseq [kind [:english-orthography :english-phonetic]]
    (with-open [builtin (llev/phonetic-rules kind)]
      (is (pos? (.size builtin)))
      (is (not-empty (llev/rewrite builtin "phone"))))))

(deftest native-error-retains-typed-and-raw-status-with-diagnostic
  (let [failure (try
                  (llev/phonetic-pattern "(")
                  nil
                  (catch NativeException native-failure native-failure))]
    (is (instance? NativeException failure))
    (when (instance? NativeException failure)
      (is (= Status/INVALID_ARGUMENT (.status ^NativeException failure)))
      (is (= (.code Status/INVALID_ARGUMENT)
             (.statusCode ^NativeException failure)))
      (is (not-empty (.getMessage ^NativeException failure))))))

(deftest query-cache-facade-preserves-bounds-counters-and-provider-requirements
  (let [current (snapshot {"cat" 1})
        capture (reify Supplier (get [_] current))]
    (with-open [dictionary (UnicodeDictionaryResource. capture)
                automaton (llev/transducer dictionary)
                cache (llev/query-cache automaton {:maximum-entries 4
                                                    :maximum-weight 4096})]
      (is (= {:requests 0N :hits 0N :misses 0N :admissions 0N
              :rejections 0N :evictions 0N :resident-entries 0N
              :resident-weight 0N}
             (llev/cache-stats cache)))
      ;; This deliberately host-defined test provider predates stable snapshot
      ;; identities. A cache must reject it diagnostically instead of serving
      ;; potentially stale results; libdictenstein resources provide identity.
      (let [failure (try
                      (llev/query cache "cat" 0)
                      nil
                      (catch NativeException native-failure native-failure))]
        (is (= Status/UNSUPPORTED (.status ^NativeException failure))))
      (is (identical? cache (llev/clear-cache! cache)))
      (is (identical? cache (llev/reset-cache-stats! cache))))))

(deftest phonetic-pattern-idempotent-close-and-closed-guard
  ;; C2/C3: close is idempotent and a closed pattern rejects use with an
  ;; IllegalStateException rather than dereferencing freed native memory.
  (let [pattern (llev/phonetic-pattern "c[ao]t")]
    (is (.matches pattern "cat"))
    (is (not (.matches pattern "cut")))
    (llev/close! pattern)
    (llev/close! pattern)
    (is (thrown? IllegalStateException (.matches pattern "cat")))))
