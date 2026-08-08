(ns vinary-tree.liblevenshtein-test
  (:require [clojure.test :refer [deftest is testing]]
            [vinary-tree.liblevenshtein :as llev])
  (:import
   (io.vinarytree.interop
    UnicodeDictionaryResource UnicodeDictionarySnapshot
    UnicodeDictionarySnapshot$Edge)
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
