(ns vinary-tree.liblevenshtein-leak-test
  "C9 leak-discipline suite for the Clojure facade (System/gc + Runtime).

  A >=10,000-cycle create/use/free loop over transducers and query cursors must
  reach a JVM-heap steady state. Native handles are freed at close (with-open);
  this asserts the managed heap (Runtime used memory after a forced collection)
  does not drift upward across the cycles, so a retained wrapper or provider
  holder would surface as unbounded growth."
  (:require [clojure.test :refer [deftest is]]
            [vinary-tree.liblevenshtein :as llev])
  (:import
   (io.vinarytree.interop UnicodeDictionaryResource UnicodeDictionarySnapshot UnicodeDictionarySnapshot$Edge)
   (java.util OptionalLong)
   (java.util.function Supplier)))

(def ^:private cycles 10000)
(def ^:private warmup 2000)
;; Generous ceiling; a per-cycle leak would accrue far more over 10k cycles.
(def ^:private max-growth-bytes (* 32 1024 1024))

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

(defn- settled-used []
  (let [runtime (Runtime/getRuntime)]
    (dotimes [_ 4]
      (System/gc)
      (Thread/sleep 15))
    (- (.totalMemory runtime) (.freeMemory runtime))))

(deftest transducer-cycle-reaches-steady-state
  (let [snap (snapshot {"cat" 1 "cot" 2 "cut" 3 "scat" nil})
        capture (reify Supplier (get [_] snap))
        run-cycle (fn []
                    (with-open [dictionary (UnicodeDictionaryResource. capture)
                                automaton (llev/transducer dictionary)]
                      (doseq [_ (llev/query automaton "cat" 2)] nil)))]
    (dotimes [_ warmup] (run-cycle))
    (let [base (settled-used)]
      (dotimes [_ cycles] (run-cycle))
      (let [growth (- (settled-used) base)]
        (is (< growth max-growth-bytes)
            (str "JVM heap grew " growth " bytes over " cycles " cycles"))))))
