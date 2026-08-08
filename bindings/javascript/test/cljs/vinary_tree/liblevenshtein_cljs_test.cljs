(ns vinary-tree.liblevenshtein-cljs-test
  (:require [cljs.test :refer-macros [deftest is run-tests]]
            [vinary-tree.libdictenstein :as dictionary]
            [vinary-tree.liblevenshtein :as levenshtein]))

(deftest clojure-and-clojurescript-facades-have-the-same-resource-shape
  (let [words (dictionary/dynamic-dawg)]
    (try
      (dictionary/put-all! words {"cat" (js/BigInt 1) "cot" (js/BigInt 2) "cut" nil})
      (is (= {:present? true :value nil} (dictionary/get words "cut")))
      (let [automaton (levenshtein/transducer words)
            cursor (levenshtein/query automaton "cat" 2 {:order :distance-then-term})
            first-result (.-value (.next cursor))]
        (try
          (dictionary/remove! words "cot")
          (dictionary/put! words "cit" (js/BigInt 4))
          (dictionary/clear! words)
          (dictionary/compact! words)
          (dictionary/put! words "after" (js/BigInt 5))
          (let [frozen (cons first-result (doall (levenshtein/cursor-seq cursor)))]
            (is (= ["cat" "cot" "cut"]
                   (mapv #(.-value (.-term %)) frozen))))
          (finally
            (levenshtein/close! cursor)
            (levenshtein/close! automaton))))
      (finally (dictionary/close! words)))))

(deftest phonetic-facade
  (let [pattern (levenshtein/phonetic-pattern "c[ao]t")]
    (try
      (is (.matches pattern "cat"))
      (is (not (.matches pattern "cut")))
      (finally (levenshtein/close! pattern)))))

(let [result (run-tests 'vinary-tree.liblevenshtein-cljs-test)]
  (set! (.-exitCode js/process) (if (zero? (+ (:fail result) (:error result))) 0 1)))
