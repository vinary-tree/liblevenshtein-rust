(ns vinary-tree.liblevenshtein-test-runner
  (:require [clojure.test :as test]
            [vinary-tree.liblevenshtein-test]
            [vinary-tree.liblevenshtein-property-test]))

(defn -main [& _]
  (let [result (test/run-tests 'vinary-tree.liblevenshtein-test
                               'vinary-tree.liblevenshtein-property-test)]
    (when (pos? (+ (:fail result) (:error result)))
      (System/exit 1))))
