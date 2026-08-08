(ns check-clojure-facades
  "Babashka-compatible source and public-contract validation.

  The JVM FFM implementation is exercised by the Clojure/JDK matrix.  This
  checker deliberately reads, rather than loads, those namespaces so Babashka
  can independently validate their syntax and idiomatic facade surface without
  pretending that Babashka implements java.lang.foreign."
  (:require [clojure.java.io :as io]
            [clojure.set :as set]))

(def ^:private end-of-file (Object.))

(defn- forms [path]
  (with-open [reader (clojure.lang.LineNumberingPushbackReader.
                      (io/reader path))]
    (loop [result []]
      (let [form (read {:eof end-of-file} reader)]
        (if (identical? form end-of-file)
          result
          (recur (conj result form)))))))

(defn- declared-namespace [source-forms]
  (some (fn [form]
          (when (and (seq? form) (= 'ns (first form)))
            (second form)))
        source-forms))

(defn- public-definitions [source-forms]
  (into #{}
        (keep (fn [form]
                (when (and (seq? form)
                           (contains? '#{defn defmacro deftype defrecord
                                         defprotocol}
                                      (first form)))
                  (second form))))
        source-forms))

(def contracts
  [{:path "bindings/clojure/src/vinary_tree/liblevenshtein.clj"
    :namespace 'vinary-tree.liblevenshtein
    :definitions '#{transducer close! ResultCursor query reduce-batches
                     phonetic-pattern llre-pattern phonetic-rules rewrite}}
   {:path "../libdictenstein/bindings/clojure/src/vinary_tree/libdictenstein.clj"
    :namespace 'vinary-tree.libdictenstein
    :definitions '#{dynamic-dawg double-array-trie scdawg
                     create-persistent-artrie open-persistent-artrie
                     create-persistent-vocabulary open-persistent-vocabulary
                     size contains? get put! put-all! remove! clear! compact!
                     checkpoint! contains-substring? frequency vocabulary-term
                     close!}}])

(doseq [{:keys [path namespace definitions]} contracts]
  (let [source-forms (forms path)
        actual-namespace (declared-namespace source-forms)
        actual-definitions (public-definitions source-forms)
        missing (set/difference definitions actual-definitions)]
    (assert (= namespace actual-namespace)
            (str path " declares " actual-namespace ", expected " namespace))
    (assert (empty? missing)
            (str path " is missing public definitions " (sort missing)))))

(println "Babashka parsed and validated both modular Clojure facade contracts")
