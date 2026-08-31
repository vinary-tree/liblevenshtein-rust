(ns vinary-tree.liblevenshtein
  "ClojureScript facade mirroring the Clojure resource/transducer API."
  (:require ["@vinary-tree/liblevenshtein" :as native]))

(defn transducer
  ([dictionary] (native/transducer dictionary "standard"))
  ([dictionary {:keys [algorithm] :or {algorithm :standard}}]
   (native/transducer dictionary (name algorithm))))

(defn query-cache
  "Retain a transducer behind a hard-bounded snapshot-aware result cache."
  ([automaton] (native/queryCache automaton))
  ([automaton {:keys [maximum-entries maximum-weight]
               :or {maximum-entries 1024 maximum-weight (* 64 1024 1024)}}]
   (native/queryCache automaton
                      #js {:maximumEntries maximum-entries
                           :maximumWeight maximum-weight})))

(defn close! [resource] (.close resource))

(defn query
  ([automaton term max-distance]
   (.query automaton term max-distance "traversal"))
  ([automaton term max-distance {:keys [order] :or {order :traversal}}]
   (.query automaton term max-distance (name order))))

(defn cache-stats [cache]
  (let [stats (.-stats cache)]
    {:requests (.-requests stats)
     :hits (.-hits stats)
     :misses (.-misses stats)
     :admissions (.-admissions stats)
     :rejections (.-rejections stats)
     :evictions (.-evictions stats)
     :resident-entries (.-residentEntries stats)
     :resident-weight (.-residentWeight stats)}))

(defn clear-cache! [cache] (.clear cache))
(defn reset-cache-stats! [cache] (.resetStats cache))

(defn reduce-batches
  "Allocation-minimizing batch reduction; borrowed views must not escape f."
  [f initial cursor]
  (try
    (.reduceBatches cursor f initial)
    (finally (close! cursor))))

(defn cursor-seq [cursor]
  (let [iterator (.call (aget cursor (.-iterator js/Symbol)) cursor)]
    (letfn [(step []
              (lazy-seq
               (let [result (.next iterator)]
                 (if (.-done result)
                   (do (close! cursor) nil)
                   (cons (.-value result) (step))))))]
      (step))))

(defn phonetic-pattern [source] (native/phoneticPattern source))
(defn llre-pattern [source] (native/llrePattern source))
(defn phonetic-rules [source] (native/phoneticRules (if (keyword? source) (name source) source)))
(defn rewrite [rules input] (.apply rules input))
