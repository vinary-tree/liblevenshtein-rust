(ns vinary-tree.liblevenshtein
  "Idiomatic reducible facade over the Java FFM resource bindings."
  (:import
   (clojure.lang IReduceInit Seqable)
   (io.vinarytree.interop
    DictionaryBatchLimits DictionaryEntriesMetadata DictionaryEntry DictionaryEntryIterator
    DictionaryKey DictionaryResource DictionarySnapshot DictionaryUnitDomain
    DictionaryValueDomain SnapshotIdentity UnsignedLong)
   (io.vinarytree.liblevenshtein
    Algorithm BorrowedBatchConsumer BorrowedMatchBatch Match PhoneticPattern
    PhoneticRuleSet PhoneticRuleSetKind QueryCursor QueryOrder Term$Bytes
    Term$U64 Term$Utf8 Transducer)
   (java.lang AutoCloseable Iterable)
   (java.util Iterator OptionalLong)
   (java.util.concurrent.atomic AtomicBoolean)))

(def ^:private algorithms
  {:standard Algorithm/STANDARD
   :transposition Algorithm/TRANSPOSITION
   :merge-and-split Algorithm/MERGE_AND_SPLIT
   :merge_and_split Algorithm/MERGE_AND_SPLIT
   :damerau-levenshtein Algorithm/DAMERAU_LEVENSHTEIN
   :damerau_levenshtein Algorithm/DAMERAU_LEVENSHTEIN})

(def ^:private orders
  {:traversal QueryOrder/TRAVERSAL
   :distance-then-term QueryOrder/DISTANCE_THEN_TERM
   :distance_then_term QueryOrder/DISTANCE_THEN_TERM})

(defn- enum-value [values value label]
  (or (get values value)
      (throw (IllegalArgumentException. (str "unknown " label ": " value)))))

(defn- unsigned-id [^OptionalLong id]
  (when (.isPresent id)
    (bigint (Long/toUnsignedString (.getAsLong id)))))

(defn- unsigned-long [^UnsignedLong value]
  (bigint (Long/toUnsignedString (.bits value))))

(defn- unit-domain [^DictionaryUnitDomain domain]
  (condp = domain
    DictionaryUnitDomain/BYTE :byte
    DictionaryUnitDomain/UNICODE_SCALAR :unicode-scalar
    DictionaryUnitDomain/U64 :u64))

(defn- value-domain [^DictionaryValueDomain domain]
  (condp = domain
    DictionaryValueDomain/UNIT :unit
    DictionaryValueDomain/OPTIONAL_U64 :optional-u64))

(defn- dictionary-key-value [^DictionaryKey key]
  (condp = (.domain key)
    DictionaryUnitDomain/BYTE (mapv #(bit-and (int %) 0xff) (.bytes key))
    DictionaryUnitDomain/UNICODE_SCALAR (.unicode key)
    DictionaryUnitDomain/U64
    (mapv #(bigint (Long/toUnsignedString %)) (.u64 key))))

(defn- dictionary-entry-value [^DictionaryEntry entry]
  (let [value (.value entry)]
    {:key (dictionary-key-value (.key entry))
     :value (when (.isPresent value) (unsigned-long (.get value)))}))

(defn- snapshot-identity [identity]
  (when (.isPresent identity)
    (let [^SnapshotIdentity value (.get identity)]
      {:producer (bigint (Long/toUnsignedString (.producer value)))
       :revision (bigint (Long/toUnsignedString (.revision value)))})))

(defn- entries-metadata [^DictionaryEntriesMetadata metadata]
  {:unit-domain (unit-domain (.unitDomain metadata))
   :value-domain (value-domain (.valueDomain metadata))
   :exact-length (unsigned-id (.exactLength metadata))
   :snapshot-identity (snapshot-identity (.snapshotIdentity metadata))})

(defn- batch-limits [{:keys [max-entries max-units max-values]
                      :or {max-entries 256 max-units 65536 max-values 256}}]
  (DictionaryBatchLimits. (long max-entries) (long max-units) (long max-values)))

(defn- term-value [term]
  (cond
    (instance? Term$Utf8 term) (.value ^Term$Utf8 term)
    (instance? Term$Bytes term) (.value ^Term$Bytes term)
    (instance? Term$U64 term) (mapv #(bigint (Long/toUnsignedString %)) (.value ^Term$U64 term))
    :else (throw (IllegalStateException. "unknown term domain"))))

(defn- match-value [^Match match]
  {:term (term-value (.term match))
   :distance (.distance match)
   :id (unsigned-id (.id match))})

(defn transducer
  "Retain a libdictenstein dictionary resource in O(1)."
  ([^DictionaryResource dictionary]
   (Transducer. dictionary))
  ([^DictionaryResource dictionary {:keys [algorithm] :or {algorithm :standard}}]
   (Transducer. dictionary (enum-value algorithms algorithm "algorithm"))))

(defn close! [resource]
  (.close ^AutoCloseable resource))

(deftype ResultCursor [^QueryCursor cursor ^AtomicBoolean claimed]
  AutoCloseable
  (close [_] (.close cursor))

  Iterable
  (iterator [_]
    (when-not (.compareAndSet claimed false true)
      (throw (IllegalStateException. "query results may only be consumed once")))
    (let [^Iterator iterator (.iterator cursor)]
      (reify Iterator
        (hasNext [_]
          (try
            (let [more (.hasNext iterator)]
              (when-not more (.close cursor))
              more)
            (catch Throwable throwable
              (.close cursor)
              (throw throwable))))
        (next [_] (match-value (.next iterator))))))

  Seqable
  (seq [this] (iterator-seq (.iterator ^Iterable this)))

  IReduceInit
  (reduce [this f initial]
    (let [source (.iterator ^Iterable this)]
      (try
        (loop [accumulator initial]
          (if (.hasNext source)
            (let [updated (f accumulator (.next source))]
              (if (reduced? updated) @updated (recur updated)))
            accumulator))
        (finally (.close cursor))))))

(deftype DictionaryEntryCursor [^DictionaryEntryIterator cursor ^AtomicBoolean claimed]
  AutoCloseable
  (close [_] (.close cursor))

  Iterable
  (iterator [_]
    (when-not (.compareAndSet claimed false true)
      (throw (IllegalStateException. "dictionary entries may only be consumed once")))
    (reify Iterator
      (hasNext [_]
        (try
          (let [more (.hasNext cursor)]
            (when-not more (.close cursor))
            more)
          (catch Throwable throwable
            (.close cursor)
            (throw throwable))))
      (next [_]
        (try
          (dictionary-entry-value (.next cursor))
          (catch Throwable throwable
            (.close cursor)
            (throw throwable))))))

  Seqable
  (seq [this] (iterator-seq (.iterator ^Iterable this)))

  IReduceInit
  (reduce [this f initial]
    (let [source (.iterator ^Iterable this)]
      (try
        (loop [accumulator initial]
          (if (.hasNext source)
            (let [updated (f accumulator (.next source))]
              (if (reduced? updated) @updated (recur updated)))
            accumulator))
        (finally (.close cursor))))))

(defn dictionary-entries
  "Return a one-shot, closeable, lazy/reducible lexicographic entry stream.
  Each element is `{:key host-value :value nil-or-unsigned-bigint}`. Close an
  abandoned seq with `close!`; `reduce` closes automatically, including on
  `reduced` early termination."
  ([^DictionaryResource dictionary]
   (dictionary-entries dictionary {}))
  ([^DictionaryResource dictionary options]
   (DictionaryEntryCursor.
    (.entryIterator dictionary (batch-limits options))
    (AtomicBoolean. false))))

(defn dictionary-snapshot
  "Materialize one revision as persistent keys/entries plus its ordered entries
  and snapshot metadata. A present key mapped to nil remains distinguishable
  from an absent key with `contains?`."
  ([^DictionaryResource dictionary]
   (dictionary-snapshot dictionary {}))
  ([^DictionaryResource dictionary options]
   (let [^DictionarySnapshot snapshot
         (.entriesSnapshot dictionary (batch-limits options))
         ordered (mapv dictionary-entry-value (.orderedEntries snapshot))]
     {:metadata (entries-metadata (.metadata snapshot))
      :keys (into #{} (map :key) ordered)
      :entries (into {} (map (juxt :key :value)) ordered)
      :ordered-entries ordered})))

(defn query
  "Return a one-shot lazy/reducible query over a query-start snapshot."
  ([^Transducer automaton term max-distance]
   (ResultCursor. (.query automaton term (long max-distance)) (AtomicBoolean. false)))
  ([^Transducer automaton term max-distance {:keys [order] :or {order :traversal}}]
   (ResultCursor.
    (.query automaton term (long max-distance) (enum-value orders order "query order"))
    (AtomicBoolean. false))))

(defn reduce-batches
  "Expert zero-copy reduction. `f` receives each BorrowedMatchBatch; views must
  not escape the call. This is the preferred allocation-minimizing surface."
  [^QueryCursor cursor f]
  (.forEachBatch cursor
                 (reify BorrowedBatchConsumer
                   (accept [_ batch] (f ^BorrowedMatchBatch batch)))))

(defn phonetic-pattern [source]
  (PhoneticPattern/compileRegex source))

(defn llre-pattern [source]
  (PhoneticPattern/compileLlre source))

(defn phonetic-rules
  ([source]
   (if (keyword? source)
     (PhoneticRuleSet/builtin
      (case source
        :english-orthography PhoneticRuleSetKind/ENGLISH_ORTHOGRAPHY
        :english-phonetic PhoneticRuleSetKind/ENGLISH_PHONETIC
        (throw (IllegalArgumentException. (str "unknown built-in rule set: " source)))))
     (PhoneticRuleSet/parse source))))

(defn rewrite [^PhoneticRuleSet rules input]
  (.apply rules input))
