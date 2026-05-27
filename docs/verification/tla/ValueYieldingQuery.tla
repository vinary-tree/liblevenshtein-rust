-------------------------------- MODULE ValueYieldingQuery --------------------------------
(***************************************************************************)
(* TLA+ specification for the value-yielding transducer query (G9).         *)
(*                                                                          *)
(* Models `ValueYieldingQueryIterator` / `Transducer::query_values`          *)
(* (src/transducer/value_filtered_query.rs): a BFS over the intersection of  *)
(* the dictionary with the Levenshtein automaton that yields                 *)
(* (term, distance, value) for each match within the edit-distance threshold,*)
(* reading the value DURING traversal and SKIPPING final nodes whose value   *)
(* is None.                                                                   *)
(*                                                                          *)
(* The genuinely new surface of G9 (relative to the already-validated        *)
(* query / value-filtered traversal) is: (i) the value read equals the       *)
(* dictionary's stored value for the matched term, and (ii) valueless finals  *)
(* are skipped. This model checks exactly those, plus soundness, dedup, and   *)
(* completeness, over a small concrete dictionary that includes valued and    *)
(* valueless finals, in-range and out-of-range finals, and a shared-term      *)
(* (dedup) case. The nondeterministic processing order makes TLC verify the   *)
(* invariants are order-independent.                                          *)
(*                                                                          *)
(* Companion to the Rust property tests in                                   *)
(* tests/proptest_value_yielding_query.rs (prop_value_yielding_value_parity,  *)
(* _soundness, _completeness, _dedup, _mixed_skip), which check the same      *)
(* facts against the REAL dictionary over generated inputs.                   *)
(*                                                                          *)
(* Corresponds to: src/transducer/value_filtered_query.rs                    *)
(*                 src/transducer/mod.rs (Transducer::query_values)          *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    MaxDistance,        \* Edit-distance threshold
    NoVal               \* Sentinel for a final node with no stored value

ASSUME MaxDistance >= 0

(***************************************************************************)
(* A small concrete dictionary (a BFS neighborhood of the query).           *)
(*                                                                          *)
(*   node 1 : term ""   non-final, root                children {2,4,7}      *)
(*   node 2 : term "a"   final, dist 0, VALUELESS       children {3}         *)
(*   node 3 : term "ab"  final, dist 1, value 100       children {}          *)
(*   node 4 : term "b"   final, dist 1, value 200       children {5}         *)
(*   node 5 : term "bc"  final, dist 2 (OUT of range)   children {6}         *)
(*   node 6 : term "bcd" final, dist 3 (OUT of range)   children {}          *)
(*   node 7 : term "b"   final, dist 1, value 200       children {5}         *)
(*                                                                          *)
(* Nodes 4 and 7 share the term "b" with identical data and children (as in  *)
(* a real trie, where a term determines its node); this exercises the        *)
(* dedup-by-term + skip logic. Terms are encoded as integers.                *)
(***************************************************************************)

Nodes == 1..7
Root  == 1
Terms == 0..5   \* 0:"" 1:"a" 2:"ab" 3:"b" 4:"bc" 5:"bcd"

NodeTerm     == (1 :> 0)     @@ (2 :> 1)    @@ (3 :> 2)   @@ (4 :> 3)
             @@ (5 :> 4)     @@ (6 :> 5)    @@ (7 :> 3)
NodeFinal    == (1 :> FALSE) @@ (2 :> TRUE) @@ (3 :> TRUE) @@ (4 :> TRUE)
             @@ (5 :> TRUE)  @@ (6 :> TRUE) @@ (7 :> TRUE)
NodeDist     == (1 :> 0)     @@ (2 :> 0)    @@ (3 :> 1)   @@ (4 :> 1)
             @@ (5 :> 2)     @@ (6 :> 3)    @@ (7 :> 1)
NodeValue    == (1 :> NoVal) @@ (2 :> NoVal) @@ (3 :> 100) @@ (4 :> 200)
             @@ (5 :> 300)   @@ (6 :> 400)   @@ (7 :> 200)
NodeChildren == (1 :> {2,4,7}) @@ (2 :> {3}) @@ (3 :> {}) @@ (4 :> {5})
             @@ (5 :> {6})     @@ (6 :> {})  @@ (7 :> {5})

ValueSet == {NoVal, 100, 200, 300, 400}
MaxNodeDist == 3

\* The dictionary's stored value for a term: the value of a final node carrying
\* that term (well-defined because same-term nodes carry identical values).
DictValue(t) == NodeValue[CHOOSE n \in Nodes : NodeFinal[n] /\ NodeTerm[n] = t]

\* A yielded result record.
ResultRecord == [term : Terms, dist : 0..MaxNodeDist, value : ValueSet]
EmitRecord(n) == [term |-> NodeTerm[n], dist |-> NodeDist[n], value |-> NodeValue[n]]

(***************************************************************************)
(* State                                                                    *)
(***************************************************************************)

VARIABLES
    pending,   \* set of frontier nodes still to (or being) explored
    done,      \* nodes already popped/processed (BFS visited set)
    seen,      \* set of terms already emitted-or-skipped (dedup discipline)
    results    \* set of yielded (term, distance, value) records

vars == <<pending, done, seen, results>>

Init ==
    /\ pending = {Root}
    /\ done = {}
    /\ seen = {}
    /\ results = {}

(***************************************************************************)
(* The step mirrors `ValueYieldingQueryIterator::next` for one popped node:  *)
(*   - final & in-range & term unseen : mark seen, queue children, and emit  *)
(*     iff the node has a value (valueless finals are skipped);              *)
(*   - final & in-range & term seen   : skip entirely (do NOT queue children);*)
(*   - otherwise (non-final, or final & out-of-range): queue children.       *)
(* `done` is the BFS visited-guard preventing re-processing (a term/node is   *)
(* reached at most once here, since same-term nodes share children).         *)
(***************************************************************************)
Step ==
    \E n \in pending \ done :
        /\ done' = done \cup {n}
        /\ IF NodeFinal[n] /\ NodeDist[n] <= MaxDistance
           THEN IF NodeTerm[n] \notin seen
                THEN /\ seen' = seen \cup {NodeTerm[n]}
                     /\ pending' = pending \cup NodeChildren[n]
                     /\ results' = IF NodeValue[n] # NoVal
                                   THEN results \cup {EmitRecord(n)}
                                   ELSE results
                ELSE /\ pending' = pending
                     /\ UNCHANGED <<seen, results>>
           ELSE /\ pending' = pending \cup NodeChildren[n]
                /\ UNCHANGED <<seen, results>>

\* All reachable nodes processed: the iterator is exhausted.
Terminated == \A n \in pending : n \in done

Next == Step \/ (Terminated /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars /\ WF_vars(Step)

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

TypeOK ==
    /\ pending \subseteq Nodes
    /\ done \subseteq Nodes
    /\ seen \subseteq Terms
    /\ results \subseteq ResultRecord

\* VALUE CORRECTNESS (the headline new property): every yielded value equals
\* the dictionary's stored value for that term -- the iterator's "read during
\* traversal, no second lookup" must agree with a fresh lookup. Mirrors the
\* Rust prop_value_yielding_value_parity.
ValueCorrectness ==
    \A r \in results : r.value = DictValue(r.term)

\* SOUNDNESS: every yielded distance is within the threshold.
Soundness ==
    \A r \in results : r.dist <= MaxDistance

\* NO VALUELESS YIELDED: valueless finals are never emitted.
NoValuelessYielded ==
    \A r \in results : r.value # NoVal

\* DEDUP: no two distinct records share a term (each term yielded at most once).
DedupInv ==
    \A r1 \in results : \A r2 \in results :
        (r1.term = r2.term) => (r1 = r2)

\* COMPLETENESS (monotone form): every processed node that is final, in range,
\* and valued has its term represented in the results. At termination (all
\* reachable nodes processed) this is full completeness over the dictionary.
CompletenessInv ==
    \A n \in done :
        (NodeFinal[n] /\ NodeDist[n] <= MaxDistance /\ NodeValue[n] # NoVal)
            => (\E r \in results : r.term = NodeTerm[n])

(***************************************************************************)
(* Temporal property: the traversal terminates (the iterator is finite).    *)
(***************************************************************************)

EventuallyTerminates == <>Terminated

(***************************************************************************)
(* Theorems (checked by TLC; see docs/verification/tla/states/).            *)
(***************************************************************************)

THEOREM Spec => []TypeOK
THEOREM Spec => []ValueCorrectness
THEOREM Spec => []Soundness
THEOREM Spec => []NoValuelessYielded
THEOREM Spec => []DedupInv
THEOREM Spec => []CompletenessInv
THEOREM Spec => EventuallyTerminates

================================================================================
