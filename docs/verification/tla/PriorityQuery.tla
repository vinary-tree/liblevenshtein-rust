-------------------------------- MODULE PriorityQuery --------------------------------
(***************************************************************************)
(* TLA+ Specification for Priority Queue A* Search                          *)
(*                                                                          *)
(* This module specifies the priority queue-based search algorithm used     *)
(* for efficient fuzzy matching. It uses A* search with an admissible       *)
(* heuristic to find matches in order of increasing edit distance.          *)
(*                                                                          *)
(* Key Properties to Verify:                                                *)
(* 1. HeapInvariant: Priority queue maintains min-heap property             *)
(* 2. AdmissibleHeuristic: h(n) <= actual_cost(n) for all nodes             *)
(* 3. Optimality: First match found has minimum edit distance               *)
(* 4. Completeness: All matches within bound are eventually found           *)
(*                                                                          *)
(* Corresponds to: src/transducer/priority_query.rs                         *)
(*                                                                          *)
(* MODEL-vs-IMPLEMENTATION NOTE (verified by property tests, see            *)
(* tests/proptest_priority_query.rs): this model is an IDEALIZED admissible  *)
(* A* whose Optimality/AdmissibleHeuristic/ResultsOrdered/FirstResultOptimal *)
(* hold by construction. The Rust `PriorityQueryIterator`, however, uses an  *)
(* INADMISSIBLE heuristic (h = query_len - max_consumed_chars,               *)
(* priority_query.rs:173) and is documented as "Distance-first (approximate  *)
(* lex)" for fast first-k results - it does NOT guarantee optimal ordering.  *)
(* The implementation that actually realizes this model's optimal/ordered    *)
(* guarantees is `OrderedQueryIterator` (Transducer::query_ordered, a        *)
(* distance-layer BFS). Read this spec as the contract of OrderedQuery /     *)
(* an idealized admissible A*, not of the approximate PriorityQueryIterator. *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MAX_COST,           \* Maximum edit distance
    WORD_LENGTH,        \* Length of query word
    DICT_SIZE,          \* Number of dictionary entries for model checking
    MAX_HEAP_SIZE       \* Maximum heap size for bounded model checking

ASSUME MAX_COST >= 0
ASSUME WORD_LENGTH >= 1
ASSUME DICT_SIZE >= 1
ASSUME MAX_HEAP_SIZE >= 1

VARIABLES
    heap,               \* Priority queue abstraction as a bounded set of nodes
    visited,            \* Set of visited states
    results,            \* Sequence of found results (in priority order)
    current_best,       \* Current best cost found
    iterations          \* Iteration counter

(***************************************************************************)
(* Type Definitions                                                         *)
(***************************************************************************)

\* A search node in the priority queue
SearchNode == [
    word_pos: 0..WORD_LENGTH,           \* Position in query word
    dict_state: 1..DICT_SIZE,           \* Current dictionary state
    g_cost: 0..MAX_COST+1,              \* Actual cost so far
    h_cost: 0..WORD_LENGTH,             \* Heuristic estimate to goal
    f_cost: 0..MAX_COST+WORD_LENGTH+1,  \* f = g + h (total estimated cost)
    is_final: BOOLEAN                    \* Is this a final dictionary state?
]

\* A result record
Result == [
    dict_state: 1..DICT_SIZE,
    cost: 0..MAX_COST
]

\* TLC needs enumerable sequence domains for result histories.
BoundedSeq(S, max_len) == UNION { [1..n -> S] : n \in 0..max_len }

(***************************************************************************)
(* Heuristic Function                                                       *)
(***************************************************************************)

\* The heuristic estimates remaining cost based on remaining characters
\* h(n) = max(0, remaining_chars - remaining_budget)
\* This is admissible because we need at least one operation per unmatched char
Heuristic(word_pos, g_cost) ==
    LET remaining == WORD_LENGTH - word_pos
        budget == MAX_COST - g_cost
    IN IF remaining > budget THEN remaining - budget ELSE 0

\* Alternative: simple remaining characters (also admissible)
SimpleHeuristic(word_pos) == WORD_LENGTH - word_pos

(***************************************************************************)
(* Heap Operations                                                          *)
(***************************************************************************)

\* The implementation uses a binary heap.  This model keeps only the semantic
\* priority-queue abstraction: a bounded finite set with an f_cost minimum.
IsMinHeap(h) ==
    /\ h \subseteq SearchNode
    /\ Cardinality(h) <= MAX_HEAP_SIZE
    /\ h = {} \/ \E n \in h : \A m \in h : n.f_cost <= m.f_cost

\* Get a minimum-priority element.
HeapMin(h) ==
    CHOOSE n \in h : \A m \in h : n.f_cost <= m.f_cost

\* Insert node into the priority-queue abstraction.
HeapInsert(h, node) ==
    h \cup {node}

\* Extract a minimum-priority element.
HeapExtractMin(h) ==
    IF h = {} THEN {}
    ELSE h \ {HeapMin(h)}

(***************************************************************************)
(* Search State Transitions                                                 *)
(***************************************************************************)

\* Dictionary transition function (abstract)
\* Returns set of (next_state, is_final, transition_cost) tuples
DictTransitions(state, word_pos, char) ==
    \* Abstract: in practice, determined by dictionary structure
    \* For model checking, we use a simplified model
    { [next_state |-> s, is_final |-> (s = DICT_SIZE), cost |-> 1]
      : s \in 1..DICT_SIZE }

\* Expand a search node to generate successors
ExpandNode(node) ==
    IF node.word_pos >= WORD_LENGTH THEN
        \* At end of word - only epsilon transitions (deletions from dict)
        {}
    ELSE
        LET
            \* Match: advance both word and dict
            match_successors == {
                [word_pos |-> node.word_pos + 1,
                 dict_state |-> t.next_state,
                 g_cost |-> node.g_cost + t.cost,
                 h_cost |-> Heuristic(node.word_pos + 1, node.g_cost + t.cost),
                 f_cost |-> node.g_cost + t.cost + Heuristic(node.word_pos + 1, node.g_cost + t.cost),
                 is_final |-> t.is_final]
                : t \in DictTransitions(node.dict_state, node.word_pos, "c")  \* representative model input symbol
            }

            \* Insert: advance word only (insertion into query)
            insert_successors == {
                [word_pos |-> node.word_pos + 1,
                 dict_state |-> node.dict_state,
                 g_cost |-> node.g_cost + 1,
                 h_cost |-> Heuristic(node.word_pos + 1, node.g_cost + 1),
                 f_cost |-> node.g_cost + 1 + Heuristic(node.word_pos + 1, node.g_cost + 1),
                 is_final |-> FALSE]
            }
        IN
            {n \in (match_successors \cup insert_successors) :
                n.g_cost <= MAX_COST}

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

\* INV1: Heap maintains min-heap property
HeapInvariant == IsMinHeap(heap)

\* INV2: Heuristic is admissible (never overestimates)
\* For our heuristic: h(n) <= remaining_characters <= actual_remaining_cost
AdmissibleHeuristic ==
    \A node \in heap :
        node.h_cost <= WORD_LENGTH - node.word_pos

\* INV3: f-costs are consistent
FCostConsistent ==
    \A node \in heap :
        node.f_cost = node.g_cost + node.h_cost

\* INV4: g-costs don't exceed max
GCostBounded ==
    \A node \in heap :
        node.g_cost <= MAX_COST + 1

\* INV5: Results are in non-decreasing cost order
ResultsOrdered ==
    \A i, j \in DOMAIN results :
        i < j => results[i].cost <= results[j].cost

\* INV6: All results have cost <= MAX_COST
ResultsBounded ==
    \A i \in DOMAIN results : results[i].cost <= MAX_COST

\* INV7: First result (if any) has optimal cost
\* (This is the key A* optimality property)
FirstResultOptimal ==
    Len(results) > 0 =>
        \A r \in {results[i] : i \in DOMAIN results} :
            results[1].cost <= r.cost

\* Type invariant
TypeInvariant ==
    /\ heap \subseteq SearchNode
    /\ Cardinality(heap) <= MAX_HEAP_SIZE
    /\ visited \subseteq (0..WORD_LENGTH) \X (1..DICT_SIZE)
    /\ results \in BoundedSeq(Result, MAX_HEAP_SIZE)
    /\ current_best \in 0..MAX_COST+1
    /\ iterations \in Nat

(***************************************************************************)
(* Initial State                                                            *)
(***************************************************************************)

\* Initial search node at start of word and dictionary
InitialNode == [
    word_pos |-> 0,
    dict_state |-> 1,  \* Root of dictionary
    g_cost |-> 0,
    h_cost |-> Heuristic(0, 0),
    f_cost |-> Heuristic(0, 0),
    is_final |-> FALSE
]

Init ==
    /\ heap = {InitialNode}
    /\ visited = {}
    /\ results = <<>>
    /\ current_best = MAX_COST + 1
    /\ iterations = 0

(***************************************************************************)
(* Actions                                                                  *)
(***************************************************************************)

\* Extract minimum and expand
SearchStep ==
    /\ heap # {}
    /\ \E n \in heap : <<n.word_pos, n.dict_state>> \notin visited
    /\ iterations < (MAX_HEAP_SIZE * 2)  \* Bound iterations
    /\ LET
           available == {n \in heap : <<n.word_pos, n.dict_state>> \notin visited}
           node == HeapMin(available)
           state == <<node.word_pos, node.dict_state>>
       IN
           /\ LET
                  \* Check if this is a result
                  new_results ==
                      IF node.is_final /\ node.word_pos = WORD_LENGTH /\ node.g_cost <= MAX_COST /\ Len(results) < MAX_HEAP_SIZE
                      THEN Append(results, [dict_state |-> node.dict_state, cost |-> node.g_cost])
                      ELSE results

                  new_best ==
                      IF node.is_final /\ node.word_pos = WORD_LENGTH /\ node.g_cost < current_best
                      THEN node.g_cost
                      ELSE current_best

                  \* Expand node
                  successors == ExpandNode(node)

                  \* Filter successors not yet visited and within cost bound
                  valid_successors == {
                      s \in successors :
                          /\ <<s.word_pos, s.dict_state>> \notin visited
                          /\ s.f_cost <= MAX_COST
                  }

                  \* Remove min from heap and add successors
                  heap_after_pop == heap \ {node}

                  \* Add all valid successors.  If the configured model cap is
                  \* exceeded, terminate this bounded search branch.
                  new_heap_set == heap_after_pop \cup valid_successors
              IN
                  /\ heap' = IF Cardinality(new_heap_set) <= MAX_HEAP_SIZE
                             THEN new_heap_set
                             ELSE {}
                  /\ visited' = visited \cup {state}
                  /\ results' = new_results
                  /\ current_best' = new_best
                  /\ iterations' = iterations + 1

\* Search complete
SearchComplete ==
    /\ \/ heap = {}
       \/ ~(\E n \in heap : <<n.word_pos, n.dict_state>> \notin visited)
       \/ iterations >= (MAX_HEAP_SIZE * 2)
    /\ UNCHANGED <<heap, visited, results, current_best, iterations>>

Next == SearchStep \/ SearchComplete

(***************************************************************************)
(* Fairness and Specification                                               *)
(***************************************************************************)

Fairness == WF_<<heap, visited, results, current_best, iterations>>(SearchStep)

Spec == Init /\ [][Next]_<<heap, visited, results, current_best, iterations>> /\ Fairness

(***************************************************************************)
(* Temporal Properties                                                      *)
(***************************************************************************)

\* Eventually search terminates
EventuallyTerminates ==
    <>(heap = {} \/ ~(\E n \in heap : <<n.word_pos, n.dict_state>> \notin visited)
       \/ iterations >= (MAX_HEAP_SIZE * 2))

\* Optimality: if we find any result, we find the best one first
OptimalityProperty ==
    [](Len(results) >= 2 => results[1].cost <= results[2].cost)

(***************************************************************************)
(* Theorems                                                                 *)
(***************************************************************************)

THEOREM Spec => []TypeInvariant
THEOREM Spec => []HeapInvariant
THEOREM Spec => []AdmissibleHeuristic
THEOREM Spec => []FCostConsistent
THEOREM Spec => []GCostBounded
THEOREM Spec => []ResultsOrdered
THEOREM Spec => []ResultsBounded
THEOREM Spec => []FirstResultOptimal
THEOREM Spec => EventuallyTerminates
THEOREM Spec => OptimalityProperty

(***************************************************************************)
(* Model Checking Configuration                                             *)
(***************************************************************************)

\* For efficient model checking:
\* MAX_COST = 2
\* WORD_LENGTH = 3
\* DICT_SIZE = 4
\* MAX_HEAP_SIZE = 20
\*
\* This keeps the state space manageable while still being able to
\* verify the key properties of the A* search algorithm.

================================================================================
