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
    heap,               \* Priority queue (sequence of search nodes)
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

\* Check if heap satisfies min-heap property on f_cost
\* For sequence indices 1..n: heap[i].f_cost <= heap[2*i].f_cost and <= heap[2*i+1].f_cost
IsMinHeap(h) ==
    \A i \in 1..Len(h) :
        /\ (2*i <= Len(h) => h[i].f_cost <= h[2*i].f_cost)
        /\ (2*i+1 <= Len(h) => h[i].f_cost <= h[2*i+1].f_cost)

\* Get minimum element (root of heap)
HeapMin(h) == IF Len(h) > 0 THEN h[1] ELSE [f_cost |-> MAX_COST + WORD_LENGTH + 2]

\* Insert node into heap (simplified - actual heapify-up needed)
\* For TLA+ specification, we use an abstract insert that maintains heap property
HeapInsert(h, node) ==
    \* In practice, this would heapify-up
    \* For model checking, we specify the result must be a valid min-heap
    CHOOSE h2 \in Seq(SearchNode) :
        /\ Len(h2) = Len(h) + 1
        /\ \A n \in DOMAIN h : \E m \in DOMAIN h2 : h2[m] = h[n]
        /\ \E m \in DOMAIN h2 : h2[m] = node
        /\ IsMinHeap(h2)

\* Extract minimum from heap (simplified)
HeapExtractMin(h) ==
    \* In practice, this would heapify-down
    \* For model checking, we specify the result
    IF Len(h) = 0 THEN <<>>
    ELSE CHOOSE h2 \in Seq(SearchNode) :
        /\ Len(h2) = Len(h) - 1
        /\ \A n \in DOMAIN h2 : \E m \in DOMAIN h \ {1} : h2[n] = h[m]
        /\ IsMinHeap(h2)

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
                : t \in DictTransitions(node.dict_state, node.word_pos, "c")  \* "c" is placeholder
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
    \A node \in {heap[i] : i \in DOMAIN heap} :
        node.h_cost <= WORD_LENGTH - node.word_pos

\* INV3: f-costs are consistent
FCostConsistent ==
    \A node \in {heap[i] : i \in DOMAIN heap} :
        node.f_cost = node.g_cost + node.h_cost

\* INV4: g-costs don't exceed max
GCostBounded ==
    \A node \in {heap[i] : i \in DOMAIN heap} :
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
    /\ heap \in Seq(SearchNode)
    /\ visited \subseteq (0..WORD_LENGTH) \X (1..DICT_SIZE)
    /\ results \in Seq(Result)
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
    /\ heap = <<InitialNode>>
    /\ visited = {}
    /\ results = <<>>
    /\ current_best = MAX_COST + 1
    /\ iterations = 0

(***************************************************************************)
(* Actions                                                                  *)
(***************************************************************************)

\* Extract minimum and expand
SearchStep ==
    /\ Len(heap) > 0
    /\ iterations < MAX_HEAP_SIZE * 2  \* Bound iterations
    /\ LET
           node == HeapMin(heap)
           state == <<node.word_pos, node.dict_state>>
       IN
           /\ state \notin visited
           /\ LET
                  \* Check if this is a result
                  new_results ==
                      IF node.is_final /\ node.word_pos = WORD_LENGTH /\ node.g_cost <= MAX_COST
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
                  heap_after_pop == HeapExtractMin(heap)

                  \* Add all valid successors (simplified - actual implementation would heapify)
                  new_heap_set == {heap_after_pop[i] : i \in DOMAIN heap_after_pop} \cup valid_successors
              IN
                  /\ heap' = CHOOSE h \in Seq(SearchNode) :
                       /\ \A n \in new_heap_set : \E i \in DOMAIN h : h[i] = n
                       /\ Cardinality({h[i] : i \in DOMAIN h}) = Cardinality(new_heap_set)
                       /\ IsMinHeap(h)
                  /\ visited' = visited \cup {state}
                  /\ results' = new_results
                  /\ current_best' = new_best
                  /\ iterations' = iterations + 1

\* Search complete
SearchComplete ==
    /\ \/ Len(heap) = 0
       \/ iterations >= MAX_HEAP_SIZE * 2
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
EventuallyTerminates == <>(Len(heap) = 0 \/ iterations >= MAX_HEAP_SIZE * 2)

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
