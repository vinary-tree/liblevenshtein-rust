-------------------------------- MODULE ProductAutomaton --------------------------------
(***************************************************************************)
(* TLA+ Specification for Product Automaton (NFA x Levenshtein)             *)
(*                                                                          *)
(* This module specifies the product construction between a phonetic NFA    *)
(* and a Levenshtein automaton. The product accepts strings that:           *)
(* 1. Are accepted by the NFA (match the phonetic pattern)                  *)
(* 2. Have edit distance <= max_distance from some NFA-accepted string      *)
(*                                                                          *)
(* Key Properties to Verify:                                                *)
(* 1. ProductCorrectness: Acceptance iff NFA accepts AND cost <= max_cost   *)
(* 2. TransitionDeterminism: Each (state, char) has defined transitions     *)
(* 3. CostMonotonicity: Cost only increases along paths                     *)
(* 4. StateSpaceBounded: |product_states| <= |nfa_states| * (max_cost+1)^2  *)
(*                                                                          *)
(* Corresponds to: src/phonetic/nfa/product.rs                              *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NFA_STATES,         \* Set of NFA states {0, 1, ..., N-1}
    INITIAL_NFA,        \* Initial NFA state
    FINAL_NFA_STATES,   \* Set of final NFA states
    MAX_COST,           \* Maximum allowed edit cost
    INPUT_LENGTH,       \* Maximum input length for model checking
    ALPHABET            \* Character alphabet

ASSUME INITIAL_NFA \in NFA_STATES
ASSUME FINAL_NFA_STATES \subseteq NFA_STATES

VARIABLES
    product_states,     \* Current set of product states
    input_position,     \* Current position in input
    accepted,           \* Whether input has been accepted
    accepted_witnesses, \* Accepting states observed so far
    cost_map            \* Map from product state to minimum cost

(***************************************************************************)
(* Type Definitions                                                         *)
(***************************************************************************)

\* A product state combines NFA state, pattern position, and accumulated cost
ProductState == [
    nfa_state: NFA_STATES,      \* Current NFA state
    pattern_pos: Nat,           \* Position in pattern string
    error_count: 0..MAX_COST+1, \* Accumulated errors
    is_special: BOOLEAN         \* Special position flag (for transposition)
]

\* NFA transitions (abstract - would be defined by specific NFA)
\* NFATransition[state][char] = set of reachable states
VARIABLE nfa_delta

(***************************************************************************)
(* Helper Functions                                                         *)
(***************************************************************************)

\* Initial product state
InitialProductState == [
    nfa_state |-> INITIAL_NFA,
    pattern_pos |-> 0,
    error_count |-> 0,
    is_special |-> FALSE
]

\* Check if product state is accepting
IsAccepting(ps) ==
    /\ ps.nfa_state \in FINAL_NFA_STATES
    /\ ps.error_count <= MAX_COST

\* Check if product state is dead (exceeded cost)
IsDead(ps) == ps.error_count > MAX_COST

\* Cost of match operation (depends on character match)
MatchCost(matches) == IF matches THEN 0 ELSE 1

\* Apply NFA epsilon transitions
EpsilonClosure(states) ==
    \* In a real implementation, this would compute the epsilon closure
    \* For TLA+ model checking, we abstract this
    states

(***************************************************************************)
(* Product Transitions                                                      *)
(***************************************************************************)

\* Generate successor product states for a given state and input character
\* This implements the product of NFA transitions with Levenshtein operations
ProductSuccessors(ps, c) ==
    IF IsDead(ps) THEN {}
    ELSE
        LET
            \* NFA successors on character c
            nfa_next == IF ps.nfa_state \in DOMAIN nfa_delta
                        THEN nfa_delta[ps.nfa_state][c]
                        ELSE {}

            \* Match/Substitute: advance both NFA and pattern, cost depends on match
            match_subs == { [
                nfa_state |-> ns,
                pattern_pos |-> ps.pattern_pos + 1,
                error_count |-> ps.error_count + 1,  \* Simplified: always 1 for substitute
                is_special |-> FALSE
            ] : ns \in nfa_next }

            \* Insert: advance input only, stay at same pattern position
            insert == { [
                nfa_state |-> ps.nfa_state,
                pattern_pos |-> ps.pattern_pos,
                error_count |-> ps.error_count + 1,
                is_special |-> FALSE
            ] }

            \* Delete: advance pattern only, don't consume input
            \* (This is an epsilon transition, handled separately)
            delete == {}  \* Handled via epsilon closure
        IN
            match_subs \cup insert

\* Epsilon transitions (delete operations, advancing pattern without input)
ProductEpsilonSuccessors(ps) ==
    IF IsDead(ps) THEN {}
    ELSE
        \* Delete: advance pattern position with cost 1
        { [
            nfa_state |-> ps.nfa_state,
            pattern_pos |-> ps.pattern_pos + 1,
            error_count |-> ps.error_count + 1,
            is_special |-> FALSE
        ] }

(***************************************************************************)
(* Subsumption Rules                                                        *)
(***************************************************************************)

\* One product state subsumes another if it represents a better path
Subsumes(ps1, ps2) ==
    /\ ps1.nfa_state = ps2.nfa_state
    /\ ps1.pattern_pos = ps2.pattern_pos
    /\ ps1.error_count <= ps2.error_count
    /\ (ps1.is_special = ps2.is_special \/ ~ps1.is_special)

\* Apply subsumption to remove dominated states
ApplySubsumption(states) ==
    { ps \in states :
        ~\E ps2 \in states : ps # ps2 /\ Subsumes(ps2, ps) }

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

\* INV1: All states have valid error counts
ErrorCountsValid ==
    \A ps \in product_states : ps.error_count <= MAX_COST + 1

\* INV2: State space bounded
StateSpaceBounded ==
    Cardinality(product_states) <= Cardinality(NFA_STATES) * (MAX_COST + 2) * (INPUT_LENGTH + 1)

\* INV3: NFA states are valid
NFAStatesValid ==
    \A ps \in product_states : ps.nfa_state \in NFA_STATES

\* INV4: Acceptance implies reaching final NFA state with bounded cost
AcceptanceValid ==
    accepted => \E ps \in accepted_witnesses :
        /\ ps.nfa_state \in FINAL_NFA_STATES
        /\ ps.error_count <= MAX_COST

\* INV5: Pattern positions are monotonic within states
PatternPositionValid ==
    \A ps \in product_states : ps.pattern_pos <= input_position + MAX_COST

\* Type invariant
TypeInvariant ==
    /\ product_states \subseteq ProductState
    /\ accepted_witnesses \subseteq ProductState
    /\ input_position \in 0..INPUT_LENGTH
    /\ accepted \in BOOLEAN

(***************************************************************************)
(* Initial State                                                            *)
(***************************************************************************)

\* Initialize NFA transitions (abstract model).
\*
\* LIMITATION: this is a placeholder total transition relation (every state goes
\* to every state on every character), so the model checks structural invariants
\* (TypeInvariant, ErrorCountsValid, StateSpaceBounded, NFAStatesValid,
\* AcceptanceValid, PatternPositionValid) but cannot meaningfully verify
\* ProductCorrectness ("accepts iff the NFA accepts AND cost <= max_cost") or
\* CostMonotonicity against a concrete NFA. Those two header goals are instead
\* verified on the real Rust construction by
\* tests/proptest_product_automaton.rs, which checks acceptance/min_distance
\* against the exact edit-distance oracle and cost monotonicity for a literal
\* NFA. Promoting this model to faithful transitions is future work.
InitNFADelta ==
    \* In a real model, this would define the NFA structure
    \* Here we use a simple placeholder
    [s \in NFA_STATES |-> [c \in ALPHABET |-> NFA_STATES]]

Init ==
    /\ product_states = {InitialProductState}
    /\ input_position = 0
    /\ accepted = FALSE
    /\ accepted_witnesses = {}
    /\ cost_map = [ps \in {InitialProductState} |-> 0]
    /\ nfa_delta = InitNFADelta

(***************************************************************************)
(* Actions                                                                  *)
(***************************************************************************)

\* Process one input character
ProcessCharacter(c) ==
    /\ input_position < INPUT_LENGTH
    /\ LET
           \* Compute epsilon closure first
           with_epsilon == UNION {ProductEpsilonSuccessors(ps) : ps \in product_states}
           closed == product_states \cup with_epsilon

           \* Compute successors on character c
           all_successors == UNION {ProductSuccessors(ps, c) : ps \in closed}

           \* Filter dead states
           alive == {ps \in all_successors : ~IsDead(ps)}

           \* Apply subsumption
           pruned == ApplySubsumption(alive)

           \* Check for acceptance
           new_witnesses == {ps \in pruned : IsAccepting(ps)}
           all_witnesses == accepted_witnesses \cup new_witnesses
           new_accepted == accepted \/ new_witnesses # {}
       IN
           /\ product_states' = pruned
           /\ input_position' = input_position + 1
           /\ accepted' = new_accepted
           /\ accepted_witnesses' = all_witnesses
           /\ cost_map' = [ps \in pruned |-> ps.error_count]
           /\ UNCHANGED nfa_delta

\* End of input - final acceptance check
CheckAcceptance ==
    /\ input_position = INPUT_LENGTH
    /\ LET
           \* Final epsilon closure
           with_epsilon == UNION {ProductEpsilonSuccessors(ps) : ps \in product_states}
           final_states == product_states \cup with_epsilon

           \* Check for accepting states
           final_witnesses == {ps \in final_states : IsAccepting(ps)}
           final_accepted == final_witnesses # {}
       IN
           /\ accepted' = accepted \/ final_accepted
           /\ accepted_witnesses' = accepted_witnesses \cup final_witnesses
           /\ UNCHANGED <<product_states, input_position, cost_map, nfa_delta>>

\* Stuttering (no change)
Stutter ==
    /\ UNCHANGED <<product_states, input_position, accepted, accepted_witnesses, cost_map, nfa_delta>>

\* Next state relation
Next ==
    \/ \E c \in ALPHABET : ProcessCharacter(c)
    \/ CheckAcceptance
    \/ Stutter

(***************************************************************************)
(* Fairness                                                                 *)
(***************************************************************************)

FairVars == <<product_states, input_position, accepted, accepted_witnesses, cost_map, nfa_delta>>

Fairness == WF_FairVars(\E c \in ALPHABET : ProcessCharacter(c))

Spec == Init /\ [][Next]_FairVars /\ Fairness

(***************************************************************************)
(* Temporal Properties                                                      *)
(***************************************************************************)

\* Eventually input is fully processed
EventuallyProcessed == <>(input_position = INPUT_LENGTH)

\* If accepted, acceptance persists (monotonicity)
AcceptancePersists == [](accepted => []accepted)

(***************************************************************************)
(* Key Theorems                                                             *)
(***************************************************************************)

\* Product correctness: the product automaton accepts iff
\* there exists a path in the NFA with cost <= MAX_COST
THEOREM Spec => []TypeInvariant
THEOREM Spec => []ErrorCountsValid
THEOREM Spec => []StateSpaceBounded
THEOREM Spec => []NFAStatesValid
THEOREM Spec => []AcceptanceValid
THEOREM Spec => AcceptancePersists
THEOREM Spec => EventuallyProcessed

================================================================================
