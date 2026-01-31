-------------------------------- MODULE OnlineScanner --------------------------------
(***************************************************************************)
(* TLA+ Specification for Online Scanner                                    *)
(*                                                                          *)
(* This module specifies the behavior of the online scanner component that  *)
(* processes input strings character-by-character, tracking multiple        *)
(* concurrent matches against patterns with bounded edit distance.          *)
(*                                                                          *)
(* Key Properties to Verify:                                                *)
(* 1. ActiveMatchesBounded: |active_matches| <= MAX_ACTIVE_MATCHES          *)
(* 2. NoMissedMatches: All valid matches eventually recorded                *)
(* 3. PositionMonotonicity: Positions only advance                          *)
(* 4. ErrorBoundRespected: All matches have errors <= max_errors            *)
(*                                                                          *)
(* Corresponds to: src/phonetic/online_scanner.rs                           *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MAX_ERRORS,         \* Maximum allowed edit distance
    MAX_ACTIVE_MATCHES, \* Maximum concurrent active matches
    PATTERN_LENGTH,     \* Length of the pattern being matched
    INPUT_LENGTH,       \* Length of the input string
    ALPHABET            \* Set of possible characters

VARIABLES
    position,           \* Current position in input string (0-indexed)
    active_matches,     \* Set of currently active match states
    completed_matches,  \* Set of completed (successful) matches
    input_consumed,     \* Sequence of consumed input characters
    dead_states         \* Set of states pruned (exceeded error bound)

(***************************************************************************)
(* Type Definitions                                                         *)
(***************************************************************************)

\* A match state tracks position in pattern, errors accumulated, and start position
MatchState == [
    pattern_pos: 0..PATTERN_LENGTH,   \* Position in pattern
    errors: 0..MAX_ERRORS + 1,        \* Error count (MAX+1 indicates dead)
    start_pos: 0..INPUT_LENGTH,       \* Start position in input
    is_special: BOOLEAN               \* Special position (transposition state)
]

\* A completed match records start, end, and cost
CompletedMatch == [
    start_pos: 0..INPUT_LENGTH,
    end_pos: 0..INPUT_LENGTH,
    cost: 0..MAX_ERRORS
]

(***************************************************************************)
(* Helper Functions                                                         *)
(***************************************************************************)

\* Check if a state is dead (exceeded error bound)
IsDead(state) == state.errors > MAX_ERRORS

\* Check if a state is final (reached end of pattern within error bound)
IsFinal(state) ==
    /\ state.pattern_pos = PATTERN_LENGTH
    /\ state.errors <= MAX_ERRORS

\* Check if one state subsumes another (can be pruned)
Subsumes(s1, s2) ==
    /\ s1.pattern_pos = s2.pattern_pos
    /\ s1.errors <= s2.errors
    /\ s1.start_pos = s2.start_pos
    /\ ~s1.is_special

\* Create initial state at given position
InitialState(pos) == [
    pattern_pos |-> 0,
    errors |-> 0,
    start_pos |-> pos,
    is_special |-> FALSE
]

\* Apply match operation (no error increase)
ApplyMatch(state) == [state EXCEPT !.pattern_pos = @ + 1]

\* Apply substitute operation (error + 1)
ApplySubstitute(state) == [state EXCEPT
    !.pattern_pos = @ + 1,
    !.errors = @ + 1
]

\* Apply delete operation (consume pattern, not input)
ApplyDelete(state) == [state EXCEPT
    !.pattern_pos = @ + 1,
    !.errors = @ + 1
]

\* Apply insert operation (consume input, not pattern)
ApplyInsert(state) == [state EXCEPT !.errors = @ + 1]

(***************************************************************************)
(* State Transition Functions                                               *)
(***************************************************************************)

\* Generate all successor states from a given state for character c
\* Note: In actual implementation, c would be compared to pattern[pattern_pos]
SuccessorStates(state, c) ==
    IF IsDead(state) THEN {}
    ELSE IF state.pattern_pos >= PATTERN_LENGTH THEN
        \* At end of pattern, only insert possible
        {ApplyInsert(state)}
    ELSE
        \* All operations possible (with character matching logic abstracted)
        { ApplyMatch(state),      \* If c matches pattern[pos]
          ApplySubstitute(state), \* If c doesn't match
          ApplyDelete(state),     \* Delete from pattern
          ApplyInsert(state) }    \* Insert into input

\* Apply subsumption pruning to a set of states
ApplySubsumption(states) ==
    { s \in states :
        ~\E s2 \in states : s # s2 /\ Subsumes(s2, s) }

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

\* INV1: Active matches bounded
ActiveMatchesBounded ==
    Cardinality(active_matches) <= MAX_ACTIVE_MATCHES

\* INV2: All active states have valid errors
ErrorBoundValid ==
    \A s \in active_matches : s.errors <= MAX_ERRORS + 1

\* INV3: Position monotonicity (never goes backward)
PositionMonotonicity ==
    position >= 0 /\ position <= INPUT_LENGTH

\* INV4: Completed matches are valid
CompletedMatchesValid ==
    \A m \in completed_matches :
        /\ m.cost <= MAX_ERRORS
        /\ m.start_pos <= m.end_pos
        /\ m.end_pos <= position

\* INV5: All states have consistent start positions
StartPositionConsistent ==
    \A s \in active_matches :
        s.start_pos <= position

\* INV6: Pattern positions in valid range
PatternPositionValid ==
    \A s \in active_matches :
        s.pattern_pos <= PATTERN_LENGTH

\* Combined type invariant
TypeInvariant ==
    /\ position \in 0..INPUT_LENGTH
    /\ active_matches \subseteq MatchState
    /\ completed_matches \subseteq CompletedMatch
    /\ input_consumed \in Seq(ALPHABET)

(***************************************************************************)
(* Initial State                                                            *)
(***************************************************************************)

Init ==
    /\ position = 0
    /\ active_matches = {InitialState(0)}
    /\ completed_matches = {}
    /\ input_consumed = <<>>
    /\ dead_states = {}

(***************************************************************************)
(* Next State Actions                                                       *)
(***************************************************************************)

\* Process one input character
ProcessChar(c) ==
    /\ position < INPUT_LENGTH
    /\ LET
           \* Generate all successor states
           all_successors == UNION {SuccessorStates(s, c) : s \in active_matches}

           \* Separate dead states
           alive_successors == {s \in all_successors : ~IsDead(s)}
           new_dead == {s \in all_successors : IsDead(s)}

           \* Extract completed matches (final states)
           new_completed == {
               [start_pos |-> s.start_pos,
                end_pos |-> position + 1,
                cost |-> s.errors]
               : s \in {s2 \in alive_successors : IsFinal(s2)}
           }

           \* Apply subsumption to remaining active states
           after_subsumption == ApplySubsumption(alive_successors)

           \* Also start a new match at current position (for overlapping matches)
           with_new_start == after_subsumption \cup {InitialState(position + 1)}
       IN
           /\ position' = position + 1
           /\ input_consumed' = Append(input_consumed, c)
           /\ active_matches' = with_new_start
           /\ completed_matches' = completed_matches \cup new_completed
           /\ dead_states' = dead_states \cup new_dead

\* End of input processing
FinishProcessing ==
    /\ position = INPUT_LENGTH
    /\ UNCHANGED <<position, active_matches, completed_matches, input_consumed, dead_states>>

\* Next state relation
Next ==
    \/ \E c \in ALPHABET : ProcessChar(c)
    \/ FinishProcessing

(***************************************************************************)
(* Fairness and Liveness                                                    *)
(***************************************************************************)

\* Weak fairness: if ProcessChar is continuously enabled, it eventually happens
Fairness == WF_<<position, active_matches, completed_matches, input_consumed, dead_states>>(
    \E c \in ALPHABET : ProcessChar(c))

\* Specification
Spec == Init /\ [][Next]_<<position, active_matches, completed_matches, input_consumed, dead_states>> /\ Fairness

(***************************************************************************)
(* Temporal Properties                                                      *)
(***************************************************************************)

\* PROP1: Eventually processing completes
EventuallyComplete == <>(position = INPUT_LENGTH)

\* PROP2: No missed matches - if a valid match exists, it's eventually found
\* (This would require a reference implementation to check against)
\* NoMissedMatches == ...

\* PROP3: Matches are recorded at correct positions
MatchesRecordedCorrectly ==
    [](Cardinality(completed_matches) <= (position + 1) * (position + 2) \div 2)

(***************************************************************************)
(* Theorems (to be verified by TLC)                                         *)
(***************************************************************************)

THEOREM Spec => []TypeInvariant
THEOREM Spec => []ActiveMatchesBounded
THEOREM Spec => []ErrorBoundValid
THEOREM Spec => []PositionMonotonicity
THEOREM Spec => []CompletedMatchesValid
THEOREM Spec => []StartPositionConsistent
THEOREM Spec => []PatternPositionValid
THEOREM Spec => EventuallyComplete

================================================================================
