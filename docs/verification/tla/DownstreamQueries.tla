------------------------- MODULE DownstreamQueries -------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

(***************************************************************************)
(* Exact finite model of the balanced DFS visitor and subsequence progress. *)
(* A frame records structural finality separately from terminal membership,  *)
(* matching PrefixPruner::permits_accept.                                    *)
(***************************************************************************)

CONSTANTS MaxDepth, QueryLength, MaxSteps

Frame == [matchesQuery: BOOLEAN, structuralFinal: BOOLEAN, terminalMember: BOOLEAN]

VARIABLES path, matchedStack, enters, leaves, accepted, steps
vars == <<path, matchedStack, enters, leaves, accepted, steps>>

Matched == matchedStack[Len(matchedStack)]

Init ==
  /\ path = <<>>
  /\ matchedStack = <<0>>
  /\ enters = 0
  /\ leaves = 0
  /\ accepted = {}
  /\ steps = 0

Enter(frame) ==
  /\ frame \in Frame
  /\ Len(path) < MaxDepth
  /\ path' = Append(path, frame)
  /\ matchedStack' = Append(
       matchedStack,
       Matched + IF frame.matchesQuery /\ Matched < QueryLength THEN 1 ELSE 0)
  /\ enters' = enters + 1
  /\ steps' = steps + 1
  /\ UNCHANGED <<leaves, accepted>>

Accept ==
  /\ Len(path) > 0
  /\ path[Len(path)].structuralFinal
  /\ path[Len(path)].terminalMember
  /\ Matched = QueryLength
  /\ accepted' = accepted \cup {[path |-> path, matched |-> Matched]}
  /\ steps' = steps + 1
  /\ UNCHANGED <<path, matchedStack, enters, leaves>>

Leave ==
  /\ Len(path) > 0
  /\ path' = SubSeq(path, 1, Len(path) - 1)
  /\ matchedStack' = SubSeq(matchedStack, 1, Len(matchedStack) - 1)
  /\ leaves' = leaves + 1
  /\ steps' = steps + 1
  /\ UNCHANGED <<enters, accepted>>

Next == steps < MaxSteps /\ ((\E frame \in Frame : Enter(frame)) \/ Accept \/ Leave)
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ path \in Seq(Frame)
  /\ matchedStack \in Seq(0..QueryLength)
  /\ enters \in Nat
  /\ leaves \in Nat
  /\ steps \in 0..MaxSteps
  /\ accepted \subseteq [path: Seq(Frame), matched: 0..QueryLength]

StackTracksPath == Len(matchedStack) = Len(path) + 1
VisitorBalance == enters = leaves + Len(path)
SubsequenceProgress == Matched <= Len(path)
CompletedTraversalBalanced == Len(path) = 0 => enters = leaves

AcceptedOnlyExactTerminals ==
  \A candidate \in accepted:
    /\ Len(candidate.path) > 0
    /\ candidate.path[Len(candidate.path)].structuralFinal
    /\ candidate.path[Len(candidate.path)].terminalMember
    /\ candidate.matched = QueryLength

=============================================================================
