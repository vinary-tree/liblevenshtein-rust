------------------------ MODULE TemporalDfsStackArena ------------------------
(***************************************************************************)
(* Bounded iterative DFS plus a query-local exact residual interner.        *)
(* Frames contain compact state identifiers. Reusing a canonical state      *)
(* pushes only a frame; committing a fresh state appends it transactionally. *)
(* Popping a frame never invalidates arena identifiers.                      *)
(***************************************************************************)

EXTENDS Integers, Sequences, TLC

CONSTANTS MaxDepth, MaxStates, ColumnCells, CellBytes, MaxScratch

VARIABLES frames, states, status, rejectedFrames, rejectedStates
vars == <<frames, states, status, rejectedFrames, rejectedStates>>

RetainedBytes == Len(states) * ColumnCells * CellBytes
ProspectiveFreshBytes == (Len(states) + 1) * ColumnCells * CellBytes
FrameIdsValid ==
  \A index \in 1..Len(frames) :
    frames[index] >= 0 /\ frames[index] < Len(states)

Init ==
  /\ frames = <<0>>
  /\ states = <<0>>
  /\ status = "Running"
  /\ rejectedFrames = <<>>
  /\ rejectedStates = <<>>

PushReused ==
  /\ status = "Running"
  /\ Len(frames) < MaxDepth
  /\ \E stateId \in 0..(Len(states) - 1) :
      /\ frames' = Append(frames, stateId)
      /\ UNCHANGED states
  /\ UNCHANGED <<status, rejectedFrames, rejectedStates>>

PushFresh ==
  /\ status = "Running"
  /\ Len(frames) < MaxDepth
  /\ Len(states) < MaxStates
  /\ ProspectiveFreshBytes <= MaxScratch
  /\ states' = Append(states, Len(states))
  /\ frames' = Append(frames, Len(states))
  /\ UNCHANGED <<status, rejectedFrames, rejectedStates>>

PopFrame ==
  /\ status = "Running"
  /\ Len(frames) > 0
  /\ frames' = SubSeq(frames, 1, Len(frames) - 1)
  /\ UNCHANGED <<states, status, rejectedFrames, rejectedStates>>

RejectFresh ==
  /\ status = "Running"
  /\ Len(frames) < MaxDepth
  /\ (Len(states) >= MaxStates \/ ProspectiveFreshBytes > MaxScratch)
  /\ status' = "Incomplete"
  /\ rejectedFrames' = frames
  /\ rejectedStates' = states
  /\ UNCHANGED <<frames, states>>

Pause ==
  /\ status = "Running"
  /\ status' = "Paused"
  /\ UNCHANGED <<frames, states, rejectedFrames, rejectedStates>>

Resume ==
  /\ status = "Paused"
  /\ status' = "Running"
  /\ UNCHANGED <<frames, states, rejectedFrames, rejectedStates>>

Finish ==
  /\ status = "Running"
  /\ Len(frames) = 0
  /\ status' = "Complete"
  /\ UNCHANGED <<frames, states, rejectedFrames, rejectedStates>>

Stutter ==
  /\ status \in {"Complete", "Incomplete"}
  /\ UNCHANGED vars

Next == PushReused \/ PushFresh \/ PopFrame \/ RejectFresh \/
        Pause \/ Resume \/ Finish \/ Stutter
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ frames \in Seq(0..MaxStates)
  /\ states \in Seq(0..MaxStates)
  /\ rejectedFrames \in Seq(0..MaxStates)
  /\ rejectedStates \in Seq(0..MaxStates)
  /\ status \in {"Running", "Paused", "Complete", "Incomplete"}

DepthBound == Len(frames) <= MaxDepth
StateBound == Len(states) <= MaxStates
ScratchBound == RetainedBytes <= MaxScratch
CompleteOnlyAfterExhaustion == status = "Complete" => Len(frames) = 0
RejectedFreshIsAtomic ==
  status = "Incomplete" =>
    frames = rejectedFrames /\ states = rejectedStates
PausedRetainsValidProduct == status = "Paused" => FrameIdsValid

THEOREM Spec => []TypeOK
THEOREM Spec => []FrameIdsValid
THEOREM Spec => []DepthBound
THEOREM Spec => []StateBound
THEOREM Spec => []ScratchBound
THEOREM Spec => []CompleteOnlyAfterExhaustion
THEOREM Spec => []RejectedFreshIsAtomic
THEOREM Spec => []PausedRetainsValidProduct

=============================================================================
