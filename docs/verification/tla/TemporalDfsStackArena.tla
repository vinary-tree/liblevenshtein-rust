------------------------ MODULE TemporalDfsStackArena ------------------------
(***************************************************************************)
(* Live-path stack arena for a bounded, iterative dictionary x temporal     *)
(* automaton product. A prospective child is admitted only if its retained  *)
(* columns fit. Frame and state arrays are pushed and popped together.       *)
(***************************************************************************)

EXTENDS Integers, Sequences, TLC

CONSTANTS MaxDepth, ColumnCells, CellBytes, MaxScratch

VARIABLES frames, states, status, rejectedFrames, rejectedStates
vars == <<frames, states, status, rejectedFrames, rejectedStates>>

RetainedBytes == Len(states) * ColumnCells * CellBytes
ProspectiveBytes == (Len(states) + 1) * ColumnCells * CellBytes

Init ==
  /\ frames = <<0>>
  /\ states = <<0>>
  /\ status = "Running"
  /\ rejectedFrames = <<>>
  /\ rejectedStates = <<>>

Push ==
  /\ status = "Running"
  /\ Len(frames) < MaxDepth
  /\ ProspectiveBytes <= MaxScratch
  /\ frames' = Append(frames, Len(frames))
  /\ states' = Append(states, Len(states))
  /\ UNCHANGED <<status, rejectedFrames, rejectedStates>>

Pop ==
  /\ status = "Running"
  /\ Len(frames) > 0
  /\ frames' = SubSeq(frames, 1, Len(frames) - 1)
  /\ states' = SubSeq(states, 1, Len(states) - 1)
  /\ UNCHANGED <<status, rejectedFrames, rejectedStates>>

RejectPush ==
  /\ status = "Running"
  /\ Len(frames) < MaxDepth
  /\ ProspectiveBytes > MaxScratch
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

Next == Push \/ Pop \/ RejectPush \/ Pause \/ Resume \/ Finish \/ Stutter
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ frames \in Seq(0..MaxDepth)
  /\ states \in Seq(0..MaxDepth)
  /\ rejectedFrames \in Seq(0..MaxDepth)
  /\ rejectedStates \in Seq(0..MaxDepth)
  /\ status \in {"Running", "Paused", "Complete", "Incomplete"}

PairedCardinality == Len(frames) = Len(states)
PairedIds == \A index \in 1..Len(frames) : frames[index] = states[index]
DepthBound == Len(frames) <= MaxDepth
ScratchBound == RetainedBytes <= MaxScratch
CompleteOnlyAfterExhaustion == status = "Complete" => Len(frames) = 0
RejectedPushIsAtomic ==
  status = "Incomplete" =>
    frames = rejectedFrames /\ states = rejectedStates
PausedRetainsProduct ==
  status = "Paused" => Len(frames) = Len(states)

THEOREM Spec => []TypeOK
THEOREM Spec => []PairedCardinality
THEOREM Spec => []PairedIds
THEOREM Spec => []DepthBound
THEOREM Spec => []ScratchBound
THEOREM Spec => []CompleteOnlyAfterExhaustion
THEOREM Spec => []RejectedPushIsAtomic
THEOREM Spec => []PausedRetainsProduct

=============================================================================
