------------------------ MODULE PhoneticStreamingBoundary ------------------------
(***************************************************************************)
(* Native phonetic online scanner and rewrite-transducer lifecycle.        *)
(* Abstracts matched text/rewrites; Rust/Julia partition properties compare *)
(* exact content with the native whole-input implementations.              *)
(***************************************************************************)
EXTENDS Integers, TLC

CONSTANT MaxBytes
ASSUME MaxBytes = 3

VARIABLES kind, phase, used, lastOp, lastStatus, outputWritten
vars == <<kind, phase, used, lastOp, lastStatus, outputWritten>>

Init ==
  /\ kind \in {"scanner", "rewrite"}
  /\ phase = "open"
  /\ used = 0
  /\ lastOp = "none"
  /\ lastStatus = "none"
  /\ outputWritten = FALSE

FeedOk ==
  /\ phase = "open"
  /\ \E amount \in 0..MaxBytes :
       /\ used + amount <= MaxBytes
       /\ used' = used + amount
  /\ lastOp' = "feed"
  /\ lastStatus' = "ok"
  /\ outputWritten' \in BOOLEAN
  /\ (kind = "scanner" => ~outputWritten')
  /\ UNCHANGED <<kind, phase>>

FeedRejected ==
  /\ lastOp' = "feed"
  /\ lastStatus' = "rejected"
  /\ outputWritten' = FALSE
  /\ UNCHANGED <<kind, phase, used>>

\* A scanner consumes its stream even if the result ceiling is too small.
ScannerFinishOk ==
  /\ kind = "scanner"
  /\ phase = "open"
  /\ phase' = "finished"
  /\ lastOp' = "finish"
  /\ lastStatus' = "ok"
  /\ outputWritten' = TRUE
  /\ UNCHANGED <<kind, used>>

ScannerFinishLimit ==
  /\ kind = "scanner"
  /\ phase = "open"
  /\ phase' = "finished"
  /\ lastOp' = "finish"
  /\ lastStatus' = "limit"
  /\ outputWritten' = FALSE
  /\ UNCHANGED <<kind, used>>

RewriteFinishOk ==
  /\ kind = "rewrite"
  /\ phase = "open"
  /\ phase' = "finished"
  /\ lastOp' = "finish"
  /\ lastStatus' = "ok"
  /\ outputWritten' = TRUE
  /\ UNCHANGED <<kind, used>>

\* Native rewrite output overflow resets buffered context, not its handle.
RewriteOverflow ==
  /\ kind = "rewrite"
  /\ phase = "open"
  /\ used' = 0
  /\ lastOp' \in {"feed", "finish"}
  /\ lastStatus' = "limit"
  /\ outputWritten' = FALSE
  /\ UNCHANGED <<kind, phase>>

Reset ==
  /\ kind = "rewrite"
  /\ phase' = "open"
  /\ used' = 0
  /\ lastOp' = "reset"
  /\ lastStatus' = "ok"
  /\ outputWritten' = FALSE
  /\ UNCHANGED kind

Next == FeedOk \/ FeedRejected \/ ScannerFinishOk \/ ScannerFinishLimit
     \/ RewriteFinishOk \/ RewriteOverflow \/ Reset
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ kind \in {"scanner", "rewrite"}
  /\ phase \in {"open", "finished"}
  /\ used \in 0..MaxBytes
  /\ lastOp \in {"none", "feed", "finish", "reset"}
  /\ lastStatus \in {"none", "ok", "rejected", "limit"}
  /\ outputWritten \in BOOLEAN

AcceptedFeedWithinBound == used <= MaxBytes
RejectedFeedNoOutput ==
  (lastOp = "feed" /\ lastStatus = "rejected") => ~outputWritten
ScannerLimitConsumes ==
  (kind = "scanner" /\ lastOp = "finish" /\ lastStatus = "limit")
    => (phase = "finished" /\ ~outputWritten)
RewriteLimitResets ==
  (kind = "rewrite" /\ lastStatus = "limit")
    => (phase = "open" /\ used = 0 /\ ~outputWritten)

=============================================================================
