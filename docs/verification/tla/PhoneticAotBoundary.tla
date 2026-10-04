--------------------------- MODULE PhoneticAotBoundary ---------------------------
(***************************************************************************)
(* Finite ownership/transaction model for revision-9 phonetic AOT calls.  *)
(* The model deliberately abstracts binary contents: Rust format parsers   *)
(* remain authoritative for magic, version, and semantic decoding.        *)
(* Rust correspondence: tests/ffi_phonetic_files_aot.rs.                  *)
(***************************************************************************)
EXTENDS Naturals, TLC

VARIABLES enabled, bytes, decoded, everEncoded, lastStatus, lastWrite, freeCount
vars == <<enabled, bytes, decoded, everEncoded, lastStatus, lastWrite, freeCount>>

Init ==
  /\ enabled \in BOOLEAN
  /\ bytes = "empty"
  /\ decoded = FALSE
  /\ everEncoded = FALSE
  /\ lastStatus = "none"
  /\ lastWrite = FALSE
  /\ freeCount = 0

SerializeOk ==
  /\ enabled
  /\ bytes = "empty"
  /\ bytes' = "owned"
  /\ everEncoded' = TRUE
  /\ lastStatus' = "ok"
  /\ lastWrite' = TRUE
  /\ UNCHANGED <<enabled, decoded, freeCount>>

\* Covers disabled builds, invalid handles, and rejected byte ceilings.
SerializeReject ==
  /\ lastStatus' = "rejected"
  /\ lastWrite' = FALSE
  /\ UNCHANGED <<enabled, bytes, decoded, everEncoded, freeCount>>

DecodeOk ==
  /\ enabled
  /\ bytes = "owned"
  /\ ~decoded
  /\ decoded' = TRUE
  /\ lastStatus' = "ok"
  /\ lastWrite' = TRUE
  /\ UNCHANGED <<enabled, bytes, everEncoded, freeCount>>

\* Covers bad magic/version, malformed input, and rejected byte ceilings.
DecodeReject ==
  /\ lastStatus' = "rejected"
  /\ lastWrite' = FALSE
  /\ UNCHANGED <<enabled, bytes, decoded, everEncoded, freeCount>>

FreeOwned ==
  /\ bytes = "owned"
  /\ bytes' = "freed"
  /\ freeCount' = freeCount + 1
  /\ lastStatus' = "ok"
  /\ lastWrite' = TRUE
  /\ UNCHANGED <<enabled, decoded, everEncoded>>

FreeAgain ==
  /\ bytes \in {"empty", "freed"}
  /\ lastStatus' = "ok"
  /\ lastWrite' = FALSE
  /\ UNCHANGED <<enabled, bytes, decoded, everEncoded, freeCount>>

Next == SerializeOk \/ SerializeReject \/ DecodeOk \/ DecodeReject \/ FreeOwned \/ FreeAgain
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ enabled \in BOOLEAN
  /\ bytes \in {"empty", "owned", "freed"}
  /\ decoded \in BOOLEAN
  /\ everEncoded \in BOOLEAN
  /\ lastStatus \in {"none", "ok", "rejected"}
  /\ lastWrite \in BOOLEAN
  /\ freeCount \in 0..1

RejectPreservesOutputs == lastStatus = "rejected" => ~lastWrite
NoDoubleFree == freeCount <= 1
DecodeRequiresValidOwnedBytes == decoded => (enabled /\ everEncoded)
DisabledBuildNeverOwnsAot == ~enabled => (bytes = "empty" /\ ~decoded)
FreedBytesHaveOneRelease == bytes = "freed" => freeCount = 1

=============================================================================
