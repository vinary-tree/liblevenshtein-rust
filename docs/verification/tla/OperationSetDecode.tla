------------------------ MODULE OperationSetDecode ------------------------
EXTENDS Naturals, FiniteSets, TLC

(***************************************************************************)
(* Finite lifecycle model of the versioned OperationSet binary decoder.    *)
(* Header fields and input sizes are nondeterministically selected in Init. *)
(* The only accepting phase requires exact input consumption, version 1,   *)
(* zero flags, semantic validation, and both modeled resource ceilings.     *)
(***************************************************************************)

CONSTANTS MaxPayload, MaxOperations

VARIABLES phase, magicOk, version, flags, declared, available,
          consumed, operations, semanticOk

vars == <<phase, magicOk, version, flags, declared, available,
          consumed, operations, semanticOk>>

Init ==
  /\ phase = "Header"
  /\ magicOk \in BOOLEAN
  /\ version \in 0..2
  /\ flags \in 0..1
  /\ declared \in 0..(MaxPayload + 1)
  /\ available \in 0..(MaxPayload + 1)
  /\ consumed = 0
  /\ operations \in 0..(MaxOperations + 1)
  /\ semanticOk \in BOOLEAN

ReadHeader ==
  /\ phase = "Header"
  /\ phase' = IF magicOk /\ version = 1 /\ flags = 0
                    /\ declared <= MaxPayload
                THEN "Payload"
                ELSE "Rejected"
  /\ UNCHANGED <<magicOk, version, flags, declared, available,
                 consumed, operations, semanticOk>>

ConsumePayload ==
  /\ phase = "Payload"
  /\ consumed < declared
  /\ consumed < available
  /\ consumed' = consumed + 1
  /\ UNCHANGED <<phase, magicOk, version, flags, declared, available,
                 operations, semanticOk>>

FinishPayload ==
  /\ phase = "Payload"
  /\ consumed = declared
  /\ phase' = "Validate"
  /\ UNCHANGED <<magicOk, version, flags, declared, available,
                 consumed, operations, semanticOk>>

RejectTruncatedPayload ==
  /\ phase = "Payload"
  /\ consumed < declared
  /\ consumed >= available
  /\ phase' = "Rejected"
  /\ UNCHANGED <<magicOk, version, flags, declared, available,
                 consumed, operations, semanticOk>>

Validate ==
  /\ phase = "Validate"
  /\ phase' = IF available = declared
                    /\ consumed = declared
                    /\ operations <= MaxOperations
                    /\ semanticOk
                THEN "Done"
                ELSE "Rejected"
  /\ UNCHANGED <<magicOk, version, flags, declared, available,
                 consumed, operations, semanticOk>>

Next == ReadHeader \/ ConsumePayload \/ FinishPayload
        \/ RejectTruncatedPayload \/ Validate

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in {"Header", "Payload", "Validate", "Done", "Rejected"}
  /\ magicOk \in BOOLEAN
  /\ version \in Nat
  /\ flags \in Nat
  /\ declared \in Nat
  /\ available \in Nat
  /\ consumed \in Nat
  /\ operations \in Nat
  /\ semanticOk \in BOOLEAN

ConsumptionNeverExceedsDeclaration == consumed <= declared

AcceptedOnlyAfterStrictValidation ==
  phase = "Done" =>
    /\ magicOk
    /\ version = 1
    /\ flags = 0
    /\ declared = available
    /\ consumed = declared
    /\ declared <= MaxPayload
    /\ operations <= MaxOperations
    /\ semanticOk

TrailingBytesNeverAccepted ==
  available > declared => phase # "Done"

TruncatedPayloadNeverAccepted ==
  available < declared => phase # "Done"

=============================================================================
