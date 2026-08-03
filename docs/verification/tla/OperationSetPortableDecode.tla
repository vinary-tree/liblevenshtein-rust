-------------------- MODULE OperationSetPortableDecode --------------------
EXTENDS Naturals, TLC

(***************************************************************************)
(* Finite lifecycle model for bincode/protobuf OperationSet admission and  *)
(* the optional single-member gzip wrapper. Protobuf allocation is modeled *)
(* as a distinct phase reachable only after known repeated/string resources *)
(* pass the non-allocating wire preflight.                                  *)
(***************************************************************************)

CONSTANTS MaxCompressed, MaxDecompressed, MaxOperations, MaxPairs

VARIABLES phase, format, compressed, checksumOk,
          compressedBytes, consumedCompressedBytes, suppliedCompressedBytes,
          decompressedBytes, wireOk, supportedFormat,
          operations, pairs, semanticOk

vars == <<phase, format, compressed, checksumOk,
          compressedBytes, consumedCompressedBytes, suppliedCompressedBytes,
          decompressedBytes, wireOk, supportedFormat,
          operations, pairs, semanticOk>>

Init ==
  /\ phase = "Outer"
  /\ format \in {"Bincode", "Protobuf"}
  /\ compressed \in BOOLEAN
  /\ checksumOk \in BOOLEAN
  /\ compressedBytes \in 0..(MaxCompressed + 1)
  /\ consumedCompressedBytes \in 0..(MaxCompressed + 1)
  /\ suppliedCompressedBytes \in 0..(MaxCompressed + 1)
  /\ decompressedBytes \in 0..(MaxDecompressed + 1)
  /\ wireOk \in BOOLEAN
  /\ supportedFormat \in BOOLEAN
  /\ operations \in 0..(MaxOperations + 1)
  /\ pairs \in 0..(MaxPairs + 1)
  /\ semanticOk \in BOOLEAN

CheckOuter ==
  /\ phase = "Outer"
  /\ phase' =
       IF ~compressed \/
          (checksumOk
           /\ compressedBytes <= MaxCompressed
           /\ consumedCompressedBytes = suppliedCompressedBytes
           /\ decompressedBytes <= MaxDecompressed)
       THEN "Preflight"
       ELSE "Rejected"
  /\ UNCHANGED <<format, compressed, checksumOk,
                 compressedBytes, consumedCompressedBytes,
                 suppliedCompressedBytes, decompressedBytes,
                 wireOk, supportedFormat, operations, pairs, semanticOk>>

Preflight ==
  /\ phase = "Preflight"
  /\ phase' =
       IF decompressedBytes <= MaxDecompressed
          /\ wireOk
          /\ (format = "Bincode" \/ supportedFormat)
          /\ operations <= MaxOperations
          /\ pairs <= MaxPairs
       THEN "Allocate"
       ELSE "Rejected"
  /\ UNCHANGED <<format, compressed, checksumOk,
                 compressedBytes, consumedCompressedBytes,
                 suppliedCompressedBytes, decompressedBytes,
                 wireOk, supportedFormat, operations, pairs, semanticOk>>

Allocate ==
  /\ phase = "Allocate"
  /\ phase' = "Validate"
  /\ UNCHANGED <<format, compressed, checksumOk,
                 compressedBytes, consumedCompressedBytes,
                 suppliedCompressedBytes, decompressedBytes,
                 wireOk, supportedFormat, operations, pairs, semanticOk>>

Validate ==
  /\ phase = "Validate"
  /\ phase' = IF semanticOk THEN "Done" ELSE "Rejected"
  /\ UNCHANGED <<format, compressed, checksumOk,
                 compressedBytes, consumedCompressedBytes,
                 suppliedCompressedBytes, decompressedBytes,
                 wireOk, supportedFormat, operations, pairs, semanticOk>>

Next == CheckOuter \/ Preflight \/ Allocate \/ Validate
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in {"Outer", "Preflight", "Allocate", "Validate", "Done", "Rejected"}
  /\ format \in {"Bincode", "Protobuf"}
  /\ compressed \in BOOLEAN
  /\ checksumOk \in BOOLEAN
  /\ compressedBytes \in Nat
  /\ consumedCompressedBytes \in Nat
  /\ suppliedCompressedBytes \in Nat
  /\ decompressedBytes \in Nat
  /\ wireOk \in BOOLEAN
  /\ supportedFormat \in BOOLEAN
  /\ operations \in Nat
  /\ pairs \in Nat
  /\ semanticOk \in BOOLEAN

AllocationOnlyAfterPreflightBounds ==
  phase \in {"Allocate", "Validate", "Done"} =>
    /\ decompressedBytes <= MaxDecompressed
    /\ operations <= MaxOperations
    /\ pairs <= MaxPairs
    /\ wireOk
    /\ (format = "Bincode" \/ supportedFormat)

AcceptedOnlyAfterSemanticValidation ==
  phase = "Done" => semanticOk

CompressedAcceptanceIsSingleBoundedMember ==
  phase = "Done" /\ compressed =>
    /\ checksumOk
    /\ compressedBytes <= MaxCompressed
    /\ decompressedBytes <= MaxDecompressed
    /\ consumedCompressedBytes = suppliedCompressedBytes

UnsupportedProtobufNeverAccepted ==
  format = "Protobuf" /\ ~supportedFormat => phase # "Done"

=============================================================================
