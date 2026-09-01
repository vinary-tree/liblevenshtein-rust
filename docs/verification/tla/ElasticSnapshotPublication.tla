--------------------- MODULE ElasticSnapshotPublication --------------------
EXTENDS FiniteSets, TLC

CONSTANT Identities

VARIABLES decodePhase, checksumMatches, staging, sealed, manifest
vars == <<decodePhase, checksumMatches, staging, sealed, manifest>>

Init ==
  /\ decodePhase = "Framed"
  /\ checksumMatches \in BOOLEAN
  /\ staging = {}
  /\ sealed = {}
  /\ manifest = "None"

VerifyChecksum ==
  /\ decodePhase = "Framed"
  /\ checksumMatches
  /\ decodePhase' = "ChecksumVerified"
  /\ UNCHANGED <<checksumMatches, staging, sealed, manifest>>

RejectChecksum ==
  /\ decodePhase = "Framed"
  /\ ~checksumMatches
  /\ decodePhase' = "Rejected"
  /\ UNCHANGED <<checksumMatches, staging, sealed, manifest>>

DecodeSemantics ==
  /\ decodePhase = "ChecksumVerified"
  /\ decodePhase' = "SemanticallyDecoded"
  /\ UNCHANGED <<checksumMatches, staging, sealed, manifest>>

VerifyBijection ==
  /\ decodePhase = "SemanticallyDecoded"
  /\ decodePhase' = "BijectionVerified"
  /\ UNCHANGED <<checksumMatches, staging, sealed, manifest>>

Accept ==
  /\ decodePhase = "BijectionVerified"
  /\ decodePhase' = "Accepted"
  /\ UNCHANGED <<checksumMatches, staging, sealed, manifest>>

BeginGeneration ==
  \E identity \in Identities :
    /\ identity \notin staging
    /\ identity \notin sealed
    /\ staging' = staging \union {identity}
    /\ UNCHANGED <<decodePhase, checksumMatches, sealed, manifest>>

SealGeneration ==
  \E identity \in staging :
    /\ staging' = staging \ {identity}
    /\ sealed' = sealed \union {identity}
    /\ UNCHANGED <<decodePhase, checksumMatches, manifest>>

PublishManifest ==
  \E identity \in sealed :
    /\ manifest' = identity
    /\ UNCHANGED <<decodePhase, checksumMatches, staging, sealed>>

Crash ==
  /\ staging' = {}
  /\ UNCHANGED <<decodePhase, checksumMatches, sealed, manifest>>

Stutter ==
  /\ decodePhase \in {"Rejected", "Accepted"}
  /\ UNCHANGED vars

Next == VerifyChecksum \/ RejectChecksum \/ DecodeSemantics \/
        VerifyBijection \/ Accept \/ BeginGeneration \/
        SealGeneration \/ PublishManifest \/ Crash \/ Stutter

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ decodePhase \in {"Framed", "ChecksumVerified", "Rejected",
                       "SemanticallyDecoded", "BijectionVerified", "Accepted"}
  /\ checksumMatches \in BOOLEAN
  /\ staging \subseteq Identities
  /\ sealed \subseteq Identities
  /\ manifest \in Identities \union {"None"}

ManifestNamesSealedGeneration == manifest # "None" => manifest \in sealed

SemanticPhaseRequiresChecksum ==
  decodePhase \in {"SemanticallyDecoded", "BijectionVerified", "Accepted"}
    => checksumMatches

RejectedNeverBecomesAccepted == decodePhase = "Rejected" => decodePhase # "Accepted"

THEOREM Spec => []TypeOK
THEOREM Spec => []ManifestNamesSealedGeneration
THEOREM Spec => []SemanticPhaseRequiresChecksum
THEOREM Spec => []RejectedNeverBecomesAccepted
=============================================================================
