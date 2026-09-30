---------------------- MODULE RangeCursorOwnership ----------------------
EXTENDS Naturals, Sequences, FiniteSets

\* Finite lifecycle abstraction for one exact range cursor. Two distinct
\* originals share a physical node. An original is owned by precisely one
\* of unseen cursor suffix, active scorer, pending result, emitted result,
\* or sound exclusion. This model does not establish Rust correspondence.
CONSTANT CountPrivate, SkipSharedMember, BypassPendingGuard

Originals == {"a", "b", "c"}
Bucket == <<"c", "a", "b">>
Eligible == {"a", "b"}
NodeOf == [original \in Originals |->
    IF original = "a" \/ original = "b" THEN "shared" ELSE "child"]
CapturedRevision == "revision-1"
NoOriginal == "none"

VARIABLES cursor, phase, active, pendingMatch, emitted, excluded,
          paused, completed, revision
vars == <<cursor, phase, active, pendingMatch, emitted, excluded,
          paused, completed, revision>>

Unseen == {Bucket[index] : index \in cursor..Len(Bucket)}
ActiveOwner == IF phase = "scoring" THEN {active} ELSE {}
PendingOwner == IF phase = "waiting" THEN {pendingMatch} ELSE {}
PrivateOwners == IF CountPrivate THEN ActiveOwner \cup PendingOwner ELSE {}
AllOwners == Unseen \cup PrivateOwners \cup emitted \cup excluded

Init ==
    /\ cursor = 1
    /\ phase = "idle"
    /\ active = NoOriginal
    /\ pendingMatch = NoOriginal
    /\ emitted = {}
    /\ excluded = {}
    /\ paused = FALSE
    /\ completed = FALSE
    /\ revision = CapturedRevision

Pick ==
    /\ ~paused /\ ~completed /\ phase = "idle"
    /\ cursor <= Len(Bucket)
    /\ active' = Bucket[cursor]
    /\ cursor' = cursor + 1
    /\ phase' = "scoring"
    /\ UNCHANGED <<pendingMatch, emitted, excluded, paused, completed, revision>>

Reject ==
    /\ ~paused /\ ~completed /\ phase = "scoring"
    /\ active \notin Eligible
    /\ excluded' = excluded \cup {active}
    /\ active' = NoOriginal
    /\ phase' = "idle"
    /\ UNCHANGED <<cursor, pendingMatch, emitted, paused, completed, revision>>

Accept ==
    /\ ~paused /\ ~completed /\ phase = "scoring"
    /\ active \in Eligible
    /\ pendingMatch' = active
    /\ active' = NoOriginal
    /\ phase' = "waiting"
    /\ UNCHANGED <<cursor, emitted, excluded, paused, completed, revision>>

Publish ==
    /\ ~paused /\ ~completed /\ phase = "waiting"
    /\ emitted' = emitted \cup {pendingMatch}
    /\ pendingMatch' = NoOriginal
    /\ phase' = "idle"
    /\ UNCHANGED <<cursor, active, excluded, paused, completed, revision>>

Pause ==
    /\ ~paused /\ ~completed
    /\ (phase = "idle" \/ phase = "waiting")
    /\ paused' = TRUE
    /\ UNCHANGED <<cursor, phase, active, pendingMatch,
                  emitted, excluded, completed, revision>>

Resume ==
    /\ paused /\ ~completed
    /\ paused' = FALSE
    /\ UNCHANGED <<cursor, phase, active, pendingMatch,
                  emitted, excluded, completed, revision>>

Finish ==
    /\ ~paused /\ ~completed
    /\ cursor > Len(Bucket)
    /\ (phase = "idle" \/ (BypassPendingGuard /\ phase = "waiting"))
    /\ completed' = TRUE
    /\ UNCHANGED <<cursor, phase, active, pendingMatch,
                  emitted, excluded, paused, revision>>

\* An invalid node-level deduplication skips the second original at the
\* shared physical node. It is enabled only in the dedicated mutant model.
SkipCollision ==
    /\ SkipSharedMember /\ ~paused /\ ~completed /\ phase = "idle"
    /\ cursor = 3 /\ NodeOf[Bucket[2]] = NodeOf[Bucket[3]]
    /\ cursor' = cursor + 1
    /\ UNCHANGED <<phase, active, pendingMatch, emitted,
                  excluded, paused, completed, revision>>

Next == Pick \/ Reject \/ Accept \/ Publish \/ Pause \/ Resume
        \/ Finish \/ SkipCollision
Spec == Init /\ [][Next]_vars
FairSpec == Spec /\ SF_vars(Pick) /\ SF_vars(Reject) /\ SF_vars(Accept)
    /\ SF_vars(Publish) /\ SF_vars(Resume) /\ SF_vars(Finish)
EventuallyCompleted == <>completed

TypeOK ==
    /\ cursor \in 1..(Len(Bucket) + 1)
    /\ phase \in {"idle", "scoring", "waiting"}
    /\ active \in Originals \cup {NoOriginal}
    /\ pendingMatch \in Originals \cup {NoOriginal}
    /\ emitted \subseteq Originals /\ excluded \subseteq Originals
    /\ paused \in BOOLEAN /\ completed \in BOOLEAN
    /\ revision = CapturedRevision

Coverage == AllOwners = Originals
UniqueOwnership ==
    Cardinality(Unseen) + Cardinality(PrivateOwners)
    + Cardinality(emitted) + Cardinality(excluded) = Cardinality(Originals)
SoundClassification == emitted \subseteq Eligible
    /\ excluded \cap Eligible = {}
PendingIsExclusive ==
    phase = "waiting" => pendingMatch \in Originals
        /\ pendingMatch \notin Unseen \cup emitted \cup excluded
NoFalseCompletion == completed => emitted = Eligible
    /\ phase = "idle" /\ cursor > Len(Bucket)

==========================================================================
