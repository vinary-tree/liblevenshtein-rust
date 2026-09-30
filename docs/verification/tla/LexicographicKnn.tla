------------------------ MODULE LexicographicKnn ------------------------
EXTENDS Naturals, FiniteSets

\* Finite executable instance of BF-1. Candidate names are deliberately
\* unrelated to tie rank, so scheduler encounter order cannot stand in for it.
Candidates == {"a", "b", "c", "d"}
Cost == [candidate \in Candidates |->
    CASE candidate = "d" -> 7 [] OTHER -> 5]
Tie == [candidate \in Candidates |->
    CASE candidate = "a" -> 4
      [] candidate = "b" -> 1
      [] candidate = "c" -> 3
      [] OTHER -> 2]
K == 2

Better(left, right) ==
    Cost[left] < Cost[right]
    \/ (Cost[left] = Cost[right] /\ Tie[left] < Tie[right])

TopK(candidates) ==
    {candidate \in candidates :
        Cardinality({other \in candidates : Better(other, candidate)}) < K}

WorstVerified(verified) ==
    CHOOSE candidate \in TopK(verified) :
        \A other \in TopK(verified) : ~Better(candidate, other)

CostValues(region) == {Cost[candidate] : candidate \in region}
TieValues(region) == {Tie[candidate] : candidate \in region}
CostFloor(region) ==
    CHOOSE value \in CostValues(region) :
        \A other \in CostValues(region) : value <= other
TieFloor(region) ==
    CHOOSE value \in TieValues(region) :
        \A other \in TieValues(region) : value <= other

LexFloorAtLeastWorst(region, verified, knownSummaries) ==
    LET worst == WorstVerified(verified) IN
        CostFloor(region) > Cost[worst]
        \/ (CostFloor(region) = Cost[worst]
            /\ region \in knownSummaries
            /\ TieFloor(region) >= Tie[worst])

VARIABLES pending, knownSummaries, verified, discarded, stopped
vars == <<pending, knownSummaries, verified, discarded, stopped>>

Init ==
    /\ pending = {Candidates, {}}
    /\ knownSummaries = {}
    /\ verified = {}
    /\ discarded = {}
    /\ stopped = FALSE

\* Splitting represents an arbitrary dictionary scheduling choice. Each
\* pending region continues to cover its candidates exactly once.
Split ==
    \E region \in pending :
        /\ Cardinality(region) > 1
        /\ \E candidate \in region :
            /\ pending' = (pending \ {region})
                \cup {{candidate}, region \ {candidate}}
            /\ knownSummaries' = knownSummaries \ {region}
            /\ UNCHANGED <<verified, discarded, stopped>>

\* Completing a structural summary certifies the exact tie floor. An
\* uncompleted or resource-limited probe leaves the region unknown.
ResolveSummary ==
    \E region \in pending \ {{}} :
        /\ region \notin knownSummaries
        /\ knownSummaries' = knownSummaries \cup {region}
        /\ UNCHANGED <<pending, verified, discarded, stopped>>

DropEmpty ==
    /\ {} \in pending
    /\ pending' = pending \ {{}}
    /\ knownSummaries' = knownSummaries \ {{}}
    /\ UNCHANGED <<verified, discarded, stopped>>

Verify ==
    \E region \in pending :
        /\ Cardinality(region) = 1
        /\ LET candidate == CHOOSE member \in region : TRUE IN
            /\ pending' = pending \ {region}
            /\ knownSummaries' = knownSummaries \ {region}
            /\ verified' = verified \cup {candidate}
            /\ UNCHANGED <<discarded, stopped>>

Prune ==
    \E region \in pending \ {{}} :
        /\ Cardinality(verified) >= K
        /\ LexFloorAtLeastWorst(region, verified, knownSummaries)
        /\ pending' = pending \ {region}
        /\ knownSummaries' = knownSummaries \ {region}
        /\ discarded' = discarded \cup region
        /\ UNCHANGED <<verified, stopped>>

Stop ==
    /\ pending = {}
    /\ stopped' = TRUE
    /\ UNCHANGED <<pending, knownSummaries, verified, discarded>>

Next == ~stopped /\ (Split \/ ResolveSummary \/ DropEmpty \/ Verify \/ Prune \/ Stop)
Spec == Init /\ [][Next]_vars

\* Negative control: an equality-cost region is discarded without a
\* completed tie floor. This is not an action of the certified Spec.
PruneCostOnlyAtEquality ==
    \E region \in pending \ {{}} :
        /\ Cardinality(verified) >= K
        /\ CostFloor(region) >= Cost[WorstVerified(verified)]
        /\ pending' = pending \ {region}
        /\ knownSummaries' = knownSummaries \ {region}
        /\ discarded' = discarded \cup region
        /\ UNCHANGED <<verified, stopped>>

UnsafeNext == ~stopped /\
    (Split \/ ResolveSummary \/ DropEmpty \/ Verify \/ PruneCostOnlyAtEquality \/ Stop)
UnsafeSpec == Init /\ [][UnsafeNext]_vars

KnownSummariesArePending == knownSummaries \subseteq pending
Coverage ==
    (UNION pending) \cup verified \cup discarded = Candidates
NoLostTopK == TopK(discarded) \cap TopK(Candidates) = {}
OrderedCompletion == stopped => TopK(verified) = TopK(Candidates)

=============================================================================
