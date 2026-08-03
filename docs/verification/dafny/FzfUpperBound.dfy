// Capacity-sensitive model of the FuzzyMatchV2 recurrence bound.

function Max(left: int, right: int): int {
  if left >= right then left else right
}

function Feasible(feasible: bool, completed: int, term: int): int {
  if feasible then term else completed
}

function Bound(completed: int, unstarted: int, active: int,
               queryLen: int, activeRemaining: int,
               capacity: int, beta: int): int {
  Max(completed,
      Max(Feasible(queryLen <= capacity, completed, unstarted),
          Feasible(activeRemaining <= capacity, completed,
                   active + activeRemaining * beta)))
}

lemma CompletedProjection(completed: int, unstarted: int, active: int,
                          queryLen: int, activeRemaining: int,
                          capacity: int, beta: int)
  ensures completed <= Bound(completed, unstarted, active,
                             queryLen, activeRemaining, capacity, beta)
{}

lemma GapProjection(completed: int, unstarted: int, active: int, child: int,
                    queryLen: int, activeRemaining: int,
                    capacity: int, beta: int)
  requires 0 <= beta
  requires activeRemaining <= capacity
  requires child <= active
  ensures child + activeRemaining * beta <=
          Bound(completed, unstarted, active,
                queryLen, activeRemaining, capacity, beta)
{}

lemma MatchProjection(completed: int, unstarted: int, active: int, child: int,
                      queryLen: int, childRemaining: int,
                      capacity: int, beta: int)
  requires 0 <= beta
  requires childRemaining + 1 <= capacity
  requires child <= active + beta
  ensures child + childRemaining * beta <=
          Bound(completed, unstarted, active,
                queryLen, childRemaining + 1, capacity, beta)
{}

lemma NewlyStartedProjection(completed: int, unstarted: int, active: int,
                             childProjection: int, queryLen: int,
                             activeRemaining: int, capacity: int, beta: int)
  requires queryLen <= capacity
  requires childProjection <= unstarted
  ensures childProjection <= Bound(completed, unstarted, active,
                                   queryLen, activeRemaining, capacity, beta)
{}

lemma PruneSound(score: int, upper: int, cutoff: int)
  requires score <= upper
  requires upper < cutoff
  ensures score < cutoff
{}

lemma ArcticDeltasTelescope(initial: int, middle: int, finalScore: int)
  ensures initial + (middle - initial) + (finalScore - middle) == finalScore
{}
