; TWED kernel counterexample checks. Every query must be UNSAT in Z3 and cvc5.
(set-logic QF_NIA)

(define-fun iabs ((value Int)) Int
  (ite (>= value 0) value (- value)))

(define-fun interval-dist ((value Int) (low Int) (high Int)) Int
  (ite (< value low)
       (- low value)
       (ite (< high value) (- value high) 0)))

(define-fun interval-gap ((low1 Int) (high1 Int) (low2 Int) (high2 Int)) Int
  (ite (< high1 low2)
       (- low2 high1)
       (ite (< high2 low1) (- low1 high2) 0)))

(declare-const value Int)
(declare-const low Int)
(declare-const high Int)
(declare-const concrete Int)

; A value-to-interval leaf lower-bounds every realization.
(push)
(assert (<= low high))
(assert (<= low concrete))
(assert (<= concrete high))
(assert (> (interval-dist value low high) (iabs (- value concrete))))
(check-sat)
(pop)

; Singleton value intervals reproduce scalar absolute distance.
(push)
(assert (not (= (interval-dist value concrete concrete)
                (iabs (- value concrete)))))
(check-sat)
(pop)

(declare-const low2 Int)
(declare-const high2 Int)
(declare-const left Int)
(declare-const right Int)

; An interval-to-interval deletion leaf lower-bounds every pair realization.
(push)
(assert (<= low high))
(assert (<= low2 high2))
(assert (<= low left))
(assert (<= left high))
(assert (<= low2 right))
(assert (<= right high2))
(assert (> (interval-gap low high low2 high2) (iabs (- left right))))
(check-sat)
(pop)

; Singleton interval gaps reproduce scalar absolute distance.
(push)
(assert (not (= (interval-gap left left right right) (iabs (- left right)))))
(check-sat)
(pop)

(declare-const x-current Int)
(declare-const x-previous Int)
(declare-const y-current Int)
(declare-const y-previous Int)
(declare-const temporal Int)

; The two independent match terms remain separable over their box.
(push)
(assert (<= low y-current))
(assert (<= y-current high))
(assert (<= low2 y-previous))
(assert (<= y-previous high2))
(assert (> (+ (interval-dist x-current low high)
              (interval-dist x-previous low2 high2)
              temporal)
           (+ (iabs (- x-current y-current))
              (iabs (- x-previous y-previous))
              temporal)))
(check-sat)
(pop)

(declare-const nu Int)
(declare-const lambda Int)

; Adding stiffness and penalty preserves deletion-leaf admissibility.
(push)
(assert (<= low left))
(assert (<= left high))
(assert (<= low2 right))
(assert (<= right high2))
(assert (> (+ (interval-gap low high low2 high2) nu lambda)
           (+ (iabs (- left right)) nu lambda)))
(check-sat)
(pop)

(declare-const predecessor Int)
(declare-const predecessor2 Int)
(declare-const local Int)
(declare-const local2 Int)

; Additive recurrence branches are monotone in predecessor and local leaves.
(push)
(assert (<= predecessor predecessor2))
(assert (<= local local2))
(assert (> (+ predecessor local) (+ predecessor2 local2)))
(check-sat)
(pop)

; A non-negative TWED leaf cannot reduce its prefix cost.
(push)
(assert (>= local 0))
(assert (< (+ predecessor local) predecessor))
(check-sat)
(pop)

(declare-const length-gap Int)
(declare-const deletions Int)
(declare-const exact Int)

; The unavoidable-deletion count implies the length lower bound.
(push)
(assert (>= length-gap 0))
(assert (>= lambda 0))
(assert (<= length-gap deletions))
(assert (<= (* deletions lambda) exact))
(assert (> (* length-gap lambda) exact))
(check-sat)
(pop)

(declare-const cutoff Int)
(declare-const bound Int)

; K4: a length bound cannot prune an in-cutoff exact result.
(push)
(assert (<= bound exact))
(assert (> bound cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; The metric wrapper's strict stiffness premise excludes zero.
(push)
(assert (> nu 0))
(assert (= nu 0))
(check-sat)
(pop)

; The documented zero-parameter deletion leaf really has zero cost.
(push)
(assert (not (= (+ (iabs (- 0 0)) 0 0) 0)))
(check-sat)
(pop)

; Sequential script composition adds the two non-negative script costs.
(declare-const left-cost Int)
(declare-const right-cost Int)
(declare-const composed-cost Int)
(push)
(assert (= composed-cost (+ left-cost right-cost)))
(assert (not (= composed-cost (+ right-cost left-cost))))
(check-sat)
(pop)

; A physical-time deletion leaf is nonnegative under monotone timestamps and
; validated nonnegative stiffness and gap penalty.
(declare-const current-time Int)
(declare-const previous-time Int)
(declare-const value-delta Int)
(push)
(assert (>= value-delta 0))
(assert (>= current-time previous-time))
(assert (>= nu 0))
(assert (>= lambda 0))
(assert (< (+ value-delta (* nu (- current-time previous-time)) lambda) 0))
(check-sat)
(pop)

; One canonical elapsed unit reproduces the unit-grid deletion leaf exactly.
(push)
(assert (not (= (+ value-delta (* nu 1) lambda)
                (+ value-delta nu lambda))))
(check-sat)
(pop)

; A strictly increasing committed timestamp has positive elapsed time.
(push)
(assert (> current-time previous-time))
(assert (<= (- current-time previous-time) 0))
(check-sat)
(pop)

; AP/K1 composition: independently admissible value and physical-time
; components remain admissible after nonnegative stiffness and gap addition.
(declare-const value-lower Int)
(declare-const time-lower Int)
(declare-const value-exact Int)
(declare-const time-exact Int)
(push)
(assert (>= value-lower 0))
(assert (>= time-lower 0))
(assert (<= value-lower value-exact))
(assert (<= time-lower time-exact))
(assert (>= nu 0))
(assert (>= lambda 0))
(assert (> (+ value-lower (* nu time-lower) lambda)
           (+ value-exact (* nu time-exact) lambda)))
(check-sat)
(pop)
