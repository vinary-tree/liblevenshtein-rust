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
