; ERP kernel counterexample checks. Every query must be UNSAT in Z3 and cvc5.
(set-logic QF_LIA)

(define-fun iabs ((value Int)) Int
  (ite (>= value 0) value (- value)))

(define-fun interval-dist ((value Int) (low Int) (high Int)) Int
  (ite (< value low)
       (- low value)
       (ite (< high value) (- value high) 0)))

(declare-const value Int)
(declare-const low Int)
(declare-const high Int)
(declare-const concrete Int)

; K1 leaf obligation: interval distance lower-bounds every realization.
(push)
(assert (<= low high))
(assert (<= low concrete))
(assert (<= concrete high))
(assert (> (interval-dist value low high) (iabs (- value concrete))))
(check-sat)
(pop)

; Degenerate bins reproduce the scalar absolute cost exactly.
(push)
(assert (not (= (interval-dist value concrete concrete)
                (iabs (- value concrete)))))
(check-sat)
(pop)

; Match edits respect the gap-mass potential by reverse triangle inequality.
(declare-const x Int)
(declare-const y Int)
(declare-const gap Int)
(push)
(assert (> (iabs (- (iabs (- x gap)) (iabs (- y gap))))
           (iabs (- x y))))
(check-sat)
(pop)

; Deleting the gap itself costs zero (the quotient's defining generator).
(push)
(assert (not (= (iabs (- gap gap)) 0)))
(check-sat)
(pop)

; Conversely, zero deletion cost identifies the value with the gap.
(push)
(assert (= (iabs (- value gap)) 0))
(assert (not (= value gap)))
(check-sat)
(pop)

; K4: a gap-mass candidate bound cannot prune an in-cutoff exact result.
(declare-const left-mass Int)
(declare-const right-mass Int)
(declare-const exact Int)
(declare-const cutoff Int)
(push)
(assert (<= (iabs (- left-mass right-mass)) exact))
(assert (> (iabs (- left-mass right-mass)) cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; Row-minimum early abandonment is sound once the row minimum bounds exact.
(declare-const row-minimum Int)
(push)
(assert (<= row-minimum exact))
(assert (> row-minimum cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; K2: a non-negative ERP edit cannot decrease accumulated cost.
(declare-const prefix Int)
(declare-const step Int)
(push)
(assert (>= step 0))
(assert (< (+ prefix step) prefix))
(check-sat)
(pop)
