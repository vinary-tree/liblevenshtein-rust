; Banded-DTW counterexample checks. Every query must be UNSAT in Z3 and cvc5.
(set-logic QF_NRA)

(define-fun iabs ((value Real)) Real
  (ite (>= value 0) value (- value)))

(define-fun imin ((left Real) (right Real)) Real
  (ite (<= left right) left right))

(define-fun min3 ((a Real) (b Real) (c Real)) Real
  (imin a (imin b c)))

(define-fun sq ((value Real)) Real
  (* value value))

(define-fun interval-dist ((value Real) (low Real) (high Real)) Real
  (ite (< value low)
       (- low value)
       (ite (< high value) (- value high) 0)))

(define-fun dtw-step ((north Real) (west Real) (diagonal Real) (local Real)) Real
  (+ (min3 north west diagonal) local))

(declare-const value Real)
(declare-const low Real)
(declare-const high Real)
(declare-const concrete Real)

; Exact interval minima remain lower bounds after squaring.
(push)
(assert (<= low high))
(assert (<= low concrete))
(assert (<= concrete high))
(assert (> (sq (interval-dist value low high)) (sq (- value concrete))))
(check-sat)
(pop)

; A point interval reproduces the exact squared local cost.
(push)
(assert (not (= (sq (interval-dist value concrete concrete))
                (sq (- value concrete)))))
(check-sat)
(pop)

; The complete additive cell is monotone in predecessor and local costs.
(declare-const n Real)
(declare-const w Real)
(declare-const d Real)
(declare-const local Real)
(declare-const n2 Real)
(declare-const w2 Real)
(declare-const d2 Real)
(declare-const local2 Real)
(push)
(assert (<= n n2))
(assert (<= w w2))
(assert (<= d d2))
(assert (<= local local2))
(assert (> (dtw-step n w d local) (dtw-step n2 w2 d2 local2)))
(check-sat)
(pop)

; Non-negative local cost makes additive accumulation inflationary.
(declare-const prefix Real)
(push)
(assert (>= prefix 0))
(assert (>= local 0))
(assert (< (+ prefix local) prefix))
(check-sat)
(pop)

; One incremental LB_Keogh step preserves admissibility.
(declare-const exact-prefix Real)
(declare-const exact-local Real)
(push)
(assert (<= prefix exact-prefix))
(assert (<= local exact-local))
(assert (> (+ prefix local) (+ exact-prefix exact-local)))
(check-sat)
(pop)

; The first gate cannot prune an exact in-cutoff descendant.
(declare-const bound Real)
(declare-const exact Real)
(declare-const cutoff Real)
(push)
(assert (<= bound exact))
(assert (> bound cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; An endpoint length gap wider than the band cannot be in-band.
(declare-const query-len Real)
(declare-const target-len Real)
(declare-const band Real)
(push)
(assert (>= query-len 0))
(assert (>= target-len 0))
(assert (>= band 0))
(assert (< band (iabs (- query-len target-len))))
(assert (<= (iabs (- query-len target-len)) band))
(check-sat)
(pop)

; Squared local cost is symmetric and non-negative.
(declare-const x Real)
(declare-const y Real)
(push)
(assert (not (= (sq (- x y)) (sq (- y x)))))
(check-sat)
(pop)
(push)
(assert (< (sq (- x y)) 0))
(check-sat)
(pop)

; Band-one witness: squared costs are 1, 0, and 2 respectively.
(push)
(assert (or
  (not (= (sq (- 0 1)) 1))
  (not (= (+ (sq (- 1 1)) (sq (- 1 1))) 0))
  (not (= (+ (sq (- 0 1)) (sq (- 0 1))) 2))
  (not (< (+ 1 0) 2))))
(check-sat)
(pop)
