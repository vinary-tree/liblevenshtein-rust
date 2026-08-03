; Discrete Frechet counterexample checks. Every query must be UNSAT in Z3 and cvc5.
(set-logic QF_LIA)

(define-fun iabs ((value Int)) Int
  (ite (>= value 0) value (- value)))

(define-fun imin ((left Int) (right Int)) Int
  (ite (<= left right) left right))

(define-fun imax ((left Int) (right Int)) Int
  (ite (<= left right) right left))

(define-fun min3 ((a Int) (b Int) (c Int)) Int
  (imin a (imin b c)))

(define-fun interval-dist ((value Int) (low Int) (high Int)) Int
  (ite (< value low)
       (- low value)
       (ite (< high value) (- value high) 0)))

(define-fun frechet-step ((north Int) (west Int) (diagonal Int) (link Int)) Int
  (imax (min3 north west diagonal) link))

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

; Degenerate bins reproduce scalar absolute distance exactly.
(push)
(assert (not (= (interval-dist value concrete concrete)
                (iabs (- value concrete)))))
(check-sat)
(pop)

; Bottleneck accumulation cannot reduce its prefix.
(declare-const prefix Int)
(declare-const link Int)
(push)
(assert (< (imax prefix link) prefix))
(check-sat)
(pop)

; The complete Fréchet cell transition is monotone in all four inputs.
(declare-const n Int)
(declare-const w Int)
(declare-const d Int)
(declare-const n2 Int)
(declare-const w2 Int)
(declare-const d2 Int)
(declare-const link2 Int)
(push)
(assert (<= n n2))
(assert (<= w w2))
(assert (<= d d2))
(assert (<= link link2))
(assert (> (frechet-step n w d link) (frechet-step n2 w2 d2 link2)))
(check-sat)
(pop)

; Both pinned endpoint links lower-bound every exact coupling.
(declare-const first Int)
(declare-const last Int)
(declare-const exact Int)
(push)
(assert (<= first exact))
(assert (<= last exact))
(assert (> (imax first last) exact))
(check-sat)
(pop)

; Combining endpoint and Hausdorff bounds with max remains admissible.
(declare-const endpoint Int)
(declare-const hausdorff Int)
(push)
(assert (<= endpoint exact))
(assert (<= hausdorff exact))
(assert (> (imax endpoint hausdorff) exact))
(check-sat)
(pop)

; K4: an admissible candidate bound cannot prune an in-cutoff result.
(declare-const bound Int)
(declare-const cutoff Int)
(push)
(assert (<= bound exact))
(assert (> bound cutoff))
(assert (<= exact cutoff))
(check-sat)
(pop)

; Pointwise triangle inequalities survive one bottleneck-composition step.
(declare-const prefix-xy Int)
(declare-const prefix-yz Int)
(declare-const prefix-xz Int)
(declare-const link-xy Int)
(declare-const link-yz Int)
(declare-const link-xz Int)
(push)
(assert (>= prefix-xy 0))
(assert (>= prefix-yz 0))
(assert (>= link-xy 0))
(assert (>= link-yz 0))
(assert (<= prefix-xz (+ prefix-xy prefix-yz)))
(assert (<= link-xz (+ link-xy link-yz)))
(assert (> (imax prefix-xz link-xz)
           (+ (imax prefix-xy link-xy) (imax prefix-yz link-yz))))
(check-sat)
(pop)

; A zero absolute link identifies its endpoints (quotient base case).
(declare-const x Int)
(declare-const y Int)
(push)
(assert (= (iabs (- x y)) 0))
(assert (not (= x y)))
(check-sat)
(pop)
