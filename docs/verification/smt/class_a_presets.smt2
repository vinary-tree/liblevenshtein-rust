; Independent Class-A preset counterexample queries.
; Every check must be UNSAT in both Z3 and cvc5.

(set-logic QF_NIA)

(define-fun mismatch ((left Int) (right Int)) Int
  (ite (= left right) 0 1))

; A coordinate mismatch cannot violate the Hamming triangle inequality.
(push)
(declare-const left Int)
(declare-const middle Int)
(declare-const right Int)
(assert (> (mismatch left right)
           (+ (mismatch left middle) (mismatch middle right))))
(check-sat)
(pop)

; Hamming coordinate cost is symmetric.
(push)
(declare-const left Int)
(declare-const right Int)
(assert (not (= (mismatch left right) (mismatch right left))))
(check-sat)
(pop)

; Insert/delete script cost bounds positive source length difference.
(push)
(declare-const kept Int)
(declare-const inserted Int)
(declare-const deleted Int)
(assert (and (>= kept 0) (>= inserted 0) (>= deleted 0)))
(define-fun source () Int (+ kept deleted))
(define-fun target () Int (+ kept inserted))
(define-fun cost () Int (+ inserted deleted))
(assert (> (- source target) cost))
(check-sat)
(pop)

; The symmetric length-difference direction is also bounded.
(push)
(declare-const kept Int)
(declare-const inserted Int)
(declare-const deleted Int)
(assert (and (>= kept 0) (>= inserted 0) (>= deleted 0)))
(define-fun source () Int (+ kept deleted))
(define-fun target () Int (+ kept inserted))
(define-fun cost () Int (+ inserted deleted))
(assert (> (- target source) cost))
(check-sat)
(pop)

; Reversing a script swaps insertion/deletion counts but preserves cost.
(push)
(declare-const inserted Int)
(declare-const deleted Int)
(assert (not (= (+ inserted deleted) (+ deleted inserted))))
(check-sat)
(pop)

; Script concatenation adds insertion/deletion cost.
(push)
(declare-const i1 Int)
(declare-const d1 Int)
(declare-const i2 Int)
(declare-const d2 Int)
(assert (not (= (+ (+ i1 i2) (+ d1 d2))
                (+ (+ i1 d1) (+ i2 d2)))))
(check-sat)
(pop)

; Source-plus-target length and indel cost have identical parity.
(push)
(declare-const kept Int)
(declare-const inserted Int)
(declare-const deleted Int)
(assert (and (>= kept 0) (>= inserted 0) (>= deleted 0)))
(assert (not (= (mod (+ (+ kept deleted) (+ kept inserted)) 2)
                (mod (+ inserted deleted) 2))))
(check-sat)
(pop)

; A bounded-skip path costs exactly the unmatched source length.
(push)
(declare-const matched Int)
(declare-const skipped Int)
(assert (and (>= matched 0) (>= skipped 0)))
(declare-const source Int)
(declare-const target Int)
(declare-const cost Int)
(assert (= source (+ matched skipped)))
(assert (= target matched))
(assert (= cost skipped))
(assert (not (= (- source target) cost)))
(check-sat)
(pop)

; A prefix of a validated aggregate cannot exceed the complete limit.
(push)
(declare-const prefix Int)
(declare-const suffix Int)
(declare-const limit Int)
(assert (and (>= prefix 0) (>= suffix 0) (>= limit 0)))
(assert (<= (+ prefix suffix) limit))
(assert (> prefix limit))
(check-sat)
(pop)

; Positive aggregate consumption means at least one coordinate advances.
(push)
(declare-const source Int)
(declare-const target Int)
(assert (and (>= source 0) (>= target 0)))
(assert (> (+ source target) 0))
(assert (and (= source 0) (= target 0)))
(check-sat)
(pop)

; Guarded aggregation cannot cross its configured ceiling.
(push)
(declare-const aggregate Int)
(declare-const source Int)
(declare-const target Int)
(declare-const limit Int)
(assert (and (>= aggregate 0) (>= source 0) (>= target 0) (>= limit 0)))
(assert (<= aggregate limit))
(assert (<= (+ source target) (- limit aggregate)))
(assert (> (+ aggregate source target) limit))
(check-sat)
(pop)

; The affordable empty-side branch returns the exact all-deletion cost.
(push)
(declare-const length Int)
(declare-const budget Int)
(assert (and (>= length 0) (>= budget 0)))
(assert (<= length budget))
(define-fun bounded-empty-result () Int (ite (<= length budget) length (- 1)))
(assert (not (= bounded-empty-result length)))
(check-sat)
(pop)

; Any prefix affordable within k remains inside the k-diagonal band.
(push)
(declare-const row Int)
(declare-const column Int)
(declare-const prefix-cost Int)
(declare-const budget Int)
(assert (and (>= row 0) (>= column 0) (>= prefix-cost 0) (>= budget 0)))
(assert (<= prefix-cost budget))
(assert (<= row (+ column prefix-cost)))
(assert (<= column (+ row prefix-cost)))
(assert (or (> row (+ column budget)) (> column (+ row budget))))
(check-sat)
(pop)
