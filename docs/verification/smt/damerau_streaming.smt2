; Cross-solver obligations for the streaming unrestricted-Damerau variant.
; Every check is expected to be UNSAT in both Z3 and cvc5.
(set-logic QF_LIA)

(declare-const error Int)
(declare-const delta Int)
(declare-const between Int)
(declare-const budget Int)
(declare-const lhs-index Int)
(declare-const rhs-index Int)
(declare-const lhs-error Int)
(declare-const rhs-error Int)
(declare-const lhs-delta Int)
(declare-const rhs-delta Int)

(define-fun entry-error ((e Int) (d Int)) Int (+ e d))
(define-fun macro-cost ((d Int) (b Int)) Int (+ d b))
(define-fun lw-cost ((d Int) (b Int)) Int (+ (- d 1) b 1))
(define-fun pending-subsumes
  ((li Int) (le Int) (ld Int) (ri Int) (re Int) (rd Int)) Bool
  (and (<= le re) (= li ri) (= ld rd)))

; A guarded entry cannot exceed the active budget.
(push)
(assert (>= error 0))
(assert (>= delta 1))
(assert (< delta 256))
(assert (<= (+ error delta) budget))
(assert (> (entry-error error delta) budget))
(check-sat)
(pop)
; Entry plus intervening dictionary insertions is the L&W macro charge.
(push)
(assert (>= delta 1))
(assert (>= between 0))
(assert (not (= (macro-cost delta between) (lw-cost delta between))))
(check-sat)
(pop)

; A resolving pending representative advances strictly forward.
(push)
(assert (>= lhs-index 0))
(assert (>= delta 1))
(assert (not (> (+ lhs-index delta 1) lhs-index)))
(check-sat)
(pop)

; Pending subsumption cannot hide unequal origins or deltas.
(push)
(assert (pending-subsumes lhs-index lhs-error lhs-delta
                          rhs-index rhs-error rhs-delta))
(assert (or (not (= lhs-index rhs-index))
            (not (= lhs-delta rhs-delta))
            (> lhs-error rhs-error)))
(check-sat)
(pop)

; Unequal deltas are incomparable even at the same origin.
(push)
(assert (not (= lhs-delta rhs-delta)))
(assert (pending-subsumes lhs-index lhs-error lhs-delta
                          lhs-index rhs-error rhs-delta))
(check-sat)
(pop)
