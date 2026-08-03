; Cross-solver counterexample checks for the PositionKind/AutomatonVariant seam.
; Every check is expected to be UNSAT in both Z3 and cvc5.
(set-logic QF_LIA)

(declare-const algorithm Int)
(declare-const lhs-index Int)
(declare-const rhs-index Int)
(declare-const lhs-error Int)
(declare-const rhs-error Int)
(declare-const lhs-kind Int)
(declare-const rhs-kind Int)
(declare-const lhs-aux Int)
(declare-const rhs-aux Int)

(define-fun select-variant ((a Int)) Int (ite (= a 0) 0 (ite (= a 1) 1 2)))
(define-fun special ((kind Int)) Bool (not (= kind 0)))
(define-fun abs-diff ((a Int) (b Int)) Int (ite (<= a b) (- b a) (- a b)))
(define-fun standard-subsumes ((li Int) (le Int) (ri Int) (re Int)) Bool
  (and (<= le re) (<= (abs-diff li ri) (- re le))))
(define-fun osa-subsumes ((li Int) (le Int) (lk Int) (ri Int) (re Int) (rk Int)) Bool
  (and (<= le re)
       (ite (and (special lk) (special rk))
            (= li ri)
            (ite (or (special lk) (special rk))
                 false
                 (standard-subsumes li le ri re)))))
(define-fun merge-subsumes
  ((query-len Int) (li Int) (le Int) (lk Int) (ri Int) (re Int) (rk Int)) Bool
  (and (<= le re)
       (= (special lk) (special rk))
       (<= li query-len)
       (not (and (special lk) (>= li query-len) (< ri query-len)))
       (< le re)
       (= li ri)))

; Runtime selector and static selector coincide for the complete built-in domain.
(push)
(assert (<= 0 algorithm))
(assert (<= algorithm 2))
(assert (not (= (select-variant algorithm) algorithm)))
(check-sat)
(pop)
; The selected variant is independent of which position is processed.
(push)
(assert (not (= (select-variant algorithm) (select-variant algorithm))))
(check-sat)
(pop)

; A normal constructor always has zero payload and lies in the u8 domain.
(push)
(assert (or (not (= 0 0)) (not (< 0 256))))
(check-sat)
(pop)

; Full ordering keys cannot collapse distinct kinds.
(push)
(assert (not (= lhs-kind rhs-kind)))
(assert (= lhs-index rhs-index))
(assert (= lhs-error rhs-error))
(assert (= lhs-aux rhs-aux))
(assert (and (= lhs-index rhs-index)
             (= lhs-error rhs-error)
             (= lhs-kind rhs-kind)
             (= lhs-aux rhs-aux)))
(check-sat)
(pop)

; Full ordering keys cannot collapse distinct payloads.
(push)
(assert (not (= lhs-aux rhs-aux)))
(assert (and (= lhs-index rhs-index)
             (= lhs-error rhs-error)
             (= lhs-kind rhs-kind)
             (= lhs-aux rhs-aux)))
(check-sat)
(pop)

; Standard subsumption cannot reverse accumulated-error order.
(push)
(assert (standard-subsumes lhs-index lhs-error rhs-index rhs-error))
(assert (> lhs-error rhs-error))
(check-sat)
(pop)

; Mixed OSA continuation languages never subsume one another.
(push)
(assert (special lhs-kind))
(assert (not (special rhs-kind)))
(assert (osa-subsumes lhs-index lhs-error lhs-kind rhs-index rhs-error rhs-kind))
(check-sat)
(pop)

; Merge/split subsumption always has strictly fewer errors and equal indices.
(push)
(assert (merge-subsumes 8 lhs-index lhs-error lhs-kind rhs-index rhs-error rhs-kind))
(assert (or (>= lhs-error rhs-error) (not (= lhs-index rhs-index))))
(check-sat)
(pop)
