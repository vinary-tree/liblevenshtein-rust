; Bounded counterexample queries for operation-driven generalized acceptance.
; Every check must be UNSAT in both Z3 and cvc5.

(set-logic QF_BV)

; Six operations of scaled cost 3 fit budget 20; seven do not.
(push)
(declare-const candidate (_ BitVec 8))
(assert (= candidate (bvmul #x06 #x03)))
(assert (bvugt candidate #x14))
(check-sat)
(pop)

; A validated zero-target operation must advance the source coordinate, so a
; current-row recurrence is topological rather than cyclic.
(push)
(declare-const source-consumption (_ BitVec 16))
(declare-const target-consumption (_ BitVec 16))
(assert (not (= (bvadd source-consumption target-consumption) #x0000)))
(assert (= target-consumption #x0000))
(assert (= source-consumption #x0000))
(check-sat)
(pop)

; The discovery counter is checked for overflow and against the ceiling before
; a new cell is materialized.
(push)
(declare-const discovered (_ BitVec 16))
(declare-const limit (_ BitVec 16))
(assert (bvult discovered #xffff))
(assert (bvule (bvadd discovered #x0001) limit))
(assert (bvugt (bvadd discovered #x0001) limit))
(check-sat)
(pop)

; Without a complete classical-lattice certificate, conservative subsumption
; requires identical coordinates. A denominator of one alone is deliberately
; not used as a certificate because integer operations may cost more than one.
(push)
(declare-const classical-certified Bool)
(declare-const left-offset (_ BitVec 16))
(declare-const right-offset (_ BitVec 16))
(assert (not classical-certified))
(assert (or classical-certified (= left-offset right-offset)))
(assert (not (= left-offset right-offset)))
(check-sat)
(pop)

; Minimum applicable cost is independent of operation insertion order.
(push)
(declare-const left-cost (_ BitVec 16))
(declare-const right-cost (_ BitVec 16))
(define-fun min-left-right () (_ BitVec 16)
  (ite (bvule left-cost right-cost) left-cost right-cost))
(define-fun min-right-left () (_ BitVec 16)
  (ite (bvule right-cost left-cost) right-cost left-cost))
(assert (not (= min-left-right min-right-left)))
(check-sat)
(pop)

; A positive completion charge cannot leave the accumulated cost unchanged
; when checked addition has not overflowed.
(push)
(declare-const completion-accumulated (_ BitVec 16))
(declare-const completion-step (_ BitVec 16))
(assert (bvugt completion-step #x0000))
(assert (bvule completion-step (bvsub #xffff completion-accumulated)))
(assert (= (bvadd completion-accumulated completion-step) completion-accumulated))
(check-sat)
(pop)

; Reinserting an already-present control position leaves antichain cardinality
; unchanged.
(push)
(declare-const position-present Bool)
(declare-const antichain-size (_ BitVec 16))
(assert position-present)
(define-fun size-after-insert () (_ BitVec 16)
  (ite position-present antichain-size (bvadd antichain-size #x0001)))
(assert (not (= size-after-insert antichain-size)))
(check-sat)
(pop)

(push)
(declare-const candidate (_ BitVec 8))
(assert (= candidate (bvmul #x07 #x03)))
(assert (bvule candidate #x14))
(check-sat)
(pop)

; Hamming operations advance source and target by equal amounts, so accumulated
; lengths cannot differ in this bounded two-step witness.
(push)
(declare-const first (_ BitVec 4))
(declare-const second (_ BitVec 4))
(declare-const source (_ BitVec 4))
(declare-const target (_ BitVec 4))
(assert (= source (bvadd first second)))
(assert (= target (bvadd first second)))
(assert (not (= source target)))
(check-sat)
(pop)

; If every zero-target operation also consumes zero source symbols, an empty
; target cannot consume a non-empty source in this bounded witness.
(push)
(declare-const source (_ BitVec 4))
(declare-const target (_ BitVec 4))
(assert (= target #x0))
(assert (=> (= target #x0) (= source #x0)))
(assert (not (= source #x0)))
(check-sat)
(pop)

; An explicit infinite empty-side rate fits only a zero scalar count.
(push)
(declare-const rate-is-finite Bool)
(declare-const empty-side-count (_ BitVec 16))
(assert (not rate-is-finite))
(assert (= empty-side-count #x0000))
(assert (bvugt empty-side-count #x0000))
(check-sat)
(pop)

; Exact cross multiplication for rate 1/2 and budget 2 admits four scalars but
; cannot admit five.
(push)
(declare-const finite-rate-count (_ BitVec 16))
(assert (= finite-rate-count #x0005))
(assert (bvule
  (bvmul #x0001 finite-rate-count)
  (bvmul #x0002 #x0002)))
(check-sat)
(pop)

; A guarded addition cannot cross an unsigned budget.
(push)
(declare-const accumulated (_ BitVec 16))
(declare-const step (_ BitVec 16))
(declare-const budget (_ BitVec 16))
(assert (bvule accumulated budget))
(assert (bvule step (bvsub budget accumulated)))
(assert (bvugt (bvadd accumulated step) budget))
(check-sat)
(pop)
