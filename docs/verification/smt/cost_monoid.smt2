; Cross-solver bounded checks for CostMonoid and CostScale arithmetic.
; Each check asks for a counterexample. The expected result is UNSAT.
(set-logic QF_BV)

; Saturating unsigned addition is associative.
(define-fun sat-add8 ((a (_ BitVec 8)) (b (_ BitVec 8))) (_ BitVec 8)
  (let ((sum (bvadd ((_ zero_extend 1) a) ((_ zero_extend 1) b))))
    (ite (bvugt sum #b011111111) #xff ((_ extract 7 0) sum))))
(declare-const a8 (_ BitVec 8))
(declare-const b8 (_ BitVec 8))
(declare-const c8 (_ BitVec 8))
(push)
(assert (not (= (sat-add8 (sat-add8 a8 b8) c8)
                (sat-add8 a8 (sat-add8 b8 c8)))))
(check-sat)
(pop)

; Saturating addition is monotone in its left operand.
(push)
(assert (bvule a8 b8))
(assert (bvugt (sat-add8 a8 c8) (sat-add8 b8 c8)))
(check-sat)
(pop)

; Unsigned max, the finite BottleneckCost model, is associative.
(define-fun max8 ((a (_ BitVec 8)) (b (_ BitVec 8))) (_ BitVec 8)
  (ite (bvule a b) b a))
(push)
(assert (not (= (max8 (max8 a8 b8) c8)
                (max8 a8 (max8 b8 c8)))))
(check-sat)
(pop)

; The all-ones TOP value absorbs either side of saturating addition.
(push)
(assert (or (not (= (sat-add8 a8 #xff) #xff))
            (not (= (sat-add8 #xff a8) #xff))))
(check-sat)
(pop)

; u8 budget scaling into usize is safe whenever the checked-multiply guard says so.
(declare-const budget8 (_ BitVec 8))
(declare-const denominator32 (_ BitVec 32))
(push)
(assert (not (= budget8 #x00)))
(assert (bvule denominator32 (bvudiv #xffffffff ((_ zero_extend 24) budget8))))
(assert (bvugt
  (bvmul ((_ zero_extend 32) denominator32)
         ((_ zero_extend 56) budget8))
  #x00000000ffffffff))
(check-sat)
(pop)
