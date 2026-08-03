; Cross-solver, bounded bit-vector checks for the Rust language product.
; Each check asks for a counterexample.  The expected result is UNSAT.
(set-logic QF_BV)

; Canonicalization preserves acceptance across a cheaper/dearer pair.
(declare-const cheaper (_ BitVec 8))
(declare-const dearer (_ BitVec 8))
(declare-const finals (_ BitVec 8))
(define-fun accepts ((states (_ BitVec 8))) Bool
  (not (= (bvand states finals) #x00)))
(push)
(assert (not
  (= (or (accepts cheaper) (accepts dearer))
     (or (accepts cheaper) (accepts (bvand dearer (bvnot cheaper)))))))
(check-sat)
(pop)

; A four-state relational image distributes over source-set union.  Each
; incoming mask describes every predecessor of one target state.
(declare-const incoming0 (_ BitVec 4))
(declare-const incoming1 (_ BitVec 4))
(declare-const incoming2 (_ BitVec 4))
(declare-const incoming3 (_ BitVec 4))
(declare-const left (_ BitVec 4))
(declare-const right (_ BitVec 4))
(define-fun hit ((states (_ BitVec 4)) (incoming (_ BitVec 4))) Bool
  (not (= (bvand states incoming) #b0000)))
(define-fun image ((states (_ BitVec 4))) (_ BitVec 4)
  (concat (ite (hit states incoming3) #b1 #b0)
    (concat (ite (hit states incoming2) #b1 #b0)
      (concat (ite (hit states incoming1) #b1 #b0)
              (ite (hit states incoming0) #b1 #b0)))))
(push)
(assert (not
  (= (image (bvor left right))
     (bvor (image left) (image right)))))
(check-sat)
(pop)

; Zero-extending a u8 budget before adding one cannot exceed 256.
(declare-const budget (_ BitVec 8))
(push)
(assert
  (bvugt (bvadd ((_ zero_extend 1) budget) #b000000001) #b100000000))
(check-sat)
(pop)

; SmallDfa reserves the sink bit and accepts real states only below 31, so its
; state bit is nonzero and the shift is defined within u32.
(declare-const state (_ BitVec 32))
(push)
(assert (bvult state #x0000001f))
(assert (= (bvshl #x00000001 state) #x00000000))
(check-sat)
(pop)
