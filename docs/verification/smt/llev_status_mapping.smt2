; Cross-solver checks for the liblevenshtein error->status mapping
; (src/ffi/index.rs `map_binding_error`, family FV obligation #6). Each check
; asks for a counterexample; the expected result is UNSAT in BOTH z3 and cvc5.
;
; LlevStatus wire codes (src/ffi/generated.rs): Ok=0 End=1 InvalidArgument=2
;   InvalidUtf8=3 NullPointer=4 Panic=5 Unsupported=6 IoError=7 Closed=8
;   LimitExceeded=9 ProviderError=10 BatchInUse=11 DomainMismatch=12.
; Interop VtStatus codes (vinary-tree-interop): Ok=0 End=1 InvalidArgument=2
;   NullPointer=3 Unsupported=4 IoError=5 Closed=6 LimitExceeded=7
;   ProviderError=8.
;
; Registry: docs/verification/ABI_INVARIANTS.tsv LLEV-STAT-1..6. Executable
; mirror: tests/ffi_provider_fault_injection.rs (the per-status pin matrix).
(set-logic QF_BV)

; ---- LlevStatus constants -------------------------------------------------
(define-fun LOk            () (_ BitVec 8) #x00)
(define-fun LEnd           () (_ BitVec 8) #x01)
(define-fun LInvalidArg    () (_ BitVec 8) #x02)
(define-fun LInvalidUtf8   () (_ BitVec 8) #x03)
(define-fun LNullPointer   () (_ BitVec 8) #x04)
(define-fun LPanic         () (_ BitVec 8) #x05)
(define-fun LUnsupported   () (_ BitVec 8) #x06)
(define-fun LIoError       () (_ BitVec 8) #x07)
(define-fun LClosed        () (_ BitVec 8) #x08)
(define-fun LLimitExceeded () (_ BitVec 8) #x09)
(define-fun LProviderError () (_ BitVec 8) #x0a)
(define-fun LBatchInUse    () (_ BitVec 8) #x0b)
(define-fun LDomainMismatch() (_ BitVec 8) #x0c)

; ---- The Provider(status) sub-map of map_binding_error --------------------
; The exact match arm: the six directly-preserved interop codes, everything
; else (Ok, End, ProviderError) to ProviderError.
(define-fun provider-map ((v (_ BitVec 8))) (_ BitVec 8)
  (ite (= v #x02) LInvalidArg        ; VtStatus::InvalidArgument
  (ite (= v #x03) LNullPointer       ; VtStatus::NullPointer
  (ite (= v #x04) LUnsupported       ; VtStatus::Unsupported
  (ite (= v #x05) LIoError           ; VtStatus::IoError
  (ite (= v #x06) LClosed            ; VtStatus::Closed
  (ite (= v #x07) LLimitExceeded     ; VtStatus::LimitExceeded
                  LProviderError))))))) ; Ok, End, ProviderError -> ProviderError

; A valid interop status the decoder (VtStatus::from_raw) admits: 0..=8.
(define-fun vt-in-range ((v (_ BitVec 8))) Bool
  (bvule v #x08))

; A valid LlevStatus code: 0..=12.
(define-fun llev-in-range ((s (_ BitVec 8))) Bool
  (bvule s #x0c))

(declare-const v (_ BitVec 8))

; ---- LLEV-STAT-1: totality -------------------------------------------------
; Every admissible interop status maps to a valid LlevStatus code.
(echo "[LLEV-STAT-1] provider-map total into the LlevStatus range")
(push)
(assert (vt-in-range v))
(assert (not (llev-in-range (provider-map v))))
(check-sat)
(pop)

; ---- LLEV-STAT-2: Ok/End are never an error mapping ------------------------
; No provider status is mapped to Ok or End (an error can never masquerade as
; success or exhaustion).
(echo "[LLEV-STAT-2] provider-map never yields Ok or End")
(push)
(assert (vt-in-range v))
(assert (or (= (provider-map v) LOk) (= (provider-map v) LEnd)))
(check-sat)
(pop)

; ---- LLEV-STAT-3: no error swallowing -------------------------------------
; A non-Ok provider status never maps to Ok (the specific swallowing hazard).
(echo "[LLEV-STAT-3] a non-Ok provider status never maps to Ok")
(push)
(assert (vt-in-range v))
(assert (not (= v LOk)))
(assert (= (provider-map v) LOk))
(check-sat)
(pop)

; ---- LLEV-STAT-4: Panic is never produced by the mapping ------------------
; Panic is reserved for boundary()'s catch_unwind arm; the error mapping never
; produces it.
(echo "[LLEV-STAT-4] provider-map never yields Panic")
(push)
(assert (vt-in-range v))
(assert (= (provider-map v) LPanic))
(check-sat)
(pop)

; ---- LLEV-STAT-5: the six preserved codes keep their meaning ---------------
; Each directly-modeled interop status maps to the LlevStatus of the same
; meaning (semantic preservation of the shared vocabulary).
(echo "[LLEV-STAT-5] the six preserved codes are semantics-preserving")
(push)
(assert (not (and
  (= (provider-map #x02) LInvalidArg)
  (= (provider-map #x03) LNullPointer)
  (= (provider-map #x04) LUnsupported)
  (= (provider-map #x05) LIoError)
  (= (provider-map #x06) LClosed)
  (= (provider-map #x07) LLimitExceeded))))
(check-sat)
(pop)

; ---- LLEV-STAT-6: End (and Ok) from an interface callback -> ProviderError -
; A provider returning Ok or End from inside an interface call is a protocol
; violation surfaced as ProviderError, not passed through.
(echo "[LLEV-STAT-6] Ok/End from a callback fold to ProviderError")
(push)
(assert (or (= v LOk) (= v LEnd)))
(assert (not (= (provider-map v) LProviderError)))
(check-sat)
(pop)
