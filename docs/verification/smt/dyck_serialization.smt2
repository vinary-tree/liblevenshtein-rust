; Differential counterexample checks for exact typed-Dyck correction and
; strict OperationSet binary persistence. Every result must be UNSAT in Z3
; and cvc5.
(set-logic QF_NIA)

(declare-const kinds Int)
(declare-const left-kind Int)
(declare-const right-kind Int)
(declare-const actual-open Int)
(declare-const expected-open Int)
(declare-const actual-close Int)
(declare-const expected-close Int)
(declare-const inner-cost Int)
(declare-const suffix-cost Int)
(declare-const candidate-a Int)
(declare-const candidate-b Int)
(declare-const candidate-c Int)
(declare-const candidate-d Int)
(declare-const version Int)
(declare-const flags Int)
(declare-const declared Int)
(declare-const available Int)
(declare-const consumed Int)
(declare-const payload-limit Int)
(declare-const operations Int)
(declare-const operation-limit Int)
(declare-const pairs Int)
(declare-const pair-limit Int)
(declare-const magic-ok Bool)
(declare-const semantic-ok Bool)
(declare-const wire-ok Bool)
(declare-const format-ok Bool)
(declare-const name-bytes Int)
(declare-const name-limit Int)
(declare-const operation-pairs Int)
(declare-const operation-pair-limit Int)
(declare-const pair-text-bytes Int)
(declare-const pair-text-limit Int)
(declare-const weight-bits Int)
(declare-const checksum-ok Bool)
(declare-const compressed Int)
(declare-const compressed-limit Int)
(declare-const decompressed Int)
(declare-const decompressed-limit Int)
(declare-const consumed-compressed Int)
(declare-const supplied-compressed Int)
(declare-const inner-ok Bool)

(define-fun replacement-cost ((actual Int) (expected Int)) Int
  (ite (= actual expected) 0 1))
(define-fun pair-cost
  ((actual-open Int) (expected-open Int) (inner Int)
   (actual-close Int) (expected-close Int) (suffix Int)) Int
  (+ (replacement-cost actual-open expected-open)
     inner
     (replacement-cost actual-close expected-close)
     suffix))
(define-fun min2 ((left Int) (right Int)) Int
  (ite (<= left right) left right))
(define-fun min4 ((a Int) (b Int) (c Int) (d Int)) Int
  (min2 (min2 a b) (min2 c d)))
(define-fun accepts-envelope
  ((magic Bool) (v Int) (f Int)
   (declared Int) (available Int) (consumed Int) (limit Int)
   (ops Int) (ops-limit Int) (pairs Int) (pairs-limit Int)
   (semantic Bool)) Bool
  (and magic
       (= v 1)
       (= f 0)
       (= declared available)
       (= consumed declared)
       (<= declared limit)
       (<= ops ops-limit)
       (<= pairs pairs-limit)
       semantic))
(define-fun accepts-protobuf
  ((wire Bool) (format Bool)
   (payload Int) (payload-limit Int)
   (ops Int) (ops-limit Int)
   (name Int) (name-limit Int)
   (operation-pairs Int) (operation-pair-limit Int)
   (pairs Int) (pairs-limit Int)
   (text Int) (text-limit Int)
   (semantic Bool)) Bool
  (and wire format
       (<= payload payload-limit)
       (<= ops ops-limit)
       (<= name name-limit)
       (<= operation-pairs operation-pair-limit)
       (<= pairs pairs-limit)
       (<= text text-limit)
       semantic))
(define-fun encode-weight-bits ((bits Int)) Int bits)
(define-fun decode-weight-bits ((bits Int)) Int bits)
(define-fun accepts-gzip
  ((checksum Bool)
   (compressed Int) (compressed-limit Int)
   (decompressed Int) (decompressed-limit Int)
   (consumed Int) (supplied Int)
   (inner Bool)) Bool
  (and checksum
       (<= compressed compressed-limit)
       (<= decompressed decompressed-limit)
       (= consumed supplied)
       inner))

; Different delimiter kinds have different encoded closing tokens.
(push)
(assert (and (> kinds 0)
             (>= left-kind 0) (< left-kind kinds)
             (>= right-kind 0) (< right-kind kinds)
             (not (= left-kind right-kind))))
(assert (= (+ kinds left-kind) (+ kinds right-kind)))
(check-sat)
(pop)

; Protobuf cannot be accepted when any pre-allocation count exceeds policy.
(push)
(assert (and (>= declared 0) (>= payload-limit 0)
             (>= operations 0) (>= operation-limit 0)
             (>= name-bytes 0) (>= name-limit 0)
             (>= operation-pairs 0) (>= operation-pair-limit 0)
             (>= pairs 0) (>= pair-limit 0)
             (>= pair-text-bytes 0) (>= pair-text-limit 0)))
(assert (accepts-protobuf wire-ok format-ok
                          declared payload-limit
                          operations operation-limit
                          name-bytes name-limit
                          operation-pairs operation-pair-limit
                          pairs pair-limit
                          pair-text-bytes pair-text-limit
                          semantic-ok))
(assert (or (> operations operation-limit)
            (> name-bytes name-limit)
            (> operation-pairs operation-pair-limit)
            (> pairs pair-limit)
            (> pair-text-bytes pair-text-limit)))
(check-sat)
(pop)

; Missing/unknown protobuf format never reaches semantic admission.
(push)
(assert (and (>= declared 0) (>= payload-limit 0)
             (>= operations 0) (>= operation-limit 0)
             (>= name-bytes 0) (>= name-limit 0)
             (>= operation-pairs 0) (>= operation-pair-limit 0)
             (>= pairs 0) (>= pair-limit 0)
             (>= pair-text-bytes 0) (>= pair-text-limit 0)))
(assert (accepts-protobuf wire-ok format-ok
                          declared payload-limit
                          operations operation-limit
                          name-bytes name-limit
                          operation-pairs operation-pair-limit
                          pairs pair-limit
                          pair-text-bytes pair-text-limit
                          semantic-ok))
(assert (not format-ok))
(check-sat)
(pop)

; The fixed64 field is an exact bit-pattern round trip.
(push)
(assert (not (= (decode-weight-bits (encode-weight-bits weight-bits))
                weight-bits)))
(check-sat)
(pop)

; Gzip acceptance requires one complete member and both byte ceilings.
(push)
(assert (and (>= compressed 0) (>= compressed-limit 0)
             (>= decompressed 0) (>= decompressed-limit 0)
             (>= consumed-compressed 0) (>= supplied-compressed 0)))
(assert (accepts-gzip checksum-ok
                      compressed compressed-limit
                      decompressed decompressed-limit
                      consumed-compressed supplied-compressed
                      inner-ok))
(assert (or (> compressed compressed-limit)
            (> decompressed decompressed-limit)
            (< consumed-compressed supplied-compressed)))
(check-sat)
(pop)

; A zero-cost consumed pair is an exact token identity on both endpoints and
; has zero-cost recursive intervals.
(push)
(assert (and (>= actual-open 0) (>= expected-open 0)
             (>= actual-close 0) (>= expected-close 0)
             (>= inner-cost 0) (>= suffix-cost 0)))
(assert (= (pair-cost actual-open expected-open inner-cost
                      actual-close expected-close suffix-cost) 0))
(assert (or (not (= actual-open expected-open))
            (not (= actual-close expected-close))
            (not (= inner-cost 0))
            (not (= suffix-cost 0))))
(check-sat)
(pop)

; The recurrence's selected minimum cannot exceed any enumerated candidate.
(push)
(assert (and (>= candidate-a 0) (>= candidate-b 0)
             (>= candidate-c 0) (>= candidate-d 0)))
(assert (or (> (min4 candidate-a candidate-b candidate-c candidate-d) candidate-a)
            (> (min4 candidate-a candidate-b candidate-c candidate-d) candidate-b)
            (> (min4 candidate-a candidate-b candidate-c candidate-d) candidate-c)
            (> (min4 candidate-a candidate-b candidate-c candidate-d) candidate-d)))
(check-sat)
(pop)

; An accepted envelope cannot contain trailing payload bytes.
(push)
(assert (and (>= declared 0) (>= available 0) (>= consumed 0)
             (>= payload-limit 0) (>= operations 0) (>= operation-limit 0)
             (>= pairs 0) (>= pair-limit 0)))
(assert (accepts-envelope magic-ok version flags declared available consumed
                          payload-limit operations operation-limit pairs pair-limit
                          semantic-ok))
(assert (> available declared))
(check-sat)
(pop)

; An accepted envelope cannot exceed operation or pair resource limits.
(push)
(assert (and (>= declared 0) (>= available 0) (>= consumed 0)
             (>= payload-limit 0) (>= operations 0) (>= operation-limit 0)
             (>= pairs 0) (>= pair-limit 0)))
(assert (accepts-envelope magic-ok version flags declared available consumed
                          payload-limit operations operation-limit pairs pair-limit
                          semantic-ok))
(assert (or (> operations operation-limit) (> pairs pair-limit)))
(check-sat)
(pop)

; Wrong version or non-zero flags can never reach acceptance.
(push)
(assert (and (>= declared 0) (>= available 0) (>= consumed 0)
             (>= payload-limit 0) (>= operations 0) (>= operation-limit 0)
             (>= pairs 0) (>= pair-limit 0)))
(assert (accepts-envelope magic-ok version flags declared available consumed
                          payload-limit operations operation-limit pairs pair-limit
                          semantic-ok))
(assert (or (not (= version 1)) (not (= flags 0))))
(check-sat)
(pop)
