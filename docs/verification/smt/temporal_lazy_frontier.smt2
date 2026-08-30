; Counterexample checks for the lazy weighted temporal-frontier refinement.
; Every check is expected to be UNSAT in both Z3 and cvc5.
(set-logic QF_LIA)

(define-fun imin ((left Int) (right Int)) Int
  (ite (<= left right) left right))
(define-fun imax ((left Int) (right Int)) Int
  (ite (<= left right) right left))
(define-fun min3 ((left Int) (middle Int) (right Int)) Int
  (imin left (imin middle right)))
(define-fun additive-cell
  ((diagonal Int) (above Int) (left Int)
   (substitution Int) (deletion Int) (insertion Int)) Int
  (min3 (+ diagonal substitution)
        (+ above deletion)
        (+ left insertion)))
(define-fun bottleneck-cell
  ((diagonal Int) (above Int) (left Int) (link Int)) Int
  (imax link (min3 diagonal above left)))

; An epsilon-dominance witness simulates every residual suffix.
(declare-const dominator-cost Int)
(declare-const reach-cost Int)
(declare-const dominated-cost Int)
(declare-const suffix-cost Int)
(push)
(assert (>= dominator-cost 0))
(assert (>= reach-cost 0))
(assert (>= dominated-cost 0))
(assert (>= suffix-cost 0))
(assert (<= (+ dominator-cost reach-cost) dominated-cost))
(assert (> (+ dominator-cost reach-cost suffix-cost)
           (+ dominated-cost suffix-cost)))
(check-sat)
(pop)

; A witness edge is accepted only when replaying its recomputed local cost
; equals the next dynamic-programming value.
(declare-const replayed-prefix Int)
(declare-const witness-edge-cost Int)
(declare-const witness-next-cell Int)
(push)
(assert (>= replayed-prefix 0))
(assert (>= witness-edge-cost 0))
(assert (= witness-next-cell (+ replayed-prefix witness-edge-cost)))
(assert (not (= (+ replayed-prefix witness-edge-cost) witness-next-cell)))
(check-sat)
(pop)

; A monotone reverse grid step strictly decreases the i+j termination measure.
(declare-const trace-query Int)
(declare-const trace-target Int)
(declare-const previous-query Int)
(declare-const previous-target Int)
(push)
(assert (>= previous-query 0))
(assert (>= previous-target 0))
(assert (<= previous-query trace-query))
(assert (<= previous-target trace-target))
(assert (or (< previous-query trace-query)
            (< previous-target trace-target)))
(assert (>= (+ previous-query previous-target)
            (+ trace-query trace-target)))
(check-sat)
(pop)

; A rejected witness reservation commits zero bytes.
(declare-const witness-limit Int)
(declare-const witness-requested Int)
(push)
(assert (>= witness-limit 0))
(assert (> witness-requested witness-limit))
(assert (not (= (ite (<= witness-requested witness-limit)
                     witness-requested
                     0)
                0)))
(check-sat)
(pop)

; Snapshot acceptance requires both content identity and semantic
; configuration identity; either mismatch fails closed.
(declare-const snapshot-checksum-ok Bool)
(declare-const snapshot-config-ok Bool)
(declare-const snapshot-accepted Bool)
(push)
(assert (= snapshot-accepted (and snapshot-checksum-ok snapshot-config-ok)))
(assert (not snapshot-checksum-ok))
(assert snapshot-accepted)
(check-sat)
(pop)

(push)
(assert (= snapshot-accepted (and snapshot-checksum-ok snapshot-config-ok)))
(assert (not snapshot-config-ok))
(assert snapshot-accepted)
(check-sat)
(pop)

; Rolling retained samples are bounded by the preregistered window, not the
; number of stream samples already consumed.
(declare-const rolling-retained Int)
(declare-const rolling-window Int)
(declare-const rolling-consumed Int)
(push)
(assert (>= rolling-retained 0))
(assert (>= rolling-window 0))
(assert (>= rolling-consumed 0))
(assert (<= rolling-retained rolling-window))
(assert (> rolling-retained rolling-window))
(check-sat)
(pop)

; Soft-DTW's partition includes every strictly positive path contribution;
; min-antichain deletion would change the analysis value.
(declare-const retained-partition Int)
(declare-const positive-contribution Int)
(push)
(assert (>= retained-partition 0))
(assert (> positive-contribution 0))
(assert (<= (+ retained-partition positive-contribution) retained-partition))
(check-sat)
(pop)

; Exact equality with an immediate non-negative vertical epsilon extension is
; a concrete dominance witness, never a heuristic comparison.
(declare-const vertical-predecessor-cost Int)
(declare-const vertical-edge-cost Int)
(declare-const vertical-cell-cost Int)
(push)
(assert (>= vertical-predecessor-cost 0))
(assert (>= vertical-edge-cost 0))
(assert (= vertical-cell-cost (+ vertical-predecessor-cost vertical-edge-cost)))
(assert (> (+ vertical-predecessor-cost vertical-edge-cost) vertical-cell-cost))
(check-sat)
(pop)

; A previous active row seeds itself and its immediate diagonal successor;
; monotone vertical closure covers every later row in the same run.
(declare-const seed-row Int)
(declare-const closure-row Int)
(push)
(assert (>= seed-row 0))
(assert (>= closure-row (+ seed-row 1)))
(assert (< closure-row (+ seed-row 1)))
(check-sat)
(pop)

(push)
(assert (>= closure-row seed-row))
(assert (< (+ closure-row 1) seed-row))
(check-sat)
(pop)

; Sparse row work is charged before evaluation.  An admitted next row cannot
; exceed the configured limit.
(declare-const sparse-used Int)
(declare-const sparse-limit Int)
(push)
(assert (>= sparse-used 0))
(assert (>= sparse-limit 0))
(assert (< sparse-used sparse-limit))
(assert (> (+ sparse-used 1) sparse-limit))
(check-sat)
(pop)

; Once the budget is exhausted, admitting another row is impossible; the
; executable machine pauses before mutating the next generation.
(push)
(assert (>= sparse-used sparse-limit))
(assert (<= (+ sparse-used 1) sparse-limit))
(check-sat)
(pop)

; A paired DFS push cannot break the live-frame/product-state bijection.
(declare-const live-frames Int)
(declare-const live-states Int)
(push)
(assert (>= live-frames 0))
(assert (>= live-states 0))
(assert (= live-frames live-states))
(assert (not (= (+ live-frames 1) (+ live-states 1))))
(check-sat)
(pop)

; A paired nonempty pop cannot break the same bijection.
(push)
(assert (> live-frames 0))
(assert (> live-states 0))
(assert (= live-frames live-states))
(assert (not (= (- live-frames 1) (- live-states 1))))
(check-sat)
(pop)

; Rejecting a prospective child above the scratch ceiling retains the current
; byte count rather than committing the child.
(declare-const scratch-limit Int)
(declare-const current-bytes Int)
(declare-const prospective-bytes Int)
(push)
(assert (>= scratch-limit 0))
(assert (>= current-bytes 0))
(assert (> prospective-bytes scratch-limit))
(assert (not (= (ite (<= prospective-bytes scratch-limit)
                     prospective-bytes
                     current-bytes)
                current-bytes)))
(check-sat)
(pop)

; An interval-relaxed additive recurrence cannot exceed its concrete cell.
(declare-const ad Int)
(declare-const aa Int)
(declare-const al Int)
(declare-const cd Int)
(declare-const ca Int)
(declare-const cl Int)
(declare-const asub Int)
(declare-const adel Int)
(declare-const ains Int)
(declare-const csub Int)
(declare-const cdel Int)
(declare-const cins Int)
(push)
(assert (>= ad 0)) (assert (>= aa 0)) (assert (>= al 0))
(assert (>= asub 0)) (assert (>= adel 0)) (assert (>= ains 0))
(assert (<= ad cd)) (assert (<= aa ca)) (assert (<= al cl))
(assert (<= asub csub)) (assert (<= adel cdel)) (assert (<= ains cins))
(assert (> (additive-cell ad aa al asub adel ains)
           (additive-cell cd ca cl csub cdel cins)))
(check-sat)
(pop)

; The bottleneck recurrence is monotone in every predecessor and local link.
(declare-const abstract-link Int)
(declare-const concrete-link Int)
(push)
(assert (>= ad 0)) (assert (>= aa 0)) (assert (>= al 0))
(assert (>= abstract-link 0))
(assert (<= ad cd)) (assert (<= aa ca)) (assert (<= al cl))
(assert (<= abstract-link concrete-link))
(assert (> (bottleneck-cell ad aa al abstract-link)
           (bottleneck-cell cd ca cl concrete-link)))
(check-sat)
(pop)

; Two live generations plus a bounded cache are independent of prefix length.
(declare-const current Int)
(declare-const next Int)
(declare-const cache Int)
(declare-const frontier-limit Int)
(declare-const cache-limit Int)
(declare-const consumed-prefix Int)
(push)
(assert (>= consumed-prefix 0))
(assert (>= current 0)) (assert (>= next 0)) (assert (>= cache 0))
(assert (>= frontier-limit 0)) (assert (>= cache-limit 0))
(assert (<= current frontier-limit))
(assert (<= next frontier-limit))
(assert (<= cache cache-limit))
(assert (> (+ current next cache) (+ (* 2 frontier-limit) cache-limit)))
(check-sat)
(pop)

; Complete is constructed exactly when the traversal reports exhaustion.
(declare-const exhausted Bool)
(declare-const complete Bool)
(push)
(assert (= complete exhausted))
(assert complete)
(assert (not exhausted))
(check-sat)
(pop)

; Sparse transition cells cannot outnumber observed source/class pairs.
(declare-const distinct-observed Int)
(declare-const observed Int)
(push)
(assert (>= distinct-observed 0))
(assert (<= distinct-observed observed))
(assert (> distinct-observed observed))
(check-sat)
(pop)
