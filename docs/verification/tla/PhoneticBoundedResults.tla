------------------------- MODULE PhoneticBoundedResults -------------------------
(***************************************************************************)
(* Bounded native phonetic result publication. The model concerns caller   *)
(* outputs, not which strings match: native Rust differential properties    *)
(* establish the semantic result count/order. Word grep has a documented   *)
(* required-capacity side channel; pointer-owned results do not.           *)
(***************************************************************************)
EXTENDS Integers, TLC

VARIABLES family, required, capacity, phase, status, payload, outCount
vars == <<family, required, capacity, phase, status, payload, outCount>>

Families == {"dictionary", "word", "online", "token", "expansion"}
Init ==
  /\ family \in Families
  /\ required \in 0..3
  /\ capacity \in 0..2
  /\ phase = "ready"
  /\ status = "none"
  /\ payload = -1
  /\ outCount = -1

CallSuccess ==
  /\ phase = "ready"
  /\ required <= capacity
  /\ phase' = "done"
  /\ status' = "ok"
  /\ payload' = required
  /\ outCount' = required
  /\ UNCHANGED <<family, required, capacity>>

CallLimit ==
  /\ phase = "ready"
  /\ required > capacity
  /\ phase' = "done"
  /\ status' = "limit"
  /\ payload' = -1
  /\ outCount' = IF family = "word" THEN required ELSE -1
  /\ UNCHANGED <<family, required, capacity>>

Next == CallSuccess \/ CallLimit
Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ family \in Families
  /\ required \in 0..3
  /\ capacity \in 0..2
  /\ phase \in {"ready", "done"}
  /\ status \in {"none", "ok", "limit"}
  /\ payload \in (-1)..3
  /\ outCount \in {-1} \cup (0..3)

SuccessIsComplete == status = "ok" => (payload = required /\ outCount = required)
LimitNeverPublishesPartialPayload == status = "limit" => payload = -1
WordLimitReportsRequired ==
  (status = "limit" /\ family = "word") => outCount = required
PointerLimitPreservesCount ==
  (status = "limit" /\ family # "word") => outCount = -1
NoOverCapacitySuccess == status = "ok" => required <= capacity

=============================================================================
