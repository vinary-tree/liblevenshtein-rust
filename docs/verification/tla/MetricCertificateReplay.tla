--------------------- MODULE MetricCertificateReplay ---------------------
EXTENDS Sequences, FiniteSets

\* Finite abstraction of the CBC certificate acceptance boundary. The
\* expression and scope alphabets are deliberately small; universal replay
\* soundness belongs to CertifiedContracts.v, not to TLC exploration.
CONSTANT BypassScopeCheck

Requested == [query |-> "q", parameters |-> "p", arithmetic |-> "nat",
              revision |-> "r1", observation |-> "score"]
ChangedRevision == [Requested EXCEPT !.revision = "r2"]
ChangedArithmetic == [Requested EXCEPT !.arithmetic = "rounded"]
WitnessRequested == [Requested EXCEPT !.observation = "witness"]

ValidStep == [rule |-> "add_zero_right", source |-> "x_plus_zero",
              target |-> "x", premise |-> TRUE]
MissingPremiseStep == [ValidStep EXCEPT !.premise = FALSE]
WrongSourceStep == [ValidStep EXCEPT !.source = "another_term"]
FabricatedStep == [ValidStep EXCEPT !.target = "zero"]
UnknownRuleStep == [ValidStep EXCEPT !.rule = "unsupported"]

RuleTarget(step) ==
    IF step.rule = "add_zero_right" /\ step.source = "x_plus_zero"
       /\ step.premise
    THEN "x"
    ELSE "invalid"

RECURSIVE Replay(_,_)
Replay(source, chain) ==
    IF Len(chain) = 0 THEN source
    ELSE LET step == Head(chain)
         IN IF source = step.source /\ RuleTarget(step) = step.target
            THEN Replay(step.target, Tail(chain))
            ELSE "invalid"

BaseExact == [kind |-> "exact", scope |-> Requested,
              source |-> "x_plus_zero", target |-> "x",
              steps |-> <<ValidStep>>, witness |-> FALSE]
BaseLower == [kind |-> "lower", scope |-> Requested,
              source |-> "x_plus_zero", target |-> "zero",
              steps |-> <<>>, witness |-> FALSE,
              lowerRule |-> "zero_floor"]

GoodExact == [certificate |-> BaseExact, requested |-> Requested,
              expected |-> TRUE]
GoodLower == [certificate |-> BaseLower, requested |-> Requested,
              expected |-> FALSE]
StaleRevision == [GoodExact EXCEPT !.certificate.scope = ChangedRevision,
                                  !.expected = FALSE]
WrongArithmetic == [GoodExact EXCEPT !.certificate.scope = ChangedArithmetic,
                                    !.expected = FALSE]
MissingWitness == [GoodExact EXCEPT !.requested = WitnessRequested,
                                   !.expected = FALSE]
ValidWitness == [GoodExact EXCEPT !.requested = WitnessRequested,
                                 !.certificate.scope = WitnessRequested,
                                 !.certificate.witness = TRUE]
WrongSource == [GoodExact EXCEPT !.certificate.steps = <<WrongSourceStep>>,
                               !.expected = FALSE]
WrongTarget == [GoodExact EXCEPT !.certificate.steps = <<FabricatedStep>>,
                               !.expected = FALSE]
MissingPremise == [GoodExact EXCEPT !.certificate.steps = <<MissingPremiseStep>>,
                                  !.expected = FALSE]
UnknownRule == [GoodExact EXCEPT !.certificate.steps = <<UnknownRuleStep>>,
                               !.expected = FALSE]
TruncatedChain == [GoodExact EXCEPT !.certificate.steps = <<>>,
                                  !.expected = FALSE]
ChangedClaim == [GoodExact EXCEPT !.certificate.target = "zero",
                               !.expected = FALSE]
ReversedLower == [BaseLower EXCEPT !.source = "zero",
                                   !.target = "x_plus_zero"]
UnsupportedLower == [BaseLower EXCEPT !.lowerRule = "fabricated"]

Cases == {GoodExact, GoodLower, StaleRevision, WrongArithmetic,
          MissingWitness, ValidWitness, WrongSource, WrongTarget,
          MissingPremise, UnknownRule, TruncatedChain, ChangedClaim}

ScopeAccepted(requested, certificate) ==
    BypassScopeCheck \/ requested = certificate.scope

AcceptExact(scenario) ==
    LET cert == scenario.certificate
    IN cert.kind = "exact"
       /\ ScopeAccepted(scenario.requested, cert)
       /\ (scenario.requested.observation # "witness" \/ cert.witness)
       /\ Replay(cert.source, cert.steps) = cert.target

AcceptLower(certificate) ==
    certificate.kind = "lower"
    /\ ScopeAccepted(Requested, certificate)
    /\ certificate.lowerRule = "zero_floor"
    /\ certificate.target = "zero"
    /\ certificate.source = "x_plus_zero"

VARIABLE selected
Init == selected \in Cases
Next == UNCHANGED selected
Spec == Init /\ [][Next]_selected

CertificateClassification == AcceptExact(selected) = selected.expected
ExactAndLowerAreDistinct == ~AcceptExact(GoodLower)
LowerRuleDirectionIsChecked ==
    AcceptLower(BaseLower)
    /\ ~AcceptLower(ReversedLower)
    /\ ~AcceptLower(UnsupportedLower)
ScopeMutationDetected == ~AcceptExact(StaleRevision)
PositiveControlsAccepted == AcceptExact(GoodExact) /\ AcceptExact(ValidWitness)

=============================================================================
