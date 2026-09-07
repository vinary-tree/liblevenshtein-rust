# Standalone generalized and universal automata

API revision 5 exposes the native generalized and universal edit automata as
owned C handles. This is the common substrate for foreign-language facades
that need custom edit grammars, exact prefix observations, universal
Levenshtein variants, or directional zero-cost substitutions without copying
either algorithm into the host language.

This surface evaluates one source against one complete or incrementally
supplied target. Dictionary-product traversal remains a separate capability:
language facades must compose the online machine with a provider walk rather
than materialize and filter every dictionary term.

## Contract at a glance

| Capability | Generalized automaton | Universal automaton |
|---|---|---|
| Unit domains | Unicode scalars | bytes, Unicode scalars, u64 tokens |
| Configuration | runtime operation set | compile-time-specialized variant plus optional owned policy |
| Costs | exact fixed-point scaling of accepted decimal `double` values | integral edit budget plus zero-cost directional equivalences |
| Prefix state | finite-lookback row ring | canonical universal antichain |
| Empty frontier | current row may later revive | permanent death |
| Retained target history | bounded by maximum target consumption | none beyond canonical state |
| Concurrency | immutable configuration is shareable; each online handle is exclusive | same |

All constructors copy their borrowed descriptors. A successful configuration
handle therefore outlives the caller's operation names, restriction strings,
and equivalence arrays. Complete evaluation creates a temporary online state;
the explicit online API exposes the identical native transition kernel.

## Generalized operation sets

Each `LlevGeneralizedOperation` declares source and target consumption,
non-negative finite weight, a diagnostic UTF-8 name, and one applicability
predicate:

| Applicability | Meaning |
|---|---|
| `LLEV_OPERATION_APPLICABILITY_ANY` | apply without comparing the consumed slices |
| `LLEV_OPERATION_APPLICABILITY_EQUAL` | apply only when both consumed slices are equal |
| `LLEV_OPERATION_APPLICABILITY_ADJACENT_TRANSPOSE` | apply only to a two-scalar adjacent transposition |
| `LLEV_OPERATION_APPLICABILITY_LISTED` | apply only to an explicitly listed directional source/target string pair |

Listed strings must contain exactly the scalars declared by their operation's
consumption pair. Direction is significant. For example, listing source
`"ph"` and target `"f"` permits that rewrite but does not implicitly permit
`"f"` to `"ph"`.

Construction fails before publishing a handle when an operation consumes
nothing, a zero-cost operation changes length, a weight is negative or cannot
be represented by the exact shared decimal scale, a name is empty or too
large, a listed restriction has the wrong arity, or aggregate resource limits
are exceeded. Consumers retrieve the precise diagnostic through
`llev_last_error_message()`.

If the shared denominator is $`d`$, an observation with numerator $`p`$
represents exact cost $`p/d`$. `has_distance` distinguishes a present zero
from the placeholder zero written when the complete source is outside the
budget. The budget supplied to the constructor is scaled by the same
denominator.

### Prefix liveness is deliberately non-monotone

`current_row_nonempty` describes only the exact target generation just
committed. It is not named `alive` because a multi-target operation can retain
an older row, observe an empty intermediate row, and reach a later row. A
consumer must continue a generalized prefix traversal after zero unless its
own independent bound proves that no configured operation can bridge the
gap.

The native regression uses a zero-cost `Equal(2, 2)` operation: source `ab`
has no current cell after target prefix `a`, then becomes accepting after `b`.
This distinguishes an exact observation from an unsound pruning signal.

## Universal variants and policies

`LlevUniversalVariant` selects one native specialization:

- `LLEV_UNIVERSAL_STANDARD` supports insert, delete, and substitute;
- `LLEV_UNIVERSAL_TRANSPOSITION` additionally supports adjacent
  transposition;
- `LLEV_UNIVERSAL_MERGE_AND_SPLIT` additionally supports symmetric one-to-two
  and two-to-one edits.

Set `policy_unit_domain` and `equivalence_count` to zero for the unrestricted
specialization. That route retains the zero-sized native policy and does not
introduce dynamic dispatch. A non-empty equivalence array binds the automaton
to `VT_UNIT_DOMAIN_BYTE`, `VT_UNIT_DOMAIN_UNICODE_SCALAR`, or
`VT_UNIT_DOMAIN_U64`; later evaluation in another domain returns
`LLEV_STATUS_DOMAIN_MISMATCH`.

An equivalence `(source, target)` permits that dictionary/source unit to match
that query/target unit at zero cost. It does not add the reverse direction.
Byte values must fit in `uint8_t`; Unicode values must be valid scalar values;
u64 values are not narrowed. The byte and Unicode policies use the native
owned substitution sets. The u64 policy stores one sorted, deduplicated owned
array and uses binary search, avoiding a hash table and nondeterministic
iteration state for a read-mostly configuration.

Unlike generalized current-row emptiness, `LlevUniversalObservation.alive ==
0` is permanent. Later advances still account for committed target units but
cannot recreate a canonical frontier.

## Limits and transactional behavior

Passing a null `LlevAutomatonLimits*` selects:

| Limit | Default | Applies to |
|---|---:|---|
| `max_source_units` | 1,000,000 | source binding and complete evaluation |
| `max_target_units` | 1,000,000 | complete evaluation and online advance |
| `max_retained_cells` | 1,000,000 | generalized row ring plus scratch row |
| `max_step_work_units` | 100,000,000 | generalized relaxations for one target scalar |

Explicit values are honored exactly; the ABI does not silently clamp a
caller's ceiling to the default. Length addition, work accounting, cost
scaling, and allocation sizing use checked arithmetic.

An invalid scalar, an out-of-domain byte, or a target-limit failure is detected
before the online handle mutates. A generalized transition that exceeds its
work ceiling computes in scratch storage and does not commit the failed step.
The observation output is zeroed before a fallible operation and populated
only with the last committed state on success.

## Ownership and host pattern

Configuration handles are immutable and may be shared by synchronized host
code. Online handles are exclusive state machines; callers must neither invoke
one concurrently nor free it while another call is active. Every successful
constructor transfers exactly one handle to the caller, and every such handle
must be passed to its matching free function exactly once. Null frees are
no-ops.

```c
LlevGeneralizedAutomaton* configuration = NULL;
LlevGeneralizedOnlineAutomaton* online = NULL;
LlevGeneralizedObservation observation = {0};

LlevStatus status = llev_generalized_automaton_new(
    2, operations, operation_count, &configuration);
if (status != LLEV_STATUS_OK) goto fail;

status = llev_generalized_online_new_utf8(
    configuration, source, source_len, NULL, &online);
if (status != LLEV_STATUS_OK) goto fail;

for (size_t i = 0; i < target_scalar_count; ++i) {
    status = llev_generalized_online_advance(
        online, target_scalars[i], &observation);
    if (status != LLEV_STATUS_OK) goto fail;
    /* Do not prune solely because current_row_nonempty is zero. */
}

llev_generalized_online_free(online);
llev_generalized_automaton_free(configuration);
return observation.accepting ? 0 : 1;

fail:
fprintf(stderr, "%s\n", llev_last_error_message());
llev_generalized_online_free(online);
llev_generalized_automaton_free(configuration);
return 2;
```

Idiomatic facades should pair these handles with deterministic scope cleanup:
RAII, `try`/`finally`, context managers, `do` blocks, or the closest host
equivalent. A finalizer is leak containment, not the primary lifecycle.

## Complexity and performance

For source length $`m`$, maximum target consumption $`r`$, operation count
$`o`$, and one committed target scalar, generalized retained memory is
$`\mathcal{O}(m(r+2)+r)`$. Work is bounded explicitly by
`max_step_work_units`; without that cutoff its coarse upper bound is
$`\mathcal{O}(mo)`$ per reachable row relaxation. Storage is retained and
reused across advances.

The universal machine retains the source plus its canonical state. Its
transition path stays monomorphized for variant, policy, and unit type. Policy
construction is $`\mathcal{O}(e \log e)`$ for $`e`$ u64 equivalences, followed
by $`\mathcal{O}(\log e)`$ policy lookup; native byte and Unicode policy
representations select their existing small-set or hashed strategy. The
unrestricted path pays neither an owned-policy allocation nor a lookup-table
indirection.

Marshalling work is confined to construction and source binding. Hosts should
reuse configurations, retain online handles across a prefix stream, and pass
domain-native contiguous buffers. Reimplementing transitions in a facade or
materializing an entire dictionary for filtering defeats the purpose of this
ABI.

## Verification and evolution

The Rust ABI tests compare complete and online observations, exact fractional
costs, generalized resurrection, rollback at target limits, all three
universal variants, all three unit domains, directional policies, invalid
policy units, and domain mismatch. `bindings/api.json` is the canonical symbol
and enum inventory; `scripts/generate-bindings.py --check` pins API revision 5
across generated language constants and headers; `scripts/check-bindings.py
--check` proves model, implementation, header, and reasoned facade absences
agree.

The record layouts in revision 5 are frozen. Future additive fields require
new entry points or new explicitly versioned records rather than making an old
caller pass a larger structure. ABI generation 1 remains unchanged.
