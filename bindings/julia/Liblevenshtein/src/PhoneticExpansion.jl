"""Explicit work/output ceilings for native reverse-phonetic expansion."""
struct PhoneticExpansionLimits
    max_input_scalars::Csize_t
    max_rules::Csize_t
    max_rule_units::Csize_t
    max_nodes::Csize_t
    max_output_bytes::Csize_t
end

function PhoneticExpansionLimits(;
    max_input_scalars::Integer=256,
    max_rules::Integer=256,
    max_rule_units::Integer=4_096,
    max_nodes::Integer=10_000,
    max_output_bytes::Integer=1_000_000)
    values = (max_input_scalars, max_rules, max_rule_units,
        max_nodes, max_output_bytes)
    all(value -> value > 0, values) ||
        throw(ArgumentError("all phonetic expansion ceilings must be positive"))
    PhoneticExpansionLimits(
        checked_csize(max_input_scalars, "max_input_scalars"),
        checked_csize(max_rules, "max_rules"),
        checked_csize(max_rule_units, "max_rule_units"),
        checked_csize(max_nodes, "max_nodes"),
        checked_csize(max_output_bytes, "max_output_bytes"))
end

"""Expand every reverse-phonetic segmentation into one native regex pattern.

The result is exact when within the supplied work/output ceilings. A limit
error does not return a truncated or partially valid pattern.
"""
function expand_phonetic_alternatives(input::AbstractString;
    rules::Union{Nothing,PhoneticRuleSet}=nothing,
    limits::PhoneticExpansionLimits=PhoneticExpansionLimits())
    bytes = text_bytes(input)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    output = Ref(RawOwnedString(C_NULL, 0))
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_expand), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, Ref{PhoneticExpansionLimits},
            Ref{RawOwnedString}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), rules_handle,
        Ref(limits), output), :llev_phonetic_expand)
    copy_and_free_owned(output)
end

"""Return `(pattern, max_cost)` from native greedy reverse expansion.

This is a different segmentation algorithm from exhaustive expansion. Its
reported cost is the sum of maximum rule costs selected by native Rust.
"""
function expand_phonetic_with_costs(input::AbstractString;
    rules::Union{Nothing,PhoneticRuleSet}=nothing,
    limits::PhoneticExpansionLimits=PhoneticExpansionLimits())
    bytes = text_bytes(input)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    output = Ref(RawOwnedString(C_NULL, 0))
    cost = Ref{Cdouble}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_expand_with_costs), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, Ref{PhoneticExpansionLimits},
            Ref{RawOwnedString}, Ref{Cdouble}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), rules_handle,
        Ref(limits), output, cost), :llev_phonetic_expand_with_costs)
    (pattern=copy_and_free_owned(output), max_cost=Float64(cost[]))
end
