function temporal_bound_algorithm(kind::Symbol)::UInt32
    kind === :erp_gap_mass && return 1
    kind === :frechet_endpoints && return 2
    kind === :frechet_hausdorff && return 3
    kind === :frechet_candidate && return 4
    kind === :keogh && return 5
    throw(ArgumentError("unknown temporal lower bound $kind"))
end

function require_temporal_bounds(operation::Symbol)
    api_revision() >= UInt32(13) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), operation,
            "temporal lower bounds require native API revision 13"))
end

"""Evaluate an admissible native temporal lower bound with hard limits.

`kind` is `:erp_gap_mass`, `:frechet_endpoints`,
`:frechet_hausdorff`, `:frechet_candidate`, or `:keogh`. ERP uses
`parameter0` as its finite gap; Keogh uses the explicit band. A finite
result is a bound, not the exact temporal distance. No alignment and
incomplete arithmetic remain distinct tagged outcomes.
"""
function temporal_lower_bound(kind::Symbol, left::AbstractVector{<:Real},
    right::AbstractVector{<:Real}; parameter0::Real=0.0, band::Integer=0,
    limits::TemporalLimits=TemporalLimits())
    require_temporal_bounds(:llev_temporal_lower_bound)
    algorithm = temporal_bound_algorithm(kind)
    length(left) <= limits.max_series_len ||
        throw(ArgumentError("left series exceeds max_series_len"))
    length(right) <= limits.max_series_len ||
        throw(ArgumentError("right series exceeds max_series_len"))
    x = Vector{Float64}(left)
    y = Vector{Float64}(right)
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = GC.@preserve x y ccall(native(:llev_temporal_lower_bound),
        Cint, (Ptr{Float64}, Csize_t, Ptr{Float64}, Csize_t,
            UInt32, Float64, Csize_t, Ref{TemporalLimits},
            Ref{RawTemporalDistanceResult}),
        isempty(x) ? C_NULL : pointer(x), length(x),
        isempty(y) ? C_NULL : pointer(y), length(y),
        algorithm, Float64(parameter0), checked_threshold(band),
        Ref(limits), output)
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_temporal_lower_bound)
    temporal_outcome(raw)
end

"""Native ERP lower bound from the difference in total gap mass."""
erp_gap_mass_lower_bound(left::AbstractVector{<:Real},
    right::AbstractVector{<:Real}, gap::Real;
    limits::TemporalLimits=TemporalLimits()) =
    temporal_lower_bound(:erp_gap_mass, left, right;
        parameter0=gap, limits)

"""Native Fréchet lower bound from the mandatory endpoint links."""
frechet_endpoint_lower_bound(left::AbstractVector{<:Real},
    right::AbstractVector{<:Real};
    limits::TemporalLimits=TemporalLimits()) =
    temporal_lower_bound(:frechet_endpoints, left, right; limits)

"""Native one-sided Hausdorff bound from the first path to the second."""
frechet_one_sided_hausdorff_lower_bound(left::AbstractVector{<:Real},
    right::AbstractVector{<:Real};
    limits::TemporalLimits=TemporalLimits()) =
    temporal_lower_bound(:frechet_hausdorff, left, right; limits)

"""Maximum of the native endpoint and one-sided Hausdorff bounds."""
frechet_candidate_lower_bound(left::AbstractVector{<:Real},
    right::AbstractVector{<:Real};
    limits::TemporalLimits=TemporalLimits()) =
    temporal_lower_bound(:frechet_candidate, left, right; limits)

"""Native root-distance Keogh bound with a fresh centered envelope."""
lb_keogh(left::AbstractVector{<:Real}, right::AbstractVector{<:Real},
    band::Integer; limits::TemporalLimits=TemporalLimits()) =
    temporal_lower_bound(:keogh, left, right; band, limits)

"""Native TWED length-only bound for two sample counts."""
function twed_length_lower_bound(left_len::Integer, right_len::Integer,
    gap_penalty::Real; limits::TemporalLimits=TemporalLimits())
    require_temporal_bounds(:llev_twed_length_lower_bound)
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = ccall(native(:llev_twed_length_lower_bound), Cint,
        (Csize_t, Csize_t, Float64, Ref{TemporalLimits},
            Ref{RawTemporalDistanceResult}),
        checked_threshold(left_len), checked_threshold(right_len),
        Float64(gap_penalty), Ref(limits), output)
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_twed_length_lower_bound)
    temporal_outcome(raw)
end

"""Reusable native Keogh envelope over one finite nonempty query."""
mutable struct KeoghPlan
    handle::Ptr{Cvoid}
    query_len::Int
    closed::Bool
end

function keogh_envelopes(query::AbstractVector{<:Real}, band::Integer;
    limits::TemporalLimits=TemporalLimits())
    require_temporal_bounds(:llev_keogh_plan_new)
    length(query) <= limits.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve values ccall(native(:llev_keogh_plan_new), Cint,
        (Ptr{Float64}, Csize_t, Csize_t, Ref{TemporalLimits},
            Ref{Ptr{Cvoid}}),
        isempty(values) ? C_NULL : pointer(values), length(values),
        checked_threshold(band), Ref(limits), output)
    checked(status, :llev_keogh_plan_new)
    plan = KeoghPlan(output[], length(values), false)
    finalizer(close!, plan)
    plan
end

Base.length(plan::KeoghPlan) = plan.query_len

"""Return an inclusive Keogh envelope interval at a 1-based target index."""
function bounds_at(plan::KeoghPlan, target_index::Integer)
    plan.closed && throw(ArgumentError("Keogh plan is closed"))
    target_index >= 1 || throw(ArgumentError("target_index must be positive"))
    index = checked_threshold(target_index - 1)
    has = Ref{UInt8}(0)
    low = Ref{Float64}(0)
    high = Ref{Float64}(0)
    status = ccall(native(:llev_keogh_plan_bounds_at), Cint,
        (Ptr{Cvoid}, Csize_t, Ref{UInt8}, Ref{Float64}, Ref{Float64}),
        plan.handle, index, has, low, high)
    checked(status, :llev_keogh_plan_bounds_at)
    has[] == 0 ? nothing : (low[], high[])
end

function score_keogh_plan(plan::KeoghPlan,
    candidate::AbstractVector{<:Real}, squared::Bool;
    limits::TemporalLimits=TemporalLimits())
    plan.closed && throw(ArgumentError("Keogh plan is closed"))
    length(candidate) <= limits.max_series_len ||
        throw(ArgumentError("candidate exceeds max_series_len"))
    values = Vector{Float64}(candidate)
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = GC.@preserve values ccall(native(:llev_keogh_plan_score), Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, UInt8,
            Ref{TemporalLimits}, Ref{RawTemporalDistanceResult}),
        plan.handle, isempty(values) ? C_NULL : pointer(values),
        length(values), UInt8(squared), Ref(limits), output)
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_keogh_plan_score)
    temporal_outcome(raw)
end

"""Score a candidate with a retained native Keogh envelope in root units."""
lb_keogh(candidate::AbstractVector{<:Real}, plan::KeoghPlan;
    limits::TemporalLimits=TemporalLimits()) =
    score_keogh_plan(plan, candidate, false; limits)

"""Score a candidate with a retained native Keogh envelope in squared units."""
lb_keogh_squared(candidate::AbstractVector{<:Real}, plan::KeoghPlan;
    limits::TemporalLimits=TemporalLimits()) =
    score_keogh_plan(plan, candidate, true; limits)

function close!(plan::KeoghPlan)
    plan.closed && return nothing
    handle = plan.handle
    plan.handle = C_NULL
    plan.closed = true
    handle == C_NULL ||
        ccall(native(:llev_keogh_plan_free), Cvoid, (Ptr{Cvoid},), handle)
    nothing
end
Base.close(plan::KeoghPlan) = close!(plan)
Base.isopen(plan::KeoghPlan) = !plan.closed
