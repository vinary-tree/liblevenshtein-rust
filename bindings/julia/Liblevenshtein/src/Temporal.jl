"""Hard per-comparison limits for native scalar temporal metrics.

All four bounds are checked before native dynamic-programming work begins.
The defaults match the C ABI contract. Lower them for a known workload.
"""
struct TemporalLimits
    max_series_len::Csize_t
    max_dp_cells::Csize_t
    max_work_units::Csize_t
    max_scratch_bytes::Csize_t
end

TemporalLimits(; max_series_len::Integer=1_000_000,
    max_dp_cells::Integer=100_000_000,
    max_work_units::Integer=200_000_000,
    max_scratch_bytes::Integer=512 * 1024 * 1024) =
    TemporalLimits(checked_threshold(max_series_len),
        checked_threshold(max_dp_cells), checked_threshold(max_work_units),
        checked_threshold(max_scratch_bytes))

struct RawTemporalConfig
    algorithm::UInt32
    reserved::UInt32
    parameter0::Float64
    parameter1::Float64
    band::Csize_t
    cutoff::Float64
end

struct RawTemporalDistanceResult
    value::Float64
    kind::UInt32
    reason::UInt32
    dp_cells::Csize_t
    work_units::Csize_t
    scratch_bytes::Csize_t
end

"""A native temporal score or exact reason no score was produced.

`kind` is `:finite`, `:above_cutoff`, `:no_alignment`, or `:incomplete`.
Only `:finite` carries `value`. An incomplete operation has `reason` equal
to `:dp_cells`, `:work_units`, `:scratch_bytes`, or `:overflow`. The work and
storage fields report whole-operation capacity reserved by the native call.
"""
struct TemporalDistanceOutcome
    kind::Symbol
    value::Union{Nothing,Float64}
    reason::Union{Nothing,Symbol}
    dp_cells::Csize_t
    work_units::Csize_t
    scratch_bytes::Csize_t
end

function temporal_algorithm(kind::Symbol)::UInt32
    kind === :msm && return 1
    kind === :erp && return 2
    kind === :twed && return 3
    kind === :dtw && return 4
    kind === :frechet && return 5
    kind === :soft_dtw && return 6
    throw(ArgumentError("unknown temporal algorithm $kind"))
end

function temporal_outcome(raw::RawTemporalDistanceResult)
    kind = if raw.kind == 0
        :finite
    elseif raw.kind == 1
        :above_cutoff
    elseif raw.kind == 2
        :no_alignment
    elseif raw.kind == 3
        :incomplete
    else
        throw(ArgumentError("native temporal result has invalid kind $(raw.kind)"))
    end
    reason = if raw.kind != 3
        nothing
    elseif raw.reason == 1
        :dp_cells
    elseif raw.reason == 2
        :work_units
    elseif raw.reason == 3
        :scratch_bytes
    elseif raw.reason == 4
        :overflow
    else
        throw(ArgumentError("native temporal result has invalid reason $(raw.reason)"))
    end
    TemporalDistanceOutcome(kind, kind === :finite ? raw.value : nothing,
        reason, raw.dp_cells, raw.work_units, raw.scratch_bytes)
end

"""Compare two finite scalar series with one native temporal kernel.

The input arrays are borrowed only for the call. Dense `Vector{Float64}`
inputs cross directly; other real vectors are converted once to contiguous
double arrays. `cutoff` is inclusive, and positive infinity requests an exact
unthresholded result. The configured limits bound native DP work and scratch
storage. No output is published for invalid input.
"""
function temporal_distance(kind::Symbol, left::AbstractVector{<:Real},
    right::AbstractVector{<:Real}; parameter0::Real=0.0,
    parameter1::Real=0.0, band::Integer=0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits())
    api_revision() >= UInt32(11) || throw(NativeError(Int32(STATUS_UNSUPPORTED),
        :llev_temporal_distance, "temporal kernels require native API revision 11"))
    length(left) <= limits.max_series_len ||
        throw(ArgumentError("left series exceeds max_series_len"))
    length(right) <= limits.max_series_len ||
        throw(ArgumentError("right series exceeds max_series_len"))
    raw_config = RawTemporalConfig(temporal_algorithm(kind), 0,
        Float64(parameter0), Float64(parameter1), checked_threshold(band),
        Float64(cutoff))
    x = left isa Vector{Float64} ? left : Vector{Float64}(left)
    y = right isa Vector{Float64} ? right : Vector{Float64}(right)
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = GC.@preserve x y ccall(native(:llev_temporal_distance), Cint,
        (Ptr{Float64}, Csize_t, Ptr{Float64}, Csize_t,
            Ref{RawTemporalConfig}, Ref{TemporalLimits},
            Ref{RawTemporalDistanceResult}),
        isempty(x) ? C_NULL : pointer(x), length(x),
        isempty(y) ? C_NULL : pointer(y), length(y),
        Ref(raw_config), Ref(limits), output)
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_temporal_distance)
    temporal_outcome(raw)
end

"""Native move-split-merge score with explicit split/merge cost and limits."""
msm_distance(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    split_merge_cost::Real=1.0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits()) =
    temporal_distance(:msm, left, right; parameter0=split_merge_cost,
        cutoff, limits)

"""Native edit-distance-with-real-penalty score for a finite gap value."""
erp_distance(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    gap::Real=0.0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits()) =
    temporal_distance(:erp, left, right; parameter0=gap, cutoff, limits)

"""Native unit-grid TWED score with nonnegative stiffness and gap penalty."""
twed_distance(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    stiffness::Real=1.0, gap_penalty::Real=0.0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits()) =
    temporal_distance(:twed, left, right; parameter0=stiffness,
        parameter1=gap_penalty, cutoff, limits)

"""Native banded dynamic-time-warping score with explicit half-width."""
dtw_distance(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    band::Integer, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits()) =
    temporal_distance(:dtw, left, right; band, cutoff, limits)

"""Native discrete Fréchet score for two scalar paths."""
frechet_distance(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    cutoff::Real=Inf, limits::TemporalLimits=TemporalLimits()) =
    temporal_distance(:frechet, left, right; cutoff, limits)

"""Native bounded Soft-DTW loss; this value may be negative and is not a metric."""
soft_dtw_loss(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    gamma::Real=1.0, cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits()) =
    temporal_distance(:soft_dtw, left, right; parameter0=gamma,
        cutoff, limits)
