"""One finite closed coordinate interval in a typed vector box."""
struct VectorInterval
    low::Float64
    high::Float64
    function VectorInterval(low::Real, high::Real)
        lower, upper = Float64(low), Float64(high)
        isfinite(lower) && isfinite(upper) && lower <= upper ||
            throw(ArgumentError("vector interval must be finite and closed"))
        new(lower, upper)
    end
end

"""Copied vector intervals associated with an exact channel/unit layout."""
struct VectorIntervalBox
    layout::Tuple
    intervals::Vector{VectorInterval}
    function VectorIntervalBox(layout::Tuple,
        intervals::Vector{VectorInterval})
        !isempty(layout) && length(layout) == length(intervals) ||
            throw(ArgumentError("vector box layout and intervals differ"))
        new(layout, copy(intervals))
    end
end

function VectorIntervalBox(metric::FixedChannelMetric,
    bounds::AbstractVector)
    length(bounds) == metric.dimension ||
        throw(ArgumentError("vector box dimension differs from metric"))
    intervals = Vector{VectorInterval}(undef, length(bounds))
    for (index, pair) in enumerate(bounds)
        pair isa Tuple || pair isa AbstractVector ||
            throw(ArgumentError("vector box interval must be a pair"))
        length(pair) == 2 ||
            throw(ArgumentError("vector box interval must have two endpoints"))
        intervals[index] = VectorInterval(pair[1], pair[2])
    end
    VectorIntervalBox(metric.layout, intervals)
end

function vector_box_refines(fine::VectorIntervalBox,
    coarse::VectorIntervalBox)
    fine.layout == coarse.layout ||
        throw(ArgumentError("vector box channel layouts differ"))
    length(fine.intervals) == length(fine.layout) &&
        length(coarse.intervals) == length(coarse.layout) ||
        throw(ArgumentError("vector box interval count differs from layout"))
    all(isfinite(interval.low) && isfinite(interval.high) &&
        interval.low <= interval.high for interval in fine.intervals) &&
        all(isfinite(interval.low) && isfinite(interval.high) &&
        interval.low <= interval.high for interval in coarse.intervals) ||
        throw(ArgumentError("vector box contains an invalid interval"))
    all(coarse.intervals[index].low <= fine.intervals[index].low &&
        fine.intervals[index].high <= coarse.intervals[index].high
        for index in eachindex(fine.intervals))
end

"""Vector box paired with one finite closed physical-time interval."""
struct TimestampedVectorIntervalBox
    box::VectorIntervalBox
    time_low::Float64
    time_high::Float64
    unit::UInt32
    function TimestampedVectorIntervalBox(box::VectorIntervalBox,
        time_low::Real, time_high::Real, unit::UInt32)
        low, high = Float64(time_low), Float64(time_high)
        isfinite(low) && isfinite(high) && low <= high ||
            throw(ArgumentError("timestamp interval must be finite and closed"))
        1 <= unit <= 4 ||
            throw(ArgumentError("unknown timestamp unit code"))
        new(box, low, high, unit)
    end
end

function TimestampedVectorIntervalBox(box::VectorIntervalBox,
    time_low::Real, time_high::Real; unit::Symbol=:seconds)
    TimestampedVectorIntervalBox(box, time_low, time_high,
        timestamp_unit_code(unit))
end

function timestamped_vector_box_refines(
    fine::TimestampedVectorIntervalBox,
    coarse::TimestampedVectorIntervalBox)
    fine.unit == coarse.unit ||
        throw(ArgumentError("timestamp units differ"))
    vector_box_refines(fine.box, coarse.box) &&
        coarse.time_low <= fine.time_low &&
        fine.time_high <= coarse.time_high
end

struct RawTimestampedVectorBoxView
    coordinates::Ptr{VectorInterval}
    dimension::Csize_t
    time_low::Float64
    time_high::Float64
    unit::UInt32
    reserved::UInt32
end

function require_vector_bounds_revision(symbol::Symbol)
    api_revision() >= UInt32(24) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), symbol,
            "typed vector bounds require native API revision 24"))
    nothing
end

function raw_timestamped_vector_box(box::TimestampedVectorIntervalBox)
    RawTimestampedVectorBoxView(pointer(box.box.intervals),
        length(box.box.intervals), box.time_low, box.time_high,
        box.unit, 0)
end

function checked_vector_box_layout(metric::FixedChannelMetric,
    box::VectorIntervalBox)
    box.layout == metric.layout ||
        throw(ArgumentError("vector box channel layout differs from metric"))
    nothing
end

"""Native fixed-channel K1 point-to-box bound."""
function vector_point_box_lower_bound(metric::FixedChannelMetric,
    point::AbstractVector{<:Real}, box::VectorIntervalBox;
    limits::VectorTemporalLimits=VectorTemporalLimits())
    require_vector_bounds_revision(:llev_vector_point_box_lower_bound)
    checked_vector_box_layout(metric, box)
    length(point) == metric.dimension ||
        throw(ArgumentError("vector point dimension differs from metric"))
    coordinates = point isa Vector{Float64} ? point : Vector{Float64}(point)
    output = Ref{Float64}(0)
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric coordinates box ccall(
            native(:llev_vector_point_box_lower_bound), Cint,
            (Ptr{Cvoid}, Ptr{Float64}, Csize_t,
                Ptr{VectorInterval}, Csize_t,
                Ref{VectorTemporalLimits}, Ref{Float64}),
            metric.handle, pointer(coordinates), length(coordinates),
            pointer(box.intervals), length(box.intervals), Ref(limits),
            output)
    end
    checked(status, :llev_vector_point_box_lower_bound)
    output[]
end

"""Native fixed-channel K1 box-to-box bound."""
function vector_box_box_lower_bound(metric::FixedChannelMetric,
    left::VectorIntervalBox, right::VectorIntervalBox;
    limits::VectorTemporalLimits=VectorTemporalLimits())
    require_vector_bounds_revision(:llev_vector_box_box_lower_bound)
    checked_vector_box_layout(metric, left)
    checked_vector_box_layout(metric, right)
    output = Ref{Float64}(0)
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric left right ccall(
            native(:llev_vector_box_box_lower_bound), Cint,
            (Ptr{Cvoid}, Ptr{VectorInterval}, Csize_t,
                Ptr{VectorInterval}, Csize_t,
                Ref{VectorTemporalLimits}, Ref{Float64}),
            metric.handle, pointer(left.intervals), length(left.intervals),
            pointer(right.intervals), length(right.intervals),
            Ref(limits), output)
    end
    checked(status, :llev_vector_box_box_lower_bound)
    output[]
end

"""K1 ERP local match, identical to the fixed-channel point-to-box bound."""
vector_erp_interval_match_lower_bound(metric::FixedChannelMetric,
    query::AbstractVector{<:Real}, candidate::VectorIntervalBox;
    kwargs...) =
    vector_point_box_lower_bound(metric, query, candidate; kwargs...)

"""K1 ERP target-gap bound against the metric's fixed gap point."""
vector_erp_interval_gap_lower_bound(metric::FixedChannelMetric,
    gap::AbstractVector{<:Real}, candidate::VectorIntervalBox;
    kwargs...) =
    vector_point_box_lower_bound(metric, gap, candidate; kwargs...)

"""K1 squared vector DTW local-cost bound, before the public square root."""
function vector_dtw_interval_local_lower_bound_squared(
    metric::FixedChannelMetric, query::AbstractVector{<:Real},
    candidate::VectorIntervalBox; kwargs...)
    distance = vector_point_box_lower_bound(metric, query, candidate;
        kwargs...)
    distance * distance
end

"""K1 vector Fréchet bottleneck-link lower bound."""
vector_frechet_interval_link_lower_bound(metric::FixedChannelMetric,
    query::AbstractVector{<:Real}, candidate::VectorIntervalBox;
    kwargs...) =
    vector_point_box_lower_bound(metric, query, candidate; kwargs...)

"""Native K4 candidate bound with exact metric-domain validation.

ERP returns the gap-mass difference; Fréchet returns the endpoint maximum.
The current native DTW and timestamped TWED K4 bounds are zero after
validating their complete candidates.
"""
function vector_candidate_lower_bound(kind::Symbol,
    metric::FixedChannelMetric, left::VectorTemporalSeries,
    right::VectorTemporalSeries;
    gap_or_sentinel::Union{Nothing,AbstractVector{<:Real}}=nothing,
    stiffness::Real=0.0, gap_penalty::Real=0.0, band::Integer=0,
    limits::VectorTemporalLimits=VectorTemporalLimits())
    require_vector_bounds_revision(
        :llev_vector_temporal_candidate_lower_bound)
    code = kind === :erp ? UInt32(2) :
        kind === :dtw ? UInt32(4) :
        kind === :frechet ? UInt32(5) :
        kind === :timestamped_twed ? UInt32(7) :
        throw(ArgumentError("unsupported vector candidate-bound algorithm"))
    size(left.samples, 1) == metric.dimension &&
        size(right.samples, 1) == metric.dimension ||
        throw(ArgumentError("vector candidate dimension differs from metric"))
    point = gap_or_sentinel === nothing ? Float64[] :
        Vector{Float64}(gap_or_sentinel)
    code in (UInt32(2), UInt32(7)) &&
        length(point) != metric.dimension &&
        throw(ArgumentError("gap or sentinel must match metric dimension"))
    code == UInt32(7) &&
        (left.timestamps === nothing || right.timestamps === nothing) &&
        throw(ArgumentError("timestamped TWED candidates need timestamps"))
    config = RawVectorTemporalConfig(code, 0,
        isempty(point) ? C_NULL : pointer(point), Float64(stiffness),
        Float64(gap_penalty), checked_threshold(band), Inf)
    output = Ref{Float64}(0)
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric left right point ccall(
            native(:llev_vector_temporal_candidate_lower_bound), Cint,
            (Ptr{Cvoid}, Ref{RawVectorSeriesView},
                Ref{RawVectorSeriesView}, Ref{RawVectorTemporalConfig},
                Ref{VectorTemporalLimits}, Ref{Float64}),
            metric.handle, Ref(raw_vector_series(left)),
            Ref(raw_vector_series(right)), Ref(config), Ref(limits), output)
    end
    checked(status, :llev_vector_temporal_candidate_lower_bound)
    output[]
end

vector_erp_candidate_lower_bound(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    gap::AbstractVector{<:Real}, kwargs...) =
    vector_candidate_lower_bound(:erp, metric, left, right;
        gap_or_sentinel=gap, kwargs...)

vector_dtw_candidate_lower_bound(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    band::Integer, kwargs...) =
    vector_candidate_lower_bound(:dtw, metric, left, right; band, kwargs...)

vector_frechet_candidate_lower_bound(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries; kwargs...) =
    vector_candidate_lower_bound(:frechet, metric, left, right; kwargs...)

vector_timestamped_twed_candidate_lower_bound(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    sentinel::AbstractVector{<:Real}, stiffness::Real,
    gap_penalty::Real=0.0, kwargs...) =
    vector_candidate_lower_bound(:timestamped_twed, metric, left, right;
        gap_or_sentinel=sentinel, stiffness, gap_penalty, kwargs...)

function raw_vector_twed_bound_config(sentinel::Vector{Float64},
    stiffness::Real, gap_penalty::Real)
    RawVectorTemporalConfig(7, 0, pointer(sentinel),
        Float64(stiffness), Float64(gap_penalty), 0, Inf)
end

function checked_timestamped_box_layout(metric::FixedChannelMetric,
    box::TimestampedVectorIntervalBox)
    checked_vector_box_layout(metric, box.box)
end

"""Native K1 delete bound between consecutive timestamped vector boxes."""
function vector_twed_interval_delete_lower_bound(
    metric::FixedChannelMetric,
    current::TimestampedVectorIntervalBox,
    previous::TimestampedVectorIntervalBox;
    sentinel::AbstractVector{<:Real}, stiffness::Real,
    gap_penalty::Real=0.0,
    limits::VectorTemporalLimits=VectorTemporalLimits())
    require_vector_bounds_revision(
        :llev_vector_twed_interval_lower_bound)
    checked_timestamped_box_layout(metric, current)
    checked_timestamped_box_layout(metric, previous)
    length(sentinel) == metric.dimension ||
        throw(ArgumentError("TWED sentinel dimension differs from metric"))
    copied = Vector{Float64}(sentinel)
    config = raw_vector_twed_bound_config(copied, stiffness, gap_penalty)
    output = Ref{Float64}(0)
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric current previous copied ccall(
            native(:llev_vector_twed_interval_lower_bound), Cint,
            (Ptr{Cvoid}, UInt32, Ptr{Float64}, Ptr{Float64},
                Csize_t, Float64, Float64,
                Ref{RawTimestampedVectorBoxView},
                Ref{RawTimestampedVectorBoxView},
                Ref{RawVectorTemporalConfig},
                Ref{VectorTemporalLimits}, Ref{Float64}),
            metric.handle, UInt32(1), C_NULL, C_NULL, 0, 0.0, 0.0,
            Ref(raw_timestamped_vector_box(current)),
            Ref(raw_timestamped_vector_box(previous)),
            Ref(config), Ref(limits), output)
    end
    checked(status, :llev_vector_twed_interval_lower_bound)
    output[]
end

"""Native K1 match bound for two exact query points and candidate boxes."""
function vector_twed_interval_match_lower_bound(
    metric::FixedChannelMetric,
    query_current::AbstractVector{<:Real},
    query_previous::AbstractVector{<:Real},
    query_current_time::Real, query_previous_time::Real,
    candidate_current::TimestampedVectorIntervalBox,
    candidate_previous::TimestampedVectorIntervalBox;
    sentinel::AbstractVector{<:Real}, stiffness::Real,
    gap_penalty::Real=0.0,
    limits::VectorTemporalLimits=VectorTemporalLimits())
    require_vector_bounds_revision(
        :llev_vector_twed_interval_lower_bound)
    checked_timestamped_box_layout(metric, candidate_current)
    checked_timestamped_box_layout(metric, candidate_previous)
    length(query_current) == metric.dimension &&
        length(query_previous) == metric.dimension &&
        length(sentinel) == metric.dimension ||
        throw(ArgumentError("TWED point dimension differs from metric"))
    current = Vector{Float64}(query_current)
    previous = Vector{Float64}(query_previous)
    copied = Vector{Float64}(sentinel)
    config = raw_vector_twed_bound_config(copied, stiffness, gap_penalty)
    output = Ref{Float64}(0)
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric candidate_current candidate_previous current previous copied ccall(
            native(:llev_vector_twed_interval_lower_bound), Cint,
            (Ptr{Cvoid}, UInt32, Ptr{Float64}, Ptr{Float64},
                Csize_t, Float64, Float64,
                Ref{RawTimestampedVectorBoxView},
                Ref{RawTimestampedVectorBoxView},
                Ref{RawVectorTemporalConfig},
                Ref{VectorTemporalLimits}, Ref{Float64}),
            metric.handle, UInt32(2), pointer(current), pointer(previous),
            metric.dimension, Float64(query_current_time),
            Float64(query_previous_time),
            Ref(raw_timestamped_vector_box(candidate_current)),
            Ref(raw_timestamped_vector_box(candidate_previous)),
            Ref(config), Ref(limits), output)
    end
    checked(status, :llev_vector_twed_interval_lower_bound)
    output[]
end
