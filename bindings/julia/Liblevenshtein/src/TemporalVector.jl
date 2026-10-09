"""One exact channel/unit identity with fixed positive scale and weight."""
struct VectorChannel
    name::String
    unit::String
    scale::Float64
    weight::Float64
end

VectorChannel(name::AbstractString, unit::AbstractString;
    scale::Real=1.0, weight::Real=1.0) =
    VectorChannel(String(name), String(unit), Float64(scale), Float64(weight))

struct RawVectorChannelView
    name::Ptr{UInt8}
    name_len::Csize_t
    unit::Ptr{UInt8}
    unit_len::Csize_t
    scale::Float64
    weight::Float64
end

struct RawVectorMetricView
    channels::Ptr{RawVectorChannelView}
    channel_count::Csize_t
    training_fold::Ptr{UInt8}
    training_fold_len::Csize_t
    estimator_revision::Ptr{UInt8}
    estimator_revision_len::Csize_t
end

"""Reusable native point metric with immutable channel schema and fold provenance."""
mutable struct FixedChannelMetric
    handle::Ptr{Cvoid}
    dimension::Int
    layout::Tuple
    lock::ReentrantLock
    closed::Bool
end

function FixedChannelMetric(channels::AbstractVector{VectorChannel};
    training_fold::AbstractString, estimator_revision::AbstractString,
    max_dimension::Integer=65_536)
    api_revision() >= UInt32(22) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :llev_vector_metric_new,
            "native vector metrics require API revision 22"))
    maximum = checked_threshold(max_dimension)
    0 < length(channels) <= maximum ||
        throw(ArgumentError("vector channel count must be 1..max_dimension"))
    names = [Vector{UInt8}(codeunits(channel.name)) for channel in channels]
    units = [Vector{UInt8}(codeunits(channel.unit)) for channel in channels]
    fold = Vector{UInt8}(codeunits(training_fold))
    revision = Vector{UInt8}(codeunits(estimator_revision))
    raw = [RawVectorChannelView(pointer(names[i]), length(names[i]),
        pointer(units[i]), length(units[i]), channels[i].scale,
        channels[i].weight) for i in eachindex(channels)]
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve names units fold revision raw ccall(
        native(:llev_vector_metric_new), Cint,
        (Ref{RawVectorMetricView}, Csize_t, Ref{Ptr{Cvoid}}),
        Ref(RawVectorMetricView(pointer(raw), length(raw), pointer(fold),
            length(fold), pointer(revision), length(revision))),
        maximum, output)
    checked(status, :llev_vector_metric_new)
    layout = Tuple((channel.name, channel.unit) for channel in channels)
    metric = FixedChannelMetric(output[], length(channels), layout,
        ReentrantLock(), false)
    finalizer(close!, metric)
    metric
end

function close!(metric::FixedChannelMetric)
    lock(metric.lock) do
        metric.closed && return nothing
        handle = metric.handle
        metric.handle = C_NULL
        metric.closed = true
        handle == C_NULL || ccall(native(:llev_vector_metric_free),
            Cvoid, (Ptr{Cvoid},), handle)
        nothing
    end
end
Base.close(metric::FixedChannelMetric) = close!(metric)
Base.isopen(metric::FixedChannelMetric) =
    lock(metric.lock) do
        !metric.closed
    end

"""Owned points in columns; each row is one named channel.

An optional timestamp vector is required for physical-time vector TWED.
The matrix is copied into contiguous Float64 storage once and reused across
native comparisons without flattening point boundaries.
"""
struct VectorTemporalSeries
    samples::Matrix{Float64}
    timestamps::Union{Nothing,Vector{Float64}}
    unit::UInt32
    origin::Float64
end

function VectorTemporalSeries(samples::AbstractMatrix{<:Real};
    timestamps::Union{Nothing,AbstractVector{<:Real}}=nothing,
    unit::Symbol=:seconds, origin::Real=0.0,
    max_series_len::Integer=1_000_000, max_dimension::Integer=65_536,
    max_input_bytes::Integer=512 * 1024 * 1024)
    dimension, count = size(samples)
    0 < dimension <= checked_threshold(max_dimension) ||
        throw(ArgumentError("vector dimension exceeds max_dimension or is zero"))
    count <= checked_threshold(max_series_len) ||
        throw(ArgumentError("vector series exceeds max_series_len"))
    timestamps === nothing || length(timestamps) == count ||
        throw(ArgumentError("vector timestamps must match sample count"))
    input_bytes = Base.checked_mul(Base.checked_mul(dimension, count), 8)
    timestamps === nothing ||
        (input_bytes = Base.checked_add(input_bytes,
            Base.checked_mul(count, 8)))
    input_bytes <= checked_threshold(max_input_bytes) ||
        throw(ArgumentError("vector input exceeds max_input_bytes"))
    matrix = Matrix{Float64}(samples)
    times = timestamps === nothing ? nothing : Vector{Float64}(timestamps)
    VectorTemporalSeries(matrix, times,
        times === nothing ? UInt32(0) : timestamp_unit_code(unit),
        Float64(origin))
end

struct RawVectorSeriesView
    coordinates::Ptr{Float64}
    sample_count::Csize_t
    dimension::Csize_t
    timestamps::Ptr{Float64}
    timestamp_unit::UInt32
    reserved::UInt32
    origin::Float64
end

struct RawVectorTemporalConfig
    algorithm::UInt32
    reserved::UInt32
    gap_or_sentinel::Ptr{Float64}
    parameter0::Float64
    parameter1::Float64
    band::Csize_t
    cutoff::Float64
end

"""Limits on vector dimension, band width, scalar work, and copied input."""
struct VectorTemporalLimits
    scalar::TemporalLimits
    max_dimension::Csize_t
    max_band_width::Csize_t
end

VectorTemporalLimits(; scalar::TemporalLimits=TemporalLimits(),
    max_dimension::Integer=65_536, max_band_width::Integer=1_000_000) =
    VectorTemporalLimits(scalar, checked_threshold(max_dimension),
        checked_threshold(max_band_width))

function raw_vector_series(series::VectorTemporalSeries)
    RawVectorSeriesView(pointer(series.samples), size(series.samples, 2),
        size(series.samples, 1),
        series.timestamps === nothing ? C_NULL : pointer(series.timestamps),
        series.unit, 0, series.origin)
end

"""Score ERP, banded DTW, Fréchet, or physical-time TWED in native Rust.

The metric schema and fold provenance are fixed by `FixedChannelMetric`.
ERP canonicalizes its gap quotient natively; Fréchet removes consecutive
stutters natively; DTW remains a nonmetric scorer. Vector MSM is unsupported
because no canonical vector betweenness relation is defined.
"""
function vector_temporal_distance(kind::Symbol, metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    gap_or_sentinel::Union{Nothing,AbstractVector{<:Real}}=nothing,
    stiffness::Real=0.0, gap_penalty::Real=0.0, band::Integer=0,
    cutoff::Real=Inf, limits::VectorTemporalLimits=VectorTemporalLimits())
    kind_code = if kind === :erp
        UInt32(2)
    elseif kind === :dtw
        UInt32(4)
    elseif kind === :frechet
        UInt32(5)
    elseif kind === :timestamped_twed
        UInt32(7)
    else
        throw(ArgumentError("unsupported vector temporal algorithm $kind"))
    end
    size(left.samples, 1) == metric.dimension &&
        size(right.samples, 1) == metric.dimension ||
        throw(ArgumentError("vector series dimension differs from metric"))
    size(left.samples, 2) <= limits.scalar.max_series_len &&
        size(right.samples, 2) <= limits.scalar.max_series_len ||
        throw(ArgumentError("vector series exceeds max_series_len"))
    metric.dimension <= limits.max_dimension ||
        throw(ArgumentError("vector series exceeds max_dimension"))
    point = gap_or_sentinel === nothing ? Float64[] :
        Vector{Float64}(gap_or_sentinel)
    kind_code in (UInt32(2), UInt32(7)) &&
        length(point) != metric.dimension &&
        throw(ArgumentError("vector gap or sentinel must match metric dimension"))
    kind_code == UInt32(7) &&
        (left.timestamps === nothing || right.timestamps === nothing) &&
        throw(ArgumentError("vector TWED requires timestamps on both series"))
    config = RawVectorTemporalConfig(kind_code, 0,
        isempty(point) ? C_NULL : pointer(point), Float64(stiffness),
        Float64(gap_penalty), checked_threshold(band), Float64(cutoff))
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = lock(metric.lock) do
        metric.closed && throw(ArgumentError("vector metric is closed"))
        GC.@preserve metric left right point ccall(
            native(:llev_vector_temporal_distance), Cint,
            (Ptr{Cvoid}, Ref{RawVectorSeriesView}, Ref{RawVectorSeriesView},
                Ref{RawVectorTemporalConfig}, Ref{VectorTemporalLimits},
                Ref{RawTemporalDistanceResult}),
            metric.handle, Ref(raw_vector_series(left)),
            Ref(raw_vector_series(right)), Ref(config), Ref(limits), output)
    end
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_vector_temporal_distance)
    temporal_outcome(raw)
end

function vector_frechet_ground_code(ground::Symbol)
    ground === :l1 && return UInt32(1)
    ground === :l2 && return UInt32(2)
    ground === :linf && return UInt32(3)
    throw(ArgumentError("Fréchet ground metric must be :l1, :l2, or :linf"))
end

"""Exact native discrete Fréchet with audited L1, L2, or L∞ point distance.

The paths must have the same positive dimension and no timestamps. Native
construction collapses consecutive equal points and charges the copied input
and exact dynamic-programming work to `limits`.
"""
function vector_frechet_ground_distance(ground::Symbol,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    cutoff::Real=Inf, limits::VectorTemporalLimits=VectorTemporalLimits())
    api_revision() >= UInt32(27) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_vector_frechet_ground_distance,
            "ground-metric vector Fréchet requires API revision 27"))
    ground_code = vector_frechet_ground_code(ground)
    left.timestamps === nothing && right.timestamps === nothing ||
        throw(ArgumentError("vector Fréchet paths must have no timestamps"))
    dimension = size(left.samples, 1)
    dimension > 0 && dimension == size(right.samples, 1) ||
        throw(ArgumentError("vector Fréchet dimensions must be positive and equal"))
    dimension <= limits.max_dimension ||
        throw(ArgumentError("vector Fréchet dimension exceeds max_dimension"))
    size(left.samples, 2) <= limits.scalar.max_series_len &&
        size(right.samples, 2) <= limits.scalar.max_series_len ||
        throw(ArgumentError("vector Fréchet path exceeds max_series_len"))
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = GC.@preserve left right ccall(
        native(:llev_vector_frechet_ground_distance), Cint,
        (UInt32, Ref{RawVectorSeriesView}, Ref{RawVectorSeriesView}, Float64,
            Ref{VectorTemporalLimits}, Ref{RawTemporalDistanceResult}),
        ground_code, Ref(raw_vector_series(left)), Ref(raw_vector_series(right)),
        Float64(cutoff), Ref(limits), output)
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_vector_frechet_ground_distance)
    temporal_outcome(raw)
end

vector_erp_distance(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    gap::AbstractVector{<:Real}, kwargs...) =
    vector_temporal_distance(:erp, metric, left, right;
        gap_or_sentinel=gap, kwargs...)

vector_dtw_distance(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    band::Integer, kwargs...) =
    vector_temporal_distance(:dtw, metric, left, right; band, kwargs...)

vector_frechet_distance(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries; kwargs...) =
    vector_temporal_distance(:frechet, metric, left, right; kwargs...)

vector_timestamped_twed_distance(metric::FixedChannelMetric,
    left::VectorTemporalSeries, right::VectorTemporalSeries;
    sentinel::AbstractVector{<:Real}, stiffness::Real,
    gap_penalty::Real=0.0, kwargs...) =
    vector_temporal_distance(:timestamped_twed, metric, left, right;
        gap_or_sentinel=sentinel, stiffness, gap_penalty, kwargs...)
