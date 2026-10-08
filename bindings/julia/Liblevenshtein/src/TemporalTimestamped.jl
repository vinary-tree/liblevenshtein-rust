"""Metric TWED configuration for explicit physical timestamps."""
struct MetricTimestampedTwedConfig
    stiffness::Float64
    gap_penalty::Float64
    function MetricTimestampedTwedConfig(stiffness::Real, gap_penalty::Real)
        nu = Float64(stiffness)
        lambda = Float64(gap_penalty)
        isfinite(nu) && nu > 0.0 ||
            throw(ArgumentError("timestamped TWED stiffness must be finite and positive"))
        isfinite(lambda) && lambda >= 0.0 ||
            throw(ArgumentError("timestamped TWED gap penalty must be finite and nonnegative"))
        new(nu, lambda)
    end
end

function timestamp_unit_code(unit::Symbol)::UInt32
    unit === :seconds && return 1
    unit === :milliseconds && return 2
    unit === :microseconds && return 3
    unit === :nanoseconds && return 4
    throw(ArgumentError("unknown timestamp unit $unit"))
end

"""Owned nonempty scalar series with strictly increasing physical timestamps."""
struct TimestampedSeries
    _values::Vector{Float64}
    _timestamps::Vector{Float64}
    _unit::Symbol
    _origin::Float64
    function TimestampedSeries(values::AbstractVector{<:Real},
        timestamps::AbstractVector{<:Real}; unit::Symbol=:seconds,
        origin::Real=0.0, max_series_len::Integer=1_000_000)
        maximum = checked_threshold(max_series_len)
        length(values) == length(timestamps) ||
            throw(ArgumentError("timestamped value and timestamp counts differ"))
        isempty(values) &&
            throw(ArgumentError("timestamped series must be nonempty"))
        length(values) <= maximum ||
            throw(ArgumentError("timestamped series exceeds max_series_len"))
        timestamp_unit_code(unit)
        start = Float64(origin)
        isfinite(start) ||
            throw(ArgumentError("timestamped origin must be finite"))
        copied_values = Vector{Float64}(undef, length(values))
        copied_times = Vector{Float64}(undef, length(timestamps))
        for (position, (raw_value, raw_time)) in
            enumerate(zip(values, timestamps))
            value = Float64(raw_value)
            time = Float64(raw_time)
            isfinite(value) ||
                throw(ArgumentError("timestamped value at index $position is nonfinite"))
            isfinite(time) ||
                throw(ArgumentError("timestamp at index $position is nonfinite"))
            if position == 1
                time >= start ||
                    throw(ArgumentError("first timestamp precedes origin"))
            else
                time > copied_times[position - 1] ||
                    throw(ArgumentError("timestamps must increase strictly"))
            end
            copied_values[position] = value
            copied_times[position] = time
        end
        new(copied_values, copied_times, unit, start)
    end
end

Base.getproperty(series::TimestampedSeries, name::Symbol) =
    name === :values ? copy(getfield(series, :_values)) :
    name === :timestamps ? copy(getfield(series, :_timestamps)) :
    name === :unit ? getfield(series, :_unit) :
    name === :origin ? getfield(series, :_origin) :
    throw(ArgumentError("unknown timestamped series property $name"))
Base.propertynames(::TimestampedSeries) =
    (:values, :timestamps, :unit, :origin)

struct RawTimestampedSeriesView
    values::Ptr{Float64}
    timestamps::Ptr{Float64}
    len::Csize_t
    unit::UInt32
    reserved::UInt32
    origin::Float64
end

"""Exact bounded TWED over physical time, retaining explicit stop reasons."""
function metric_timestamped_twed_distance(config::MetricTimestampedTwedConfig,
    left::TimestampedSeries, right::TimestampedSeries;
    cutoff::Real=Inf, limits::TemporalLimits=TemporalLimits())
    api_revision() >= UInt32(16) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_timestamped_twed_distance,
            "timestamped TWED requires native API revision 16"))
    length(getfield(left, :_values)) <= limits.max_series_len ||
        throw(ArgumentError("left timestamped series exceeds max_series_len"))
    length(getfield(right, :_values)) <= limits.max_series_len ||
        throw(ArgumentError("right timestamped series exceeds max_series_len"))
    left.unit === right.unit ||
        throw(ArgumentError("timestamped series units differ"))
    left.origin == right.origin ||
        throw(ArgumentError("timestamped series origins differ"))
    score_cutoff = Float64(cutoff)
    !isnan(score_cutoff) && score_cutoff >= 0.0 ||
        throw(ArgumentError("timestamped TWED cutoff must be nonnegative"))
    x = getfield(left, :_values)
    xt = getfield(left, :_timestamps)
    y = getfield(right, :_values)
    yt = getfield(right, :_timestamps)
    first = RawTimestampedSeriesView(pointer(x), pointer(xt), length(x),
        timestamp_unit_code(left.unit), 0, left.origin)
    second = RawTimestampedSeriesView(pointer(y), pointer(yt), length(y),
        timestamp_unit_code(right.unit), 0, right.origin)
    output = Ref(RawTemporalDistanceResult(0.0, 0, 0, 0, 0, 0))
    status = GC.@preserve x xt y yt ccall(
        native(:llev_timestamped_twed_distance), Cint,
        (Ref{RawTimestampedSeriesView}, Ref{RawTimestampedSeriesView},
            Float64, Float64, Float64, Ref{TemporalLimits},
            Ref{RawTemporalDistanceResult}),
        Ref(first), Ref(second), config.stiffness, config.gap_penalty,
        score_cutoff, Ref(limits), output)
    raw = output[]
    status == Int32(STATUS_OK) ||
        (status == Int32(STATUS_LIMIT_EXCEEDED) && raw.kind == 3) ||
        checked(status, :llev_timestamped_twed_distance)
    temporal_outcome(raw)
end
