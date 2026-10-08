function metric_finite_samples(raw::AbstractVector{<:Real},
    max_series_len::Integer)
    maximum = checked_threshold(max_series_len)
    length(raw) <= maximum ||
        throw(ArgumentError("metric series exceeds max_series_len"))
    samples = Vector{Float64}(undef, length(raw))
    for (position, sample) in enumerate(raw)
        value = Float64(sample)
        isfinite(value) ||
            throw(ArgumentError("metric series contains a nonfinite sample at index $position"))
        samples[position] = value
    end
    samples
end

"""MSM metric witness with finite positive split/merge cost."""
struct MetricMsmConfig
    split_merge_cost::Float64
    function MetricMsmConfig(split_merge_cost::Real)
        value = Float64(split_merge_cost)
        isfinite(value) && value > 0.0 ||
            throw(ArgumentError("metric MSM split/merge cost must be finite and positive"))
        new(value)
    end
end

"""Unit-grid TWED metric witness with positive stiffness and nonnegative gap cost."""
struct MetricTwedConfig
    stiffness::Float64
    gap_penalty::Float64
    function MetricTwedConfig(stiffness::Real, gap_penalty::Real)
        nu = Float64(stiffness)
        lambda = Float64(gap_penalty)
        isfinite(nu) && nu > 0.0 ||
            throw(ArgumentError("metric TWED stiffness must be finite and positive"))
        isfinite(lambda) && lambda >= 0.0 ||
            throw(ArgumentError("metric TWED gap penalty must be finite and nonnegative"))
        new(nu, lambda)
    end
end

"""Bounded exact MSM score on its nonempty metric domain."""
function metric_msm_distance(config::MetricMsmConfig,
    left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    cutoff::Real=Inf, limits::TemporalLimits=TemporalLimits())
    isempty(left) && throw(ArgumentError("metric MSM left series must be nonempty"))
    isempty(right) && throw(ArgumentError("metric MSM right series must be nonempty"))
    msm_distance(left, right; split_merge_cost=config.split_merge_cost,
        cutoff, limits)
end

"""Bounded exact unit-grid TWED score under a validated metric configuration."""
metric_twed_distance(config::MetricTwedConfig,
    left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
    cutoff::Real=Inf, limits::TemporalLimits=TemporalLimits()) =
    twed_distance(left, right; stiffness=config.stiffness,
        gap_penalty=config.gap_penalty, cutoff, limits)

"""A finite ERP gap defining one quotient metric domain."""
struct MetricErpConfig
    gap::Float64
    function MetricErpConfig(gap::Real)
        value = Float64(gap)
        isfinite(value) || throw(ArgumentError("ERP quotient gap must be finite"))
        new(value)
    end
end

"""Owned ERP representative with every occurrence of its gap removed."""
struct ErpQuotientSeries
    _gap::Float64
    _samples::Vector{Float64}
    function ErpQuotientSeries(raw::AbstractVector{<:Real}, gap::Real;
        max_series_len::Integer=1_000_000)
        config = MetricErpConfig(gap)
        samples = metric_finite_samples(raw, max_series_len)
        filter!(value -> value != config.gap, samples)
        new(config.gap, samples)
    end
end

Base.getproperty(series::ErpQuotientSeries, name::Symbol) =
    name === :gap ? getfield(series, :_gap) :
    name === :samples ? canonical_samples(series) :
    throw(ArgumentError("unknown ERP quotient property $name"))
Base.propertynames(::ErpQuotientSeries) = (:gap, :samples)

"""Return an owned copy of a canonical metric-domain representative."""
canonical_samples(series::ErpQuotientSeries) = copy(getfield(series, :_samples))
representative(config::MetricErpConfig, raw::AbstractVector{<:Real};
    max_series_len::Integer=1_000_000) =
    ErpQuotientSeries(raw, config.gap; max_series_len)

function metric_erp_gap_matches(config::MetricErpConfig,
    series::ErpQuotientSeries)
    reinterpret(UInt64, config.gap) == reinterpret(UInt64,
        getfield(series, :_gap)) ||
        throw(ArgumentError("ERP quotient representative uses a different gap"))
    nothing
end

"""Bounded exact ERP metric score between representatives of one gap quotient."""
function metric_erp_distance(config::MetricErpConfig,
    left::ErpQuotientSeries, right::ErpQuotientSeries;
    cutoff::Real=Inf, limits::TemporalLimits=TemporalLimits())
    metric_erp_gap_matches(config, left)
    metric_erp_gap_matches(config, right)
    erp_distance(getfield(left, :_samples), getfield(right, :_samples);
        gap=config.gap, cutoff, limits)
end

"""Owned, nonempty discrete Fréchet representative with runs collapsed."""
struct FrechetStutterClass
    _samples::Vector{Float64}
    function FrechetStutterClass(raw::AbstractVector{<:Real};
        max_series_len::Integer=1_000_000)
        values = metric_finite_samples(raw, max_series_len)
        isempty(values) &&
            throw(ArgumentError("Fréchet metric quotient requires a nonempty path"))
        retained = 0
        for position in eachindex(values)
            value = values[position]
            if retained == 0 || values[retained] != value
                retained += 1
                values[retained] = value
            end
        end
        resize!(values, retained)
        new(values)
    end
end

Base.getproperty(series::FrechetStutterClass, name::Symbol) =
    name === :samples ? canonical_samples(series) :
    throw(ArgumentError("unknown Fréchet stutter property $name"))
Base.propertynames(::FrechetStutterClass) = (:samples,)

canonical_samples(series::FrechetStutterClass) =
    copy(getfield(series, :_samples))

"""Bounded exact discrete Fréchet metric score on stutter classes."""
metric_frechet_distance(left::FrechetStutterClass,
    right::FrechetStutterClass; cutoff::Real=Inf,
    limits::TemporalLimits=TemporalLimits()) =
    frechet_distance(getfield(left, :_samples),
        getfield(right, :_samples); cutoff, limits)

"""Freezable native ERP range index restricted to one canonical gap quotient."""
struct MetricErpIndex
    _index::TemporalIndex
    config::MetricErpConfig
    function MetricErpIndex(config::MetricErpConfig; kwargs...)
        new(TemporalIndex(:erp; parameter0=config.gap, kwargs...), config)
    end
end
MetricErpIndex(gap::Real; kwargs...) =
    MetricErpIndex(MetricErpConfig(gap); kwargs...)

Base.getproperty(index::MetricErpIndex, name::Symbol) =
    name === :config ? getfield(index, :config) :
    throw(ArgumentError("unknown metric ERP index property $name"))
Base.propertynames(::MetricErpIndex) = (:config,)

function insert!(index::MetricErpIndex, id::Integer,
    series::ErpQuotientSeries)
    metric_erp_gap_matches(index.config, series)
    insert!(getfield(index, :_index), id, getfield(series, :_samples))
    index
end

function insert!(index::MetricErpIndex, id::Integer,
    raw::AbstractVector{<:Real})
    inner = getfield(index, :_index)
    inner.closed && throw(ArgumentError("metric ERP index is closed"))
    inner.frozen && throw(ArgumentError("metric ERP index is frozen"))
    insert!(index, id, representative(index.config, raw;
        max_series_len=inner.max_series_len))
end

function query_metric_range(index::MetricErpIndex,
    query::ErpQuotientSeries; kwargs...)
    metric_erp_gap_matches(index.config, query)
    query_index_range(getfield(index, :_index),
        getfield(query, :_samples); kwargs...)
end

function query_metric_range(index::MetricErpIndex,
    raw::AbstractVector{<:Real};
    cutoff::Real=Inf, limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    inner = getfield(index, :_index)
    inner.closed && throw(ArgumentError("metric ERP index is closed"))
    inner.frozen || throw(ArgumentError("metric ERP index must be frozen"))
    query = representative(index.config, raw;
        max_series_len=limits.max_series_len)
    query_metric_range(index, query; cutoff, limits,
        page_work_units, page_results)
end

freeze!(index::MetricErpIndex) =
    (freeze!(getfield(index, :_index)); index)
close!(index::MetricErpIndex) = close!(getfield(index, :_index))
Base.close(index::MetricErpIndex) = close!(index)
Base.isopen(index::MetricErpIndex) = isopen(getfield(index, :_index))

"""Freezable native Fréchet range index over nonempty stutter classes."""
struct MetricFrechetIndex
    _index::TemporalIndex
    function MetricFrechetIndex(; kwargs...)
        new(TemporalIndex(:frechet; kwargs...))
    end
end

Base.getproperty(::MetricFrechetIndex, name::Symbol) =
    throw(ArgumentError("unknown metric Fréchet index property $name"))
Base.propertynames(::MetricFrechetIndex) = ()

function insert!(index::MetricFrechetIndex, id::Integer,
    series::FrechetStutterClass)
    insert!(getfield(index, :_index), id, getfield(series, :_samples))
    index
end

function insert!(index::MetricFrechetIndex, id::Integer,
    raw::AbstractVector{<:Real})
    inner = getfield(index, :_index)
    inner.closed && throw(ArgumentError("metric Fréchet index is closed"))
    inner.frozen && throw(ArgumentError("metric Fréchet index is frozen"))
    insert!(index, id, FrechetStutterClass(raw;
        max_series_len=inner.max_series_len))
end

query_metric_range(index::MetricFrechetIndex,
    query::FrechetStutterClass; kwargs...) =
    query_index_range(getfield(index, :_index),
        getfield(query, :_samples); kwargs...)

function query_metric_range(index::MetricFrechetIndex,
    raw::AbstractVector{<:Real};
    cutoff::Real=Inf, limits::TemporalSearchLimits=TemporalSearchLimits(),
    page_work_units::Integer=100_000, page_results::Integer=256)
    inner = getfield(index, :_index)
    inner.closed && throw(ArgumentError("metric Fréchet index is closed"))
    inner.frozen || throw(ArgumentError("metric Fréchet index must be frozen"))
    query = FrechetStutterClass(raw;
        max_series_len=limits.max_series_len)
    query_metric_range(index, query; cutoff, limits,
        page_work_units, page_results)
end

freeze!(index::MetricFrechetIndex) =
    (freeze!(getfield(index, :_index)); index)
close!(index::MetricFrechetIndex) = close!(getfield(index, :_index))
Base.close(index::MetricFrechetIndex) = close!(index)
Base.isopen(index::MetricFrechetIndex) = isopen(getfield(index, :_index))

"""Freezable native MSM index over the nonempty metric domain."""
struct MetricMsmIndex
    _index::TemporalIndex
    config::MetricMsmConfig
    function MetricMsmIndex(config::MetricMsmConfig; kwargs...)
        new(TemporalIndex(:msm;
            parameter0=config.split_merge_cost, kwargs...), config)
    end
end

Base.getproperty(index::MetricMsmIndex, name::Symbol) =
    name === :config ? getfield(index, :config) :
    throw(ArgumentError("unknown metric MSM index property $name"))
Base.propertynames(::MetricMsmIndex) = (:config,)

function insert!(index::MetricMsmIndex, id::Integer,
    raw::AbstractVector{<:Real})
    isempty(raw) && throw(ArgumentError("metric MSM series must be nonempty"))
    insert!(getfield(index, :_index), id, raw)
    index
end

function query_metric_range(index::MetricMsmIndex,
    raw::AbstractVector{<:Real}; kwargs...)
    isempty(raw) && throw(ArgumentError("metric MSM query must be nonempty"))
    query_index_range(getfield(index, :_index), raw; kwargs...)
end

freeze!(index::MetricMsmIndex) =
    (freeze!(getfield(index, :_index)); index)
close!(index::MetricMsmIndex) = close!(getfield(index, :_index))
Base.close(index::MetricMsmIndex) = close!(index)
Base.isopen(index::MetricMsmIndex) = isopen(getfield(index, :_index))

"""Freezable native index under a validated unit-grid TWED metric."""
struct MetricTwedIndex
    _index::TemporalIndex
    config::MetricTwedConfig
    function MetricTwedIndex(config::MetricTwedConfig; kwargs...)
        new(TemporalIndex(:twed; parameter0=config.stiffness,
            parameter1=config.gap_penalty, kwargs...), config)
    end
end

Base.getproperty(index::MetricTwedIndex, name::Symbol) =
    name === :config ? getfield(index, :config) :
    throw(ArgumentError("unknown metric TWED index property $name"))
Base.propertynames(::MetricTwedIndex) = (:config,)

function insert!(index::MetricTwedIndex, id::Integer,
    raw::AbstractVector{<:Real})
    insert!(getfield(index, :_index), id, raw)
    index
end

query_metric_range(index::MetricTwedIndex,
    raw::AbstractVector{<:Real}; kwargs...) =
    query_index_range(getfield(index, :_index), raw; kwargs...)

freeze!(index::MetricTwedIndex) =
    (freeze!(getfield(index, :_index)); index)
close!(index::MetricTwedIndex) = close!(getfield(index, :_index))
Base.close(index::MetricTwedIndex) = close!(index)
Base.isopen(index::MetricTwedIndex) = isopen(getfield(index, :_index))
