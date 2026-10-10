"""Uniform temporal quantizer with the native Rust bin and outlier rules."""
struct QuantizationConfig
    min_value::Float64
    max_value::Float64
    num_bins::UInt32
    width::Float64
    clamp_outliers::Bool
end

"""Return `nothing` for an invalid uniform range, count, or bin width."""
function try_uniform_quantizer(min_value::Real, max_value::Real,
    num_bins::Integer; clamp_outliers::Bool=true)
    0 < num_bins <= typemax(UInt32) || return nothing
    low = Float64(min_value)
    high = Float64(max_value)
    isfinite(low) && isfinite(high) && low < high || return nothing
    width = (high - low) / Float64(num_bins)
    isfinite(width) && width > 0 || return nothing
    QuantizationConfig(low, high, UInt32(num_bins), width, clamp_outliers)
end

"""Construct a uniform quantizer; invalid configurations raise ArgumentError."""
function QuantizationConfig(min_value::Real, max_value::Real,
    num_bins::Integer; clamp_outliers::Bool=true)
    config = try_uniform_quantizer(min_value, max_value, num_bins;
        clamp_outliers)
    config === nothing && throw(ArgumentError("invalid uniform quantizer"))
    config
end

QuantizationConfig() = QuantizationConfig(0.0, 1.0, 256)
quantizer_u8(low::Real, high::Real) = QuantizationConfig(low, high, 256)
quantizer_u16(low::Real, high::Real) = QuantizationConfig(low, high, 65_536)

"""Fit a uniform quantizer to finite samples under a hard input length cap."""
function quantizer_from_data(data::AbstractVector{<:Real}, num_bins::Integer,
    margin::Real; max_samples::Integer=1_000_000)
    length(data) <= checked_threshold(max_samples) ||
        throw(ArgumentError("quantizer input exceeds max_samples"))
    isempty(data) && return nothing
    0 < num_bins <= typemax(UInt32) || return nothing
    gap = Float64(margin)
    isfinite(gap) || return nothing
    low = Inf
    high = -Inf
    for value in data
        sample = Float64(value)
        if isfinite(sample)
            low = min(low, sample)
            high = max(high, sample)
        end
    end
    isfinite(low) && isfinite(high) && low < high || return nothing
    extra = (high - low) * gap
    isfinite(extra) || return nothing
    try_uniform_quantizer(low - extra, high + extra, num_bins)
end

bin_width(config::QuantizationConfig) = config.width
max_error(config::QuantizationConfig) = config.width / 2

"""Map one value to a zero-based bin; NaN and negative infinity map to zero."""
function quantize(config::QuantizationConfig, value::Real)::UInt32
    sample = Float64(value)
    (isnan(sample) || sample == -Inf) && return UInt32(0)
    sample == Inf && return config.num_bins - UInt32(1)
    if config.clamp_outliers
        sample <= config.min_value && return UInt32(0)
        sample >= config.max_value && return config.num_bins - UInt32(1)
    end
    normalized = (sample - config.min_value) / config.width
    (!isfinite(normalized) && normalized < 0) && return UInt32(0)
    (!isfinite(normalized) && normalized > 0) &&
        return config.num_bins - UInt32(1)
    (isnan(normalized) || normalized <= 0) && return UInt32(0)
    maximum_bin = config.num_bins - UInt32(1)
    normalized >= Float64(maximum_bin) && return maximum_bin
    UInt32(floor(Int, normalized))
end

function quantize_u8(config::QuantizationConfig, value::Real)::UInt8
    config.num_bins <= 256 ||
        throw(ArgumentError("quantizer has more than 256 bins"))
    UInt8(quantize(config, value))
end

function quantize_u16(config::QuantizationConfig, value::Real)::UInt16
    config.num_bins <= 65_536 ||
        throw(ArgumentError("quantizer has more than 65,536 bins"))
    UInt16(quantize(config, value))
end

function checked_bin(config::QuantizationConfig, bin::Integer)
    bin >= 0 || throw(ArgumentError("bin must be nonnegative"))
    min(bin, Int(config.num_bins) - 1)
end

"""Return the center of a bin, clamping indexes above the final bin."""
dequantize(config::QuantizationConfig, bin::Integer) =
    config.min_value + (checked_bin(config, bin) + 0.5) * config.width

"""Return an admissible bin interval; extreme bins absorb all outliers."""
function bin_bounds(config::QuantizationConfig, bin::Integer)
    index = checked_bin(config, bin)
    lower = index == 0 ? -Inf :
        config.min_value + index * config.width
    upper = index == Int(config.num_bins) - 1 ? Inf :
        config.min_value + (index + 1) * config.width
    (lower, upper)
end

"""Saturating conversion of an absolute value difference to bin units."""
function value_diff_to_bins(config::QuantizationConfig, difference::Real)::UInt32
    bins = abs(Float64(difference)) / config.width
    (!isfinite(bins) || bins >= Float64(typemax(UInt32))) &&
        return typemax(UInt32)
    UInt32(ceil(Int, bins))
end

"""Lazily encode a bounded, copied series as byte bins."""
function encode_u8(config::QuantizationConfig,
    series::AbstractVector{<:Real}; max_samples::Integer=1_000_000)
    config.num_bins <= 256 ||
        throw(ArgumentError("quantizer has more than 256 bins"))
    values = checked_encoding_snapshot(Float64, series, max_samples)
    (quantize_u8(config, value) for value in values)
end

"""Lazily encode a bounded, copied series as UInt32 bins."""
function encode_u32(config::QuantizationConfig,
    series::AbstractVector{<:Real}; max_samples::Integer=1_000_000)
    values = checked_encoding_snapshot(Float64, series, max_samples)
    (quantize(config, value) for value in values)
end

"""Lazily decode a bounded, copied byte-bin series to bin centers."""
function decode_u8(config::QuantizationConfig,
    encoded::AbstractVector{UInt8}; max_samples::Integer=1_000_000)
    words = checked_encoding_snapshot(UInt8, encoded, max_samples)
    (dequantize(config, bin) for bin in words)
end

"""Lazily decode a bounded, copied UInt32-bin series to bin centers."""
function decode_u32(config::QuantizationConfig,
    encoded::AbstractVector{UInt32}; max_samples::Integer=1_000_000)
    words = checked_encoding_snapshot(UInt32, encoded, max_samples)
    (dequantize(config, bin) for bin in words)
end
