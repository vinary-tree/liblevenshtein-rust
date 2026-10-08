"""Iterate over a reconstructed series from an initial value and copied deltas."""
struct DeltaReconstruction{T,F}
    initial::Float64
    values::Vector{T}
    decode::F
end

Base.IteratorSize(::Type{<:DeltaReconstruction}) = Base.HasLength()
Base.IteratorEltype(::Type{<:DeltaReconstruction}) = Base.HasEltype()
Base.eltype(::Type{<:DeltaReconstruction}) = Float64
Base.length(series::DeltaReconstruction) = length(series.values) + 1

function Base.iterate(series::DeltaReconstruction, state=(1, series.initial))
    index, previous = state
    index > length(series) && return nothing
    if index == 1
        return (series.initial, (2, series.initial))
    end
    value = previous + series.decode(series.values[index - 1])
    (value, (index + 1, value))
end

"""Iterate over consecutive differences from a bounded copied series."""
function compute_deltas(series::AbstractVector{<:Real};
    max_samples::Integer=1_000_000)
    values = checked_encoding_snapshot(Float64, series, max_samples)
    (values[i] - values[i - 1] for i in 2:length(values))
end

"""Reconstruct a series lazily from a bounded copied delta sequence."""
function reconstruct_from_deltas(initial::Real,
    deltas::AbstractVector{<:Real}; max_deltas::Integer=1_000_000)
    values = checked_encoding_snapshot(Float64, deltas, max_deltas)
    DeltaReconstruction(Float64(initial), values, identity)
end

"""Return the first sample and lazy byte-quantized consecutive differences."""
function encode_deltas_u8(series::AbstractVector{<:Real},
    config::QuantizationConfig; max_samples::Integer=1_000_000)
    config.num_bins <= 256 ||
        throw(ArgumentError("delta quantizer has more than 256 bins"))
    values = checked_encoding_snapshot(Float64, series, max_samples)
    isempty(values) && return (0.0, (UInt8(0) for _ in 1:0))
    (values[1], (quantize_u8(config, values[i] - values[i - 1])
        for i in 2:length(values)))
end

"""Reconstruct a series lazily from bounded copied quantized deltas."""
function decode_deltas_u8(initial::Real, encoded::AbstractVector{UInt8},
    config::QuantizationConfig; max_deltas::Integer=1_000_000)
    words = checked_encoding_snapshot(UInt8, encoded, max_deltas)
    DeltaReconstruction(Float64(initial), words,
        bin -> dequantize(config, bin))
end
