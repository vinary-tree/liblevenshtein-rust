const SAX_BREAKPOINTS = (
    (0.0,),
    (-0.43, 0.43),
    (-0.67, 0.0, 0.67),
    (-0.84, -0.25, 0.25, 0.84),
    (-0.97, -0.43, 0.0, 0.43, 0.97),
    (-1.07, -0.57, -0.18, 0.18, 0.57, 1.07),
    (-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15),
    (-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22),
    (-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28),
)

"""Return a copy of native SAX breakpoints for alphabet sizes two through ten."""
function sax_breakpoints(alphabet_size::Integer)
    2 <= alphabet_size <= 10 || return nothing
    collect(SAX_BREAKPOINTS[Int(alphabet_size) - 1])
end

function sax_stats(values::Vector{Float64})
    isempty(values) && return (0.0, 0.0)
    total = 0.0
    for value in values
        total += value
    end
    mean = total / length(values)
    variance_sum = 0.0
    for value in values
        offset = value - mean
        variance_sum += offset * offset
    end
    (mean, sqrt(variance_sum / length(values)))
end

"""Lazily return zero-mean, unit-variance samples from a bounded snapshot."""
function sax_normalize(series::AbstractVector{<:Real};
    max_samples::Integer=1_000_000)
    values = checked_encoding_snapshot(Float64, series, max_samples)
    mean, deviation = sax_stats(values)
    deviation < 1e-10 ? (0.0 for _ in values) :
        ((value - mean) / deviation for value in values)
end

struct SaxPaaIterator
    values::Vector{Float64}
    segments::Int
    mean::Float64
    deviation::Float64
    normalized::Bool
end

Base.IteratorSize(::Type{SaxPaaIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{SaxPaaIterator}) = Base.HasEltype()
Base.eltype(::Type{SaxPaaIterator}) = Float64
Base.length(paa::SaxPaaIterator) = paa.segments

function scaled_sax_boundary(segment::Int, count::Int, segments::Int)
    segment * (count ÷ segments) +
        Int((UInt128(segment) * UInt128(count % segments)) ÷
            UInt128(segments))
end

function Base.iterate(paa::SaxPaaIterator, index::Int=0)
    index >= paa.segments && return nothing
    count = length(paa.values)
    start = min(scaled_sax_boundary(index, count, paa.segments),
        count - 1)
    stop = min(max(scaled_sax_boundary(index + 1, count,
        paa.segments), start + 1), count)
    total = 0.0
    for position in (start + 1):stop
        value = paa.values[position]
        if paa.normalized
            value = paa.deviation < 1e-10 ? 0.0 :
                (value - paa.mean) / paa.deviation
        end
        total += value
    end
    (total / (stop - start), index + 1)
end

function sax_paa_snapshot(series::AbstractVector{<:Real},
    num_segments::Integer, max_samples::Integer, max_segments::Integer,
    normalized::Bool)
    segment_limit = checked_threshold(max_segments)
    count = checked_threshold(num_segments)
    count <= segment_limit ||
        throw(ArgumentError("SAX word exceeds max_segments"))
    values = checked_encoding_snapshot(Float64, series, max_samples)
    segments = isempty(values) ? 0 : Int(count)
    mean, deviation = normalized ? sax_stats(values) : (0.0, 1.0)
    SaxPaaIterator(values, segments, mean, deviation, normalized)
end

"""Lazily compute native-style piecewise aggregate approximation."""
function sax_paa(series::AbstractVector{<:Real}, num_segments::Integer;
    max_samples::Integer=1_000_000,
    max_segments::Integer=1_000_000)
    sax_paa_snapshot(series, num_segments, max_samples, max_segments,
        false)
end

function sax_symbol(value::Float64, breakpoints)
    isnan(value) && return UInt8(0)
    for (index, breakpoint) in enumerate(breakpoints)
        value < breakpoint && return UInt8(index - 1)
    end
    UInt8(length(breakpoints))
end

"""Lazily encode a bounded source as a SAX word of the requested length."""
function sax_encode(series::AbstractVector{<:Real},
    num_segments::Integer, alphabet_size::Integer;
    max_samples::Integer=1_000_000,
    max_segments::Integer=1_000_000)
    breakpoints = sax_breakpoints(alphabet_size)
    breakpoints === nothing && return (UInt8(0) for _ in 1:0)
    paa = sax_paa_snapshot(series, num_segments, max_samples,
        max_segments, true)
    (sax_symbol(value, breakpoints) for value in paa)
end

"""Native SAX MINDIST lower bound for two bounded symbol snapshots."""
function sax_mindist(first::AbstractVector{UInt8},
    second::AbstractVector{UInt8}, series_len::Integer,
    alphabet_size::Integer; max_word_len::Integer=1_000_000)
    length(first) <= checked_threshold(max_word_len) ||
        throw(ArgumentError("first SAX word exceeds max_word_len"))
    length(second) <= checked_threshold(max_word_len) ||
        throw(ArgumentError("second SAX word exceeds max_word_len"))
    count = checked_threshold(series_len)
    length(first) == length(second) && !isempty(first) || return Inf
    breakpoints = sax_breakpoints(alphabet_size)
    breakpoints === nothing && return Inf
    left = copy(first)
    right = copy(second)
    total = 0.0
    for (a, b) in zip(left, right)
        small = Int(min(a, b))
        large = Int(max(a, b))
        if large - small > 1
            distance = if large > 0 && small < length(breakpoints)
                upper = large <= length(breakpoints) ?
                    breakpoints[large] : Inf
                abs(upper - breakpoints[small + 1])
            else
                0.0
            end
            total += distance * distance
        end
    end
    sqrt(Float64(count) / length(left)) * sqrt(total)
end
