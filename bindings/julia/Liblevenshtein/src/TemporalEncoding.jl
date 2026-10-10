"""Lossless IEEE-754 time-series encodings matching native `float_encoding`."""

"""Encode one Float32 as its exact unsigned IEEE-754 bit pattern."""
encode_f32(value::Float32) = reinterpret(UInt32, value)

"""Decode an unsigned IEEE-754 bit pattern to Float32 without changing NaN payloads."""
decode_f32(bits::UInt32) = reinterpret(Float32, bits)

"""Encode one Float64 as its exact unsigned IEEE-754 bit pattern."""
encode_f64(value::Float64) = reinterpret(UInt64, value)

"""Decode an unsigned IEEE-754 bit pattern to Float64."""
decode_f64(bits::UInt64) = reinterpret(Float64, bits)

function checked_encoding_snapshot(::Type{T}, series::AbstractVector,
    max_samples::Integer) where {T}
    limit = checked_threshold(max_samples)
    length(series) <= limit ||
        throw(ArgumentError("encoding input exceeds max_samples"))
    Vector{T}(series)
end

"""Iterate over exact Float32 bit patterns from a bounded source snapshot."""
function encode_f32_series(series::AbstractVector{<:Real};
    max_samples::Integer=1_000_000)
    values = checked_encoding_snapshot(Float32, series, max_samples)
    (encode_f32(value) for value in values)
end

"""Iterate over Float32 values decoded from a bounded bit-pattern snapshot."""
function decode_f32_series(encoded::AbstractVector{UInt32};
    max_samples::Integer=1_000_000)
    words = checked_encoding_snapshot(UInt32, encoded, max_samples)
    (decode_f32(bits) for bits in words)
end

"""Iterate over high/low UInt32 pairs from a bounded Float64 snapshot."""
function encode_f64_series_as_u32_pairs(series::AbstractVector{<:Real};
    max_samples::Integer=1_000_000)
    values = checked_encoding_snapshot(Float64, series, max_samples)
    Iterators.flatten(((UInt32(encode_f64(value) >> 32),
        UInt32(encode_f64(value) & 0xffff_ffff)) for value in values))
end

"""Decode high/low UInt32 pairs lazily; an unmatched final word is ignored."""
function decode_u32_pairs_to_f64(encoded::AbstractVector{UInt32};
    max_words::Integer=2_000_000)
    words = checked_encoding_snapshot(UInt32, encoded, max_words)
    (decode_f64((UInt64(words[2i - 1]) << 32) | UInt64(words[2i]))
        for i in 1:(length(words) ÷ 2))
end

"""Encode a nonnegative Float32 for unsigned-ordered trie lookup."""
function encode_f32_ordered(value::Float32)
    value >= 0 || throw(ArgumentError("ordered Float32 must be nonnegative"))
    encode_f32(value)
end

"""Encode a Float32 so unsigned word order follows IEEE total bit order."""
function encode_f32_total_order(value::Float32)
    bits = encode_f32(value)
    bits & 0x8000_0000 != 0 ? ~bits : bits ⊻ 0x8000_0000
end

"""Reverse the order-preserving Float32 bit transform exactly."""
function decode_f32_total_order(encoded::UInt32)
    bits = encoded & 0x8000_0000 != 0 ?
        encoded ⊻ 0x8000_0000 : ~encoded
    decode_f32(bits)
end
