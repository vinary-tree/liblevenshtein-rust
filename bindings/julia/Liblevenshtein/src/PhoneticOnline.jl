"""A copied character-level match with zero-based end-exclusive source spans."""
struct PhoneticOnlineMatch
    original_text::String
    normalized_text::String
    byte_range::Tuple{Int,Int}
    char_range::Tuple{Int,Int}
    distance::Int
end

struct RawPhoneticOnlineMatch
    byte_start::Csize_t
    byte_end::Csize_t
    char_start::Csize_t
    char_end::Csize_t
    original_text::RawOwnedString
    normalized_text::RawOwnedString
    distance::UInt8
    reserved::NTuple{7,UInt8}
end

"""Reusable native substring matcher with phonetic normalization and edit distance."""
mutable struct PhoneticOnlineGrep
    handle::Ptr{Cvoid}
    closed::Bool
end

function PhoneticOnlineGrep(pattern::AbstractString;
    rules::Union{Nothing,PhoneticRuleSet}=nothing,
    max_distance::Integer=0, case_insensitive::Bool=false,
    max_pattern_scalars::Integer=4_096)
    0 <= max_distance <= typemax(UInt8) ||
        throw(ArgumentError("max_distance must fit UInt8"))
    max_pattern_scalars > 0 || throw(ArgumentError("max_pattern_scalars must be positive"))
    bytes = text_bytes(pattern)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_online_new), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, UInt8, UInt8, Csize_t, Ref{Ptr{Cvoid}}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), rules_handle,
        UInt8(max_distance), UInt8(case_insensitive),
        checked_csize(max_pattern_scalars, "max_pattern_scalars"), output),
        :llev_phonetic_online_new)
    grep = PhoneticOnlineGrep(output[], false)
    finalizer(close!, grep)
    grep
end

function require_open(grep::PhoneticOnlineGrep)
    grep.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :phonetic_online, "online matcher is closed"))
    grep.handle
end

function close!(grep::PhoneticOnlineGrep)
    grep.closed && return nothing
    handle = grep.handle
    grep.handle = C_NULL
    grep.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_online_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(grep::PhoneticOnlineGrep) = close!(grep)
Base.isopen(grep::PhoneticOnlineGrep) = !grep.closed

"""Return a copied query after the native rewrite rules are applied."""
function normalized_query(grep::PhoneticOnlineGrep)
    output = Ref(RawOwnedString(C_NULL, 0))
    checked(ccall(native(:llev_phonetic_online_normalized_query), Cint,
        (Ptr{Cvoid}, Ref{RawOwnedString}), require_open(grep), output),
        :llev_phonetic_online_normalized_query)
    try
        copied_owned_string(output[])
    finally
        ccall(native(:llev_owned_string_free), Cvoid,
            (Ref{RawOwnedString},), output)
    end
end

function copy_online_matches(pointer::Ptr{RawPhoneticOnlineMatch}, count::Csize_t)
    try
        results = PhoneticOnlineMatch[]
        sizehint!(results, Int(count))
        for index in 1:Int(count)
            item = unsafe_load(pointer, index)
            push!(results, PhoneticOnlineMatch(
                copied_owned_string(item.original_text),
                copied_owned_string(item.normalized_text),
                (Int(item.byte_start), Int(item.byte_end)),
                (Int(item.char_start), Int(item.char_end)),
                Int(item.distance)))
        end
        results
    finally
        ccall(native(:llev_phonetic_online_matches_free), Cvoid,
            (Ptr{RawPhoneticOnlineMatch}, Csize_t), pointer, count)
    end
end

"""Scan an entire UTF-8 document for native non-word-boundary matches."""
function scan(grep::PhoneticOnlineGrep, document::AbstractString;
    max_input_bytes::Integer=1_000_000, max_matches::Integer=100_000)
    max_input_bytes > 0 || throw(ArgumentError("max_input_bytes must be positive"))
    max_matches > 0 || throw(ArgumentError("max_matches must be positive"))
    bytes = text_bytes(document)
    output = Ref{Ptr{RawPhoneticOnlineMatch}}(C_NULL)
    count = Ref{Csize_t}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_online_scan), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Csize_t,
            Ref{Ptr{RawPhoneticOnlineMatch}}, Ref{Csize_t}),
        require_open(grep), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        checked_csize(max_input_bytes, "max_input_bytes"),
        checked_csize(max_matches, "max_matches"), output, count),
        :llev_phonetic_online_scan)
    copy_online_matches(output[], count[])
end

"""A bounded chunk-fed native scanner. `finish!` consumes the scanner once."""
mutable struct PhoneticOnlineStream
    handle::Ptr{Cvoid}
    finished::Bool
    closed::Bool
end

function streaming(grep::PhoneticOnlineGrep; max_total_bytes::Integer=1_000_000)
    max_total_bytes > 0 || throw(ArgumentError("max_total_bytes must be positive"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    checked(ccall(native(:llev_phonetic_online_stream_new), Cint,
        (Ptr{Cvoid}, Csize_t, Ref{Ptr{Cvoid}}), require_open(grep),
        checked_csize(max_total_bytes, "max_total_bytes"), output),
        :llev_phonetic_online_stream_new)
    stream = PhoneticOnlineStream(output[], false, false)
    finalizer(close!, stream)
    stream
end

function require_open(stream::PhoneticOnlineStream)
    stream.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :phonetic_online_stream, "online stream is closed"))
    stream.handle
end

function close!(stream::PhoneticOnlineStream)
    stream.closed && return nothing
    handle = stream.handle
    stream.handle = C_NULL
    stream.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_online_stream_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(stream::PhoneticOnlineStream) = close!(stream)
Base.isopen(stream::PhoneticOnlineStream) = !stream.closed

"""Append a complete UTF-8 chunk; chunks may split phonetic rewrites but not scalars."""
function feed!(stream::PhoneticOnlineStream, chunk::AbstractString)
    stream.finished && throw(ArgumentError("stream was finished"))
    bytes = text_bytes(chunk)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_online_stream_feed), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t), require_open(stream),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes)),
        :llev_phonetic_online_stream_feed)
    stream
end

"""Finalize and copy all matches; a result-limit error still consumes the stream."""
function finish!(stream::PhoneticOnlineStream; max_matches::Integer=100_000)
    stream.finished && throw(ArgumentError("stream was finished"))
    max_matches > 0 || throw(ArgumentError("max_matches must be positive"))
    output = Ref{Ptr{RawPhoneticOnlineMatch}}(C_NULL)
    count = Ref{Csize_t}(0)
    status = ccall(native(:llev_phonetic_online_stream_finish), Cint,
        (Ptr{Cvoid}, Csize_t, Ref{Ptr{RawPhoneticOnlineMatch}}, Ref{Csize_t}),
        require_open(stream), checked_csize(max_matches, "max_matches"), output, count)
    stream.finished = true
    checked(status, :llev_phonetic_online_stream_finish)
    copy_online_matches(output[], count[])
end
