"""Copied native per-token evidence for a token-query match."""
struct PhoneticTokenDetail
    token_index::Int
    byte_range::Tuple{Int,Int}
    original_text::String
    normalized_text::String
    distance::Int
end

"""Copied native token-query match and its per-token details."""
struct PhoneticTokenMatch
    byte_range::Tuple{Int,Int}
    total_distance::Int
    matched_text::String
    details::Vector{PhoneticTokenDetail}
end

struct RawPhoneticTokenDetail
    token_index::Csize_t
    byte_start::Csize_t
    byte_end::Csize_t
    original_text::RawOwnedString
    normalized_text::RawOwnedString
    distance::UInt8
    reserved::NTuple{7,UInt8}
end

struct RawPhoneticTokenMatch
    byte_start::Csize_t
    byte_end::Csize_t
    total_distance::UInt8
    reserved::NTuple{7,UInt8}
    matched_text::RawOwnedString
    details::Ptr{RawPhoneticTokenDetail}
    detail_count::Csize_t
end

"""Reusable native token-aware phonetic grep with its original query grammar."""
mutable struct PhoneticTokenGrep
    handle::Ptr{Cvoid}
    closed::Bool
end

function PhoneticTokenGrep(query::AbstractString;
    rules::Union{Nothing,PhoneticRuleSet}=nothing,
    default_distance::Integer=0, max_query_bytes::Integer=4_096)
    0 <= default_distance <= typemax(UInt8) ||
        throw(ArgumentError("default_distance must fit UInt8"))
    max_query_bytes > 0 || throw(ArgumentError("max_query_bytes must be positive"))
    bytes = text_bytes(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_token_new), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, UInt8, Csize_t, Ref{Ptr{Cvoid}}),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes), rules_handle,
        UInt8(default_distance), checked_csize(max_query_bytes, "max_query_bytes"), output),
        :llev_phonetic_token_new)
    grep = PhoneticTokenGrep(output[], false)
    finalizer(close!, grep)
    grep
end

function require_open(grep::PhoneticTokenGrep)
    grep.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :phonetic_token, "token matcher is closed"))
    grep.handle
end

function close!(grep::PhoneticTokenGrep)
    grep.closed && return nothing
    handle = grep.handle
    grep.handle = C_NULL
    grep.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_token_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(grep::PhoneticTokenGrep) = close!(grep)
Base.isopen(grep::PhoneticTokenGrep) = !grep.closed

"""Scan a UTF-8 document; return copied native matches with per-token details."""
function scan(grep::PhoneticTokenGrep, document::AbstractString;
    max_input_bytes::Integer=1_000_000,
    max_matches::Integer=100_000,
    max_details::Integer=100_000)
    max_input_bytes > 0 || throw(ArgumentError("max_input_bytes must be positive"))
    max_matches > 0 || throw(ArgumentError("max_matches must be positive"))
    max_details > 0 || throw(ArgumentError("max_details must be positive"))
    bytes = text_bytes(document)
    output = Ref{Ptr{RawPhoneticTokenMatch}}(C_NULL)
    count = Ref{Csize_t}(0)
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_token_scan), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Csize_t, Csize_t,
            Ref{Ptr{RawPhoneticTokenMatch}}, Ref{Csize_t}),
        require_open(grep), isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        checked_csize(max_input_bytes, "max_input_bytes"),
        checked_csize(max_matches, "max_matches"),
        checked_csize(max_details, "max_details"), output, count),
        :llev_phonetic_token_scan)
    try
        matches = PhoneticTokenMatch[]
        sizehint!(matches, Int(count[]))
        for index in 1:Int(count[])
            item = unsafe_load(output[], index)
            details = PhoneticTokenDetail[]
            sizehint!(details, Int(item.detail_count))
            for detail_index in 1:Int(item.detail_count)
                detail = unsafe_load(item.details, detail_index)
                push!(details, PhoneticTokenDetail(Int(detail.token_index),
                    (Int(detail.byte_start), Int(detail.byte_end)),
                    copied_owned_string(detail.original_text),
                    copied_owned_string(detail.normalized_text), Int(detail.distance)))
            end
            push!(matches, PhoneticTokenMatch(
                (Int(item.byte_start), Int(item.byte_end)),
                Int(item.total_distance), copied_owned_string(item.matched_text), details))
        end
        matches
    finally
        ccall(native(:llev_phonetic_token_matches_free), Cvoid,
            (Ptr{RawPhoneticTokenMatch}, Csize_t), output[], count[])
    end
end
