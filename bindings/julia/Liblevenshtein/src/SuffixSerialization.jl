"""Closeable native snapshot of suffix-automaton source texts."""
const SuffixSourceTexts = DictionaryTerms

const SUFFIX_FORMAT_BINCODE_V1 = UInt32(1)
const SUFFIX_FORMAT_PROTOBUF_V1 = UInt32(2)

function suffix_format_id(format::Symbol)
    format === :bincode_v1 && return SUFFIX_FORMAT_BINCODE_V1
    format === :protobuf_v1 && return SUFFIX_FORMAT_PROTOBUF_V1
    throw(ArgumentError("unknown suffix source format: $(format)"))
end

"""Serialize the indexed source texts of a native suffix automaton.

The result uses the native Bincode V1 or Protobuf V1 suffix format. Source
texts remain in insertion order; the accepted substring language is not
enumerated. All resource ceilings apply before native construction.
"""
function suffix_source_bytes(texts; format::Symbol=:bincode_v1,
    max_terms::Integer=100_000, max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)::Vector{UInt8}
    format_id = suffix_format_id(format)
    require_dictionary_format(format_id)
    limits = dictionary_limits(; max_terms, max_term_bytes,
        max_total_term_bytes, max_payload_bytes)
    raw, anchored = bounded_dictionary_source(texts, limits)
    output = Ref(RawOwnedBytes(C_NULL, 0))
    try
        GC.@preserve raw anchored checked(ccall(native(:llev_suffix_source_serialize), Cint,
            (UInt32, Ptr{RawDictionaryTerm}, Csize_t,
                Ref{DictionaryLimits}, Ref{RawOwnedBytes}),
            format_id, isempty(raw) ? C_NULL : pointer(raw), length(raw),
            Ref(limits), output), :llev_suffix_source_serialize)
        result = output[]
        result.len == 0 && return UInt8[]
        result.data != C_NULL || error("native suffix serializer returned null bytes")
        copy(unsafe_wrap(Vector{UInt8}, result.data, Int(result.len); own=false))
    finally
        ccall(native(:llev_owned_bytes_free), Cvoid, (Ref{RawOwnedBytes},), output)
    end
end

"""Decode native suffix bytes into a closeable immutable source-text snapshot."""
function suffix_source_texts(source::AbstractVector{UInt8};
    format::Symbol=:bincode_v1, max_terms::Integer=100_000,
    max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)::SuffixSourceTexts
    format_id = suffix_format_id(format)
    require_dictionary_format(format_id)
    limits = dictionary_limits(; max_terms, max_term_bytes,
        max_total_term_bytes, max_payload_bytes)
    length(source) <= limits.max_payload_bytes ||
        throw(ArgumentError("suffix source input exceeds max_payload_bytes"))
    input = source isa Vector{UInt8} ? source : collect(UInt8, source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve input checked(ccall(native(:llev_suffix_source_deserialize), Cint,
        (UInt32, Ptr{UInt8}, Csize_t, Ref{DictionaryLimits}, Ref{Ptr{Cvoid}}),
        format_id, isempty(input) ? C_NULL : pointer(input), length(input),
        Ref(limits), output), :llev_suffix_source_deserialize)
    count = Ref{Csize_t}(0)
    try
        checked(ccall(native(:llev_decoded_dictionary_terms_len), Cint,
            (Ptr{Cvoid}, Ref{Csize_t}), output[], count),
            :llev_decoded_dictionary_terms_len)
    catch
        ccall(native(:llev_decoded_dictionary_terms_free), Cvoid,
            (Ptr{Cvoid},), output[])
        rethrow()
    end
    result = DictionaryTerms(output[], Int(count[]), ReentrantLock())
    finalizer(close, result)
    result
end
