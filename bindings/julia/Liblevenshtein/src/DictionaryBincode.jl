"""Borrowed UTF-8 bytes passed to the native dictionary bincode bridge."""
struct RawBincodeTerm
    data::Ptr{UInt8}
    len::Csize_t
end

"""Hard native input and output ceilings for one bincode operation."""
struct BincodeDictionaryLimits
    max_terms::Csize_t
    max_term_bytes::Csize_t
    max_total_term_bytes::Csize_t
    max_payload_bytes::Csize_t
end

function bincode_limits(; max_terms::Integer=100_000,
    max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)
    limit = BincodeDictionaryLimits(
        checked_csize(max_terms, "max_terms"),
        checked_csize(max_term_bytes, "max_term_bytes"),
        checked_csize(max_total_term_bytes, "max_total_term_bytes"),
        checked_csize(max_payload_bytes, "max_payload_bytes"))
    limit.max_payload_bytes >= 8 || throw(ArgumentError("max_payload_bytes must be at least 8"))
    limit.max_term_bytes <= limit.max_total_term_bytes ||
        throw(ArgumentError("max_term_bytes exceeds max_total_term_bytes"))
    limit
end

function require_bincode_dictionary()
    api_revision() >= 32 && (build_features() & BUILD_FEATURE_SERIALIZATION) != 0 ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :dictionary_bincode,
            "native dictionary bincode requires API revision 32 and the SERIALIZATION build feature"))
    nothing
end

function bounded_bincode_source(source, limits::BincodeDictionaryLimits)
    raw = RawBincodeTerm[]
    bytes = Vector{UInt8}[]
    total = 0
    iterator = source isa AbstractDict ? keys(source) : source
    for term in iterator
        term isa AbstractString || throw(ArgumentError("dictionary bincode terms must be strings"))
        length(raw) < limits.max_terms || throw(ArgumentError("dictionary bincode term count limit"))
        count = ncodeunits(term)
        count <= limits.max_term_bytes || throw(ArgumentError("dictionary bincode term byte limit"))
        total = Base.checked_add(total, count)
        total <= limits.max_total_term_bytes ||
            throw(ArgumentError("dictionary bincode total term byte limit"))
        owned = Vector{UInt8}(codeunits(String(term)))
        push!(bytes, owned)
        push!(raw, RawBincodeTerm(isempty(owned) ? C_NULL : pointer(owned), length(owned)))
    end
    raw, bytes
end

"""Serialize UTF-8 terms or an `AbstractDict`'s keys using native bincode V1.

The native encoder first constructs the accepted byte-dictionary language, so
duplicate terms collapse exactly as they do for Rust dictionaries. The caller
chooses explicit term and payload ceilings. The result owns its binary bytes.
"""
function bincode_dictionary_bytes(source; format_version::Integer=1,
    max_terms::Integer=100_000, max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)::Vector{UInt8}
    require_bincode_dictionary()
    limits = bincode_limits(; max_terms, max_term_bytes,
        max_total_term_bytes, max_payload_bytes)
    raw, anchored = bounded_bincode_source(source, limits)
    output = Ref(RawOwnedBytes(C_NULL, 0))
    try
        GC.@preserve raw anchored checked(ccall(native(:llev_dictionary_bincode_serialize), Cint,
            (UInt32, Ptr{RawBincodeTerm}, Csize_t,
                Ref{BincodeDictionaryLimits}, Ref{RawOwnedBytes}),
            UInt32(format_version), isempty(raw) ? C_NULL : pointer(raw), length(raw),
            Ref(limits), output), :llev_dictionary_bincode_serialize)
        result = output[]
        result.len == 0 && return UInt8[]
        result.data != C_NULL || error("native bincode returned null nonempty bytes")
        copy(unsafe_wrap(Vector{UInt8}, result.data, Int(result.len); own=false))
    finally
        ccall(native(:llev_owned_bytes_free), Cvoid, (Ref{RawOwnedBytes},), output)
    end
end

"""Closeable, immutable native snapshot of decoded bincode terms."""
mutable struct BincodeDictionaryTerms
    handle::Ptr{Cvoid}
    len::Int
    lock::ReentrantLock
end

function Base.close(terms::BincodeDictionaryTerms)
    lock(terms.lock) do
        terms.handle == C_NULL && return nothing
        ccall(native(:llev_decoded_bincode_terms_free), Cvoid,
            (Ptr{Cvoid},), terms.handle)
        terms.handle = C_NULL
    end
    nothing
end

Base.length(terms::BincodeDictionaryTerms) = terms.len
Base.IteratorSize(::Type{BincodeDictionaryTerms}) = Base.HasLength()
Base.eltype(::Type{BincodeDictionaryTerms}) = String

function Base.iterate(terms::BincodeDictionaryTerms, index::Int=1)
    lock(terms.lock) do
        terms.handle != C_NULL || throw(ArgumentError("decoded bincode terms are closed"))
        index > terms.len && return nothing
        view = Ref(RawBincodeTerm(C_NULL, 0))
        checked(ccall(native(:llev_decoded_bincode_term_at), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawBincodeTerm}),
            terms.handle, index - 1, view), :llev_decoded_bincode_term_at)
        value = view[]
        bytes = value.len == 0 ? UInt8[] :
            copy(unsafe_wrap(Vector{UInt8}, value.data, Int(value.len); own=false))
        (String(bytes), index + 1)
    end
end

"""Decode one complete native bincode V1 dictionary payload under explicit
input, term, and output ceilings. Terms are copied only as the iterator is read;
close its result when finished. Malformed, corrupt, or trailing bytes fail
before native dictionary reconstruction.
"""
function bincode_dictionary_terms(source::AbstractVector{UInt8};
    format_version::Integer=1, max_terms::Integer=100_000,
    max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)::BincodeDictionaryTerms
    require_bincode_dictionary()
    limits = bincode_limits(; max_terms, max_term_bytes,
        max_total_term_bytes, max_payload_bytes)
    length(source) <= limits.max_payload_bytes ||
        throw(ArgumentError("dictionary bincode input exceeds max_payload_bytes"))
    input = source isa Vector{UInt8} ? source : collect(UInt8, source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve input checked(ccall(native(:llev_dictionary_bincode_deserialize), Cint,
        (UInt32, Ptr{UInt8}, Csize_t, Ref{BincodeDictionaryLimits}, Ref{Ptr{Cvoid}}),
        UInt32(format_version), isempty(input) ? C_NULL : pointer(input),
        length(input), Ref(limits), output), :llev_dictionary_bincode_deserialize)
    count = Ref{Csize_t}(0)
    try
        checked(ccall(native(:llev_decoded_bincode_terms_len), Cint,
            (Ptr{Cvoid}, Ref{Csize_t}), output[], count),
            :llev_decoded_bincode_terms_len)
    catch
        ccall(native(:llev_decoded_bincode_terms_free), Cvoid,
            (Ptr{Cvoid},), output[])
        rethrow()
    end
    result = BincodeDictionaryTerms(output[], Int(count[]), ReentrantLock())
    finalizer(close, result)
    result
end
