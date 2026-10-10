"""Borrowed UTF-8 bytes passed to the native dictionary serializer."""
struct RawDictionaryTerm
    data::Ptr{UInt8}
    len::Csize_t
end

"""Native input and output ceilings for one binary dictionary operation."""
struct DictionaryLimits
    max_terms::Csize_t
    max_term_bytes::Csize_t
    max_total_term_bytes::Csize_t
    max_payload_bytes::Csize_t
end

const BincodeDictionaryLimits = DictionaryLimits

const DICTIONARY_FORMAT_BINCODE_V1 = UInt32(1)
const DICTIONARY_FORMAT_PROTOBUF_V1 = UInt32(2)
const DICTIONARY_FORMAT_PROTOBUF_V2 = UInt32(3)
const DICTIONARY_FORMAT_GZIP_BINCODE_V1 = UInt32(4)
const DICTIONARY_FORMAT_GZIP_PROTOBUF_V1 = UInt32(5)
const DICTIONARY_FORMAT_GZIP_PROTOBUF_V2 = UInt32(6)
const DICTIONARY_FORMAT_PROTOBUF_DAT_V1 = UInt32(7)

const DICTIONARY_FORMAT_IDS = Dict(
    :bincode_v1 => DICTIONARY_FORMAT_BINCODE_V1,
    :protobuf_v1 => DICTIONARY_FORMAT_PROTOBUF_V1,
    :protobuf_v2 => DICTIONARY_FORMAT_PROTOBUF_V2,
    :gzip_bincode_v1 => DICTIONARY_FORMAT_GZIP_BINCODE_V1,
    :gzip_protobuf_v1 => DICTIONARY_FORMAT_GZIP_PROTOBUF_V1,
    :gzip_protobuf_v2 => DICTIONARY_FORMAT_GZIP_PROTOBUF_V2,
    :protobuf_dat_v1 => DICTIONARY_FORMAT_PROTOBUF_DAT_V1)

function dictionary_format_id(format::Symbol)
    get(DICTIONARY_FORMAT_IDS, format) do
        throw(ArgumentError("unknown dictionary binary format: $(format)"))
    end
end

function dictionary_limits(; max_terms::Integer=100_000,
    max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)
    limit = DictionaryLimits(
        checked_csize(max_terms, "max_terms"),
        checked_csize(max_term_bytes, "max_term_bytes"),
        checked_csize(max_total_term_bytes, "max_total_term_bytes"),
        checked_csize(max_payload_bytes, "max_payload_bytes"))
    limit.max_payload_bytes >= 8 || throw(ArgumentError("max_payload_bytes must be at least 8"))
    limit.max_term_bytes <= limit.max_total_term_bytes ||
        throw(ArgumentError("max_term_bytes exceeds max_total_term_bytes"))
    limit
end

function require_dictionary_format(format_id::UInt32)
    api_revision() >= 32 && (build_features() & BUILD_FEATURE_SERIALIZATION) != 0 ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :dictionary_serialization,
            "native dictionary formats require API revision 32 and SERIALIZATION"))
    if format_id in (DICTIONARY_FORMAT_PROTOBUF_V1, DICTIONARY_FORMAT_PROTOBUF_V2,
        DICTIONARY_FORMAT_GZIP_PROTOBUF_V1, DICTIONARY_FORMAT_GZIP_PROTOBUF_V2,
        DICTIONARY_FORMAT_PROTOBUF_DAT_V1)
        (build_features() & BUILD_FEATURE_PROTOBUF) != 0 ||
            throw(NativeError(Int32(STATUS_UNSUPPORTED), :dictionary_serialization,
                "native dictionary protobuf requires the PROTOBUF build feature"))
    end
    if format_id in (DICTIONARY_FORMAT_GZIP_BINCODE_V1,
        DICTIONARY_FORMAT_GZIP_PROTOBUF_V1, DICTIONARY_FORMAT_GZIP_PROTOBUF_V2)
        (build_features() & BUILD_FEATURE_COMPRESSION) != 0 ||
            throw(NativeError(Int32(STATUS_UNSUPPORTED), :dictionary_serialization,
                "native dictionary gzip requires the COMPRESSION build feature"))
    end
    nothing
end

function bounded_dictionary_source(source, limits::DictionaryLimits)
    raw = RawDictionaryTerm[]
    bytes = Vector{UInt8}[]
    total = 0
    iterator = source isa AbstractDict ? keys(source) : source
    for term in iterator
        term isa AbstractString || throw(ArgumentError("dictionary terms must be strings"))
        length(raw) < limits.max_terms || throw(ArgumentError("dictionary term count limit"))
        count = ncodeunits(term)
        count <= limits.max_term_bytes || throw(ArgumentError("dictionary term byte limit"))
        total = Base.checked_add(total, count)
        total <= limits.max_total_term_bytes ||
            throw(ArgumentError("dictionary total term byte limit"))
        owned = Vector{UInt8}(codeunits(String(term)))
        push!(bytes, owned)
        push!(raw, RawDictionaryTerm(isempty(owned) ? C_NULL : pointer(owned), length(owned)))
    end
    raw, bytes
end

"""Serialize UTF-8 terms or an `AbstractDict`'s keys using a native binary format.

The native encoder first constructs the accepted byte-dictionary language, so
duplicate terms collapse exactly as they do for Rust dictionaries. The caller
chooses explicit term and payload ceilings. The result owns its binary bytes.
"""
function dictionary_bytes(source; format::Symbol=:bincode_v1,
    max_terms::Integer=100_000, max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)::Vector{UInt8}
    format_id = dictionary_format_id(format)
    require_dictionary_format(format_id)
    limits = dictionary_limits(; max_terms, max_term_bytes,
        max_total_term_bytes, max_payload_bytes)
    raw, anchored = bounded_dictionary_source(source, limits)
    output = Ref(RawOwnedBytes(C_NULL, 0))
    try
        GC.@preserve raw anchored checked(ccall(native(:llev_dictionary_serialize), Cint,
            (UInt32, Ptr{RawDictionaryTerm}, Csize_t,
                Ref{DictionaryLimits}, Ref{RawOwnedBytes}),
            format_id, isempty(raw) ? C_NULL : pointer(raw), length(raw),
            Ref(limits), output), :llev_dictionary_serialize)
        result = output[]
        result.len == 0 && return UInt8[]
        result.data != C_NULL || error("native dictionary returned null nonempty bytes")
        copy(unsafe_wrap(Vector{UInt8}, result.data, Int(result.len); own=false))
    finally
        ccall(native(:llev_owned_bytes_free), Cvoid, (Ref{RawOwnedBytes},), output)
    end
end

"""Closeable, immutable native snapshot of decoded dictionary terms."""
mutable struct DictionaryTerms
    handle::Ptr{Cvoid}
    len::Int
    lock::ReentrantLock
end

const BincodeDictionaryTerms = DictionaryTerms

function Base.close(terms::DictionaryTerms)
    lock(terms.lock) do
        terms.handle == C_NULL && return nothing
        ccall(native(:llev_decoded_dictionary_terms_free), Cvoid,
            (Ptr{Cvoid},), terms.handle)
        terms.handle = C_NULL
    end
    nothing
end

Base.length(terms::DictionaryTerms) = terms.len
Base.IteratorSize(::Type{DictionaryTerms}) = Base.HasLength()
Base.eltype(::Type{DictionaryTerms}) = String

function Base.iterate(terms::DictionaryTerms, index::Int=1)
    lock(terms.lock) do
        terms.handle != C_NULL || throw(ArgumentError("decoded dictionary terms are closed"))
        index > terms.len && return nothing
        view = Ref(RawDictionaryTerm(C_NULL, 0))
        checked(ccall(native(:llev_decoded_dictionary_term_at), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawDictionaryTerm}),
            terms.handle, index - 1, view), :llev_decoded_dictionary_term_at)
        value = view[]
        bytes = value.len == 0 ? UInt8[] :
            copy(unsafe_wrap(Vector{UInt8}, value.data, Int(value.len); own=false))
        (String(bytes), index + 1)
    end
end

"""Decode one complete native dictionary payload under explicit
input, term, and output ceilings. Terms are copied only as the iterator is read;
close its result when finished. Malformed, corrupt, or trailing bytes fail
before native dictionary reconstruction.
"""
function dictionary_terms(source::AbstractVector{UInt8};
    format::Symbol=:bincode_v1, max_terms::Integer=100_000,
    max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=96 * 1024 * 1024)::DictionaryTerms
    format_id = dictionary_format_id(format)
    require_dictionary_format(format_id)
    limits = dictionary_limits(; max_terms, max_term_bytes,
        max_total_term_bytes, max_payload_bytes)
    length(source) <= limits.max_payload_bytes ||
        throw(ArgumentError("dictionary input exceeds max_payload_bytes"))
    input = source isa Vector{UInt8} ? source : collect(UInt8, source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve input checked(ccall(native(:llev_dictionary_deserialize), Cint,
        (UInt32, Ptr{UInt8}, Csize_t, Ref{DictionaryLimits}, Ref{Ptr{Cvoid}}),
        format_id, isempty(input) ? C_NULL : pointer(input),
        length(input), Ref(limits), output), :llev_dictionary_deserialize)
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

"""Encode a byte-domain dictionary in native bincode V1 format."""
function bincode_dictionary_bytes(source; format_version::Integer=1, kwargs...)
    format_version == 1 || throw(NativeError(Int32(STATUS_INVALID_ARGUMENT),
        :bincode_dictionary_bytes, "unsupported bincode format version"))
    dictionary_bytes(source; format=:bincode_v1, kwargs...)
end

"""Decode native bincode V1 bytes into a closeable term snapshot."""
function bincode_dictionary_terms(source::AbstractVector{UInt8};
    format_version::Integer=1, kwargs...)
    format_version == 1 || throw(NativeError(Int32(STATUS_INVALID_ARGUMENT),
        :bincode_dictionary_terms, "unsupported bincode format version"))
    dictionary_terms(source; format=:bincode_v1, kwargs...)
end

"""Encode a byte-domain dictionary with native Protobuf V1 or V2.

V1 is the portable interchange format. V2 is the compact native format.
Set `gzip=true` for the native gzip wrapper over the selected version.
"""
function protobuf_dictionary_bytes(source; version::Integer=1,
    gzip::Bool=false, kwargs...)
    version in (1, 2) || throw(NativeError(Int32(STATUS_INVALID_ARGUMENT),
        :protobuf_dictionary_bytes, "unsupported protobuf format version"))
    format = gzip ? (version == 1 ? :gzip_protobuf_v1 : :gzip_protobuf_v2) :
        (version == 1 ? :protobuf_v1 : :protobuf_v2)
    dictionary_bytes(source; format, kwargs...)
end

"""Decode native Protobuf V1 or V2 bytes into a closeable term snapshot."""
function protobuf_dictionary_terms(source::AbstractVector{UInt8};
    version::Integer=1, gzip::Bool=false, kwargs...)
    version in (1, 2) || throw(NativeError(Int32(STATUS_INVALID_ARGUMENT),
        :protobuf_dictionary_terms, "unsupported protobuf format version"))
    format = gzip ? (version == 1 ? :gzip_protobuf_v1 : :gzip_protobuf_v2) :
        (version == 1 ? :protobuf_v1 : :protobuf_v2)
    dictionary_terms(source; format, kwargs...)
end

"""Encode a byte-domain dictionary with native gzip-wrapped bincode V1."""
gzip_bincode_dictionary_bytes(source; kwargs...) =
    dictionary_bytes(source; format=:gzip_bincode_v1, kwargs...)

"""Decode native gzip-wrapped bincode V1 bytes into a term snapshot."""
gzip_bincode_dictionary_terms(source::AbstractVector{UInt8}; kwargs...) =
    dictionary_terms(source; format=:gzip_bincode_v1, kwargs...)

"""Encode the native DAT-specific Protocol Buffers term format."""
dat_protobuf_dictionary_bytes(source; kwargs...) =
    dictionary_bytes(source; format=:protobuf_dat_v1, kwargs...)

"""Decode native DAT Protocol Buffers bytes into a term snapshot."""
dat_protobuf_dictionary_terms(source::AbstractVector{UInt8}; kwargs...) =
    dictionary_terms(source; format=:protobuf_dat_v1, kwargs...)
