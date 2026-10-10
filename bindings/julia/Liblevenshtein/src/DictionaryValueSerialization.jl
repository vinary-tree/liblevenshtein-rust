"""Caller-selected ceilings for native value-preserving Bincode."""
struct ValuedDictionaryLimits
    max_entries::Csize_t
    max_term_bytes::Csize_t
    max_total_term_bytes::Csize_t
    max_value_bytes::Csize_t
    max_total_value_bytes::Csize_t
    max_payload_bytes::Csize_t
end

struct RawValueEntryInput
    term_data::Ptr{UInt8}
    term_len::Csize_t
    value_data::Ptr{UInt8}
    value_len::Csize_t
    value_u64::UInt64
end

struct RawValueEntryView
    term_data::Ptr{UInt8}
    term_len::Csize_t
    value_data::Ptr{UInt8}
    value_len::Csize_t
    value_u64::UInt64
    value_kind::UInt32
    reserved::UInt32
end

RawValueEntryView() = RawValueEntryView(C_NULL, 0, C_NULL, 0, 0, 0, 0)

const VALUE_PERSIST_U64 = UInt32(1)
const VALUE_PERSIST_BYTES = UInt32(2)
const VALUE_PERSIST_KINDS = Dict(:u64 => VALUE_PERSIST_U64,
    :bytes => VALUE_PERSIST_BYTES)
const VALUE_PERSIST_UNITS = Dict(:byte => UInt32(1), :unicode => UInt32(2))

function value_persist_kind(kind::Symbol)
    get(VALUE_PERSIST_KINDS, kind) do
        throw(ArgumentError("unknown valued Bincode value kind: $(kind)"))
    end
end

function value_persist_unit(domain::Symbol)
    get(VALUE_PERSIST_UNITS, domain) do
        throw(ArgumentError("unknown valued Bincode unit domain: $(domain)"))
    end
end

function valued_dictionary_limits(; max_entries::Integer=100_000,
    max_term_bytes::Integer=1_048_576,
    max_total_term_bytes::Integer=64 * 1024 * 1024,
    max_value_bytes::Integer=1_048_576,
    max_total_value_bytes::Integer=64 * 1024 * 1024,
    max_payload_bytes::Integer=128 * 1024 * 1024)
    limits = ValuedDictionaryLimits(
        checked_csize(max_entries, "max_entries"),
        checked_csize(max_term_bytes, "max_term_bytes"),
        checked_csize(max_total_term_bytes, "max_total_term_bytes"),
        checked_csize(max_value_bytes, "max_value_bytes"),
        checked_csize(max_total_value_bytes, "max_total_value_bytes"),
        checked_csize(max_payload_bytes, "max_payload_bytes"))
    limits.max_term_bytes <= limits.max_total_term_bytes ||
        throw(ArgumentError("max_term_bytes exceeds max_total_term_bytes"))
    limits.max_value_bytes <= limits.max_total_value_bytes ||
        throw(ArgumentError("max_value_bytes exceeds max_total_value_bytes"))
    limits.max_payload_bytes >= 8 ||
        throw(ArgumentError("max_payload_bytes must be at least eight"))
    limits
end

function require_valued_bincode(format_version::Integer)
    format_version == 1 || throw(NativeError(Int32(STATUS_INVALID_ARGUMENT),
        :valued_dictionary_bytes, "unsupported valued Bincode version"))
    api_revision() >= 33 && (build_features() & BUILD_FEATURE_SERIALIZATION) != 0 ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :valued_dictionary_bytes,
            "native valued Bincode requires API revision 33 and SERIALIZATION"))
    nothing
end

function checked_value_u64(value)
    value isa Integer || throw(ArgumentError("u64 dictionary values must be integers"))
    value >= 0 || throw(ArgumentError("u64 dictionary values must be nonnegative"))
    value <= typemax(UInt64) || throw(OverflowError("u64 dictionary value overflow"))
    UInt64(value)
end

function borrowed_value_entries(source, kind::UInt32, limits::ValuedDictionaryLimits)
    raw = RawValueEntryInput[]
    terms = Vector{UInt8}[]
    values = Vector{UInt8}[]
    total_terms = 0
    total_values = 0
    for entry in source
        entry isa Pair || throw(ArgumentError("valued entries must be key => value pairs"))
        key = entry.first
        key isa AbstractString || throw(ArgumentError("valued dictionary keys must be strings"))
        length(raw) < limits.max_entries || throw(ArgumentError("entry count limit"))
        term = Vector{UInt8}(codeunits(String(key)))
        term_len = length(term)
        total_terms = Base.checked_add(total_terms, term_len)
        term_len <= limits.max_term_bytes && total_terms <= limits.max_total_term_bytes ||
            throw(ArgumentError("term byte limit"))
        if kind == VALUE_PERSIST_U64
            limits.max_value_bytes >= 8 || throw(ArgumentError("u64 value byte limit"))
            total_values = Base.checked_add(total_values, 8)
            total_values <= limits.max_total_value_bytes ||
                throw(ArgumentError("total value byte limit"))
            push!(raw, RawValueEntryInput(data_pointer(term), term_len,
                C_NULL, 0, checked_value_u64(entry.second)))
        else
            entry.second isa AbstractVector{UInt8} ||
                throw(ArgumentError("byte dictionary values must be UInt8 vectors"))
            value = Vector{UInt8}(entry.second)
            value_len = length(value)
            total_values = Base.checked_add(total_values, value_len)
            value_len <= limits.max_value_bytes &&
                total_values <= limits.max_total_value_bytes ||
                throw(ArgumentError("value byte limit"))
            push!(values, value)
            push!(raw, RawValueEntryInput(data_pointer(term), term_len,
                data_pointer(value), value_len, 0))
        end
        push!(terms, term)
    end
    raw, terms, values
end

"""Encode a byte or Unicode dictionary with u64 or raw-byte values.

The wire format is native Bincode `Vec<(String, V)>`; `V` is `u64` for
`:u64` and `Vec<u8>` for `:bytes`. Record the value kind and unit domain with
the resulting bytes, because the Bincode payload has no type tag.
"""
function valued_dictionary_bytes(source; value_kind::Symbol,
    unit_domain::Symbol=:byte, format_version::Integer=1,
    kwargs...)::Vector{UInt8}
    require_valued_bincode(format_version)
    kind = value_persist_kind(value_kind)
    domain = value_persist_unit(unit_domain)
    limits = valued_dictionary_limits(; kwargs...)
    raw, terms, values = borrowed_value_entries(source, kind, limits)
    output = Ref(RawOwnedBytes(C_NULL, 0))
    GC.@preserve raw terms values checked(ccall(native(:llev_valued_dictionary_serialize),
        Cint,
        (UInt32, UInt32, Ptr{RawValueEntryInput}, Csize_t,
            Ref{ValuedDictionaryLimits}, Ref{RawOwnedBytes}),
        domain, kind, data_pointer(raw), length(raw), Ref(limits), output),
        :llev_valued_dictionary_serialize)
    try
        encoded = output[]
        encoded.len == 0 && return UInt8[]
        encoded.data != C_NULL || error("native valued Bincode returned null nonempty bytes")
        copy(unsafe_wrap(Vector{UInt8}, encoded.data, Int(encoded.len); own=false))
    finally
        ccall(native(:llev_owned_bytes_free), Cvoid, (Ref{RawOwnedBytes},), output)
    end
end

"""Closeable native snapshot of decoded valued dictionary entries."""
mutable struct ValuedDictionaryEntries{V}
    handle::Ptr{Cvoid}
    len::Int
    value_kind::UInt32
    lock::ReentrantLock
end

function Base.close(entries::ValuedDictionaryEntries)
    lock(entries.lock) do
        entries.handle == C_NULL && return nothing
        ccall(native(:llev_decoded_value_entries_free), Cvoid,
            (Ptr{Cvoid},), entries.handle)
        entries.handle = C_NULL
    end
    nothing
end

Base.length(entries::ValuedDictionaryEntries) = entries.len
Base.IteratorSize(::Type{<:ValuedDictionaryEntries}) = Base.HasLength()
Base.eltype(::Type{ValuedDictionaryEntries{V}}) where {V} = Pair{String,V}

function copy_valued_term(data::Ptr{UInt8}, len::Csize_t)
    len == 0 && return ""
    data != C_NULL || error("native valued entry has null nonempty term")
    String(copy(unsafe_wrap(Vector{UInt8}, data, Int(len); own=false)))
end

function Base.iterate(entries::ValuedDictionaryEntries{V}, index::Int=1) where {V}
    lock(entries.lock) do
        entries.handle != C_NULL || throw(ArgumentError("valued dictionary entries are closed"))
        index > entries.len && return nothing
        output = Ref(RawValueEntryView())
        checked(ccall(native(:llev_decoded_value_entry_at), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawValueEntryView}),
            entries.handle, index - 1, output), :llev_decoded_value_entry_at)
        view = output[]
        view.value_kind == entries.value_kind ||
            error("native valued dictionary entry kind mismatch")
        term = copy_valued_term(view.term_data, view.term_len)
        value = if V === UInt64
            view.value_u64
        else
            view.value_len == 0 ? UInt8[] :
                copy(unsafe_wrap(Vector{UInt8}, view.value_data,
                    Int(view.value_len); own=false))
        end
        (term => value, index + 1)
    end
end

"""Decode one complete native valued Bincode payload under explicit ceilings."""
function valued_dictionary_entries(source::AbstractVector{UInt8};
    value_kind::Symbol, unit_domain::Symbol=:byte,
    format_version::Integer=1, kwargs...)
    require_valued_bincode(format_version)
    kind = value_persist_kind(value_kind)
    domain = value_persist_unit(unit_domain)
    limits = valued_dictionary_limits(; kwargs...)
    length(source) <= limits.max_payload_bytes ||
        throw(ArgumentError("valued Bincode input exceeds max_payload_bytes"))
    input = source isa Vector{UInt8} ? source : collect(UInt8, source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve input checked(ccall(native(:llev_valued_dictionary_deserialize), Cint,
        (UInt32, UInt32, Ptr{UInt8}, Csize_t,
            Ref{ValuedDictionaryLimits}, Ref{Ptr{Cvoid}}),
        domain, kind, data_pointer(input), length(input), Ref(limits), output),
        :llev_valued_dictionary_deserialize)
    count = Ref{Csize_t}(0)
    try
        checked(ccall(native(:llev_decoded_value_entries_len), Cint,
            (Ptr{Cvoid}, Ref{Csize_t}), output[], count),
            :llev_decoded_value_entries_len)
    catch
        ccall(native(:llev_decoded_value_entries_free), Cvoid,
            (Ptr{Cvoid},), output[])
        rethrow()
    end
    V = kind == VALUE_PERSIST_U64 ? UInt64 : Vector{UInt8}
    result = ValuedDictionaryEntries{V}(output[], Int(count[]), kind, ReentrantLock())
    finalizer(close, result)
    result
end
