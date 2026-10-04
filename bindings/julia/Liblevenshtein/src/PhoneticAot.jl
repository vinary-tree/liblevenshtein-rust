"""An owned native byte buffer; only `llev_owned_bytes_free` releases it."""
struct RawPhoneticBytes
    data::Ptr{UInt8}
    len::Csize_t
end

function require_phonetic_aot()
    api_revision() >= 9 && (build_features() & BUILD_FEATURE_PHONETIC_AOT) != 0 ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :phonetic_aot,
            "native phonetic AOT requires API revision 9 and the PHONETIC_AOT build feature"))
    nothing
end

function compiled_phonetic_bytes(value::Union{PhoneticRuleSet,PhoneticPattern};
    max_output_bytes::Integer=16 * 1024 * 1024)::Vector{UInt8}
    require_phonetic_aot()
    symbol = value isa PhoneticRuleSet ? :llev_phonetic_rules_to_bytes :
        :llev_phonetic_pattern_to_bytes
    output = Ref(RawPhoneticBytes(C_NULL, 0))
    try
        checked(ccall(native(symbol), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawPhoneticBytes}), require_open(value),
            checked_csize(max_output_bytes, "max_output_bytes"), output), symbol)
        bytes = output[]
        bytes.len == 0 && return UInt8[]
        bytes.data != C_NULL || error("native AOT returned null nonempty bytes")
        copy(unsafe_wrap(Vector{UInt8}, bytes.data, Int(bytes.len); own=false))
    finally
        ccall(native(:llev_owned_bytes_free), Cvoid, (Ref{RawPhoneticBytes},), output)
    end
end

function load_compiled_phonetic(type::Type{T}, bytes::AbstractVector{UInt8},
    max_input_bytes::Integer, symbol::Symbol)::T where {T<:Union{PhoneticRuleSet,PhoneticPattern}}
    require_phonetic_aot()
    input = bytes isa Vector{UInt8} ? bytes : collect(UInt8, bytes)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve input checked(ccall(native(symbol), Cint,
        (Ptr{UInt8}, Csize_t, Csize_t, Ref{Ptr{Cvoid}}),
        isempty(input) ? C_NULL : pointer(input), length(input),
        checked_csize(max_input_bytes, "max_input_bytes"), output), symbol)
    value = T(output[], false)
    finalizer(close!, value)
    value
end

"""Restore versioned native `.llev` rule bytes. Rejects wrong magic/version or a
buffer exceeding `max_input_bytes` (hard ceiling 16 MiB). The caller retains
ownership of `bytes`; the resulting native rule set owns its decoded state.
"""
load_compiled_phonetic_rules(bytes::AbstractVector{UInt8};
    max_input_bytes::Integer=16 * 1024 * 1024) =
    load_compiled_phonetic(PhoneticRuleSet, bytes, max_input_bytes,
        :llev_phonetic_rules_from_bytes)

"""Restore versioned native `.llre` pattern bytes with the native NFA-state
ceiling. The caller retains ownership of `bytes`.
"""
load_compiled_phonetic_pattern(bytes::AbstractVector{UInt8};
    max_input_bytes::Integer=16 * 1024 * 1024) =
    load_compiled_phonetic(PhoneticPattern, bytes, max_input_bytes,
        :llev_phonetic_pattern_from_bytes)
