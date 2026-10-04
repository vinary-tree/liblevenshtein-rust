"""Native stateful phonetic rewrite transducer with contextual delayed emission."""
mutable struct PhoneticTransducer
    handle::Ptr{Cvoid}
    finished::Bool
    closed::Bool
end

function PhoneticTransducer(; rules::Union{Nothing,PhoneticRuleSet}=nothing)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    rules_handle = rules === nothing ? Ptr{Cvoid}(C_NULL) : require_open(rules)
    checked(ccall(native(:llev_phonetic_transducer_new), Cint,
        (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), rules_handle, output),
        :llev_phonetic_transducer_new)
    transducer = PhoneticTransducer(output[], false, false)
    finalizer(close!, transducer)
    transducer
end

function require_open(transducer::PhoneticTransducer)
    transducer.closed && throw(NativeError(Int32(STATUS_CLOSED),
        :phonetic_transducer, "phonetic transducer is closed"))
    transducer.handle
end

function close!(transducer::PhoneticTransducer)
    transducer.closed && return nothing
    handle = transducer.handle
    transducer.handle = C_NULL
    transducer.closed = true
    handle == C_NULL || ccall(native(:llev_phonetic_transducer_free), Cvoid,
        (Ptr{Cvoid},), handle)
    nothing
end

Base.close(transducer::PhoneticTransducer) = close!(transducer)
Base.isopen(transducer::PhoneticTransducer) = !transducer.closed

function copy_and_free_owned(output::Ref{RawOwnedString})
    try
        copied_owned_string(output[])
    finally
        ccall(native(:llev_owned_string_free), Cvoid,
            (Ref{RawOwnedString},), output)
    end
end

"""Feed a complete UTF-8 chunk; return only characters ready for emission.

On an output-limit error, native state is reset and the logical stream must be
restarted. Contextual rules may emit nothing until a later chunk or `finish!`.
"""
function feed!(transducer::PhoneticTransducer, chunk::AbstractString;
    max_input_scalars::Integer=1_000_000,
    max_output_bytes::Integer=1_000_000)
    transducer.finished && throw(ArgumentError("transducer was finished; call reset!"))
    max_input_scalars > 0 || throw(ArgumentError("max_input_scalars must be positive"))
    max_output_bytes > 0 || throw(ArgumentError("max_output_bytes must be positive"))
    bytes = text_bytes(chunk)
    output = Ref(RawOwnedString(C_NULL, 0))
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_transducer_feed), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Csize_t, Ref{RawOwnedString}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes), checked_csize(max_input_scalars, "max_input_scalars"),
        checked_csize(max_output_bytes, "max_output_bytes"), output),
        :llev_phonetic_transducer_feed)
    copy_and_free_owned(output)
end

"""Flush buffered context once; call `reset!` before a new logical stream."""
function finish!(transducer::PhoneticTransducer;
    max_output_bytes::Integer=1_000_000)
    transducer.finished && throw(ArgumentError("transducer was finished; call reset!"))
    max_output_bytes > 0 || throw(ArgumentError("max_output_bytes must be positive"))
    output = Ref(RawOwnedString(C_NULL, 0))
    checked(ccall(native(:llev_phonetic_transducer_finish), Cint,
        (Ptr{Cvoid}, Csize_t, Ref{RawOwnedString}),
        require_open(transducer), checked_csize(max_output_bytes, "max_output_bytes"),
        output), :llev_phonetic_transducer_finish)
    transducer.finished = true
    copy_and_free_owned(output)
end

"""Reset native state while retaining the compiled rule set."""
function reset!(transducer::PhoneticTransducer)
    checked(ccall(native(:llev_phonetic_transducer_reset), Cint,
        (Ptr{Cvoid},), require_open(transducer)),
        :llev_phonetic_transducer_reset)
    transducer.finished = false
    transducer
end

"""Normalize a separate string without changing incremental stream state."""
function normalize(transducer::PhoneticTransducer, input::AbstractString;
    max_input_scalars::Integer=1_000_000,
    max_output_bytes::Integer=1_000_000)
    max_input_scalars > 0 || throw(ArgumentError("max_input_scalars must be positive"))
    max_output_bytes > 0 || throw(ArgumentError("max_output_bytes must be positive"))
    bytes = text_bytes(input)
    output = Ref(RawOwnedString(C_NULL, 0))
    GC.@preserve bytes checked(ccall(native(:llev_phonetic_transducer_normalize), Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Csize_t, Ref{RawOwnedString}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes), checked_csize(max_input_scalars, "max_input_scalars"),
        checked_csize(max_output_bytes, "max_output_bytes"), output),
        :llev_phonetic_transducer_normalize)
    copy_and_free_owned(output)
end
