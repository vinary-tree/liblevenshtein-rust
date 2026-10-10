"""Caller-selected native limits for generalized operation-set persistence."""
struct OperationSetLimits
    max_payload_bytes::Csize_t
    max_operations::Csize_t
    max_operation_name_bytes::Csize_t
    max_restriction_pairs_per_operation::Csize_t
    max_total_restriction_pairs::Csize_t
    max_restriction_text_bytes::Csize_t
end

const OPERATION_SET_FORMAT_BINARY_V1 = UInt32(1)
const OPERATION_SET_FORMAT_PROTOBUF_V1 = UInt32(2)
const OPERATION_SET_FORMAT_GZIP_BINARY_V1 = UInt32(3)
const OPERATION_SET_FORMAT_GZIP_PROTOBUF_V1 = UInt32(4)

const OPERATION_SET_FORMAT_IDS = Dict(
    :binary_v1 => OPERATION_SET_FORMAT_BINARY_V1,
    :protobuf_v1 => OPERATION_SET_FORMAT_PROTOBUF_V1,
    :gzip_binary_v1 => OPERATION_SET_FORMAT_GZIP_BINARY_V1,
    :gzip_protobuf_v1 => OPERATION_SET_FORMAT_GZIP_PROTOBUF_V1)

function operation_set_format_id(format::Symbol)
    get(OPERATION_SET_FORMAT_IDS, format) do
        throw(ArgumentError("unknown operation-set binary format: $(format)"))
    end
end

function operation_set_limits(; max_payload_bytes::Integer=64 * 1024 * 1024,
    max_operations::Integer=4096, max_operation_name_bytes::Integer=1024,
    max_restriction_pairs_per_operation::Integer=1_048_576,
    max_total_restriction_pairs::Integer=1_048_576,
    max_restriction_text_bytes::Integer=64 * 1024 * 1024)
    OperationSetLimits(
        checked_csize(max_payload_bytes, "max_payload_bytes"),
        checked_csize(max_operations, "max_operations"),
        checked_csize(max_operation_name_bytes, "max_operation_name_bytes"),
        checked_csize(max_restriction_pairs_per_operation,
            "max_restriction_pairs_per_operation"),
        checked_csize(max_total_restriction_pairs, "max_total_restriction_pairs"),
        checked_csize(max_restriction_text_bytes, "max_restriction_text_bytes"))
end

function require_operation_set_format(format_id::UInt32)
    api_revision() >= 33 && (build_features() & BUILD_FEATURE_SERIALIZATION) != 0 ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED), :operation_set_serialization,
            "native operation-set formats require API revision 33 and SERIALIZATION"))
    if format_id in (OPERATION_SET_FORMAT_PROTOBUF_V1,
        OPERATION_SET_FORMAT_GZIP_PROTOBUF_V1)
        (build_features() & BUILD_FEATURE_PROTOBUF) != 0 ||
            throw(NativeError(Int32(STATUS_UNSUPPORTED), :operation_set_serialization,
                "native operation-set protobuf requires PROTOBUF"))
    end
    if format_id in (OPERATION_SET_FORMAT_GZIP_BINARY_V1,
        OPERATION_SET_FORMAT_GZIP_PROTOBUF_V1)
        (build_features() & BUILD_FEATURE_COMPRESSION) != 0 ||
            throw(NativeError(Int32(STATUS_UNSUPPORTED), :operation_set_serialization,
                "native operation-set gzip requires COMPRESSION"))
    end
    nothing
end

struct RawSerializedOperationView
    consume_source::Csize_t
    consume_target::Csize_t
    weight::Cdouble
    name_data::Ptr{UInt8}
    name_len::Csize_t
    applicability::UInt32
    restriction_count::Csize_t
end

RawSerializedOperationView() = RawSerializedOperationView(0, 0, 0, C_NULL, 0, 0, 0)

struct RawSerializedRestrictionView
    kind::UInt32
    source_byte::UInt8
    target_byte::UInt8
    reserved::NTuple{2,UInt8}
    source_data::Ptr{UInt8}
    source_len::Csize_t
    target_data::Ptr{UInt8}
    target_len::Csize_t
end

RawSerializedRestrictionView() = RawSerializedRestrictionView(
    0, 0, 0, (0, 0), C_NULL, 0, C_NULL, 0)

"""One exact raw-byte restriction from a persisted native operation set."""
struct SerializedByteRestriction
    source::UInt8
    target::UInt8
end

"""One immutable operation copied from a decoded native set."""
struct SerializedOperation
    consume_source::Int
    consume_target::Int
    weight::Float64
    name::String
    applicability::OperationApplicability
    restrictions::Tuple{Vararg{Union{GeneralizedRestriction,SerializedByteRestriction}}}
end

"""Closeable native snapshot of a versioned generalized operation set."""
mutable struct OperationSetSnapshot
    handle::Ptr{Cvoid}
    len::Int
    lock::ReentrantLock
end

function Base.close(snapshot::OperationSetSnapshot)
    lock(snapshot.lock) do
        snapshot.handle == C_NULL && return nothing
        ccall(native(:llev_decoded_operation_set_free), Cvoid,
            (Ptr{Cvoid},), snapshot.handle)
        snapshot.handle = C_NULL
    end
    nothing
end

Base.length(snapshot::OperationSetSnapshot) = snapshot.len
Base.IteratorSize(::Type{OperationSetSnapshot}) = Base.HasLength()
Base.eltype(::Type{OperationSetSnapshot}) = SerializedOperation

function copied_native_utf8(data::Ptr{UInt8}, len::Csize_t)
    len == 0 && return ""
    data != C_NULL || error("native operation-set view has null nonempty text")
    String(copy(unsafe_wrap(Vector{UInt8}, data, Int(len); own=false)))
end

function Base.iterate(snapshot::OperationSetSnapshot, index::Int=1)
    lock(snapshot.lock) do
        snapshot.handle != C_NULL || throw(ArgumentError("operation-set snapshot is closed"))
        index > snapshot.len && return nothing
        raw = Ref(RawSerializedOperationView())
        checked(ccall(native(:llev_decoded_operation_set_operation_at), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawSerializedOperationView}),
            snapshot.handle, index - 1, raw),
            :llev_decoded_operation_set_operation_at)
        view = raw[]
        restrictions = Union{GeneralizedRestriction,SerializedByteRestriction}[]
        sizehint!(restrictions, Int(view.restriction_count))
        for pair_index in 0:(Int(view.restriction_count) - 1)
            pair = Ref(RawSerializedRestrictionView())
            checked(ccall(native(:llev_decoded_operation_set_restriction_at), Cint,
                (Ptr{Cvoid}, Csize_t, Csize_t, Ref{RawSerializedRestrictionView}),
                snapshot.handle, index - 1, pair_index, pair),
                :llev_decoded_operation_set_restriction_at)
            item = pair[]
            if item.kind == 1
                push!(restrictions,
                    SerializedByteRestriction(item.source_byte, item.target_byte))
            elseif item.kind == 2
                push!(restrictions, GeneralizedRestriction(
                    copied_native_utf8(item.source_data, item.source_len),
                    copied_native_utf8(item.target_data, item.target_len)))
            else
                error("native operation-set restriction has unknown kind")
            end
        end
        operation = SerializedOperation(Int(view.consume_source),
            Int(view.consume_target), Float64(view.weight),
            copied_native_utf8(view.name_data, view.name_len),
            OperationApplicability(view.applicability), Tuple(restrictions))
        (operation, index + 1)
    end
end

"""Copy a text-only persisted grammar into the Unicode automaton model.

Raw-byte restrictions have no Unicode-scalar interpretation and are rejected.
The native snapshot remains open and independently owned.
"""
function GeneralizedOperationSet(snapshot::OperationSetSnapshot)
    operations = GeneralizedOperation[]
    for operation in snapshot
        all(pair -> pair isa GeneralizedRestriction, operation.restrictions) ||
            throw(ArgumentError("raw-byte restrictions cannot configure a Unicode automaton"))
        push!(operations, GeneralizedOperation(operation.consume_source,
            operation.consume_target, operation.weight, operation.name;
            applicability=operation.applicability,
            restrictions=operation.restrictions))
    end
    GeneralizedOperationSet(operations)
end

function take_operation_set_bytes!(output::Ref{RawOwnedBytes})
    try
        value = output[]
        value.len == 0 && return UInt8[]
        value.data != C_NULL || error("native operation-set returned null nonempty bytes")
        copy(unsafe_wrap(Vector{UInt8}, value.data, Int(value.len); own=false))
    finally
        ccall(native(:llev_owned_bytes_free), Cvoid, (Ref{RawOwnedBytes},), output)
    end
end

"""Encode a generalized edit grammar with a native versioned codec."""
function operation_set_bytes(source::GeneralizedOperationSet;
    format::Symbol=:binary_v1, kwargs...)::Vector{UInt8}
    format_id = operation_set_format_id(format)
    require_operation_set_format(format_id)
    limits = operation_set_limits(; kwargs...)
    raw, names, sources, targets, restrictions = marshal_operations(source)
    output = Ref(RawOwnedBytes(C_NULL, 0))
    GC.@preserve raw names sources targets restrictions checked(ccall(
        native(:llev_operation_set_serialize), Cint,
        (UInt32, Ptr{RawGeneralizedOperation}, Csize_t,
            Ref{OperationSetLimits}, Ref{RawOwnedBytes}),
        format_id, data_pointer(raw), length(raw), Ref(limits), output),
        :llev_operation_set_serialize)
    take_operation_set_bytes!(output)
end

"""Re-encode a native snapshot, preserving byte and string restrictions."""
function operation_set_bytes(source::OperationSetSnapshot;
    format::Symbol=:binary_v1, kwargs...)::Vector{UInt8}
    format_id = operation_set_format_id(format)
    require_operation_set_format(format_id)
    limits = operation_set_limits(; kwargs...)
    lock(source.lock) do
        source.handle != C_NULL || throw(ArgumentError("operation-set snapshot is closed"))
        output = Ref(RawOwnedBytes(C_NULL, 0))
        checked(ccall(native(:llev_decoded_operation_set_serialize), Cint,
            (Ptr{Cvoid}, UInt32, Ref{OperationSetLimits}, Ref{RawOwnedBytes}),
            source.handle, format_id, Ref(limits), output),
            :llev_decoded_operation_set_serialize)
        take_operation_set_bytes!(output)
    end
end

"""Decode one complete native payload into an owned operation-set snapshot."""
function operation_set_snapshot(source::AbstractVector{UInt8};
    format::Symbol=:binary_v1, kwargs...)::OperationSetSnapshot
    format_id = operation_set_format_id(format)
    require_operation_set_format(format_id)
    limits = operation_set_limits(; kwargs...)
    length(source) <= limits.max_payload_bytes ||
        throw(ArgumentError("operation-set input exceeds max_payload_bytes"))
    input = source isa Vector{UInt8} ? source : collect(UInt8, source)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve input checked(ccall(native(:llev_operation_set_deserialize), Cint,
        (UInt32, Ptr{UInt8}, Csize_t, Ref{OperationSetLimits}, Ref{Ptr{Cvoid}}),
        format_id, data_pointer(input), length(input), Ref(limits), output),
        :llev_operation_set_deserialize)
    count = Ref{Csize_t}(0)
    try
        checked(ccall(native(:llev_decoded_operation_set_len), Cint,
            (Ptr{Cvoid}, Ref{Csize_t}), output[], count),
            :llev_decoded_operation_set_len)
    catch
        ccall(native(:llev_decoded_operation_set_free), Cvoid,
            (Ptr{Cvoid},), output[])
        rethrow()
    end
    snapshot = OperationSetSnapshot(output[], Int(count[]), ReentrantLock())
    finalizer(close, snapshot)
    snapshot
end
