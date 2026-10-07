mutable struct ValueFilterState
    predicate::Any
    failure::Any
end

function value_filter_callback(pointer::Ptr{Cvoid}, has_id::UInt8,
    id::UInt64)::UInt8
    state = unsafe_pointer_to_objref(pointer)::ValueFilterState
    state.failure === nothing || return 0x02
    try
        UInt8(Bool(state.predicate(has_id == 0 ? nothing : id)))
    catch error
        state.failure = (error, catch_backtrace())
        0x02
    end
end

const VALUE_FILTER_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)

"""A closeable native Unicode query that tests values before constructing terms."""
mutable struct FilteredCursor
    inner::QueryCursor
    state::ValueFilterState
    closed::Bool
end

Base.IteratorSize(::Type{FilteredCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{FilteredCursor}) = Base.HasEltype()
Base.eltype(::Type{FilteredCursor}) = Match

"""Filter native Unicode matches by their optional provider ID.

`predicate(id)` receives `UInt64` or `nothing` at the accepted dictionary
node. Rejected terms are not materialized by the native traversal. The result
cursor retains one query-start dictionary snapshot and is in traversal order.
Close or cancel it if iteration stops early.
"""
function query_filtered(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer, predicate)
    0 <= maximum_distance <= typemax(Int) ||
        throw(ArgumentError("maximum distance must be nonnegative and fit Int"))
    state = ValueFilterState(predicate, nothing)
    bytes = text_bytes(input)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve bytes state ccall(native(:llev_transducer_query_filtered_utf8),
        Cint, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid},
            Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes), maximum_distance, VALUE_FILTER_CALLBACK[],
        pointer_from_objref(state), output)
    checked(status, :llev_transducer_query_filtered_utf8)
    cursor = FilteredCursor(QueryCursor(output[]), state, false)
    finalizer(cursor) do item
        try
            close!(item)
        catch
        end
    end
    cursor
end

function throw_filter_failure!(cursor::FilteredCursor)
    failure = cursor.state.failure
    failure === nothing && return nothing
    cursor.state.failure = nothing
    try
        close!(cursor)
    catch
    end
    throw(failure[1])
end

function next_batch!(cursor::FilteredCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :query_filtered,
        "cursor is closed"))
    result = try
        state = cursor.state
        GC.@preserve state next_batch!(cursor.inner, maximum)
    catch
        throw_filter_failure!(cursor)
        rethrow()
    end
    throw_filter_failure!(cursor)
    if result === nothing
        close!(cursor)
    end
    result
end

function Base.iterate(cursor::FilteredCursor, state=nothing)
    cursor.closed && return nothing
    result = try
        callback_state = cursor.state
        GC.@preserve callback_state iterate(cursor.inner)
    catch
        throw_filter_failure!(cursor)
        rethrow()
    end
    throw_filter_failure!(cursor)
    if result === nothing
        close!(cursor)
    end
    result
end

"""Reduce borrowed native filtered batches and close the cursor."""
function reduce_batches!(function_value, initial, cursor::FilteredCursor;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :query_filtered,
        "cursor is closed"))
    try
        callback_state = cursor.state
        result = try
            GC.@preserve callback_state reduce_batches!(function_value, initial,
                cursor.inner; batch_size=batch_size)
        catch
            throw_filter_failure!(cursor)
            rethrow()
        end
        throw_filter_failure!(cursor)
        result
    finally
        close!(cursor)
    end
end

function close!(cursor::FilteredCursor)
    cursor.closed && return nothing
    callback_state = cursor.state
    GC.@preserve callback_state close!(cursor.inner)
    cursor.closed = true
    nothing
end

Base.close(cursor::FilteredCursor) = close!(cursor)
Base.isopen(cursor::FilteredCursor) = !cursor.closed
cancel!(cursor::FilteredCursor) = close!(cursor)
