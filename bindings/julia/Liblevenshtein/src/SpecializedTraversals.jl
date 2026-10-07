"""Context-dependent edit costs supplied by Julia callbacks.

Each callback runs synchronously on the thread advancing the native cursor.
`nothing` forbids an edit. All nonzero allowed costs must be at least
`minimum_nonzero_cost`, which must be finite and strictly positive. A false
lower bound can cause native pruning to omit matches.
"""
struct ContextualCosts <: AbstractEditCostDomain
    substitution::Any
    insertion::Any
    deletion::Any
    minimum_nonzero_cost::Float64
    function ContextualCosts(substitution, insertion, deletion;
        minimum_nonzero_cost::Real)
        minimum = Float64(minimum_nonzero_cost)
        isfinite(minimum) && minimum > 0 ||
            throw(ArgumentError("minimum_nonzero_cost must be finite and positive"))
        new(substitution, insertion, deletion, minimum)
    end
end

mutable struct TraversalCallbackState
    callbacks::Any
    failure::Any
    generation::UInt64
    active_prefix_depth::Int
end

"""A scalar sequence borrowed only during one host traversal callback."""
struct BorrowedScalars <: AbstractVector{Char}
    data::Ptr{UInt32}
    length::Int
    state::TraversalCallbackState
    generation::UInt64
end

Base.IndexStyle(::Type{BorrowedScalars}) = IndexLinear()
Base.size(value::BorrowedScalars) = (value.length,)
function Base.getindex(value::BorrowedScalars, index::Int)
    value.state.generation == value.generation ||
        throw(ArgumentError("borrowed scalar context has expired"))
    @boundscheck checkbounds(value, index)
    Char(unsafe_load(value.data, index))
end

"""Query and dictionary-left context for one native edit operation.

`query_index` is one-based. `query` and `dictionary_prefix` are borrowed scalar
views that expire when the callback returns. `dictionary_unit` is the current
edge scalar, or `nothing` when the operation has no dictionary edge.
"""
struct ContextualEditContext
    query::BorrowedScalars
    query_index::Int
    dictionary_prefix::BorrowedScalars
    dictionary_unit::Union{Nothing,Char}
end

struct RawEditContext
    query_units::Ptr{UInt32}
    query_len::Csize_t
    query_index::Csize_t
    dictionary_prefix::Ptr{UInt32}
    prefix_len::Csize_t
    dictionary_unit::UInt32
    has_dictionary_unit::UInt8
    reserved::NTuple{3,UInt8}
end

function contextual_cost_callback(pointer::Ptr{Cvoid}, operation::UInt32,
    view_pointer::Ptr{RawEditContext}, query_unit::UInt32,
    candidate_unit::UInt32)::Cdouble
    state = unsafe_pointer_to_objref(pointer)::TraversalCallbackState
    state.failure === nothing || return NaN
    state.generation += 1
    generation = state.generation
    try
        view = unsafe_load(view_pointer)
        context = ContextualEditContext(
            BorrowedScalars(view.query_units, Int(view.query_len), state, generation),
            Int(view.query_index) + 1,
            BorrowedScalars(view.dictionary_prefix, Int(view.prefix_len), state, generation),
            view.has_dictionary_unit == 0 ? nothing : Char(view.dictionary_unit))
        costs = state.callbacks::ContextualCosts
        result = if operation == 0
            costs.substitution(context, Char(query_unit), Char(candidate_unit))
        elseif operation == 1
            costs.insertion(context, Char(candidate_unit))
        elseif operation == 2
            costs.deletion(context, Char(query_unit))
        else
            throw(ArgumentError("unknown native contextual operation"))
        end
        result === nothing ? NaN : Float64(result)
    catch error
        state.failure = (error, catch_backtrace())
        NaN
    finally
        state.generation += 1
    end
end

const CONTEXTUAL_COST_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)

"""Balanced Julia visitor for a native prefix-pruned fuzzy DFS.

`enter(unit, depth)` and `leave(unit, depth)` are paired even for rejected
subtrees. `matches(candidate, query)` defaults to exact scalar equality.
`permits(prefix)` decides terminal membership and `score(prefix)` may return a
number or `nothing`. Prefix views expire when each callback returns.
"""
struct PrefixVisitor
    enter::Any
    leave::Any
    matches::Any
    permits::Any
    score::Any
end

PrefixVisitor(enter, leave; matches=(candidate, query) -> candidate == query,
    permits=(_prefix) -> true, score=(_prefix) -> nothing) =
    PrefixVisitor(enter, leave, matches, permits, score)

function prefix_callback(pointer::Ptr{Cvoid}, operation::UInt32,
    candidate_unit::UInt32, query_unit::UInt32, depth::Csize_t,
    prefix_pointer::Ptr{UInt32}, prefix_len::Csize_t,
    out_score::Ptr{Cdouble})::UInt8
    state = unsafe_pointer_to_objref(pointer)::TraversalCallbackState
    (state.failure === nothing || operation == 2) || return 0x00
    (operation != 2 || Int(depth) <= state.active_prefix_depth) || return 0x00
    state.generation += 1
    generation = state.generation
    try
        visitor = state.callbacks::PrefixVisitor
        if operation == 0
            return UInt8(Bool(visitor.matches(Char(candidate_unit), Char(query_unit))))
        elseif operation == 1
            state.active_prefix_depth = Int(depth)
            return UInt8(Bool(visitor.enter(Char(candidate_unit), Int(depth))))
        elseif operation == 2
            visitor.leave(Char(candidate_unit), Int(depth))
            return 0x00
        elseif operation == 3 || operation == 4
            prefix = BorrowedScalars(prefix_pointer, Int(prefix_len), state, generation)
            if operation == 3
                return UInt8(Bool(visitor.permits(prefix)))
            end
            score = visitor.score(prefix)
            score === nothing && return 0x00
            unsafe_store!(out_score, Float64(score))
            return 0x01
        end
        throw(ArgumentError("unknown native prefix operation"))
    catch error
        state.failure === nothing && (state.failure = (error, catch_backtrace()))
        0x00
    finally
        if operation == 2
            state.active_prefix_depth = Int(depth) - 1
        end
        state.generation += 1
    end
end

const PREFIX_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)

struct RawSpecializedMatch
    term_data::Ptr{UInt8}
    byte_len::Csize_t
    cost::Cdouble
    score::Cdouble
    has_score::UInt8
    reserved::NTuple{7,UInt8}
end

struct RawSpecializedBatch
    matches::Ptr{RawSpecializedMatch}
    len::Csize_t
    generation::UInt64
end

"""An owned Unicode result from a contextual or prefix-pruned traversal."""
struct SpecializedMatch
    term::String
    cost::Float64
    score::Union{Nothing,Float64}
end

mutable struct BorrowedSpecializedBatch
    matches::Ptr{RawSpecializedMatch}
    length::Int
    active::Bool
end

struct BorrowedSpecializedMatch
    batch::BorrowedSpecializedBatch
    index::Int
end

Base.length(batch::BorrowedSpecializedBatch) = batch.active ? batch.length : 0
Base.eltype(::Type{BorrowedSpecializedBatch}) = BorrowedSpecializedMatch
function Base.getindex(batch::BorrowedSpecializedBatch, index::Int)
    batch.active || throw(ArgumentError("borrowed specialized batch has expired"))
    checkbounds(1:batch.length, index)
    BorrowedSpecializedMatch(batch, index)
end
function Base.iterate(batch::BorrowedSpecializedBatch, index::Int=1)
    index > length(batch) ? nothing : (batch[index], index + 1)
end

function materialize(match::BorrowedSpecializedMatch)
    match.batch.active || throw(ArgumentError("borrowed specialized match has expired"))
    materialize(unsafe_load(match.batch.matches, match.index))
end

function materialize(raw::RawSpecializedMatch)
    term = raw.byte_len == 0 ? "" : unsafe_string(raw.term_data, raw.byte_len)
    SpecializedMatch(term, raw.cost, raw.has_score == 0 ? nothing : raw.score)
end

"""A closeable native specialized cursor over one query-start snapshot."""
mutable struct SpecializedCursor
    handle::Ptr{Cvoid}
    state::TraversalCallbackState
    pending::Vector{SpecializedMatch}
    offset::Int
    closed::Bool
end

Base.IteratorSize(::Type{SpecializedCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{SpecializedCursor}) = Base.HasEltype()
Base.eltype(::Type{SpecializedCursor}) = SpecializedMatch

function SpecializedCursor(handle::Ptr{Cvoid}, state::TraversalCallbackState)
    handle == C_NULL && throw(ArgumentError("specialized cursor handle is null"))
    cursor = SpecializedCursor(handle, state, SpecializedMatch[], 1, false)
    finalizer(cursor) do item
        try
            close!(item)
        catch
        end
    end
    cursor
end

function require_open(cursor::SpecializedCursor)
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :specialized_cursor,
        "cursor is closed"))
    cursor.handle
end

function throw_callback_failure!(cursor::SpecializedCursor)
    failure = cursor.state.failure
    failure === nothing && return nothing
    cursor.state.failure = nothing
    try
        close!(cursor)
    catch
    end
    throw(failure[1])
end

"""Start a native context-dependent Unicode query with a bounded cursor."""
function query_contextual(transducer::Transducer, input::AbstractString,
    maximum_cost::Real, costs::ContextualCosts)
    max_cost = Float64(maximum_cost)
    isfinite(max_cost) && max_cost >= 0 ||
        throw(ArgumentError("maximum contextual cost must be finite and nonnegative"))
    state = TraversalCallbackState(costs, nothing, 0, 0)
    bytes = text_bytes(input)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve bytes state ccall(native(:llev_transducer_query_contextual_utf8),
        Cint, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Cdouble, Cdouble,
            Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes), max_cost, costs.minimum_nonzero_cost,
        CONTEXTUAL_COST_CALLBACK[], pointer_from_objref(state), output)
    if state.failure !== nothing
        if output[] != C_NULL
            GC.@preserve state ccall(native(:llev_specialized_cursor_free), Cint,
                (Ptr{Cvoid},), output[])
        end
        throw(state.failure[1])
    end
    checked(status, :llev_transducer_query_contextual_utf8)
    SpecializedCursor(output[], state)
end

"""Start a native Unicode fuzzy DFS with a balanced prefix visitor."""
function query_pruned(transducer::Transducer, input::AbstractString,
    maximum_distance::Integer, visitor::PrefixVisitor)
    0 <= maximum_distance <= typemax(Int) ||
        throw(ArgumentError("maximum distance must be nonnegative and fit Int"))
    state = TraversalCallbackState(visitor, nothing, 0, 0)
    bytes = text_bytes(input)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve bytes state ccall(native(:llev_transducer_query_pruned_utf8),
        Cint, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Csize_t,
            Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        require_open(transducer), isempty(bytes) ? C_NULL : pointer(bytes),
        length(bytes), maximum_distance,
        PREFIX_CALLBACK[], pointer_from_objref(state), output)
    checked(status, :llev_transducer_query_pruned_utf8)
    SpecializedCursor(output[], state)
end

function next_batch!(cursor::SpecializedCursor,
    maximum::Integer=DEFAULT_MATCH_BATCH)
    maximum > 0 || throw(ArgumentError("maximum batch size must be positive"))
    view = Ref(RawSpecializedBatch(Ptr{RawSpecializedMatch}(C_NULL), 0, 0))
    callback_state = cursor.state
    status = GC.@preserve callback_state ccall(native(:llev_specialized_cursor_next_batch),
        Cint, (Ptr{Cvoid}, Csize_t, Ref{RawSpecializedBatch}),
        require_open(cursor), maximum, view)
    if status == Cint(STATUS_END)
        throw_callback_failure!(cursor)
        return nothing
    end
    if status != Cint(STATUS_OK)
        throw_callback_failure!(cursor)
        checked(status, :llev_specialized_cursor_next_batch)
    end
    batch = view[]
    output = SpecializedMatch[]
    try
        sizehint!(output, batch.len)
        for index in 1:Int(batch.len)
            push!(output, materialize(unsafe_load(batch.matches, index)))
        end
    finally
        checked(ccall(native(:llev_specialized_cursor_release_batch),
            Cint, (Ptr{Cvoid}, UInt64), cursor.handle, batch.generation),
            :llev_specialized_cursor_release_batch)
    end
    throw_callback_failure!(cursor)
    output
end

function Base.iterate(cursor::SpecializedCursor, state=nothing)
    cursor.closed && return nothing
    if cursor.offset > length(cursor.pending)
        batch = next_batch!(cursor)
        if batch === nothing
            close!(cursor)
            return nothing
        end
        cursor.pending = batch
        cursor.offset = 1
    end
    value = cursor.pending[cursor.offset]
    cursor.offset += 1
    (value, nothing)
end

function specialized_reducer_callback(pointer::Ptr{Cvoid},
    matches::Ptr{RawSpecializedMatch}, len::Csize_t)::Cint
    state = unsafe_pointer_to_objref(pointer)::ReducerState
    batch = BorrowedSpecializedBatch(matches, Int(len), true)
    try
        state.accumulator = state.function_value(state.accumulator, batch)
        Cint(STATUS_OK)
    catch error
        state.failure = (error, catch_backtrace())
        Cint(STATUS_END)
    finally
        batch.active = false
    end
end

const SPECIALIZED_REDUCER_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)

"""Reduce native borrowed specialized batches and close the cursor."""
function reduce_batches!(function_value, initial, cursor::SpecializedCursor;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    state = ReducerState(function_value, initial, nothing)
    count = Ref{Csize_t}(0)
    status = Cint(STATUS_OK)
    try
        callback_state = cursor.state
        GC.@preserve callback_state state begin
            status = ccall(native(:llev_specialized_cursor_reduce), Cint,
                (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ref{Csize_t}),
                require_open(cursor), batch_size, SPECIALIZED_REDUCER_CALLBACK[],
                pointer_from_objref(state), count)
        end
        throw_callback_failure!(cursor)
        checked(status, :llev_specialized_cursor_reduce)
        state.failure === nothing || throw(state.failure[1])
        state.accumulator
    finally
        close!(cursor)
    end
end

function close!(cursor::SpecializedCursor)
    cursor.closed && return nothing
    callback_state = cursor.state
    status = GC.@preserve callback_state ccall(native(:llev_specialized_cursor_free),
        Cint, (Ptr{Cvoid},), cursor.handle)
    checked(status, :llev_specialized_cursor_free)
    cursor.handle = C_NULL
    cursor.closed = true
    empty!(cursor.pending)
    if cursor.state.failure !== nothing
        failure = cursor.state.failure
        cursor.state.failure = nothing
        throw(failure[1])
    end
    nothing
end

Base.close(cursor::SpecializedCursor) = close!(cursor)
Base.isopen(cursor::SpecializedCursor) = !cursor.closed
cancel!(cursor::SpecializedCursor) = close!(cursor)
