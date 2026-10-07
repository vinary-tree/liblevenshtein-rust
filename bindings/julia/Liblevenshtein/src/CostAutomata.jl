struct RawAffineCosts
    gap_open::Cdouble
    gap_extend::Cdouble
    substitution::Cdouble
    scale_denominator::UInt32
    reserved::UInt32
end

"""Exact decimal affine-gap costs validated by the native automaton.

A gap of `k` units costs `gap_open + k * gap_extend`. `scale_denominator=nothing`
selects the least exact decimal scale; an explicit positive denominator must
represent every configured weight exactly. Query budgets use that same scale.
"""
struct AffineGapCosts <: AbstractEditCostDomain
    gap_open::Float64
    gap_extend::Float64
    substitution::Float64
    scale_denominator::UInt32
    function AffineGapCosts(gap_open::Real, gap_extend::Real, substitution::Real;
        scale_denominator::Union{Nothing,Integer}=nothing)
        requested = if scale_denominator === nothing
            UInt32(0)
        else
            1 <= scale_denominator <= typemax(UInt32) ||
                throw(ArgumentError("scale denominator must fit a positive UInt32"))
            UInt32(scale_denominator)
        end
        raw = RawAffineCosts(Float64(gap_open), Float64(gap_extend),
            Float64(substitution), requested, 0)
        denominator = Ref{UInt32}(0)
        checked(ccall(native(:llev_affine_costs_validate), Cint,
            (Ref{RawAffineCosts}, Ref{UInt32}), Ref(raw), denominator),
            :llev_affine_costs_validate)
        new(raw.gap_open, raw.gap_extend, raw.substitution, denominator[])
    end
end

RawAffineCosts(value::AffineGapCosts) = RawAffineCosts(value.gap_open,
    value.gap_extend, value.substitution, value.scale_denominator, 0)

struct RawOperationCostsF64
    match_cost::Cdouble
    substitution::Cdouble
    insertion::Cdouble
    deletion::Cdouble
    transposition::Cdouble
    split::Cdouble
    merge::Cdouble
end

"""Typed native operation costs for float-weighted fuzzy matching.

Every field must be finite and nonnegative; `match_cost` is exactly zero.
The selected transducer algorithm determines which optional edit operations
are available. Use `WeightedOperationCosts(:standard)`, `:typo`, or `:ocr` to
obtain native presets, or pass six custom operation costs.
"""
struct WeightedOperationCosts <: AbstractEditCostDomain
    match_cost::Float64
    substitution::Float64
    insertion::Float64
    deletion::Float64
    transposition::Float64
    split::Float64
    merge::Float64
    function WeightedOperationCosts(substitution::Real, insertion::Real,
        deletion::Real, transposition::Real, split::Real, merge::Real;
        match_cost::Real=0.0)
        raw = RawOperationCostsF64(Float64(match_cost), Float64(substitution),
            Float64(insertion), Float64(deletion), Float64(transposition),
            Float64(split), Float64(merge))
        checked(ccall(native(:llev_operation_costs_validate), Cint,
            (Ref{RawOperationCostsF64},), Ref(raw)),
            :llev_operation_costs_validate)
        new(raw.match_cost, raw.substitution, raw.insertion, raw.deletion,
            raw.transposition, raw.split, raw.merge)
    end
end

function WeightedOperationCosts(preset::Symbol)
    code = if preset === :standard
        UInt32(0)
    elseif preset === :typo
        UInt32(1)
    elseif preset === :ocr
        UInt32(2)
    else
        throw(ArgumentError("unknown weighted cost preset $preset"))
    end
    raw = Ref{RawOperationCostsF64}()
    checked(ccall(native(:llev_operation_costs_preset), Cint,
        (UInt32, Ref{RawOperationCostsF64}), code, raw),
        :llev_operation_costs_preset)
    value = raw[]
    WeightedOperationCosts(value.substitution, value.insertion, value.deletion,
        value.transposition, value.split, value.merge; match_cost=value.match_cost)
end

RawOperationCostsF64(value::WeightedOperationCosts) =
    RawOperationCostsF64(value.match_cost, value.substitution, value.insertion,
        value.deletion, value.transposition, value.split, value.merge)

struct RawCostMatch
    term_data::Ptr{Cvoid}
    term_len::Csize_t
    byte_len::Csize_t
    cost::Cdouble
    scaled_cost::Csize_t
    scale_denominator::UInt32
    unit_domain::UInt32
    has_scaled_cost::UInt8
    reserved::NTuple{7,UInt8}
end

struct RawCostBatch
    matches::Ptr{RawCostMatch}
    len::Csize_t
    generation::UInt64
end

"""Owned result from an affine or weighted native automaton.

`scaled_cost` is present only for affine queries. In that case the exact
distance is `scaled_cost / scale_denominator`; `cost` is its f64 presentation.
"""
struct CostMatch{T}
    term::T
    cost::Float64
    scaled_cost::Union{Nothing,UInt}
    scale_denominator::UInt32
    unit_domain::VTI.UnitDomain
end

function materialize(value::RawCostMatch)
    domain = VTI.UnitDomain(value.unit_domain)
    term = if domain == VTI.UNIT_UNICODE_SCALAR
        value.byte_len == 0 ? "" : unsafe_string(Ptr{UInt8}(value.term_data), value.byte_len)
    elseif domain == VTI.UNIT_BYTE
        value.byte_len == 0 ? UInt8[] : copy(unsafe_wrap(Vector{UInt8},
            Ptr{UInt8}(value.term_data), value.byte_len; own=false))
    elseif domain == VTI.UNIT_U64
        value.term_len == 0 ? UInt64[] : copy(unsafe_wrap(Vector{UInt64},
            Ptr{UInt64}(value.term_data), value.term_len; own=false))
    else
        throw(ArgumentError("unknown native unit domain $(value.unit_domain)"))
    end
    CostMatch(term, value.cost,
        value.has_scaled_cost == 0 ? nothing : UInt(value.scaled_cost),
        value.scale_denominator, domain)
end

mutable struct BorrowedCostBatch
    matches::Ptr{RawCostMatch}
    length::Int
    active::Bool
end

struct BorrowedCostMatch
    batch::BorrowedCostBatch
    index::Int
end

Base.length(batch::BorrowedCostBatch) = batch.active ? batch.length : 0
Base.eltype(::Type{BorrowedCostBatch}) = BorrowedCostMatch
function Base.getindex(batch::BorrowedCostBatch, index::Int)
    batch.active || throw(ArgumentError("borrowed cost batch has expired"))
    checkbounds(1:batch.length, index)
    BorrowedCostMatch(batch, index)
end
Base.iterate(batch::BorrowedCostBatch, index::Int=1) =
    index > length(batch) ? nothing : (batch[index], index + 1)
function materialize(match::BorrowedCostMatch)
    match.batch.active || throw(ArgumentError("borrowed cost match has expired"))
    materialize(unsafe_load(match.batch.matches, match.index))
end

"""Closeable native cost query cursor over one captured dictionary revision."""
mutable struct CostCursor
    handle::Ptr{Cvoid}
    pending::Vector{CostMatch}
    offset::Int
    closed::Bool
end

Base.IteratorSize(::Type{CostCursor}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{CostCursor}) = Base.HasEltype()
Base.eltype(::Type{CostCursor}) = CostMatch

function CostCursor(handle::Ptr{Cvoid})
    handle == C_NULL && throw(ArgumentError("native cost cursor handle is null"))
    cursor = CostCursor(handle, CostMatch[], 1, false)
    finalizer(close!, cursor)
    cursor
end

function require_open(cursor::CostCursor)
    cursor.closed && throw(NativeError(Int32(STATUS_CLOSED), :cost_cursor,
        "cursor is closed"))
    cursor.handle
end

function next_batch!(cursor::CostCursor, maximum::Integer=DEFAULT_MATCH_BATCH)
    maximum > 0 || throw(ArgumentError("maximum batch size must be positive"))
    view = Ref(RawCostBatch(Ptr{RawCostMatch}(C_NULL), 0, 0))
    status = ccall(native(:llev_cost_cursor_next_batch), Cint,
        (Ptr{Cvoid}, Csize_t, Ref{RawCostBatch}), require_open(cursor),
        maximum, view)
    checked(status, :llev_cost_cursor_next_batch; allow_end=true) || return nothing
    batch = view[]
    output = CostMatch[]
    try
        sizehint!(output, batch.len)
        for index in 1:Int(batch.len)
            push!(output, materialize(unsafe_load(batch.matches, index)))
        end
    finally
        checked(ccall(native(:llev_cost_cursor_release_batch), Cint,
            (Ptr{Cvoid}, UInt64), cursor.handle, batch.generation),
            :llev_cost_cursor_release_batch)
    end
    output
end

function Base.iterate(cursor::CostCursor, state=nothing)
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

function close!(cursor::CostCursor)
    cursor.closed && return nothing
    checked(ccall(native(:llev_cost_cursor_free), Cint,
        (Ptr{Cvoid},), cursor.handle), :llev_cost_cursor_free)
    cursor.handle = C_NULL
    cursor.closed = true
    empty!(cursor.pending)
    nothing
end

Base.close(cursor::CostCursor) = close!(cursor)
Base.isopen(cursor::CostCursor) = !cursor.closed
cancel!(cursor::CostCursor) = close!(cursor)

function cost_query(symbol::Symbol, transducer::Transducer,
    input::AbstractString, maximum_cost::Real, costs)
    bytes = text_bytes(input)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    raw = Ref(costs)
    status = GC.@preserve bytes raw ccall(native(symbol), Cint,
        (Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Csize_t, Cdouble,
            Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        require_open(transducer), UInt32(VTI.UNIT_UNICODE_SCALAR),
        isempty(bytes) ? C_NULL : pointer(bytes), length(bytes),
        Float64(maximum_cost),
        Ptr{Cvoid}(Base.unsafe_convert(Ptr{typeof(costs)}, raw)), output)
    checked(status, symbol)
    CostCursor(output[])
end

function cost_query(symbol::Symbol, transducer::Transducer,
    input::AbstractVector{U}, maximum_cost::Real, costs) where {U<:Union{UInt8,UInt64}}
    units = ffi_units(input)
    domain = U == UInt8 ? VTI.UNIT_BYTE : VTI.UNIT_U64
    output = Ref{Ptr{Cvoid}}(C_NULL)
    raw = Ref(costs)
    status = GC.@preserve units raw ccall(native(symbol), Cint,
        (Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Csize_t, Cdouble,
            Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        require_open(transducer), UInt32(domain),
        isempty(units) ? C_NULL : pointer(units), length(units),
        Float64(maximum_cost),
        Ptr{Cvoid}(Base.unsafe_convert(Ptr{typeof(costs)}, raw)), output)
    checked(status, symbol)
    CostCursor(output[])
end

"""Start exact scaled affine matching over Unicode, byte, or u64 dictionaries."""
query_affine(transducer::Transducer, input, maximum_cost::Real,
    costs::AffineGapCosts) = cost_query(:llev_transducer_query_affine,
    transducer, input, maximum_cost, RawAffineCosts(costs))

"""Start float-weighted matching over Unicode, byte, or u64 dictionaries."""
query_weighted(transducer::Transducer, input, maximum_cost::Real,
    costs::WeightedOperationCosts) = cost_query(:llev_transducer_query_weighted,
    transducer, input, maximum_cost, RawOperationCostsF64(costs))

"""Select the native edit-cost kernel by the type of its cost configuration.

Unit costs use `query`, affine costs use exact scaled arithmetic, weighted
operation costs use the native floating kernel, and contextual costs use the
Unicode callback kernel. Result cursors retain their native result types.
"""
query_cost(transducer::Transducer, input, maximum_distance::Integer,
    ::UnitEditCosts; order::QueryOrder=ORDER_TRAVERSAL) =
    query(transducer, input, maximum_distance; order)
query_cost(transducer::Transducer, input, maximum_cost::Real,
    costs::AffineGapCosts) = query_affine(transducer, input, maximum_cost, costs)
query_cost(transducer::Transducer, input, maximum_cost::Real,
    costs::WeightedOperationCosts) = query_weighted(transducer, input, maximum_cost, costs)
query_cost(transducer::Transducer, input::AbstractString, maximum_cost::Real,
    costs::ContextualCosts) = query_contextual(transducer, input, maximum_cost, costs)

function cost_reducer_callback(pointer::Ptr{Cvoid},
    matches::Ptr{RawCostMatch}, len::Csize_t)::Cint
    state = unsafe_pointer_to_objref(pointer)::ReducerState
    batch = BorrowedCostBatch(matches, Int(len), true)
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

const COST_REDUCER_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)

"""Reduce borrowed native cost batches and close the cursor."""
function reduce_batches!(function_value, initial, cursor::CostCursor;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    state = ReducerState(function_value, initial, nothing)
    count = Ref{Csize_t}(0)
    try
        status = GC.@preserve state ccall(native(:llev_cost_cursor_reduce), Cint,
            (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ref{Csize_t}),
            require_open(cursor), batch_size, COST_REDUCER_CALLBACK[],
            pointer_from_objref(state), count)
        checked(status, :llev_cost_cursor_reduce)
        state.failure === nothing || throw(state.failure[1])
        state.accumulator
    finally
        close!(cursor)
    end
end
