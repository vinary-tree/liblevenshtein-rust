struct RawApproxMsmIndexConfig
    segments::Csize_t
    candidate_limit::Csize_t
    split_merge_cost::Float64
    max_entries::Csize_t
    max_total_samples::Csize_t
    max_series_len::Csize_t
    max_total_features::Csize_t
end

struct RawApproxMsmNeighbor
    id::UInt64
    insertion_index::Csize_t
    distance::Float64
end

struct RawApproxMsmOutcome
    kind::UInt32
    reason::UInt32
    neighbor_count::Csize_t
    indexed_entries::Csize_t
    candidate_entries::Csize_t
    exact_reranked::Csize_t
    dp_cells::Csize_t
    work_units::Csize_t
    scratch_bytes::Csize_t
    candidates::Csize_t
    results::Csize_t
end

"""One exact MSM score from a feature-ranked candidate set."""
struct ApproxMsmNeighbor
    id::UInt64
    insertion_index::Csize_t
    distance::Float64
end

"""Tagged approximate MSM outcome with exact-neighbor and recall evidence.

Only `:exhaustive` proves exact top-k recall. `:advisory` may be empty despite
available neighbors. `:incomplete` may carry exact partial neighbors, but
cannot establish recall. Coverage counts exact MSM rerank decisions.
"""
struct ApproxMsmResult
    kind::Symbol
    neighbors::Vector{ApproxMsmNeighbor}
    indexed_entries::Csize_t
    candidate_entries::Csize_t
    exact_reranked::Csize_t
    reason::Symbol
    usage::NamedTuple
end

Base.length(result::ApproxMsmResult) = length(result.neighbors)
Base.isempty(result::ApproxMsmResult) = isempty(result.neighbors)
Base.IteratorSize(::Type{ApproxMsmResult}) = Base.HasLength()
Base.IteratorEltype(::Type{ApproxMsmResult}) = Base.HasEltype()
Base.eltype(::Type{ApproxMsmResult}) = ApproxMsmNeighbor
Base.iterate(result::ApproxMsmResult, state::Int=1) =
    state > length(result.neighbors) ? nothing :
    (result.neighbors[state], state + 1)

proves_recall(result::ApproxMsmResult) =
    result.kind === :exhaustive &&
    result.candidate_entries == result.indexed_entries &&
    result.exact_reranked == result.indexed_entries

"""Fold bounded batches of already certified exact neighbor scores."""
function reduce_batches!(function_value, initial, result::ApproxMsmResult;
    batch_size::Integer=DEFAULT_MATCH_BATCH)
    size = checked_threshold(batch_size)
    0 < size <= 65_536 ||
        throw(ArgumentError("batch_size must be 1..65,536"))
    accumulator = initial
    position = 1
    while position <= length(result.neighbors)
        last_position = min(length(result.neighbors), position + Int(size) - 1)
        accumulator = function_value(accumulator,
            result.neighbors[position:last_position])
        position = last_position + 1
    end
    accumulator
end

"""Frozen native PAA selector with exact MSM reranking of admitted episodes."""
mutable struct ApproxMsmIndex
    handle::Ptr{Cvoid}
    lock::ReentrantLock
    entries::Csize_t
    max_series_len::Csize_t
    closed::Bool
end

function ApproxMsmIndex(entries;
    segments::Integer=16, candidate_limit::Integer=128,
    split_merge_cost::Real=1.0, max_entries::Integer=100_000,
    max_total_samples::Integer=1_000_000,
    max_series_len::Integer=1_000_000,
    max_total_features::Integer=1_600_000)
    api_revision() >= UInt32(20) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_approx_msm_index_new,
            "approximate MSM indexes require native API revision 20"))
    parts = checked_threshold(segments)
    pool = checked_threshold(candidate_limit)
    entry_limit = checked_threshold(max_entries)
    sample_limit = checked_threshold(max_total_samples)
    series_limit = checked_threshold(max_series_len)
    feature_limit = checked_threshold(max_total_features)
    cost = Float64(split_merge_cost)
    isfinite(cost) && cost >= 0.0 ||
        throw(ArgumentError("MSM split/merge cost must be finite and nonnegative"))
    raw = RawApproxMsmIndexConfig(parts, pool, cost, entry_limit,
        sample_limit, series_limit, feature_limit)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(native(:llev_approx_msm_index_new), Cint,
        (Ref{RawApproxMsmIndexConfig}, Ref{Ptr{Cvoid}}),
        Ref(raw), output)
    checked(status, :llev_approx_msm_index_new)
    index = ApproxMsmIndex(output[], ReentrantLock(), 0, series_limit, false)
    finalizer(close!, index)
    samples_used = UInt(0)
    try
        for entry in entries
            entry isa Pair || entry isa Tuple && length(entry) == 2 ||
                throw(ArgumentError("approximate MSM entries must be ID => samples pairs"))
            id = first(entry)
            samples = last(entry)
            id isa Integer && 0 <= id <= typemax(UInt64) ||
                throw(ArgumentError("approximate MSM ID must fit UInt64"))
            samples isa AbstractVector{<:Real} ||
                throw(ArgumentError("approximate MSM samples must be a real vector"))
            index.entries < entry_limit ||
                throw(ArgumentError("approximate MSM source exceeds max_entries"))
            length(samples) <= series_limit ||
                throw(ArgumentError("approximate MSM series exceeds max_series_len"))
            length(samples) <= sample_limit - samples_used ||
                throw(ArgumentError("approximate MSM source exceeds max_total_samples"))
            parts == 0 || index.entries + 1 <= feature_limit ÷ parts ||
                throw(ArgumentError("approximate MSM source exceeds max_total_features"))
            values = Float64.(samples)
            all(isfinite, values) ||
                throw(ArgumentError("approximate MSM samples must be finite"))
            position = Ref{Csize_t}(0)
            status = GC.@preserve values ccall(
                native(:llev_approx_msm_index_insert), Cint,
                (Ptr{Cvoid}, UInt64, Ptr{Float64}, Csize_t, Ref{Csize_t}),
                index.handle, UInt64(id),
                isempty(values) ? C_NULL : pointer(values),
                length(values), position)
            checked(status, :llev_approx_msm_index_insert)
            position[] == index.entries ||
                error("native approximate MSM insertion position changed")
            index.entries += 1
            samples_used += length(values)
        end
        status = ccall(native(:llev_approx_msm_index_freeze), Cint,
            (Ptr{Cvoid},), index.handle)
        checked(status, :llev_approx_msm_index_freeze)
        index
    catch
        close!(index)
        rethrow()
    end
end

Base.length(index::ApproxMsmIndex) = Int(index.entries)
Base.isempty(index::ApproxMsmIndex) = index.entries == 0

function close!(index::ApproxMsmIndex)
    lock(index.lock) do
        index.closed && return nothing
        handle = index.handle
        index.handle = C_NULL
        index.closed = true
        handle == C_NULL || ccall(native(:llev_approx_msm_index_free),
            Cvoid, (Ptr{Cvoid},), handle)
        nothing
    end
end
Base.close(index::ApproxMsmIndex) = close!(index)
Base.isopen(index::ApproxMsmIndex) = lock(index.lock) do
    !index.closed
end

"""Run strict bounded feature selection and exact MSM reranking."""
function query_approx_msm_knn(index::ApproxMsmIndex,
    query::AbstractVector{<:Real}, k::Integer;
    limits::TemporalSearchLimits=TemporalSearchLimits())
    lock(index.lock) do
        index.closed && throw(ArgumentError("approximate MSM index is closed"))
        length(query) <= limits.max_series_len ||
            throw(ArgumentError("approximate MSM query exceeds max_series_len"))
        count = checked_threshold(k)
        values = Float64.(query)
        all(isfinite, values) ||
            throw(ArgumentError("approximate MSM query samples must be finite"))
        capacity = min(count, index.entries)
        neighbors = Vector{RawApproxMsmNeighbor}(undef, capacity)
        output = isempty(neighbors) ? Ptr{RawApproxMsmNeighbor}(C_NULL) :
            pointer(neighbors)
        raw = Ref(RawApproxMsmOutcome(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        status = GC.@preserve values neighbors ccall(
            native(:llev_approx_msm_index_query_knn), Cint,
            (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Csize_t,
                Ref{TemporalSearchLimits}, Ptr{RawApproxMsmNeighbor},
                Csize_t, Ref{RawApproxMsmOutcome}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), count, Ref(limits), output, capacity, raw)
        checked(status, :llev_approx_msm_index_query_knn)
        outcome = raw[]
        kind = outcome.kind == 1 ? :exhaustive :
            outcome.kind == 2 ? :advisory :
            outcome.kind == 3 ? :incomplete :
            error("unknown native approximate MSM outcome kind")
        result = [ApproxMsmNeighbor(neighbors[i].id,
            neighbors[i].insertion_index, neighbors[i].distance)
            for i in 1:Int(outcome.neighbor_count)]
        usage = (dp_cells=outcome.dp_cells, work_units=outcome.work_units,
            scratch_bytes=outcome.scratch_bytes, candidates=outcome.candidates,
            results=outcome.results)
        ApproxMsmResult(kind, result, outcome.indexed_entries,
            outcome.candidate_entries, outcome.exact_reranked,
            kind === :incomplete ? native_index_reason(outcome.reason) : :none,
            usage)
    end
end
