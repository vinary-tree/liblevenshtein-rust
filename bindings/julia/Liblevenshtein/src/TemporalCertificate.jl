"""Cumulative limits for a complete exact temporal range certificate."""
struct TemporalCertificateLimits
    search::TemporalSearchLimits
    max_witness_bytes::Csize_t
    max_records::Csize_t
    max_path_bytes::Csize_t
    max_work_units::Csize_t
end

function TemporalCertificateLimits(; search::TemporalSearchLimits=TemporalSearchLimits(),
    max_witness_bytes::Integer=64 * 1024 * 1024,
    max_records::Integer=100_000, max_path_bytes::Integer=64 * 1024 * 1024,
    max_work_units::Integer=200_000_000)
    TemporalCertificateLimits(search, checked_threshold(max_witness_bytes),
        checked_threshold(max_records), checked_threshold(max_path_bytes),
        checked_threshold(max_work_units))
end

"""Exact certificate shape and native resource accounting."""
struct TemporalCertificateInfo
    query_len::Csize_t
    evidence_len::Csize_t
    result_len::Csize_t
    cutoff_native::Float64
    work_units::Csize_t
    path_bytes::Csize_t
    witness_bytes::Csize_t
    snapshot_present::UInt8
    reserved::NTuple{7, UInt8}
    snapshot_identity::NTuple{32, UInt8}
end

struct RawTemporalCertificateEvidenceHeader
    kind::UInt32
    reserved::UInt32
    path_len::Csize_t
    stable_id::UInt64
    lower_bound::Float64
    exact::Float64
    has_exact::UInt8
    survived::UInt8
    reserved_tail::NTuple{6, UInt8}
end

struct RawTemporalCertificateEvidenceView
    header::RawTemporalCertificateEvidenceHeader
    path::Ptr{UInt8}
end

struct RawTemporalCertificateView
    info::TemporalCertificateInfo
    query_bits::Ptr{UInt64}
    evidence::Ptr{RawTemporalCertificateEvidenceView}
    results::Ptr{RawTemporalIndexMatch}
end

"""One ordered K1–K4 native decision with an owned quantized path."""
struct TemporalCertificateEvidence
    header::RawTemporalCertificateEvidenceHeader
    path::Vector{UInt8}
end

"""Caller-owned complete projection for native replay and tamper detection."""
struct TemporalCertificateProjection
    info::TemporalCertificateInfo
    query_bits::Vector{UInt64}
    evidence::Vector{TemporalCertificateEvidence}
    matches::Vector{RawTemporalIndexMatch}
end

"""Native exact range certificate retaining its frozen index snapshot."""
mutable struct TemporalCertificate
    handle::Ptr{Cvoid}
    info::TemporalCertificateInfo
    closed::Bool
    lock::ReentrantLock
end

"""Certify a complete exact range query on a frozen MSM, ERP, TWED, DTW, or
Fréchet index. Every ceiling is cumulative. The certificate survives closure
of the index; failure returns no partial evidence. DTW cutoff is supplied in
public root-distance units and stored in native squared units.
"""
function query_index_certified(index::TemporalIndex,
    query::AbstractVector{<:Real}; cutoff::Real=Inf,
    limits::TemporalCertificateLimits=TemporalCertificateLimits())
    api_revision() >= UInt32(26) ||
        throw(NativeError(Int32(STATUS_UNSUPPORTED),
            :llev_temporal_index_query_certified,
            "native temporal certificates require API revision 26"))
    length(query) <= limits.search.max_series_len ||
        throw(ArgumentError("query exceeds max_series_len"))
    values = Vector{Float64}(query)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = lock(index.lock) do
        index.closed && throw(ArgumentError("temporal index is closed"))
        index.frozen || throw(ArgumentError("temporal index must be frozen"))
        GC.@preserve values ccall(native(:llev_temporal_index_query_certified),
            Cint, (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Float64,
                Ref{TemporalCertificateLimits}, Ref{Ptr{Cvoid}}),
            index.handle, isempty(values) ? C_NULL : pointer(values),
            length(values), Float64(cutoff), Ref(limits), output)
    end
    checked(status, :llev_temporal_index_query_certified)
    info = Ref{TemporalCertificateInfo}()
    try
        checked(ccall(native(:llev_temporal_certificate_info), Cint,
            (Ptr{Cvoid}, Ref{TemporalCertificateInfo}), output[], info),
            :llev_temporal_certificate_info)
    catch
        ccall(native(:llev_temporal_certificate_free), Cvoid,
            (Ptr{Cvoid},), output[])
        rethrow()
    end
    certificate = TemporalCertificate(output[], info[], false, ReentrantLock())
    finalizer(close!, certificate)
    certificate
end

function close!(certificate::TemporalCertificate)
    lock(certificate.lock) do
        if !certificate.closed
            handle = certificate.handle
            certificate.handle = C_NULL
            certificate.closed = true
            handle == C_NULL || ccall(native(:llev_temporal_certificate_free),
                Cvoid, (Ptr{Cvoid},), handle)
        end
        nothing
    end
end
Base.close(certificate::TemporalCertificate) = close!(certificate)
Base.isopen(certificate::TemporalCertificate) = lock(certificate.lock) do
    !certificate.closed
end
certificate_info(certificate::TemporalCertificate) = certificate.info

function certificate_query_bits(certificate::TemporalCertificate)
    lock(certificate.lock) do
        certificate.closed && throw(ArgumentError("temporal certificate is closed"))
        result = Vector{UInt64}(undef, certificate.info.query_len)
        written = Ref{Csize_t}(0)
        GC.@preserve result checked(ccall(
            native(:llev_temporal_certificate_query_bits), Cint,
            (Ptr{Cvoid}, Csize_t, Ptr{UInt64}, Csize_t, Ref{Csize_t}),
            certificate.handle, 0, isempty(result) ? C_NULL : pointer(result),
            length(result), written), :llev_temporal_certificate_query_bits)
        written[] == length(result) || error("incomplete native certificate query")
        result
    end
end

"""Copy one bounded page of exact matches in public distance units."""
function certificate_match_page(certificate::TemporalCertificate,
    start::Integer, maximum::Integer=DEFAULT_MATCH_BATCH)
    0 <= start <= certificate.info.result_len || throw(BoundsError())
    0 < maximum <= 65_536 || throw(ArgumentError("page maximum must be 1..65,536"))
    lock(certificate.lock) do
        certificate.closed && throw(ArgumentError("temporal certificate is closed"))
        capacity = min(maximum, certificate.info.result_len - start)
        raw = Vector{RawTemporalIndexMatch}(undef, capacity)
        written = Ref{Csize_t}(0)
        GC.@preserve raw checked(ccall(native(:llev_temporal_certificate_matches),
            Cint, (Ptr{Cvoid}, Csize_t, Ptr{RawTemporalIndexMatch},
                Csize_t, Ref{Csize_t}), certificate.handle, start,
            isempty(raw) ? C_NULL : pointer(raw), length(raw), written),
            :llev_temporal_certificate_matches)
        resize!(raw, written[])
        [TemporalMatch(item.id, item.distance) for item in raw]
    end
end

"""Copy one bounded K1–K4 evidence record; paths are owned by the caller."""
function certificate_evidence_at(certificate::TemporalCertificate, index::Integer)
    0 <= index < certificate.info.evidence_len || throw(BoundsError())
    lock(certificate.lock) do
        certificate.closed && throw(ArgumentError("temporal certificate is closed"))
        header = Ref{RawTemporalCertificateEvidenceHeader}()
        written = Ref{Csize_t}(0)
        checked(ccall(native(:llev_temporal_certificate_evidence_at), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawTemporalCertificateEvidenceHeader},
                Ptr{UInt8}, Csize_t, Ref{Csize_t}),
            certificate.handle, index, header, C_NULL, 0, written),
            :llev_temporal_certificate_evidence_at)
        path = Vector{UInt8}(undef, written[])
        GC.@preserve path checked(ccall(
            native(:llev_temporal_certificate_evidence_at), Cint,
            (Ptr{Cvoid}, Csize_t, Ref{RawTemporalCertificateEvidenceHeader},
                Ptr{UInt8}, Csize_t, Ref{Csize_t}),
            certificate.handle, index, header,
            isempty(path) ? C_NULL : pointer(path), length(path), written),
            :llev_temporal_certificate_evidence_at)
        written[] == length(path) || error("incomplete native certificate path")
        TemporalCertificateEvidence(header[], path)
    end
end

struct TemporalCertificateEvidenceIterator
    certificate::TemporalCertificate
end
Base.IteratorSize(::Type{TemporalCertificateEvidenceIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TemporalCertificateEvidenceIterator}) = Base.HasEltype()
Base.eltype(::Type{TemporalCertificateEvidenceIterator}) = TemporalCertificateEvidence
function Base.iterate(iterator::TemporalCertificateEvidenceIterator, offset::Int=0)
    offset >= iterator.certificate.info.evidence_len && return nothing
    (certificate_evidence_at(iterator.certificate, offset), offset + 1)
end
certificate_evidence(certificate::TemporalCertificate) =
    TemporalCertificateEvidenceIterator(certificate)

struct TemporalCertificateMatchIterator
    certificate::TemporalCertificate
    page_size::Int
end
Base.IteratorSize(::Type{TemporalCertificateMatchIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TemporalCertificateMatchIterator}) = Base.HasEltype()
Base.eltype(::Type{TemporalCertificateMatchIterator}) = TemporalMatch
function Base.iterate(iterator::TemporalCertificateMatchIterator,
    state::Tuple{Int, Vector{TemporalMatch}, Int}=(0, TemporalMatch[], 1))
    next_offset, page, within = state
    if within > length(page)
        next_offset >= iterator.certificate.info.result_len && return nothing
        page = certificate_match_page(iterator.certificate, next_offset,
            iterator.page_size)
        isempty(page) && return nothing
        next_offset += length(page)
        within = 1
    end
    (page[within], (next_offset, page, within + 1))
end
function certificate_matches(certificate::TemporalCertificate;
    page_size::Integer=DEFAULT_MATCH_BATCH)
    0 < page_size <= 65_536 || throw(ArgumentError("page size must be 1..65,536"))
    TemporalCertificateMatchIterator(certificate, Int(page_size))
end

function reduce_certificate_evidence(function_value, initial,
    certificate::TemporalCertificate)
    accumulator = initial
    for record in certificate_evidence(certificate)
        accumulator = function_value(accumulator, record)
    end
    accumulator
end

"""Materialize an explicitly bounded complete projection for replay."""
function read_certificate(certificate::TemporalCertificate)
    info = certificate.info
    matches = RawTemporalIndexMatch[]
    for match in certificate_matches(certificate)
        push!(matches, RawTemporalIndexMatch(match.id, match.distance))
    end
    TemporalCertificateProjection(info, certificate_query_bits(certificate),
        collect(certificate_evidence(certificate)), matches)
end

"""Replay caller-supplied evidence and results against the retained snapshot.
An altered well-formed projection returns false. Malformed vector lengths also
return false before crossing the native boundary.
"""
function verify_certificate(certificate::TemporalCertificate,
    proposal::TemporalCertificateProjection)
    info = proposal.info
    length(proposal.query_bits) == info.query_len || return false
    length(proposal.evidence) == info.evidence_len || return false
    length(proposal.matches) == info.result_len || return false
    all(length(item.path) == item.header.path_len for item in proposal.evidence) ||
        return false
    views = [RawTemporalCertificateEvidenceView(item.header,
        isempty(item.path) ? C_NULL : pointer(item.path))
        for item in proposal.evidence]
    raw = RawTemporalCertificateView(info,
        isempty(proposal.query_bits) ? C_NULL : pointer(proposal.query_bits),
        isempty(views) ? C_NULL : pointer(views),
        isempty(proposal.matches) ? C_NULL : pointer(proposal.matches))
    lock(certificate.lock) do
        certificate.closed && throw(ArgumentError("temporal certificate is closed"))
        valid = Ref{UInt8}(0)
        GC.@preserve proposal views checked(ccall(
            native(:llev_temporal_certificate_verify), Cint,
            (Ptr{Cvoid}, Ref{RawTemporalCertificateView}, Ref{UInt8}),
            certificate.handle, Ref(raw), valid),
            :llev_temporal_certificate_verify)
        valid[] != 0
    end
end
