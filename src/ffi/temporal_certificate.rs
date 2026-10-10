//! Replayable, bounded evidence for frozen scalar temporal range indexes.

use super::{
    index::{boundary, slice},
    temporal_index::{
        Frozen, IndexState, LlevTemporalIndex, LlevTemporalIndexMatch, LlevTemporalSearchLimits,
    },
    LlevStatus,
};
use crate::time_series::{
    ElasticCertificateError, ElasticCertificateLimits, ElasticRangeCertificate,
    ElasticRangeEvidence, ResourceLimits, TemporalValidationError,
};

/// Cumulative native range-certificate ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalCertificateLimits {
    /// Common query, traversal, and result ceilings.
    pub search: LlevTemporalSearchLimits,
    /// Maximum charged logical witness storage.
    pub max_witness_bytes: usize,
    /// Maximum K1--K4 evidence records.
    pub max_records: usize,
    /// Maximum total quantized-path bytes.
    pub max_path_bytes: usize,
    /// Certificate-specific cumulative work ceiling.
    pub max_work_units: usize,
}

impl From<LlevTemporalCertificateLimits> for ElasticCertificateLimits {
    fn from(raw: LlevTemporalCertificateLimits) -> Self {
        let mut resources = ResourceLimits::from(raw.search);
        resources.max_witness_bytes = raw.max_witness_bytes;
        Self {
            resources,
            max_records: raw.max_records,
            max_path_bytes: raw.max_path_bytes,
            max_work_units: raw.max_work_units,
        }
    }
}

/// Shape and accounting of one complete native certificate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalCertificateInfo {
    /// Number of exact query IEEE-754 words.
    pub query_len: usize,
    /// Number of ordered K1--K4 decisions.
    pub evidence_len: usize,
    /// Number of complete exact range matches.
    pub result_len: usize,
    /// Native kernel cutoff; DTW uses squared-distance units.
    pub cutoff_native: f64,
    /// Charged native work.
    pub work_units: usize,
    /// Total quantized-path bytes.
    pub path_bytes: usize,
    /// Total charged logical witness bytes.
    pub witness_bytes: usize,
    /// One when bound to a complete persistent snapshot.
    pub snapshot_present: u8,
    /// Must be zero.
    pub reserved: [u8; 7],
    /// SHA-256 snapshot identity, or all zero for an in-memory index.
    pub snapshot_identity: [u8; 32],
}

/// One K1--K4 decision; its quantized path is transferred separately.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalCertificateEvidenceHeader {
    /// 1 prefix, 2 subtree, 3 terminal, 4 candidate, 5 exact candidate.
    pub kind: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Number of bytes in the ordered quantized path.
    pub path_len: usize,
    /// Candidate identity for kinds 4 and 5; zero otherwise.
    pub stable_id: u64,
    /// K1, K2, terminal, or K4 bound in native kernel units.
    pub lower_bound: f64,
    /// Exact K3 distance when `has_exact` is one.
    pub exact: f64,
    /// One when the exact candidate carries a finite score.
    pub has_exact: u8,
    /// One when the exact candidate survived the closed cutoff.
    pub survived: u8,
    /// Must be zero.
    pub reserved_tail: [u8; 6],
}

/// One caller-supplied evidence record for replay verification.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalCertificateEvidenceView {
    /// Scalar decision fields.
    pub header: LlevTemporalCertificateEvidenceHeader,
    /// `header.path_len` readable bytes, or null when the length is zero.
    pub path: *const u8,
}

/// Caller-supplied complete certificate and result projection for replay.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalCertificateView {
    /// Exact identity, counts, cutoff, and accounting.
    pub info: LlevTemporalCertificateInfo,
    /// `info.query_len` exact IEEE-754 words.
    pub query_bits: *const u64,
    /// `info.evidence_len` ordered evidence records.
    pub evidence: *const LlevTemporalCertificateEvidenceView,
    /// `info.result_len` exact public-unit matches.
    pub results: *const LlevTemporalIndexMatch,
}

/// Owned frozen snapshot, exact matches, and replayable evidence.
pub struct LlevTemporalRangeCertificate {
    frozen: Frozen,
    results: Vec<(u64, f64)>,
    certificate: ElasticRangeCertificate<f64>,
    limits: ElasticCertificateLimits,
    squared_dtw: bool,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

fn certificate_error(error: ElasticCertificateError) -> (LlevStatus, String) {
    let status = match error {
        ElasticCertificateError::Validation(TemporalValidationError::SeriesTooLong { .. })
        | ElasticCertificateError::BudgetExceeded { .. }
        | ElasticCertificateError::ArithmeticOverflow { .. }
        | ElasticCertificateError::AllocationFailed { .. }
        | ElasticCertificateError::NumericOverflow => LlevStatus::LimitExceeded,
        ElasticCertificateError::Validation(_) => LlevStatus::InvalidArgument,
        ElasticCertificateError::Unsupported => LlevStatus::Unsupported,
        ElasticCertificateError::InvalidStoredData => LlevStatus::ProviderError,
    };
    (status, error.to_string())
}

fn info(handle: &LlevTemporalRangeCertificate) -> LlevTemporalCertificateInfo {
    let certificate = &handle.certificate;
    LlevTemporalCertificateInfo {
        query_len: certificate.query_bits.len(),
        evidence_len: certificate.evidence.len(),
        result_len: handle.results.len(),
        cutoff_native: certificate.cutoff,
        work_units: certificate.work_units,
        path_bytes: certificate.path_bytes,
        witness_bytes: certificate.witness_bytes,
        snapshot_present: u8::from(certificate.snapshot_identity.is_some()),
        reserved: [0; 7],
        snapshot_identity: certificate
            .snapshot_identity
            .map_or([0; 32], |identity| identity.0),
    }
}

fn evidence_fields(
    evidence: &ElasticRangeEvidence<f64>,
) -> (LlevTemporalCertificateEvidenceHeader, &[u8]) {
    let mut header = LlevTemporalCertificateEvidenceHeader::default();
    let path = match evidence {
        ElasticRangeEvidence::PrefixPruned {
            quantized_path,
            lower_bound,
        } => {
            header.kind = 1;
            header.lower_bound = *lower_bound;
            quantized_path
        }
        ElasticRangeEvidence::SubtreePruned {
            quantized_path,
            lower_bound,
        } => {
            header.kind = 2;
            header.lower_bound = *lower_bound;
            quantized_path
        }
        ElasticRangeEvidence::TerminalPruned {
            quantized_path,
            lower_bound,
        } => {
            header.kind = 3;
            header.lower_bound = *lower_bound;
            quantized_path
        }
        ElasticRangeEvidence::CandidatePruned {
            quantized_path,
            stable_id,
            candidate_bound,
        } => {
            header.kind = 4;
            header.stable_id = *stable_id;
            header.lower_bound = *candidate_bound;
            quantized_path
        }
        ElasticRangeEvidence::ExactCandidate {
            quantized_path,
            stable_id,
            candidate_bound,
            exact,
            survived,
        } => {
            header.kind = 5;
            header.stable_id = *stable_id;
            header.lower_bound = *candidate_bound;
            header.exact = exact.unwrap_or(0.0);
            header.has_exact = u8::from(exact.is_some());
            header.survived = u8::from(*survived);
            quantized_path
        }
    };
    header.path_len = path.len();
    (header, path)
}

/// Search one frozen index and retain complete native K1--K4 evidence.
///
/// # Safety
/// All pointers must be valid for their declared lengths and disjoint from
/// the writable output. The index must not be freed concurrently with this
/// call; the returned certificate retains its own frozen snapshot thereafter.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_query_certified(
    index: *const LlevTemporalIndex,
    query: *const f64,
    query_len: usize,
    cutoff: f64,
    raw_limits: *const LlevTemporalCertificateLimits,
    out_certificate: *mut *mut LlevTemporalRangeCertificate,
) -> LlevStatus {
    boundary(|| {
        let output = out_certificate
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "certificate output is null".into()))?;
        *output = std::ptr::null_mut();
        let index = index
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal index is null".into()))?;
        let query = slice(query, query_len, "certificate query")?;
        let limits = ElasticCertificateLimits::from(*raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "certificate limits are null".into(),
        ))?);
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(invalid("certificate cutoff must be nonnegative"));
        }
        let frozen = match &index.state {
            IndexState::Frozen(value) => value,
            _ => {
                return Err(invalid(
                    "temporal index must be frozen before certification",
                ))
            }
        };
        let squared_dtw = matches!(frozen, Frozen::Dtw(_));
        let native_cutoff = if squared_dtw { cutoff * cutoff } else { cutoff };
        if cutoff.is_finite() && !native_cutoff.is_finite() {
            return Err((
                LlevStatus::LimitExceeded,
                "squared DTW certificate cutoff overflows".into(),
            ));
        }
        let (results, certificate) = match frozen {
            Frozen::Msm(value) => value.search_range_with_certificate(query, native_cutoff, limits),
            Frozen::Erp(value) => value.search_range_with_certificate(query, native_cutoff, limits),
            Frozen::Twed(value) => {
                value.search_range_with_certificate(query, native_cutoff, limits)
            }
            Frozen::Dtw(value) => value.search_range_with_certificate(query, native_cutoff, limits),
            Frozen::Frechet(value) => {
                value.search_range_with_certificate(query, native_cutoff, limits)
            }
        }
        .map_err(certificate_error)?;
        *output = Box::into_raw(Box::new(LlevTemporalRangeCertificate {
            frozen: frozen.clone(),
            results,
            certificate,
            limits,
            squared_dtw,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Read the fixed metadata for one complete certificate.
///
/// # Safety
/// Both pointers must be valid; `out_info` must be writable and disjoint.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_certificate_info(
    certificate: *const LlevTemporalRangeCertificate,
    out_info: *mut LlevTemporalCertificateInfo,
) -> LlevStatus {
    boundary(|| {
        let output = out_info.as_mut().ok_or((
            LlevStatus::NullPointer,
            "certificate info output is null".into(),
        ))?;
        *output = LlevTemporalCertificateInfo::default();
        let handle = certificate
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "certificate is null".into()))?;
        *output = info(handle);
        Ok(LlevStatus::Ok)
    })
}

fn page_count(start: usize, total: usize, capacity: usize) -> Result<usize, (LlevStatus, String)> {
    if start > total {
        return Err(invalid("certificate page start is beyond the sequence"));
    }
    Ok((total - start).min(capacity))
}

fn check_output_pointer<T>(
    pointer: *mut T,
    count: usize,
    name: &str,
) -> Result<(), (LlevStatus, String)> {
    if count == 0 {
        return Ok(());
    }
    if pointer.is_null() {
        return Err((LlevStatus::NullPointer, format!("{name} is null")));
    }
    if !(pointer as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(invalid(format!("{name} is not aligned")));
    }
    Ok(())
}

/// Copy a bounded page of exact query words.
///
/// # Safety
/// The handle, count output, and any nonempty destination must be valid and
/// disjoint. A zero-capacity page may use a null destination.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_certificate_query_bits(
    certificate: *const LlevTemporalRangeCertificate,
    start: usize,
    out_words: *mut u64,
    capacity: usize,
    out_written: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let written = out_written.as_mut().ok_or((
            LlevStatus::NullPointer,
            "query-bit count output is null".into(),
        ))?;
        *written = 0;
        let handle = certificate
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "certificate is null".into()))?;
        let count = page_count(start, handle.certificate.query_bits.len(), capacity)?;
        check_output_pointer(out_words, count, "query-bit output")?;
        if count > 0 {
            std::ptr::copy_nonoverlapping(
                handle.certificate.query_bits.as_ptr().add(start),
                out_words,
                count,
            );
        }
        *written = count;
        Ok(LlevStatus::Ok)
    })
}

/// Copy a bounded page of complete exact matches in public distance units.
///
/// # Safety
/// The handle, count output, and any nonempty destination must be valid and
/// disjoint. A zero-capacity page may use a null destination.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_certificate_matches(
    certificate: *const LlevTemporalRangeCertificate,
    start: usize,
    out_matches: *mut LlevTemporalIndexMatch,
    capacity: usize,
    out_written: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let written = out_written.as_mut().ok_or((
            LlevStatus::NullPointer,
            "certificate match count output is null".into(),
        ))?;
        *written = 0;
        let handle = certificate
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "certificate is null".into()))?;
        let count = page_count(start, handle.results.len(), capacity)?;
        check_output_pointer(out_matches, count, "certificate match output")?;
        for offset in 0..count {
            let (id, distance) = handle.results[start + offset];
            out_matches.add(offset).write(LlevTemporalIndexMatch {
                id,
                distance: if handle.squared_dtw {
                    distance.sqrt()
                } else {
                    distance
                },
            });
        }
        *written = count;
        Ok(LlevStatus::Ok)
    })
}

/// Read one ordered evidence record and optionally its quantized path.
///
/// A null path buffer with zero capacity reads only the header and required
/// path length. A short nonzero buffer fails without writing path bytes.
///
/// # Safety
/// All pointers must be valid for their declared lengths and disjoint from
/// each other. `out_header` and `out_path_written` are mandatory.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_certificate_evidence_at(
    certificate: *const LlevTemporalRangeCertificate,
    index: usize,
    out_header: *mut LlevTemporalCertificateEvidenceHeader,
    out_path: *mut u8,
    path_capacity: usize,
    out_path_written: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let header = out_header.as_mut().ok_or((
            LlevStatus::NullPointer,
            "certificate evidence header output is null".into(),
        ))?;
        *header = LlevTemporalCertificateEvidenceHeader::default();
        let written = out_path_written.as_mut().ok_or((
            LlevStatus::NullPointer,
            "certificate path count output is null".into(),
        ))?;
        *written = 0;
        let handle = certificate
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "certificate is null".into()))?;
        let evidence = handle
            .certificate
            .evidence
            .get(index)
            .ok_or_else(|| invalid("certificate evidence index is out of range"))?;
        let (value, path) = evidence_fields(evidence);
        *header = value;
        *written = path.len();
        if path_capacity == 0 && out_path.is_null() {
            return Ok(LlevStatus::Ok);
        }
        if path_capacity < path.len() {
            return Err((
                LlevStatus::LimitExceeded,
                "certificate path output is too short".into(),
            ));
        }
        check_output_pointer(out_path, path.len(), "certificate path output")?;
        if !path.is_empty() {
            std::ptr::copy_nonoverlapping(path.as_ptr(), out_path, path.len());
        }
        Ok(LlevStatus::Ok)
    })
}

fn same_info(left: LlevTemporalCertificateInfo, right: LlevTemporalCertificateInfo) -> bool {
    left.query_len == right.query_len
        && left.evidence_len == right.evidence_len
        && left.result_len == right.result_len
        && left.cutoff_native.to_bits() == right.cutoff_native.to_bits()
        && left.work_units == right.work_units
        && left.path_bytes == right.path_bytes
        && left.witness_bytes == right.witness_bytes
        && left.snapshot_present == right.snapshot_present
        && left.reserved == right.reserved
        && left.snapshot_identity == right.snapshot_identity
}

fn same_evidence(
    left: LlevTemporalCertificateEvidenceHeader,
    right: LlevTemporalCertificateEvidenceHeader,
) -> bool {
    left.kind == right.kind
        && left.reserved == right.reserved
        && left.path_len == right.path_len
        && left.stable_id == right.stable_id
        && left.lower_bound.to_bits() == right.lower_bound.to_bits()
        && left.exact.to_bits() == right.exact.to_bits()
        && left.has_exact == right.has_exact
        && left.survived == right.survived
        && left.reserved_tail == right.reserved_tail
}

/// Replay a caller-supplied complete certificate against the retained frozen
/// index. Every supplied byte and scalar decision must match the canonical
/// certificate before the native verifier recomputes the full K1--K4 walk.
/// A well-formed changed certificate returns `out_valid=0`.
///
/// # Safety
/// The handle and view must be live; each view pointer must address its
/// declared length and remain stable for this call. The writable validity
/// output must be disjoint from every input.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_certificate_verify(
    certificate: *const LlevTemporalRangeCertificate,
    raw_view: *const LlevTemporalCertificateView,
    out_valid: *mut u8,
) -> LlevStatus {
    boundary(|| {
        let valid = out_valid.as_mut().ok_or((
            LlevStatus::NullPointer,
            "certificate validity output is null".into(),
        ))?;
        *valid = 0;
        let handle = certificate
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "certificate is null".into()))?;
        let view = *raw_view.as_ref().ok_or((
            LlevStatus::NullPointer,
            "certificate replay view is null".into(),
        ))?;
        let expected = info(handle);
        if !same_info(view.info, expected) {
            return Ok(LlevStatus::Ok);
        }
        let query_bits = slice(
            view.query_bits,
            expected.query_len,
            "certificate query bits",
        )?;
        if query_bits != handle.certificate.query_bits {
            return Ok(LlevStatus::Ok);
        }
        let results = slice(view.results, expected.result_len, "certificate results")?;
        for (raw, &(id, distance)) in results.iter().zip(&handle.results) {
            let public_distance = if handle.squared_dtw {
                distance.sqrt()
            } else {
                distance
            };
            if raw.id != id || raw.distance.to_bits() != public_distance.to_bits() {
                return Ok(LlevStatus::Ok);
            }
        }
        let supplied = slice(view.evidence, expected.evidence_len, "certificate evidence")?;
        for (raw, original) in supplied.iter().zip(&handle.certificate.evidence) {
            let (header, path) = evidence_fields(original);
            if !same_evidence(raw.header, header) {
                return Ok(LlevStatus::Ok);
            }
            let supplied_path = slice(raw.path, path.len(), "certificate evidence path")?;
            if supplied_path != path {
                return Ok(LlevStatus::Ok);
            }
        }
        let query_bytes = expected
            .query_len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or((
                LlevStatus::LimitExceeded,
                "certificate query size overflow".into(),
            ))?;
        if query_bytes > handle.limits.resources.max_scratch_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "certificate replay query exceeds scratch limit".into(),
            ));
        }
        let mut query = Vec::new();
        query.try_reserve_exact(expected.query_len).map_err(|_| {
            (
                LlevStatus::LimitExceeded,
                "certificate replay query allocation failed".into(),
            )
        })?;
        query.extend(query_bits.iter().copied().map(f64::from_bits));
        let replay = match &handle.frozen {
            Frozen::Msm(value) => value.verify_range_certificate(
                &query,
                handle.certificate.cutoff,
                &handle.certificate,
                handle.limits,
            ),
            Frozen::Erp(value) => value.verify_range_certificate(
                &query,
                handle.certificate.cutoff,
                &handle.certificate,
                handle.limits,
            ),
            Frozen::Twed(value) => value.verify_range_certificate(
                &query,
                handle.certificate.cutoff,
                &handle.certificate,
                handle.limits,
            ),
            Frozen::Dtw(value) => value.verify_range_certificate(
                &query,
                handle.certificate.cutoff,
                &handle.certificate,
                handle.limits,
            ),
            Frozen::Frechet(value) => value.verify_range_certificate(
                &query,
                handle.certificate.cutoff,
                &handle.certificate,
                handle.limits,
            ),
        }
        .map_err(certificate_error)?;
        *valid = u8::from(replay);
        Ok(LlevStatus::Ok)
    })
}

/// Release a complete certificate and its retained frozen snapshot once.
///
/// # Safety
/// The pointer must come from `llev_temporal_index_query_certified` and must
/// be freed exactly once after concurrent readers have stopped.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_certificate_free(
    certificate: *mut LlevTemporalRangeCertificate,
) {
    if !certificate.is_null() {
        drop(Box::from_raw(certificate));
    }
}
