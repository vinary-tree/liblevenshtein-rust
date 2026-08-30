use super::cost::CanonicalCost;

/// One weighted control position in a canonical temporal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TemporalPosition {
    pub(crate) query_index: u32,
    pub(crate) cost: CanonicalCost,
}

impl TemporalPosition {
    #[inline]
    pub(crate) fn new(query_index: usize, cost: f64) -> Option<Self> {
        Some(Self {
            query_index: u32::try_from(query_index).ok()?,
            cost: CanonicalCost::new(cost)?,
        })
    }

    #[inline]
    pub(crate) fn query_index(self) -> usize {
        usize::try_from(self.query_index).expect("u32 query index fits usize")
    }
}

/// Canonical state retained by a query-local arena.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CanonicalTemporalState<C> {
    pub(crate) context: C,
    pub(crate) positions: Vec<TemporalPosition>,
}
