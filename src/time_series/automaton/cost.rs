use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Finite binary64 value with a unique signed-zero representation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CanonicalFinite(f64);

impl CanonicalFinite {
    #[inline]
    pub(crate) fn new(value: f64) -> Option<Self> {
        value
            .is_finite()
            .then_some(Self(if value == 0.0 { 0.0 } else { value }))
    }

    #[inline]
    pub(crate) fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for CanonicalFinite {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for CanonicalFinite {}

impl PartialOrd for CanonicalFinite {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CanonicalFinite {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Hash for CanonicalFinite {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Finite, non-negative path cost used in canonical state identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CanonicalCost(CanonicalFinite);

impl CanonicalCost {
    #[inline]
    pub(crate) fn new(value: f64) -> Option<Self> {
        (value >= 0.0)
            .then(|| CanonicalFinite::new(value))
            .flatten()
            .map(Self)
    }

    #[inline]
    pub(crate) fn get(self) -> f64 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_zero_is_one_exact_key() {
        assert_eq!(CanonicalFinite::new(0.0), CanonicalFinite::new(-0.0));
        assert_eq!(CanonicalCost::new(0.0), CanonicalCost::new(-0.0));
    }

    #[test]
    fn invalid_costs_are_not_constructible() {
        assert!(CanonicalCost::new(-1.0).is_none());
        assert!(CanonicalCost::new(f64::NAN).is_none());
        assert!(CanonicalCost::new(f64::INFINITY).is_none());
    }
}
