//! Context-dependent edit costs for the opt-in contextual query surface.

use super::OperationCostsF64;
use libdictenstein::CharUnit;

/// Context available while evaluating one trie edge against one query unit.
///
/// Query left and right context are both available because the query is fixed.
/// Dictionary left context is the already-descended trie prefix. Dictionary
/// right context is unavailable: the traversal has not selected a child edge
/// beyond `dictionary_unit`.
#[derive(Clone, Copy, Debug)]
pub struct EditContext<'a, U: CharUnit> {
    query: &'a [U],
    query_index: usize,
    dictionary_prefix: &'a [U],
    dictionary_unit: Option<U>,
}

impl<'a, U: CharUnit> EditContext<'a, U> {
    /// Construct context for a DP cell.
    pub fn new(
        query: &'a [U],
        query_index: usize,
        dictionary_prefix: &'a [U],
        dictionary_unit: Option<U>,
    ) -> Self {
        Self {
            query,
            query_index,
            dictionary_prefix,
            dictionary_unit,
        }
    }

    /// Complete query.
    pub fn query(&self) -> &'a [U] {
        self.query
    }

    /// Zero-based query position associated with this edit.
    pub fn query_index(&self) -> usize {
        self.query_index
    }

    /// Query unit immediately to the left, if any.
    pub fn query_left(&self) -> Option<U> {
        self.query_index
            .checked_sub(1)
            .and_then(|index| self.query.get(index).copied())
    }

    /// Query unit immediately to the right, if any.
    pub fn query_right(&self) -> Option<U> {
        self.query.get(self.query_index.saturating_add(1)).copied()
    }

    /// Already-consumed dictionary prefix, excluding `dictionary_unit`.
    pub fn dictionary_prefix(&self) -> &'a [U] {
        self.dictionary_prefix
    }

    /// Dictionary unit on the edge currently being evaluated.
    pub fn dictionary_unit(&self) -> Option<U> {
        self.dictionary_unit
    }

    /// Dictionary unit immediately to the left, if any.
    pub fn dictionary_left(&self) -> Option<U> {
        self.dictionary_prefix.last().copied()
    }

    /// Dictionary right context is not known during trie descent.
    pub fn dictionary_right(&self) -> Option<U> {
        None
    }
}

/// Context-dependent, non-negative edit costs.
///
/// `None` forbids an operation. The strictly positive finite lower bound is
/// mandatory because offset-based subsumption can only realign positions when
/// `abs(i-j) * min_nonzero_cost() <= cost_slack`.
pub trait ContextualCost<U: CharUnit> {
    /// Match or substitute one query unit with the current dictionary unit.
    fn substitution_cost(
        &self,
        context: &EditContext<'_, U>,
        query: U,
        dictionary: U,
    ) -> Option<f64>;

    /// Insert the current dictionary unit relative to the query.
    fn insertion_cost(&self, context: &EditContext<'_, U>, dictionary: U) -> Option<f64>;

    /// Delete one query unit.
    fn deletion_cost(&self, context: &EditContext<'_, U>, query: U) -> Option<f64>;

    /// Strictly positive finite lower bound for every non-zero allowed edit.
    fn min_nonzero_cost(&self) -> f64;
}

impl<U> ContextualCost<U> for OperationCostsF64
where
    U: CharUnit + Eq,
{
    #[inline]
    fn substitution_cost(
        &self,
        _context: &EditContext<'_, U>,
        query: U,
        dictionary: U,
    ) -> Option<f64> {
        Some(if query == dictionary {
            self.match_cost
        } else {
            self.substitution
        })
    }

    #[inline]
    fn insertion_cost(&self, _context: &EditContext<'_, U>, _dictionary: U) -> Option<f64> {
        Some(self.insertion)
    }

    #[inline]
    fn deletion_cost(&self, _context: &EditContext<'_, U>, _query: U) -> Option<f64> {
        Some(self.deletion)
    }

    #[inline]
    fn min_nonzero_cost(&self) -> f64 {
        OperationCostsF64::min_nonzero_cost(self)
    }
}

/// English soft-c reference surface.
///
/// Substituting query `c` with dictionary `s` is discounted when the *query's*
/// next scalar is `e`, `i`, or `y`. The trigger intentionally uses query right
/// context; arbitrary dictionary right-context rules are not implementable
/// during trie descent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnglishSoftC {
    soft_cost: f64,
    ordinary_cost: f64,
}

impl Default for EnglishSoftC {
    fn default() -> Self {
        Self {
            soft_cost: 0.25,
            ordinary_cost: 1.0,
        }
    }
}

impl ContextualCost<char> for EnglishSoftC {
    fn substitution_cost(
        &self,
        context: &EditContext<'_, char>,
        query: char,
        dictionary: char,
    ) -> Option<f64> {
        if query == dictionary {
            return Some(0.0);
        }
        let soft_query_c = matches!(query, 'c' | 'C')
            && matches!(dictionary, 's' | 'S')
            && context
                .query_right()
                .is_some_and(|right| matches!(right, 'e' | 'E' | 'i' | 'I' | 'y' | 'Y'));
        Some(if soft_query_c {
            self.soft_cost
        } else {
            self.ordinary_cost
        })
    }

    fn insertion_cost(&self, _context: &EditContext<'_, char>, _dictionary: char) -> Option<f64> {
        Some(self.ordinary_cost)
    }

    fn deletion_cost(&self, _context: &EditContext<'_, char>, _query: char) -> Option<f64> {
        Some(self.ordinary_cost)
    }

    fn min_nonzero_cost(&self) -> f64 {
        self.soft_cost
    }
}

/// Reference surface discounting deletion of a query-final silent `e`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionalSilentE {
    silent_e_cost: f64,
    ordinary_cost: f64,
}

impl Default for PositionalSilentE {
    fn default() -> Self {
        Self {
            silent_e_cost: 0.25,
            ordinary_cost: 1.0,
        }
    }
}

impl ContextualCost<char> for PositionalSilentE {
    fn substitution_cost(
        &self,
        _context: &EditContext<'_, char>,
        query: char,
        dictionary: char,
    ) -> Option<f64> {
        Some(if query == dictionary {
            0.0
        } else {
            self.ordinary_cost
        })
    }

    fn insertion_cost(&self, _context: &EditContext<'_, char>, _dictionary: char) -> Option<f64> {
        Some(self.ordinary_cost)
    }

    fn deletion_cost(&self, context: &EditContext<'_, char>, query: char) -> Option<f64> {
        Some(
            if matches!(query, 'e' | 'E') && context.query_right().is_none() {
                self.silent_e_cost
            } else {
                self.ordinary_cost
            },
        )
    }

    fn min_nonzero_cost(&self) -> f64 {
        self.silent_e_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_availability_is_explicit() {
        let query = ['c', 'i'];
        let prefix = ['x'];
        let context = EditContext::new(&query, 0, &prefix, Some('s'));
        assert_eq!(context.query_right(), Some('i'));
        assert_eq!(context.dictionary_left(), Some('x'));
        assert_eq!(context.dictionary_right(), None);
        assert_eq!(
            EnglishSoftC::default().substitution_cost(&context, 'c', 's'),
            Some(0.25)
        );
    }
}
