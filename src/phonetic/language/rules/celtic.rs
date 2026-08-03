//! Per-language rule aggregators for Celtic languages.
//!
//! Covers: Welsh, Irish.

use crate::phonetic::types::RewriteRuleChar;

/// Get Welsh rule set.
///
/// Returns the complete phonetic normalization rules for Welsh (Cymraeg):
/// - 8 digraphs as letters: ch→CH, dd→DH, ff→F, ng→NG, ll→LL, ph→F, rh→RH, th→TH
/// - Unique LL: voiceless lateral fricative /ɬ/
/// - F = V sound (ff = F sound)
/// - W and Y as vowels
/// - Circumflex for long vowels: â, ê, î, ô, û, ŵ, ŷ
pub(crate) fn welsh_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::welsh;
    welsh::base().rules.clone()
}

/// Get Irish rule set.
///
/// Returns the complete phonetic normalization rules for Irish (Gaeilge):
/// - Séimhiú (lenition): bh→v, ch→CH, dh→GH, fh→(silent!), gh→GH, mh→v, ph→f, sh→h, th→h
/// - FH is completely silent
/// - Fadas (acute accent): á, é, í, ó, ú for long vowels
pub(crate) fn irish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::irish;
    irish::base().rules.clone()
}
