//! Per-language rule aggregators for Romance languages.
//!
//! Covers: Castilian Spanish, Latin American Spanish, Standard French,
//! Canadian French, European Portuguese, Brazilian Portuguese, Italian,
//! Catalan, Romanian.

use crate::phonetic::types::RewriteRuleChar;

/// Combine Castilian Spanish rule sets into a single vector.
///
/// This includes:
/// - Base Spanish orthographic rules
/// - Castilian dialect rules (distinción: z, c(e,i) → θ)
pub(crate) fn castilian_spanish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::spanish;
    spanish::combined_castilian().rules.clone()
}

/// Combine Latin American Spanish rule sets into a single vector.
///
/// This includes:
/// - Base Spanish orthographic rules
/// - Latin American dialect rules (seseo: z, c(e,i) → s)
pub(crate) fn latin_american_spanish_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::spanish;
    spanish::combined_latin_american().rules.clone()
}

/// Combine Standard French rule sets into a single vector.
///
/// This includes:
/// - Base French orthographic rules (silent letters, nasal vowels, digraphs)
/// - Standard French dialect rules
pub(crate) fn standard_french_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::french;
    french::combined_standard().rules.clone()
}

/// Combine Canadian French rule sets into a single vector.
///
/// This includes:
/// - Base French orthographic rules
/// - Canadian dialect rules (affrication: ti → tsi, tu → tsu, di → dzi)
pub(crate) fn canadian_french_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::french;
    french::combined_canadian().rules.clone()
}

/// Combine European Portuguese rule sets into a single vector.
///
/// This includes:
/// - Base Portuguese orthographic rules
/// - European dialect rules (vowel reduction, uvular R, s → sh patterns)
pub(crate) fn european_portuguese_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::portuguese;
    portuguese::combined_european().rules.clone()
}

/// Combine Brazilian Portuguese rule sets into a single vector.
///
/// This includes:
/// - Base Portuguese orthographic rules
/// - Brazilian dialect rules (t/d palatalization, l vocalization, h-like R)
pub(crate) fn brazilian_portuguese_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::portuguese;
    portuguese::combined_brazilian().rules.clone()
}

/// Get Standard Italian rule set.
///
/// Italian has relatively uniform pronunciation across regions,
/// so only a single comprehensive rule set is provided.
pub(crate) fn italian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::italian;
    italian::base().rules.clone()
}

/// Get Catalan rule set.
///
/// Returns the complete phonetic normalization rules for Catalan (Català):
/// - L·L (ela geminada) → LL (unique to Catalan)
/// - Digraphs: ny→NY, tx→CH, ll→Y, ss→S, tg/tj→J
/// - Ç → S (c cedilla)
/// - X → SH (like English "sh")
/// - H → (silent)
/// - V → B (b/v merger in most dialects)
/// - Accented vowels normalized
pub(crate) fn catalan_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::catalan;
    catalan::base().rules.clone()
}

/// Get Romanian rule set.
///
/// Returns the complete phonetic normalization rules for Romanian (Română):
/// - Special vowels: ă→a (schwa), â/î→i (close central unrounded)
/// - Special consonants: ș→SH, ț→TS
/// - Digraphs: ch→K (before e, i), gh→G (before e, i)
/// - J → ZH (like French "j")
/// - CE, CI → CHE, CHI (soft c)
pub(crate) fn romanian_rules() -> Vec<RewriteRuleChar> {
    use crate::phonetic::rules::romanian;
    romanian::base().rules.clone()
}
