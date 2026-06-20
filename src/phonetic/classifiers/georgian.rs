//! Georgian script vowel classifier.
//!
//! Supports Georgian (ქართული) language.
//! Georgian uses the Mkhedruli script, a unique alphabet with:
//! - No uppercase/lowercase distinction in modern usage
//! - 33 letters (originally 38)
//! - 5 vowels and 28 consonants
//! - Ejective consonants (unique feature)
//! - Nearly phonemic orthography

use super::VowelClassifier;

/// Vowel classifier for Georgian Mkhedruli script.
///
/// Georgian has 5 vowels:
/// - **ა (ani)**: /a/
/// - **ე (eni)**: /e/
/// - **ი (ini)**: /i/
/// - **ო (oni)**: /o/
/// - **უ (uni)**: /u/
///
/// Georgian is notable for:
/// - **Ejective consonants**: ყ, წ, ჭ, პ, ტ, კ (voiceless with glottal closure)
/// - **Harmonic clusters**: Consonant clusters are common and predictable
/// - **No grammatical gender**: Unlike most Indo-European languages
#[derive(Debug, Clone, Copy, Default)]
pub struct GeorgianClassifier;

impl GeorgianClassifier {
    /// Create a new Georgian classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Georgian vowels (Mkhedruli script).
static GEORGIAN_VOWELS: &[char] = &[
    'ა', // ani - a
    'ე', // eni - e
    'ი', // ini - i
    'ო', // oni - o
    'უ', // uni - u
];

impl VowelClassifier for GeorgianClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Mkhedruli vowels
            0x10D0 => true, // ა (ani) - a
            0x10D4 => true, // ე (eni) - e
            0x10D8 => true, // ი (ini) - i
            0x10DD => true, // ო (oni) - o
            0x10E3 => true, // უ (uni) - u

            // Asomtavruli (archaic uppercase) vowels - rarely used
            0x10A0 => true, // Ⴀ (ani) - a
            0x10A4 => true, // Ⴄ (eni) - e
            0x10A8 => true, // Ⴈ (ini) - i
            0x10AD => true, // Ⴍ (oni) - o
            0x10B3 => true, // Ⴓ (uni) - u

            // Nuskhuri (ecclesiastical) vowels - rarely used
            0x2D00 => true, // ⴀ (ani) - a
            0x2D04 => true, // ⴄ (eni) - e
            0x2D08 => true, // ⴈ (ini) - i
            0x2D0D => true, // ⴍ (oni) - o
            0x2D13 => true, // ⴓ (uni) - u

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Georgian"
    }

    fn vowels(&self) -> &[char] {
        GEORGIAN_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Mkhedruli consonants (28 consonants)
            // Georgian consonants are at non-contiguous code points.

            // Letters before ა (ani)
            // (none - ა is first)

            // Letters between ა and ე
            0x10D1 => true, // ბ (bani) - b
            0x10D2 => true, // გ (gani) - g
            0x10D3 => true, // დ (doni) - d

            // Letters between ე and ი
            0x10D5 => true, // ვ (vini) - v
            0x10D6 => true, // ზ (zeni) - z
            0x10D7 => true, // თ (tani) - t (aspirated)

            // Letters between ი and ო
            0x10D9 => true, // კ (kani) - k (ejective)
            0x10DA => true, // ლ (lasi) - l
            0x10DB => true, // მ (mani) - m
            0x10DC => true, // ნ (nari) - n

            // Letters between ო and უ
            0x10DE => true, // პ (pari) - p (ejective)
            0x10DF => true, // ჟ (zhani) - zh
            0x10E0 => true, // რ (rae) - r
            0x10E1 => true, // ს (sani) - s
            0x10E2 => true, // ტ (tari) - t (ejective)

            // Letters after უ
            0x10E4 => true, // ფ (pari) - p (aspirated)
            0x10E5 => true, // ქ (kani) - k (aspirated)
            0x10E6 => true, // ღ (ghani) - gh (voiced velar fricative)
            0x10E7 => true, // ყ (qari) - q (ejective uvular)
            0x10E8 => true, // შ (shini) - sh
            0x10E9 => true, // ჩ (chini) - ch (aspirated)
            0x10EA => true, // ც (tsani) - ts (aspirated)
            0x10EB => true, // ძ (dzili) - dz
            0x10EC => true, // წ (tsili) - ts (ejective)
            0x10ED => true, // ჭ (chari) - ch (ejective)
            0x10EE => true, // ხ (khani) - kh (voiceless velar fricative)
            0x10EF => true, // ჯ (jani) - j
            0x10F0 => true, // ჰ (hae) - h

            // Archaic/obsolete letters (rarely used)
            0x10F1 => true, // ჱ (he) - obsolete
            0x10F2 => true, // ჲ (hie) - obsolete
            0x10F3 => true, // ჳ (we) - obsolete
            0x10F4 => true, // ჴ (har) - obsolete
            0x10F5 => true, // ჵ (hoe) - obsolete
            0x10F6 => true, // ჶ (fi) - obsolete

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vowels() {
        let c = GeorgianClassifier::new();
        assert!(c.is_vowel('ა')); // ani - a
        assert!(c.is_vowel('ე')); // eni - e
        assert!(c.is_vowel('ი')); // ini - i
        assert!(c.is_vowel('ო')); // oni - o
        assert!(c.is_vowel('უ')); // uni - u
    }

    #[test]
    fn test_consonants_not_vowels() {
        let c = GeorgianClassifier::new();
        assert!(!c.is_vowel('ბ')); // bani - b
        assert!(!c.is_vowel('გ')); // gani - g
        assert!(!c.is_vowel('დ')); // doni - d
        assert!(!c.is_vowel('კ')); // kani - k (ejective)
        assert!(!c.is_vowel('პ')); // pari - p (ejective)
        assert!(!c.is_vowel('ტ')); // tari - t (ejective)
    }

    #[test]
    fn test_is_consonant() {
        let c = GeorgianClassifier::new();
        // Regular consonants
        assert!(c.is_consonant('ბ')); // bani - b
        assert!(c.is_consonant('გ')); // gani - g
        assert!(c.is_consonant('დ')); // doni - d
        assert!(c.is_consonant('ვ')); // vini - v
        assert!(c.is_consonant('ზ')); // zeni - z
        assert!(c.is_consonant('ლ')); // lasi - l
        assert!(c.is_consonant('მ')); // mani - m
        assert!(c.is_consonant('ნ')); // nari - n
        assert!(c.is_consonant('რ')); // rae - r
        assert!(c.is_consonant('ს')); // sani - s
        assert!(c.is_consonant('ჰ')); // hae - h

        // Ejective consonants
        assert!(c.is_consonant('კ')); // kani - k (ejective)
        assert!(c.is_consonant('პ')); // pari - p (ejective)
        assert!(c.is_consonant('ტ')); // tari - t (ejective)
        assert!(c.is_consonant('ყ')); // qari - q (ejective uvular)
        assert!(c.is_consonant('წ')); // tsili - ts (ejective)
        assert!(c.is_consonant('ჭ')); // chari - ch (ejective)

        // Aspirated consonants
        assert!(c.is_consonant('თ')); // tani - t (aspirated)
        assert!(c.is_consonant('ფ')); // pari - p (aspirated)
        assert!(c.is_consonant('ქ')); // kani - k (aspirated)
        assert!(c.is_consonant('ც')); // tsani - ts (aspirated)
        assert!(c.is_consonant('ჩ')); // chini - ch (aspirated)
    }

    #[test]
    fn test_sibilants_and_fricatives() {
        let c = GeorgianClassifier::new();
        assert!(c.is_consonant('ს')); // sani - s
        assert!(c.is_consonant('შ')); // shini - sh
        assert!(c.is_consonant('ზ')); // zeni - z
        assert!(c.is_consonant('ჟ')); // zhani - zh
        assert!(c.is_consonant('ხ')); // khani - kh
        assert!(c.is_consonant('ღ')); // ghani - gh
    }

    #[test]
    fn test_affricates() {
        let c = GeorgianClassifier::new();
        assert!(c.is_consonant('ც')); // tsani - ts (aspirated)
        assert!(c.is_consonant('წ')); // tsili - ts (ejective)
        assert!(c.is_consonant('ძ')); // dzili - dz
        assert!(c.is_consonant('ჩ')); // chini - ch (aspirated)
        assert!(c.is_consonant('ჭ')); // chari - ch (ejective)
        assert!(c.is_consonant('ჯ')); // jani - j
    }

    #[test]
    fn test_vowels_not_consonants() {
        let c = GeorgianClassifier::new();
        assert!(!c.is_consonant('ა')); // ani - a
        assert!(!c.is_consonant('ე')); // eni - e
        assert!(!c.is_consonant('ი')); // ini - i
        assert!(!c.is_consonant('ო')); // oni - o
        assert!(!c.is_consonant('უ')); // uni - u
    }

    #[test]
    fn test_script_name() {
        let c = GeorgianClassifier::new();
        assert_eq!(c.script_name(), "Georgian");
    }
}
