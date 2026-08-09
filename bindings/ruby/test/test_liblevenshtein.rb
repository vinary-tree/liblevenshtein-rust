require "minitest/autorun"
require "vinary_tree/liblevenshtein"

class LiblevenshteinTest < Minitest::Test
  def test_distances
    assert_equal 3, VinaryTree::Liblevenshtein.distance("kitten", "sitting")
    assert_equal 1, VinaryTree::Liblevenshtein.damerau_distance("ab", "ba")
    assert_equal 2, VinaryTree::Liblevenshtein.true_damerau_distance("ca", "abc")
  end

  def test_phonetic_pattern
    pattern = VinaryTree::Liblevenshtein::PhoneticPattern.compile_regex("cat")
    assert pattern.matches?("cat")
    refute pattern.matches?("cot")
  ensure
    pattern&.close
  end

  # C6: the ABI counts Unicode scalar values, so the facade passes the UTF-8 byte
  # length while distances stay char-based (guards against UTF-16/byte length).
  def test_unicode_distances
    assert_equal 1, VinaryTree::Liblevenshtein.distance("café", "cafe")
    assert_equal 1, VinaryTree::Liblevenshtein.distance("🦀", "x")
    assert_equal 1, VinaryTree::Liblevenshtein.distance("é", "e")
  end

  # C6: threshold functions pass the native sentinel (usize::MAX - 1) through when
  # the exact distance exceeds the bound.
  def test_threshold_sentinel
    exceeded = (1 << (Fiddle::SIZEOF_SIZE_T * 8)) - 2
    assert_equal 3, VinaryTree::Liblevenshtein.distance_threshold("kitten", "sitting", 3)
    assert_equal exceeded, VinaryTree::Liblevenshtein.distance_threshold("kitten", "sitting", 2)
    # "ca" -> "abc": OSA is 3 but unrestricted Damerau-Levenshtein is 2.
    assert_equal exceeded, VinaryTree::Liblevenshtein.damerau_distance_threshold("ca", "abc", 2)
    assert_equal 2, VinaryTree::Liblevenshtein.true_damerau_distance_threshold("ca", "abc", 2)
  end

  # C2/C3: size accessor, idempotent close, and closed-handle error (no crash).
  def test_phonetic_pattern_size_and_idempotent_close
    pattern = VinaryTree::Liblevenshtein::PhoneticPattern.compile_regex("c[ao]t")
    states, transitions = pattern.size
    assert_operator states, :>, 0
    assert_operator transitions, :>, 0
    pattern.close
    pattern.close # idempotent
    assert_raises(IOError) { pattern.matches?("cat") }
  end

  # C6/C2/C3: rule-set parse/length/apply, idempotent close, closed-handle error.
  def test_phonetic_rule_set
    rules = VinaryTree::Liblevenshtein::PhoneticRuleSet.parse("ph -> f\ngh ->\n")
    assert_equal 2, rules.length
    assert_equal "f", rules.apply("phgh")
    rules.close
    rules.close # idempotent
    assert_raises(IOError) { rules.length }
  end

  def test_builtin_rule_set
    rules = VinaryTree::Liblevenshtein::PhoneticRuleSet.builtin(
      VinaryTree::Liblevenshtein::PhoneticRuleSet::ENGLISH_ORTHOGRAPHY
    )
    assert_operator rules.length, :>, 0
    refute_empty rules.apply("phone")
  ensure
    rules&.close
  end
end
