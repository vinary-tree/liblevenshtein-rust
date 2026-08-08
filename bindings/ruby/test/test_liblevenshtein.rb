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
end
