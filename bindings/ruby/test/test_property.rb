# C8 native property-based tests for the Ruby facade: a seeded pseudo-PBT loop
# (Minitest + a fixed-seed Random) checked against an in-language Levenshtein
# oracle.
#
#   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, and threshold
#       consistency dThreshold(a,b,k) == (d<=k ? d : sentinel) across the
#       Levenshtein/Damerau/true-Damerau variants, plus standard oracle
#       agreement;
#   (b) query result set == {t in dict : lev(query,t) <= k} over a random
#       libdictenstein DynamicDawg dictionary, with exact distances and value
#       round-trips;
#   (c) u64 value round-trips, with 0 and 2**64-1 pinned.
#
# The PRNG is seeded from a constant, so a failing run reproduces. Ruby strings
# expose a non-null buffer even when empty, so empty operands are exercised.

require "fiddle"
require "minitest/autorun"
require "vinary_tree/liblevenshtein"
require "vinary_tree/libdictenstein"

class PropertyTest < Minitest::Test
  M = VinaryTree::Liblevenshtein
  D = VinaryTree::Libdictenstein
  ALPHABET = %w[a b c é].freeze
  SEED = 20_260_809
  U64_MAX = (1 << 64) - 1
  SENTINEL = (1 << (Fiddle::SIZEOF_SIZE_T * 8)) - 2 # threshold over-bound

  def setup
    @rng = Random.new(SEED)
  end

  def gen_string(min, max)
    Array.new(@rng.rand(min..max)) { ALPHABET.sample(random: @rng) }.join
  end

  def levenshtein(a, b)
    return 0 if a == b
    left = a.chars
    right = b.chars
    return right.length if left.empty?
    return left.length if right.empty?
    previous = (0..right.length).to_a
    left.each_with_index do |ca, i|
      current = [i + 1]
      right.each_with_index do |cb, j|
        cost = ca == cb ? 0 : 1
        current << [previous[j + 1] + 1, current[j] + 1, previous[j] + cost].min
      end
      previous = current
    end
    previous[right.length]
  end

  VARIANTS = [
    %i[distance distance_threshold],
    %i[damerau_distance damerau_distance_threshold],
    %i[true_damerau_distance true_damerau_distance_threshold]
  ].freeze

  def test_distance_symmetry_identity_threshold
    300.times do
      a = gen_string(0, 6)
      b = gen_string(0, 6)
      k = @rng.rand(0..3)
      assert_equal levenshtein(a, b), M.distance(a, b), "standard oracle agreement"
      VARIANTS.each do |distance, thresholded|
        full = M.public_send(distance, a, b)
        assert_equal full, M.public_send(distance, b, a), "#{distance} symmetry"
        assert_equal 0, M.public_send(distance, a, a), "#{distance} identity"
        bounded = M.public_send(thresholded, a, b, k)
        assert_equal(full <= k ? full : SENTINEL, bounded, "#{thresholded} consistency")
      end
    end
  end

  def run_query(entries, query, k)
    dictionary = D::DynamicDawg.new
    entries.each { |term, value| dictionary.put(term, value) }
    transducer = M::Transducer.new(dictionary)
    result = {}
    transducer.query(query, k).each { |match| result[match.term] = [match.distance, match.id] }
    result
  ensure
    transducer&.close
    dictionary&.close
  end

  def test_query_matches_oracle
    150.times do
      entries = {}
      @rng.rand(0..8).times do
        entries[gen_string(0, 6)] = @rng.rand(4).zero? ? nil : @rng.rand(0..U64_MAX)
      end
      query = gen_string(0, 6)
      k = @rng.rand(0..3)
      got = run_query(entries, query, k)
      expected = entries.select { |term, _| levenshtein(query, term) <= k }
      assert_equal expected.keys.sort, got.keys.sort, "result set equals oracle"
      got.each do |term, (distance, id)|
        assert_equal levenshtein(query, term), distance, "exact distance for #{term.inspect}"
        if expected[term].nil?
          assert_nil id, "value round-trip for #{term.inspect}"
        else
          assert_equal expected[term], id, "value round-trip for #{term.inspect}"
        end
      end
    end
  end

  def test_u64_value_round_trip
    [0, 1, 1 << 63, U64_MAX].each do |value|
      assert_equal({ "term" => [0, value] }, run_query({ "term" => value }, "term", 0))
    end
  end
end
