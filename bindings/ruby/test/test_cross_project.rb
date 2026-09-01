# frozen_string_literal: true

require "minitest/autorun"
require "set"
require "vinary_tree/libdictenstein"
require "vinary_tree/liblevenshtein"

class LiblevenshteinCrossProjectTest < Minitest::Test
  LD = VinaryTree::Libdictenstein
  LL = VinaryTree::Liblevenshtein

  def collect(query)
    query.map { |match| [match.term, match.distance, match.id, match.domain] }
  end

  def text_dictionary(entries)
    dictionary = LD::DynamicDawg.new
    dictionary.put_all(entries)
    dictionary
  end

  def test_distances_statuses_and_generated_selections
    assert_equal 3, LL.distance("kitten", "sitting")
    assert_equal 3, LL.distance_threshold("kitten", "sitting", 3)
    assert_equal 1, LL.damerau_distance("ab", "ba")
    assert_equal 1, LL.damerau_distance_threshold("ab", "ba", 1)
    assert_equal 2, LL.true_damerau_distance("ca", "abc")
    assert_equal 2, LL.true_damerau_distance_threshold("ca", "abc", 2)

    assert_equal [0, 1, 2, 3], [
      LL::Algorithm::STANDARD,
      LL::Algorithm::TRANSPOSITION,
      LL::Algorithm::MERGE_AND_SPLIT,
      LL::Algorithm::DAMERAU_LEVENSHTEIN
    ]
    assert_equal [0, 1], [LL::QueryOrder::TRAVERSAL, LL::QueryOrder::DISTANCE_THEN_TERM]
    assert_equal [0, 1], [
      LL::PhoneticRuleSetKind::ENGLISH_ORTHOGRAPHY,
      LL::PhoneticRuleSetKind::ENGLISH_PHONETIC
    ]
    assert_equal((0..12).to_a, [
      LL::Status::OK, LL::Status::END_OF_STREAM, LL::Status::INVALID_ARGUMENT,
      LL::Status::INVALID_UTF8, LL::Status::NULL_POINTER, LL::Status::PANIC,
      LL::Status::UNSUPPORTED, LL::Status::IO_ERROR, LL::Status::CLOSED,
      LL::Status::LIMIT_EXCEEDED, LL::Status::PROVIDER_ERROR,
      LL::Status::BATCH_IN_USE, LL::Status::DOMAIN_MISMATCH
    ])

    error = assert_raises(LL::Error) { LL::PhoneticPattern.compile_regex("(") }
    assert_equal LL::Status::INVALID_ARGUMENT, error.status
    refute_empty error.message
    assert_equal 0xffff_ffff, LL::Error.new(0xffff_ffff).status
  end

  def test_every_algorithm_and_distance_then_term_order
    dictionary = text_dictionary([
      ["cat", nil], ["cot", nil], ["cut", nil], ["scat", nil],
      ["ab", nil], ["ba", nil], ["c", nil], ["abc", nil],
      ["bat", nil], ["cats", nil]
    ])

    standard = LL::Transducer.new(dictionary, algorithm: LL::Transducer::STANDARD)
    refute_includes collect(standard.query("ba", 1)).map(&:first), "ab"

    transposition = LL::Transducer.new(dictionary, algorithm: LL::Transducer::TRANSPOSITION)
    assert_includes collect(transposition.query("ba", 1)).map { |term, distance,| [term, distance] }, ["ab", 1]

    merge_and_split = LL::Transducer.new(dictionary, algorithm: LL::Transducer::MERGE_AND_SPLIT)
    assert_includes collect(merge_and_split.query("ab", 1)).map { |term, distance,| [term, distance] }, ["c", 1]

    damerau = LL::Transducer.new(dictionary, algorithm: LL::Transducer::DAMERAU_LEVENSHTEIN)
    assert_includes collect(damerau.query("ca", 2)).map { |term, distance,| [term, distance] }, ["abc", 2]

    ordered = collect(standard.query("cat", 1, order: LL::QueryOrder::DISTANCE_THEN_TERM))
    assert_equal ["cat", 0], ordered.first.take(2)
    assert_equal ordered.map { |term, distance,| [distance, term] }.sort,
                 ordered.map { |term, distance,| [distance, term] }
  ensure
    [standard, transposition, merge_and_split, damerau].compact.each(&:close)
    dictionary&.close
  end

  def test_query_start_snapshot_survives_mutation_and_owner_close
    dictionary = text_dictionary([["cat", 1], ["cot", 2], ["cut", 3], ["scat", nil]])
    transducer = LL::Transducer.new(dictionary)
    query = transducer.query("cat", 2)

    dictionary.remove("cot")
    dictionary.put("cut", 30)
    dictionary.put("cit", 5)
    dictionary.close
    transducer.close

    assert_equal Set["cat", "cot", "cut", "scat"], query.map(&:term).to_set
    assert_raises(IOError) { query.each.to_a }
  ensure
    query&.close
    transducer&.close
    dictionary&.close
  end

  def test_raw_byte_and_u64_queries_preserve_payloads_and_values
    byte_dictionary = LD::DynamicDawg.new(domain: LD::BYTE)
    byte_dictionary.put("\xff\x00\x7f".b, 9)
    byte_transducer = LL::Transducer.new(byte_dictionary)
    ordered_bytes_error = assert_raises(LL::Error) do
      byte_transducer.query_bytes(
        "\xff\x00\x7e".b,
        1,
        order: LL::QueryOrder::DISTANCE_THEN_TERM
      )
    end
    assert_equal LL::Status::UNSUPPORTED, ordered_bytes_error.status
    byte_matches = collect(byte_transducer.query_bytes("\xff\x00\x7e".b, 1))
    assert_equal [["\xff\x00\x7f".b, 1, 9, LL::BYTE_DOMAIN]], byte_matches

    token_dictionary = LD::DynamicDawg.new(domain: LD::U64)
    token_dictionary.put_u64([0, 0xffff_ffff_ffff_ffff], 7)
    token_transducer = LL::Transducer.new(token_dictionary)
    ordered_tokens_error = assert_raises(LL::Error) do
      token_transducer.query_u64(
        [0, 0xffff_ffff_ffff_fffe],
        1,
        order: LL::QueryOrder::DISTANCE_THEN_TERM
      )
    end
    assert_equal LL::Status::UNSUPPORTED, ordered_tokens_error.status
    token_matches = collect(token_transducer.query_u64([0, 0xffff_ffff_ffff_fffe], 1))
    assert_equal [[[0, 0xffff_ffff_ffff_ffff], 1, 7, LL::U64_DOMAIN]], token_matches
  ensure
    byte_transducer&.close
    byte_dictionary&.close
    token_transducer&.close
    token_dictionary&.close
  end

  def test_llre_product_automaton_and_rule_sets
    llre = LL::PhoneticPattern.compile_llre("@name \"Greeting\"\n^hello$")
    assert llre.matches?("hello")
    states, transitions = llre.size
    assert_operator states, :>, 0
    assert_operator transitions, :>, 0

    dictionary = text_dictionary([["cat", nil], ["cot", nil], ["cut", nil]])
    transducer = LL::Transducer.new(dictionary)
    regex = LL::PhoneticPattern.compile_regex("c[ao]t")
    assert_equal Set["cat", "cot"], transducer.query_pattern(regex, 0).map(&:term).to_set

    rules = LL::PhoneticRuleSet.parse("ph -> f\ngh ->\n")
    assert_equal 2, rules.length
    assert_equal "f", rules.apply("phgh")
    rules.close
    rules.close

    [
      LL::PhoneticRuleSet::ENGLISH_ORTHOGRAPHY,
      LL::PhoneticRuleSet::ENGLISH_PHONETIC
    ].each do |kind|
      builtin = LL::PhoneticRuleSet.builtin(kind)
      assert_operator builtin.length, :>, 0
      refute_empty builtin.apply("phone")
      builtin.close
    end
  ensure
    rules&.close
    regex&.close
    llre&.close
    transducer&.close
    dictionary&.close
  end
end
