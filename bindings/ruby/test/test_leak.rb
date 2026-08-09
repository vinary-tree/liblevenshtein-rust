# C9 leak-discipline suite for the Ruby facade, measured with GC + ObjectSpace.
# A >=10,000-cycle create/use/free loop must reach a steady state: after a full
# GC no facade wrapper (Transducer, Query) may remain live beyond the warmup
# baseline, and the total live-object count must not drift upward. Native
# handles are freed at close() (and by finalizers); ObjectSpace confirms the
# managed wrappers are reclaimed rather than accumulating.

require "minitest/autorun"
require "vinary_tree/liblevenshtein"
require "vinary_tree/libdictenstein"

class LeakTest < Minitest::Test
  M = VinaryTree::Liblevenshtein
  D = VinaryTree::Libdictenstein
  CYCLES = 10_000
  WARMUP = 2_000
  # Generous slot-growth ceiling; a per-cycle leak would accrue far more.
  MAX_SLOT_GROWTH = 200_000

  def full_gc
    GC.start(full_mark: true, immediate_sweep: true)
  end

  def live_count(klass)
    full_gc
    ObjectSpace.each_object(klass).count
  end

  def transducer_cycle
    dictionary = D::DynamicDawg.new
    %w[cat cot cut scat].each_with_index { |term, index| dictionary.put(term, index) }
    transducer = M::Transducer.new(dictionary)
    transducer.query("cat", 2).each { |_match| }
    transducer.close
    dictionary.close
  end

  def test_transducer_cycle_reclaims_wrappers_and_holds_steady
    WARMUP.times { transducer_cycle }
    baseline_transducers = live_count(M::Transducer)
    baseline_queries = live_count(M::Query)
    full_gc
    baseline_slots = GC.stat(:heap_live_slots)

    CYCLES.times { transducer_cycle }

    assert_operator live_count(M::Transducer), :<=, baseline_transducers,
                    "Transducer wrappers accumulated across #{CYCLES} cycles"
    assert_operator live_count(M::Query), :<=, baseline_queries,
                    "Query wrappers accumulated across #{CYCLES} cycles"
    full_gc
    growth = GC.stat(:heap_live_slots) - baseline_slots
    assert_operator growth, :<, MAX_SLOT_GROWTH,
                    "live slots grew #{growth} across #{CYCLES} cycles"
  end

  def test_phonetic_cycle_holds_steady
    cycle = lambda do
      pattern = M::PhoneticPattern.compile_regex("c[ao]t")
      pattern.matches?("cat")
      pattern.close
      rules = M::PhoneticRuleSet.builtin(M::PhoneticRuleSet::ENGLISH_ORTHOGRAPHY)
      rules.apply("phone")
      rules.close
    end
    WARMUP.times { cycle.call }
    full_gc
    baseline_slots = GC.stat(:heap_live_slots)
    CYCLES.times { cycle.call }
    full_gc
    growth = GC.stat(:heap_live_slots) - baseline_slots
    assert_operator growth, :<, MAX_SLOT_GROWTH,
                    "live slots grew #{growth} across #{CYCLES} cycles"
  end
end
