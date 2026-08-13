#!/usr/bin/env ruby
# frozen_string_literal: true

# Ruby harness for the cross-language benchmark program.
#
# Implements harnesses/common/PROTOCOL.md over the fiddle facades
# (VinaryTree::Liblevenshtein + VinaryTree::Libdictenstein). The runner
# provides RUBYLIB to both binding lib trees plus the
# LIBLEVENSHTEIN_LIBRARY / LIBDICTENSTEIN_LIBRARY release overrides.
#
# Fairness notes (PROTOCOL.md section 10): default MRI, NO --yjit (the primary
# row; a labeled yjit sensitivity run is reported separately by the runner).
# Dictionary population uses the facade's single batch call
# (DynamicDawg#put_all -> ldict_dictionary_insert_text_batch).

require "json"
require "vinary_tree/liblevenshtein"
require "vinary_tree/libdictenstein"

LL = VinaryTree::Liblevenshtein
LD = VinaryTree::Libdictenstein

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
BATCH_SIZE = 256
WALL_CAP_SECONDS = 300.0
SAMPLE_DEFINITION =
  "one full pass over the query set; every cursor fully drained and " \
  "(term, distance) materialized"

ALGORITHMS = {
  "standard" => LL::Transducer::STANDARD,
  "transposition" => LL::Transducer::TRANSPOSITION,
  "merge_and_split" => LL::Transducer::MERGE_AND_SPLIT,
  "damerau_levenshtein" => LL::Transducer::DAMERAU_LEVENSHTEIN,
}.freeze

# ---------------------------------------------------------------------------
# Checksum primitives (PROTOCOL.md section 8) + self-test (section 2)
# ---------------------------------------------------------------------------

def fnv1a64(bytes)
  hash = FNV_OFFSET
  bytes.each_byte { |byte| hash = ((hash ^ byte) * FNV_PRIME) & MASK64 }
  hash
end

def entry_hash(term, distance)
  hash = FNV_OFFSET
  term.each_byte { |byte| hash = ((hash ^ byte) * FNV_PRIME) & MASK64 }
  hash = (hash * FNV_PRIME) & MASK64 # separator: XOR with 0x00 is identity
  8.times do |i|
    hash = ((hash ^ ((distance >> (8 * i)) & 0xFF)) * FNV_PRIME) & MASK64
  end
  hash
end

def self_test
  abort_unless = lambda do |actual, wanted, label|
    die(format("checksum self-test failed for %s: got %016x, want %016x",
               label, actual, wanted)) unless actual == wanted
  end
  abort_unless.call(fnv1a64(""), 0xCBF29CE484222325, 'fnv1a64("")')
  abort_unless.call(fnv1a64("a"), 0xAF63DC4C8601EC8C, 'fnv1a64("a")')
  abort_unless.call(entry_hash("cat", 1), 0x9697FA3E50464BC4, "entry(cat,1)")
  abort_unless.call(entry_hash("cat", 0), 0xB592C1475B3595E5, "entry(cat,0)")
  abort_unless.call(entry_hash("cot", 1), 0xB8ACC5D3816BCDEA, "entry(cot,1)")
  abort_unless.call((entry_hash("cat", 0) + entry_hash("cot", 1)) & MASK64,
                    0x6E3F871ADCA163CF, "checksum{2}")
  abort_unless.call(0, 0x0000000000000000, "checksum{}")
end

# ---------------------------------------------------------------------------
# CLI (PROTOCOL.md section 1)
# ---------------------------------------------------------------------------

def die(message)
  warn("bench-cross-ruby: #{message}")
  exit(2)
end

def mono_ns
  Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)
end

def parse_args(argv)
  args = {
    mode: nil, algorithm: nil, max_distance: -1, dictionary: nil, queries: nil,
    backend: nil, out: nil, samples: 30, warmup_seconds: 3.0, gate_limit: 200,
    reps: 10, cells: nil,
  }
  index = 0
  while index < argv.length
    die("flag requires a value: #{argv[index]}") if index + 1 >= argv.length
    flag = argv[index]
    value = argv[index + 1]
    case flag
    when "--mode" then args[:mode] = value
    when "--algorithm" then args[:algorithm] = value
    when "--max-distance" then args[:max_distance] = Integer(value)
    when "--dictionary" then args[:dictionary] = value
    when "--queries" then args[:queries] = value
    when "--backend" then args[:backend] = value
    when "--out" then args[:out] = value
    when "--samples" then args[:samples] = Integer(value)
    when "--warmup-seconds" then args[:warmup_seconds] = Float(value)
    when "--gate-limit" then args[:gate_limit] = Integer(value)
    when "--reps" then args[:reps] = Integer(value)
    when "--cells" then args[:cells] = value
    else die("unknown flag: #{flag}")
    end
    index += 2
  end
  if args[:mode].nil? || args[:dictionary].nil? || args[:backend].nil?
    die("--mode, --dictionary, --backend are required")
  end
  args
end

# ---------------------------------------------------------------------------
# Input loading (PROTOCOL.md section 3)
# ---------------------------------------------------------------------------

def read_lines(path)
  data = File.read(path, encoding: Encoding::UTF_8)
  lines = data.split("\n").reject(&:empty?)
  die("#{path} contains no lines") if lines.empty?
  lines
end

def assert_strictly_sorted(lines, path)
  (0...(lines.length - 1)).each do |i|
    # String#<=> on binary representations is bytewise: the required order.
    if (lines[i].b <=> lines[i + 1].b) >= 0
      die("#{path} is not strictly byte-sorted at line #{i + 1}: " \
          "#{lines[i].inspect} >= #{lines[i + 1].inspect}")
    end
  end
end

# ---------------------------------------------------------------------------
# Dictionary + transducer side (PROTOCOL.md section 4)
# ---------------------------------------------------------------------------

class Side
  def initialize
    @dictionary = nil
    @transducer = nil
    @prepared_entries = nil
  end

  def prepared_entries(terms)
    # Built once, reused across construct-mode rebuilds (section 3.4).
    @prepared_entries ||= terms.map { |term| [term, nil] }
  end

  def build_dictionary(terms, backend)
    case backend
    when "dynamic_dawg"
      dawg = LD::DynamicDawg.new
      inserted = dawg.put_all(prepared_entries(terms)) # ONE batch call (section 4)
      die("batch insert count mismatch: #{inserted} != #{terms.length}") if inserted != terms.length
      @dictionary = dawg
    when "double_array_trie"
      @dictionary = LD::DoubleArrayTrie.new(prepared_entries(terms))
    else
      die("unknown backend: #{backend}")
    end
  end

  def free_dictionary
    unless @transducer.nil?
      @transducer.close
      @transducer = nil
    end
    unless @dictionary.nil?
      @dictionary.close
      @dictionary = nil
    end
  end

  def create_transducer(algorithm)
    @transducer&.close
    mapped = ALGORITHMS[algorithm]
    die("unknown algorithm: #{algorithm}") if mapped.nil?
    @transducer = LL::Transducer.new(@dictionary, algorithm: mapped)
  end

  # One full pass: every cursor fully drained, (term, distance) materialized;
  # the facade's Query#each closes the one-shot cursor on completion. Timed
  # passes accumulate only the O(1) triple; the checksum is computed
  # exclusively in untimed gate/verify contexts.
  def full_pass(queries, limit, max_distance, with_checksum)
    matches = 0
    term_bytes = 0
    distance_sum = 0
    checksum = 0
    index = 0
    transducer = @transducer
    while index < limit
      transducer.query(queries[index], max_distance).each do |match|
        term = match.term
        matches += 1
        term_bytes += term.bytesize # UTF-8 byte length
        distance_sum += match.distance
        checksum = (checksum + entry_hash(term, match.distance)) & MASK64 if with_checksum
      end
      index += 1
    end
    [matches, term_bytes, distance_sum, checksum]
  end
end

# ---------------------------------------------------------------------------
# Result JSON (PROTOCOL.md section 11 — runner post-fills run_id, sha256s,
# cell_snapshot, environment_ref, and the memory object)
# ---------------------------------------------------------------------------

def render_result(args, mode, algorithm, max_distance, queries_path, query_count,
                  term_count, backend, construct_ns, warmup_passes, samples_ns,
                  triple, checksum, construct_times, status, notes)
  samples_requested =
    case mode
    when "construct" then args[:reps]
    when "query" then args[:samples]
    else 0
    end
  dictionary = {
    "file" => args[:dictionary],
    "term_count" => term_count,
    "structure" => backend,
    "unit_domain" => "unicode_scalar",
  }
  dictionary["construct_ns"] = construct_ns unless construct_ns.nil?
  result = {
    "schema_version" => "1.0.0",
    "suite" => "cross-language-v1",
    "timestamp_utc" => Time.now.utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "target" => {
      "language" => "ruby",
      "implementation" => "vinary-tree",
      "backend" => "fiddle",
      "runtime_version" => "ruby #{RUBY_VERSION}",
      "library_version" => "0.10.0",
      "artifact" => { "kind" => "local-build", "id" => "vinary-tree-liblevenshtein@0.10.0" },
    },
    "dictionary" => dictionary,
    "workload" => {
      "queryset" => File.basename(queries_path, ".txt"),
      "file" => queries_path,
      "query_count" => query_count,
    },
    "algorithm" => algorithm,
    "max_distance" => max_distance,
    "mode" => mode == "memory-child" ? "memory" : mode,
    "protocol" => {
      "timer" => "monotonic",
      "harness" => "self-timed",
      "warmup_seconds_min" => args[:warmup_seconds],
      "warmup_passes" => warmup_passes,
      "samples_requested" => samples_requested,
      "sample_definition" => SAMPLE_DEFINITION,
      "batch_size" => BATCH_SIZE,
      "wall_cap_seconds" => WALL_CAP_SECONDS.to_i,
    },
    "status" => status,
    "notes" => notes,
  }
  if construct_times.nil?
    result["measurements"] = {
      "samples_ns" => samples_ns,
      "sample_count" => samples_ns.length,
      "matches_per_pass" => triple[0],
      "term_bytes_per_pass" => triple[1],
      "distance_sum_per_pass" => triple[2],
      "checksum_hex" => format("%016x", checksum),
    }
  else
    result["construct"] = {
      "reps" => construct_times.length,
      "times_ns" => construct_times,
      "term_count" => term_count,
    }
  end
  result
end

def write_result(out_path, result)
  require "fileutils"
  FileUtils.mkdir_p(File.dirname(out_path))
  File.write(out_path, JSON.pretty_generate(result) + "\n")
end

# ---------------------------------------------------------------------------
# Modes (PROTOCOL.md sections 6-7)
# ---------------------------------------------------------------------------

def run_query_cell(side, args, queries, algorithm, max_distance, queries_path, out_path,
                   term_count, construct_ns, base_notes)
  gate = side.full_pass(queries, queries.length, max_distance, true) # untimed gate pass
  gate_triple = gate[0, 3]

  warm_start = mono_ns
  warmup_budget_ns = (args[:warmup_seconds] * 1e9).round
  warmup_passes = 0
  last_pass_ns = 0
  while mono_ns - warm_start < warmup_budget_ns || warmup_passes < 2
    t0 = mono_ns
    triple = side.full_pass(queries, queries.length, max_distance, false)[0, 3]
    last_pass_ns = mono_ns - t0
    die("nondeterministic result during warmup") unless triple == gate_triple
    warmup_passes += 1
  end

  sample_count = args[:samples]
  status = "ok"
  notes = base_notes.dup
  last_pass_seconds = last_pass_ns / 1e9
  if sample_count * last_pass_seconds > WALL_CAP_SECONDS
    reduced = [10, (WALL_CAP_SECONDS / last_pass_seconds).to_i].max
    notes << format("samples reduced from %d to %d by the %ds wall cap (estimated pass %.3fs)",
                    sample_count, reduced, WALL_CAP_SECONDS.to_i, last_pass_seconds)
    sample_count = reduced
    status = "degraded"
  end

  samples_ns = Array.new(sample_count) # preallocated (section 3.4)
  sample_count.times do |i|
    t0 = mono_ns
    triple = side.full_pass(queries, queries.length, max_distance, false)[0, 3]
    samples_ns[i] = mono_ns - t0
    die("nondeterministic result during measurement") unless triple == gate_triple
  end

  write_result(out_path, render_result(args, "query", algorithm, max_distance, queries_path,
                                       queries.length, term_count, args[:backend], construct_ns,
                                       warmup_passes, samples_ns, gate_triple, gate[3], nil,
                                       status, notes))
end

def main
  self_test
  args = parse_args(ARGV)

  terms = read_lines(args[:dictionary])
  assert_strictly_sorted(terms, args[:dictionary])
  side = Side.new
  yjit_enabled = defined?(RubyVM::YJIT) && RubyVM::YJIT.enabled?
  base_notes = [
    "fiddle facade (bindings/ruby)",
    "default MRI, --yjit #{yjit_enabled ? 'ENABLED (sensitivity row)' : 'disabled'} " \
    "(fairness rule 6: primary row is no-yjit)",
  ]

  if args[:mode] == "construct"
    die("--out is required for construct mode") if args[:out].nil?
    side.build_dictionary(terms, args[:backend]) # warmup build (section 6.2)
    side.free_dictionary
    times = Array.new(args[:reps]) # preallocated
    args[:reps].times do |r|
      t0 = mono_ns
      side.build_dictionary(terms, args[:backend])
      times[r] = mono_ns - t0
      side.free_dictionary
    end
    notes = base_notes +
            ["construct mode: timed region is the build from the pre-sorted in-memory list only"]
    write_result(args[:out], render_result(args, "construct", "standard", 1,
                                           args[:queries] || "workload/queries/hits.txt", 1,
                                           terms.length, args[:backend], nil, 1, [],
                                           [0, 0, 0], 0, times, "ok", notes))
    return
  end

  build_start = mono_ns
  side.build_dictionary(terms, args[:backend])
  construct_ns = mono_ns - build_start

  run_one = lambda do |algorithm, max_distance, queries_path, out_path|
    side.create_transducer(algorithm)
    queries = read_lines(queries_path)
    case args[:mode]
    when "verify"
      limit = [args[:gate_limit], queries.length].min
      m, b, d, checksum = side.full_pass(queries, limit, max_distance, true)
      write_result(out_path, render_result(args, "verify", algorithm, max_distance,
                                           queries_path, limit, terms.length, args[:backend],
                                           construct_ns, 0, [], [m, b, d], checksum, nil,
                                           "ok", base_notes))
    when "memory-child"
      m, b, d, checksum = side.full_pass(queries, queries.length, max_distance, true)
      write_result(out_path, render_result(args, "memory-child", algorithm, max_distance,
                                           queries_path, queries.length, terms.length,
                                           args[:backend], construct_ns, 0, [], [m, b, d],
                                           checksum, nil, "ok", base_notes))
    when "query"
      run_query_cell(side, args, queries, algorithm, max_distance, queries_path, out_path,
                     terms.length, construct_ns, base_notes)
    else
      die("unknown mode: #{args[:mode]}")
    end
  end

  if args[:cells]
    File.readlines(args[:cells], chomp: true).each do |line|
      line = line.strip
      next if line.empty? || line.start_with?("#")
      fields = line.split("\t")
      die("cells row needs 4 tab-separated fields: #{line}") if fields.length != 4
      run_one.call(fields[0], Integer(fields[1]), fields[2], fields[3])
    end
  else
    if args[:algorithm].nil? || args[:max_distance] < 0 || args[:queries].nil? || args[:out].nil?
      die("--algorithm, --max-distance, --queries, --out are required")
    end
    run_one.call(args[:algorithm], args[:max_distance], args[:queries], args[:out])
  end
end

main
