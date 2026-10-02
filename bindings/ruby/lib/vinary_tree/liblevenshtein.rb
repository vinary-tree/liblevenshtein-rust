require "thread"
require_relative "liblevenshtein/version"
require_relative "liblevenshtein/native"
require_relative "liblevenshtein/generated_enums"

##
# Bindings for the Vinary Tree native library family.
module VinaryTree
  ##
  # Unicode edit distances and snapshot-consistent fuzzy dictionary search.
  # The distance methods accept valid UTF-8 Ruby strings and count Unicode
  # scalar values; invalid UTF-8 returns the native +SIZE_MAX+ sentinel.
  # Thresholded distance methods use +SIZE_MAX-1+ for an over-bound result.
  # query_bytes and query_u64 preserve their distinct input domains.
  #
  # A producer such as +libdictenstein+ supplies the dictionary resource:
  #
  #   require "vinary_tree/libdictenstein"
  #   require "vinary_tree/liblevenshtein"
  #   dictionary = VinaryTree::Libdictenstein::DynamicDawg.new
  #   dictionary.put("cat", 7)
  #   transducer = VinaryTree::Liblevenshtein::Transducer.new(dictionary)
  #   begin
  #     transducer.query("cot", 1).each { |match| p [match.term, match.distance] }
  #   ensure
  #     transducer.close
  #     dictionary.close
  #   end
  module Liblevenshtein
    # Native status indicating success.
    OK = Status::OK
    # Native status indicating that a finite cursor is exhausted.
    END_STATUS = Status::END_OF_STREAM
    # A match term is an arbitrary byte string.
    BYTE_DOMAIN = 1
    # A match term is valid UTF-8, representing Unicode scalar values.
    UNICODE_DOMAIN = 2
    # A match term is an array of unsigned 64-bit tokens.
    U64_DOMAIN = 3

    # A native operation failed. Inspect #status, not the human-readable message,
    # to branch on the failure; future numeric status values remain representable.
    class Error < StandardError
      # The original native status value, including values unknown to this gem.
      attr_reader :status

      # Build an exception from +status+ and the current thread's native diagnostic.
      def initialize(status)
        @status = status
        super("liblevenshtein status #{status}: #{Native.llev_last_error_message.to_s}")
      end
    end

    # Raise Error unless a native status indicates success. This is primarily
    # used by the facade; applications normally call a typed public method.
    def self.check(status)
      raise Error, status unless status == OK
    end

    # Standard Levenshtein distance in Unicode scalar values.
    #
    #   VinaryTree::Liblevenshtein.distance("kitten", "sitting") # => 3
    #
    # Inputs must encode valid UTF-8; this method does not normalize
    # canonically equivalent Unicode text.
    def self.distance(source, target)
      Native.llev_distance(source.b, source.bytesize, target.b, target.bytesize)
    end

    # Standard distance up to +threshold+; returns the native over-bound
    # sentinel (+SIZE_MAX-1+) when the exact distance exceeds it.
    #
    #   VinaryTree::Liblevenshtein.distance_threshold("cat", "cut", 1) # => 1
    def self.distance_threshold(source, target, threshold)
      Native.llev_distance_threshold(source.b, source.bytesize, target.b, target.bytesize, threshold)
    end

    # Optimal-string-alignment distance, allowing one adjacent transposition
    # but disallowing repeated edits of a substring. This is not unrestricted
    # Damerau-Levenshtein distance and is not a metric.
    def self.damerau_distance(source, target)
      Native.llev_damerau_distance(source.b, source.bytesize, target.b, target.bytesize)
    end

    # Thresholded optimal-string-alignment distance; returns +SIZE_MAX-1+
    # if the distance exceeds +threshold+.
    def self.damerau_distance_threshold(source, target, threshold)
      Native.llev_damerau_distance_threshold(source.b, source.bytesize, target.b, target.bytesize, threshold)
    end

    # Unrestricted Damerau-Levenshtein distance; unlike #damerau_distance,
    # transpositions can participate in later edits.
    #
    #   VinaryTree::Liblevenshtein.true_damerau_distance("ca", "abc") # => 2
    def self.true_damerau_distance(source, target)
      Native.llev_true_damerau_distance(source.b, source.bytesize, target.b, target.bytesize)
    end

    # Thresholded unrestricted Damerau-Levenshtein distance; returns
    # +SIZE_MAX-1+ when the exact distance exceeds +threshold+.
    def self.true_damerau_distance_threshold(source, target, threshold)
      Native.llev_true_damerau_distance_threshold(source.b, source.bytesize, target.b, target.bytesize, threshold)
    end

    # An owned, immutable result. +term+ is UTF-8 text, a binary string, or an
    # Array of unsigned 64-bit integers according to +domain+. +distance+ is
    # the edit cost; +id+ is a dictionary value or +nil+ if absent. Results
    # remain valid after the cursor and source dictionary close.
    Match = Data.define(:term, :distance, :id, :domain)

    # Immutable counters returned by QueryCache#stats. +requests+ is the sum
    # of +hits+ and +misses+; +resident_entries+ and +resident_weight+ describe
    # the current bounded resident set, not cumulative admissions.
    QueryCacheStats = Data.define(
      :requests, :hits, :misses, :admissions, :rejections, :evictions,
      :resident_entries, :resident_weight
    )

    module Finalizer # :nodoc:
      module_function
      def for(box, function)
        proc do
          pointer = box[0]
          function.call(pointer) unless pointer.nil? || pointer.zero?
          box[0] = 0
        rescue StandardError
          nil
        end
      end
    end

    # A lifetime gate that permits concurrent native calls and only coordinates
    # close with in-flight operations. It never serializes ordinary reads.
    class ConcurrentHandle # :nodoc:
      def initialize(pointer, releaser)
        @pointer = pointer
        @releaser = releaser
        @active = 0
        @closing = false
        @mutex = Mutex.new
        @condition = ConditionVariable.new
      end

      def with_pointer
        pointer = @mutex.synchronize do
          raise IOError, "native handle is closed" if @closing || @pointer.zero?
          @active += 1
          @pointer
        end
        yield pointer
      ensure
        @mutex.synchronize do
          @active -= 1
          @condition.broadcast if @active.zero?
        end if pointer
      end

      def close
        pointer = @mutex.synchronize do
          return 0 if @pointer.zero?
          @closing = true
          @condition.wait(@mutex) until @active.zero?
          result = @pointer
          @pointer = 0
          result
        end
        @releaser.call(pointer) unless pointer.zero?
        pointer
      end

      def self.finalizer(handle)
        proc { handle.close rescue nil }
      end
    end

    # A reusable fuzzy matcher over a retained dictionary resource. A producer
    # must provide +with_resource { |context, vtable| }+; this constructor
    # retains that resource without copying dictionary entries. Every query
    # captures one immutable dictionary revision at its start, so an open cursor
    # remains valid after later mutations or after the dictionary closes.
    #
    #   dictionary = VinaryTree::Libdictenstein::DynamicDawg.new
    #   dictionary.put("cat", 7)
    #   transducer = VinaryTree::Liblevenshtein::Transducer.new(dictionary)
    #   begin
    #     transducer.query("cot", 1).each { |match| p match.term }
    #   ensure
    #     transducer.close
    #     dictionary.close
    #   end
    class Transducer
      # Standard insertion, deletion, and substitution edits.
      STANDARD = Algorithm::STANDARD
      # Optimal-string-alignment edits with adjacent transposition.
      TRANSPOSITION = Algorithm::TRANSPOSITION
      # Standard edits plus two-to-one merge and one-to-two split.
      MERGE_AND_SPLIT = Algorithm::MERGE_AND_SPLIT
      # Unrestricted Damerau-Levenshtein edits.
      DAMERAU_LEVENSHTEIN = Algorithm::DAMERAU_LEVENSHTEIN

      # Retain +dictionary+ and select one Algorithm constant. Raises
      # ArgumentError unless the producer implements +with_resource+; native
      # negotiation failures raise Error. Call #close when finished.
      def initialize(dictionary, algorithm: STANDARD)
        raise ArgumentError, "dictionary must respond to with_resource" unless dictionary.respond_to?(:with_resource)
        output = Native.pointer_output
        dictionary.with_resource do |context, vtable|
          resource = Native::VtResource.malloc
          resource.context = context
          resource.vtable = vtable
          Liblevenshtein.check(Native.llev_transducer_new(resource, algorithm, output))
        end
        @handle = ConcurrentHandle.new(Native.read_pointer(output), Native.method(:llev_transducer_free))
        ObjectSpace.define_finalizer(self, ConcurrentHandle.finalizer(@handle))
      end

      # Start a one-shot UTF-8 query at edit distance +maximum_distance+.
      # Returns a Query whose Match#term values are UTF-8 strings. +order+
      # selects QueryOrder::TRAVERSAL or QueryOrder::DISTANCE_THEN_TERM.
      def query(text, maximum_distance, order: QueryOrder::TRAVERSAL)
        start_query(:llev_transducer_query_utf8, text.b, maximum_distance, order)
      end

      # Query arbitrary binary terms, including embedded zero bytes. Results
      # contain binary strings; no UTF-8 decoding or normalization occurs.
      # Distance-then-term ordering is currently unsupported in this domain.
      def query_bytes(bytes, maximum_distance, order: QueryOrder::TRAVERSAL)
        start_query(:llev_transducer_query_bytes, bytes.b, maximum_distance, order)
      end

      # Query a sequence of unsigned 64-bit tokens. Results contain token
      # arrays, preserving zero and the full unsigned 64-bit range.
      # Distance-then-term ordering is currently unsupported in this domain.
      def query_u64(tokens, maximum_distance, order: QueryOrder::TRAVERSAL)
        packed = tokens.pack("Q*")
        start_query(:llev_transducer_query_u64, packed, maximum_distance, order, tokens.length)
      end

      # Match a compiled PhoneticPattern against this dictionary revision.
      # +maximum_distance+ must fit in the native unsigned-byte range 0..255.
      #
      #   pattern = VinaryTree::Liblevenshtein::PhoneticPattern.compile_regex("c[ao]t")
      #   begin
      #     transducer.query_pattern(pattern, 0).each { |match| p match.term }
      #   ensure
      #     pattern.close
      #   end
      def query_pattern(pattern, maximum_distance)
        raise ArgumentError, "maximum distance must be between 0 and 255" unless (0..255).cover?(maximum_distance)
        with_pointer do |pointer|
          pattern.__send__(:with_pointer) do |pattern_pointer|
            output = Native.pointer_output
            Liblevenshtein.check(Native.llev_transducer_query_pattern(pointer, pattern_pointer, maximum_distance, output))
            Query.new(Native.read_pointer(output))
          end
        end
      end

      # Release the retained native resource. Closing twice is harmless;
      # subsequent query starts raise IOError. Already-started queries own
      # independent snapshots and remain valid.
      def close
        @handle.close
        ObjectSpace.undefine_finalizer(self)
        nil
      end

      private

      def with_pointer
        @handle.with_pointer { |pointer| yield pointer }
      end

      def start_query(function, input, maximum_distance, order, units = input.bytesize)
        raise ArgumentError, "maximum distance must be nonnegative" if maximum_distance.negative?
        with_pointer do |pointer|
          output = Native.pointer_output
          Liblevenshtein.check(Native.public_send(function, pointer, input, units, maximum_distance, order, output))
          Query.new(Native.read_pointer(output))
        end
      end
    end

    # Exclusive synchronization-free bounded memo for complete repeated
    # queries. Limits apply independently to each result-order shard. Use one
    # cache per worker for parallel workloads; sharing one cache across workers
    # is unsupported. A hit returns semantically identical owned matches to a
    # cold query, without exposing a borrowed native result buffer.
    #
    #   cache = VinaryTree::Liblevenshtein::QueryCache.new(transducer,
    #     max_entries: 128, max_weight: 1 << 20)
    #   begin
    #     cache.query("cot", 1).each { |match| p match.term }
    #     p cache.stats.hits
    #   ensure
    #     cache.close
    #   end
    class QueryCache
      # Default maximum number of resident entries per order shard.
      DEFAULT_ENTRIES = 1024
      # Default maximum resident weight in bytes per order shard.
      DEFAULT_WEIGHT = 64 * 1024 * 1024

      # Retain +transducer+ with independent nonnegative entry/byte limits.
      # A zero limit disables admissions without changing query correctness.
      def initialize(transducer, max_entries: DEFAULT_ENTRIES, max_weight: DEFAULT_WEIGHT)
        raise ArgumentError, "max_entries must be nonnegative" if max_entries.negative?
        raise ArgumentError, "max_weight must be nonnegative" if max_weight.negative?
        output = Native.pointer_output
        transducer.__send__(:with_pointer) do |pointer|
          Liblevenshtein.check(
            Native.llev_query_cache_new(pointer, max_entries, max_weight, output)
          )
        end
        @box = [Native.read_pointer(output)]
        ObjectSpace.define_finalizer(
          self, Finalizer.for(@box, Native.method(:llev_query_cache_free))
        )
      end

      # Return a QueryCacheStats snapshot of requests, decisions, and residency.
      # The cache must be open.
      def stats
        raw = Native::QueryCacheStats.malloc
        Liblevenshtein.check(Native.llev_query_cache_stats(pointer, raw))
        QueryCacheStats.new(
          raw.requests, raw.hits, raw.misses, raw.admissions, raw.rejections,
          raw.evictions, raw.resident_entries, raw.resident_weight
        )
      end

      # Number of currently resident entries.
      def length
        stats.resident_entries
      end
      # Alias for #length.
      alias size length
      # Whether the cache has no resident entries.
      def empty?
        length.zero?
      end

      # Evict all resident results, preserving cumulative statistics.
      # Returns +self+ for chaining.
      def clear
        Liblevenshtein.check(Native.llev_query_cache_clear(pointer))
        self
      end

      # Zero counters while preserving resident results. Returns +self+.
      def reset_stats
        Liblevenshtein.check(Native.llev_query_cache_reset_stats(pointer))
        self
      end

      # Query Unicode text, returning a one-shot Query. Cache identity includes
      # the captured dictionary revision, term, distance bound, and result order.
      def query(text, maximum_distance, order: QueryOrder::TRAVERSAL)
        start_query(:llev_query_cache_query_utf8, text.b, maximum_distance, order)
      end

      # Query byte-domain terms without text decoding; returns a Query.
      # Distance-then-term ordering is unsupported in this domain.
      def query_bytes(bytes, maximum_distance, order: QueryOrder::TRAVERSAL)
        start_query(:llev_query_cache_query_bytes, bytes.b, maximum_distance, order)
      end

      # Query unsigned 64-bit token terms; returns a Query.
      # Distance-then-term ordering is unsupported in this domain.
      def query_u64(tokens, maximum_distance, order: QueryOrder::TRAVERSAL)
        start_query(
          :llev_query_cache_query_u64,
          tokens.pack("Q*"), maximum_distance, order, tokens.length
        )
      end

      # Release cache storage. Closing twice is harmless; later operations
      # raise IOError. The source transducer has an independent lifetime.
      def close
        return nil if @box[0].zero?
        Native.llev_query_cache_free(@box[0])
        @box[0] = 0
        ObjectSpace.undefine_finalizer(self)
        nil
      end

      private

      def pointer
        raise IOError, "query cache is closed" if @box[0].zero?
        @box[0]
      end

      def start_query(function, input, maximum_distance, order, units = input.bytesize)
        raise ArgumentError, "maximum distance must be nonnegative" if maximum_distance.negative?
        output = Native.pointer_output
        Liblevenshtein.check(
          Native.public_send(
            function, pointer, input, units, maximum_distance, order, output
          )
        )
        Query.new(Native.read_pointer(output))
      end
    end

    # A one-shot, closeable Enumerable over an immutable dictionary snapshot.
    # Iteration borrows at most BATCH_SIZE native matches at once, copies each
    # into an owned Match, and releases the batch before returning. A cursor
    # closes on normal exhaustion, +break+, or an exception in the block.
    # Calling +each+ without a block returns an Enumerator; consuming it claims
    # the cursor. Keep each cursor on one Ruby thread or fiber.
    #
    #   query = transducer.query("cot", 1)
    #   query.each { |match| p [match.term, match.distance, match.id] }
    class Query
      include Enumerable
      # Maximum native matches copied before the next batch lease.
      BATCH_SIZE = 256

      # Wrap an owned native cursor. Applications normally obtain queries from
      # Transducer or QueryCache rather than calling this initializer directly.
      def initialize(pointer)
        @box = [pointer]
        @claimed = false
        @mutex = Mutex.new
        ObjectSpace.define_finalizer(self, Finalizer.for(@box, proc { |value| Native.llev_query_cursor_free(value) }))
      end

      # Yield owned Match records exactly once. With no block, return an
      # Enumerator. A second traversal raises IOError; early termination still
      # closes the cursor. Enumerable methods such as +map+ and +find+ work.
      def each
        return enum_for(__method__) unless block_given?
        @mutex.synchronize do
          raise IOError, "query is one-shot" if @claimed
          @claimed = true
        end
        begin
          loop do
            batch = Native::Batch.malloc
            status = Native.llev_query_cursor_next_batch(@box[0], BATCH_SIZE, batch)
            break if status == END_STATUS
            Liblevenshtein.check(status)
            begin
              batch.len.times do |index|
                item = Native::Match.new(batch.matches + index * Native::Match.size)
                yield materialize(item)
              end
            ensure
              Liblevenshtein.check(Native.llev_query_cursor_release_batch(@box[0], batch.generation))
            end
          end
        ensure
          close
        end
      end

      # Release the cursor if still open. Safe to call after iteration and more
      # than once; it does not close the source dictionary or transducer.
      def close
        @mutex.synchronize do
          return if @box[0].zero?
          Liblevenshtein.check(Native.llev_query_cursor_free(@box[0]))
          @box[0] = 0
          ObjectSpace.undefine_finalizer(self)
        end
      end

      private

      def materialize(item)
        pointer = Fiddle::Pointer.new(item.term_data)
        term = case item.unit_domain
               when BYTE_DOMAIN then pointer[0, item.byte_len].b
               when UNICODE_DOMAIN then pointer[0, item.byte_len].dup.force_encoding(Encoding::UTF_8)
               when U64_DOMAIN then pointer[0, item.term_len * 8].unpack("Q*")
               else raise IOError, "unknown native term domain #{item.unit_domain}"
               end
        Match.new(term, item.distance, item.has_id.zero? ? nil : item.id, item.unit_domain)
      end
    end

    # Immutable, compiled phonetic matcher. Compile once for repeated matching
    # or dictionary queries, and close it deterministically.
    #
    #   pattern = VinaryTree::Liblevenshtein::PhoneticPattern.compile_regex("c[ao]t")
    #   begin
    #     p pattern.matches?("cat") # => true
    #   ensure
    #     pattern.close
    #   end
    class PhoneticPattern
      # Compile a regex source into an immutable native pattern.
      def self.compile_regex(source)
        compile(source, false)
      end
      # Compile a source in the library's LLRE phonetic-pattern language.
      def self.compile_llre(source)
        compile(source, true)
      end
      # :nodoc:
      def self.compile(source, llre)
        output = Native.pointer_output
        status = llre ? Native.llev_phonetic_pattern_compile_llre(source.b, source.bytesize, output) : Native.llev_phonetic_pattern_compile_regex(source.b, source.bytesize, output)
        Liblevenshtein.check(status)
        new(Native.read_pointer(output))
      end

      # Wrap an owned native pattern. Prefer .compile_regex or .compile_llre.
      def initialize(pointer)
        @handle = ConcurrentHandle.new(pointer, Native.method(:llev_phonetic_pattern_free))
        ObjectSpace.define_finalizer(self, ConcurrentHandle.finalizer(@handle))
      end
      # Whether the compiled pattern accepts UTF-8 +text+.
      def matches?(text)
        with_pointer do |pointer|
          output = Fiddle::Pointer.malloc(1, Fiddle::RUBY_FREE)
          Liblevenshtein.check(Native.llev_phonetic_pattern_matches(pointer, text.b, text.bytesize, output))
          output[0].positive?
        end
      end
      # Return [state_count, transition_count] for the compiled automaton.
      def size
        with_pointer do |pointer|
          states = Native.size_output; transitions = Native.size_output
          Liblevenshtein.check(Native.llev_phonetic_pattern_size(pointer, states, transitions))
          [Native.read_size(states), Native.read_size(transitions)]
        end
      end
      # Release the native pattern. Idempotent; further inspection raises
      # IOError. A query already started from this pattern retains its snapshot.
      def close
        @handle.close; ObjectSpace.undefine_finalizer(self); nil
      end
      private
      def with_pointer
        @handle.with_pointer { |pointer| yield pointer }
      end
    end

    # A reusable set of phonetic rewrite rules. Parsing/selection is separate
    # from applying rules, so reuse one set across input strings.
    #
    #   rules = VinaryTree::Liblevenshtein::PhoneticRuleSet.parse("ph -> f\n")
    #   begin
    #     p rules.apply("phone") # => "fone"
    #   ensure
    #     rules.close
    #   end
    class PhoneticRuleSet
      # Built-in English orthographic normalization rules.
      ENGLISH_ORTHOGRAPHY = PhoneticRuleSetKind::ENGLISH_ORTHOGRAPHY
      # Built-in English phonetic transformation rules.
      ENGLISH_PHONETIC = PhoneticRuleSetKind::ENGLISH_PHONETIC
      # Parse native rewrite-rule source into a reusable rule set.
      # Invalid rules raise Error with a status and native diagnostic.
      def self.parse(source)
        output = Native.pointer_output; Liblevenshtein.check(Native.llev_phonetic_rules_parse(source.b, source.bytesize, output)); new(Native.read_pointer(output))
      end
      # Load a PhoneticRuleSetKind constant without parsing application text.
      def self.builtin(kind)
        output = Native.pointer_output; Liblevenshtein.check(Native.llev_phonetic_rules_builtin(kind, output)); new(Native.read_pointer(output))
      end
      # Wrap an owned native rule set. Prefer .parse or .builtin.
      def initialize(pointer)
        @handle = ConcurrentHandle.new(pointer, Native.method(:llev_phonetic_rules_free))
        ObjectSpace.define_finalizer(self, ConcurrentHandle.finalizer(@handle))
      end
      # Number of compiled rewrite rules.
      def length
        with_pointer do |pointer|
          output = Native.size_output; Liblevenshtein.check(Native.llev_phonetic_rules_len(pointer, output)); Native.read_size(output)
        end
      end
      # Rewrite UTF-8 +text+ and return an independently owned Ruby String.
      def apply(text)
        with_pointer do |pointer|
          output = Native::OwnedString.malloc
          Liblevenshtein.check(Native.llev_phonetic_rules_apply(pointer, text.b, text.bytesize, output))
          begin Fiddle::Pointer.new(output.data)[0, output.len].force_encoding(Encoding::UTF_8)
          ensure Native.llev_owned_string_free(output)
          end
        end
      end
      # Release the native rule set. Idempotent; later operations raise IOError.
      def close
        @handle.close; ObjectSpace.undefine_finalizer(self); nil
      end
      private
      def with_pointer
        @handle.with_pointer { |pointer| yield pointer }
      end
    end
  end
end
