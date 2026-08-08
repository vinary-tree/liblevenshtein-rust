require "thread"
require_relative "liblevenshtein/version"
require_relative "liblevenshtein/native"

module VinaryTree
  module Liblevenshtein
    OK = 0
    END_STATUS = 1
    BYTE_DOMAIN = 1
    UNICODE_DOMAIN = 2
    U64_DOMAIN = 3

    class Error < StandardError
      attr_reader :status
      def initialize(status)
        @status = status
        super("liblevenshtein status #{status}: #{Native.llev_last_error_message.to_s}")
      end
    end

    def self.check(status)
      raise Error, status unless status == OK
    end

    class << self
      def distance(source, target) = Native.llev_distance(source.b, source.bytesize, target.b, target.bytesize)
      def distance_threshold(source, target, threshold) = Native.llev_distance_threshold(source.b, source.bytesize, target.b, target.bytesize, threshold)
      def damerau_distance(source, target) = Native.llev_damerau_distance(source.b, source.bytesize, target.b, target.bytesize)
      def damerau_distance_threshold(source, target, threshold) = Native.llev_damerau_distance_threshold(source.b, source.bytesize, target.b, target.bytesize, threshold)
      def true_damerau_distance(source, target) = Native.llev_true_damerau_distance(source.b, source.bytesize, target.b, target.bytesize)
      def true_damerau_distance_threshold(source, target, threshold) = Native.llev_true_damerau_distance_threshold(source.b, source.bytesize, target.b, target.bytesize, threshold)
    end

    Match = Data.define(:term, :distance, :id, :domain)

    module Finalizer
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
    class ConcurrentHandle
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

      def self.finalizer(handle) = proc { handle.close rescue nil }
    end

    class Transducer
      STANDARD = 0
      TRANSPOSITION = 1
      MERGE_AND_SPLIT = 2
      DAMERAU_LEVENSHTEIN = 3

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

      def query(text, maximum_distance, order: 0)
        start_query(:llev_transducer_query_utf8, text.b, maximum_distance, order)
      end

      def query_bytes(bytes, maximum_distance)
        start_query(:llev_transducer_query_bytes, bytes.b, maximum_distance, 0)
      end

      def query_u64(tokens, maximum_distance)
        packed = tokens.pack("Q*")
        start_query(:llev_transducer_query_u64, packed, maximum_distance, 0, tokens.length)
      end

      def query_pattern(pattern, maximum_distance)
        with_pointer do |pointer|
          pattern.__send__(:with_pointer) do |pattern_pointer|
            output = Native.pointer_output
            Liblevenshtein.check(Native.llev_transducer_query_pattern(pointer, pattern_pointer, maximum_distance, output))
            Query.new(Native.read_pointer(output))
          end
        end
      end

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
        with_pointer do |pointer|
          output = Native.pointer_output
          Liblevenshtein.check(Native.public_send(function, pointer, input, units, maximum_distance, order, output))
          Query.new(Native.read_pointer(output))
        end
      end
    end

    class Query
      include Enumerable
      BATCH_SIZE = 256

      def initialize(pointer)
        @box = [pointer]
        @claimed = false
        @mutex = Mutex.new
        ObjectSpace.define_finalizer(self, Finalizer.for(@box, proc { |value| Native.llev_query_cursor_free(value) }))
      end

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

    class PhoneticPattern
      def self.compile_regex(source) = compile(source, false)
      def self.compile_llre(source) = compile(source, true)
      def self.compile(source, llre)
        output = Native.pointer_output
        status = llre ? Native.llev_phonetic_pattern_compile_llre(source.b, source.bytesize, output) : Native.llev_phonetic_pattern_compile_regex(source.b, source.bytesize, output)
        Liblevenshtein.check(status)
        new(Native.read_pointer(output))
      end

      def initialize(pointer)
        @handle = ConcurrentHandle.new(pointer, Native.method(:llev_phonetic_pattern_free))
        ObjectSpace.define_finalizer(self, ConcurrentHandle.finalizer(@handle))
      end
      def matches?(text)
        with_pointer do |pointer|
          output = Fiddle::Pointer.malloc(1, Fiddle::RUBY_FREE)
          Liblevenshtein.check(Native.llev_phonetic_pattern_matches(pointer, text.b, text.bytesize, output))
          output[0].positive?
        end
      end
      def size
        with_pointer do |pointer|
          states = Native.size_output; transitions = Native.size_output
          Liblevenshtein.check(Native.llev_phonetic_pattern_size(pointer, states, transitions))
          [Native.read_size(states), Native.read_size(transitions)]
        end
      end
      def close
        @handle.close; ObjectSpace.undefine_finalizer(self); nil
      end
      private
      def with_pointer
        @handle.with_pointer { |pointer| yield pointer }
      end
    end

    class PhoneticRuleSet
      ENGLISH_ORTHOGRAPHY = 0
      ENGLISH_PHONETIC = 1
      def self.parse(source)
        output = Native.pointer_output; Liblevenshtein.check(Native.llev_phonetic_rules_parse(source.b, source.bytesize, output)); new(Native.read_pointer(output))
      end
      def self.builtin(kind)
        output = Native.pointer_output; Liblevenshtein.check(Native.llev_phonetic_rules_builtin(kind, output)); new(Native.read_pointer(output))
      end
      def initialize(pointer)
        @handle = ConcurrentHandle.new(pointer, Native.method(:llev_phonetic_rules_free))
        ObjectSpace.define_finalizer(self, ConcurrentHandle.finalizer(@handle))
      end
      def length
        with_pointer do |pointer|
          output = Native.size_output; Liblevenshtein.check(Native.llev_phonetic_rules_len(pointer, output)); Native.read_size(output)
        end
      end
      def apply(text)
        with_pointer do |pointer|
          output = Native::OwnedString.malloc
          Liblevenshtein.check(Native.llev_phonetic_rules_apply(pointer, text.b, text.bytesize, output))
          begin Fiddle::Pointer.new(output.data)[0, output.len].force_encoding(Encoding::UTF_8)
          ensure Native.llev_owned_string_free(output)
          end
        end
      end
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
