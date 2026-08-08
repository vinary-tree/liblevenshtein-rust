require "fiddle/import"
require "rbconfig"

module VinaryTree
  module Liblevenshtein
    module Native
      extend Fiddle::Importer

      def self.library_candidates
        explicit = ENV["LIBLEVENSHTEIN_LIBRARY"]
        os, extension, filename = case RbConfig::CONFIG["host_os"]
                                  when /darwin/ then ["darwin", "dylib", "libliblevenshtein.dylib"]
                                  when /mswin|mingw/ then ["windows", "dll", "liblevenshtein.dll"]
                                  else ["linux", "so", "libliblevenshtein.so"]
                                  end
        cpu = case RbConfig::CONFIG["host_cpu"]
              when /x86_64|amd64/ then "x64"
              when /aarch64|arm64/ then "arm64"
              else RbConfig::CONFIG["host_cpu"]
              end
        platform = "#{os}-#{cpu}"
        bundled = File.expand_path("native/#{platform}/#{filename}", __dir__)
        [explicit, bundled, filename, "liblevenshtein.#{extension}"].compact
      end

      loaded = library_candidates.find do |candidate|
        begin
          dlload candidate
          true
        rescue Fiddle::DLError
          false
        end
      end
      raise Fiddle::DLError, "unable to load liblevenshtein (tried #{library_candidates.join(', ')})" unless loaded

      VtResource = struct ["void* context", "void* vtable"]
      Match = struct [
        "void* term_data", "size_t term_len", "size_t byte_len",
        "size_t distance", "uint64_t id", "uint32_t unit_domain",
        "uint8_t has_id", "uint8_t reserved[3]"
      ]
      Batch = struct ["void* matches", "size_t len", "uint64_t generation"]
      OwnedString = struct ["void* data", "size_t len"]

      extern "const char* llev_last_error_message(void)"
      extern "size_t llev_distance(const char*, size_t, const char*, size_t)"
      extern "size_t llev_distance_threshold(const char*, size_t, const char*, size_t, size_t)"
      extern "size_t llev_damerau_distance(const char*, size_t, const char*, size_t)"
      extern "size_t llev_damerau_distance_threshold(const char*, size_t, const char*, size_t, size_t)"
      extern "size_t llev_true_damerau_distance(const char*, size_t, const char*, size_t)"
      extern "size_t llev_true_damerau_distance_threshold(const char*, size_t, const char*, size_t, size_t)"
      extern "uint32_t llev_transducer_new(const void*, uint32_t, void*)"
      extern "void llev_transducer_free(void*)"
      extern "uint32_t llev_transducer_query_utf8(void*, const char*, size_t, size_t, uint32_t, void*)"
      extern "uint32_t llev_transducer_query_bytes(void*, const void*, size_t, size_t, uint32_t, void*)"
      extern "uint32_t llev_transducer_query_u64(void*, const void*, size_t, size_t, uint32_t, void*)"
      extern "uint32_t llev_query_cursor_next_batch(void*, size_t, void*)"
      extern "uint32_t llev_query_cursor_release_batch(void*, uint64_t)"
      extern "uint32_t llev_query_cursor_free(void*)"
      extern "uint32_t llev_phonetic_pattern_compile_regex(const char*, size_t, void*)"
      extern "uint32_t llev_phonetic_pattern_compile_llre(const char*, size_t, void*)"
      extern "void llev_phonetic_pattern_free(void*)"
      extern "uint32_t llev_phonetic_pattern_size(void*, void*, void*)"
      extern "uint32_t llev_phonetic_pattern_matches(void*, const char*, size_t, void*)"
      extern "uint32_t llev_transducer_query_pattern(void*, void*, uint8_t, void*)"
      extern "uint32_t llev_phonetic_rules_parse(const char*, size_t, void*)"
      extern "uint32_t llev_phonetic_rules_builtin(uint32_t, void*)"
      extern "void llev_phonetic_rules_free(void*)"
      extern "uint32_t llev_phonetic_rules_len(void*, void*)"
      extern "uint32_t llev_phonetic_rules_apply(void*, const char*, size_t, void*)"
      extern "void llev_owned_string_free(void*)"

      module_function

      def pointer_output
        Fiddle::Pointer.malloc(Fiddle::SIZEOF_VOIDP, Fiddle::RUBY_FREE)
      end

      def read_pointer(output)
        output[0, Fiddle::SIZEOF_VOIDP].unpack1("J")
      end

      def size_output
        Fiddle::Pointer.malloc(Fiddle::SIZEOF_SIZE_T, Fiddle::RUBY_FREE)
      end

      def read_size(output)
        output[0, Fiddle::SIZEOF_SIZE_T].unpack1("J")
      end
    end
  end
end
