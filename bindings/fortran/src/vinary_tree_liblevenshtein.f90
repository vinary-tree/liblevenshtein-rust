module vinary_tree_liblevenshtein
  use, intrinsic :: iso_c_binding
  use vinary_tree_interop, only: vt_resource, vt_unit_byte, vt_unit_unicode_scalar, vt_unit_u64
  implicit none
  private

  integer(c_int32_t), parameter, public :: llev_ok = 0, llev_end = 1
  integer(c_int32_t), parameter, public :: llev_invalid_argument = 2
  integer(c_int32_t), parameter, public :: llev_invalid_utf8 = 3
  integer(c_int32_t), parameter, public :: llev_null_pointer = 4
  integer(c_int32_t), parameter, public :: llev_panic = 5
  integer(c_int32_t), parameter, public :: llev_unsupported = 6
  integer(c_int32_t), parameter, public :: llev_io_error = 7
  integer(c_int32_t), parameter, public :: llev_closed = 8
  integer(c_int32_t), parameter, public :: llev_limit_exceeded = 9
  integer(c_int32_t), parameter, public :: llev_provider_error = 10
  integer(c_int32_t), parameter, public :: llev_batch_in_use = 11
  integer(c_int32_t), parameter, public :: llev_domain_mismatch = 12
  integer(c_int32_t), parameter, public :: llev_standard = 0
  integer(c_int32_t), parameter, public :: llev_transposition = 1
  integer(c_int32_t), parameter, public :: llev_merge_and_split = 2
  integer(c_int32_t), parameter, public :: llev_damerau_levenshtein = 3
  integer(c_int32_t), parameter, public :: llev_traversal = 0
  integer(c_int32_t), parameter, public :: llev_distance_then_term = 1
  integer(c_int32_t), parameter, public :: llev_english_orthography = 0
  integer(c_int32_t), parameter, public :: llev_english_phonetic = 1

  type, bind(c) :: c_match
    type(c_ptr) :: term_data
    integer(c_size_t) :: term_len
    integer(c_size_t) :: byte_len
    integer(c_size_t) :: distance
    integer(c_int64_t) :: id
    integer(c_int32_t) :: unit_domain
    integer(c_int8_t) :: has_id
    integer(c_int8_t) :: reserved(3)
  end type c_match

  type, bind(c) :: c_batch
    type(c_ptr) :: matches
    integer(c_size_t) :: len
    integer(c_int64_t) :: generation
  end type c_batch

  type, bind(c) :: c_owned_string
    type(c_ptr) :: data
    integer(c_size_t) :: len
  end type c_owned_string

  type, public :: levenshtein_match
    character(kind=c_char, len=:), allocatable :: text
    integer(c_int8_t), allocatable :: bytes(:)
    integer(c_int64_t), allocatable :: tokens(:)
    integer(c_size_t) :: distance = 0
    integer(c_int64_t) :: id = 0
    logical :: has_id = .false.
    integer(c_int32_t) :: unit_domain = 0
  end type levenshtein_match

  type, public :: query_iterator
    private
    type(c_ptr) :: handle = c_null_ptr
    type(c_batch) :: batch
    integer(c_size_t) :: index = 0
    logical :: leased = .false.
  contains
    procedure, public :: next => iterator_next
    procedure, public :: close => iterator_close
    final :: iterator_finalize
  end type query_iterator

  type, public :: transducer
    private
    type(c_ptr) :: handle = c_null_ptr
  contains
    procedure, public :: query_text
    procedure, public :: query_bytes
    procedure, public :: query_u64
    procedure, public :: query_pattern
    procedure, public :: close => transducer_close
    final :: transducer_finalize
  end type transducer

  type, public :: phonetic_pattern
    private
    type(c_ptr) :: handle = c_null_ptr
  contains
    procedure, public :: matches => pattern_matches
    procedure, public :: size => pattern_size
    procedure, public :: close => pattern_close
    final :: pattern_finalize
  end type phonetic_pattern

  type, public :: phonetic_rule_set
    private
    type(c_ptr) :: handle = c_null_ptr
  contains
    procedure, public :: length => rules_length
    procedure, public :: apply => rules_apply
    procedure, public :: close => rules_close
    final :: rules_finalize
  end type phonetic_rule_set

  public :: new_transducer, compile_phonetic_regex, compile_phonetic_llre
  public :: parse_phonetic_rules, builtin_phonetic_rules
  public :: levenshtein_distance, levenshtein_distance_threshold
  public :: damerau_distance, damerau_distance_threshold
  public :: true_damerau_distance, true_damerau_distance_threshold

  interface
    function c_distance(a, an, b, bn) bind(c, name="llev_distance") result(value)
      import c_ptr, c_size_t
      type(c_ptr), value :: a, b
      integer(c_size_t), value :: an, bn
      integer(c_size_t) :: value
    end function
    function c_distance_threshold(a, an, b, bn, threshold) bind(c, name="llev_distance_threshold") result(value)
      import c_ptr, c_size_t
      type(c_ptr), value :: a, b
      integer(c_size_t), value :: an, bn, threshold
      integer(c_size_t) :: value
    end function
    function c_damerau(a, an, b, bn) bind(c, name="llev_damerau_distance") result(value)
      import c_ptr, c_size_t
      type(c_ptr), value :: a, b
      integer(c_size_t), value :: an, bn
      integer(c_size_t) :: value
    end function
    function c_damerau_threshold(a, an, b, bn, threshold) bind(c, name="llev_damerau_distance_threshold") result(value)
      import c_ptr, c_size_t
      type(c_ptr), value :: a, b
      integer(c_size_t), value :: an, bn, threshold
      integer(c_size_t) :: value
    end function
    function c_true_damerau(a, an, b, bn) bind(c, name="llev_true_damerau_distance") result(value)
      import c_ptr, c_size_t
      type(c_ptr), value :: a, b
      integer(c_size_t), value :: an, bn
      integer(c_size_t) :: value
    end function
    function c_true_damerau_threshold(a, an, b, bn, threshold) bind(c, name="llev_true_damerau_distance_threshold") result(value)
      import c_ptr, c_size_t
      type(c_ptr), value :: a, b
      integer(c_size_t), value :: an, bn, threshold
      integer(c_size_t) :: value
    end function
    function c_transducer_new(resource, algorithm, output) bind(c, name="llev_transducer_new") result(status)
      import vt_resource, c_int32_t, c_ptr
      type(vt_resource), intent(in) :: resource
      integer(c_int32_t), value :: algorithm
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    subroutine c_transducer_free(handle) bind(c, name="llev_transducer_free")
      import c_ptr
      type(c_ptr), value :: handle
    end subroutine
    function c_query_text(handle, input, count, maximum, order, output) bind(c, name="llev_transducer_query_utf8") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: handle, input
      integer(c_size_t), value :: count, maximum
      integer(c_int32_t), value :: order
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_query_bytes(handle, input, count, maximum, order, output) bind(c, name="llev_transducer_query_bytes") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: handle, input
      integer(c_size_t), value :: count, maximum
      integer(c_int32_t), value :: order
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_query_u64(handle, input, count, maximum, order, output) bind(c, name="llev_transducer_query_u64") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: handle, input
      integer(c_size_t), value :: count, maximum
      integer(c_int32_t), value :: order
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_next(handle, maximum, batch) bind(c, name="llev_query_cursor_next_batch") result(status)
      import c_ptr, c_size_t, c_int32_t, c_batch
      type(c_ptr), value :: handle
      integer(c_size_t), value :: maximum
      type(c_batch), intent(out) :: batch
      integer(c_int32_t) :: status
    end function
    function c_release(handle, generation) bind(c, name="llev_query_cursor_release_batch") result(status)
      import c_ptr, c_int64_t, c_int32_t
      type(c_ptr), value :: handle
      integer(c_int64_t), value :: generation
      integer(c_int32_t) :: status
    end function
    function c_cursor_free(handle) bind(c, name="llev_query_cursor_free") result(status)
      import c_ptr, c_int32_t
      type(c_ptr), value :: handle
      integer(c_int32_t) :: status
    end function
    function c_pattern_compile(input, count, output) bind(c, name="llev_phonetic_pattern_compile_regex") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: input
      integer(c_size_t), value :: count
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_pattern_compile_llre(input, count, output) bind(c, name="llev_phonetic_pattern_compile_llre") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: input
      integer(c_size_t), value :: count
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    subroutine c_pattern_free(handle) bind(c, name="llev_phonetic_pattern_free")
      import c_ptr
      type(c_ptr), value :: handle
    end subroutine
    function c_pattern_matches(handle, input, count, output) bind(c, name="llev_phonetic_pattern_matches") result(status)
      import c_ptr, c_size_t, c_int32_t, c_int8_t
      type(c_ptr), value :: handle, input
      integer(c_size_t), value :: count
      integer(c_int8_t), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_pattern_size(handle, states, transitions) bind(c, name="llev_phonetic_pattern_size") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: handle
      integer(c_size_t), intent(out) :: states, transitions
      integer(c_int32_t) :: status
    end function
    function c_query_pattern(handle, pattern, maximum, output) bind(c, name="llev_transducer_query_pattern") result(status)
      import c_ptr, c_int8_t, c_int32_t
      type(c_ptr), value :: handle, pattern
      integer(c_int8_t), value :: maximum
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_rules_parse(input, count, output) bind(c, name="llev_phonetic_rules_parse") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: input
      integer(c_size_t), value :: count
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_rules_builtin(kind, output) bind(c, name="llev_phonetic_rules_builtin") result(status)
      import c_ptr, c_int32_t
      integer(c_int32_t), value :: kind
      type(c_ptr), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    subroutine c_rules_free(handle) bind(c, name="llev_phonetic_rules_free")
      import c_ptr
      type(c_ptr), value :: handle
    end subroutine
    function c_rules_len(handle, output) bind(c, name="llev_phonetic_rules_len") result(status)
      import c_ptr, c_size_t, c_int32_t
      type(c_ptr), value :: handle
      integer(c_size_t), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    function c_rules_apply(handle, input, count, output) bind(c, name="llev_phonetic_rules_apply") result(status)
      import c_ptr, c_size_t, c_int32_t, c_owned_string
      type(c_ptr), value :: handle, input
      integer(c_size_t), value :: count
      type(c_owned_string), intent(out) :: output
      integer(c_int32_t) :: status
    end function
    subroutine c_owned_string_free(output) bind(c, name="llev_owned_string_free")
      import c_owned_string
      type(c_owned_string), intent(inout) :: output
    end subroutine
  end interface

contains

  function byte_pointer(bytes) result(pointer)
    character(kind=c_char), target, intent(in) :: bytes(:)
    type(c_ptr) :: pointer
    if (size(bytes) == 0) then; pointer = c_null_ptr; else; pointer = c_loc(bytes(1)); end if
  end function

  function string_bytes(text) result(bytes)
    character(len=*), intent(in) :: text
    character(kind=c_char), allocatable, target :: bytes(:)
    integer :: index
    allocate(bytes(len(text)))
    do index = 1, len(text); bytes(index) = text(index:index); end do
  end function

  function levenshtein_distance(source, target) result(value)
    character(len=*), intent(in) :: source, target
    integer(c_size_t) :: value
    character(kind=c_char), allocatable, target :: a(:), b(:)
    a = string_bytes(source); b = string_bytes(target)
    value = c_distance(byte_pointer(a), int(size(a), c_size_t), byte_pointer(b), int(size(b), c_size_t))
  end function

  function levenshtein_distance_threshold(source, target, threshold) result(value)
    character(len=*), intent(in) :: source, target
    integer(c_size_t), intent(in) :: threshold
    integer(c_size_t) :: value
    character(kind=c_char), allocatable, target :: a(:), b(:)
    a = string_bytes(source); b = string_bytes(target)
    value = c_distance_threshold(byte_pointer(a), int(size(a), c_size_t), byte_pointer(b), int(size(b), c_size_t), threshold)
  end function

  function damerau_distance(source, target) result(value)
    character(len=*), intent(in) :: source, target
    integer(c_size_t) :: value
    character(kind=c_char), allocatable, target :: a(:), b(:)
    a = string_bytes(source); b = string_bytes(target)
    value = c_damerau(byte_pointer(a), int(size(a), c_size_t), byte_pointer(b), int(size(b), c_size_t))
  end function

  function damerau_distance_threshold(source, target, threshold) result(value)
    character(len=*), intent(in) :: source, target
    integer(c_size_t), intent(in) :: threshold
    integer(c_size_t) :: value
    character(kind=c_char), allocatable, target :: a(:), b(:)
    a = string_bytes(source); b = string_bytes(target)
    value = c_damerau_threshold(byte_pointer(a), int(size(a), c_size_t), byte_pointer(b), int(size(b), c_size_t), threshold)
  end function

  function true_damerau_distance(source, target) result(value)
    character(len=*), intent(in) :: source, target
    integer(c_size_t) :: value
    character(kind=c_char), allocatable, target :: a(:), b(:)
    a = string_bytes(source); b = string_bytes(target)
    value = c_true_damerau(byte_pointer(a), int(size(a), c_size_t), byte_pointer(b), int(size(b), c_size_t))
  end function

  function true_damerau_distance_threshold(source, target, threshold) result(value)
    character(len=*), intent(in) :: source, target
    integer(c_size_t), intent(in) :: threshold
    integer(c_size_t) :: value
    character(kind=c_char), allocatable, target :: a(:), b(:)
    a = string_bytes(source); b = string_bytes(target)
    value = c_true_damerau_threshold(byte_pointer(a), int(size(a), c_size_t), byte_pointer(b), int(size(b), c_size_t), threshold)
  end function

  subroutine new_transducer(value, resource, algorithm, status)
    type(transducer), intent(out) :: value
    type(vt_resource), intent(in) :: resource
    integer(c_int32_t), intent(in), optional :: algorithm
    integer(c_int32_t), intent(out), optional :: status
    integer(c_int32_t) :: code, selected
    selected = llev_standard; if (present(algorithm)) selected = algorithm
    code = c_transducer_new(resource, selected, value%handle)
    if (present(status)) status = code
  end subroutine

  subroutine query_text(self, query, maximum, iterator, order, status)
    class(transducer), intent(in) :: self
    character(len=*), intent(in) :: query
    integer(c_size_t), intent(in) :: maximum
    integer(c_int32_t), intent(in), optional :: order
    integer(c_int32_t), intent(out), optional :: status
    type(query_iterator), intent(out) :: iterator
    character(kind=c_char), allocatable, target :: input(:)
    integer(c_int32_t) :: code, selected
    selected = llev_traversal; if (present(order)) selected = order
    input = string_bytes(query)
    code = c_query_text(self%handle, byte_pointer(input), int(size(input), c_size_t), maximum, selected, iterator%handle)
    if (present(status)) status = code
  end subroutine

  subroutine query_bytes(self, query, maximum, iterator, status)
    class(transducer), intent(in) :: self
    integer(c_int8_t), target, intent(in) :: query(:)
    integer(c_size_t), intent(in) :: maximum
    integer(c_int32_t), intent(out), optional :: status
    type(query_iterator), intent(out) :: iterator
    integer(c_int32_t) :: code
    type(c_ptr) :: input
    if (size(query) == 0) then; input = c_null_ptr; else; input = c_loc(query(1)); end if
    code = c_query_bytes(self%handle, input, int(size(query), c_size_t), maximum, llev_traversal, iterator%handle)
    if (present(status)) status = code
  end subroutine

  subroutine query_u64(self, query, maximum, iterator, status)
    class(transducer), intent(in) :: self
    integer(c_int64_t), target, intent(in) :: query(:)
    integer(c_size_t), intent(in) :: maximum
    integer(c_int32_t), intent(out), optional :: status
    type(query_iterator), intent(out) :: iterator
    integer(c_int32_t) :: code
    type(c_ptr) :: input
    if (size(query) == 0) then; input = c_null_ptr; else; input = c_loc(query(1)); end if
    code = c_query_u64(self%handle, input, int(size(query), c_size_t), maximum, llev_traversal, iterator%handle)
    if (present(status)) status = code
  end subroutine

  subroutine query_pattern(self, pattern, maximum, iterator, status)
    class(transducer), intent(in) :: self
    type(phonetic_pattern), intent(in) :: pattern
    integer(c_int8_t), intent(in) :: maximum
    integer(c_int32_t), intent(out), optional :: status
    type(query_iterator), intent(out) :: iterator
    integer(c_int32_t) :: code
    code = c_query_pattern(self%handle, pattern%handle, maximum, iterator%handle)
    if (present(status)) status = code
  end subroutine

  subroutine iterator_next(self, value, found, status)
    class(query_iterator), intent(inout) :: self
    type(levenshtein_match), intent(out) :: value
    logical, intent(out) :: found
    integer(c_int32_t), intent(out), optional :: status
    type(c_match), pointer :: matches(:)
    integer(c_int32_t) :: code
    found = .false.; code = llev_ok
    if (.not. c_associated(self%handle)) then; code = llev_closed; goto 100; end if
    if (.not. self%leased .or. self%index >= self%batch%len) then
      if (self%leased) then
        code = c_release(self%handle, self%batch%generation)
        self%leased = .false.
        if (code /= llev_ok) goto 100
      end if
      code = c_next(self%handle, 256_c_size_t, self%batch)
      if (code == llev_end) then; code = llev_ok; call self%close(); goto 100; end if
      if (code /= llev_ok) goto 100
      self%leased = .true.; self%index = 0
    end if
    call c_f_pointer(self%batch%matches, matches, [int(self%batch%len)])
    self%index = self%index + 1
    call materialize_match(matches(int(self%index)), value)
    found = .true.
100 if (present(status)) status = code
  end subroutine

  subroutine materialize_match(source, value)
    type(c_match), intent(in) :: source
    type(levenshtein_match), intent(out) :: value
    character(kind=c_char), pointer :: chars(:)
    integer(c_int8_t), pointer :: bytes(:)
    integer(c_int64_t), pointer :: tokens(:)
    integer :: index
    value%distance = source%distance; value%id = source%id
    value%has_id = source%has_id /= 0; value%unit_domain = source%unit_domain
    select case (source%unit_domain)
    case (vt_unit_byte)
      call c_f_pointer(source%term_data, bytes, [int(source%byte_len)])
      allocate(value%bytes(size(bytes))); value%bytes = bytes
    case (vt_unit_unicode_scalar)
      call c_f_pointer(source%term_data, chars, [int(source%byte_len)])
      allocate(character(kind=c_char, len=int(source%byte_len)) :: value%text)
      do index = 1, size(chars); value%text(index:index) = chars(index); end do
    case (vt_unit_u64)
      call c_f_pointer(source%term_data, tokens, [int(source%term_len)])
      allocate(value%tokens(size(tokens))); value%tokens = tokens
    end select
  end subroutine

  subroutine iterator_close(self)
    class(query_iterator), intent(inout) :: self
    integer(c_int32_t) :: ignored
    if (.not. c_associated(self%handle)) return
    if (self%leased) then; ignored = c_release(self%handle, self%batch%generation); self%leased = .false.; end if
    ignored = c_cursor_free(self%handle); self%handle = c_null_ptr
  end subroutine
  subroutine iterator_finalize(self); type(query_iterator), intent(inout) :: self; call self%close(); end subroutine
  subroutine transducer_close(self)
    class(transducer), intent(inout) :: self
    if (c_associated(self%handle)) call c_transducer_free(self%handle)
    self%handle = c_null_ptr
  end subroutine
  subroutine transducer_finalize(self); type(transducer), intent(inout) :: self; call self%close(); end subroutine

  subroutine compile_phonetic_regex(value, source, status)
    type(phonetic_pattern), intent(out) :: value
    character(len=*), intent(in) :: source
    integer(c_int32_t), intent(out), optional :: status
    character(kind=c_char), allocatable, target :: input(:)
    integer(c_int32_t) :: code
    input = string_bytes(source); code = c_pattern_compile(byte_pointer(input), int(size(input), c_size_t), value%handle)
    if (present(status)) status = code
  end subroutine
  subroutine compile_phonetic_llre(value, source, status)
    type(phonetic_pattern), intent(out) :: value
    character(len=*), intent(in) :: source
    integer(c_int32_t), intent(out), optional :: status
    character(kind=c_char), allocatable, target :: input(:)
    integer(c_int32_t) :: code
    input = string_bytes(source); code = c_pattern_compile_llre(byte_pointer(input), int(size(input), c_size_t), value%handle)
    if (present(status)) status = code
  end subroutine
  function pattern_matches(self, text, status) result(accepted)
    class(phonetic_pattern), intent(in) :: self
    character(len=*), intent(in) :: text
    integer(c_int32_t), intent(out), optional :: status
    logical :: accepted
    integer(c_int8_t) :: output
    integer(c_int32_t) :: code
    character(kind=c_char), allocatable, target :: input(:)
    input = string_bytes(text); code = c_pattern_matches(self%handle, byte_pointer(input), int(size(input), c_size_t), output)
    accepted = output /= 0; if (present(status)) status = code
  end function
  subroutine pattern_size(self, states, transitions, status)
    class(phonetic_pattern), intent(in) :: self
    integer(c_size_t), intent(out) :: states, transitions
    integer(c_int32_t), intent(out), optional :: status
    integer(c_int32_t) :: code
    code = c_pattern_size(self%handle, states, transitions); if (present(status)) status = code
  end subroutine
  subroutine pattern_close(self)
    class(phonetic_pattern), intent(inout) :: self
    if (c_associated(self%handle)) call c_pattern_free(self%handle)
    self%handle = c_null_ptr
  end subroutine
  subroutine pattern_finalize(self); type(phonetic_pattern), intent(inout) :: self; call self%close(); end subroutine

  subroutine parse_phonetic_rules(value, source, status)
    type(phonetic_rule_set), intent(out) :: value
    character(len=*), intent(in) :: source
    integer(c_int32_t), intent(out), optional :: status
    character(kind=c_char), allocatable, target :: input(:)
    integer(c_int32_t) :: code
    input = string_bytes(source); code = c_rules_parse(byte_pointer(input), int(size(input), c_size_t), value%handle)
    if (present(status)) status = code
  end subroutine
  subroutine builtin_phonetic_rules(value, kind, status)
    type(phonetic_rule_set), intent(out) :: value
    integer(c_int32_t), intent(in) :: kind
    integer(c_int32_t), intent(out), optional :: status
    integer(c_int32_t) :: code
    code = c_rules_builtin(kind, value%handle); if (present(status)) status = code
  end subroutine
  function rules_length(self, status) result(value)
    class(phonetic_rule_set), intent(in) :: self
    integer(c_int32_t), intent(out), optional :: status
    integer(c_size_t) :: value
    integer(c_int32_t) :: code
    code = c_rules_len(self%handle, value); if (present(status)) status = code
  end function
  function rules_apply(self, text, status) result(value)
    class(phonetic_rule_set), intent(in) :: self
    character(len=*), intent(in) :: text
    integer(c_int32_t), intent(out), optional :: status
    character(kind=c_char, len=:), allocatable :: value
    character(kind=c_char), allocatable, target :: input(:)
    character(kind=c_char), pointer :: output_chars(:)
    type(c_owned_string) :: output
    integer(c_int32_t) :: code
    integer :: index
    input = string_bytes(text); code = c_rules_apply(self%handle, byte_pointer(input), int(size(input), c_size_t), output)
    if (code == llev_ok) then
      call c_f_pointer(output%data, output_chars, [int(output%len)])
      allocate(character(kind=c_char, len=int(output%len)) :: value)
      do index = 1, size(output_chars); value(index:index) = output_chars(index); end do
      call c_owned_string_free(output)
    else
      allocate(character(kind=c_char, len=0) :: value)
    end if
    if (present(status)) status = code
  end function
  subroutine rules_close(self)
    class(phonetic_rule_set), intent(inout) :: self
    if (c_associated(self%handle)) call c_rules_free(self%handle)
    self%handle = c_null_ptr
  end subroutine
  subroutine rules_finalize(self); type(phonetic_rule_set), intent(inout) :: self; call self%close(); end subroutine
end module vinary_tree_liblevenshtein
