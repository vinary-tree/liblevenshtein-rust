program test_cross_project
  use, intrinsic :: iso_c_binding
  use vinary_tree_interop, only: vt_resource
  use vinary_tree_libdictenstein
  use vinary_tree_liblevenshtein
  implicit none

  type(dictionary) :: dict, dat, suffix, persistent
  type(transducer) :: machine
  type(query_iterator) :: iterator
  type(levenshtein_match) :: item
  type(vt_resource) :: resource
  character(len=4) :: initial(4)
  integer(c_int64_t) :: values(4), mapped
  logical :: has_values(4), found, has_value, inserted, removed, seen(4)
  integer(c_int32_t) :: status
  integer(c_size_t) :: inserted_count
  integer :: count, seed
  character(len=128) :: path

  initial = [character(len=4) :: "cat", "cot", "cut", "scat"]
  values = [1_c_int64_t, 2_c_int64_t, 3_c_int64_t, 0_c_int64_t]
  has_values = [.true., .true., .true., .false.]
  call require(levenshtein_distance("kitten", "sitting") == 3_c_size_t, "Levenshtein distance")
  call require(levenshtein_distance_threshold("kitten", "sitting", 2_c_size_t) == -2_c_size_t, &
       "Levenshtein threshold")
  call require(damerau_distance("ab", "ba") == 1_c_size_t, "OSA Damerau distance")
  call require(damerau_distance_threshold("ca", "abc", 2_c_size_t) == -2_c_size_t, "OSA threshold")
  call require(true_damerau_distance("ca", "abc") == 2_c_size_t, "true Damerau distance")
  call require(true_damerau_distance_threshold("ca", "abc", 2_c_size_t) == 2_c_size_t, "true Damerau threshold")
  call new_dynamic_dawg(dict, status=status)
  call require(status == ldict_ok, "dynamic constructor")
  call dict%put_all(initial, values, has_values, inserted_count, status)
  call require(status == ldict_ok .and. inserted_count == 4, "batch insert")
  call dict%resource(resource, status)
  call require(status == ldict_ok, "resource")
  call new_transducer(machine, resource, status=status)
  call require(status == llev_ok, "transducer")
  call machine%query_text("cat", 2_c_size_t, iterator, status=status)
  call require(status == llev_ok, "query")
  call iterator%next(item, found, status)
  call require(status == llev_ok .and. found, "first result")
  seen = .false.; call mark(item%text, seen)
  call dict%remove_text("cot", removed, status)
  call dict%put_text("cut", 30_c_int64_t, inserted=inserted, status=status)
  call dict%put_text("cit", 5_c_int64_t, inserted=inserted, status=status)
  inserted_count = dict%compact(status)
  do
    call iterator%next(item, found, status)
    call require(status == llev_ok, "drain status")
    if (.not. found) exit
    call mark(item%text, seen)
  end do
  call require(all(seen), "query-start snapshot changed")

  call assert_algorithms_and_order(dict)
  call assert_query_cache(machine)
  call assert_non_text_domains()
  call assert_phonetic_surface(machine)

  do seed = 0, 63
    call property_trace(seed)
  end do

  call new_double_array_trie(dat, [character(len=5) :: "alpha", "beta"], &
       [7_c_int64_t, 0_c_int64_t], [.true., .false.], status=status)
  call dat%get_text("alpha", found, mapped, has_value, status)
  call require(status == ldict_ok .and. found .and. has_value .and. mapped == 7, "DAT")
  call new_scdawg(suffix, status=status)
  call suffix%put_text("banana", 1_c_int64_t, inserted=inserted, status=status)
  call require(suffix%contains_substring("ana", status), "SCDAWG contains")
  call require(suffix%substring_frequency("ana", status) == 2, "SCDAWG frequency")

  call system_clock(count)
  write(path, '(a,i0)') '.vinary-tree-fortran-', count
  call create_persistent_artrie(persistent, trim(path), status=status)
  call persistent%put_text("durable", 9_c_int64_t, inserted=inserted, status=status)
  call persistent%checkpoint(status); call persistent%close()
  call open_persistent_artrie(persistent, trim(path), status=status)
  call persistent%get_text("durable", found, mapped, has_value, status)
  call require(status == ldict_ok .and. found .and. mapped == 9, "persistent reopen")
  call persistent%close()
  call delete_file(trim(path))
  call delete_file(trim(path)//'.wal')
  call delete_file(trim(path)//'.wlock')

  call machine%close(); call dict%close(); call dat%close(); call suffix%close()
  print *, "Fortran cross-project snapshot conformance passed"

contains
  subroutine assert_query_cache(source)
    type(transducer), intent(in) :: source
    type(query_cache) :: cache
    type(query_cache_stats) :: counters
    type(query_iterator) :: stream
    type(levenshtein_match) :: current
    integer(c_int32_t) :: local_status
    logical :: more
    integer :: pass, count

    call new_query_cache(cache, source, 8_c_size_t, 1048576_c_size_t, local_status)
    call require(local_status == llev_ok, "query-cache constructor")
    do pass = 1, 2
      call cache%query_text("cat", 1_c_size_t, stream, status=local_status)
      call require(local_status == llev_ok, "cached query")
      count = 0
      do
        call stream%next(current, more, local_status)
        call require(local_status == llev_ok, "cached cursor")
        if (.not. more) exit
        count = count + 1
      end do
      call require(count > 0, "cached result is empty")
    end do
    counters = cache%stats(local_status)
    call require(local_status == llev_ok, "query-cache stats")
    call require(counters%requests == 2 .and. counters%hits == 1 .and. &
         counters%misses == 1, "query-cache hit accounting")
    call require(counters%resident_entries == 1 .and. &
         counters%resident_weight > 0, "query-cache residency")
    call cache%reset_stats(local_status)
    counters = cache%stats(local_status)
    call require(counters%requests == 0 .and. counters%resident_entries == 1, &
         "query-cache reset preserves residency")
    call cache%clear(local_status)
    counters = cache%stats(local_status)
    call require(counters%resident_entries == 0, "query-cache clear")
    call cache%close()
  end subroutine

  subroutine require(condition, message)
    logical, intent(in) :: condition; character(len=*), intent(in) :: message
    if (.not. condition) error stop message
  end subroutine

  subroutine mark(text, values_seen)
    character(kind=c_char, len=*), intent(in) :: text
    logical, intent(inout) :: values_seen(4)
    select case (text)
    case ("cat"); values_seen(1) = .true.
    case ("cot"); values_seen(2) = .true.
    case ("cut"); values_seen(3) = .true.
    case ("scat"); values_seen(4) = .true.
    case default; error stop "old iterator observed a new term"
    end select
  end subroutine

  subroutine assert_algorithms_and_order(subject)
    type(dictionary), intent(inout) :: subject
    type(vt_resource) :: local_resource
    type(transducer) :: local_machine
    type(query_iterator) :: stream
    type(levenshtein_match) :: current
    integer(c_int32_t), parameter :: algorithms(4) = [llev_standard, llev_transposition, &
         llev_merge_and_split, llev_damerau_levenshtein]
    character(kind=c_char, len=:), allocatable :: previous
    integer(c_size_t) :: previous_distance
    integer(c_int32_t) :: local_status
    integer :: index, result_count
    logical :: changed, more

    call subject%put_text("ab", inserted=changed, status=local_status)
    call subject%put_text("ba", inserted=changed, status=local_status)
    call subject%put_text("c", inserted=changed, status=local_status)
    call subject%put_text("abc", inserted=changed, status=local_status)
    call subject%put_text("bat", inserted=changed, status=local_status)
    call subject%put_text("cats", inserted=changed, status=local_status)
    call subject%put_text("cot", inserted=changed, status=local_status)
    call require(local_status == ldict_ok, "algorithm fixtures")
    call subject%resource(local_resource, local_status)
    call require(local_status == ldict_ok, "algorithm resource")

    do index = 1, size(algorithms)
      call new_transducer(local_machine, local_resource, algorithms(index), local_status)
      call require(local_status == llev_ok, "algorithm constructor")
      select case (algorithms(index))
      case (llev_standard)
        call require(.not. cursor_has_text(local_machine, "ba", 1_c_size_t, "ab", 1_c_size_t), &
             "standard unexpectedly accepted a transposition")
      case (llev_transposition)
        call require(cursor_has_text(local_machine, "ba", 1_c_size_t, "ab", 1_c_size_t), &
             "transposition algorithm")
      case (llev_merge_and_split)
        call require(cursor_has_text(local_machine, "ab", 1_c_size_t, "c", 1_c_size_t), &
             "merge-and-split algorithm")
      case (llev_damerau_levenshtein)
        call require(cursor_has_text(local_machine, "ca", 2_c_size_t, "abc", 2_c_size_t), &
             "true Damerau-Levenshtein algorithm")
      end select
      call local_machine%close()
    end do

    call new_transducer(local_machine, local_resource, llev_standard, local_status)
    call local_machine%query_text("cat", 1_c_size_t, stream, llev_distance_then_term, local_status)
    call require(local_status == llev_ok, "distance ordering query")
    previous_distance = 0_c_size_t
    result_count = 0
    do
      call stream%next(current, more, local_status)
      call require(local_status == llev_ok, "distance ordering status")
      if (.not. more) exit
      result_count = result_count + 1
      if (result_count == 1) then
        call require(current%text == "cat" .and. current%distance == 0, "exact result was not first")
      else
        call require(current%distance >= previous_distance, "distance ordering regressed")
        if (current%distance == previous_distance) then
          call require(previous <= current%text, "term tie-break ordering regressed")
        end if
      end if
      previous_distance = current%distance
      previous = current%text
    end do
    call require(result_count > 1, "distance ordering fixture was not discriminating")
    call local_machine%close()
  end subroutine

  function cursor_has_text(machine_value, query, maximum, expected, expected_distance) result(present)
    type(transducer), intent(in) :: machine_value
    character(len=*), intent(in) :: query, expected
    integer(c_size_t), intent(in) :: maximum, expected_distance
    type(query_iterator) :: stream
    type(levenshtein_match) :: current
    integer(c_int32_t) :: local_status
    logical :: present, more
    present = .false.
    call machine_value%query_text(query, maximum, stream, order=llev_traversal, status=local_status)
    call require(local_status == llev_ok, "algorithm query")
    do
      call stream%next(current, more, local_status)
      call require(local_status == llev_ok, "algorithm cursor")
      if (.not. more) exit
      if (current%text == expected .and. current%distance == expected_distance) present = .true.
    end do
  end function

  subroutine assert_non_text_domains()
    type(dictionary) :: byte_dictionary, token_dictionary
    type(vt_resource) :: local_resource
    type(transducer) :: local_machine
    type(query_iterator) :: stream
    type(levenshtein_match) :: current
    integer(c_int8_t), parameter :: expected_bytes(3) = [-1_c_int8_t, 0_c_int8_t, 127_c_int8_t]
    integer(c_int8_t), target :: byte_query(3)
    integer(c_int64_t), target :: token_term(2), token_query(2)
    character(kind=c_char, len=3) :: byte_term
    integer(c_int32_t) :: local_status
    logical :: changed, more

    byte_term = transfer(expected_bytes, byte_term)
    byte_query = [-1_c_int8_t, 0_c_int8_t, 126_c_int8_t]
    call new_dynamic_dawg(byte_dictionary, vt_unit_byte, local_status)
    call byte_dictionary%put_text(byte_term, -1_c_int64_t, inserted=changed, status=local_status)
    call require(local_status == ldict_ok .and. changed, "byte dictionary insert")
    call byte_dictionary%resource(local_resource, local_status)
    call new_transducer(local_machine, local_resource, llev_standard, local_status)
    call local_machine%query_bytes(byte_query, 1_c_size_t, stream, local_status)
    call require(local_status == llev_ok, "byte query")
    call byte_dictionary%close()
    call stream%next(current, more, local_status)
    call require(local_status == llev_ok .and. more, "byte query result")
    call require(current%unit_domain == vt_unit_byte, "byte result domain")
    call require(all(current%bytes == expected_bytes), "byte result payload")
    call require(current%distance == 1 .and. current%has_id .and. current%id == -1, "byte metadata")
    call stream%close(); call local_machine%close()

    token_term = [0_c_int64_t, -1_c_int64_t]
    token_query = [0_c_int64_t, -2_c_int64_t]
    call new_dynamic_dawg(token_dictionary, vt_unit_u64, local_status)
    call token_dictionary%put_u64(token_term, 7_c_int64_t, inserted=changed, status=local_status)
    call require(local_status == ldict_ok .and. changed, "u64 dictionary insert")
    call token_dictionary%resource(local_resource, local_status)
    call new_transducer(local_machine, local_resource, llev_standard, local_status)
    call local_machine%query_u64(token_query, 1_c_size_t, stream, local_status)
    call require(local_status == llev_ok, "u64 query")
    call token_dictionary%close()
    call stream%next(current, more, local_status)
    call require(local_status == llev_ok .and. more, "u64 query result")
    call require(current%unit_domain == vt_unit_u64, "u64 result domain")
    call require(all(current%tokens == token_term), "u64 result payload")
    call require(current%distance == 1 .and. current%has_id .and. current%id == 7, "u64 metadata")
    call stream%close(); call local_machine%close()
  end subroutine

  subroutine assert_phonetic_surface(text_machine)
    type(transducer), intent(in) :: text_machine
    type(phonetic_pattern) :: llre, regex
    type(phonetic_rule_set) :: rules
    type(query_iterator) :: stream
    type(levenshtein_match) :: current
    character(kind=c_char, len=:), allocatable :: rewritten
    character(len=:), allocatable :: rule_source, llre_source
    integer(c_size_t) :: states, transitions, rule_count
    integer(c_int32_t) :: local_status, kind
    logical :: accepted, more, saw_cat, saw_cot

    llre_source = '@name "Greeting"' // new_line('a') // '^hello$'
    call compile_phonetic_llre(llre, llre_source, local_status)
    call require(local_status == llev_ok, "LLRE compilation")
    accepted = llre%matches("hello", local_status)
    call require(local_status == llev_ok .and. accepted, "LLRE match")
    call llre%size(states, transitions, local_status)
    call require(local_status == llev_ok .and. states > 0 .and. transitions > 0, "LLRE size")
    call llre%close(); call llre%close()

    call compile_phonetic_regex(regex, "c[ao]t", local_status)
    call text_machine%query_pattern(regex, 0_c_int8_t, stream, local_status)
    call require(local_status == llev_ok, "product query")
    saw_cat = .false.; saw_cot = .false.
    do
      call stream%next(current, more, local_status)
      call require(local_status == llev_ok, "product cursor")
      if (.not. more) exit
      if (current%text == "cat") saw_cat = .true.
      if (current%text == "cot") saw_cot = .true.
    end do
    call require(saw_cat .and. saw_cot, "product automaton results")
    call regex%close()

    rule_source = "ph -> f" // new_line('a') // "gh ->" // new_line('a')
    call parse_phonetic_rules(rules, rule_source, local_status)
    call require(local_status == llev_ok, "parsed rule construction")
    rule_count = rules%length(local_status)
    call require(local_status == llev_ok .and. rule_count == 2, "parsed rules")
    rewritten = rules%apply("phgh", local_status)
    call require(local_status == llev_ok .and. rewritten == "f", "parsed rule application")
    call rules%close(); call rules%close()

    do kind = llev_english_orthography, llev_english_phonetic
      call builtin_phonetic_rules(rules, kind, local_status)
      call require(local_status == llev_ok, "built-in rule construction")
      rule_count = rules%length(local_status)
      call require(local_status == llev_ok .and. rule_count > 0, "built-in rules")
      rewritten = rules%apply("phone", local_status)
      call require(local_status == llev_ok .and. len(rewritten) > 0, "built-in rule application")
      call rules%close()
    end do
  end subroutine

  subroutine property_trace(trace_seed)
    integer, intent(in) :: trace_seed
    type(dictionary) :: subject
    type(transducer) :: local_machine
    type(query_iterator) :: stream
    type(levenshtein_match) :: current
    type(vt_resource) :: local_resource
    character(len=3) :: terms(32), removed_term, inserted_term
    integer(c_int64_t) :: ids(32)
    logical :: present_values(32), more, changed
    integer(c_size_t) :: batch_count
    integer(c_int32_t) :: local_status
    integer :: index, result_count
    do index = 1, 32
      write(terms(index), '(a1,i2.2)') 't', index - 1
      ids(index) = index - 1; present_values(index) = .true.
    end do
    call new_dynamic_dawg(subject, status=local_status)
    call subject%put_all(terms, ids, present_values, batch_count, local_status)
    call subject%resource(local_resource, local_status)
    call new_transducer(local_machine, local_resource, status=local_status)
    call local_machine%query_text("t00", 3_c_size_t, stream, status=local_status)
    call stream%next(current, more, local_status)
    call require(more, "property cursor empty")
    result_count = 1
    write(removed_term, '(a1,i2.2)') 't', modulo(trace_seed, 32)
    write(inserted_term, '(a1,i2.2)') 'x', trace_seed
    call subject%remove_text(removed_term, changed, local_status)
    call subject%put_text(inserted_term, int(trace_seed, c_int64_t), inserted=changed, status=local_status)
    do
      call stream%next(current, more, local_status)
      call require(local_status == llev_ok, "property cursor status")
      if (.not. more) exit
      call require(current%text(1:1) == 't', "property observed insertion")
      result_count = result_count + 1
    end do
    call require(result_count == 32, "property snapshot size")
    call local_machine%close(); call subject%close()
  end subroutine

  subroutine delete_file(filename)
    character(len=*), intent(in) :: filename
    integer :: unit, io
    open(newunit=unit, file=filename, status='old', action='readwrite', iostat=io)
    if (io == 0) close(unit, status='delete')
  end subroutine
end program test_cross_project
