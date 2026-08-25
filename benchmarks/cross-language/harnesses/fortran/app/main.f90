! Fortran harness for the cross-language benchmark program.
!
! Implements harnesses/common/PROTOCOL.md over the iso_c_binding facades
! (liblevenshtein + libdictenstein, sharing the
! vinary_tree_interop resource type). Fairness notes (PROTOCOL.md section
! 10): fpm release profile, no runtime tuning; dynamic_dawg construction is
! ONE facade put_all batch call (section 4).
!
! Clock (PROTOCOL.md section 9): system_clock with int64 count/count_rate,
! reduced to nanoseconds (xl_clock). Checksums (section 8): integer(int64)
! with ieor/iand/ishft and half-decomposed wrapping multiply (xl_checksum).
program bench_cross_fortran
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_size_t
  use, intrinsic :: iso_fortran_env, only: int64, real64, compiler_version
  use xl_diag, only: die
  use xl_checksum, only: checksum_self_test, entry_hash, wrap_add64, to_hex16
  use xl_clock, only: clock_init, now_ticks, ticks_to_ns, seconds_to_ticks
  use xl_lines, only: line_list, read_lines, assert_strictly_sorted
  use xl_json, only: cell_output, write_result, add_note, note_width
  use vinary_tree_interop, only: vt_resource
  use vinary_tree_libdictenstein, only: dictionary, new_dynamic_dawg, &
    new_double_array_trie, ldict_ok, last_error_message
  use vinary_tree_liblevenshtein, only: transducer, query_iterator, &
    levenshtein_match, new_transducer, llev_ok, llev_standard, &
    llev_transposition, llev_merge_and_split, llev_damerau_levenshtein
  implicit none

  character(len=1), parameter :: tab_char = achar(9)
  real(real64), parameter :: wall_cap_seconds = 300.0_real64

  ! CLI state (PROTOCOL.md section 1)
  character(len=:), allocatable :: mode, algorithm_name, dictionary_path
  character(len=:), allocatable :: queries_path, backend, out_path, cells_path
  integer :: max_distance = -1
  integer :: samples = 30
  integer :: gate_limit = 200
  integer :: reps = 10
  real(real64) :: warmup_seconds = 3.0_real64

  type(line_list) :: terms
  type(dictionary) :: dict
  type(vt_resource) :: resource
  character(len=:), allocatable :: runtime_description
  integer(int64) :: construct_ns = 0_int64
  integer(c_int32_t) :: status
  integer(int64) :: build_start
  integer :: row

  call checksum_self_test()   ! PROTOCOL.md section 2, before touching inputs
  call clock_init()
  call parse_cli()
  runtime_description = compiler_version()

  call read_lines(dictionary_path, terms)
  call assert_strictly_sorted(terms, dictionary_path)

  if (mode == 'construct') then
    call run_construct()
  else
    build_start = now_ticks()
    call build_dictionary(dict)
    construct_ns = ticks_to_ns(now_ticks() - build_start)
    call dict%resource(resource, status)
    if (status /= ldict_ok) call die('dictionary_resource failed: '//last_error_message())

    if (allocated(cells_path)) then
      block
        type(line_list) :: rows
        character(len=:), allocatable :: row_algorithm, row_queries, row_out
        integer :: row_distance
        call read_lines(cells_path, rows)
        do row = 1, rows%count
          if (rows%text(row)(1:1) == '#') cycle
          call split_cells_row(rows%text(row)(1:rows%length(row)), &
                               row_algorithm, row_distance, row_queries, row_out)
          call run_one(row_algorithm, row_distance, row_queries, row_out)
        end do
      end block
    else
      if (.not. allocated(algorithm_name)) call die('--algorithm is required')
      if (max_distance < 0) call die('--max-distance is required')
      if (.not. allocated(queries_path)) call die('--queries is required')
      if (.not. allocated(out_path)) call die('--out is required')
      call run_one(algorithm_name, max_distance, queries_path, out_path)
    end if
    call dict%close()
  end if

contains

  ! ------------------------------------------------------------------
  ! CLI (PROTOCOL.md section 1); unknown flags abort.
  ! ------------------------------------------------------------------

  subroutine parse_cli()
    integer :: argc, i
    character(len=4096) :: flag, value
    argc = command_argument_count()
    i = 1
    do while (i <= argc)
      call get_command_argument(i, flag)
      if (i + 1 > argc) call die('flag requires a value: '//trim(flag))
      call get_command_argument(i + 1, value)
      select case (trim(flag))
      case ('--mode')
        mode = trim(value)
      case ('--algorithm')
        algorithm_name = trim(value)
      case ('--max-distance')
        max_distance = parse_int(trim(value))
      case ('--dictionary')
        dictionary_path = trim(value)
      case ('--queries')
        queries_path = trim(value)
      case ('--backend')
        backend = trim(value)
      case ('--out')
        out_path = trim(value)
      case ('--samples')
        samples = parse_int(trim(value))
      case ('--warmup-seconds')
        warmup_seconds = parse_real(trim(value))
      case ('--gate-limit')
        gate_limit = parse_int(trim(value))
      case ('--reps')
        reps = parse_int(trim(value))
      case ('--cells')
        cells_path = trim(value)
      case default
        call die('unknown flag: '//trim(flag))
      end select
      i = i + 2
    end do
    if (.not. allocated(mode)) call die('--mode is required')
    if (.not. allocated(dictionary_path)) call die('--dictionary is required')
    if (.not. allocated(backend)) call die('--backend is required')
  end subroutine parse_cli

  function parse_int(text) result(value)
    character(len=*), intent(in) :: text
    integer :: value, io
    read (text, *, iostat=io) value
    if (io /= 0) call die('not an integer: '//text)
  end function parse_int

  function parse_real(text) result(value)
    character(len=*), intent(in) :: text
    real(real64) :: value
    integer :: io
    read (text, *, iostat=io) value
    if (io /= 0) call die('not a number: '//text)
  end function parse_real

  function algorithm_code(name) result(code)
    character(len=*), intent(in) :: name
    integer(c_int32_t) :: code
    select case (name)
    case ('standard')
      code = llev_standard
    case ('transposition')
      code = llev_transposition
    case ('merge_and_split')
      code = llev_merge_and_split
    case ('damerau_levenshtein')
      code = llev_damerau_levenshtein
    case default
      code = llev_standard
      call die('unknown algorithm: '//name)
    end select
  end function algorithm_code

  subroutine split_cells_row(row_text, row_algorithm, row_distance, row_queries, row_out)
    character(len=*), intent(in) :: row_text
    character(len=:), allocatable, intent(out) :: row_algorithm, row_queries, row_out
    integer, intent(out) :: row_distance
    integer :: tab1, tab2, tab3
    tab1 = index(row_text, tab_char)
    if (tab1 == 0) call die('cells row needs 4 tab-separated fields: '//row_text)
    tab2 = index(row_text(tab1 + 1:), tab_char)
    if (tab2 == 0) call die('cells row needs 4 tab-separated fields: '//row_text)
    tab2 = tab1 + tab2
    tab3 = index(row_text(tab2 + 1:), tab_char)
    if (tab3 == 0) call die('cells row needs 4 tab-separated fields: '//row_text)
    tab3 = tab2 + tab3
    row_algorithm = row_text(1:tab1 - 1)
    row_distance = parse_int(row_text(tab1 + 1:tab2 - 1))
    row_queries = row_text(tab2 + 1:tab3 - 1)
    row_out = row_text(tab3 + 1:)
  end subroutine split_cells_row

  ! ------------------------------------------------------------------
  ! Dictionary construction (PROTOCOL.md section 4)
  ! ------------------------------------------------------------------

  subroutine build_dictionary(target_dict)
    type(dictionary), intent(out) :: target_dict
    integer(c_int32_t) :: build_status
    integer(c_size_t) :: inserted
    select case (backend)
    case ('dynamic_dawg')
      call new_dynamic_dawg(target_dict, status=build_status)
      if (build_status /= ldict_ok) &
        call die('ldict_dynamic_dawg_new failed: '//last_error_message())
      ! ONE facade batch call; len_trim inside put_all recovers each padded
      ! term's true byte length (all-ASCII workload, no trailing blanks).
      call target_dict%put_all(terms%text, inserted=inserted, status=build_status)
      if (build_status /= ldict_ok) &
        call die('insert_text_batch failed: '//last_error_message())
      if (inserted /= int(terms%count, c_size_t)) call die('batch insert count mismatch')
    case ('double_array_trie')
      call new_double_array_trie(target_dict, terms%text, status=build_status)
      if (build_status /= ldict_ok) &
        call die('ldict_double_array_trie_new failed: '//last_error_message())
    case default
      call die('unknown backend: '//backend)
    end select
  end subroutine build_dictionary

  ! ------------------------------------------------------------------
  ! The pass (PROTOCOL.md section 5): drain every cursor, materialize
  ! (term, distance), accumulate the O(1) triple (+ checksum when untimed).
  ! ------------------------------------------------------------------

  subroutine full_pass(machine, queries, limit, distance, with_checksum, &
                       matches, term_bytes, distance_sum, checksum)
    type(transducer), intent(in) :: machine
    type(line_list), intent(in) :: queries
    integer, intent(in) :: limit, distance
    logical, intent(in) :: with_checksum
    integer(int64), intent(out) :: matches, term_bytes, distance_sum, checksum
    type(query_iterator) :: iterator
    type(levenshtein_match) :: item
    logical :: found
    integer(c_int32_t) :: pass_status
    integer :: i
    matches = 0_int64
    term_bytes = 0_int64
    distance_sum = 0_int64
    checksum = 0_int64
    do i = 1, limit
      call machine%query_text(queries%text(i)(1:queries%length(i)), &
                              int(distance, c_size_t), iterator, status=pass_status)
      if (pass_status /= llev_ok) call die('query_text failed')
      do
        call iterator%next(item, found, pass_status)
        if (pass_status /= llev_ok) call die('query iterator failed')
        if (.not. found) exit
        matches = matches + 1_int64
        term_bytes = term_bytes + int(len(item%text), int64)  ! UTF-8 byte length
        distance_sum = distance_sum + item%distance
        if (with_checksum) then
          checksum = wrap_add64(checksum, entry_hash(item%text, item%distance))
        end if
      end do
      call iterator%close()  ! idempotent: next() already closed at END
    end do
  end subroutine full_pass

  ! ------------------------------------------------------------------
  ! Shared result-cell scaffolding
  ! ------------------------------------------------------------------

  subroutine fill_common(cell, mode_name, algo_name, distance, qpath, qcount)
    type(cell_output), intent(out) :: cell
    character(len=*), intent(in) :: mode_name, algo_name, qpath
    integer, intent(in) :: distance, qcount
    cell%mode = mode_name
    cell%structure = backend
    cell%algorithm = algo_name
    cell%dictionary_file = dictionary_path
    cell%queries_file = qpath
    cell%status = 'ok'
    cell%runtime_version = runtime_description
    cell%max_distance = distance
    cell%term_count = terms%count
    cell%query_count = qcount
    cell%warmup_seconds = warmup_seconds
    call add_note(cell, 'iso_c_binding facade over the shared C ABI')
    call add_note(cell, 'clock: system_clock int64 count/count_rate reduced to nanoseconds')
    if (backend == 'dynamic_dawg') then
      call add_note(cell, 'dynamic_dawg populated with ONE facade put_all batch call')
    end if
  end subroutine fill_common

  ! ------------------------------------------------------------------
  ! Modes (PROTOCOL.md section 6)
  ! ------------------------------------------------------------------

  subroutine run_construct()
    type(dictionary) :: build_target
    type(cell_output) :: cell
    integer(int64) :: t0
    integer :: r
    character(len=:), allocatable :: construct_queries
    if (.not. allocated(out_path)) call die('--out is required for construct mode')
    if (allocated(queries_path)) then
      construct_queries = queries_path
    else
      construct_queries = 'workload/queries/hits.txt'
    end if
    call build_dictionary(build_target)  ! warmup build
    call build_target%close()
    call fill_common(cell, 'construct', 'standard', 1, construct_queries, 1)
    allocate (cell%construct_times(reps))  ! preallocated (section 3.4)
    do r = 1, reps
      t0 = now_ticks()
      call build_dictionary(build_target)
      cell%construct_times(r) = ticks_to_ns(now_ticks() - t0)
      call build_target%close()
    end do
    cell%warmup_passes = 1
    cell%samples_requested = reps
    call add_note(cell, &
      'construct mode: timed region is the build from the pre-sorted in-memory list only')
    call write_result(out_path, cell)
  end subroutine run_construct

  subroutine run_one(algo_name, distance, qpath, opath)
    character(len=*), intent(in) :: algo_name, qpath, opath
    integer, intent(in) :: distance
    type(transducer) :: machine
    type(line_list) :: queries
    integer(c_int32_t) :: cell_status
    call new_transducer(machine, resource, algorithm=algorithm_code(algo_name), &
                        status=cell_status)
    if (cell_status /= llev_ok) call die('llev_transducer_new failed')
    call read_lines(qpath, queries)
    select case (mode)
    case ('verify')
      call run_gate_cell(machine, queries, 'verify', min(gate_limit, queries%count), &
                         algo_name, distance, qpath, opath)
    case ('memory-child')
      call run_gate_cell(machine, queries, 'memory-child', queries%count, &
                         algo_name, distance, qpath, opath)
    case ('query')
      call run_query_cell(machine, queries, algo_name, distance, qpath, opath)
    case default
      call die('unknown mode: '//mode)
    end select
    call machine%close()
  end subroutine run_one

  ! verify (section 6.4) and memory-child (section 6.3): one untimed gate
  ! pass with checksum + triple; no timing fields.
  subroutine run_gate_cell(machine, queries, mode_name, limit, algo_name, &
                           distance, qpath, opath)
    type(transducer), intent(in) :: machine
    type(line_list), intent(in) :: queries
    character(len=*), intent(in) :: mode_name, algo_name, qpath, opath
    integer, intent(in) :: limit, distance
    type(cell_output) :: cell
    integer(int64) :: matches, term_bytes, distance_sum, checksum
    call full_pass(machine, queries, limit, distance, .true., &
                   matches, term_bytes, distance_sum, checksum)
    call fill_common(cell, mode_name, algo_name, distance, qpath, limit)
    cell%has_construct_ns = .true.
    cell%construct_ns = construct_ns
    cell%matches = matches
    cell%term_bytes = term_bytes
    cell%distance_sum = distance_sum
    cell%checksum_hex = to_hex16(checksum)
    call write_result(opath, cell)
  end subroutine run_gate_cell

  ! query (section 6.1): gate pass, warmup to the deadline (>= 2 passes),
  ! deterministic wall-cap arithmetic, timed samples with triple asserts.
  subroutine run_query_cell(machine, queries, algo_name, distance, qpath, opath)
    type(transducer), intent(in) :: machine
    type(line_list), intent(in) :: queries
    character(len=*), intent(in) :: algo_name, qpath, opath
    integer, intent(in) :: distance
    type(cell_output) :: cell
    integer(int64) :: ref_matches, ref_bytes, ref_distance, checksum
    integer(int64) :: matches, term_bytes, distance_sum, ignored
    integer(int64) :: warm_start, warmup_tick_budget, t0, last_pass_ticks, last_pass_ns
    integer :: warmup_passes, sample_count, reduced, i
    real(real64) :: last_pass_seconds
    character(len=note_width) :: cap_note

    call full_pass(machine, queries, queries%count, distance, .true., &
                   ref_matches, ref_bytes, ref_distance, checksum)

    warm_start = now_ticks()
    warmup_tick_budget = seconds_to_ticks(warmup_seconds)
    warmup_passes = 0
    last_pass_ticks = 0_int64
    do while (now_ticks() - warm_start < warmup_tick_budget .or. warmup_passes < 2)
      t0 = now_ticks()
      call full_pass(machine, queries, queries%count, distance, .false., &
                     matches, term_bytes, distance_sum, ignored)
      last_pass_ticks = now_ticks() - t0
      if (matches /= ref_matches .or. term_bytes /= ref_bytes .or. &
          distance_sum /= ref_distance) call die('nondeterministic result during warmup')
      warmup_passes = warmup_passes + 1
    end do

    call fill_common(cell, 'query', algo_name, distance, qpath, queries%count)
    sample_count = samples
    last_pass_ns = ticks_to_ns(last_pass_ticks)
    last_pass_seconds = real(last_pass_ns, real64) / 1.0e9_real64
    if (real(sample_count, real64) * last_pass_seconds > wall_cap_seconds) then
      reduced = max(10, int(wall_cap_seconds / last_pass_seconds))
      write (cap_note, '(a,i0,a,i0,a,f0.3,a)') 'samples reduced from ', sample_count, &
        ' to ', reduced, ' by the 300s wall cap (estimated pass ', last_pass_seconds, 's)'
      call add_note(cell, trim(cap_note))
      sample_count = reduced
      cell%status = 'degraded'
    end if

    allocate (cell%samples_ns(sample_count))  ! preallocated (section 3.4)
    do i = 1, sample_count
      t0 = now_ticks()
      call full_pass(machine, queries, queries%count, distance, .false., &
                     matches, term_bytes, distance_sum, ignored)
      cell%samples_ns(i) = ticks_to_ns(now_ticks() - t0)
      if (matches /= ref_matches .or. term_bytes /= ref_bytes .or. &
          distance_sum /= ref_distance) call die('nondeterministic result during measurement')
    end do

    cell%warmup_passes = warmup_passes
    cell%samples_requested = samples
    cell%has_construct_ns = .true.
    cell%construct_ns = construct_ns
    cell%matches = ref_matches
    cell%term_bytes = ref_bytes
    cell%distance_sum = ref_distance
    cell%checksum_hex = to_hex16(checksum)
    call write_result(opath, cell)
  end subroutine run_query_cell

end program bench_cross_fortran
