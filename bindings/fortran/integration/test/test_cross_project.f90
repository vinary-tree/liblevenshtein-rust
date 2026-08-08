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
  write(path, '(a,i0)') '/tmp/vinary-tree-fortran-', count
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
