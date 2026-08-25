! Result JSON emission for the Fortran cross-language benchmark harness
! (PROTOCOL.md section 11, harness-filled fields only; the runner post-fills
! run_id, sha256 digests, environment_ref, cell_snapshot, and memory).
! Field order mirrors the C harness; plain formatted writes, two-space
! indent, LF records.
module xl_json
  use, intrinsic :: iso_fortran_env, only: int64, real64
  use xl_diag, only: die
  implicit none
  private
  public :: cell_output, write_result, note_width, max_notes, add_note

  integer, parameter :: note_width = 256
  integer, parameter :: max_notes = 8
  character(len=*), parameter :: library_version = '0.10.0'
  character(len=*), parameter :: sample_definition = &
    'one full pass over the query set; every cursor fully drained and '// &
    '(term, distance) materialized'

  type :: cell_output
    character(len=:), allocatable :: mode
    character(len=:), allocatable :: structure
    character(len=:), allocatable :: algorithm
    character(len=:), allocatable :: dictionary_file
    character(len=:), allocatable :: queries_file
    character(len=:), allocatable :: status
    character(len=:), allocatable :: runtime_version
    integer :: max_distance = 1
    integer :: term_count = 0
    logical :: has_construct_ns = .false.
    integer(int64) :: construct_ns = 0_int64
    integer :: query_count = 0
    integer :: warmup_passes = 0
    integer :: samples_requested = 0
    real(real64) :: warmup_seconds = 3.0_real64
    integer(int64), allocatable :: samples_ns(:)     ! measurements block
    integer(int64) :: matches = 0_int64
    integer(int64) :: term_bytes = 0_int64
    integer(int64) :: distance_sum = 0_int64
    character(len=16) :: checksum_hex = '0000000000000000'
    integer(int64), allocatable :: construct_times(:) ! construct block instead
    character(len=note_width) :: notes(max_notes) = ''
    integer :: note_count = 0
  end type cell_output

contains

  subroutine add_note(cell, note)
    type(cell_output), intent(inout) :: cell
    character(len=*), intent(in) :: note
    if (cell%note_count >= max_notes) call die('too many result notes')
    cell%note_count = cell%note_count + 1
    cell%notes(cell%note_count) = note
  end subroutine add_note

  subroutine write_result(out_path, cell)
    character(len=*), intent(in) :: out_path
    type(cell_output), intent(in) :: cell
    integer :: unit, io, i
    character(len=:), allocatable :: mode_label

    open (newunit=unit, file=out_path, action='write', status='replace', &
          form='formatted', iostat=io)
    if (io /= 0) call die('cannot write '//out_path)

    if (cell%mode == 'memory-child') then
      mode_label = 'memory'
    else
      mode_label = cell%mode
    end if

    write (unit, '(a)') '{'
    write (unit, '(a)') '  "schema_version": "1.0.0",'
    write (unit, '(a)') '  "suite": "cross-language-v1",'
    write (unit, '(a)') '  "timestamp_utc": "'//utc_timestamp()//'",'
    write (unit, '(a)') '  "target": {'
    write (unit, '(a)') '    "language": "fortran",'
    write (unit, '(a)') '    "implementation": "vinary-tree",'
    write (unit, '(a)') '    "backend": "iso_c_binding",'
    write (unit, '(a)') '    "runtime_version": "'//json_escape(cell%runtime_version)//'",'
    write (unit, '(a)') '    "library_version": "'//library_version//'",'
    write (unit, '(a)') '    "artifact": { "kind": "local-build", '// &
      '"id": "liblevenshtein-fpm@'//library_version//'" }'
    write (unit, '(a)') '  },'
    write (unit, '(a)') '  "dictionary": {'
    write (unit, '(a)') '    "file": "'//json_escape(cell%dictionary_file)//'",'
    write (unit, '(a)') '    "term_count": '//itoa(int(cell%term_count, int64))//','
    write (unit, '(a)') '    "structure": "'//cell%structure//'",'
    if (cell%has_construct_ns) then
      write (unit, '(a)') '    "unit_domain": "unicode_scalar",'
      write (unit, '(a)') '    "construct_ns": '//itoa(cell%construct_ns)
    else
      write (unit, '(a)') '    "unit_domain": "unicode_scalar"'
    end if
    write (unit, '(a)') '  },'
    write (unit, '(a)') '  "workload": {'
    write (unit, '(a)') '    "queryset": "'//queryset_stem(cell%queries_file)//'",'
    write (unit, '(a)') '    "file": "'//json_escape(cell%queries_file)//'",'
    write (unit, '(a)') '    "query_count": '//itoa(int(cell%query_count, int64))
    write (unit, '(a)') '  },'
    write (unit, '(a)') '  "algorithm": "'//cell%algorithm//'",'
    write (unit, '(a)') '  "max_distance": '//itoa(int(cell%max_distance, int64))//','
    write (unit, '(a)') '  "mode": "'//mode_label//'",'
    write (unit, '(a)') '  "protocol": {'
    write (unit, '(a)') '    "timer": "monotonic",'
    write (unit, '(a)') '    "harness": "self-timed",'
    write (unit, '(a)') '    "warmup_seconds_min": '//rtoa(cell%warmup_seconds)//','
    write (unit, '(a)') '    "warmup_passes": '//itoa(int(cell%warmup_passes, int64))//','
    write (unit, '(a)') '    "samples_requested": '//itoa(int(cell%samples_requested, int64))//','
    write (unit, '(a)') '    "sample_definition": "'//sample_definition//'",'
    write (unit, '(a)') '    "batch_size": 256,'
    write (unit, '(a)') '    "wall_cap_seconds": 300'
    write (unit, '(a)') '  },'
    if (allocated(cell%construct_times)) then
      write (unit, '(a)') '  "construct": {'
      write (unit, '(a)') '    "reps": '//itoa(int(size(cell%construct_times), int64))//','
      write (unit, '(a)', advance='no') '    "times_ns": ['
      do i = 1, size(cell%construct_times)
        if (i > 1) write (unit, '(a)', advance='no') ', '
        write (unit, '(a)', advance='no') itoa(cell%construct_times(i))
      end do
      write (unit, '(a)') '],'
      write (unit, '(a)') '    "term_count": '//itoa(int(cell%term_count, int64))
      write (unit, '(a)') '  },'
    else
      write (unit, '(a)') '  "measurements": {'
      write (unit, '(a)', advance='no') '    "samples_ns": ['
      if (allocated(cell%samples_ns)) then
        do i = 1, size(cell%samples_ns)
          if (i > 1) write (unit, '(a)', advance='no') ', '
          write (unit, '(a)', advance='no') itoa(cell%samples_ns(i))
        end do
      end if
      write (unit, '(a)') '],'
      if (allocated(cell%samples_ns)) then
        write (unit, '(a)') '    "sample_count": '//itoa(int(size(cell%samples_ns), int64))//','
      else
        write (unit, '(a)') '    "sample_count": 0,'
      end if
      write (unit, '(a)') '    "matches_per_pass": '//itoa(cell%matches)//','
      write (unit, '(a)') '    "term_bytes_per_pass": '//itoa(cell%term_bytes)//','
      write (unit, '(a)') '    "distance_sum_per_pass": '//itoa(cell%distance_sum)//','
      write (unit, '(a)') '    "checksum_hex": "'//cell%checksum_hex//'"'
      write (unit, '(a)') '  },'
    end if
    write (unit, '(a)') '  "status": "'//cell%status//'",'
    write (unit, '(a)', advance='no') '  "notes": ['
    do i = 1, cell%note_count
      if (i > 1) write (unit, '(a)', advance='no') ', '
      write (unit, '(a)', advance='no') '"'//json_escape(trim(cell%notes(i)))//'"'
    end do
    write (unit, '(a)') ']'
    write (unit, '(a)') '}'
    close (unit)
  end subroutine write_result

  ! ------------------------------------------------------------------
  ! Formatting helpers
  ! ------------------------------------------------------------------

  function itoa(value) result(text)
    integer(int64), intent(in) :: value
    character(len=:), allocatable :: text
    character(len=24) :: buffer
    write (buffer, '(i0)') value
    text = trim(buffer)
  end function itoa

  ! Compact JSON number for the warmup-seconds value: integral values print
  ! as integers (mirroring C's %g), fractional values keep three decimals.
  function rtoa(value) result(text)
    real(real64), intent(in) :: value
    character(len=:), allocatable :: text
    character(len=32) :: buffer
    if (abs(value - nint(value, int64)) < 1.0e-9_real64) then
      write (buffer, '(i0)') nint(value, int64)
    else
      write (buffer, '(f0.3)') value
      if (buffer(1:1) == '.') buffer = '0'//buffer  ! f0.x drops the leading 0
    end if
    text = trim(buffer)
  end function rtoa

  function json_escape(text) result(escaped)
    character(len=*), intent(in) :: text
    character(len=:), allocatable :: escaped
    character(len=6) :: unicode_escape
    integer :: i, code
    escaped = ''
    do i = 1, len(text)
      code = iachar(text(i:i))
      select case (text(i:i))
      case ('"')
        escaped = escaped//'\"'
      case ('\')
        escaped = escaped//'\\'
      case default
        if (code == 10) then
          escaped = escaped//'\n'
        else if (code == 13) then
          escaped = escaped//'\r'
        else if (code == 9) then
          escaped = escaped//'\t'
        else if (code < 32) then
          write (unicode_escape, '(a2,z4.4)') '\u', code
          escaped = escaped//lowercase_hex(unicode_escape)
        else
          escaped = escaped//text(i:i)
        end if
      end select
    end do
  end function json_escape

  pure function lowercase_hex(text) result(lowered)
    character(len=*), intent(in) :: text
    character(len=len(text)) :: lowered
    integer :: i, code
    do i = 1, len(text)
      code = iachar(text(i:i))
      if (code >= iachar('A') .and. code <= iachar('F')) then
        lowered(i:i) = achar(code + 32)
      else
        lowered(i:i) = text(i:i)
      end if
    end do
  end function lowercase_hex

  ! Basename of the queries path without its extension (the schema queryset).
  function queryset_stem(path) result(stem)
    character(len=*), intent(in) :: path
    character(len=:), allocatable :: stem
    integer :: slash, dot
    slash = index(path, '/', back=.true.)
    dot = index(path, '.', back=.true.)
    if (dot <= slash) dot = len(path) + 1
    stem = path(slash + 1:dot - 1)
  end function queryset_stem

  ! ------------------------------------------------------------------
  ! UTC timestamp: date_and_time gives local time plus the UTC offset in
  ! minutes; convert through a day serial (Howard Hinnant's civil-days
  ! algorithm) so the subtraction rolls dates over correctly.
  ! ------------------------------------------------------------------

  function utc_timestamp() result(stamp)
    character(len=20) :: stamp
    integer :: values(8), year, month, day, hour, minute, second
    integer(int64) :: days, seconds
    call date_and_time(values=values)
    days = days_from_civil(values(1), values(2), values(3))
    seconds = days * 86400_int64 + values(5) * 3600_int64 &
              + values(6) * 60_int64 + values(7) - values(4) * 60_int64
    days = seconds / 86400_int64
    if (mod(seconds, 86400_int64) < 0_int64) days = days - 1_int64  ! floor
    call civil_from_days(days, year, month, day)
    seconds = seconds - days * 86400_int64
    hour = int(seconds / 3600_int64)
    minute = int(mod(seconds, 3600_int64) / 60_int64)
    second = int(mod(seconds, 60_int64))
    write (stamp, '(i4.4,"-",i2.2,"-",i2.2,"T",i2.2,":",i2.2,":",i2.2,"Z")') &
      year, month, day, hour, minute, second
  end function utc_timestamp

  pure function days_from_civil(year_in, month, day) result(days)
    integer, intent(in) :: year_in, month, day
    integer(int64) :: days
    integer(int64) :: year, era, year_of_era, day_of_year, day_of_era
    year = int(year_in, int64)
    if (month <= 2) year = year - 1_int64
    era = year / 400_int64                       ! contemporary dates: year > 0
    year_of_era = year - era * 400_int64
    if (month > 2) then
      day_of_year = (153_int64 * int(month - 3, int64) + 2_int64) / 5_int64 &
                    + int(day - 1, int64)
    else
      day_of_year = (153_int64 * int(month + 9, int64) + 2_int64) / 5_int64 &
                    + int(day - 1, int64)
    end if
    day_of_era = year_of_era * 365_int64 + year_of_era / 4_int64 &
                 - year_of_era / 100_int64 + day_of_year
    days = era * 146097_int64 + day_of_era - 719468_int64
  end function days_from_civil

  pure subroutine civil_from_days(days_in, year, month, day)
    integer(int64), intent(in) :: days_in
    integer, intent(out) :: year, month, day
    integer(int64) :: z, era, day_of_era, year_of_era, day_of_year, month_prime
    z = days_in + 719468_int64
    era = z / 146097_int64
    day_of_era = z - era * 146097_int64
    year_of_era = (day_of_era - day_of_era / 1460_int64 + day_of_era / 36524_int64 &
                   - day_of_era / 146096_int64) / 365_int64
    year = int(year_of_era + era * 400_int64)
    day_of_year = day_of_era - (365_int64 * year_of_era + year_of_era / 4_int64 &
                                - year_of_era / 100_int64)
    month_prime = (5_int64 * day_of_year + 2_int64) / 153_int64
    day = int(day_of_year - (153_int64 * month_prime + 2_int64) / 5_int64 + 1_int64)
    if (month_prime < 10_int64) then
      month = int(month_prime + 3_int64)
    else
      month = int(month_prime - 9_int64)
    end if
    if (month <= 2) year = year + 1
  end subroutine civil_from_days

end module xl_json
