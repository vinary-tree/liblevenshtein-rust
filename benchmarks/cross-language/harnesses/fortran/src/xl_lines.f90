! Input loading for the Fortran cross-language benchmark harness
! (PROTOCOL.md section 3): whole-file stream read, newline split with the
! single trailing empty line dropped, strict byte-ascending assertion.
module xl_lines
  use, intrinsic :: iso_fortran_env, only: int64
  use xl_diag, only: die
  implicit none
  private
  public :: line_list, read_lines, assert_strictly_sorted

  character(len=1), parameter :: newline_char = achar(10)

  ! Fixed-width line storage: item i is text(i)(1:length(i)); width is the
  ! longest line. The workload is lowercase ASCII, so the blank padding is
  ! inert: the libdictenstein facade's put_all recovers true lengths via
  ! len_trim, and every other consumer slices with length(i).
  type :: line_list
    character(len=:), allocatable :: text(:)
    integer, allocatable :: length(:)
    integer :: count = 0
  end type line_list

contains

  subroutine read_lines(path, list)
    character(len=*), intent(in) :: path
    type(line_list), intent(out) :: list
    character(len=:), allocatable :: raw
    integer(int64) :: file_size
    integer :: unit, io, count, width, index
    logical :: exists

    inquire (file=path, exist=exists, size=file_size)
    if (.not. exists .or. file_size < 0) call die('cannot open '//path)
    open (newunit=unit, file=path, access='stream', form='unformatted', &
          action='read', status='old', iostat=io)
    if (io /= 0) call die('cannot open '//path)
    allocate (character(len=int(file_size)) :: raw)
    if (file_size > 0) then
      read (unit, iostat=io) raw
      if (io /= 0) call die('short read on '//path)
    end if
    close (unit)

    ! Pass 1: count non-empty lines and the maximum width, so both arrays
    ! are allocated exactly once (PROTOCOL.md section 3.4 preallocation).
    call scan_lines(raw, count, width)
    if (count == 0) call die(path//' contains no lines')
    allocate (character(len=width) :: list%text(count))
    allocate (list%length(count))
    list%count = count

    ! Pass 2: materialize the (blank-padded) lines and their byte lengths.
    call split_lines(raw, list, index)
    if (index /= count) call die('internal line-count mismatch reading '//path)
  end subroutine read_lines

  subroutine scan_lines(raw, count, width)
    character(len=*), intent(in) :: raw
    integer, intent(out) :: count, width
    integer :: position, start, line_len
    count = 0
    width = 0
    start = 1
    do position = 1, len(raw) + 1
      if (position <= len(raw)) then
        if (raw(position:position) /= newline_char) cycle
      end if
      line_len = position - start
      if (line_len > 0) then
        count = count + 1
        if (line_len > width) width = line_len
      end if
      start = position + 1
    end do
  end subroutine scan_lines

  subroutine split_lines(raw, list, index)
    character(len=*), intent(in) :: raw
    type(line_list), intent(inout) :: list
    integer, intent(out) :: index
    integer :: position, start, line_len
    index = 0
    start = 1
    do position = 1, len(raw) + 1
      if (position <= len(raw)) then
        if (raw(position:position) /= newline_char) cycle
      end if
      line_len = position - start
      if (line_len > 0) then
        index = index + 1
        list%text(index) = raw(start:position - 1)  ! assignment blank-pads
        list%length(index) = line_len
      end if
      start = position + 1
    end do
  end subroutine split_lines

  ! PROTOCOL.md section 3.2: strict byte-ascending order, asserted
  ! everywhere. LLT is the ASCII collating order, and blank padding sorts
  ! below every printable workload byte, so prefix < extension holds —
  ! bytewise semantics for this all-ASCII workload.
  subroutine assert_strictly_sorted(list, path)
    type(line_list), intent(in) :: list
    character(len=*), intent(in) :: path
    integer :: i
    character(len=32) :: line_number
    do i = 1, list%count - 1
      if (.not. llt(list%text(i)(1:list%length(i)), &
                    list%text(i + 1)(1:list%length(i + 1)))) then
        write (line_number, '(i0)') i
        call die(path//' is not strictly byte-sorted at line '//trim(line_number)// &
                 ': "'//list%text(i)(1:list%length(i))//'" >= "'// &
                 list%text(i + 1)(1:list%length(i + 1))//'"')
      end if
    end do
  end subroutine assert_strictly_sorted

end module xl_lines
