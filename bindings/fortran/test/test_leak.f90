! C9 leak-discipline suite for the Fortran facade. Fortran has no garbage
! collector, so this exercises the create/use/free discipline directly: a
! >=10,000-cycle loop compiles a phonetic pattern, matches against it, and
! closes it each cycle. Resident memory is read from /proc/self/statm before and
! after the measured loop (after a warmup so the allocator has reached a working
! set); a per-cycle native handle leak would push resident pages up without
! bound, whereas correct release keeps them flat within a generous ceiling.
program test_leak
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_long
  use vinary_tree_liblevenshtein
  implicit none

  integer, parameter :: cycles = 10000
  integer, parameter :: warmup = 2000
  ! 4096 pages (~16 MiB) is generous against allocator noise; a real per-cycle
  ! leak would accrue far more over 10k cycles.
  integer(c_long), parameter :: max_growth_pages = 4096_c_long

  integer :: iteration
  integer(c_long) :: baseline, growth

  do iteration = 1, warmup
    call phonetic_cycle()
  end do
  baseline = resident_pages()
  do iteration = 1, cycles
    call phonetic_cycle()
  end do
  growth = resident_pages() - baseline
  if (growth > max_growth_pages) error stop "phonetic cycle leaked resident memory"

  print *, "Fortran leak loop completed cycles"

contains

  subroutine phonetic_cycle()
    type(phonetic_pattern) :: pattern
    integer(c_int32_t) :: status
    logical :: accepted
    call compile_phonetic_regex(pattern, "c[ao]t", status)
    if (status /= llev_ok) error stop "pattern compilation"
    accepted = pattern%matches("cat", status)
    if (status /= llev_ok .or. .not. accepted) error stop "pattern match"
    call pattern%close()
  end subroutine phonetic_cycle

  function resident_pages() result(pages)
    integer(c_long) :: pages
    integer :: unit, ios
    integer(c_long) :: total
    pages = -1_c_long
    open (newunit=unit, file="/proc/self/statm", status="old", action="read", iostat=ios)
    if (ios /= 0) return
    read (unit, *, iostat=ios) total, pages
    close (unit)
  end function resident_pages

end program test_leak
