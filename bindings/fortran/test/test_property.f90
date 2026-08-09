! C8 native property-based tests for the Fortran facade: a seeded pseudo-PBT
! loop checked against an in-language Levenshtein oracle.
!
!   (a) distance symmetry d(a,b)==d(b,a), identity d(a,a)==0, and threshold
!       consistency dThreshold(a,b,k) == (d<=k ? d : -2) across the
!       Levenshtein/Damerau/true-Damerau variants, plus standard oracle
!       agreement. The native usize::MAX-1 over-bound sentinel reads as -2
!       under the signed c_size_t kind.
!
! The intrinsic PRNG is seeded from a constant so failures reproduce. An ASCII
! alphabet keeps byte == scalar, so the oracle needs no UTF-8 decoding; the
! scalar-vs-byte counting is pinned by the conformance test. Distance operands
! may be empty: the facade marshals an empty string as a null data pointer, and
! as of the LLEV-B18 fix the native distance entry points accept NULL+0 as the
! empty string (the transducer query path always did), so random_term generates
! possibly-empty terms.
program test_property
  use, intrinsic :: iso_c_binding, only: c_size_t
  use vinary_tree_liblevenshtein
  implicit none

  integer, parameter :: max_length = 6
  character(len=4), parameter :: alphabet = "abcd"
  integer(c_size_t), parameter :: sentinel = -2_c_size_t

  integer :: iteration, seed_size, k
  integer, allocatable :: seed(:)
  character(len=:), allocatable :: a, b

  call random_seed(size=seed_size)
  allocate (seed(seed_size))
  seed = 20260809
  call random_seed(put=seed)

  do iteration = 1, 2000
    a = random_term()
    b = random_term()
    k = random_int(0, 3)

    ! Standard Levenshtein: symmetry, identity, oracle agreement, threshold.
    if (levenshtein_distance(a, b) /= levenshtein_distance(b, a)) error stop "levenshtein symmetry"
    if (levenshtein_distance(a, a) /= 0_c_size_t) error stop "levenshtein identity"
    if (levenshtein_distance(a, b) /= oracle(a, b)) error stop "levenshtein oracle"
    call check_threshold(levenshtein_distance(a, b), &
                         levenshtein_distance_threshold(a, b, int(k, c_size_t)), k)

    ! Damerau (OSA): self-consistency.
    if (damerau_distance(a, b) /= damerau_distance(b, a)) error stop "damerau symmetry"
    if (damerau_distance(a, a) /= 0_c_size_t) error stop "damerau identity"
    call check_threshold(damerau_distance(a, b), &
                         damerau_distance_threshold(a, b, int(k, c_size_t)), k)

    ! Unrestricted Damerau-Levenshtein: self-consistency.
    if (true_damerau_distance(a, b) /= true_damerau_distance(b, a)) error stop "true damerau symmetry"
    if (true_damerau_distance(a, a) /= 0_c_size_t) error stop "true damerau identity"
    call check_threshold(true_damerau_distance(a, b), &
                         true_damerau_distance_threshold(a, b, int(k, c_size_t)), k)
  end do

  print *, "Fortran property tests passed"

contains

  subroutine check_threshold(full, bounded, k)
    integer(c_size_t), intent(in) :: full, bounded
    integer, intent(in) :: k
    if (full <= int(k, c_size_t)) then
      if (bounded /= full) error stop "threshold within-bound"
    else
      if (bounded /= sentinel) error stop "threshold sentinel"
    end if
  end subroutine check_threshold

  function random_term() result(term)
    character(len=:), allocatable :: term
    integer :: length, index, choice
    real :: sample
    call random_number(sample)
    length = int(sample * real(max_length + 1))
    if (length > max_length) length = max_length
    allocate (character(len=length) :: term)
    do index = 1, length
      call random_number(sample)
      choice = 1 + int(sample * 4.0)
      if (choice > 4) choice = 4
      term(index:index) = alphabet(choice:choice)
    end do
  end function random_term

  function random_int(low, high) result(value)
    integer, intent(in) :: low, high
    integer :: value
    real :: sample
    call random_number(sample)
    value = low + int(sample * real(high - low + 1))
    if (value > high) value = high
  end function random_int

  function oracle(left, right) result(distance)
    character(len=*), intent(in) :: left, right
    integer(c_size_t) :: distance
    integer :: la, lb, i, j, cost
    integer, allocatable :: previous(:), current(:)
    la = len(left)
    lb = len(right)
    if (la == 0) then
      distance = int(lb, c_size_t)
      return
    end if
    if (lb == 0) then
      distance = int(la, c_size_t)
      return
    end if
    allocate (previous(0:lb), current(0:lb))
    do j = 0, lb
      previous(j) = j
    end do
    do i = 1, la
      current(0) = i
      do j = 1, lb
        cost = merge(0, 1, left(i:i) == right(j:j))
        current(j) = min(previous(j) + 1, current(j - 1) + 1, previous(j - 1) + cost)
      end do
      previous = current
    end do
    distance = int(previous(lb), c_size_t)
  end function oracle

end program test_property
