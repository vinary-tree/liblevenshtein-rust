! Checksum primitives for the Fortran cross-language benchmark harness
! (PROTOCOL.md section 8, normative bit-level spec).
!
! All arithmetic is unsigned 64-bit mod 2^64 carried in integer(int64) bit
! patterns. Fortran's signed multiply/add overflow is formally undefined
! (gfortran wraps without -ftrapv), so the wrapping multiply and add are
! decomposed into 32-bit halves: every intermediate stays below 2^42 and the
! recombination uses only ieor/iand/ior/ishft, which the standard defines on
! the bit pattern. ISHFT is a logical (zero-fill) shift in both directions.
module xl_checksum
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: fnv_offset, fnv_update, fnv1a64, entry_hash, wrap_add64, to_hex16
  public :: checksum_self_test

  integer(int64), parameter :: mask32 = int(z'FFFFFFFF', int64)
  ! FNV-1a 64-bit offset basis 0xcbf29ce484222325, assembled from 32-bit
  ! halves (the full pattern exceeds huge(0_int64) as a literal).
  integer(int64), parameter :: fnv_offset = &
    ior(ishft(int(z'CBF29CE4', int64), 32), int(z'84222325', int64))
  ! FNV prime 0x00000100000001b3 = 2^40 + 0x1b3, split at bit 32.
  integer(int64), parameter :: prime_hi = int(z'100', int64)
  integer(int64), parameter :: prime_lo = int(z'1B3', int64)

contains

  ! (a + b) mod 2^64 without signed overflow: add the 32-bit halves and
  ! carry, then recombine bit patterns.
  pure function wrap_add64(a, b) result(total)
    integer(int64), intent(in) :: a, b
    integer(int64) :: total
    integer(int64) :: low, high
    low = iand(a, mask32) + iand(b, mask32)                    ! <= 2^33 - 2
    high = ishft(a, -32) + ishft(b, -32) + ishft(low, -32)     ! <= 2^33
    total = ior(ishft(iand(high, mask32), 32), iand(low, mask32))
  end function wrap_add64

  ! (h * FNV_PRIME) mod 2^64 via half decomposition. With h = hi*2^32 + lo
  ! and PRIME = 0x100*2^32 + 0x1b3, the hi*hi term lands at 2^64 and drops
  ! out; the cross terms stay below 2^42 so no signed overflow can occur.
  pure function wrap_mul_prime(h) result(product)
    integer(int64), intent(in) :: h
    integer(int64) :: product
    integer(int64) :: h_lo, h_hi, low_part, cross
    h_lo = iand(h, mask32)
    h_hi = ishft(h, -32)
    low_part = h_lo * prime_lo                       ! < 2^41
    cross = h_lo * prime_hi + h_hi * prime_lo        ! < 2^42
    product = wrap_add64(low_part, ishft(cross, 32)) ! ishft drops bits >= 2^32 of cross: mod 2^64
  end function wrap_mul_prime

  ! One FNV-1a step: XOR first, then the wrapping prime multiply.
  pure function fnv_update(h, byte) result(updated)
    integer(int64), intent(in) :: h, byte
    integer(int64) :: updated
    updated = wrap_mul_prime(ieor(h, byte))
  end function fnv_update

  ! FNV-1a 64 over the bytes of a character string.
  pure function fnv1a64(text) result(h)
    character(len=*), intent(in) :: text
    integer(int64) :: h
    integer :: i
    h = fnv_offset
    do i = 1, len(text)
      h = fnv_update(h, iand(int(iachar(text(i:i)), int64), 255_int64))
    end do
  end function fnv1a64

  ! entry(term, distance) = FNV-1a64 over utf8(term) || 0x00 || LE64(distance).
  pure function entry_hash(term, distance) result(h)
    character(len=*), intent(in) :: term
    integer(int64), intent(in) :: distance
    integer(int64) :: h
    integer :: i
    h = fnv1a64(term)
    h = fnv_update(h, 0_int64)                                  ! separator
    do i = 0, 7                                                 ! LE64(distance)
      h = fnv_update(h, iand(ishft(distance, -8 * i), 255_int64))
    end do
  end function entry_hash

  ! 16 lowercase, zero-padded hex digits of the 64-bit pattern.
  pure function to_hex16(value) result(hex)
    integer(int64), intent(in) :: value
    character(len=16) :: hex
    character(len=16), parameter :: digits = '0123456789abcdef'
    integer :: i, nibble
    do i = 1, 16
      nibble = int(iand(ishft(value, -4 * (16 - i)), 15_int64))
      hex(i:i) = digits(nibble + 1:nibble + 1)
    end do
  end function to_hex16

  ! Startup self-test (PROTOCOL.md section 2): the seven confirmed vectors.
  ! Expected values are assembled from 32-bit halves like fnv_offset.
  subroutine checksum_self_test()
    use xl_diag, only: die
    if (fnv1a64('') /= u64(int(z'CBF29CE4', int64), int(z'84222325', int64))) &
      call die('checksum self-test failed: fnv1a64("")')
    if (fnv1a64('a') /= u64(int(z'AF63DC4C', int64), int(z'8601EC8C', int64))) &
      call die('checksum self-test failed: fnv1a64("a")')
    if (entry_hash('cat', 1_int64) /= u64(int(z'9697FA3E', int64), int(z'50464BC4', int64))) &
      call die('checksum self-test failed: entry(cat,1)')
    if (entry_hash('cat', 0_int64) /= u64(int(z'B592C147', int64), int(z'5B3595E5', int64))) &
      call die('checksum self-test failed: entry(cat,0)')
    if (entry_hash('cot', 1_int64) /= u64(int(z'B8ACC5D3', int64), int(z'816BCDEA', int64))) &
      call die('checksum self-test failed: entry(cot,1)')
    if (wrap_add64(entry_hash('cat', 0_int64), entry_hash('cot', 1_int64)) /= &
        u64(int(z'6E3F871A', int64), int(z'DCA163CF', int64))) &
      call die('checksum self-test failed: checksum{(cat,0),(cot,1)}')
    if (0_int64 /= u64(0_int64, 0_int64)) &
      call die('checksum self-test failed: checksum{}')
  end subroutine checksum_self_test

  pure function u64(hi, lo) result(value)
    integer(int64), intent(in) :: hi, lo
    integer(int64) :: value
    value = ior(ishft(hi, 32), lo)
  end function u64

end module xl_checksum
