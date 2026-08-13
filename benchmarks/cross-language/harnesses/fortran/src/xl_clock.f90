! Monotonic clock for the Fortran cross-language benchmark harness.
! PROTOCOL.md section 9 pins Fortran to system_clock with int64
! count/count_rate. Raw ticks are kept for interval arithmetic and reduced
! to nanoseconds only at the edges (exactly when the tick rate is already
! 1 GHz, as gfortran reports for int64 arguments on Linux).
module xl_clock
  use, intrinsic :: iso_fortran_env, only: int64, real64
  use xl_diag, only: die
  implicit none
  private
  public :: clock_init, now_ticks, ticks_to_ns, seconds_to_ticks

  integer(int64) :: tick_rate = 0_int64

contains

  subroutine clock_init()
    integer(int64) :: count
    call system_clock(count=count, count_rate=tick_rate)
    if (tick_rate <= 0_int64) call die('system_clock reports no usable int64 counter')
  end subroutine clock_init

  function now_ticks() result(ticks)
    integer(int64) :: ticks
    call system_clock(count=ticks)
  end function now_ticks

  ! Convert a tick INTERVAL to nanoseconds. Interval magnitudes here are
  ! bounded by the 300 s wall cap, far inside real64's 2^53 exact-integer
  ! range, so the non-1GHz fallback loses nothing measurable.
  function ticks_to_ns(delta) result(ns)
    integer(int64), intent(in) :: delta
    integer(int64) :: ns
    if (tick_rate == 1000000000_int64) then
      ns = delta
    else
      ns = int(real(delta, real64) * (1.0e9_real64 / real(tick_rate, real64)), int64)
    end if
  end function ticks_to_ns

  function seconds_to_ticks(seconds) result(ticks)
    real(real64), intent(in) :: seconds
    integer(int64) :: ticks
    ticks = int(seconds * real(tick_rate, real64), int64)
  end function seconds_to_ticks

end module xl_clock
