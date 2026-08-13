! Diagnostics for the Fortran cross-language benchmark harness.
! PROTOCOL.md section 1: all diagnostics go to stderr; stdout stays empty.
module xl_diag
  use, intrinsic :: iso_fortran_env, only: error_unit
  implicit none
  private
  public :: die

contains

  ! Print one prefixed diagnostic line and abort with a nonzero exit status.
  subroutine die(message)
    character(len=*), intent(in) :: message
    write (error_unit, '(a)') 'bench-cross-fortran: '//message
    error stop 2
  end subroutine die

end module xl_diag
