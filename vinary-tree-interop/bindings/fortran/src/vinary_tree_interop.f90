module vinary_tree_interop
  use, intrinsic :: iso_c_binding, only: c_ptr
  implicit none
  private

  integer, parameter, public :: vt_unit_byte = 1
  integer, parameter, public :: vt_unit_unicode_scalar = 2
  integer, parameter, public :: vt_unit_u64 = 3

  type, bind(c), public :: vt_resource
    type(c_ptr) :: context
    type(c_ptr) :: vtable
  end type vt_resource
end module vinary_tree_interop
