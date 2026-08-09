program test_distance
  use, intrinsic :: iso_c_binding, only: c_size_t, c_int32_t
  use vinary_tree_liblevenshtein
  implicit none
  type(phonetic_pattern) :: pattern
  integer(c_int32_t) :: status = -1
  logical :: accepted
  if (levenshtein_distance("kitten", "sitting") /= 3_c_size_t) error stop "distance"
  if (damerau_distance("ab", "ba") /= 1_c_size_t) error stop "Damerau"
  if (true_damerau_distance("ca", "abc") /= 2_c_size_t) error stop "true Damerau"
  ! C6: distances count Unicode scalars, not the UTF-8 bytes the facade passes.
  if (levenshtein_distance("café", "cafe") /= 1_c_size_t) error stop "unicode distance"
  if (levenshtein_distance("🦀", "x") /= 1_c_size_t) error stop "astral distance"
  ! C6: the exceeded-bound native sentinel (usize::MAX - 1) reads as -2 under the
  ! signed c_size_t kind; "ca" -> "abc" separates OSA 3 from unrestricted 2.
  if (levenshtein_distance_threshold("kitten", "sitting", 3_c_size_t) /= 3_c_size_t) error stop "threshold within"
  if (levenshtein_distance_threshold("kitten", "sitting", 2_c_size_t) /= -2_c_size_t) error stop "threshold sentinel"
  if (damerau_distance_threshold("ca", "abc", 2_c_size_t) /= -2_c_size_t) error stop "OSA threshold sentinel"
  if (true_damerau_distance_threshold("ca", "abc", 2_c_size_t) /= 2_c_size_t) error stop "true Damerau threshold"
  call compile_phonetic_regex(pattern, "cat", status)
  if (status /= llev_ok) error stop "pattern compilation"
  accepted = pattern%matches("cat", status)
  if (status /= llev_ok .or. .not. accepted) error stop "pattern"
  accepted = pattern%matches("cot", status)
  if (status /= llev_ok .or. accepted) error stop "pattern rejection"
  call pattern%close()
  print *, "Fortran binding conformance passed"
end program test_distance
