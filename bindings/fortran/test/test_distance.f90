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
  call compile_phonetic_regex(pattern, "cat", status)
  if (status /= llev_ok) error stop "pattern compilation"
  accepted = pattern%matches("cat", status)
  if (status /= llev_ok .or. .not. accepted) error stop "pattern"
  accepted = pattern%matches("cot", status)
  if (status /= llev_ok .or. accepted) error stop "pattern rejection"
  call pattern%close()
  print *, "Fortran binding conformance passed"
end program test_distance
