program test_distance
  use, intrinsic :: iso_c_binding, only: c_size_t, c_int32_t
  use vinary_tree_liblevenshtein
  implicit none
  type(phonetic_pattern) :: pattern
  integer(c_int32_t) :: status = -1
  logical :: accepted
  if (llev_ok /= 0 .or. llev_end /= 1) error stop "terminal statuses"
  if (llev_invalid_argument /= 2 .or. llev_invalid_utf8 /= 3) error stop "input statuses"
  if (llev_null_pointer /= 4 .or. llev_panic /= 5) error stop "boundary statuses"
  if (llev_unsupported /= 6 .or. llev_io_error /= 7) error stop "capability statuses"
  if (llev_closed /= 8 .or. llev_limit_exceeded /= 9) error stop "lifecycle statuses"
  if (llev_provider_error /= 10 .or. llev_batch_in_use /= 11) error stop "provider statuses"
  if (llev_domain_mismatch /= 12) error stop "domain status"
  if (llev_english_orthography /= 0 .or. llev_english_phonetic /= 1) error stop "rule-set kinds"
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
