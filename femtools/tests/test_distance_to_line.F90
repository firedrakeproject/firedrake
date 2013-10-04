subroutine test_distance_to_line

  use unittest_tools
  use surfacelabels, only: minimum_distance_to_line_segment
  implicit none

  real, dimension(3) :: a, b, c
  real :: correct, computed
  logical :: fail

  a = (/0.0, 0.0, 0.0/)
  b = (/1.0, 0.0, 0.0/)
  c = (/2.0, 0.0, 0.0/)
  correct = 1.0
  computed = minimum_distance_to_line_segment(c, a, b)
  fail = (correct .fne. computed)
  call report_test("[distance_to_line]", fail, .false., "Dumdum")

  c = (/0.5, 0.5, 0.0/)
  correct = 0.5
  computed = minimum_distance_to_line_segment(c, a, b)
  fail = (correct .fne. computed)
  call report_test("[distance_to_line]", fail, .false., "Dumdum")

  c = (/5.0, 1.0, 0.0/)
  correct = sqrt(17.0)
  computed = minimum_distance_to_line_segment(c, a, b)
  fail = (correct .fne. computed)
  call report_test("[distance_to_line]", fail, .false., "Dumdum")

end subroutine test_distance_to_line
