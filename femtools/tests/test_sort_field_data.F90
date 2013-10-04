subroutine test_sort_field_data

  use quicksort
  use unittest_tools

  implicit none

  integer, dimension(5) :: permutation
  real, dimension(5, 3) :: field_data

  field_data(1, :) = (/0.0, 0.0, 0.0/)
  field_data(2, :) = (/0.0, 2.0, 0.0/)
  field_data(3, :) = (/0.0, 1.0, 0.0/)
  field_data(4, :) = (/0.0, 1.0, 3.0/)
  field_data(5, :) = (/0.0, 1.0, 2.0/)

  call sort(field_data, permutation)
  call report_test("[sort_field_data]", any(permutation /= (/1, 3, 5, 4, 2/)), .false., "Invalid permutation")

end subroutine test_sort_field_data
