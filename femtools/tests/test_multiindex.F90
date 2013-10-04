subroutine test_multiindex

  use unittest_tools
  use futils
  implicit none

  integer, dimension(3) :: result
  integer :: charcount
  logical :: fail

  charcount = count_chars("This is a string", " ")
  if (charcount /= 3) then
    fail = .true.
  else
    fail = .false.
  end if

  call report_test("[count_chars]", fail, .false., &
  & "Give the right answer, please")

  result = multiindex("This is a string", " ")
  if (any(result /= (/5, 8, 10/))) then
    write(0,*) "result == ", result
    fail = .true.
  else
    fail = .false.
  end if

  call report_test("[multi-index]", fail, .false., &
  & "Multiindex should give the right answer.")

end subroutine test_multiindex
