subroutine test_integer_set

  use data_structures
  use unittest_tools
  implicit none

  type(integer_set) :: iset
  integer :: len, i
  logical :: fail, changed

  call allocate(iset)
  call insert(iset, 4)
  call insert(iset, 5)
  call insert(iset, 6)

  len = key_count(iset)
  fail = (len /= 3)
  call report_test("[key_count]", fail, .false., "Should be 3")

  do i=1,len
    fail = (fetch(iset, i) /= i+3)
    call report_test("[fetch]", fail, .false., "Should give i+3")
  end do

  fail = has_value(iset, 99)
  call report_test("[integer_set_has_value]", fail, .false., "Should be .false.!")

  fail = .not. has_value(iset, 5)
  call report_test("[integer_set_has_value]", fail, .false., "Should be .true.!")

  call insert(iset, 4, changed=changed)
  fail = changed
  call report_test("[integer_set_insert]", fail, .false., "Should not change")

  len = key_count(iset)
  fail = (len /= 3)
  call report_test("[key_count]", fail, .false., "Should be 3")

  call remove(iset, 4)
  len = key_count(iset)
  fail = (len /= 2) .or. has_value(iset, 4)
  call report_test("[key_count]", fail, .false., "Should change")

  call deallocate(iset)
end subroutine test_integer_set
