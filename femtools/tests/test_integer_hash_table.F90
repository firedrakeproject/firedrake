subroutine test_integer_hash_table

  use data_structures
  use unittest_tools
  implicit none

  type(integer_hash_table) :: ihash
  integer :: len, i
  logical :: fail

  call allocate(ihash)
  call insert(ihash, 4, 40)
  call insert(ihash, 5, 50)
  call insert(ihash, 6, 60)

  len = key_count(ihash)
  fail = (len /= 3)
  call report_test("[key_count]", fail, .false., "Should be 3")

  do i=4,6
    fail = (fetch(ihash, i) /= i*10)
    call report_test("[fetch]", fail, .false., "Should give i*10")
  end do

  fail = has_key(ihash, 99)
  call report_test("[integer_hash_table_has_value]", fail, .false., "Should be .false.!")

  fail = .not. has_key(ihash, 5)
  call report_test("[integer_hash_table_has_value]", fail, .false., "Should be .true.!")

  call remove(ihash, 5)
  fail = has_key(ihash, 5)
  call report_test("[integer_hash_table_has_value]", fail, .false., "Should be .false.!")

  call deallocate(ihash)
end subroutine test_integer_hash_table
