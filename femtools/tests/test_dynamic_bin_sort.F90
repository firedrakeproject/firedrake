subroutine test_dynamic_bin_sort()
! tests the dynamic bin sort algorithm
! (a bin sort where entries may jump bin during the sort)
use dynamic_bin_sort_module
use unittest_tools
implicit none

  type(dynamic_bin_type) dbin
  integer, parameter:: NBINS=10, NELEMENTS=10000
  integer, dimension(1:NELEMENTS):: binlist, sorted_list
  logical sorted
  real rnd
  integer i, elm, bin_no

  do i=1, NELEMENTS
    call random_number(rnd)
    binlist(i)=floor(rnd*NBINS)+1
  end do

  call allocate(dbin, binlist)

  do i=1, NELEMENTS
    call pull_element(dbin, sorted_list(i), bin_no)

    ! select random element:
    call random_number(rnd)
    elm=floor(rnd*NELEMENTS)+1

    if (.not. element_pulled(dbin, elm)) then
       ! element not pulled yet
       ! move to somewhere >=bin_no
       call random_number(rnd)
       call move_element(dbin, elm, floor(rnd*(NBINS+1-bin_no))+bin_no)

    end if

  end do

  ! check if the resulting list is sorted:
  sorted=.true.
  do i=1, NELEMENTS-1
    sorted=sorted .and. (binlist(sorted_list(i))<=binlist(sorted_list(i+1)))
  end do

  call deallocate(dbin)

  call report_test("[test_dynamic_bin_sort]", .not. sorted, .false., "resulting list not sorted")

end subroutine test_dynamic_bin_sort

