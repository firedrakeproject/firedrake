module element_set

  implicit none

  external :: ele_add_to_set, ele_get_size, ele_fetch_list

  contains

  subroutine eleset_add(i)
    !!< Add i to the set of elements to be considered.
    integer, intent(in) :: i

    call ele_add_to_set(i)
  end subroutine

  subroutine eleset_get_size(size)
    integer, intent(out) :: size

    call ele_get_size(size)
  end subroutine

  subroutine eleset_fetch_list(arr)
    !!< Fetch the list and clear it.
    integer, dimension(:), intent(out) :: arr
    call ele_fetch_list(arr)
  end subroutine

  subroutine eleset_get_ele(i, ele)
    !!< Get element i in the set.
    integer, intent(in) :: i
    integer, intent(out) :: ele
    call ele_get_ele(i, ele)
  end subroutine eleset_get_ele


end module element_set
