#include "fdebug.h"

module data_structures

  use integer_set_module
  use integer_hash_table_module

  implicit none

  interface invert_set
    module procedure invert_set_vector, invert_set_iset
  end interface invert_set

contains

  subroutine invert_set_vector(vector, ihash)
    ! A vector (/n1, n2, .../) implicitly defines a map
    ! 1 -> n1
    ! 2 -> n2
    ! ...

    ! Here we invert it to give the hash table that maps
    ! n1 -> 1
    ! n2 -> 2
    ! ...

    integer, dimension(:), intent(in) :: vector
    type(integer_hash_table), intent(out) :: ihash

    integer :: i

    call allocate(ihash)
    do i = 1, size(vector)
      call insert(ihash, vector(i), i)
    end do

  end subroutine invert_set_vector

  subroutine invert_set_iset(iset, ihash)
    ! A set {n1, n2, ...} implicitly defines a map
    ! 1 -> n1
    ! 2 -> n2
    ! ...

    ! Here we invert it to give the hash table that maps
    ! n1 -> 1
    ! n2 -> 2

    type(integer_set), intent(in) :: iset
    type(integer_hash_table), intent(out) :: ihash
    integer :: i

    call allocate(ihash)
    do i=1,key_count(iset)
      call insert(ihash, fetch(iset, i), i)
    end do
  end subroutine invert_set_iset

end module data_structures
