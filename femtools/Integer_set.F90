#include "fdebug.h"

module integer_set_module
  ! Don't use this directly, use data_structures
  use iso_c_binding, only: c_ptr,c_null_ptr
  use fldebug
  type integer_set
    type(c_ptr) :: address=c_null_ptr
  end type integer_set

  type integer_set_vector
     type(integer_set), dimension(:), pointer :: sets
  end type integer_set_vector

  interface
    subroutine integer_set_create_c(i)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(out) :: i
    end subroutine integer_set_create_c

    subroutine integer_set_delete_c(i)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(inout) :: i
    end subroutine integer_set_delete_c

    subroutine integer_set_insert_c(i, v, c)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(inout) :: i
      integer, intent(in) :: v
      integer, intent(out) :: c
    end subroutine integer_set_insert_c

    pure subroutine integer_set_length_c(i, l)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(out) :: l
    end subroutine integer_set_length_c

    subroutine integer_set_fetch_c(i, idx, val)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(in) :: idx
      integer, intent(out) :: val
    end subroutine integer_set_fetch_c

    subroutine integer_set_remove_c(i, idx, stat)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(in) :: idx
      integer, intent(out) :: stat
    end subroutine integer_set_remove_c

    subroutine integer_set_has_value_c(i, val, bool)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(in) :: val
      integer, intent(out) :: bool
    end subroutine integer_set_has_value_c
  end interface

  interface allocate
    module procedure integer_set_allocate_single,&
         & integer_set_allocate_vector, integer_set_allocate_array
  end interface

  interface insert
    module procedure integer_set_insert, integer_set_insert_multiple, &
      integer_set_insert_set
  end interface

  interface deallocate
    module procedure integer_set_delete_single, integer_set_delete_vector,&
         & integer_set_delete_array
  end interface

  interface has_value
    module procedure integer_set_has_value, integer_set_has_value_multiple
  end interface

  interface key_count
    module procedure integer_set_length_single, integer_set_length_vector
  end interface

  interface fetch
    module procedure integer_set_fetch
  end interface

  interface remove
    module procedure integer_set_remove
  end interface

  interface copy
    module procedure integer_set_copy, integer_set_copy_multiple
  end interface

  interface set_intersection
    module procedure set_intersection_two, set_intersection_multiple
  end interface

  private
  public :: integer_set, allocate, deallocate, has_value, key_count, fetch, insert, &
          & set_complement, set2vector, set_intersection, set_minus, remove, copy, &
          & integer_set_vector

  contains

  subroutine integer_set_allocate_single(iset)
    type(integer_set), intent(out) :: iset
    iset = integer_set_create()
  end subroutine integer_set_allocate_single

  subroutine integer_set_allocate_vector(iset)
    type(integer_set), dimension(:), intent(out) :: iset

    integer :: i

    do i = 1, size(iset)
      call allocate(iset(i))
    end do

  end subroutine integer_set_allocate_vector

  subroutine integer_set_allocate_array(iset)
    type(integer_set), dimension(:,:), intent(out) :: iset

    integer :: i,j

    do i = 1, size(iset,1)
       do j = 1, size(iset,2)
          call allocate(iset(i,j))
       end do
    end do

  end subroutine integer_set_allocate_array

  function integer_set_create() result(iset)
    type(integer_set) :: iset
    call integer_set_create_c(iset%address)
  end function integer_set_create

  subroutine integer_set_delete_single(iset)
    type(integer_set), intent(inout) :: iset
    call integer_set_delete_c(iset%address)
  end subroutine integer_set_delete_single

  subroutine integer_set_delete_vector(iset)
    type(integer_set), dimension(:), intent(inout) :: iset

    integer :: i

    do i = 1, size(iset)
      call deallocate(iset(i))
    end do

  end subroutine integer_set_delete_vector

  subroutine integer_set_delete_array(iset)
    type(integer_set), dimension(:,:), intent(inout) :: iset

    integer :: i,j

    do i = 1, size(iset,1)
       do j = 1, size(iset,2)
          call deallocate(iset(i,j))
       end do
    end do

  end subroutine integer_set_delete_array

  subroutine integer_set_insert(iset, val, changed)
    type(integer_set), intent(inout) :: iset
    integer, intent(in) :: val
    logical, intent(out), optional :: changed
    integer :: lchanged

    call integer_set_insert_c(iset%address, val, lchanged)

    if (present(changed)) then
      changed = (lchanged == 1)
    end if
  end subroutine integer_set_insert

  subroutine integer_set_insert_multiple(iset, values)
    type(integer_set), intent(inout) :: iset
    integer, dimension(:), intent(in) :: values
    integer :: i

    do i=1,size(values)
      call insert(iset, values(i))
    end do
  end subroutine integer_set_insert_multiple

  subroutine integer_set_insert_set(iset, value_set)
    type(integer_set), intent(inout) :: iset
    type(integer_set), intent(in) :: value_set
    integer :: i

    do i=1, key_count(value_set)
      call insert(iset, fetch(value_set,i))
    end do
  end subroutine integer_set_insert_set

  pure function integer_set_length_single(iset) result(len)
    type(integer_set), intent(in) :: iset
    integer :: len

    call integer_set_length_c(iset%address, len)
  end function integer_set_length_single

  pure function integer_set_length_vector(iset) result(len)
    type(integer_set), dimension(:), intent(in) :: iset

    integer, dimension(size(iset)) :: len

    integer :: i

    do i = 1, size(iset)
      len(i) = key_count(iset(i))
    end do

  end function integer_set_length_vector

  function integer_set_fetch(iset, idx) result(val)
    type(integer_set), intent(in) :: iset
    integer, intent(in) :: idx
    integer :: val

    call integer_set_fetch_c(iset%address, idx, val)
  end function integer_set_fetch

  subroutine integer_set_remove(iset, idx)
    type(integer_set), intent(in) :: iset
    integer, intent(in) :: idx
    integer :: stat

    call integer_set_remove_c(iset%address, idx, stat)
    assert(stat == 1)
  end subroutine integer_set_remove

  function integer_set_has_value(iset, val) result(bool)
    type(integer_set), intent(in) :: iset
    integer, intent(in) :: val
    logical :: bool

    integer :: lbool
    call integer_set_has_value_c(iset%address, val, lbool)
    bool = (lbool == 1)
  end function integer_set_has_value

  function integer_set_has_value_multiple(iset, val) result(bool)
    type(integer_set), intent(in) :: iset
    integer, dimension(:), intent(in) :: val
    logical, dimension(size(val)) :: bool

    integer:: i

    do i=1, size(val)
      bool(i)=integer_set_has_value(iset, val(i))
    end do
  end function integer_set_has_value_multiple

  subroutine set_complement(complement, universe, current)
    ! complement = universe \ current
    type(integer_set), intent(out) :: complement
    type(integer_set), intent(in) :: universe, current
    integer :: i, val

    call allocate(complement)
    do i=1,key_count(universe)
      val = fetch(universe, i)
      if (.not. has_value(current, val)) then
        call insert(complement, val)
      end if
    end do
  end subroutine set_complement

  subroutine set_intersection_two(intersection, A, B)
    ! intersection = A n B
    type(integer_set), intent(out) :: intersection
    type(integer_set), intent(in) :: A, B
    integer :: i, val

    call allocate(intersection)
    do i=1,key_count(A)
      val = fetch(A, i)
      if (has_value(B, val)) then
        call insert(intersection, val)
      end if
    end do
  end subroutine set_intersection_two

  subroutine set_intersection_multiple(intersection, isets, mask)
    ! intersection = isets(i) n isets(j), forall i /= j
    type(integer_set), intent(out) :: intersection
    type(integer_set), dimension(:), intent(in) :: isets
    logical, dimension(:), intent(in), optional :: mask
    integer :: i, n

    ! Ring buffer of isets.
    type(integer_set), dimension(2) :: tmp_iset(0:1)
    integer :: r, oldr

    oldr=1
    r=0

    if (present(mask)) then
       assert(size(mask)==size(isets))

       if (all(.not.mask))then
          call allocate(intersection)
          return
       end if

       do n=1, size(isets)
          if (mask(n)) then
             call copy(tmp_iset(r), isets(n))
             exit
          end if
       end do

    else
       n=1
       call copy(tmp_iset(r), isets(n))
    end if

    do i = n+1, size(isets)
       if (present(mask)) then
          if (.not.mask(i)) cycle
       end if
       oldr=r
       r=mod(r+1,2)
       call set_intersection(tmp_iset(r), tmp_iset(oldr), isets(i))
       call deallocate(tmp_iset(oldr))
    end do

    intersection = tmp_iset(r)

  end subroutine set_intersection_multiple

  subroutine integer_set_copy(iset_copy, iset)
    type(integer_set), intent(out) :: iset_copy
    type(integer_set), intent(in) :: iset

    integer :: i, val

    call allocate(iset_copy)

    do i = 1, key_count(iset)
      val = fetch(iset, i)
      call insert(iset_copy, val)
    end do

  end subroutine integer_set_copy

  subroutine integer_set_copy_multiple(iset_copy, iset)
    type(integer_set), dimension(:), intent(out) :: iset_copy
    type(integer_set), dimension(:), intent(in) :: iset

    integer :: i

    assert(size(iset)==size(iset_copy))

    do i = 1, size(iset)
       call integer_set_copy(iset_copy(i), iset(i))
    end do

  end subroutine integer_set_copy_multiple

  subroutine set_minus(minus, A, B)
  ! minus = A \ B
    type(integer_set), intent(out) :: minus
    type(integer_set), intent(in) :: A, B
    integer :: i, val

    call allocate(minus)
    do i=1,key_count(A)
      val = fetch(A, i)
      if (.not. has_value(B, val)) then
        call insert(minus, val)
      end if
    end do
  end subroutine set_minus

  function set2vector(iset) result(vec)
    type(integer_set), intent(in) :: iset
    integer, dimension(key_count(iset)) :: vec
    integer :: i

    do i=1,key_count(iset)
      vec(i) = fetch(iset, i)
    end do
  end function set2vector

end module integer_set_module
