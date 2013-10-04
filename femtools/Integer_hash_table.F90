#include "fdebug.h"

module integer_hash_table_module
  ! Don't use this directly, use data_structures
  use iso_c_binding, only: c_ptr
  use fldebug
  type integer_hash_table
    type(c_ptr) :: address
  end type integer_hash_table

  interface
    subroutine integer_hash_table_create_c(i)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(out) :: i
    end subroutine integer_hash_table_create_c

    subroutine integer_hash_table_delete_c(i)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(inout) :: i
    end subroutine integer_hash_table_delete_c

    subroutine integer_hash_table_insert_c(i, k, v)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(inout) :: i
      integer, intent(in) :: k, v
    end subroutine integer_hash_table_insert_c

    pure subroutine integer_hash_table_length_c(i, l)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(out) :: l
    end subroutine integer_hash_table_length_c

    subroutine integer_hash_table_fetch_c(i, key, val)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(in) :: key
      integer, intent(out) :: val
    end subroutine integer_hash_table_fetch_c

    subroutine integer_hash_table_remove_c(i, key, stat)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(inout) :: i
      integer, intent(in) :: key
      integer, intent(out) :: stat
    end subroutine integer_hash_table_remove_c

    subroutine integer_hash_table_has_key_c(i, val, bool)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(in) :: val
      integer, intent(out) :: bool
    end subroutine integer_hash_table_has_key_c

    subroutine integer_hash_table_fetch_pair_c(i, idx, key, val)
      use iso_c_binding, only: c_ptr
      type(c_ptr), intent(in) :: i
      integer, intent(in) :: idx
      integer, intent(out) :: key, val
    end subroutine integer_hash_table_fetch_pair_c
  end interface

  interface allocate
    module procedure integer_hash_table_allocate
  end interface

  interface insert
    module procedure integer_hash_table_insert
  end interface

  interface remove
    module procedure integer_hash_table_remove
  end interface

  interface deallocate
    module procedure integer_hash_table_delete
  end interface

  interface has_key
    module procedure integer_hash_table_has_key
  end interface

  interface key_count
    module procedure integer_hash_table_length
  end interface

  interface fetch
    module procedure integer_hash_table_fetch, integer_hash_table_fetch_v
  end interface

  interface fetch_pair
    module procedure integer_hash_table_fetch_pair
  end interface

  interface print
    module procedure print_hash_table
  end interface

  interface copy
    module procedure integer_hash_table_copy
  end interface

  private
  public :: integer_hash_table, allocate, deallocate, has_key, key_count, fetch, insert, &
            fetch_pair, print, remove, copy

  contains

  subroutine integer_hash_table_copy(ihash_copy, ihash)
    type(integer_hash_table), intent(out) :: ihash_copy
    type(integer_hash_table), intent(in) :: ihash

    integer :: ind, key, key_val

    call allocate(ihash_copy)
    do ind = 1, key_count(ihash)
      call fetch_pair(ihash, ind, key, key_val)
      call insert(ihash_copy, key, key_val)
    end do

  end subroutine integer_hash_table_copy

  subroutine integer_hash_table_allocate(ihash)
    type(integer_hash_table), intent(out) :: ihash
    ihash = integer_hash_table_create()
  end subroutine integer_hash_table_allocate

  function integer_hash_table_create() result(ihash)
    type(integer_hash_table) :: ihash
    call integer_hash_table_create_c(ihash%address)
  end function integer_hash_table_create

  subroutine integer_hash_table_delete(ihash)
    type(integer_hash_table), intent(inout) :: ihash
    call integer_hash_table_delete_c(ihash%address)
  end subroutine integer_hash_table_delete

  subroutine integer_hash_table_insert(ihash, key, val)
    type(integer_hash_table), intent(inout) :: ihash
    integer, intent(in) :: key, val

    call integer_hash_table_insert_c(ihash%address, key, val)
  end subroutine integer_hash_table_insert

  pure function integer_hash_table_length(ihash) result(len)
    type(integer_hash_table), intent(in) :: ihash
    integer :: len

    call integer_hash_table_length_c(ihash%address, len)
  end function integer_hash_table_length

  function integer_hash_table_fetch(ihash, key) result(val)
    type(integer_hash_table), intent(in) :: ihash
    integer, intent(in) :: key
    integer :: val

    call integer_hash_table_fetch_c(ihash%address, key, val)
  end function integer_hash_table_fetch

  subroutine integer_hash_table_remove(ihash, key)
    type(integer_hash_table), intent(inout) :: ihash
    integer, intent(in) :: key
    integer :: stat

    call integer_hash_table_remove_c(ihash%address, key, stat)
    assert(stat == 1)
  end subroutine integer_hash_table_remove

  function integer_hash_table_fetch_v(ihash, keys) result(vals)
    type(integer_hash_table), intent(in) :: ihash
    integer, intent(in), dimension(:) :: keys
    integer, dimension(size(keys)) :: vals
    integer :: i

    do i=1,size(keys)
      call integer_hash_table_fetch_c(ihash%address, keys(i), vals(i))
    end do
  end function integer_hash_table_fetch_v

  function integer_hash_table_has_key(ihash, key) result(bool)
    type(integer_hash_table), intent(in) :: ihash
    integer, intent(in) :: key
    logical :: bool

    integer :: lbool
    call integer_hash_table_has_key_c(ihash%address, key, lbool)
    bool = (lbool == 1)
  end function integer_hash_table_has_key

  subroutine integer_hash_table_fetch_pair(ihash, idx, key, val)
    type(integer_hash_table), intent(in) :: ihash
    integer, intent(in) :: idx
    integer, intent(out) :: key, val

    call integer_hash_table_fetch_pair_c(ihash%address, idx, key, val)
  end subroutine integer_hash_table_fetch_pair

  subroutine print_hash_table(ihash, priority)
    type(integer_hash_table), intent(in) :: ihash
    integer, intent(in) :: priority

    integer :: i, key, val

    ewrite(priority,*) "Writing hash table: "
    do i=1,key_count(ihash)
      call fetch_pair(ihash, i, key, val)
      ewrite(priority,*) key, " --> ", val
    end do
  end subroutine print_hash_table
end module integer_hash_table_module
