!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineering
!    Imperial College London
!
!    amcgsoftware@imperial.ac.uk
!
!    This library is free software; you can redistribute it and/or
!    modify it under the terms of the GNU Lesser General Public
!    License as published by the Free Software Foundation,
!    version 2.1 of the License.
!
!    This library is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!    Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with this library; if not, write to the Free Software
!    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!    USA

#include "fdebug.h"

module node_ownership

  use adjacency_lists
  use data_structures
  use fields
  use fldebug
  use linked_lists
  use node_owner_finder
  use pickers
  use sparse_tools

  implicit none

  private

  public :: find_node_ownership, find_node_ownership_brute_force, &
    & find_node_ownership_rtree, find_node_ownership_af, find_node_ownership_if
  public :: ownership_predicate

  real, parameter, public :: default_ownership_tolerance = 1.0e2 * epsilon(0.0)

  interface find_node_ownership
    module procedure find_node_ownership_rtree_no_tolerance, &
      & find_node_ownership_rtree_tolerance
  end interface find_node_ownership

  interface find_node_ownership_rtree
    module procedure find_node_ownership_rtree_no_tolerance, &
      & find_node_ownership_rtree_tolerance
  end interface find_node_ownership_rtree

contains

  subroutine find_node_ownership_brute_force(positions_a, positions_b, map, ownership_tolerance)
    !!< Find the elements in positions_a containing the nodes in positions_b.
    !!< Uses an brute force algorithm.

    type(vector_field), intent(inout) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, dimension(node_count(positions_b)), intent(out) :: map
    real, optional, intent(in) :: ownership_tolerance

    integer :: ele_a, node_b
    real :: lownership_tolerance

    ewrite(1, *) "In find_node_ownership_brute_force"

    if(present(ownership_tolerance)) then
      lownership_tolerance = ownership_tolerance
    else
      lownership_tolerance = default_ownership_tolerance
    end if

    map = -1

    node_b_loop: do node_b = 1, node_count(positions_b)
      do ele_a = 1, ele_count(positions_a)
        if(ownership_predicate(positions_a, positions_b, ele_a, node_b, lownership_tolerance)) then
          map(node_b) = ele_a
          cycle node_b_loop
        end if
      end do
    end do node_b_loop

    ewrite(1, *) "Exiting find_node_ownership_brute_force"

  end subroutine find_node_ownership_brute_force

  subroutine find_node_ownership_rtree_no_tolerance(positions_a, positions_b, map)
    !!< Find the elements in positions_a containing the nodes in positions_b.
    !!< Uses an rtree algorithm.
    !!< This does not use ownership tolerances - instead, it determines the
    !!< "best" owning elements (those that are the smallest distance in ideal
    !!< space from test nodes).

    type(vector_field), intent(inout) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, dimension(node_count(positions_b)), intent(out) :: map

    ewrite(1, *) "In find_node_ownership_rtree_no_tolerance"

    call picker_inquire(positions_a, positions_b, map, global = .false.)

    ewrite(1, *) "Exiting find_node_ownership_rtree_no_tolerance"

  end subroutine find_node_ownership_rtree_no_tolerance

  subroutine find_node_ownership_rtree_tolerance(positions_a, positions_b, map, ownership_tolerance)
    !!< Find all elements in positions_a within ownership_tolerance (in ideal
    !!< space) of nodes in positions_b.
    !!< Uses an rtree algorithm.

    type(vector_field), intent(inout) :: positions_a
    type(vector_field), intent(in) :: positions_b
    type(integer_set), dimension(node_count(positions_b)), intent(out) :: map
    real, intent(in) :: ownership_tolerance

    ewrite(1, *) "In find_node_ownership_rtree_tolerance"

    call picker_inquire(positions_a, positions_b, map, ownership_tolerance = ownership_tolerance)

    ewrite(2, *) "Min. elements: ", minval(key_count(map))
    ewrite(2, *) "Max. elements: ", maxval(key_count(map))

    ewrite(1, *) "Exiting find_node_ownership_rtree_tolerance"

  end subroutine find_node_ownership_rtree_tolerance

  subroutine find_node_ownership_af(positions_a, positions_b, map, ownership_tolerance, seed_b)
    !!< Find the elements in positions_a containing the nodes in positions_b.
    !!< Uses a simple advancing front algorithm.

    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, dimension(node_count(positions_b)), intent(out) :: map
    real, optional, intent(in) :: ownership_tolerance
    integer, optional, intent(in) :: seed_b

    integer :: ele_a, node_b
    real :: lownership_tolerance
    type(csr_sparsity), pointer :: eelist_a, nnlist_b
    type(ilist) :: null_ilist

    ! The advancing front
    logical, dimension(:), allocatable :: ele_a_in_list, tested_ele_a
    logical, dimension(:), allocatable :: node_b_in_list
    type(ilist) :: next_possible_node_b, possible_node_b
    type(ilist) :: next_ele_a

    ewrite(1, *) "In find_node_ownership_af"

    if(present(ownership_tolerance)) then
      lownership_tolerance = ownership_tolerance
    else
      lownership_tolerance = default_ownership_tolerance
    end if

    ! Advancing front seed
    if(present(seed_b)) then
      node_b = seed_b
    else
      node_b = 1
    end if
    assert(node_b > 0 .and. node_b <= node_count(positions_b))

    ! Initialisation
    allocate(tested_ele_a(ele_count(positions_a)))
    tested_ele_a = .false.
    allocate(ele_a_in_list(ele_count(positions_a)))
    ele_a_in_list = .false.
    allocate(node_b_in_list(node_count(positions_b)))
    node_b_in_list = .false.

    map = -1

    eelist_a => extract_eelist(positions_a)
    nnlist_b => extract_nnlist(positions_b)

    ! Step 1: Brute force search for the owner of the seed
    map(node_b) = brute_force_search(positions_a, positions_b, node_b, lownership_tolerance)

    ele_a = map(node_b)
    call insert(next_ele_a, ele_a)
    ele_a_in_list(ele_a) = .true.
    call advance_node_b_front(node_b)

    ! Step 2: The advancing front
    do while(next_ele_a%length > 0)
      ele_a = pop(next_ele_a)
      assert(.not. tested_ele_a(ele_a))

      do while(possible_node_b%length > 0)
        node_b = pop(possible_node_b)
        assert(map(node_b) < 0)

        if(ownership_predicate(positions_a, positions_b, ele_a, node_b, lownership_tolerance)) then
          map(node_b) = ele_a
          call advance_node_b_front(node_b)
        else if(map(node_b) < 0) then
          call insert(next_possible_node_b, node_b)
        end if
      end do

      call advance_ele_a_front(ele_a)
      possible_node_b = next_possible_node_b
      next_possible_node_b = null_ilist
    end do

    ! Cleanup
    assert(next_ele_a%length == 0)
    assert(possible_node_b%length == 0)
    deallocate(tested_ele_a)
    deallocate(ele_a_in_list)
    deallocate(node_b_in_list)

    ewrite(1, *) "Exiting find_node_ownership_af"

  contains

    function brute_force_search(positions_a, positions_b, node_b, ownership_tolerance) result(ele_a)
      type(vector_field), intent(in) :: positions_a
      type(vector_field), intent(in) :: positions_b
      integer, intent(in) :: node_b
      real, intent(in) :: ownership_tolerance

      integer :: ele_a

      do ele_a = 1, ele_count(positions_a)
        if(ownership_predicate(positions_a, positions_b, ele_a, node_b, ownership_tolerance)) then
          return
        end if
      end do

      ewrite(-1, *) "For node ", node_b
      FLAbort("Brute force ownership search failed")

    end function brute_force_search

    subroutine advance_ele_a_front(ele_a)
      integer, intent(in) :: ele_a

      integer :: i
      integer, dimension(:), pointer :: neigh

      tested_ele_a(ele_a) = .true.
      neigh => row_m_ptr(eelist_a, ele_a)
      do i = 1, size(neigh)
        if(neigh(i) <= 0) cycle
        if(ele_a_in_list(neigh(i))) cycle
        if(tested_ele_a(neigh(i))) cycle

        call insert(next_ele_a, neigh(i))
        ele_a_in_list(neigh(i)) = .true.
      end do

    end subroutine advance_ele_a_front

    subroutine advance_node_b_front(node_b)
      integer, intent(in) :: node_b

      integer :: i
      integer, dimension(:), pointer :: neigh

      neigh => row_m_ptr(nnlist_b, node_b)
      do i = 1, size(neigh)
        if(node_b_in_list(neigh(i))) cycle
        if(map(neigh(i)) > 0) cycle

        call insert(possible_node_b, neigh(i))
        node_b_in_list(neigh(i)) = .true.
      end do

    end subroutine advance_node_b_front

  end subroutine find_node_ownership_af

  subroutine find_node_ownership_if(positions_a, positions_b, map, ownership_tolerance, map_ab)
    !!< Find the elements in positions_a containing the nodes in positions_b.
    !!< Uses the element intersection finder.

    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, dimension(node_count(positions_b)), intent(out) :: map
    real, optional, intent(in) :: ownership_tolerance
    type(ilist), dimension(ele_count(positions_a)), optional, intent(in) :: map_ab

    integer :: ele_a, ele_b, node_b
    integer, dimension(:), pointer :: ele_bs
    real :: lownership_tolerance
    type(csr_sparsity), pointer :: nelist_b
    type(inode), pointer :: node
    type(ilist), dimension(:), allocatable :: map_ba

    ewrite(1, *) "In find_node_ownership_if"

    if(present(ownership_tolerance)) then
      lownership_tolerance = ownership_tolerance
    else
      lownership_tolerance = default_ownership_tolerance
    end if

    allocate(map_ba(ele_count(positions_b)))
    if(present(map_ab)) then
      ! We need the inverse map here
      do ele_a = 1, ele_count(positions_a)
        node => map_ab(ele_a)%firstnode
        do while(associated(node))
          ele_b = node%value
          call insert(map_ba(ele_b), ele_a)
          node => node%next
        end do
      end do
    else
      map_ba = intersection_finder(positions_b, positions_a)
    end if

    nelist_b => extract_nelist(positions_b)

    map = -1
    node_b_loop: do node_b = 1, node_count(positions_b)
      ele_bs => row_m_ptr(nelist_b, node_b)
      assert(size(ele_bs) > 0)
      ele_b = ele_bs(1)

      node => map_ba(ele_b)%firstnode
      do while(associated(node))
        ele_a = node%value
        if(ownership_predicate(positions_a, positions_b, ele_a, node_b, lownership_tolerance)) then
          map(node_b) = ele_a
          cycle node_b_loop
        end if

        node => node%next
      end do
    end do node_b_loop

    do ele_b = 1, size(map_ba)
      call deallocate(map_ba(ele_b))
    end do
    deallocate(map_ba)

    ewrite(1, *) "Exiting find_node_ownership_if"

  end subroutine find_node_ownership_if

end module node_ownership
