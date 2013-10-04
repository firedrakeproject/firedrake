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

module node_owner_finder

  use data_structures
  use fields
  use fldebug
  use global_parameters, only : real_4, real_8
  use mpi_interfaces

  implicit none

  private

  public :: node_owner_finder_reset, cnode_owner_finder_set_input, &
    & cnode_owner_finder_find, cnode_owner_finder_query_output, &
    & cnode_owner_finder_get_output
  public :: node_owner_finder_set_input, node_owner_finder_find
  public :: out_of_bounds_tolerance, rtree_tolerance
  public :: ownership_predicate

  !! If a test node is more than this distance (in ideal space) outside of a
  !! test element in ownership tests, then the test element cannot own the test
  !! node
  real, parameter :: out_of_bounds_tolerance = 0.1
  !! Factor by which element bounding boxes are expanded in setting up the rtree
  real, parameter :: rtree_tolerance = 0.1

  interface node_owner_finder_reset
    subroutine cnode_owner_finder_reset(id)
      implicit none
      integer, intent(in) :: id
    end subroutine cnode_owner_finder_reset
  end interface node_owner_finder_reset

  interface cnode_owner_finder_set_input
    module procedure node_owner_finder_set_input_sp

    subroutine cnode_owner_finder_set_input(id, positions, enlist, dim, loc, nnodes, nelements)
      use global_parameters, only : real_8
      implicit none
      integer, intent(out) :: id
      integer, intent(in) :: dim
      integer, intent(in) :: loc
      integer, intent(in) :: nnodes
      integer, intent(in) :: nelements
      real(kind = real_8), dimension(nnodes * dim), intent(in) :: positions
      integer, dimension(nelements * loc), intent(in) :: enlist
    end subroutine cnode_owner_finder_set_input
  end interface cnode_owner_finder_set_input

  interface node_owner_finder_set_input
    module procedure node_owner_finder_set_input_positions
  end interface node_owner_finder_set_input

  interface cnode_owner_finder_find
    module procedure node_owner_finder_find_sp

    subroutine cnode_owner_finder_find(id, position, dim)
      use global_parameters, only : real_8
      implicit none
      integer, intent(in) :: id
      integer, intent(in) :: dim
      real(kind = real_8), dimension(dim), intent(in) :: position
    end subroutine cnode_owner_finder_find
  end interface cnode_owner_finder_find

  interface node_owner_finder_find
    module procedure node_owner_finder_find_single_position, &
      & node_owner_finder_find_multiple_positions, &
      & node_owner_finder_find_single_position_tolerance, &
      & node_owner_finder_find_multiple_positions_tolerance, &
      & node_owner_finder_find_node, node_owner_finder_find_nodes, &
      & node_owner_finder_find_node_tolerance, &
      & node_owner_finder_find_nodes_tolerance
  end interface node_owner_finder_find

  interface cnode_owner_finder_query_output
    subroutine cnode_owner_finder_query_output(id, nelms)
      implicit none
      integer, intent(in) :: id
      integer, intent(out) :: nelms
    end subroutine cnode_owner_finder_query_output
  end interface cnode_owner_finder_query_output

  interface cnode_owner_finder_get_output
    subroutine cnode_owner_finder_get_output(id, ele_id, index)
      implicit none
      integer, intent(in) :: id
      integer, intent(out) :: ele_id
      integer, intent(in) :: index
    end subroutine cnode_owner_finder_get_output
  end interface cnode_owner_finder_get_output

  interface ownership_predicate
    module procedure ownership_predicate_position, ownership_predicate_node
  end interface ownership_predicate

contains

  subroutine node_owner_finder_set_input_sp(id, positions, enlist, dim, loc, nnodes, nelements)
    integer, intent(out) :: id
    integer, intent(in) :: dim
    integer, intent(in) :: loc
    integer, intent(in) :: nnodes
    integer, intent(in) :: nelements
    real(kind = real_4), dimension(nnodes * dim), intent(in) :: positions
    integer, dimension(nelements * loc), intent(in) :: enlist

    call cnode_owner_finder_set_input(id, real(positions, kind = real_8), enlist, dim, loc, nnodes, nelements)

  end subroutine node_owner_finder_set_input_sp

  subroutine node_owner_finder_set_input_positions(id, positions)
    !!< Generate a new node owner finder for the supplied positions. Returns the
    !!< node owner finder ID in id.

    integer, intent(out) :: id
    type(vector_field), intent(in) :: positions

    integer :: i
    real, dimension(:), allocatable :: lpositions

    allocate(lpositions(node_count(positions) * positions%dim))
    do i = 1, node_count(positions)
      lpositions((i - 1) * positions%dim + 1:i * positions%dim) = node_val(positions, i)
    end do

    call cnode_owner_finder_set_input(id, lpositions, positions%mesh%ndglno, positions%dim, ele_loc(positions, 1), node_count(positions), ele_count(positions))

    deallocate(lpositions)

  end subroutine node_owner_finder_set_input_positions

  subroutine node_owner_finder_find_sp(id, position, dim)
    integer, intent(in) :: id
    integer, intent(in) :: dim
    real(kind = real_4), dimension(dim), intent(in) :: position

    call cnode_owner_finder_find(id, real(position, kind = real_8), dim)

  end subroutine node_owner_finder_find_sp

  subroutine node_owner_finder_find_single_position(id, positions_a, position, ele_id, global)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element ID owning the given position.
    !!< This does not use ownership tolerances - instead, it determines the
    !!< "best" owning elements (those that are the smallest distance in ideal
    !!< space from test nodes).

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    real, dimension(:), intent(in) :: position
    integer, intent(out) :: ele_id
    !! If present and .false., do not perform a global ownership test across all
    !! processes
    logical, optional, intent(in) :: global

    integer, dimension(1) :: lele_id

    call node_owner_finder_find(id, positions_a, spread(position, 2, 1), lele_id, global = global)
    ele_id = lele_id(1)

  end subroutine node_owner_finder_find_single_position

  subroutine node_owner_finder_find_multiple_positions(id, positions_a, positions, ele_ids, global)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element IDs owning the given positions.
    !!< This does not use ownership tolerances - instead, it determines the
    !!< "best" owning elements (those that are the smallest distance in ideal
    !!< space from test nodes).

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    real, dimension(:, :), intent(in) :: positions
    integer, dimension(size(positions, 2)), intent(out) :: ele_ids
    !! If present and .false., do not perform a global ownership test across all
    !! processes
    logical, optional, intent(in) :: global

    if(.not. present_and_false(global) .and. isparallel()) then
      call find_parallel()
    else
      call find_serial()
    end if

 contains

    ! Separate serial and parallel versions, as in parallel we need to keep a
    ! record of the closest_misses

    subroutine find_serial()
      integer :: closest_ele_id, i, j, nele_ids, possible_ele_id
      real :: closest_miss, miss

      ele_ids = -1
      positions_loop: do i = 1, size(positions, 2)
        call cnode_owner_finder_find(id, positions(:, i), size(positions, 1))
        call cnode_owner_finder_query_output(id, nele_ids)

        closest_ele_id = -1
        ! We don't tolerate very large ownership failures
        closest_miss = out_of_bounds_tolerance
        do j = 1, nele_ids
          call cnode_owner_finder_get_output(id, possible_ele_id, j)
          ! Zero tolerance - we're not using an "epsilon-ball" approach here
          if(ownership_predicate(positions_a, possible_ele_id, positions(:, i), 0.0, miss = miss)) then
            ele_ids(i) = possible_ele_id
            ! We've found an owner - no need to worry about the closest miss
            cycle positions_loop
          else if(miss < closest_miss) then
            ! We didn't find an owner, but did find the closest miss so far
            closest_ele_id = possible_ele_id
            closest_miss = miss
          end if
        end do

        ! We didn't find an owner, so choose the element with the closest miss
        ele_ids(i) = closest_ele_id

      end do positions_loop

    end subroutine find_serial

    subroutine find_parallel()
      integer :: closest_ele_id, i, j, nele_ids, possible_ele_id
      real :: miss
      real, dimension(:), allocatable :: closest_misses
#ifdef HAVE_MPI
      integer :: communicator, ierr, rank
      real, dimension(2, size(positions, 2)) :: misses_send, minlocs_recv
#endif

      ele_ids = -1
      allocate(closest_misses(size(positions, 2)))
      ! We don't tolerate very large ownership failures
      closest_misses = out_of_bounds_tolerance

      positions_loop: do i = 1, size(positions, 2)
        call cnode_owner_finder_find(id, positions(:, i), size(positions, 1))
        call cnode_owner_finder_query_output(id, nele_ids)

        closest_ele_id = -1
        possible_elements_loop: do j = 1, nele_ids
          call cnode_owner_finder_get_output(id, possible_ele_id, j)
          ! If this process does not own this possible_ele_id element then
          ! don't consider it.  This filter is needed to make this subroutine work in
          ! parallel without having to use universal numbers, which aren't defined
          ! for all the meshes that use this subroutine.
          if(.not.element_owned(positions_a,possible_ele_id)) then
             assert(isparallel())
             cycle possible_elements_loop
          end if
          ! Zero tolerance - we're not using an "epsilon-ball" approach here
          if(ownership_predicate(positions_a, possible_ele_id, positions(:, i), 0.0, miss = miss)) then
            ele_ids(i) = possible_ele_id
            ! We've found an owner - no need to worry about the closest miss
            closest_misses(i) = 0.0
            cycle positions_loop
          else if(miss < closest_misses(i)) then
            ! We didn't find an owner, but did find the closest miss so far
            closest_ele_id = possible_ele_id
            closest_misses(i) = miss
          end if
        end do possible_elements_loop

        ! We didn't find an owner, so choose the element with the closest miss
        ele_ids(i) = closest_ele_id

      end do positions_loop

      ! Find which processes have the smallest miss for each coordinate
#ifdef HAVE_MPI
      communicator = halo_communicator(positions_a)
      rank = getrank(communicator = communicator)

      misses_send(1, :) = closest_misses
      misses_send(2, :) = float(rank)

      call mpi_allreduce(misses_send, minlocs_recv, size(misses_send, 2), &
#ifdef DOUBLEP
        & MPI_2DOUBLE_PRECISION, &
#else
        & MPI_2REAL, &
#endif
        & MPI_MINLOC, communicator, ierr)
      assert(ierr == MPI_SUCCESS)

      do i = 1, size(minlocs_recv, 2)
        if(int(minlocs_recv(2, i)) /= rank) then
          ! Another processes has a smaller miss for this coordinate
          ele_ids(i) = -1
        end if
        ! if no process has closest_misses(i) < out_of_bounds_tolerance
        ! then ele_ids(i) is already set to -1 in positions_loop above
        ! on all processes.  This matches the find_serial(0) behaviour.
      end do
#endif

      deallocate(closest_misses)

    end subroutine find_parallel

  end subroutine node_owner_finder_find_multiple_positions

  subroutine node_owner_finder_find_single_position_tolerance(id, positions_a, position, ele_ids, ownership_tolerance)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element ID owning the given position using an
    !!< ownership tolerance. This performs a strictly local (this process)
    !!< ownership test.

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    real, dimension(:), intent(in) :: position
    type(integer_set), intent(out) :: ele_ids
    real, intent(in) :: ownership_tolerance

    type(integer_set), dimension(1) :: lele_ids

    call node_owner_finder_find(id, positions_a, spread(position, 2, 1), lele_ids, ownership_tolerance = ownership_tolerance)
    ele_ids = lele_ids(1)

  end subroutine node_owner_finder_find_single_position_tolerance

  subroutine node_owner_finder_find_multiple_positions_tolerance(id, positions_a, positions, ele_ids, ownership_tolerance)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element IDs owning the given positions using an
    !!< ownership tolerance. This performs a strictly local (this process)
    !!< ownership test.

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    real, dimension(:, :), intent(in) :: positions
    type(integer_set), dimension(size(positions, 2)), intent(out) :: ele_ids
    real, intent(in) :: ownership_tolerance

    integer ::  i, j, nele_ids, possible_ele_id

    ! Elements will be missed by the rtree query if ownership_tolerance is too
    ! big
    assert(ownership_tolerance <= rtree_tolerance)

    call allocate(ele_ids)
    positions_loop: do i = 1, size(positions, 2)
      call cnode_owner_finder_find(id, positions(:, i), size(positions, 1))
      call cnode_owner_finder_query_output(id, nele_ids)

      do j = 1, nele_ids
        call cnode_owner_finder_get_output(id, possible_ele_id, j)
        if(ownership_predicate(positions_a, possible_ele_id, positions(:, i), ownership_tolerance)) then
          ! We've found an owner
          call insert(ele_ids(i), possible_ele_id)
        end if
      end do

    end do positions_loop

  end subroutine node_owner_finder_find_multiple_positions_tolerance

  subroutine node_owner_finder_find_node(id, positions_a, positions_b, ele_a, node_b, global)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element ID owning the given node in positions_b.
    !!< This does not use ownership tolerances - instead, it determines the
    !!< "best" owning elements (those that are the smallest distance in ideal
    !!< space from test nodes).

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, intent(out) :: ele_a
    integer, intent(in) :: node_b
    !! If present and .false., do not perform a global ownership test across all
    !! processes
    logical, optional, intent(in) :: global

    call node_owner_finder_find(id, positions_a, node_val(positions_b, node_b), ele_a, global = global)

  end subroutine node_owner_finder_find_node

  subroutine node_owner_finder_find_nodes(id, positions_a, positions_b, eles_a, global)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element ID owning the nodes in positions_b.
    !!< This does not use ownership tolerances - instead, it determines the
    !!< "best" owning elements (those that are the smallest distance in ideal
    !!< space from test nodes).

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, dimension(node_count(positions_b)), intent(out) :: eles_a
    !! If present and .false., do not perform a global ownership test across all
    !! processes
    logical, optional, intent(in) :: global

    integer :: node_b
    real, dimension(:, :), allocatable :: lpositions

    allocate(lpositions(positions_b%dim, node_count(positions_b)))

    do node_b = 1, node_count(positions_b)
      lpositions(:, node_b) = node_val(positions_b, node_b)
    end do

    call node_owner_finder_find(id, positions_a, lpositions, eles_a, global = global)

    deallocate(lpositions)

  end subroutine node_owner_finder_find_nodes

  subroutine node_owner_finder_find_node_tolerance(id, positions_a, positions_b, eles_a, node_b, ownership_tolerance)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element ID owning the given node in positions_b
    !!< using an ownership tolerance. This performs a strictly local (this
    !!< process) ownership test.

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    type(integer_set), intent(out) :: eles_a
    integer, intent(in) :: node_b
    real, intent(in) :: ownership_tolerance

    call node_owner_finder_find(id, positions_a, node_val(positions_b, node_b), eles_a, ownership_tolerance = ownership_tolerance)

  end subroutine node_owner_finder_find_node_tolerance

  subroutine node_owner_finder_find_nodes_tolerance(id, positions_a, positions_b, eles_a, ownership_tolerance)
    !!< For the node owner finder with ID id corresponding to positions
    !!< positions_a, find the element ID owning the nodes in positions_b
    !!< using an ownership tolerance. This performs a strictly local (this
    !!< process) ownership test.

    integer, intent(in) :: id
    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    type(integer_set), dimension(node_count(positions_b)), intent(out) :: eles_a
    real, intent(in) :: ownership_tolerance

    integer :: node_b
    real, dimension(:, :), allocatable :: lpositions

    allocate(lpositions(positions_b%dim, node_count(positions_b)))

    do node_b = 1, node_count(positions_b)
      lpositions(:, node_b) = node_val(positions_b, node_b)
    end do

    call node_owner_finder_find(id, positions_a, lpositions, eles_a, ownership_tolerance = ownership_tolerance)

    deallocate(lpositions)

  end subroutine node_owner_finder_find_nodes_tolerance

  function ownership_predicate_position(positions_a, ele_a, position, ownership_tolerance, miss, l_coords) result(owned)
    !!< Node ownership predicate. Returns .true. if the given position is
    !!< contained within element ele_a of positions_a to within tolerance
    !!< ownership_tolerance.

    type(vector_field), intent(in) :: positions_a
    integer, intent(in) :: ele_a
    real, dimension(positions_a%dim), intent(in) :: position
    real, intent(in) :: ownership_tolerance
    !!< Return the "miss" - the distance (in ideal space) of the test position
    !!< from the test element
    real, optional, intent(out) :: miss
    !!< Return the coordinate (in ideal space) of the test position
    !!< in the test element
    real, dimension(positions_a%dim + 1), optional, intent(out) :: l_coords

    logical :: owned

    real :: lmiss
    real, dimension(positions_a%dim + 1) :: ll_coords

    assert(ownership_tolerance >= 0.0)

    ll_coords = local_coords(positions_a, ele_a, position)

    assert(cell_family(positions_a, ele_a) == FAMILY_SIMPLEX)
    if(any(ll_coords < 0.0)) then
      lmiss = -minval(ll_coords)
      if(lmiss < ownership_tolerance) then
        owned = .true.
      else
        owned = .false.
      end if
      if(present(miss)) miss = lmiss
    else
      owned = .true.
      if(present(miss)) miss = 0.0
    end if

    if(present(l_coords)) l_coords = ll_coords

  end function ownership_predicate_position

  function ownership_predicate_node(positions_a, positions_b, ele_a, node_b, ownership_tolerance, miss, l_coords) result(owned)
    !!< Node ownership predicate. Returns .true. if the given node in
    !!< positions_b is contained within element ele_a of positions_a to within
    !!< tolerance ownership_tolerance.

    type(vector_field), intent(in) :: positions_a
    type(vector_field), intent(in) :: positions_b
    integer, intent(in) :: ele_a
    integer, intent(in) :: node_b
    real, intent(in) :: ownership_tolerance
    !!< Return the "miss" - the distance (in ideal space) of the test position
    !!< from the test element
    real, optional, intent(out) :: miss
    !!< Return the coordinate (in ideal space) of the test position
    !!< in the test element
    real, dimension(positions_a%dim + 1), optional, intent(out) :: l_coords

    logical :: owned

    owned = ownership_predicate(positions_a, ele_a, node_val(positions_b, node_b), ownership_tolerance, &
      & miss = miss, l_coords = l_coords)

  end function ownership_predicate_node

end module node_owner_finder
