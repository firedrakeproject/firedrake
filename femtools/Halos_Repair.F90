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
!    License as published by the Free Software Foundation; either
!    version 2.1 of the License, or (at your option) any later version.
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

module halos_repair

  use fields_data_types
  use fields_base
  use parallel_tools
  use fldebug
  use halo_data_types
  use halos_base
  use halos_debug
  use halos_numbering
  use halos_ownership
  use mpi_interfaces
  use quicksort

  implicit none

  private

  public :: reorder_halo, reorder_l1_from_l2_halo, reorder_element_halo, &
    & reorder_halo_receives, reorder_halo_from_element_halo

  interface reorder_halo
    module procedure reorder_halo_vector, reorder_halo_halo
  end interface reorder_halo

contains

  subroutine reorder_halo_vector(halo, repair_field)
    !!< Reorder the halo sends and receives for consistency with the supplied
    !!< repair field. Sends and receives are changed, although no communication
    !!< is involved.
    !!< IMPORTANT NOTE: This assumes that the repair field in the halo region is
    !!< *floating point equal* across processes.

    type(halo_type), intent(inout) :: halo
    type(vector_field), intent(in) :: repair_field

    integer :: i, j, receives_count, sends_count
    integer, dimension(:), allocatable :: permutation, receives, sends
    real, dimension(:, :), allocatable :: receive_data, send_data

    assert(any(halo_data_type(halo) == (/HALO_TYPE_CG_NODE, HALO_TYPE_DG_NODE/)))
    assert(.not. has_global_to_universal_numbering(halo))
    assert(.not. has_ownership(halo))

    do i = 1, halo_proc_count(halo)
      ! Step 1: Extract the current halo sends
      sends_count = halo_send_count(halo, i)
      allocate(sends(sends_count))
      sends = halo_sends(halo, i)

      ! Step 2: Pull out the data we currently have for the sends
      allocate(send_data(sends_count, repair_field%dim))
      do j = 1, sends_count
        send_data(j, :) = node_val(repair_field, sends(j))
      end do

      ! Step 3: Sort them into order
      allocate(permutation(sends_count))
      call sort(send_data, permutation)
      deallocate(send_data)
      call apply_permutation(sends, permutation)
      deallocate(permutation)

      ! Step 4: Set the halo sends
      call set_halo_sends(halo, i, sends)
      deallocate(sends)

      ! Step 5: Extract the current halo receives
      receives_count = halo_receive_count(halo, i)
      allocate(receives(receives_count))
      receives = halo_receives(halo, i)

      ! Step 6: Pull out the data we currently have for the receives
      allocate(receive_data(receives_count, repair_field%dim))
      do j = 1, receives_count
        receive_data(j, :) = node_val(repair_field, receives(j))
      end do

      ! Step 7: Sort them into order
      allocate(permutation(receives_count))
      call sort(receive_data, permutation)
      deallocate(receive_data)
      call apply_permutation(receives, permutation)
      deallocate(permutation)

      ! Step 8: Set the halo receives
      call set_halo_receives(halo, i, receives)
      deallocate(receives)
    end do

    ! The halo sends and recevies are now sorted on all processes for all
    ! processes, and hence must be in a consistent order (provided the halo
    ! is actually valid on the supplied repair field)

  end subroutine reorder_halo_vector

  subroutine reorder_halo_halo(halo, repair_halo)
    !!< Using the order information from the supplied repair halo, rerorder the
    !!< sends and receives in halo into a consistent order

    type(halo_type), intent(inout) :: halo
    type(halo_type), intent(in) :: repair_halo

    integer :: i
    integer, dimension(:), allocatable :: permutation
    integer, dimension(:), pointer :: halo_nodes

    assert(has_global_to_universal_numbering(repair_halo))
    assert(.not. has_global_to_universal_numbering(halo))
    assert(.not. has_ownership(halo))

    do i = 1, halo_proc_count(halo)
      allocate(permutation(halo_send_count(halo, i)))
      halo_nodes => halo_sends(halo, i)
      call qsort(halo_universal_numbers(repair_halo, halo_nodes), permutation)
      call apply_permutation(halo_nodes, permutation)
      deallocate(permutation)

      allocate(permutation(halo_receive_count(halo, i)))
      halo_nodes => halo_receives(halo, i)
      call qsort(halo_universal_numbers(repair_halo, halo_nodes), permutation)
      call apply_permutation(halo_nodes, permutation)
      deallocate(permutation)
    end do

  end subroutine reorder_halo_halo

  subroutine reorder_l1_from_l2_halo(l1_halo, l2_halo, sorted_l1_halo)
    !!< Use the supplied consistently ordered l2 halo to reorder the supplied l1
    !!< halo

    type(halo_type), intent(inout) :: l1_halo
    type(halo_type), intent(in) :: l2_halo
    !! If present and .true., indicates that the l1 halo nodes are already
    !! sorted into order
    logical, optional, intent(in) :: sorted_l1_halo

    integer :: i, index, j
    integer, dimension(:), allocatable :: l2_halo_nodes, permutation
    integer, dimension(:), pointer :: l1_halo_nodes
    logical :: lsorted_l1_halo

    lsorted_l1_halo = present_and_true(sorted_l1_halo)

    assert(.not. has_global_to_universal_numbering(l1_halo))
    assert(.not. has_ownership(l1_halo))
    assert(halo_proc_count(l1_halo) == halo_proc_count(l2_halo))

    do i = 1, halo_proc_count(l1_halo)
      ! Extract and (if necessary) sort the l1 halo sends
      l1_halo_nodes => halo_sends(l1_halo, i)
      if(size(l1_halo_nodes) == 0) cycle
      if(.not. lsorted_l1_halo) then
        allocate(permutation(size(l1_halo_nodes)))
        call qsort(l1_halo_nodes, permutation)
        call apply_permutation(l1_halo_nodes, permutation)
        deallocate(permutation)
      end if

      ! Extract and sort the l2 halo sends
      allocate(l2_halo_nodes(halo_send_count(l2_halo, i)))
      l2_halo_nodes = halo_sends(l2_halo, i)
      allocate(permutation(size(l2_halo_nodes)))
      call qsort(l2_halo_nodes, permutation)
      call apply_permutation(l2_halo_nodes, permutation)
      ! Zero out the pure l2 sends in the l2 halo, to leave just the l1 halo sends
      index = 1
      do j = 1, size(l2_halo_nodes)
        if(l1_halo_nodes(index) == l2_halo_nodes(j)) then
          index = index + 1
          if(index > size(l1_halo_nodes)) then
            l2_halo_nodes(j + 1:) = 0
            exit
          end if
        else
          l2_halo_nodes(j) = 0
        end if
      end do
      ! Permute the l2 halo sends (with pure l2 sends zerod out) back into their
      ! original order
      call apply_permutation(l2_halo_nodes, inverse_permutation(permutation))
      deallocate(permutation)
      ! Collapse the l1 halo sends (stripping out the zeros)
      index = 1
      do j = 1, size(l2_halo_nodes)
        if(l2_halo_nodes(j) > 0) then
          l1_halo_nodes(index) = l2_halo_nodes(j)
          index = index + 1
          if(index > size(l1_halo_nodes)) exit
        end if
      end do
      deallocate(l2_halo_nodes)
    end do

    do i = 1, halo_proc_count(l1_halo)
      ! Extract and (if necessary) sort the l1 halo receives
      l1_halo_nodes => halo_receives(l1_halo, i)
      if(size(l1_halo_nodes) == 0) cycle
      if(.not. lsorted_l1_halo) then
        allocate(permutation(size(l1_halo_nodes)))
        call qsort(l1_halo_nodes, permutation)
        call apply_permutation(l1_halo_nodes, permutation)
        deallocate(permutation)
      end if

      ! Extract and sort the l2 halo receives
      allocate(l2_halo_nodes(halo_receive_count(l2_halo, i)))
      l2_halo_nodes = halo_receives(l2_halo, i)
      allocate(permutation(size(l2_halo_nodes)))
      call qsort(l2_halo_nodes, permutation)
      call apply_permutation(l2_halo_nodes, permutation)
      ! Zero out the pure l2 receives in the l2 halo, to leave just the l1 halo receives
      index = 1
      do j = 1, size(l2_halo_nodes)
        if(l1_halo_nodes(index) == l2_halo_nodes(j)) then
          index = index + 1
          if(index > size(l1_halo_nodes)) then
            l2_halo_nodes(j + 1:) = 0
            exit
          end if
        else
          l2_halo_nodes(j) = 0
        end if
      end do
      ! Permute the l2 halo receives (with pure l2 receives zerod out) back into their
      ! original order
      call apply_permutation(l2_halo_nodes, inverse_permutation(permutation))
      deallocate(permutation)
      ! Collapse the l1 halo receives (stripping out the zeros)
      index = 1
      do j = 1, size(l2_halo_nodes)
        if(l2_halo_nodes(j) > 0) then
          l1_halo_nodes(index) = l2_halo_nodes(j)
          index = index + 1
          if(index > size(l1_halo_nodes)) exit
        end if
      end do
      deallocate(l2_halo_nodes)
    end do

  end subroutine reorder_l1_from_l2_halo

  subroutine reorder_halo_from_element_halo(node_halo, element_halo, mesh)
    !!< Using the order information in the element halo, reorder the sends
    !!< and receives in halo into a consistent order.
    !!<
    !!< This has the side effect of also defining the universal numbering on
    !!< node_halo.

    type(halo_type), intent(inout) :: node_halo
    type(halo_type), intent(in) :: element_halo
    type(mesh_type), intent(in) :: mesh

    integer :: p, nprocs, n
    integer, dimension(:), allocatable ::  global_numbers, order

    assert(any(halo_data_type(node_halo) == (/HALO_TYPE_CG_NODE, HALO_TYPE_DG_NODE/)))
    assert(halo_data_type(element_halo) == HALO_TYPE_ELEMENT)
    assert(.not. has_global_to_universal_numbering(node_halo))
    assert(.not. has_ownership(node_halo))

    nprocs = halo_proc_count(node_halo)

    ! First we need to establish the universal numbers of owned nodes.
    call create_global_to_universal_numbering(node_halo, local_only=.true.)

    call communicate_universal_numbers(node_halo, element_halo, mesh)

#ifdef DDEBUG
    select case(halo_ordering_scheme(node_halo))
      case(HALO_ORDER_GENERAL)
        assert(minval(node_halo%gnn_to_unn)>0)
      case(HALO_ORDER_TRAILING_RECEIVES)
        assert(minval(node_halo%receives_gnn_to_unn)>0)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select
#endif

    ! Now that we have the universal numbers, we can sort the send and
    ! receive lists into universal number order.
    do p=1, nprocs
       ! Sort receives for this processor.
       allocate(global_numbers(halo_receive_count(node_halo,p)))
       allocate(order(halo_receive_count(node_halo,p)))
       do n=1,size(global_numbers)
          global_numbers(n)=halo_universal_number(node_halo,&
               halo_receive(node_halo,p,n))
       end do

       call qsort(global_numbers, order)
       call apply_permutation(node_halo%receives(p)%ptr, order)
       deallocate(global_numbers,order)

       ! Sort sends for this processor.
       allocate(global_numbers(halo_send_count(node_halo,p)))
       allocate(order(halo_send_count(node_halo,p)))
       do n=1,size(global_numbers)
          global_numbers(n)=halo_universal_number(node_halo,&
               halo_send(node_halo,p,n))
       end do

       call qsort(global_numbers, order)
       call apply_permutation(node_halo%sends(p)%ptr, order)
       deallocate(global_numbers,order)
    end do

#ifdef DDEBUG
    if(halo_ordering_scheme(node_halo) == HALO_ORDER_TRAILING_RECEIVES) then
      assert(trailing_receives_consistent(node_halo))
    end if
#endif

    call create_ownership(node_halo)

  end subroutine reorder_halo_from_element_halo

  subroutine communicate_universal_numbers(node_halo, element_halo, mesh)
    ! Communicate the universal numbers of halos. Due to the problem of
    ! receive nodes in send elements, it is necessary to do this twice as
    ! some numbers will travel indirectly via third processors.

    type(halo_type), intent(inout) :: node_halo
    type(halo_type), intent(in) :: element_halo
    type(mesh_type), intent(in) :: mesh

#ifdef HAVE_MPI
    type(integer_vector), dimension(:), allocatable :: send_data, receive_data
    integer :: p, nloc, nprocs, communicator, rank, ierr, pos, n, e, stat, i
    integer :: current_unn, new_unn, total_halo, sends, receives
    integer, dimension(:), pointer :: nodes
    integer, dimension(:), allocatable :: requests, statuses
    integer tag(2)

    nprocs = halo_proc_count(node_halo)
    communicator = halo_communicator(node_halo)

    allocate(send_data(nprocs), receive_data(nprocs))

    ! Note that this won't work for mixed element meshes.
    nloc=ele_loc(mesh,1)

    ! Establish lists of the universal numbers for both send and receive elements.
    !
    ! Because the boundary between processors is in a slightly different
    ! location for element and node halos, it is necessary to do a double
    ! communication in which universal numbers corresponding to elements in
    ! the entire send and receive components of the element halo are transmitted.
    do p=1, nprocs
       total_halo=halo_send_count(element_halo,p)+halo_receive_count(element_halo,p)
       allocate(send_data(p)%ptr(nloc*total_halo))
       allocate(receive_data(p)%ptr(nloc*total_halo))
    end do

    doubleloop: do i = 1, 2
       do p=1, nprocs
          do e=1,halo_send_count(element_halo,p)
             nodes=>ele_nodes(mesh,halo_send(element_halo,p,e))
             do n=1,nloc
                pos=(e-1)*nloc+n
                send_data(p)%ptr(pos)=halo_universal_number(node_halo,nodes(n))
             end do
          end do

          sends=halo_send_count(element_halo,p)
          do e=1,halo_receive_count(element_halo,p)
             nodes=>ele_nodes(mesh,halo_receive(element_halo,p,e))
             do n=1,nloc
                pos=(e+sends-1)*nloc+n
                send_data(p)%ptr(pos)=halo_universal_number(node_halo,nodes(n))
             end do
          end do

       end do

       ! Actually communicate the data
       rank = getrank(communicator)
       tag(i) = next_mpi_tag()

       allocate(requests(2*nprocs))
       requests = MPI_REQUEST_NULL
       do p = 1, nprocs
          if(size(send_data(p)%ptr) > 0) then
            ! Non-blocking sends
            call mpi_isend(send_data(p)%ptr, size(send_data(p)%ptr), getpinteger()&
                 &, p - 1, tag(i), communicator, &
                 & requests(p), ierr)
            assert(ierr == MPI_SUCCESS)
          end if

          ! Non-blocking receives
          if(size(receive_data(p)%ptr) > 0) then
            call mpi_irecv(receive_data(p)%ptr, size(receive_data(p)%ptr),&
                 & getpinteger(), p-1, tag(i), communicator, requests(p+nprocs), ierr)
            assert(ierr == MPI_SUCCESS)
          end if
       end do

       ! Wait for all non-blocking communications to complete
       allocate(statuses(MPI_STATUS_SIZE * size(requests)))
       call mpi_waitall(size(requests), requests, statuses, ierr)
       assert(ierr == MPI_SUCCESS)
       deallocate(statuses)
       deallocate(requests)

       ! Now that we have all the communications, walk through them and set
       ! the corresponding universal numbers.
       do p=1, nprocs
          do e=1,halo_receive_count(element_halo,p)
             nodes=>ele_nodes(mesh,halo_receive(element_halo,p,e))

             nodeloop: do n=1,nloc
                pos=(e-1)*nloc+n
                new_unn=receive_data(p)%ptr(pos)
                if (new_unn>0) then
                   ! We have real data (unknown quantities are transmitted as
                   !-1)
                   current_unn=halo_universal_number(node_halo,nodes(n))
                   if(new_unn==current_unn) then
                      ! Already got this information
                      cycle nodeloop
                   else if (current_unn<0) then
                      call set_halo_universal_number(node_halo, nodes(n),&
                           & new_unn, stat)
                      ! We don't bother to check stat as it could legitimately
                      ! be 1 in the case where the halo does not cover the
                      ! whole mesh.
                   else
                      FLAbort("Universal node number mismatch")
                   end if
                end if

             end do nodeloop
          end do
          receives=halo_receive_count(element_halo,p)
          do e=1,halo_send_count(element_halo,p)
             nodes=>ele_nodes(mesh,halo_send(element_halo,p,e))

             receive_nodeloop: do n=1,nloc
                pos=(e+receives-1)*nloc+n
                new_unn=receive_data(p)%ptr(pos)
                if (new_unn>0) then
                   ! We have real data (unknown quantities are transmitted as
                   !-1)
                   current_unn=halo_universal_number(node_halo,nodes(n))
                   if(new_unn==current_unn) then
                      ! Already got this information
                      cycle receive_nodeloop
                   else if (current_unn<0) then
                      call set_halo_universal_number(node_halo, nodes(n),&
                           & new_unn, stat)
                      ! We don't bother to check stat as it could legitimately
                      ! be 1 in the case where the halo does not cover the
                      ! whole mesh.
                   else
                      FLAbort("Universal node number mismatch")
                   end if
                end if

             end do receive_nodeloop
          end do
       end do
    end do doubleloop

    do p=1, nprocs
       deallocate(send_data(p)%ptr)
       deallocate(receive_data(p)%ptr)
    end do
#else
    FLAbort("Communicating universal numbers makes no sense without MPI")
#endif

  end subroutine communicate_universal_numbers

  subroutine reorder_element_halo(element_halo, node_halo, mesh)
    !!< Reorder the halo sends and receives in the supplied element halo for
    !!< consistency with the universal numbering of the supplied node halo.

    type(halo_type), intent(inout) :: element_halo
    type(halo_type), intent(in) :: node_halo
    type(mesh_type), intent(in) :: mesh

    integer :: i, j, k, loc, receives_count, sends_count
    integer, dimension(:), allocatable :: permutation, receives, sends, unns, unns_permutation
    integer, dimension(:, :), allocatable :: receive_data, send_data

    assert(halo_data_type(element_halo) == HALO_TYPE_ELEMENT)
    assert(.not. has_global_to_universal_numbering(element_halo))
    assert(.not. has_ownership(element_halo))

    ! Hybrid meshes are not supported by this algorithm. This could be done
    ! if a csr sparsity sort were written to swap out for sort_integer_array,
    ! but as femtools doesn't yet support hybrid meshes anyway ...
    loc = ele_loc(mesh, 1)
#ifdef DDEBUG
    do i = 2, ele_count(mesh)
      assert(ele_loc(mesh, i) == loc)
    end do
#endif
    allocate(unns(loc))
    allocate(unns_permutation(loc))

    ! This is basically the same as reorder_halo, except we're repairing using
    ! the universal numbering of the node halo instead of a field

    do i = 1, halo_proc_count(element_halo)
      ! Step 1: Extract the current halo sends
      sends_count = halo_send_count(element_halo, i)
      allocate(sends(sends_count))
      sends = halo_sends(element_halo, i)

      ! Step 2: Collect the ordered universal numbering on the sends
      allocate(send_data(sends_count, loc))
      do j = 1, sends_count
        unns = halo_universal_numbers(node_halo, ele_nodes(mesh, sends(j)))
        call qsort(unns, unns_permutation)
        do k = 1, loc
          send_data(j, k) = unns(unns_permutation(k))
        end do
      end do

      ! Step 3: Sort them into order
      allocate(permutation(sends_count))
      call sort(send_data, permutation)
      deallocate(send_data)
      call apply_permutation(sends, permutation)
      deallocate(permutation)

      ! Step 4: Set the halo sends
      call set_halo_sends(element_halo, i, sends)
      deallocate(sends)

      ! Step 5: Extract the current halo receives
      receives_count = halo_receive_count(element_halo, i)
      allocate(receives(receives_count))
      receives = halo_receives(element_halo, i)

      ! Step 6: Collect the ordered universal numbering on the receives
      allocate(receive_data(receives_count, loc))
      do j = 1, receives_count
        unns = halo_universal_numbers(node_halo, ele_nodes(mesh, receives(j)))
        call qsort(unns, unns_permutation)
        do k = 1, loc
          receive_data(j, k) = unns(unns_permutation(k))
        end do
      end do

      ! Step 7: Sort them into order
      allocate(permutation(receives_count))
      call sort(receive_data, permutation)
      deallocate(receive_data)
      call apply_permutation(receives, permutation)
      deallocate(permutation)

      ! Step 8: Set the halo receives
      call set_halo_receives(element_halo, i, receives)
      deallocate(receives)
    end do

    deallocate(unns)
    deallocate(unns_permutation)

  end subroutine reorder_element_halo

  subroutine reorder_halo_receives(halo, repair_field)
    !!< Reorder the halo receives for consistency with the supplied repair
    !!< field (which will typically be Coordinate). The sends are unchanged in
    !!< the repair, although communication is involved.

    type(halo_type), intent(inout) :: halo
    type(vector_field), intent(in) :: repair_field

    assert(halo_valid_for_communication(halo))
    assert(.not. pending_communication(halo))
    assert(.not. has_global_to_universal_numbering(halo))
    assert(.not. has_ownership(halo))

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        FLAbort("Halo receive reordering is not yet available for halos with general ordering")
      case(HALO_ORDER_TRAILING_RECEIVES)
        call reorder_halo_receives_order_trailing_receives(halo, repair_field)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

   end subroutine reorder_halo_receives

  subroutine reorder_halo_receives_order_trailing_receives(halo, repair_field)
    !!< Reorder the halo receives for consistency with the supplied repair
    !!< field (which will typically be Coordinate). The sends are unchanged in
    !!< the repair, although communication is involved.
    !!< IMPORTANT NOTE: This assumes that the repair field in the halo region is
    !!< *floating point equal* across processes.

    type(halo_type), intent(inout) :: halo
    type(vector_field), intent(in) :: repair_field

#ifdef HAVE_MPI
    integer :: communicator, i, ierr, j, nprocs, nsends, rank, receives_count
    integer, dimension(:), allocatable :: receives, requests, &
      & send_types, start_indices, statuses, permutation, &
      & permutation_inverse
    real, dimension(:, :), allocatable :: current_receive_data, correct_receive_data
    integer tag

    assert(trailing_receives_consistent(halo))

    nprocs = halo_proc_count(halo)
    communicator = halo_communicator(halo)

    ! Step 1: Extract the current halo receives

    receives_count = halo_all_receives_count(halo)
    allocate(receives(receives_count))
    allocate(start_indices(nprocs))
    call extract_all_halo_receives(halo, receives, start_indices = start_indices)

    ! Step 2: Pull out the data we currently have for the receives

    allocate(current_receive_data(receives_count, repair_field%dim))
    do i = 1, receives_count
      current_receive_data(i, :) = node_val(repair_field, receives(i))
    end do

    ! Step 3: Communicate the data from other processes, indicating what we
    ! should have retrieved on the receives

    ! Create indexed MPI types defining the indices into real data to be
    ! sent/received
    allocate(send_types(nprocs))
    send_types = MPI_DATATYPE_NULL
    do i = 1, nprocs
      nsends = halo_send_count(halo, i)
      if(nsends > 0) then
        call mpi_type_create_indexed_block(nsends, 1, &
          & halo_sends(halo, i) - 1, getpreal(), send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Set up non-blocking communications
    allocate(correct_receive_data(receives_count, repair_field%dim))
    allocate(requests(nprocs * 2 * repair_field%dim))
    requests = MPI_REQUEST_NULL
    rank = getrank(communicator)
    tag = next_mpi_tag()

    do i = 1, nprocs
      do j = 1, repair_field%dim
        ! Non-blocking sends
        if(halo_send_count(halo, i) > 0) then
          call mpi_isend(repair_field%val(j,:), 1, send_types(i), i - 1, &
               tag, communicator, requests((i - 1) * repair_field%dim + j), ierr)
          assert(ierr == MPI_SUCCESS)
        end if

        ! Non-blocking receives
        if(halo_receive_count(halo, i) > 0) then
          call mpi_irecv(correct_receive_data(start_indices(i):, j), &
               halo_receive_count(halo, i), getpreal(), i - 1, tag, &
               communicator, requests((i - 1 + nprocs) * repair_field%dim + j), ierr)
          assert(ierr == MPI_SUCCESS)
        end if
      end do
    end do
    deallocate(start_indices)

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(statuses)
    deallocate(requests)

    ! Free the indexed MPI types
    do i = 1, nprocs
      if(send_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do
    deallocate(send_types)

    ! If all is as expected, current_receive_data and correct_receive_data now
    ! contain the same data in different orders.
#ifdef DDEBUG
    if(all(abs(current_receive_data - correct_receive_data) < epsilon(0.0))) then
      ewrite(0, *) "Warning: reorder_halo_receives called for halo with correctly orderered receives"
    end if
#endif

    ! Step 4: Sort them - this is the fiddly bit

    ! Sort 1: Sort the current receive data
    allocate(permutation(receives_count))
    call sort(current_receive_data, permutation)
    ! Apply the sort to the current receives
    call apply_permutation(receives, permutation)
#ifdef DDEBUG
    ! Sort the current receive data as well to enabled checking below
    call apply_permutation(current_receive_data, permutation)
#endif

    ! Sort 2: Sort the correct receive data using the same sorting algorithm
    call sort(correct_receive_data, permutation)
    ! Invert it
    allocate(permutation_inverse(receives_count))
    permutation_inverse = inverse_permutation(permutation)
    ! Apply the inverse sort to the sorted current receives
    call apply_permutation(receives, permutation_inverse)
#ifdef DDEBUG
    ! Again, sort the current receive data
    call apply_permutation(current_receive_data, permutation_inverse)
    ! We can now check that current_receive_data and correct_receive data
    ! actually contained the same data in different orders
    assert(all(abs(current_receive_data - correct_receive_data) < epsilon(0.0)))
#endif
    deallocate(permutation)
    deallocate(permutation_inverse)

    deallocate(current_receive_data)
    deallocate(correct_receive_data)

    ! Step 5: We now have our copy of receives in the correct order. Set the
    ! halo receives from this.
    call set_all_halo_receives(halo, receives)

    deallocate(receives)
#else
    if(.not. valid_serial_halo(halo)) then
      FLAbort("Cannot reorder halo receives without MPI support")
    end if
#endif

  end subroutine reorder_halo_receives_order_trailing_receives

end module halos_repair
