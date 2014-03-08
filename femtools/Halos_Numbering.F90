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

module halos_numbering

  use data_structures
  use fldebug
  use futils
  use halo_data_types
  use halos_allocates
  use halos_base
  use halos_communications
  use halos_debug
  use mpi_interfaces
  use parallel_tools
  use quicksort
  use global_parameters, only: malloc
  use iso_c_binding
  implicit none

#ifdef __INTEL_COMPILER
  intrinsic sizeof
#define c_sizeof sizeof
#endif

  private

  public :: create_global_to_universal_numbering, &
    & has_global_to_universal_numbering, universal_numbering_count, &
    & halo_universal_number, halo_universal_numbers, get_universal_numbering, &
    & get_universal_numbering_inverse, set_halo_universal_number, &
    & ewrite_universal_numbers, valid_global_to_universal_numbering

  interface halo_universal_number
     module procedure halo_universal_number, halo_universal_number_vector
  end interface

  interface get_universal_numbering
     module procedure get_universal_numbering, get_universal_numbering_multiple_components
  end interface

contains

  subroutine create_global_to_universal_numbering(halo, local_only)
    !!< Create the global to universal node numbering, and cache it on the halo
    !!<
    !!< If local_only is present and .true. then only the universal numbers
    !!< for the owned nodes will be calculated. This is required when the
    !!< halos are not yet consistent and the universal numbers are to be
    !!< used to coordinate the halos.

    type(halo_type), intent(inout) :: halo
    logical, intent(in), optional :: local_only

    !assert(halo_valid_for_communication(halo))
    assert(.not. pending_communication(halo))

    if(has_global_to_universal_numbering(halo)) return

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        call create_global_to_universal_numbering_order_general(halo, local_only)
      case(HALO_ORDER_TRAILING_RECEIVES)
        call create_global_to_universal_numbering_order_trailing_receives&
             (halo, local_only)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

#ifdef DDEBUG
    if(.not. present_and_true(local_only)) then
      assert(valid_global_to_universal_numbering(halo))
    end if
#endif

  end subroutine create_global_to_universal_numbering

  subroutine create_global_to_universal_numbering_order_general(halo, local_only)
    type(halo_type), intent(inout) :: halo
    logical, intent(in), optional :: local_only

#ifdef HAVE_MPI
    integer :: communicator, i, ierr, nowned_nodes, nprocs, nreceives, nsends, &
      & rank, count, nnodes
    integer, dimension(:), allocatable :: receive_types, requests,&
         & send_types, statuses
    logical, dimension(:), allocatable :: local_nodes
    integer tag

    ewrite(1,*) "Creating universal numbering for a general order halo"

    nprocs = halo_proc_count(halo)
    communicator = halo_communicator(halo)
    rank = getrank(communicator)
    nowned_nodes = halo_nowned_nodes(halo)

    call set_universal_numbering_count(halo)

    ! Calculate the base universal node number for the owned nodes. The i th
    ! owned node then has universal node number equal to the base + i.
    call mpi_scan(nowned_nodes, halo%my_owned_nodes_unn_base, 1, getpinteger(), MPI_SUM, communicator, ierr)
    assert(ierr == MPI_SUCCESS)
    halo%my_owned_nodes_unn_base = halo%my_owned_nodes_unn_base - nowned_nodes
    ! gather this information from/to all other processors:
    allocate(halo%owned_nodes_unn_base(1:nprocs+1))
    call mpi_allgather(halo%my_owned_nodes_unn_base, 1, getpinteger(), &
      halo%owned_nodes_unn_base, 1, getpinteger(), communicator, ierr)
    assert(ierr == MPI_SUCCESS)
    assert( halo%owned_nodes_unn_base(rank+1)==halo%my_owned_nodes_unn_base )
    ! extra entry for convenience, such that number of owned nodes on a process
    ! can be derived from subtracting its unn_base from the next (similar to findrm):
    halo%owned_nodes_unn_base(nprocs+1)=universal_numbering_count(halo)
    assert( halo%owned_nodes_unn_base(nprocs)<=halo%owned_nodes_unn_base(nprocs+1) )

    nnodes = node_count(halo)

    allocate(local_nodes(nnodes))
    local_nodes=.true.
    assert(max_halo_receive_node(halo) <= nnodes)
    do i = 1, nprocs
      local_nodes(halo_receives(halo, i)) = .false.
    end do

    allocate(halo%gnn_to_unn(nnodes))
    halo%gnn_to_unn=-1

    count=halo%my_owned_nodes_unn_base
    do i=1, nnodes
       if (local_nodes(i)) then
          count=count+1
          halo%gnn_to_unn(i)=count
       end if
    end do

    if (present_and_true(local_only)) then
       return
    end if

    ! Create indexed MPI types defining the indices into halo%gnn_to_unn to be sent/received
    allocate(send_types(nprocs))
    allocate(receive_types(nprocs))
    send_types = MPI_DATATYPE_NULL
    receive_types = MPI_DATATYPE_NULL
    do i = 1, nprocs
      nsends = halo_send_count(halo, i)
      if(nsends > 0) then
        call mpi_type_create_indexed_block(nsends, 1, &
          & halo_sends(halo, i) - lbound(halo%gnn_to_unn, 1), &
          & getpinteger(), send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      nreceives = halo_receive_count(halo, i)
      if(nreceives > 0) then
        call mpi_type_create_indexed_block(nreceives, 1, &
          & halo_receives(halo, i) - lbound(halo%gnn_to_unn, 1), &
          & getpinteger(), receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Set up non-blocking communications
    allocate(requests(nprocs * 2))
    requests = MPI_REQUEST_NULL
    tag = next_mpi_tag()

    do i = 1, nprocs
      ! Non-blocking sends
      if(halo_send_count(halo, i) > 0) then
        call mpi_isend(halo%gnn_to_unn, 1, send_types(i), i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      ! Non-blocking receives
      if(halo_receive_count(halo, i) > 0) then
        call mpi_irecv(halo%gnn_to_unn, 1, receive_types(i), i - 1, tag, communicator, requests(i + nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

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

      if(receive_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do
    deallocate(send_types)
    deallocate(receive_types)

#else
    if(valid_serial_halo(halo)) then
      allocate(halo%owned_nodes_unn_base(halo_proc_count(halo)))
      halo%owned_nodes_unn_base = 0
      halo%my_owned_nodes_unn_base = 0
      halo%receives_gnn_to_unn_c = malloc(halo_all_receives_count(halo) * c_sizeof(1_c_int))
      call c_f_pointer(halo%receives_gnn_to_unn_c, halo%receives_gnn_to_unn, &
           [halo_all_receives_count(halo)])
    else
      FLAbort("Cannot create global to universal numbering without MPI support")
    end if
#endif

  end subroutine create_global_to_universal_numbering_order_general

  subroutine create_global_to_universal_numbering_order_trailing_receives&
       (halo, local_only)
    type(halo_type), intent(inout) :: halo
    logical, intent(in), optional :: local_only

#ifdef HAVE_MPI
    integer :: communicator, i, ierr, nowned_nodes, nprocs, rank
    integer, dimension(:), allocatable :: requests, statuses
    type(integer_vector), dimension(:), allocatable :: receives_unn, sends_unn
    integer tag

    ewrite(1,*) "Creating universal numbering for a trailing receives halo"
    assert(trailing_receives_consistent(halo))
    assert(halo_valid_for_communication(halo))

    nprocs = halo_proc_count(halo)
    communicator = halo_communicator(halo)
    rank = getrank(communicator)
    nowned_nodes = halo_nowned_nodes(halo)

    call set_universal_numbering_count(halo)

    ! Calculate the base universal node number for the owned nodes. The i th
    ! owned node then has universal node number equal to the base + i.
    call mpi_scan(nowned_nodes, halo%my_owned_nodes_unn_base, 1, getpinteger(), MPI_SUM, communicator, ierr)
    assert(ierr == MPI_SUCCESS)
    halo%my_owned_nodes_unn_base = halo%my_owned_nodes_unn_base - nowned_nodes
    ! gather this information from/to all other processors:
    allocate(halo%owned_nodes_unn_base(1:nprocs+1))
    call mpi_allgather(halo%my_owned_nodes_unn_base, 1, getpinteger(), &
      halo%owned_nodes_unn_base, 1, getpinteger(), communicator, ierr)
    assert(ierr == MPI_SUCCESS)
    assert( halo%owned_nodes_unn_base(rank+1)==halo%my_owned_nodes_unn_base )
    ! extra entry for convenience, such that number of owned nodes on a process
    ! can be derived from subtracting its unn_base from the next (similar to findrm):
    halo%owned_nodes_unn_base(nprocs+1)=universal_numbering_count(halo)
    assert( halo%owned_nodes_unn_base(nprocs)<=halo%owned_nodes_unn_base(nprocs+1) )

    ewrite(2, "(a,i0)") "Owned nodes universal node number base = ", &
      & halo%my_owned_nodes_unn_base
    ewrite(2, "(a,i0)") "Total receive_nodes = ", halo_all_receives_count(halo)
    halo%receives_gnn_to_unn_c = malloc((max_halo_node(halo) - halo%nowned_nodes) * c_sizeof(1_c_int))
    call c_f_pointer(halo%receives_gnn_to_unn_c, halo%receives_gnn_to_unn, &
         [max_halo_node(halo) - halo%nowned_nodes])

    if(present_and_true(local_only)) then
      halo%receives_gnn_to_unn = -1
      return
    end if

    ! Communicate the universal node numbers of the receive nodes across
    ! processes
    allocate(sends_unn(nprocs))
    do i = 1, nprocs
      allocate(sends_unn(i)%ptr(halo_send_count(halo, i)))
      sends_unn(i)%ptr = halo%my_owned_nodes_unn_base + halo_sends(halo, i)
      assert(all(sends_unn(i)%ptr > halo%my_owned_nodes_unn_base .and. sends_unn(i)%ptr <= halo%my_owned_nodes_unn_base + nowned_nodes))
    end do
    allocate(receives_unn(nprocs))
    allocate(requests(nprocs * 2))
    requests = MPI_REQUEST_NULL
    rank = getrank(communicator)
    tag = next_mpi_tag()
    do i = 1, nprocs
      allocate(receives_unn(i)%ptr(halo_receive_count(halo, i)))

      ! Non-blocking sends
      if(halo_send_count(halo, i) > 0) then
        call mpi_isend(sends_unn(i)%ptr, size(sends_unn(i)%ptr), getpinteger(), i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      ! Non-blocking receives
      if(halo_receive_count(halo, i) > 0) then
        call mpi_irecv(receives_unn(i)%ptr, halo_receive_count(halo, i),&
             & getpinteger(), i - 1, tag, communicator, requests(i +&
             & nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)

    deallocate(statuses)
    deallocate(requests)

    do i = 1, nprocs
      assert(all(receives_unn(i)%ptr <= halo%my_owned_nodes_unn_base .or. receives_unn(i)%ptr > halo%my_owned_nodes_unn_base + nowned_nodes))
      halo%receives_gnn_to_unn(halo_receives(halo, i) - nowned_nodes) = receives_unn(i)%ptr
      deallocate(sends_unn(i)%ptr)
      deallocate(receives_unn(i)%ptr)
    end do
    deallocate(sends_unn)
    deallocate(receives_unn)

#else
    if(valid_serial_halo(halo)) then
      allocate(halo%owned_nodes_unn_base(halo_proc_count(halo)))
      halo%owned_nodes_unn_base = 0
      halo%my_owned_nodes_unn_base = 0
      halo%receives_gnn_to_unn_c = malloc(halo_all_receives_count(halo) * c_sizeof(1_c_int))
      call c_f_pointer(halo%receives_gnn_to_unn_c, halo%receives_gnn_to_unn, &
           [halo_all_receives_count(halo)])
    else
      FLAbort("Cannot create global to universal numbering without MPI support")
    end if
#endif

  end subroutine create_global_to_universal_numbering_order_trailing_receives

  function valid_global_to_universal_numbering(halo) result(valid)
    !!< Return whether the global to universal numbering cache for the supplied
    !!< halo is valid

    type(halo_type), intent(in) :: halo

    logical :: valid

    integer, dimension(max_halo_node(halo)) :: unns

    call get_universal_numbering(halo, unns)
    valid = halo_verifies(halo, unns)

  end function valid_global_to_universal_numbering

  function has_global_to_universal_numbering(halo) result(has_gnn_to_unn)
    !!< Return whether the supplied halo has global to universal node numbering
    !!< data

    type(halo_type), intent(in) :: halo

    logical :: has_gnn_to_unn

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        has_gnn_to_unn = associated(halo%owned_nodes_unn_base)
#ifdef DDEBUG
        if(has_gnn_to_unn) then
          assert(associated(halo%gnn_to_unn))
          assert(halo%my_owned_nodes_unn_base>=0)
        end if
#endif
      case(HALO_ORDER_TRAILING_RECEIVES)
        has_gnn_to_unn = associated(halo%receives_gnn_to_unn)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end function has_global_to_universal_numbering

  pure function universal_numbering_count(halo) result(unn_count)
    !!< Return the global (universal) number of nodes

    type(halo_type), intent(in) :: halo

    integer :: unn_count

    unn_count = halo%unn_count

  end function universal_numbering_count

  subroutine set_universal_numbering_count(halo)
    !!< Set the universal numbering count for the supplied halo

    type(halo_type), intent(inout) :: halo

    halo%unn_count = halo_nowned_nodes(halo)
    call allsum(halo%unn_count, communicator = halo_communicator(halo))

  end subroutine set_universal_numbering_count

  function halo_universal_number(halo, global_number) result(unn)
    !!< For the supplied halo, return the corresponding universal node number
    !!< for the supplied global node number

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: global_number

    integer :: unn

    assert(global_number > 0)

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        unn = halo_universal_number_order_general(halo, global_number)
      case(HALO_ORDER_TRAILING_RECEIVES)
        unn = halo_universal_number_order_trailing_receives(halo, global_number)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end function halo_universal_number

  function halo_universal_number_vector(halo, global_number) result(unn)
    !!< Version of halo_universal_number which returns a vector of
    !!< universal numbers corresponding to the supplied vector of global
    !!< numbers.
    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(in) :: global_number

    integer, dimension(size(global_number)) :: unn

    integer :: i

    do i = 1, size(global_number)
       unn(i) = halo_universal_number(halo, global_number(i))
    end do

  end function halo_universal_number_vector

  function halo_universal_number_order_general(halo, global_number) result(unn)
    type(halo_type), intent(in) :: halo
    integer, intent(in) :: global_number

    integer :: unn

    assert(has_global_to_universal_numbering(halo))
    if (global_number<=size(halo%gnn_to_unn)) then
       unn = halo%gnn_to_unn(global_number)
    else
       unn = -1
    end if

  end function halo_universal_number_order_general

  function halo_universal_number_order_trailing_receives(halo, global_number) result(unn)
    type(halo_type), intent(in) :: halo
    integer, intent(in) :: global_number

    integer :: unn

    assert(has_global_to_universal_numbering(halo))


    if(global_number <= halo_nowned_nodes(halo)) then
      unn = halo%my_owned_nodes_unn_base + global_number
    else if(global_number - halo_nowned_nodes(halo) > size(halo%receives_gnn_to_unn)) then
      unn = - 1
    else
      unn = halo%receives_gnn_to_unn(global_number - halo_nowned_nodes(halo))
    end if

  end function halo_universal_number_order_trailing_receives

  function halo_universal_numbers(halo, global_numbers) result(unns)
    !!< For the supplied halo, return the corresponding universal node numbers
    !!< for the supplied global node numbers

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(in) :: global_numbers

    integer :: i
    integer, dimension(size(global_numbers)) :: unns

    do i = 1, size(global_numbers)
      unns(i) = halo_universal_number(halo, global_numbers(i))
    end do

  end function halo_universal_numbers

  subroutine get_universal_numbering(halo, unns)
    !!< For the supplied halo, retrieve the complete universal node numbering
    !!< list

    type(halo_type), intent(in) :: halo
    integer, dimension(node_count(halo)), intent(out) :: unns

    assert(has_global_to_universal_numbering(halo))

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        call get_universal_numbering_order_general(halo, unns)
      case(HALO_ORDER_TRAILING_RECEIVES)
        call get_universal_numbering_order_trailing_receives(halo, unns)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine get_universal_numbering

  subroutine get_universal_numbering_order_general(halo, unns)
    type(halo_type), intent(in) :: halo
    integer, dimension(node_count(halo)), intent(out) :: unns

    unns=halo%gnn_to_unn

  end subroutine get_universal_numbering_order_general

  subroutine get_universal_numbering_order_trailing_receives(halo, unns)
    type(halo_type), intent(in) :: halo
    integer, dimension(max_halo_node(halo)), intent(out) :: unns

    integer :: i

    assert(trailing_receives_consistent(halo))

    unns = -1
    do i = 1, halo_nowned_nodes(halo)
      unns(i) = halo%my_owned_nodes_unn_base + i
    end do
    unns(halo_nowned_nodes(halo) + 1:&
         halo_nowned_nodes(halo) + size(halo%receives_gnn_to_unn)) = &
         halo%receives_gnn_to_unn

  end subroutine get_universal_numbering_order_trailing_receives

  subroutine get_universal_numbering_multiple_components(halo, unns)
    !!< For the supplied halo, retrieve the complete universal numbering
    !!< of the degrees of freedom in a multi-component field,
    !!< in such a way that the universal ordering is:
    !!< - 1st component of all owned nodes on process 0
    !!< - 2nd component of all owned nodes on process 0
    !!< - ...
    !!< - n-th component of all owned nodes on process 0
    !!< - 1st component of all owned nodes on process 1
    !!< - ...
    !!< - n-th component of all owned nodes on the last process
    !!< The number of components n is determined by size(unns,2)

    type(halo_type), intent(in) :: halo
    integer, dimension(:,:), intent(out) :: unns

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        call get_unn_multiple_components_order_general(halo, unns)
      case(HALO_ORDER_TRAILING_RECEIVES)
        call get_unn_multiple_components_order_trailing_receives(halo, unns)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine get_universal_numbering_multiple_components

  subroutine get_unn_multiple_components_order_general(halo, unns)
    !!< For the supplied halo, retrieve the complete universal numbering
    !!< of the degrees of freedom in a multi-component field

    type(halo_type), intent(in) :: halo
    integer, dimension(:,:), intent(out) :: unns

    integer, dimension(:), pointer :: receives
    integer :: owned_nodes, ncomponents, out_unn, out_unn_base, remote_gnn
    integer :: i, j, k, unn

    assert(has_global_to_universal_numbering(halo))
    assert(size(unns,1)>=size(halo%gnn_to_unn))

    ncomponents = size(unns, 2)

    unns = -1

    ! first the receiving nodes
    do i = 1, halo_proc_count(halo)

      receives => halo_receives(halo, i)
      ! base for the created multi-component unns of owned nodes on proces i:
      out_unn_base = halo%owned_nodes_unn_base(i)*ncomponents
      ! nodes owned by process i:
      owned_nodes = halo%owned_nodes_unn_base(i+1)-halo%owned_nodes_unn_base(i)

      do j = 1, size(receives)

        unn = halo%gnn_to_unn(receives(j))
        ! global no as owned node on process i:
        remote_gnn = unn-halo%owned_nodes_unn_base(i)

        ! start with unn for component 1
        out_unn = out_unn_base+remote_gnn
        do k = 1, ncomponents
          unns(receives(j), k) = out_unn
          out_unn = out_unn + owned_nodes
        end do

      end do
    end do

    ! the rest must be owned nodes
    owned_nodes = halo_nowned_nodes(halo)
    do i=1, size(halo%gnn_to_unn)
      if (unns(i,1)==-1) then
        ! the multi-component unn should be using base unn_base*ncomponents
        ! so we add the missing bit:
        out_unn = halo%gnn_to_unn(i)+(ncomponents-1)*halo%my_owned_nodes_unn_base
        do k=1, ncomponents
          unns(i, k) = out_unn
          out_unn = out_unn + owned_nodes
        end do
      end if
    end do

  end subroutine get_unn_multiple_components_order_general

  subroutine get_unn_multiple_components_order_trailing_receives(halo, unns)
    !!< For the supplied halo, retrieve the complete universal numbering
    !!< of the degrees of freedom in a multi-component field

    type(halo_type), intent(in) :: halo
    integer, dimension(:,:), intent(out) :: unns

    integer, dimension(:), pointer :: receives
    integer :: owned_nodes, ncomponents, out_unn, out_unn_base, remote_gnn
    integer :: i, j, k, unn, my_nowned_nodes

    assert(trailing_receives_consistent(halo))
    assert(size(unns) >= halo_nowned_nodes(halo) + size(halo%receives_gnn_to_unn))
    assert(has_global_to_universal_numbering(halo))

    ncomponents = size(unns, 2)

    ! first our owned nodes
    my_nowned_nodes = halo_nowned_nodes(halo)
    do i = 1, my_nowned_nodes
      ! the multi-component unn uses a base of unn_base*ncomponents
      out_unn = ncomponents*halo%my_owned_nodes_unn_base + i
      do k = 1, ncomponents
        unns(i, k) = out_unn
        out_unn = out_unn + my_nowned_nodes
      end do
    end do

    ! then fill in the receiving nodes
    do i = 1, halo_proc_count(halo)
      receives => halo_receives(halo, i)
      ! base for the created multi-component unns of owned nodes on proces i:
      out_unn_base = halo%owned_nodes_unn_base(i)*ncomponents
      ! nodes owned by process i:
      owned_nodes = halo%owned_nodes_unn_base(i+1)-halo%owned_nodes_unn_base(i)
      do j = 1, size(receives)

        unn = halo%receives_gnn_to_unn(receives(j)-my_nowned_nodes)
        ! global no as owned node on process i:
        remote_gnn = unn-halo%owned_nodes_unn_base(i)

        ! start with unn for component 1
        out_unn = out_unn_base+remote_gnn
        do k = 1, ncomponents
          unns(receives(j), k) = out_unn
          out_unn = out_unn + owned_nodes
        end do

      end do
    end do

  end subroutine get_unn_multiple_components_order_trailing_receives

  subroutine get_universal_numbering_inverse(halo, gnns)
    !!< For the supplied halo, retrieve the complete universal node numbering
    !!< list inverse

    type(halo_type), intent(in) :: halo
    type(integer_hash_table), intent(out) :: gnns

    integer, dimension(:), allocatable :: unns

    allocate(unns(node_count(halo)))
    call get_universal_numbering(halo, unns)
    call invert_set(unns, gnns)
    deallocate(unns)

  end subroutine get_universal_numbering_inverse

  subroutine set_halo_universal_number(halo, node, universal_number, stat)
    !!< Set a single universal number in halo. This is useful for external
    !!< routines which set up universal numberings such as
    !!< reorder_halo_from_element_halo.
    !!<
    !!< The stat argument, if present, returns 1 if the node is outside the
    !!< range of the halo and 0 otherwise.
    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: node, universal_number
    integer, intent(out), optional :: stat

    assert(has_global_to_universal_numbering(halo))
    if (present(stat)) stat=0

    select case(halo_ordering_scheme(halo))
    case(HALO_ORDER_GENERAL)
       if (node>size(halo%gnn_to_unn)) then
          if (present(stat)) then
             stat=1
             return
          else
             FLAbort("Illegal node number in set_halo_universal_number")
          end if
       end if

       halo%gnn_to_unn(node)=universal_number

    case(HALO_ORDER_TRAILING_RECEIVES)

       if (node>halo%nowned_nodes)&
            & then
          if (node-halo%nowned_nodes>size(halo%receives_gnn_to_unn)) then
             if (present(stat)) then
                stat=1
                return
             else
                FLAbort("Illegal node number in set_halo_universal_number")
             end if
          end if

          halo%receives_gnn_to_unn(node-halo%nowned_nodes) &
               = universal_number
       else
          assert(universal_number==node+halo%my_owned_nodes_unn_base)
       end if

    case default
       FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine set_halo_universal_number

  subroutine ewrite_universal_numbers(halo, debug_level)
    !!< Print the universal number cache for this process with the supplied
    !!< debug level

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: debug_level

    assert(has_global_to_universal_numbering(halo))

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        assert(associated(halo%gnn_to_unn))

        ewrite(debug_level, *) "Global to universal numbering map:"
        ewrite(debug_level, *) halo%gnn_to_unn
      case(HALO_ORDER_TRAILING_RECEIVES)
        assert(associated(halo%receives_gnn_to_unn))

        ewrite(debug_level, *) "Owned nodes universal node number base = ", halo%my_owned_nodes_unn_base
        ewrite(debug_level, *) "Receives global to universal numbering map:"
        ewrite(debug_level, *) halo%receives_gnn_to_unn
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine ewrite_universal_numbers

end module halos_numbering
