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

module halos_debug

  use fldebug
  use halo_data_types
  use halos_base
  use mpi_interfaces
  use parallel_tools

  implicit none

  private

  public :: valid_serial_halo, pending_communication, valid_halo_communicator, &
    & valid_halo_node_counts, halo_valid_for_communication, &
    & trailing_receives_consistent, print_halo

  interface pending_communication
    module procedure pending_communication_halo
  end interface pending_communication

contains

  function valid_serial_halo(halo) result(valid)
    !!< Return whether the supplied halo is valid as a serial halo.

    type(halo_type), intent(in) :: halo
    logical :: valid
    valid = halo_proc_count(halo) <= 1 .and. halo_all_sends_count(halo) == 0 .and. halo_all_receives_count(halo) == 0

  end function valid_serial_halo

  function pending_communication_halo(halo) result(pending)
    !!< Return whether there is a pending communication for the supplied halo.

    type(halo_type), intent(in) :: halo

    logical :: pending

#ifdef HAVE_MPI
    assert(valid_halo_communicator(halo))
    pending = pending_communication(communicator = halo_communicator(halo))
#else
    pending = .false.
#endif

  end function pending_communication_halo

  function valid_halo_communicator(halo) result(valid)
    !!< Return whether the communicator for the supplied halo corresponds to
    !!< a valid MPI communicator and is consistent with the halo number of
    !!< processes - i.e. whether the halo communicator can be used for
    !!< communication on the halo.

    type(halo_type), intent(in) :: halo

    logical :: valid

    valid = valid_communicator(halo_communicator(halo)) .and. &
      & getnprocs(halo_communicator(halo)) == halo_proc_count(halo)

  end function valid_halo_communicator

  function valid_halo_node_counts(halo) result(valid)
    !!< Return whether the halo has consistent node sizes across processors. A
    !!< moderately expensive operation (involes communication of nprocs integers
    !!< from/to each processors).

    type(halo_type), intent(in) :: halo

    logical :: valid

#ifdef HAVE_MPI
    integer :: ierr, nprocs
    integer, dimension(:), allocatable :: communicated_nreceives, nreceives, nsends

    assert(valid_halo_communicator(halo))
    assert(.not. pending_communication(halo))

    nprocs = halo_proc_count(halo)

    ! Read nsends from the halo
    allocate(nsends(nprocs))
    call halo_send_counts(halo, nsends)

    ! Read nsends from other processes and assemble onto communcated_nreceives
    allocate(communicated_nreceives(nprocs))
    call mpi_alltoall(nsends, 1, getpinteger(), communicated_nreceives, 1, getpinteger(), halo_communicator(halo), ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(nsends)

    ! Check that the communicated nreceives is consistent with that that read
    ! from the halo
    allocate(nreceives(nprocs))
    call halo_receive_counts(halo, nreceives)

    valid = all(nreceives == communicated_nreceives)
    if(.not. valid) then
      ewrite(2, *) "Invalid halo node counts"
    end if

    deallocate(nreceives)
    deallocate(communicated_nreceives)
#else
    valid = valid_serial_halo(halo)
#endif

  end function valid_halo_node_counts

  function receive_nodes_unique(halo) result(unique)
    !!< Return whether the receive nodes in the supplied halo are unique

    type(halo_type), intent(in) :: halo

    logical :: unique

    unique = (halo_all_unique_receives_count(halo) == halo_all_receives_count(halo))

  end function receive_nodes_unique

  function halo_valid_for_communication(halo) result(valid)
    !!< Return whether the supplied halo is valid for data communication.

    type(halo_type), intent(in) :: halo

    logical :: valid

    if(.not. valid_halo_communicator(halo)) then
      ewrite(0, *) "Invalid communicator"
      valid = .false.
    else if(.not. valid_halo_node_counts(halo)) then
      ewrite(0, *) "Invalid halo node counts"
      valid = .false.
    else
      valid = .true.
    end if

  end function halo_valid_for_communication

  function trailing_receives_consistent(halo) result(consistent)
    !!< Return whether the supplied halo is consistent with trailing receives
    !!< ordering

    type(halo_type), intent(in) :: halo

    logical :: consistent

    ewrite(1, *) "Checking nodes in halo " // halo_name(halo) // " for consistency with trailing receive ordering"

    if(.not. has_nowned_nodes(halo)) then
      ewrite(1, *) "Owned nodes not set"
      consistent = .false.
    else if(max_halo_send_node(halo) > halo_nowned_nodes(halo)) then
      ewrite(1, *) "Not all send nodes are owned"
      consistent = .false.
    else if(halo_all_receives_count(halo) == 0) then
      consistent = .true.
    else if(min_halo_receive_node(halo) <= halo_nowned_nodes(halo)) then
      ewrite(1, *) "At least one receive node is owned"
      consistent = .false.
    else if(.not. receive_nodes_unique(halo)) then
      ewrite(1, *) "Receive nodes are not unique"
      consistent = .false.
!!$    else if(max_halo_receive_node(halo) /= node_count(halo)) then
!!$      ewrite(1, *) "Not all non-owned nodes are receive nodes"
!!$      consistent = .false.
    else
      consistent = .true.
    end if

    call alland(consistent, communicator = halo_communicator(halo))

    if(consistent) then
      ewrite(1, *) "Halo nodes are consistent with trailing receive ordering"
    else
      ewrite(1, *) "Halo nodes are not consistent with trailing receive ordering"
    end if

  end function trailing_receives_consistent

  subroutine print_halo(halo, priority)
    type(halo_type), intent(in) :: halo
    integer, intent(in) :: priority

    integer :: i

    ewrite(priority,*) "Halo name: ", halo_name(halo)
    ewrite(priority,*) "Owned nodes: ", halo_nowned_nodes(halo)

    do i = 1, halo_proc_count(halo)
       ewrite(priority,*) "Sends to process ", i
       ewrite(priority,*) halo_sends(halo, i)
    end do

    do i = 1, halo_proc_count(halo)
       ewrite(priority,*) "Receives from process ", i
       ewrite(priority,*) halo_receives(halo, i)
    end do

  end subroutine print_halo

end module halos_debug
