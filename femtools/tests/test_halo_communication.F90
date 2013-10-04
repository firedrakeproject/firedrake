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

subroutine test_halo_communication
  !!< Test communication using halo_type derived type. Parallel test - requires
  !!< exactly two processes.

#ifdef HAVE_MPI
  use futils
  use halos
  use mpi_interfaces
  use parallel_tools
  use unittest_tools

  implicit none

  integer :: i, ierr, nprocs, procno
  integer, dimension(2) :: nreceives, nsends
  integer, dimension(7) :: integer_data
  integer :: communicator = MPI_COMM_FEMTOOLS
  logical :: fail
  real, dimension(7) :: real_data
  type(halo_type) :: halo

  call mpi_comm_size(communicator, nprocs, ierr)
  call report_test("[mpi_comm_size]", ierr /= MPI_SUCCESS, .false., "Failed to read communicator size")
  call report_test("[2 processes]", nprocs /= 2, .false., "Incorrect number of processes")

  procno = getprocno(communicator)

  ! Construct a halo
  if(procno == 1) then
    nsends(1) = 0
    nsends(2) = 2
    nreceives(1) = 0
    nreceives(2) = 3

    call allocate(halo, nsends, nreceives, communicator = communicator, name = "TestHalo")

    call zero(halo)
    call set_halo_send(halo, 2, 1, 2)
    call set_halo_send(halo, 2, 2, 6)
    call set_halo_receive(halo, 2, 1, 3)
    call set_halo_receive(halo, 2, 2, 5)
    call set_halo_receive(halo, 2, 3, 4)
  else
    nsends(1) = 3
    nsends(2) = 0
    nreceives(1) = 2
    nreceives(2) = 0

    call allocate(halo, nsends, nreceives, communicator = communicator, name = "TestHalo")

    call zero(halo)
    call set_halo_send(halo, 1, 1, 3)
    call set_halo_send(halo, 1, 2, 5)
    call set_halo_send(halo, 1, 3, 4)
    call set_halo_receive(halo, 1, 1, 2)
    call set_halo_receive(halo, 1, 2, 6)
  end if

  call report_test("[valid_halo_communicator]", .not. valid_halo_communicator(halo), .false., "Invalid halo communicator")
  call report_test("[valid_halo_node_counts]", .not. valid_halo_node_counts(halo), .false., "Invalid halo node counts")
  call report_test("[halo_valid_for_communication]", .not. halo_valid_for_communication(halo), .false., "Halo not valid for communication")

  ! Construct integer data to send/receive
  if(procno == 1) then
    integer_data(1) = 1
    integer_data(2) = 2
    integer_data(3) = -3
    integer_data(4) = -4
    integer_data(5) = -5
    integer_data(6) = 6
    integer_data(7) = 7
  else
    integer_data(1) = 1
    integer_data(2) = -2
    integer_data(3) = 3
    integer_data(4) = 4
    integer_data(5) = 5
    integer_data(6) = -6
    integer_data(7) = 7
  end if

  call halo_update(halo, integer_data)

  fail = .false.
  do i = 1, 5
    if(integer_data(i) /= i) then
      fail = .true.
      exit
    end if
  end do

  call report_test("[Integer array halo communication]", fail, .false., "Error in halo communication")

  ! Construct real data to send/receive
  if(procno == 1) then
    real_data(1) = 1.0
    real_data(2) = 2.0
    real_data(3) = -3.0
    real_data(4) = -4.0
    real_data(5) = -5.0
    real_data(6) = 6.0
    real_data(7) = 7.0
  else
    real_data(1) = 1.0
    real_data(2) = -2.0
    real_data(3) = 3.0
    real_data(4) = 4.0
    real_data(5) = 5.0
    real_data(6) = -6.0
    real_data(7) = 7.0
  end if

  call halo_update(halo, real_data)

  fail = .false.
  do i = 1, 5
    if(real_data(i) .fne. float(i)) then
      fail = .true.
      exit
    end if
  end do

  call report_test("[Real array halo communication]", fail, .false., "Error in halo communication")

  call report_test("[No pending communications]", pending_communication(halo), .false., "Pending communications")

  call deallocate(halo)

  call report_test_no_references()

#else
  use unittest_tools

  implicit none

  call report_test("[test disabled]", .false., .true., "Test compiled without MPI support")
#endif

end subroutine test_halo_communication
