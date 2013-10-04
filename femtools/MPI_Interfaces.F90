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

module mpi_interfaces
  !!< Interfaces for MPI routines

  implicit none

#ifdef HAVE_MPI
  include "mpif.h"

  interface
    subroutine mpi_barrier(communicator, ierr)
      implicit none
      integer, intent(in) :: communicator
      integer, intent(out) :: ierr
    end subroutine mpi_barrier

    subroutine mpi_comm_rank(communicator, rank, ierr)
      implicit none
      integer, intent(in) :: communicator
      integer, intent(out) :: rank
      integer, intent(out) :: ierr
    end subroutine mpi_comm_rank

    subroutine mpi_comm_size(communicator, size, ierr)
      implicit none
      integer, intent(in) :: communicator
      integer, intent(out) :: size
      integer, intent(out) :: ierr
    end subroutine mpi_comm_size

    subroutine mpi_comm_test_inter(communicator, inter_communicator, ierr)
      implicit none
      integer, intent(in) :: communicator
      logical, intent(out) :: inter_communicator
      integer, intent(out) :: ierr
    end subroutine mpi_comm_test_inter

    subroutine mpi_finalize(ierr)
      implicit none
      integer, intent(out) :: ierr
    end subroutine mpi_finalize

    subroutine mpi_init(ierr)
      implicit none
      integer, intent(out) :: ierr
    end subroutine mpi_init

    subroutine mpi_initialized(initialized, ierr)
      implicit none
      logical, intent(out) :: initialized
      integer, intent(out) :: ierr
    end subroutine mpi_initialized

    subroutine mpi_iprobe(source, tag, communicator, result, status, ierr)
      implicit none
      include "mpif.h"
      integer, intent(in) :: source
      integer, intent(in) :: tag
      integer, intent(in) :: communicator
      integer, intent(out) :: result
      integer, dimension(MPI_STATUS_SIZE), intent(out) :: status
      integer, intent(out) :: ierr
    end subroutine mpi_iprobe

    function mpi_tick()
      use global_parameters, only : real_8
      implicit none
      real(kind = real_8) :: mpi_tick
    end function mpi_tick

    subroutine mpi_type_commit(type, ierr)
      implicit none
      integer, intent(in) :: type
      integer, intent(out) :: ierr
    end subroutine mpi_type_commit

    subroutine mpi_type_indexed(displacements_size, entries_per_displacement, displacements, old_type, new_type, ierr)
      implicit none
      integer, intent(in) :: displacements_size
      integer, dimension(displacements_size), intent(in) :: entries_per_displacement
      integer, dimension(displacements_size), intent(in) :: displacements
      integer, intent(in) :: old_type
      integer, intent(out) :: new_type
      integer, intent(out) :: ierr
    end subroutine mpi_type_indexed

    subroutine mpi_type_free(type, ierr)
      implicit none
      integer, intent(in) :: type
      integer, intent(out) :: ierr
    end subroutine mpi_type_free

    subroutine mpi_type_vector(blocks, block_size, stride, old_type, new_type, ierr)
      implicit none
      integer, intent(in) :: blocks
      integer, intent(in) :: block_size
      integer, intent(in) :: stride
      integer, intent(in) :: old_type
      integer, intent(out) :: new_type
      integer, intent(out) :: ierr
    end subroutine mpi_type_vector
  end interface

  external :: mpi_allreduce
  !subroutine mpi_allreduce(send, receive, size, type, operation, communicator, ierr)
  !  implicit none
  !  integer, intent(in) :: size
  !  (type), dimension(size), intent(in) :: send
  !  (type), dimension(size), intent(out) :: receive
  !  integer, intent(in) :: type
  !  integer, intent(in) :: operation
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: ierr
  !end subroutine mpi_allreduce

  external :: mpi_alltoall
  !subroutine mpi_alltoall(send_buffer, send_buffer_size, send_type, receive_buffer, receive_buffer_size, receive_type, communicator, ierr)
  !  implicit none
  !  integer, intent(in) :: send_buffer_size
  !  integer, intent(in) :: receive_buffer_size
  !  (send_type), dimension(send_buffer_size), intent(in) :: send_buffer
  !  integer, intent(in) :: send_type
  !  (receive_type), dimension(receive_buffer_size), intent(out) :: receive_buffer
  !  integer, intent(in) :: receive_type
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: ierr
  !end subroutine mpi_alltoall

  external :: mpi_bcast
  !subroutine mpi_bcast(buffer, buffer_size, type, source, communicator, ierr)
  !  implicit none
  !  integer, intent(in) :: buffer_size
  !  (type), dimension(buffer_size), intent(inout) :: buffer
  !  integer, intent(in) :: type
  !  integer, intent(in) :: source
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: ierr
  !end subroutine mpi_bcast

  external :: mpi_gather
  !subroutine mpi_gather(send_buffer, send_buffer_size, send_type, receive_buffer, receive_buffer_size, receive_type, source, communicator, ierr)
  !  implicit none
  !  integer, intent(in) :: send_buffer_size
  !  (send_type), dimension(send_buffer_size), intent(in) :: send_buffer
  !  integer, intent(in) :: send_type
  !  (receive_type), dimension(*), intent(out) :: receive_buffer
  !  integer, intent(in) :: receive_buffer_size
  !  integer, intent(in) :: receive_type
  !  integer, intent(in) :: source
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: ierr
  !end subroutine mpi_gather

  external :: mpi_irecv
  !subroutine mpi_irecv(buffer, buffer_size, type, source, tag, communicator, request, ierr)
  !  implicit none
  !  integer, intent(in) :: buffer_size
  !  (type), dimension(buffer_size), intent(out) :: buffer
  !  integer, intent(in) :: type
  !  integer, intent(in) :: source
  !  integer, intent(in) :: tag
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: request
  !  integer, intent(out) :: ierr
  !end subroutine mpi_irecv

  external :: mpi_isend
  !subroutine mpi_isend(buffer, buffer_size, type, destination, tag, communicator, request, ierr)
  !  implicit none
  !  integer, intent(in) :: buffer_size
  !  (type), dimension(buffer_size), intent(out) :: buffer
  !  integer, intent(in) :: type
  !  integer, intent(in) :: destination
  !  integer, intent(in) :: tag
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: request
  !  integer, intent(out) :: ierr
  !end subroutine mpi_isend

  external :: mpi_scan
  !subroutine mpi_scan(send_buffer, receive_buffer, send_buffer_size, type, operation, communicator, ierr)
  !  implicit none
  !  integer, intent(in) :: send_buffer_size
  !  (type), dimension(send_buffer_size), intent(in) :: send_buffer
  !  (type), intent(out) :: receive_buffer
  !  integer, intent(in) :: type
  !  integer, intent(in) :: operation
  !  integer, intent(in) :: communicator
  !  integer, intent(out) :: ierr
  !end subroutine mpi_scan

  external :: mpi_waitall
  !subroutine mpi_waitall(requests_size, requests, statuses, ierr)
  !  implicit none
  !  include "mpif.h"
  !  integer, intent(in) :: requests_size
  !  integer, dimension(requests_size), intent(inout) :: requests
  !  integer, dimension(requests_size * MPI_STATUS_SIZE), intent(out) :: statuses
  !  integer, intent(out) :: ierr
  !end subroutine mpi_waitall

  ! It seems that mpif.h can declare this external
  !external :: mpi_wtime
  !function mpi_wtime
  !  implicit none
  !  real :: mpi_wtime
  !end function mpi_wtime

#ifdef HAVE_MPI2
  interface
    subroutine mpi_type_create_indexed_block(displacements_size, entries_per_displacement, displacements, old_type, new_type, ierr)
      implicit none
      integer, intent(in) :: displacements_size
      integer, intent(in) :: entries_per_displacement
      integer, dimension(displacements_size), intent(in) :: displacements
      integer, intent(in) :: old_type
      integer, intent(out) :: new_type
      integer, intent(out) :: ierr
    end subroutine mpi_type_create_indexed_block
  end interface
#endif

! Note: Make sure the contains statement is only seen if the contains part is
! non-empty (empty contains part is Fortran 2008)
#ifndef HAVE_MPI2
contains

  subroutine mpi_type_create_indexed_block(displacements_size, &
    & entries_per_displacement, displacements, old_type, new_type, ierr)
    integer, intent(in) :: displacements_size
    integer, intent(in) :: entries_per_displacement
    integer, dimension(displacements_size), intent(in) :: displacements
    integer, intent(in) :: old_type
    integer, intent(out) :: new_type
    integer, intent(out) :: ierr

    call mpi_type_indexed(displacements_size, &
      & spread(entries_per_displacement, 1, displacements_size), &
      & displacements, old_type, new_type, ierr)

  end subroutine mpi_type_create_indexed_block
#endif
#endif

end module mpi_interfaces
