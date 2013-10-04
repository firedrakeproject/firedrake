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

module tictoc

  use fldebug
  use mpi_interfaces
  use parallel_tools
  use timers

  implicit none

  private

  public :: tic, toc, tictoc_reset, tictoc_clear, tictoc_time, &
    & tictoc_imbalance, tictoc_report

  integer, parameter :: MAX_TIC_ID = 1024
  integer, parameter, public :: TICTOC_ID_SIMULATION = 1, &
    & TICTOC_ID_SERIAL_ADAPT = 2, TICTOC_ID_DATA_MIGRATION = 3, &
    & TICTOC_ID_ADAPT = 4, TICTOC_ID_IO_READ = 5, TICTOC_ID_DATA_REMAP = 6, &
    & TICTOC_ID_INTERPOLATION = 7, TICTOC_ID_ASSEMBLE_METRIC = 8

  real :: starttime(MAX_TIC_ID) = 0.0, totaltime(MAX_TIC_ID) = 0.0
#ifdef DDEBUG
  logical :: timer_running(MAX_TIC_ID) = .false.
#endif

contains

  subroutine tic(id)
    integer, intent(in) :: id

    assert(id > 0)
    assert(id <= MAX_TIC_ID)

    starttime(id) = wall_time()

#ifdef DDEBUG
    timer_running(id) = .true.
#endif

  end subroutine tic

  subroutine toc(id)
    integer, intent(in) :: id

    real :: finish_time

    assert(id > 0)
    assert(id <= MAX_TIC_ID)
#ifdef DDEBUG
    assert(timer_running(id))
#endif

    finish_time = wall_time()
    totaltime(id) = totaltime(id) + (finish_time - starttime(id))

#ifdef DDEBUG
    timer_running(id) = .false.
#endif

  end subroutine toc

  subroutine tictoc_reset()
    starttime = 0.0
    totaltime = 0.0
  end subroutine tictoc_reset

  subroutine tictoc_clear(id)
    integer, intent(in) :: id

    assert(id > 0)
    assert(id <= MAX_TIC_ID)

    starttime(id) = 0.0
    totaltime(id) = 0.0

  end subroutine tictoc_clear

  real function tictoc_time(id)
    integer, intent(in) :: id

    assert(id > 0)
    assert(id <= MAX_TIC_ID)

    tictoc_time = totaltime(id)

  end function tictoc_time


  real function tictoc_imbalance(id)
    integer, intent(in) :: id
#ifdef HAVE_MPI
    real :: dt, max_time, mean_time
    real, dimension(:), allocatable :: times
    integer :: i, nprocs, rank, ierr
#endif

    assert(id > 0)
    assert(id <= MAX_TIC_ID)

    tictoc_imbalance = 0.0

#ifdef HAVE_MPI
    if(isparallel()) then
      dt = tictoc_time(id)
      nprocs = getnprocs()
      rank = getrank()
      allocate(times(nprocs))
      call MPI_Gather(dt, 1, getpreal(), times, 1, getpreal(), 0, MPI_COMM_FEMTOOLS, ierr)
      assert(ierr == MPI_SUCCESS)

      if(rank == 0) then
         mean_time = times(1)
         max_time = times(1)

         do i = 2, nprocs
            mean_time = mean_time + times(i)
            max_time = max(max_time, times(i))
         end do

         mean_time = mean_time / nprocs

         tictoc_imbalance = (max_time - mean_time) / mean_time
      end if

      call MPI_BCast(tictoc_imbalance, 1, getpreal(), 0, MPI_COMM_FEMTOOLS, ierr)
      assert(ierr == MPI_SUCCESS)

      deallocate(times)
   end if
#endif

  end function tictoc_imbalance

  subroutine tictoc_report(debug_level, id)

    integer, intent(in) :: debug_level
    integer, intent(in) :: id

    real :: imbalance, max_time, min_time, time

    assert(id > 0)
    assert(id <= MAX_TIC_ID)

    if(debug_level > current_debug_level) return

    time = tictoc_time(id)

    if(isparallel()) then
      min_time = time
      max_time = time
      call allmin(min_time)
      call allmax(max_time)
      imbalance = tictoc_imbalance(id)
    end if

    if(getrank() == 0) then
       select case(id)
         case(TICTOC_ID_SIMULATION)
           ewrite(debug_level, *) "For TICTOC_ID_SIMULATION"
         case(TICTOC_ID_SERIAL_ADAPT)
           ewrite(debug_level, *) "For TICTOC_ID_SERIAL_ADAPT"
         case(TICTOC_ID_DATA_MIGRATION)
           ewrite(debug_level, *) "For TICTOC_DATA_MIGRATION"
         case(TICTOC_ID_DATA_REMAP)
           ewrite(debug_level, *) "For TICTOC_DATA_REMAP"
         case(TICTOC_ID_ADAPT)
           ewrite(debug_level, *) "For TICTOC_ID_ADAPT"
         case(TICTOC_ID_IO_READ)
           ewrite(debug_level, *) "For TICTOC_ID_IO_READ"
         case(TICTOC_ID_INTERPOLATION)
           ewrite(debug_level, *) "For TICTOC_ID_INTERPOLATION"
         case(TICTOC_ID_ASSEMBLE_METRIC)
           ewrite(debug_level, *) "For TICTOC_ID_ASSEMBLE_METRIC"
         case default
           ewrite(debug_level, "(a,i0)") "For tictoc ID: ", id
       end select

       if(isparallel()) then
         ewrite(debug_level, *) "Time (process 0) = ", time
         ewrite(debug_level, *) "Min. time = ", min_time
         ewrite(debug_level, *) "Max. time = ", max_time
         ewrite(debug_level, *) "Imbalance = ", imbalance
       else
         ewrite(debug_level, *) "Time = ", time
       end if
    end if

  end subroutine tictoc_report

end module tictoc
