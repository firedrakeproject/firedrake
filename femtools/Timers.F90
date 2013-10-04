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

module timers
!  !!< This module contains routines which time the fluidity run

  use fldebug
  use global_parameters, only : real_8
  use mpi_interfaces

  implicit none

  private

  public :: wall_time, wall_time_supported

contains

  function wall_time()
    ! This function returns the wall clock time from when the
    ! simulation started.
    !
    ! It must be called at the start of the simulation to get the clock
    ! running.
    real(kind = real_8):: wall_time
    logical, save :: started=.false.
#ifdef HAVE_MPI
    real(kind = real_8), save :: wall_time0

    wall_time = MPI_Wtime()
    if(.not.started) then
       wall_time0 = wall_time
       wall_time = 0.0
       started=.true.
    else
       wall_time = wall_time - wall_time0
    endif
#else
    integer, save :: clock0
    integer, save :: clock1,clockmax,clockrate, ticks
    real secs
    logical, save :: clock_support=.true.

    ! Initialize
    wall_time = -1.0

    ! Return -1.0 if no clock support
    IF(.not.clock_support) return

    IF(.not.started) THEN
       call system_clock(count_max=clockmax, count_rate=clockrate)
       call system_clock(clock0)

       IF(clockrate==0) THEN
          clock_support=.false.
          ewrite(-1, *) "No wall time support"
       else
          wall_time = 0.0
       ENDIF

       started=.true.
    ELSE
       call system_clock(clock1)
       ticks=clock1-clock0
       ! reset -ve numbers
       ticks=mod(ticks+clockmax, clockmax)
       secs=  real(ticks)/real(clockrate)
       wall_time=secs
    ENDIF
#endif
  end function wall_time

  function wall_time_supported() result(supported)
!    !!< Return whether wall time is supported

    logical :: supported

#ifdef HAVE_MPI
    supported = .true.
#else
    supported = (wall_time() >= 0.0)
#endif

  end function wall_time_supported

end module timers
