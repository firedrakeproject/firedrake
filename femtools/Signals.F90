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
#include "confdefs.h"

module signals
  ! This module sets up signal handling.
  use signal_vars
  implicit none

  logical, save, private :: initialised=.false.

#ifdef SIGNAL_HAVE_FLAG
  interface
     function signal(signum, proc, flag)
       integer :: signal
       integer, intent(in):: signum, flag
       interface
          function proc(signum)
            integer :: proc
            integer, intent(in) :: signum
          end function proc
       end interface
     end function signal
  end interface
!#else
!  interface
!     function signal(signum, proc)
!       integer :: signal
!       integer, intent(in):: signum
!       interface
!          function proc(signum)
!            integer :: proc
!            integer, intent(in) :: signum
!          end function proc
!       end interface
!     end function signal
!  end interface
#endif

contains

  subroutine initialise_signals
    ! Register the signal handlers.
    interface
       function handle_sigint(signum)
         integer :: handle_sigint
         integer, intent(in) :: signum
       end function handle_sigint
       function handle_sigterm(signum)
         integer :: handle_sigterm
         integer, intent(in) :: signum
       end function handle_sigterm
    end interface
    interface
       function handle_sighup(signum)
         integer :: handle_sighup
         integer, intent(in) :: signum
       end function handle_sighup
    end interface
    interface
       function handle_sigfpe(signum)
         integer :: handle_sigfpe
         integer, intent(in) :: signum
       end function handle_sigfpe
    end interface

    integer :: result

    ! SIGHUP support is still to come.
    ! call signal(SIGHUP, handle_sighup, -1)

#ifdef SIGNAL_HAVE_FLAG
    result= signal(SIGINT, handle_sigint, -1)
#else
    result= signal(SIGINT, handle_sigint)
#endif

#ifdef SIGNAL_HAVE_FLAG
    result= signal(SIGTERM, handle_sigterm, -1)
#else
    result= signal(SIGTERM, handle_sigterm)
#endif

    ! We don't check result because we don't know if it has the same
    ! meaning on all platforms.

!DEBUG
#ifdef SIGNAL_HAVE_FLAG
    result= signal(SIGFPE, handle_sigfpe, -1)
#else
    result= signal(SIGFPE, handle_sigfpe)
#endif

  end subroutine initialise_signals

end module signals

! Dummy procedure to enable compilers which don't support the signal
! function to compile without error.

#ifndef SIGNAL
function signal(signum, proc, flag)
  integer :: signal
  integer, intent(in):: signum, flag
  interface
     function proc(signum)
       integer :: proc
       integer, intent(in) :: signum
     end function proc
  end interface

  signal=0

end function signal
#endif
