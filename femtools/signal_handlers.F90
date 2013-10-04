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

! This file contains signal handlers. These are required to be external
! functions by the signal function
#include "fdebug.h"

function handle_sighup(signal)
  use signal_vars

  implicit none
  integer :: handle_sighup
  integer, intent(in) :: signal

  sig_hup=.true.

  handle_sighup=0

end function handle_sighup


function handle_sigint(signal)
  use signal_vars

  implicit none
  integer :: handle_sigint
  integer, intent(in) :: signal

  sig_int=.true.

  handle_sigint=0

end function handle_sigint

function handle_sigterm(signal)
  use signal_vars

  implicit none
  integer :: handle_sigterm
  integer, intent(in) :: signal

  sig_int=.true.

  handle_sigterm=0

end function handle_sigterm

function handle_sigfpe(signal)
  use FLDebug
  use signal_vars

  implicit none
  integer :: handle_sigfpe
  integer, intent(in) :: signal

  handle_sigfpe = 0

  FLAbort("Floating point exception")

  handle_sigfpe=0

end function handle_sigfpe
