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

module eventcounter
  !---------------------------------------------------------------------
  !
  !!  This module is used to keep a record of events (eg. how often a task was performed)
  !
  !---------------------------------------------------------------------

  use fldebug

  implicit none

  private

  ! List of identifiers
  integer, parameter, public ::EVENT_ADAPTIVITY=1
  integer, parameter, public ::EVENT_MESH_MOVEMENT=2
  integer, parameter :: MAXIMUM_NUMBER_OF_EVENTS=2
  ! Data arrays
  integer, save :: events(MAXIMUM_NUMBER_OF_EVENTS) = 0

  public :: incrementeventcounter, geteventcounter, seteventcounter, eventcount

contains

  subroutine incrementeventcounter(event)
    integer, intent(in)::event

    assert(event > 0 .and. event <= MAXIMUM_NUMBER_OF_EVENTS)
    events(event) = events(event) + 1

  end subroutine incrementeventcounter

  subroutine geteventcounter(event, cnt)
    integer, intent(in)::event
    integer, intent(out)::cnt

    assert(event > 0 .and. event <= MAXIMUM_NUMBER_OF_EVENTS)
    cnt = events(event)

  end subroutine geteventcounter

  subroutine seteventcounter(event, cnt)
    integer, intent(in)::event, cnt

    assert(event > 0 .and. event <= MAXIMUM_NUMBER_OF_EVENTS)
    assert(cnt > 0)
    events(event) = cnt

  end subroutine seteventcounter

  function eventcount(event) result(cnt)
    integer, intent(in) :: event

    integer :: cnt

    call geteventcounter(event, cnt)

  end function eventcount

end module eventcounter

