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

module pickers_base

  use fldebug
  use picker_data_types

  implicit none

  private

  public :: picker_name, set_picker_name

contains

  pure function picker_name(picker)
    !!< Return the name of the supplied picker

    type(picker_type), intent(in) :: picker

    character(len = len_trim(picker%name)) :: picker_name

    picker_name = picker%name

  end function picker_name

  subroutine set_picker_name(picker, name)
    !!< Set the name of the supplied picker

    type(picker_type), intent(inout) :: picker
    character(len = *), intent(in) :: name

    picker%name = name

  end subroutine set_picker_name

end module pickers_base
