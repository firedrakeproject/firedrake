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

module picker_data_types

  use fldebug
  use global_parameters, only : FIELD_NAME_LEN
  use reference_counting

  implicit none

  private

  public :: picker_type, picker_ptr

  !! Picker (spatial index interface) information
  type picker_type
    !! Name of this picker
    character(len = FIELD_NAME_LEN) :: name
    !! Reference count for picker
    type(refcount_type), pointer :: refcount => null()
    !! Node owner finder ID for this picker
    integer :: picker_id = 0
    !! Last mesh movement event - used to keep track of when a new picker must
    !! be generated
    integer :: last_mesh_movement = 0
  end type picker_type

  type picker_ptr
    type(picker_type), pointer :: ptr => null()
  end type picker_ptr

end module picker_data_types
