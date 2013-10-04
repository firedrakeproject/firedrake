!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineeringp
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
module fields
  !!< This module contains abstracted field types which carry shape and
  !!< connectivity with them.

  ! this is a wrapper module providing the routines specified in the following modules:
  use fields_data_types   ! the derived types of the basic objects
  use fields_base         ! all basic enquiry functions and field evaluation at nodes and elements
  use fields_allocates    ! allocates, deallocates and all other routines creating field or mesh objects
  use fields_manipulation ! all routines that do operations on existing fields to change their values
  use fields_calculations ! all calculation routines that return values or complete new fields

  implicit none

end module fields
