!    Copyright (C) 2007 Imperial College London and others.
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
module generic_interface
  !!< This module provides routines for setting fields and boundary
  !!< conditions using generic functions provided in strings.
  use fields
  implicit none

  interface set_from_generic_function
     module procedure set_from_generic_function_scalar,&
          & set_values_from_generic_scalar
  end interface

  interface
     subroutine set_from_external_function_scalar(function, function_len,&
          & nodes, x, y, z, result, stat)
       !! Interface to c wrapper function.
       integer, intent(in) :: function_len
       character(len=function_len) :: function
       integer, intent(in) :: nodes
       real, dimension(nodes), intent(in) :: x, y, z
       real, dimension(nodes), intent(out) :: result
       integer, intent(out) :: stat
     end subroutine set_from_external_function_scalar
  end interface

contains

  subroutine set_from_generic_function_scalar(field, func, position)
    !!< Set the values at the nodes of field using the generic function
    !!< specified in the string func. The position field is used to
    !!< determine the locations of the nodes.
    type(scalar_field), intent(inout) :: field
    !! Func is the string to execute on the command line to start the
    !! generic function.
    character(len=*), intent(in) :: func
    type(vector_field), intent(in), target :: position

    type(vector_field) :: lposition
    real, dimension(:), pointer :: x, y, z
    integer :: stat, dim

    dim=mesh_dim(position)

    if (dim/=3) then
       FLExit("Generic functions are only supported for 3d scalar fields")
    end if

    if (field%mesh==position%mesh) then
       x=>position%val(1,:)

       if (dim>1) then
          y=>position%val(2,:)

          if (dim>2) then
             z=>position%val(3,:)
          end if
       end if
    else
       ! Remap position first.
       call allocate(lposition, dim, field%mesh, "Local Position")
       call remap_field(position, lposition)

       x=>lposition%val(1,:)

       if (dim>1) then
          y=>lposition%val(2,:)

          if (dim>2) then
             z=>lposition%val(3,:)
          end if
       end if
    end if

    call set_from_external_function_scalar(func, len(func), &
            & node_count(field), x, y, z, field%val, stat)

    if (stat/=0) then
       ewrite(0,*) "Generic error, function was:"
       ewrite(0,*) func
       FLExit("Dying")
    end if

    if (has_references(lposition)) then
       call deallocate(lposition)
    end if

  end subroutine set_from_generic_function_scalar

  subroutine set_values_from_generic_scalar(values, func, x, y, z)
    !!< Given a list of positions evaluate the generic function
    !!< specified in the string func at those points.
    real, dimension(:), intent(inout) :: values
    !! Func is the string to execute on the command line to start the
    !! generic function.
    character(len=*), intent(in) :: func
    real, dimension(size(values)), target :: x
    real, dimension(size(values)), optional, target :: y
    real, dimension(size(values)), optional, target :: z

    real, dimension(:), pointer :: lx, ly, lz
    integer :: stat, dim

    if (dim/=3) then
       FLExit("Generic functions are only supported for 3d scalar fields")
    end if

    lx=>x
    ly=>y
    lz=>z

    call set_from_external_function_scalar(func, len(func), &
            & size(values), lx, ly, lz, values, stat)

    if (stat/=0) then
       ewrite(0,*) "Generic error, function was:"
       ewrite(0,*) func
       FLExit("Dying")
    end if

  end subroutine set_values_from_generic_scalar

end module generic_interface
