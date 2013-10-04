!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    David Ham
!    Department of Computing
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
module fiat_interface
  !!< This module imports elements from FIAT.
  use elements
  use python_utils
  implicit none

  private

  public make_element_shape_fiat

  character(len=42), parameter :: fiat_type(8) = [&
       "lagrange                      ", &
       "nonconforming_not_implemented ", &
       "bubble                        ", &
       "control_volume_not_implemented",&
       "control_volume_not_implemented",&
       "control_volume_not_implemented",&
       "discontinuous lagrange        ",&
       "trace_not_implemented         "]

contains

  function make_element_shape_fiat(cell, type, degree) result (element)
    type(element_type) :: element
    type(cell_type), intent(in) :: cell
    integer, intent(in) :: type
    integer, intent(in) :: degree

    character(len=1024) :: python_string
    integer :: d, e, dof

    call python_run_string("import FIAT")

    write(python_string,"(5a,i0,a)") "element = FIAT.",&
         fiat_type(type), ", ",trim(cell%fiat_name), ", ", degree, ")"

    call python_run_string(python_string)

    ! Ensure deallocation doesn't fail.
    call addref(element)

    element%dim = cell%dimension
    element%ndof = python_fetch_integer("element.space_dimension()")
    element%ngi = 0 ! No quadrature
    element%degree = degree
    element%type = type

    allocate(element%entity2dofs(0:ubound(cell%entities,1),size(cell%entities,2)))

    do d = 0, element%dim
       write (python_string, "(a,i0,a)") "len(element.entity_dofs()[", &
            d, "])"

       do e = 1, python_fetch_integer(python_string)
          write(python_string, "(a,i0,a,i0,a)") &
               "len(element.entity_dofs()[", &
               d, "][", e, "])"

          allocate(element%entity2dofs(d,e)%dofs(&
               python_fetch_integer(python_string)))

          do dof = 1, python_fetch_integer(python_string)
             write(python_string, "(a,i0,a,i0,a,i0,a)") &
                  "element.entity_dofs()[", &
                  d, "][", e, "][", dof, "]"

             element%entity2dofs(d,e)%dofs(dof) = &
                  python_fetch_integer(python_string)
          end do
       end do

    end do

  end function make_element_shape_fiat


end module fiat_interface
