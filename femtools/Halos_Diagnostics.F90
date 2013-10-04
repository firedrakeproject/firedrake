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
module halos_diagnostics
  !!< this module contains routines for diagnosing halo problems.
  use vtk_interfaces
  use fields_data_types
  use fields_allocates
  use fields_base
  use shape_functions
  use halos_numbering
  use halo_data_types
  use fields_manipulation
  implicit none

  private
  public write_universal_numbering

contains

  subroutine write_universal_numbering(halo, mesh, position, name)
    !!< Dump a vtu file containing the universal numbering of halo on mesh.
    type(halo_type), intent(in) :: halo
    type(mesh_type), intent(inout) :: mesh
    type(vector_field), intent(in) :: position
    character(len=*), intent(in) :: name

    type(scalar_field) :: field
    type(mesh_type) :: lmesh
    type(element_type) :: shape
    integer :: node, ele

    assert(has_global_to_universal_numbering(halo))

    if (halo%data_type==HALO_TYPE_ELEMENT) then
       shape=make_element_shape(mesh%shape, degree=0)
       lmesh=make_mesh(position%mesh, shape=shape, continuity=-1)
       call allocate(field, lmesh, name="UniversalNumber")

       ! Drop excess mesh and shape references
       call deallocate(lmesh)
       call deallocate(shape)

       ! Note that for degree 0 shape functions, nodes==elements
       do ele=1, element_count(field)
          call set(field, ele, real(halo_universal_number(halo, ele)))
       end do

    else

       lmesh=mesh
       call allocate(field, mesh, name="UniversalNumber")

       ! Note that for degree 0 shape functions, nodes==elements
       do node=1, node_count(field)
          call set(field, node, real(halo_universal_number(halo, node)))
       end do

    end if

    call vtk_write_fields(name, position=position, model=lmesh,&
         & sfields=(/field/))

    call deallocate(field)

  end subroutine write_universal_numbering

end module halos_diagnostics
