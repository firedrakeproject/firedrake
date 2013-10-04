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

subroutine test_vtk_read_surface

  use fields
  use fldebug
  use state_module
  use unittest_tools
  use vtk_interfaces

  !use read_triangle

  implicit none

  type(mesh_type), pointer :: mesh
  type(state_type) :: state

  !type(vector_field), target :: coordinate

  call vtk_read_state("data/tet.vtu", state = state)
  mesh => extract_mesh(state, "Mesh")
  !coordinate = read_triangle_files("data/tet", quad_degree = 1)
  !mesh => coordinate%mesh

  call report_test("[node_count]", node_count(mesh) /= 4, .false., "Incorrect element count")
  call report_test("[ele_count]", ele_count(mesh) /= 1, .false., "Incorrect element count")
  call report_test("[surface_element_count]", surface_element_count(mesh) /= 4, .false., "Incorrect element count")

  call deallocate(state)
  !call deallocate(coordinate)

  call report_test_no_references()

end subroutine test_vtk_read_surface
