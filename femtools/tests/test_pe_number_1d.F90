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

subroutine test_pe_number_1d

  use diagnostic_fields
  use fields
  use fldebug
  use spud
  use state_module
  use unittest_tools
  use field_options

  implicit none

  integer :: stat
  type(element_type) :: shape
  type(quadrature_type) :: quad
  type(mesh_type) :: coordinate_mesh, velocity_mesh
  type(scalar_field) :: pe_no, phi
  type(state_type) :: state
  type(tensor_field) :: diffusivity
  type(vector_field) :: positions, velocity

  quad = make_quadrature(vertices = 2, dim  = 1, degree = 2)
  shape = make_element_shape(vertices = 2, dim  = 1, degree = 1, quad = quad)
  call deallocate(quad)

  call allocate(coordinate_mesh, nodes = 4, elements = 3, shape = shape, name = "CoordinateMesh")
  call deallocate(shape)

  call set_ele_nodes(coordinate_mesh, 1, (/1, 2/))
  call set_ele_nodes(coordinate_mesh, 2, (/2, 3/))
  call set_ele_nodes(coordinate_mesh, 3, (/3, 4/))

  velocity_mesh = piecewise_constant_mesh(coordinate_mesh, name = "PeNumberMesh")

  call allocate(positions, 1, coordinate_mesh, name = "Coordinate")
  call allocate(velocity, 1, velocity_mesh, name = "Velocity")
  call allocate(diffusivity, velocity_mesh, name = "PhiDiffusivity")
  call allocate(phi, velocity_mesh, name = "Phi")
  call allocate(pe_no, velocity_mesh, name = "PeNumber")

  call deallocate(coordinate_mesh)
  call deallocate(velocity_mesh)

  call set(positions, (/1, 2, 3, 4/), spread((/0.0, 1.0, 11.0, 111.0/), 1, 1))
  call set(velocity, (/1, 2, 3/), spread((/1.0, 1.0, 5.0/), 1, 1))

  call insert(state, positions, name = positions%name)
  call insert(state, velocity, name = velocity%name)
  call insert(state, diffusivity, name = diffusivity%name)
  call insert(state, phi, name = phi%name)
  call deallocate(positions)
  call deallocate(velocity)
  call deallocate(phi)

  pe_no%option_path = "/material_phase::Fluid/scalar_field::GridPecletNumber"
  call add_option(trim(pe_no%option_path) // "/diagnostic", stat = stat)
  assert(stat == SPUD_NEW_KEY_WARNING)
  call set_option(trim(complete_field_path(pe_no%option_path)) // "/field_name", "Phi", stat = stat)
  assert(stat == SPUD_NEW_KEY_WARNING)

  call set(diffusivity, (/1, 2, 3/), spread(spread((/1.0, 1.0, 1.0/), 1, 1), 1, 1))

  call calculate_diagnostic_variable(state, "GridPecletNumber", pe_no)
  call report_test("[pe no]", node_val(pe_no, (/1, 2, 3/)) .fne. (/1.0, 10.0, 500.0/), .false., "Incorrect pe number")

  call set(diffusivity, (/1, 2, 3/), spread(spread((/2.0, 5.0, 10.0/), 1, 1), 1, 1))

  call calculate_diagnostic_variable(state, "GridPecletNumber", pe_no)
  call report_test("[pe no]", node_val(pe_no, (/1, 2, 3/)) .fne. (/0.5, 2.0, 50.0/), .false., "Incorrect pe number")

  call deallocate(state)
  call deallocate(diffusivity)
  call deallocate(pe_no)

  call report_test_no_references()

end subroutine test_pe_number_1d
