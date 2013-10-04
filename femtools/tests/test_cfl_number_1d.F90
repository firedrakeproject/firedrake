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

subroutine test_cfl_number_1d

  use diagnostic_fields
  use fields
  use fldebug
  use state_module
  use unittest_tools

  implicit none

  type(element_type) :: shape
  type(quadrature_type) :: quad
  type(mesh_type) :: coordinate_mesh, velocity_mesh
  type(scalar_field) :: cfl_no
  type(state_type) :: state
  type(vector_field) :: positions, velocity

  quad = make_quadrature(vertices = 2, dim  = 1, degree = 2)
  shape = make_element_shape(vertices = 2, dim  = 1, degree = 1, quad = quad)
  call deallocate(quad)

  call allocate(coordinate_mesh, nodes = 4, elements = 3, shape = shape, name = "CoordinateMesh")
  call deallocate(shape)

  call set_ele_nodes(coordinate_mesh, 1, (/1, 2/))
  call set_ele_nodes(coordinate_mesh, 2, (/2, 3/))
  call set_ele_nodes(coordinate_mesh, 3, (/3, 4/))

  velocity_mesh = piecewise_constant_mesh(coordinate_mesh, name = "CFLNumberMesh")

  call allocate(positions, 1, coordinate_mesh, name = "Coordinate")
  call allocate(velocity, 1, velocity_mesh, name = "Velocity")
  call allocate(cfl_no, velocity_mesh, name = "CFLNumber")

  call deallocate(coordinate_mesh)
  call deallocate(velocity_mesh)

  call set(positions, (/1, 2, 3, 4/), spread((/0.0, 1.0, 11.0, 111.0/), 1, 1))
  call set(velocity, (/1, 2, 3/), spread((/1.0, 1.0, 5.0/), 1, 1))

  call insert(state, positions, name = positions%name)
  call insert(state, velocity, name = velocity%name)
  call deallocate(positions)
  call deallocate(velocity)

  call calculate_cfl_number(state, cfl_no, dt = 1.0)
  call report_test("[cfl no]", node_val(cfl_no, (/1, 2, 3/)) .fne. (/1.0, 0.1, 0.05/), .false., "Incorrect CFL number")

  call calculate_cfl_number(state, cfl_no, dt = 10.0)
  call report_test("[cfl no]", node_val(cfl_no, (/1, 2, 3/)) .fne. (/10.0, 1.0, 0.5/), .false., "Incorrect CFL number")

  call deallocate(state)
  call deallocate(cfl_no)

  call report_test_no_references()

end subroutine test_cfl_number_1d
