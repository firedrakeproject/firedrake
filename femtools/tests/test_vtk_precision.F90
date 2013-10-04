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

subroutine test_vtk_precision
  !!< Test the precision of VTK I/O

  use elements
  use fields
  use fields_data_types
  use global_parameters, only : real_4, real_8
  use state_module
  use unittest_tools
  use vtk_interfaces

  implicit none

  character(len = 255) :: filename
  integer :: i, stat
  integer, parameter :: D = real_8, S = real_4
  type(element_type) :: shape
  type(mesh_type) :: mesh
  type(quadrature_type) :: quad
  type(scalar_field) :: written_s_field
  type(scalar_field), pointer :: read_s_field
  type(state_type) :: read_state, written_state
  type(vector_field) :: mesh_field, written_v_field
  type(vector_field), pointer :: read_v_field

  filename = "data/test_vtk_precision_out.vtu"

  ! Allocate a mesh
  quad = make_quadrature(vertices = 3, dim  = 2, degree = 1)
  shape = make_element_shape(vertices = 3, dim  = 2, degree = 1, quad = quad)
  call allocate(mesh, nodes = 3, elements = 1, shape = shape, name = "CoordinateMesh")
  call allocate(mesh_field, mesh_dim(mesh), mesh, "Coordinate")

  ! Create a single triangle mesh
  do i = 1, size(mesh%ndglno)
    mesh%ndglno(i) = i
  end do
  call set(mesh_field, 1, (/0.0, 0.0/))
  call set(mesh_field, 2, (/1.0, 0.0/))
  call set(mesh_field, 3, (/1.0, 1.0/))

  call deallocate(quad)
  call deallocate(shape)
  call deallocate(mesh)

  call insert(written_state, mesh_field%mesh, "CoordinateMesh")
  call insert(written_state, mesh_field, "Coordinate")

  ! Insert empty fields into the state to be written
  call allocate(written_s_field,  mesh_field%mesh, "TestScalarField", field_type = FIELD_TYPE_CONSTANT)
  call zero(written_s_field)
  call insert(written_state, written_s_field, "TestScalarField")
  call allocate(written_v_field, mesh_dim(mesh_field%mesh), mesh_field%mesh, "TestVectorField", field_type = FIELD_TYPE_CONSTANT)
  call zero(written_v_field)
  call insert(written_state, written_v_field, "TestVectorField")

  ! Clean existing output
  call vtk_write_state(filename, model = "CoordinateMesh", state = (/written_state/), stat = stat)
  call report_test("[Clean]", stat /= 0, .false., "Failed to clean output")
  call vtk_read_state(filename, state = read_state)
  read_v_field => extract_vector_field(read_state, "Coordinate")
  call report_test("[Clean]", abs(read_v_field%val(1,1)) > 0.0, .false., "[Failed to clean output]")
  read_s_field => extract_scalar_field(read_state, "TestScalarField")
  call report_test("[Clean]", any(abs(read_s_field%val) > 0.0), .false., "[Failed to clean output]")
  read_v_field => extract_vector_field(read_state, "TestVectorField")
  do i = 1, mesh_dim(mesh_field)
    call report_test("[Clean]", any(abs(read_v_field%val(1,:)) > 0.0), .false., "[Failed to clean output]")
  end do
  call deallocate(read_state)
  nullify(read_s_field)
  nullify(read_v_field)

  call set(mesh_field, 1, real((/tiny(0.0_S) * 1.0_S, 0.0_S/)))
  call set(written_s_field, real(tiny(0.0_S) * 1.0_S))
  call set(written_v_field, real((/tiny(0.0_S) * 1.0_S, 0.0_S/)))
  call vtk_write_state(filename, model = "CoordinateMesh", state = (/written_state/), stat = stat)
  call report_test("[vtk_write_state]", stat /= 0, .false., "Failed to write state")

  call vtk_read_state(filename, state = read_state)
  read_v_field => extract_vector_field(read_state, "Coordinate")
  call report_test("[Coordinate field, tiny, real_4 precision]", read_v_field%val(1,1) < tiny(0.0_S), .false., "[Insufficient precision]")
  read_s_field => extract_scalar_field(read_state, "TestScalarField")
  read_v_field => extract_vector_field(read_state, "TestVectorField")
  call report_test("[Scalar field, tiny, real_4 precision]", any(read_s_field%val < tiny(0.0_S)), .false., "[Insufficient precision]")
  call report_test("[Vector field, tiny, real_4 precision]", any(read_v_field%val(1,:) < tiny(0.0_S)), .false., "[Insufficient precision]")
  call deallocate(read_state)
  nullify(read_s_field)
  nullify(read_v_field)

#ifdef DOUBLEP
  call set(mesh_field, 1, (/tiny(0.0_D) * 1.0_D, 0.0_D/))
  call set(written_s_field, tiny(0.0_D) * 1.0_D)
  call set(written_v_field, (/tiny(0.0_D) * 1.0_D, 0.0_D/))
  call vtk_write_state(filename, model = "CoordinateMesh", state = (/written_state/), stat = stat)
  call report_test("[vtk_write_state]", stat /= 0, .false., "Failed to write state")

  call vtk_read_state(filename, state = read_state)
  read_v_field => extract_vector_field(read_state, "Coordinate")
  call report_test("[Coordinate field, tiny, real_8 precision]", read_v_field%val(1,1) < tiny(0.0_D), .false., "[Insufficient precision]")
  read_s_field => extract_scalar_field(read_state, "TestScalarField")
  read_v_field => extract_vector_field(read_state, "TestVectorField")
  call report_test("[Scalar field, tiny, real_8 precision]", any(read_s_field%val < tiny(0.0_D)), .false., "[Insufficient precision]")
  call report_test("[Vector field, tiny, real_8 precision]", any(read_v_field%val(1,:) < tiny(0.0_D)), .false., "[Insufficient precision]")
  call deallocate(read_state)
  nullify(read_s_field)
  nullify(read_v_field)
#endif

  ! TODO: Similar tests for tensor fields, similar tests using epsilon

  call deallocate(written_s_field)
  call deallocate(written_v_field)
  call deallocate(mesh_field)
  call deallocate(written_state)

  call report_test_no_references()

end subroutine test_vtk_precision
