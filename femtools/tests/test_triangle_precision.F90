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

subroutine test_triangle_precision
  !!< Test the precision of triangle I/O

  use elements
  use fields
  use fields_data_types
  use global_parameters, only : real_4, real_8
  use read_triangle
  use unittest_tools
  use write_triangle

  implicit none

  character(len = 255) :: filename
  integer :: i
  integer, parameter :: D = real_8, S = real_4
  type(element_type) :: shape
  type(mesh_type) :: mesh
  type(quadrature_type) :: quad
  type(vector_field) :: read_mesh_field, written_mesh_field

  filename = "data/test_triangle_precision_out"

  ! Allocate a mesh
  quad = make_quadrature(vertices = 3, dim  = 2, degree = 1)
  shape = make_element_shape(vertices = 3, dim  = 2, degree = 1, quad = quad)
  call allocate(mesh, nodes = 3, elements = 1, shape = shape, name = "CoordinateMesh")
  call allocate(written_mesh_field, mesh_dim(mesh), mesh, "Coordinate")

  ! Create a single triangle mesh
  do i = 1, size(mesh%ndglno)
    mesh%ndglno(i) = i
  end do
  call set(written_mesh_field, 1, (/0.0, 0.0/))
  call set(written_mesh_field, 2, (/1.0, 0.0/))
  call set(written_mesh_field, 3, (/1.0, 1.0/))

  call deallocate(quad)
  call deallocate(shape)
  call deallocate(mesh)

  ! Clean existing output
  call write_triangle_files(filename, written_mesh_field)
  read_mesh_field = read_triangle_files(filename, quad_degree = 1)
  call report_test("[Clean]", abs(read_mesh_field%val(1,1)) > 0.0, .false., "[Failed to clean output]")
  call deallocate(read_mesh_field)

  call set(written_mesh_field, 1, real((/1.0_S * tiny(0.0_S), 0.0_S/)))
  call write_triangle_files(filename, written_mesh_field)
  read_mesh_field = read_triangle_files(filename, quad_degree = 1)
  call report_test("[tiny, real_4 precision]", read_mesh_field%val(1,1) < tiny(0.0_S), .false., "Insufficient precision")
  call deallocate(read_mesh_field)

  call set(written_mesh_field, 1, real((/1.0_S + epsilon(1.0_S), 0.0_S/)))
  call write_triangle_files(filename, written_mesh_field)
  read_mesh_field = read_triangle_files(filename, quad_degree = 1)
  call report_test("[epsilon, real_4 precision]", read_mesh_field%val(1,1) < 1.0_S + epsilon(1.0_S), .false., "Insufficient precision")
  call deallocate(read_mesh_field)

#ifdef DOUBLEP
  call set(written_mesh_field, 1, (/1.0_D * tiny(0.0_D), 0.0_D/))
  call write_triangle_files(filename, written_mesh_field)
  read_mesh_field = read_triangle_files(filename, quad_degree = 1)
  call report_test("[tiny, real_8 precision]", read_mesh_field%val(1,1) < tiny(0.0_D), .false., "Insufficient precision")
  call deallocate(read_mesh_field)

  call set(written_mesh_field, 1, (/1.0_D + epsilon(1.0_D), 0.0_D/))
  call write_triangle_files(filename, written_mesh_field)
  read_mesh_field = read_triangle_files(filename, quad_degree = 1)
  call report_test("[epsilon, real_8 precision]", read_mesh_field%val(1,1) < 1.0_S + epsilon(1.0_D), .false., "Insufficient precision")
  call deallocate(read_mesh_field)
#endif

  call deallocate(written_mesh_field)

  call report_test_no_references()

end subroutine test_triangle_precision
