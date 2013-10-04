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
!    Foundation, Inc., 59 Temple Place, Suite 332, Boston, MA  02111-1327
!    USA

#include "fdebug.h"

subroutine test_surface_integrals_3d
  !!< Test 3D surface integrals

  use fields
  use fields_data_types
  use read_triangle
  use surface_integrals
  use unittest_tools

  implicit none

  integer :: i
  real :: integral
  real, dimension(:), allocatable :: pos
  type(scalar_field) :: test_s_field
  type(vector_field) :: mesh_field, test_v_field

  mesh_field = read_triangle_files("data/square-cavity", quad_degree = 4)
  assert(mesh_dim(mesh_field) == 3)

  call allocate(test_s_field, mesh_field%mesh, "TestScalar")

  call zero(test_s_field)

  integral = surface_integral(test_s_field, mesh_field)
  call report_test("[Zero valued scalar, whole mesh]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Zero valued scalar, single present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Zero valued scalar, multiple present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  call set(test_s_field, 1.0)

  integral = surface_integral(test_s_field, mesh_field)
  call report_test("[Constant valued scalar, whole mesh]", integral .fne. 2.2, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Constant valued scalar, single present surface ID]", integral .fne. 1.0, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Constant valued scalar, multiple present surface IDs]", integral .fne. 2.0, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/34/))
  call report_test("[Constant valued scalar, single non-present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/34, 35/))
  call report_test("[Constant valued scalar, multiple non-present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29, 34, 35/))
  call report_test("[Constant valued scalar, mix of present and non-present surface IDs]", integral .fne. 2.0, .false., "Incorrect integral")

  allocate(pos(mesh_dim(mesh_field)))
  do i = 1, node_count(mesh_field)
    pos = node_val(mesh_field, i)
    call set(test_s_field, i, pos(1))
  end do

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Linearly varying scalar, single present surface ID]", integral .fne. 0.5, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Linearly varying scalar, multiple present surface IDs]", integral .fne. 1.0, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/34/))
  call report_test("[Linearly varying scalar, single non-present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/34, 35/))
  call report_test("[Linearly varying scalar, multiple non-present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29, 34, 35/))
  call report_test("[Linearly varying scalar, mix of present and non-present surface IDs]", integral .fne. 1.0, .false., "Incorrect integral")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field)
  call report_test("[Gradient of linearly varying scalar, whole mesh]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field, surface_ids = (/30/)) ! Left
  call report_test("[Gradient of linearly varying scalar, single present surface ID]", integral .fne. -0.05, .false., "Incorrect integral")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field, surface_ids = (/32/)) ! Right
  call report_test("[Gradient of linearly varying scalar, single present surface ID]", integral .fne. 0.05, .false., "Incorrect integral")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field, surface_ids = (/30, 32/)) ! Left, right
  call report_test("[Gradient of linearly varying scalar, multiple present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field, surface_ids = (/34/))
  call report_test("[Gradient of linearly varying scalar, single non-present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field, surface_ids = (/34, 35/))
  call report_test("[Gradient of linearly varying scalar, multiple non-present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = gradient_normal_surface_integral(test_s_field, mesh_field, surface_ids = (/32, 34, 35/))
  call report_test("[Gradient of linearly varying scalar, mix of present and non-present surface IDs]", integral .fne. 0.05, .false., "Incorrect integral")

  do i = 1, node_count(mesh_field)
    pos = node_val(mesh_field, i)
    call set(test_s_field, i, pos(1) + pos(3))
  end do

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Bilinearly varying scalar, single present surface ID]", integral .fne. 1.0, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Bilinearly varying scalar, multiple present surface IDs]", integral .fne. 2.0, .false., "Incorrect integral")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/34/))
  call report_test("[Bilinearly varying scalar, single non-present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/34, 35/))
  call report_test("[Bilinearly varying scalar, multiple non-present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = surface_integral(test_s_field, mesh_field, surface_ids = (/28, 29, 34, 35/))
  call report_test("[Bilinearly varying scalar, mix of present and non-present surface IDs]", integral .fne. 2.0, .false., "Incorrect integral")

  call deallocate(test_s_field)
  call allocate(test_v_field, mesh_dim(mesh_field), mesh_field%mesh, "TestVector")

  call zero(test_v_field)

  integral = normal_surface_integral(test_v_field, mesh_field)
  call report_test("[Zero valued vector, whole mesh]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Zero valued vector, single present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Zero valued vector, multiple present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  call set(test_v_field, (/0.0, 1.0, 0.0/))

  integral = normal_surface_integral(test_v_field, mesh_field)
  call report_test("[Constant valued vector, whole mesh]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Constant valued vector, single present surface ID]", integral .fne. 1.0, .false., "Incorrect integral")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/29/)) ! Back
  call report_test("[Constant valued vector, single present surface ID]", integral .fne. - 1.0, .false., "Incorrect integral")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Constant valued vector, multiple present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/34/))
  call report_test("[Constant valued vector, single non-present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/34, 35/))
  call report_test("[Constant valued vector, multiple non-present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28, 34, 35/))
  call report_test("[Constant valued vector, mix of present and non-present surface IDs]", integral .fne. 1.0, .false., "Incorrect integral")

  call set(test_v_field, (/1.0, 1.0, 1.0/))

  integral = normal_surface_integral(test_v_field, mesh_field)
  call report_test("[Constant valued vector, whole mesh]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28/)) ! Front
  call report_test("[Constant valued vector, single present surface ID]", integral .fne. 1.0, .false., "Incorrect integral")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/29/)) ! Back
  call report_test("[Constant valued vector, single present surface ID]", integral .fne. - 1.0, .false., "Incorrect integral")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28, 29/)) ! Front, back
  call report_test("[Constant valued vector, multiple present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/34/))
  call report_test("[Constant valued vector, single non-present surface ID]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/34, 35/))
  call report_test("[Constant valued vector, multiple non-present surface IDs]", integral .fne. 0.0, .false., "Integral non-zero")

  integral = normal_surface_integral(test_v_field, mesh_field, surface_ids = (/28, 34, 35/))
  call report_test("[Constant valued vector, mix of present and non-present surface IDs]", integral .fne. 1.0, .false., "Incorrect integral")

  call deallocate(test_v_field)

  deallocate(pos)

  call deallocate(mesh_field)

end subroutine test_surface_integrals_3d
