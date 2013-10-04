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

subroutine test_smooth_scalar

  use fields
  use fldebug
  use read_triangle
  use solvers
  use smoothing_module
  use unittest_tools

  implicit none

  character(len = *), parameter :: path = "/dummy"
  integer :: i
  integer, parameter :: quad_degree = 2
  logical :: fail
  real, dimension(2) :: minmax
  real :: alpha
  type(scalar_field) :: s_field, smoothed_s_field
  type(vector_field) :: positions

  positions = read_triangle_files("data/square-2d_A", quad_degree = quad_degree)

  call allocate(s_field, positions%mesh, "Scalar")
  do i = 1, node_count(s_field)
    call set(s_field, i, node_val(positions, 1, i) ** 3)
  end do

  alpha = 0.0

  call allocate(smoothed_s_field, positions%mesh, "SmoothedScalar")

  call set_solver_options(path, ksptype = "cg", pctype = "sor", atol = epsilon(0.0), rtol = 0.0, max_its = 2000, start_from_zero = .true.)
  call smooth_scalar(s_field, positions, smoothed_s_field, alpha, path)

  fail = .false.
  do i = 1, node_count(s_field)
    fail = fnequals(node_val(s_field, i), node_val(smoothed_s_field, i))
    if(fail) exit
  end do
  call report_test("[Not smoothed]", fail, .false., "Smoothed")

  alpha = 1.0e10

  call smooth_scalar(s_field, positions, smoothed_s_field, alpha, path)

  minmax = (/huge(0.0), -huge(0.0)/)
  do i = 1, node_count(smoothed_s_field)
    minmax(1) = min(minmax(1), node_val(smoothed_s_field, i))
    minmax(2) = max(minmax(2), node_val(smoothed_s_field, i))
  end do

  call report_test("[Smoothed]", fnequals(minmax(2), minmax(1), tol = 1.0e-6), .false., "Not smoothed")

  call deallocate(smoothed_s_field)
  call deallocate(s_field)
  call deallocate(positions)

  call report_test_no_references()

end subroutine test_smooth_scalar
