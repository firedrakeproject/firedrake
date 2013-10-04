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

subroutine test_get_connected_surface_eles

  use fields
  use fldebug
  use read_triangle
  use surfacelabels
  use unittest_tools

  implicit none

  integer :: i
  integer, parameter :: quad_degree = 1
  type(integer_vector), dimension(:), allocatable :: connected_surface_eles
  type(vector_field) :: positions

  positions = read_triangle_files("data/interval", quad_degree = quad_degree)

  call get_connected_surface_eles(positions%mesh, connected_surface_eles)

  call report_test("[Correct number of surfaces]", size(connected_surface_eles) /= 2, .false., "Incorrect number of surfaces")
  call report_test("[Correct surface]", any(connected_surface_eles(1)%ptr /= (/1/)), .false., "Incorrect surface")
  call report_test("[Correct surface]", any(connected_surface_eles(2)%ptr /= (/2/)), .false., "Incorrect surface")

  do i = 1, size(connected_surface_eles)
    deallocate(connected_surface_eles(i)%ptr)
  end do
  deallocate(connected_surface_eles)
  call deallocate(positions)

  call report_test_no_references()

  positions = read_triangle_files("data/tet", quad_degree = quad_degree)

  call get_connected_surface_eles(positions%mesh, connected_surface_eles)

  call report_test("[Correct number of surfaces]", size(connected_surface_eles) /= 1, .false., "Incorrect number of surfaces")
  call report_test("[Correct surface]", any(connected_surface_eles(1)%ptr /= (/1, 2, 3, 4/)), .false., "Incorrect surface")

  do i = 1, size(connected_surface_eles)
    deallocate(connected_surface_eles(i)%ptr)
  end do
  deallocate(connected_surface_eles)
  call deallocate(positions)

  call report_test_no_references()

end subroutine test_get_connected_surface_eles
