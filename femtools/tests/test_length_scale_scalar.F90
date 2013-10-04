!    Copyright (C) 2009 Imperial College London and others.
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

subroutine test_length_scale_scalar

   use unittest_tools
   use read_triangle
   use fields
   use smoothing_module
   implicit none

   type(vector_field) :: positions
   real :: expected_result, computed_result
   logical :: fail

   positions = read_triangle_files("data/structured", quad_degree=3)

   ! We'll just choose the first element here - each of them should have the same area
   computed_result = length_scale_scalar(positions, 1)
   expected_result = 0.5*(0.392699081699**2)

   fail = .not.fequals(computed_result, expected_result, 1.0e-9)
   call report_test("[length_scale_scalar]", fail, .false., "Result from length_scale_scalar is incorrect.")

end subroutine test_length_scale_scalar

