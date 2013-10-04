!    Copyright (C) 2006-2007 Imperial College London and others.
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

subroutine test_1d
  !!< Test that basic integration and differentiation of 1d elements works.
  use shape_functions
  use fetools
  use fields
  use read_triangle
  use vector_tools
  use unittest_tools
  implicit none

  logical :: fail
  real :: integral

  type(vector_field) :: X
  type(scalar_field) :: T, dT
  interface
     function func_1d(X)
       real :: func_1d
       real, dimension(:), intent(in) :: X
     end function func_1d
  end interface

  integer :: ele
  real, dimension(:,:), allocatable :: mass
  real, dimension(:), allocatable :: detwei
  integer, dimension(:), pointer :: T_ele
  real, dimension(:,:,:), allocatable :: dT_ele
  type(element_type), pointer :: T_shape

  X=read_triangle_files("data/interval", quad_degree=4)

  call allocate(T, X%mesh, "tracer")

  call set_from_function(T, func_1d, X)

  ! Test 1 Integrate T over the interval.

  integral=field_integral(T,X)

  fail=integral/=0.5

  call report_test("[test_1d Integral]", fail, .false., "int_0^1 x dx should&
       & be 0.5.")

  ! Test 2 Calculate the derivative of T over the interval.

  allocate(mass(node_count(T),node_count(T)))
  allocate(detwei(ele_ngi(T,1)))
  allocate(dT_ele(ele_loc(T,1), ele_ngi(T,1), X%dim))

  dT=clone(T)

  call zero(dT)

  mass=0.0

  do ele=1,element_count(T)

     T_ele=>ele_nodes(T,ele)
     T_shape=>ele_shape(T,ele)

     call transform_to_physical(X, ele, T_shape, &
          dm_t=dT_ele, detwei=detwei)

     mass(T_ele, T_ele)=mass(T_ele, T_ele) &
          + shape_shape(T_shape, T_shape, detwei)

     call addto(dT, T_ele, matmul(sum(shape_dshape(T_shape, dT_ele, detwei),1),&
          & ele_val(T, ele)))

  end do

  call invert(mass)

  dT%val=matmul(mass,dT%val)

  fail=any(abs(dT%val-1.0)>1e-14)
  call report_test("[test_1d Derivative]", fail, .false., "dx/dx should&
       & be 1.0.")

  deallocate(mass)
  deallocate(detwei)
  deallocate(dT_ele)


end subroutine test_1d

function func_1d(x)
  real :: func_1d
  real, dimension(:), intent(in) :: x

  func_1d=X(1)

end function func_1d
