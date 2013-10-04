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

subroutine test_make_mesh_quad

  use fields
  use element_numbering
  use fldebug
  use read_triangle
  use unittest_tools

  implicit none

  integer :: degree, ele, node
  integer, parameter :: min_degree = 1, max_degree = 9
  logical :: fail
  real, dimension(:, :), allocatable :: l_coords, quad_l_coords
  type(element_type) :: derived_shape
  type(element_type), pointer :: base_shape
  type(mesh_type) :: derived_mesh
  type(mesh_type), pointer :: base_mesh
  type(vector_field), target :: positions

  positions = read_triangle_files("data/quad_grid", quad_degree = 1)
  base_mesh => positions%mesh
  base_shape => ele_shape(base_mesh, 1)
  call report_test("[Linear quad input mesh]", &
    & ele_numbering_family(base_shape) /= FAMILY_CUBE .or. base_shape%degree /= 1 .or. base_shape%dim /= 2, .false., &
    & "Input mesh not composed of linear quads")

  do degree = min_degree, max_degree
    print "(a,i0)", "Degree = ", degree

    derived_shape = make_element_shape(base_shape, degree = degree)
    call report_test("[Derived loc]", &
      & derived_shape%ndof /= (degree + 1)**2, .false., &
      & "Incorrect local node count")

    derived_mesh = make_mesh(base_mesh, derived_shape)
    call report_test("[Derived ele_count]", &
      & ele_count(derived_mesh) /= ele_count(base_mesh), .false., &
      & "Incorrect element count")

    allocate(quad_l_coords(base_shape%dim, derived_shape%ndof))
    quad_l_coords = quad_local_coords(degree)
    allocate(l_coords(base_shape%dim, derived_shape%ndof))
    fail = .false.
    ele_loop: do ele = 1, ele_count(derived_mesh)
      fail = ele_loc(derived_mesh, ele) /= derived_shape%ndof
      if(fail) exit ele_loop

      do node = 1, size(l_coords, 2)
         l_coords(:,node) = local_coords(node,derived_shape%numbering)
      end do
      fail = fnequals(l_coords, quad_l_coords, tol = 1.0e3 * epsilon(0.0))
      if(fail) then
        do node = 1, size(l_coords, 2)
          print *, node, l_coords(:, node)
          print *, node, quad_l_coords(:, node)
        end do
        exit ele_loop
      end if
    end do ele_loop
    deallocate(l_coords)
    deallocate(quad_l_coords)

    call report_test("[Derived mesh numbering]", fail, .false., "Invalid derived mesh numbering, failed on element " // int2str(ele))

    call deallocate(derived_shape)
    call deallocate(derived_mesh)
  end do

  call deallocate(positions)
  call report_test_no_references()

contains

  function quad_local_coords(degree) result(l_coords)
    !!< Return the node local coords

    integer, intent(in) :: degree

    integer :: i, index, j
    real, dimension(2, (degree + 1)**2) :: l_coords

    index = 1
    do i = 0, degree
      do j = 0, degree
        assert(index <= size(l_coords, 2))
        l_coords(2, index) = float(i) / float(degree)
        l_coords(1, index) = float(j) / float(degree)
        index = index + 1
      end do
    end do
    assert(index == size(l_coords, 2) + 1)

  end function quad_local_coords

end subroutine test_make_mesh_quad
