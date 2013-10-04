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

subroutine test_find_node_ownership_rtree

  use fields
  use fldebug
  use node_ownership
  use read_triangle
  use unittest_tools

  implicit none

  integer :: i
  integer, dimension(:), allocatable :: l_coords, node_ownership
  logical :: fail
  type(vector_field) :: positions1, positions2

  positions1 = read_triangle_files("data/rotated_square.1", quad_degree = 1)
  positions2 = read_triangle_files("data/rotated_square.2", quad_degree = 1)

  allocate(node_ownership(node_count(positions2)))
  call find_node_ownership_rtree(positions1, positions2, node_ownership)

  call report_test("[All node owners found]", any(node_ownership < 0), .false., "Not all node owners found")

  call report_test("[Correct map size]", size(node_ownership) /= node_count(positions2), .false., "Incorrect map size")
  fail = .false.
  do i = 1, node_count(positions2)
    allocate(l_coords(ele_loc(positions1, node_ownership(i))))
    l_coords = local_coords(positions1, node_ownership(i), node_val(positions2, i))
    if(any(l_coords < - default_ownership_tolerance)) then
      fail = .true.
      exit
    end if
    deallocate(l_coords)
  end do
  call report_test("[Valid map]", fail, .false., "Invalid map")

  deallocate(node_ownership)

  call deallocate(positions1)
  call deallocate(positions2)

  call report_test_no_references()

end subroutine test_find_node_ownership_rtree
