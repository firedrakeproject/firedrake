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

subroutine test_intersection_finder_seeds

  use elements
  use fields
  use linked_lists
  use quadrature
  use unittest_tools

  implicit none

  integer :: i
  integer, dimension(2) :: seeds_vec
  logical, dimension(4) :: ele_found
  type(element_type) :: shape
  type(ilist) :: seeds
  type(ilist), dimension(4) :: map
  type(inode), pointer :: node
  type(mesh_type) :: mesh
  type(quadrature_type) :: quad
  type(vector_field) :: positions

  quad = make_quadrature(vertices = 2, dim  = 1, degree = 1)
  shape = make_element_shape(vertices = 2, dim  = 1, degree = 1, quad = quad)
  call deallocate(quad)

  call allocate(mesh, nodes = 6, elements = 4, shape = shape, name = "CoordinateMesh")
  call deallocate(shape)
  call set_ele_nodes(mesh, 1, (/1, 2/))
  call set_ele_nodes(mesh, 2, (/2, 3/))
  call set_ele_nodes(mesh, 3, (/4, 5/))
  call set_ele_nodes(mesh, 4, (/5, 6/))

  call allocate(positions, 1, mesh, "Coordinate")
  call deallocate(mesh)
  do i = 1, node_count(positions)
    call set(positions, i, spread(float(i), 1, 1))
  end do

  seeds = advancing_front_intersection_finder_seeds(positions)

  call report_test("[number of seeds]", seeds%length /= 2, .false., "Incorrect number of seeds")
  seeds_vec = list2vector(seeds)
  call report_test("[seed]", seeds_vec(1) /= 1, .false., "Incorrect seed")
  call report_test("[seed]", seeds_vec(2) /= 3, .false., "Incorrect seed")

  map = advancing_front_intersection_finder(positions, positions, seed = seeds_vec(1))
  ele_found = .false.
  do i = 1, size(map)
    node => map(i)%firstnode
    do while(associated(node))
      ele_found(node%value) = .true.

      node => node%next
    end do
  end do
  call report_test("[intersection_finder]", .not. all(ele_found .eqv. (/.true., .true., .false., .false./)), .false., "Incorrect intersections reported")
  do i = 1, size(map)
    call deallocate(map(i))
  end do

  map = advancing_front_intersection_finder(positions, positions, seed = seeds_vec(2))
  ele_found = .false.
  do i = 1, size(map)
    node => map(i)%firstnode
    do while(associated(node))
      ele_found(node%value) = .true.

      node => node%next
    end do
  end do
  call report_test("[intersection_finder]", .not. all(ele_found .eqv. (/.false., .false., .true., .true./)), .false., "Incorrect intersections reported")
  do i = 1, size(map)
    call deallocate(map(i))
  end do

  call deallocate(seeds)
  call deallocate(positions)

  call report_test_no_references()

end subroutine test_intersection_finder_seeds
