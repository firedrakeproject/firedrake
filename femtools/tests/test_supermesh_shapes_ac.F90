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

subroutine test_supermesh_shapes_ac

  use fields
  use fldebug
  use supermesh_assembly
  use unittest_tools

  implicit none

  integer :: degree, i
  real, dimension(:), allocatable :: shape_rhs_integral_a, shape_rhs_integral_c
  real, dimension(:, :), allocatable :: dshape_dot_dshape_integral_a, dshape_dot_dshape_integral_c
  type(element_type) :: shape_a, shape_c
  type(element_type), dimension(:), allocatable :: shapes_c
  type(mesh_type) :: mesh_a, mesh_c, shape_mesh
  type(quadrature_type) :: quad
  type(vector_field) :: positions_a, positions_c

  do degree = 0, 4
    print *, "Degree = ", degree

    quad = make_quadrature(vertices = 3, dim  = 2, degree = max(degree * 2, 1))
    shape_a = make_element_shape(vertices = 3, dim  = 2, degree = 1, quad = quad)
    shape_c = make_element_shape(vertices = 3, dim  = 2, degree = degree, quad = quad)
    call deallocate(quad)

    call allocate(mesh_a, nodes = 3, elements = 1, shape = shape_a, name = "TargetMesh")
    call set_ele_nodes(mesh_a, 1, (/1, 2, 3/))

    call allocate(positions_a, dim = 2, mesh = mesh_a, name = "TargetCoordinate")
    call deallocate(mesh_a)
    call set(positions_a, 1, (/0.0, 0.0/))
    call set(positions_a, 2, (/1.0, 0.0/))
    call set(positions_a, 3, (/0.0, 1.0/))

    call allocate(mesh_c, nodes = 4, elements = 3, shape = shape_a, name = "Supermesh")
    call deallocate(shape_a)
    call set_ele_nodes(mesh_c, 1, (/1, 2, 4/))
    call set_ele_nodes(mesh_c, 2, (/4, 2, 3/))
    call set_ele_nodes(mesh_c, 3, (/3, 1, 4/))
    allocate(mesh_c%region_ids(3))
    mesh_c%region_ids = (/1, 1, 1/)

    call allocate(positions_c, dim = 2, mesh = mesh_c, name = "SupermeshCoordinate")
    call deallocate(mesh_c)
    call set(positions_c, 1, (/0.0, 0.0/))
    call set(positions_c, 2, (/1.0, 0.0/))
    call set(positions_c, 3, (/0.0, 1.0/))
    call set(positions_c, 4, (/0.25, 0.25/))

    shape_mesh = make_mesh(positions_a%mesh, shape = shape_c, continuity = -1, name = "ShapeMesh")
    call deallocate(shape_c)

    call project_donor_shape_to_supermesh(positions_a, shape_mesh, positions_c, &
      & shapes_c, form_dn = .true.)

    allocate(shape_rhs_integral_a(ele_loc(shape_mesh, 1)))
    shape_rhs_integral_a = shape_rhs_integral_ele(1, positions_a, ele_shape(shape_mesh, 1))

    allocate(shape_rhs_integral_c(ele_loc(shape_mesh, 1)))
    shape_rhs_integral_c = 0.0
    do i = 1, ele_count(positions_c)
      shape_rhs_integral_c = shape_rhs_integral_c + shape_rhs_integral_ele(i, positions_c, shapes_c(i))
    end do

    call report_test("[shape_rhs on supermesh]", shape_rhs_integral_a .fne. shape_rhs_integral_c, .false., "Incorrect integral")

    deallocate(shape_rhs_integral_a)
    deallocate(shape_rhs_integral_c)

    allocate(dshape_dot_dshape_integral_a(ele_loc(shape_mesh, 1), ele_loc(shape_mesh, 1)))
    dshape_dot_dshape_integral_a = dshape_dot_dshape_integral_ele(1, positions_a, ele_shape(shape_mesh, 1))

    allocate(dshape_dot_dshape_integral_c(ele_loc(shape_mesh, 1), ele_loc(shape_mesh, 1)))
    dshape_dot_dshape_integral_c = 0.0
    do i = 1, ele_count(positions_c)
      dshape_dot_dshape_integral_c = dshape_dot_dshape_integral_c + dshape_dot_dshape_integral_ele(i, positions_c,  shapes_c(i))
    end do

    call report_test("[dshape_dot_dshape on supermesh]", fnequals(dshape_dot_dshape_integral_a, dshape_dot_dshape_integral_c, tol = 1.0e3 * epsilon(0.0)), .false., "Incorrect integral")

    deallocate(dshape_dot_dshape_integral_a)
    deallocate(dshape_dot_dshape_integral_c)

    do i = 1, size(shapes_c)
      call deallocate(shapes_c(i))
    end do
    deallocate(shapes_c)

    call deallocate(positions_a)
    call deallocate(positions_c)
    call deallocate(shape_mesh)

    call report_test_no_references()
  end do

contains

  function shape_rhs_integral_ele(ele, positions, shape) result(integral)
    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(element_type), intent(in) :: shape

    real, dimension(shape%ndof) :: integral

    real, dimension(ele_ngi(positions, ele)) :: detwei

    call transform_to_physical(positions, ele, detwei = detwei)

    integral = shape_rhs(shape, detwei)

  end function shape_rhs_integral_ele

  function dshape_dot_dshape_integral_ele(ele, positions, shape) result(integral)
    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(element_type), intent(in) :: shape

    real, dimension(shape%ndof, shape%ndof) :: integral

    real, dimension(ele_ngi(positions, ele)) :: detwei
    real, dimension(shape%ndof, ele_ngi(positions, ele), positions%dim) :: dn

    call transform_to_physical(positions, ele, shape, &
      & dshape = dn, detwei = detwei)

    integral = dshape_dot_dshape(dn, dn, detwei)

  end function dshape_dot_dshape_integral_ele

end subroutine test_supermesh_shapes_ac
