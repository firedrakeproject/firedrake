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

  subroutine test_ispcolouring
  use fields_manipulation
  use state_module
  use vtk_interfaces
  use colouring
  use sparsity_patterns
  use unittest_tools
  use read_triangle
  implicit none

  type(vector_field) :: positions
  type(mesh_type)  :: mesh
  type(csr_sparsity) :: sparsity
  type(csr_sparsity) :: isp_sparsity
  logical :: fail=.false.
  type(scalar_field) :: node_colour
  integer :: no_colours

  positions = read_triangle_files('data/square-cavity-2d', quad_degree=4)
  mesh = piecewise_constant_mesh(positions%mesh, "P0Mesh")
  sparsity = make_sparsity_compactdgdouble(mesh, "cdG Sparsity")

  isp_sparsity=mat_sparsity_to_isp_sparsity(sparsity)

  call colour_sparsity(isp_sparsity, mesh, node_colour, no_colours)

  fail=.not. verify_colour_sparsity(isp_sparsity, node_colour)
  call report_test("colour sets", fail, .false., "the adjacency sparsity colouring is not valid")

  fail=.not. verify_colour_ispsparsity(sparsity, node_colour)
  call report_test("colour sets", fail, .false., "the csr sparcity colouring is not valid")

  end subroutine test_ispcolouring
