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

  subroutine test_colouring
  use fields_manipulation
  use state_module
  use vtk_interfaces
  use colouring
  use sparsity_patterns
  use unittest_tools
  use read_triangle
  use data_structures

  implicit none

  type(vector_field) :: positions
  type(mesh_type)  :: mesh
  type(csr_sparsity) :: sparsity
  integer :: maxdgr, i, j, len, sum1, sum2
  logical :: fail=.false.
  type(scalar_field) :: node_colour
  integer :: no_colours
  type(integer_set), dimension(:), allocatable :: clr_sets

  !positions = read_triangle_files("data/pslgA", quad_degree=4)
  positions = read_triangle_files('data/square-cavity-2d', quad_degree=4)
  mesh = piecewise_constant_mesh(positions%mesh, "P0Mesh")
  sparsity = make_sparsity_compactdgdouble(mesh, "cdG Sparsity")

  ! The sparsity matrix is the adjacency matrix of the graph and should therefore have dimension nodes X nodes
  assert(size(sparsity,1)==size(sparsity,2))

  maxdgr=0
  do i=1, size(sparsity, 1)
     maxdgr=max(maxdgr, row_length(sparsity, i))
  enddo
  call colour_sparsity(sparsity, mesh, node_colour, no_colours)

  if (no_colours > maxdgr+1) fail = .true. ! The +1 is needed for sparsities with zeros on the diagonal
  call report_test("colour sets", fail, .false., "there are more colours than the degree of the graph")

  fail=.not. verify_colour_sparsity(sparsity, node_colour)
  call report_test("colour sets", fail, .false., "the colouring is not valid")

  fail= .false.
  allocate(clr_sets(no_colours))
  call allocate(clr_sets)
  clr_sets=colour_sets(sparsity, node_colour, no_colours)

  sum1=0
  sum2=0
  do i=1, size(sparsity, 1)
     sum1=sum1+i
  enddo

  do i=1, no_colours
     len=key_count(clr_sets(i))
     do j= 1, len
        sum2=sum2+fetch(clr_sets(i), j)
     enddo
  enddo

  fail = .not.(sum1 .eq. sum2)
  call report_test("colour sets", fail, .false., "there are something wrong in  construction of colour_sets")
  call deallocate(clr_sets)
  deallocate(clr_sets)

  end subroutine test_colouring
