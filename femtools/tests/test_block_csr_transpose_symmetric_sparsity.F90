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

subroutine test_block_csr_transpose_symmetric_sparsity

  use Sparse_Tools
  use unittest_tools

  type(csr_sparsity) :: sparsity
  type(csr_matrix) :: A, B
  type(block_csr_matrix) :: block_mat, block_mat_T, block_mat_TT
  integer :: i, j

  call allocate(sparsity, 2, 2, nnz = (/ 2, 1 /), name="Sparsity")
  sparsity%colm = (/ 2, 1, 1 /)
  call report_test("[sparsity is symmetric]", .not. is_symmetric(sparsity), .false., "sparsity is not symmetric")
  call report_test("[sparsity is not yet sorted]", is_sorted(sparsity), .false., "sparsity should be not sorted before calling sort(sparsity).")
  call sparsity_sort(sparsity)
  call report_test("[sparsity is sorted]", .not. is_sorted(sparsity), .false., "sparsity is not sorted after calling sort(sparsity).")


  sparsity%sorted_rows = .true.

  call allocate(A, sparsity, name="A")
  call set(A, (/ 1 /) , (/ 2 /) , reshape( (/ 1.0, 2.0 /), (/ 1, 2/) ) )
  call set(A, (/ 2 /), (/ 1 /), reshape( (/ 3.0 /), (/ 1, 1 /) ) )

  call allocate(block_mat, sparsity, (/ 1, 3 /), name="BlockMat")
  call set(block_mat, 1, 1, A)
  call set(block_mat, 1, 2, A)
  call set(block_mat, 1, 3, A)

  block_mat_T = transpose(block_mat, symmetric_sparsity=.true.)
  block_mat_TT = transpose(block_mat_T, symmetric_sparsity=.true.)

  call report_test("[blocks are the same]", .not. all(block_mat%blocks == block_mat_TT%blocks), .false., "the blocks do not match")
  do i=1,block_mat%blocks(1)
    do j=1,block_mat%blocks(2)
    call report_test("[values are the same]", .not. all(block_mat%val(i,j)%ptr == block_mat_TT%val(i, j)%ptr), .false., "the values do not match")
    end do
  end do

  call deallocate(block_mat_T)
  call deallocate(block_mat)

end subroutine test_block_csr_transpose_symmetric_sparsity

