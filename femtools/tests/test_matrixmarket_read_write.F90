subroutine test_matrixmarket_read_write

  use sparse_tools
  use unittest_tools
  implicit none

  type(dynamic_csr_matrix):: A, B
  logical fail

  call mmread('data/matrix.mm', A)

  call allocate(B, 5, 6)

  call set(B, 1, 1, 1.0)
  call set(B, 1, 2, 3.0)
  call set(B, 2, 1, 0.0)
  call set(B, 4, 5, 6.0)
  call set(B, 5, 3, -99.0)
  call set(B, 5, 4, -5e20)
  call set(B, 5, 6, 20.0/3.0)

  fail= .not. fequals(A, B, 1e-8)

  call report_test("[matrixmarket_read]", fail, .false., &
    "Read matrix is not the same as in file matrix.mm")

  call deallocate(A)
  call deallocate(B)

  A=random_sparse_matrix(99, 100, 1001)

  call mmwrite('data/matrix2.mm', A)
  call mmread('data/matrix2.mm', B)

  fail= .not. fequals(A, B, 1e-8)

  call report_test("[matrixmarket_read_write]", fail, .false., &
    "Written matrix is not the same after reading back in.")

end subroutine test_matrixmarket_read_write
