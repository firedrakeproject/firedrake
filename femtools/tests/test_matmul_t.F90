subroutine test_matmul_t

  use vector_tools
  use sparse_tools
  use unittest_tools
  implicit none

  real, dimension(3, 3) :: A, B, C, D
  type(dynamic_csr_matrix) :: A_d, B_d, C_d
  integer :: i, j, k
  logical :: fail
  character(len=20) :: buf

  call allocate(A_d, 3, 3)
  call allocate(B_d, 3, 3)

  do k=1,5
    write(buf,'(i0)') k
    fail = .false.

    A = random_matrix(3)
    B = random_matrix(3)

    call zero(A_d)
    call zero(B_d)

    do i=1,3
       do j=1,3
          call addto(A_d,i,j,A(i,j))
          call addto(B_d,i,j,B(i,j))
       end do
    end do

    C_d = matmul_T(A_d,B_d)

    C=dense(C_d)

    call deallocate(C_d)

    D = matmul(A, transpose(B))

    if (any(C /= D)) fail = .true.

    call report_test("[matmul_t " // trim(buf) // "]", fail, .false., "The output of matmul_t and matmul should be identical.")
  end do

end subroutine test_matmul_t
