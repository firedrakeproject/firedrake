subroutine test_matmul_t_sparse

  use vector_tools
  use sparse_tools
  use unittest_tools
  implicit none

  integer, parameter :: size_mat = 50
  integer, parameter :: n_samples = 100
  real, dimension(size_mat,size_mat) :: A, B, C, D
  type(dynamic_csr_matrix) :: A_d, B_d, C_d
  type(csr_matrix) :: A_c, B_c, C_dc
  integer :: i, j, k, n
  logical :: fail,fail1,fail2, fail3
  character(len=size_mat) :: buf
  real, dimension(4) :: rand0

  do k=1,5

     call allocate(A_d, size_mat,size_mat)
     call allocate(B_d, size_mat,size_mat)

    write(buf,'(i0)') k
    fail = .false.
    fail1 = .false.
    fail2 = .false.
    fail3 = .false.

    A = 0.
    B = 0.
    call zero(A_d)
    call zero(B_d)

    do n = 1, n_samples
       call random_number(rand0)
       i = ceiling(rand0(1)*size_mat)
       j = ceiling(rand0(2)*size_mat)
       A(i,j) = A(i,j) +1.0
       B(i,j) = B(i,j) + 1.0
       !call addto(A_d,i,j,rand0(3))
       !call addto(B_d,i,j,rand0(4))
       call addto(A_d,i,j,1.0)
       call addto(B_d,i,j,1.0)
    end do

    if (any(A /= dense(A_d))) then
       write(0,*) 'A assembly bungled'
       write(0,*) '---'
       write(0,*) A
       write(0,*) '---'
       write(0,*) dense(A_d)
       stop
    end if
    if (any(B /= dense(B_d))) then
       write(0,*) 'B assembly bungled'
       stop
    end if

    A_c = dcsr2csr(A_d)
    B_c = dcsr2csr(B_d)

    C_d = matmul_T(A_d,B_d)
    !C_c = matmul_T(A_c,B_c)

    C_dc = dcsr2csr(C_d)

    C=dense(C_d)
    !E = dense(C_c)

    if (any(abs(C-dense(C_dc))>1.0e-14)) fail3 = .true.


    do i = 1, size(B_d%colm)
       if(any(C_d%val(i)%ptr<0.5)) fail2=.true.
    end do

    call deallocate(C_d)
    call deallocate(A_c)
    call deallocate(B_c)
    !call deallocate(C_c)
    call deallocate(C_dc)

    D = matmul(A, transpose(B))

    if (any(abs(C-D)>1.0e-14)) fail = .true.
    !if (any(abs(E-D)>1.0e-14)) fail2 = .true.
    do i = 1, size(A_d%colm)
       do j = 2, size(A_d%colm(i)%ptr)
          if(A_d%colm(i)%ptr(j).le.A_d%colm(i)%ptr(j-1)) then
             fail1 = .true.
          end if
       end do
    end do

    do i = 1, size(B_d%colm)
       do j = 2, size(B_d%colm(i)%ptr)
          if(B_d%colm(i)%ptr(j).le.B_d%colm(i)%ptr(j-1)) then
             fail1 = .true.
          end if
       end do
    end do

    call report_test("[matmul_t_sparse dcsr " // trim(buf) // "]", fail, .false., "The output of matmul_t and matmul should be identical.")
    call report_test("[matmul_t_sparse dcsr dense " // trim(buf) // "]", fail3, .false., "The output of dense(dscr_matrix) and dense(dcsr2csr(dcsr_matrix)) should be identical.")
    call report_test("[matmul_t_sparse zeros " // trim(buf) // "]", fail2, .false., "The sparsity pattern should not contain zeros")
    call report_test("[matmul_t_sparse " // trim(buf) // " ordering]", fail1, .false., "We expect the rows to be incrementally ordered in dcsr matrices")

     call deallocate(A_d)
     call deallocate(B_d)

  end do

end subroutine test_matmul_t_sparse
