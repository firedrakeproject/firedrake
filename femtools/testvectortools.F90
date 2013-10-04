#include "fdebug.h"
program testvectortools
  use vector_tools
  implicit none

  integer, parameter :: N=50
  real, dimension(N,N) :: ident, A, Ainv
  integer :: i,j,k

  ident=0.0

  forall(i=1:N) ident(i,i)=1.0

  k=0
  do i=1,N
     do j=1,N
        k=k+1
        A(i,j)=-1
     end do
  end do

  forall(i=1:N)  A(i,i)=N

  Ainv=A

  call invert(Ainv)

  print *,'Error in invert:', maxval(abs(matmul(A,Ainv)-ident))

  Ainv=A

  call cholesky_factor(Ainv)

  print *,'Error in cholesky_factor:', &
       maxval(abs(matmul(transpose(Ainv),Ainv)-A))

end program testvectortools
