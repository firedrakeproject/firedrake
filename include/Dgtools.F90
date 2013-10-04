#include "fdebug.h"

module dgtools

use elements
use sparse_tools
implicit none

contains

  function local_node_map(m, m_f, bdy, bdy_2) result(local_glno)
    ! Fill in the number map for the DG double element.
    type(element_type), intent(in) :: m, m_f
    integer, dimension(m_f%ndof) :: bdy, bdy_2
    integer, dimension(m%ndof,2) :: local_glno

    integer :: i,j

    local_glno=0

    ! First m_f%ndof places are for the bdy between the elements.
    forall(i=1:m_f%ndof)
       local_glno(bdy(i),1)=i
    end forall

    ! Remaining spots go to elements.
    j=m_f%ndof
    do i=1, m%ndof
       if(local_glno(i,1)==0) then
          j=j+1
          local_glno(i,1)=j
       end if
    end do

    ASSERT(j==m%ndof)

    ! First m_f%ndof places are for the bdy between the elements.
    forall(i=1:m_f%ndof)
       local_glno(bdy_2(i),2)=i
    end forall

    ! Remaining spots go to elements.
    j=m%ndof
    do i=1, m%ndof
       if(local_glno(i,2)==0) then
          j=j+1
          local_glno(i,2)=j
       end if
    end do

    ASSERT(j==2*m%ndof-m_f%ndof)

  end function local_node_map

  subroutine local_node_map_nc(nh_f,nu,b_seg, b_seg_2, &
       u_ele, u_ele_2, &
       local_glno, local2global_glno)
    ! Fill in the number map for the NC double element.
    ! only works for the P1nc element
    ! higher order elements require deeper thinking

    type(element_type) :: nu, nh_f   ! Shape Functions
    integer, dimension(nh_f%ndof), intent(in) :: b_seg, b_seg_2
    integer, dimension(:), intent(in) :: u_ele, u_ele_2
    integer, dimension(nu%ndof,2), intent(out) :: local_glno
    integer, dimension(2*nu%ndof-1), intent(out) :: local2global_glno

    integer :: i,j

    local_glno=0

    ! First 2 places are for nodes in element 1 away from the
    ! shared b_seg
    forall(i=1:2)
       local_glno(b_seg(i),1)=i
       local2global_glno(i) = u_ele(b_seg(i))
    end forall

    ! spot 3 goes to the node on the shared b_seg
    do i=1, 3
       if(local_glno(i,1)==0) then
          local_glno(i,1)=3
          local2global_glno(3) = u_ele(i)
       end if
    end do

    ! last 2 places are for the nodes in element 2 away from the
    ! shared b_seg
    forall(i=1:2)
       local_glno(b_seg_2(i),2)=i+3
       local2global_glno(i+3) = u_ele_2(b_seg_2(i))
    end forall

    ! spot 3 goes to the node on the shared b_seg
    do i=1, 3
       if(local_glno(i,2)==0) then
          local_glno(i,2)=3
       end if
    end do

  end subroutine local_node_map_nc

  function get_nc_coefficients(b_seg_n,loc,f_loc) result(coeff)
    implicit none

    ! need a local b_seg ordering
    ! nodes not on the b_seg, node on the b_seg.
    ! b_seg_nh_lno gives local node numbers for h on b_seg
    ! these are the same as the local node numbers which are
    ! not on the b_seg for u
    ! on the b_seg, basis functions for u nodes not on the b_seg
    ! take value 0.5 on the side of the b_seg they are on, and
    ! value -0.5 on the other side
    ! we'll need to calculate which one is which
    ! we already have that information as the local node number
    ! for the b_seg h node is the same as the volume u node opposite
    ! the basis function corresponding to the node that is on
    ! the b_seg is ==1 on the b_seg

    integer, intent(in):: loc, f_loc
    integer, dimension(f_loc), intent(in) :: b_seg_n
    real, dimension(loc,2) :: coeff

    !local variables
    integer, dimension(loc) :: flag
    integer :: i

    flag = 0.

    do i = 1,f_loc
       flag(b_seg_n(i)) = i
    end do

    do i = 1,loc
       select case (flag(i))
       case (0)
          coeff(i,:) = 1.0
       case (1)
          coeff(i,:) = (/ -0.5, 0.5 /)
       case (2)
          coeff(i,:) = (/ 0.5, -0.5 /)
       case default
          ERROR('NC coefficients disaster -- cjc')
       end select
    end do

  end function get_nc_coefficients

  subroutine solve(A,B)
    ! Solve Ax=b for multiple right hand sides B putting the result in B.
    !
    ! This is simply a wrapper for lapack.
    real, dimension(:,:), intent(in) :: A
    real, dimension(:,:), intent(inout) :: B

    integer, dimension(size(A,1)) :: ipiv
    integer :: info

    interface
#ifdef DOUBLEP
       SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
         INTEGER :: INFO, LDA, LDB, N, NRHS
         INTEGER :: IPIV( * )
         REAL ::  A( LDA, * ), B( LDB, * )
       END SUBROUTINE DGESV
#else
       SUBROUTINE SGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
         INTEGER :: INFO, LDA, LDB, N, NRHS
         INTEGER :: IPIV( * )
         REAL ::  A( LDA, * ), B( LDB, * )
       END SUBROUTINE SGESV
#endif
    end interface

    ASSERT(size(A,1)==size(A,2))
    ASSERT(size(A,1)==size(B,1))

#ifdef DOUBLEP
    call dgesv(size(A,1), size(B,2), A, size(A,1), ipiv, B, size(B,1),&
         & info)
#else
    call sgesv(size(A,1), size(B,2), A, size(A,1), ipiv, B, size(B,1),&
         & info)
#endif

    ASSERT(info==0)

  end subroutine solve

end module dgtools

