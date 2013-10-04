!    Copyright (C) 2007 Imperial College London and others.
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
module tensors
  !!< This module provides tensor operations on arrays.
  use FLDebug
  implicit none

  interface tensormul
     module procedure tensormul_3_1, tensormul_3_2, tensormul_4_1, tensormul_3_1_last
  end interface

contains

  pure function exclude(i, j)
    !!< Choose dimension for the result of tensor contraction.
    integer, intent(in) :: i, j
    integer :: exclude

    if (i < j) then
      exclude = i
    else
      exclude = i +1
    end if
  end function exclude

  pure function tensormul_3_1(tensor1, vec, d) result(prod)
    !!< Tensor contraction on two tensors by contraction of the index specified.
    real, dimension(:, :, :), intent(in) :: tensor1
    real, dimension(:), intent(in) :: vec
    integer, intent(in) :: d

    integer, pointer :: m1, m2
    integer, dimension(3), target :: n
    integer :: i

    real, dimension(size(tensor1, exclude(1, d)), size(tensor1, exclude(2, d))) :: prod

    prod = 0.0

    n = 1
    m1 => n(exclude(1, d)); m2 => n(exclude(2,d))

    do i=1,size(tensor1)
      prod(m1, m2) = prod(m1, m2) + tensor1(n(1), n(2), n(3)) * vec(n(d))
      n(1) = n(1) + 1
      if (n(1) > size(tensor1, 1)) then
        n(1) = 1
        n(2) = n(2) + 1
        if (n(2) > size(tensor1, 2)) then
          n(2) = 1
          n(3) = n(3) + 1
        end if
      end if
    end do
  end function tensormul_3_1

  function tensormul_3_1_last(tensor1, vec) result(prod)
    real, dimension(:, :, :), intent(in) :: tensor1
    real, dimension(:), intent(in) :: vec
    real, dimension(size(tensor1, 1), size(tensor1, 2)) :: prod
    integer :: i

    prod = 0.0
    do i=1,size(vec)
      prod = prod + vec(i) * tensor1(:, :, i)
    end do
  end function tensormul_3_1_last

  pure function tensormul_3_2(tensor1, tensor2) result (product)
    !!< Tensor contraction on two tensors by innermost dimension.
    real, dimension(:,:,:), intent(in) :: tensor1
    real, dimension(:,:), intent(in) :: tensor2
    real, dimension(size(tensor1,1), size(tensor1,2), size(tensor2,2)) ::&
         & product

    integer :: i

    forall (i=1:size(tensor1,1))
       product(i,:,:)=matmul(tensor1(i,:,:),tensor2)
    end forall

  end function tensormul_3_2

  pure function tensormul_4_1(tensor1, vec, d) result(prod)
    !!< Tensor contraction on two tensors by contraction of the index specified.
    real, dimension(:, :, :, :), intent(in) :: tensor1
    real, dimension(:), intent(in) :: vec
    integer, intent(in) :: d

    integer, pointer :: m1, m2, m3
    integer, dimension(4), target :: n
    integer :: i

    real, dimension(size(tensor1, exclude(1, d)), size(tensor1, exclude(2, d)), size(tensor1, exclude(3, d))) :: prod

    prod = 0.0

    n = 1
    m1 => n(exclude(1, d)); m2 => n(exclude(2,d)); m3 => n(exclude(3,d))

    do i=1,size(tensor1)
      prod(m1, m2, m3) = prod(m1, m2, m3) + tensor1(n(1), n(2), n(3), n(4)) * vec(n(d))
      n(1) = n(1) + 1
      if (n(1) > size(tensor1, 1)) then
        n(1) = 1
        n(2) = n(2) + 1
        if (n(2) > size(tensor1, 2)) then
          n(2) = 1
          n(3) = n(3) + 1
          if (n(3) > size(tensor1, 3)) then
            n(3) = 1
            n(4) = n(4) + 1
          end if
        end if
      end if
    end do
  end function tensormul_4_1

end module tensors
