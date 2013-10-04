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

subroutine test_spev

  use futils
  use unittest_tools

  implicit none

  interface spev
#ifdef DOUBLEP
    subroutine dspev( &
#else
    subroutine sspev( &
#endif
      & jobz, uplo, n, ap, w, z, ldz, work, info)
      implicit none
      ! Calculation type
      character, intent(in) :: jobz  ! "N" = eigenvalues
                                     ! "V" = eigenvalues + eigenvectors
      ! Input type
      character, intent(in) :: uplo  ! "U" = upper triangle input
                                     ! "L" = lower triangle input
      integer, intent(in) :: n
      ! Upper or lower triangle of input matrix
      real, dimension((n * (n + 1)) / 2), intent(inout) :: ap
      ! Output eigenvalues
      real, dimension(n), intent(out) :: w
      integer, intent(in) :: ldz
      ! Output eigenvectors
      real, dimension(ldz, n), intent(out) :: z
      real, dimension(3 * n), intent(inout) :: work
      integer, intent(out) :: info
#ifdef DOUBLEP
    end subroutine dspev
#else
    end subroutine sspev
#endif
  end interface spev

  character(len = 255) :: buffer
  character(len = real_format_len()) :: r_format
  integer :: i, info, j, n, k
  logical :: fail
  real, dimension(:), allocatable :: ap, w, work
  real, dimension(:, :), allocatable :: a, z

  r_format = real_format()

  n = 3
  allocate(a(n, n))
  allocate(ap((n * (n + 1)) / 2))
  allocate(w(n))
  allocate(work(3 * n))
  allocate(z(size(w), n))

  ! Form a unit matrix a
  a = 0.0
  do i = 1, n
    a(i, i) = 1.0
  end do

  ! Extract the upper triangular of matrix a
  k=1
  do j = 1, n
    do i = 1, j
      ap(k) = a(i, j)
      k=k+1
    end do
  end do

  ! Zero, to check output
  w = 0
  z = 0
  info = 0

  call spev("V", "U", n, ap, w, z, n, work, info)
  call report_test("[spev]", info /= 0, .false., "spev failure")

  fail = w .fne. 1.0
  buffer = "Incorrect eigenvalues"
  if(fail) then
    write(buffer, "(a,a," // trim(r_format) // ")") trim(buffer), ", Max deviation = ", maxval(abs(abs(w) - 1.0))
  end if
  call report_test("[Unit matrix eigenvalues]", fail, .false., trim(buffer))

  fail = .false.
  do i = 1, n
    do j = i + 1, n
      if(dot_product(z(i, :), z(j, :)) .fne. 0.0) then
        fail = .true.
        exit
      end if
    end do
    if(fail) exit
  end do
  call report_test("[Orthogonal eigenvectors]", fail, .false., "Incorrect eigenvectors")

  fail = .false.
  do i = 1, n
    if(dot_product(z(i, :), z(i, :)) .fne. 1.0) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Normalised eigenvectors]", fail, .false., "Incorrect eigenvectors")

  deallocate(a)
  deallocate(ap)
  deallocate(w)
  deallocate(work)
  deallocate(z)

end subroutine test_spev
