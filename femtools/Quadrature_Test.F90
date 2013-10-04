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

module quadrature_test
  !!< Support module for all unit tests related to quadrature.
  !!< Provides auxiliary routines needed by these tests and separates these
  !!< from the actual module, thereby reducing dependencies.
  use quadrature
  implicit none

  ! Power is used by the test functions.
  integer, save :: power=0

  contains

  !------------------------------------------------------------------------
  ! Test procedures
  !------------------------------------------------------------------------

  function quad_integrate(integrand, quad) result (integral)
    ! Integrate the function integrand over an element using the
    ! specified quadrature.
    real :: integral
    interface
       function integrand(coords)
         real :: integrand
         real, dimension(:), intent(in) :: coords
       end function integrand
    end interface
    type(quadrature_type) :: quad

    integer :: i

    integral=0

    do i=1, size(quad%weight)
       integral=integral+quad%weight(i)*integrand(quad%l(i,:))
    end do

  end function quad_integrate

  function monic(coords)
    ! Calculate x^n
    real :: monic
    real, dimension(:), intent(in) :: coords

    monic=coords(1)**power

  end function monic

  function cube_monic(coords)
    ! Calculate.
    real :: cube_monic
    real, dimension(:), intent(in) :: coords

    cube_monic=((1-coords(1))/2.0)**power

  end function cube_monic

end module quadrature_test
