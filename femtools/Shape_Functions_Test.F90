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
module shape_functions_test
  !!< Support module for all unit tests related to shape_functions.
  !!< Provides auxiliary routines needed by these tests and separates these
  !!< from the actual module, thereby reducing dependencies.
  use shape_functions
  use spud, only: option_count, get_option

  ! Power is used by the test functions.
  integer, save :: power=0

contains

  !------------------------------------------------------------------------
  ! Test procedures
  !------------------------------------------------------------------------

  function shape_integrate(integrand, element) result (integral)
    !!< Integrate the function integrand over an element using the
    !!< specified shape functions and quadrature.
    real :: integral
    interface
       function integrand(coords)
         real :: integrand
         real, dimension(:), intent(in) :: coords
       end function integrand
    end interface
    type(element_type), intent(in) :: element

    real :: tmpval
    integer :: i,j

    integral=0.0

    do i=1, element%ndof

       tmpval=integrand(local_coords(i,element))

       do j=1, element%quadrature%ngi

          integral=integral+element%quadrature%weight(j)*tmpval*element%n(i,j)

       end do
    end do

  end function shape_integrate

  function shape_integrate_diff(integrand, element, dim) result (integral)
    !!< Integrate the function derivative of integrand with respect to dim
    !!< over an element using the specified shape functions and quadrature.
    real :: integral
    interface
       function integrand(coords)
         real :: integrand
         real, dimension(:), intent(in) :: coords
       end function integrand
    end interface
    type(element_type), intent(in) :: element
    integer, intent(in) :: dim

    real :: tmpval
    integer :: i,j

    integral=0.0

    do i=1, element%ndof

       tmpval=integrand(local_coords(i,element))

       do j=1, element%quadrature%ngi

          integral=integral&
               +element%quadrature%weight(j)*tmpval*element%dn(i,j,dim)

       end do
    end do

  end function shape_integrate_diff

  function shape_integrate_surface(integrand, element, dim,face) &
      result (integral)
    !!< Integrate the function derivative of integrand with respect to dim
    !!< over an element using the specified shape functions and quadrature.
    real :: integral
    interface
       function integrand(coords)
         real :: integrand
         real, dimension(:), intent(in) :: coords
       end function integrand
    end interface
    type(element_type), intent(in) :: element
    integer, intent(in) :: dim
    integer, optional, intent(in) :: face

    real :: tmpval
    integer :: i,j,k

    integral=0.0

    do i=1, element%ndof

       tmpval=integrand(local_coords(i,element))

       if (present(face)) then
          do j=1, element%surface_quadrature%ngi
             integral=integral&
                  +element%quadrature%weight(j)*tmpval&
                  *element%dn_s(i,j,face,dim)
          end do
       else

       do k=1, element%dim+1
          do j=1, element%surface_quadrature%ngi
             integral=integral&
                  +element%quadrature%weight(j)*tmpval*element%n_s(i,j,k)
          end do
       end do
    end if
    end do

  end function shape_integrate_surface

  function shape_integrate_surface_diff(integrand, element, dim,face) &
      result (integral)
    !!< Integrate the function derivative of integrand with respect to dim
    !!< over an element using the specified shape functions and quadrature.
    real :: integral
    interface
       function integrand(coords)
         real :: integrand
         real, dimension(:), intent(in) :: coords
       end function integrand
    end interface
    type(element_type), intent(in) :: element
    integer, intent(in) :: dim
    integer, optional, intent(in) :: face

    real :: tmpval
    integer :: i,j,k

    integral=0.0

    do i=1, element%ndof

       tmpval=integrand(local_coords(i,element))

       if (present(face)) then
          do j=1, element%surface_quadrature%ngi
             integral=integral&
                  +element%quadrature%weight(j)*tmpval&
                  *element%dn_s(i,j,face,dim)
          end do
       else

       do k=1, element%dim+1
          do j=1, element%surface_quadrature%ngi
             integral=integral&
                  +element%quadrature%weight(j)*tmpval*element%dn_s(i,j,k,dim)
          end do
       end do
    end if
    end do

  end function shape_integrate_surface_diff

  function monic(coords)
    !!< Calculate x^n
    real :: monic
    real, dimension(:), intent(in) :: coords

    monic=coords(1)**power

  end function monic

  function cube_monic(coords)
    ! Calculate.
    real :: cube_monic
    real, dimension(:), intent(in) :: coords

    cube_monic=(1-coords(1))**power

  end function cube_monic

  subroutine shape_functions_test_check_options

    integer :: quaddegree, degree, stat, i, nmesh

    call get_option("/geometry/quadrature/degree", quaddegree)
    nmesh = option_count("/geometry/mesh")
    do i = 1, nmesh
      call get_option("/geometry/mesh["//int2str(i-1)//&
                      "]/from_mesh/mesh_shape/polynomial_degree", &
                      degree, stat)
      if(stat==0) then
        if (quaddegree<2*degree) then
          ewrite(0,"(a,i0,a,i0)") "Warning: quadrature of degree ",quaddegree&
                &," may be incomplete for elements of degree ",degree
        end if
      end if
    end do


  end subroutine shape_functions_test_check_options

end module shape_functions_test
