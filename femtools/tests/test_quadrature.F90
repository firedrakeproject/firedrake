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
subroutine test_quadrature
  use FLDebug
  use quadrature
  use quadrature_test
  use unittest_tools

  type(quadrature_type) :: quad
  type(quadrature_template), dimension(:), pointer :: template

  integer :: dim, vertices, degree, stat, i

  character(len=254) :: test_message, error_message
  logical :: fail

  call construct_quadrature_templates

  ! Test for simplices.
  do dim=1,3

     vertices=dim+1
     degree=0

     degreeloop:do
        degree=degree+1

        quad=make_quadrature(vertices, dim, degree=degree, stat=stat)

        select case (stat)
        case (QUADRATURE_DEGREE_ERROR)
           ! Reached highest available degree.
           exit degreeloop
        case (0)
           ! Success
           continue
        case default
           ! Some other error
           FLAbort(quadrature_error_message)
        end select

        ! Skip any degrees which don't exist.
        degree=quad%degree

        do power=0,degree

           if(quad_integrate(monic, quad) .fne. simplex_answer()) then
              write(error_message,'(e15.7)') &
                   quad_integrate(monic, quad)-simplex_answer()
              fail=.true.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-simplex, qua&
                &d degree ",degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))

        end do

        call deallocate(quad)

     end do degreeloop

  end do

  ! Test for hypercubes
  do dim=2,3

     vertices=2**dim

     select case(dim)
     case(2)
        template=>quad_quads
     case(3)
        template=>hex_quads
     end select

     quadloop: do i=1,size(template)
        degree=template(i)%degree

        quad=make_quadrature(vertices, dim, degree=degree, stat=stat)

        select case (stat)
        case (0)
           ! Success
           continue
        case default
           ! Some other error
           FLAbort(quadrature_error_message)
        end select

        do power=0,degree

           if(quad_integrate(cube_monic, quad) .fne. cube_answer()) then
              write(error_message,'(e15.7)') &
                   quad_integrate(cube_monic, quad)-cube_answer()
              fail=.true.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(4(a,i0),a)') "[",dim,"-cube, qua&
                &d number ",i," degree ",degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))

        end do

        call deallocate(quad)

     end do quadloop

  end do

contains

  recursive function factorial(n) result(f)
    ! Calculate n!
    integer :: f
    integer, intent(in) :: n

    if (n==0) then
       f=1
    else
       f=n*factorial(n-1)
    end if

  end function factorial

  function simplex_answer()
    ! Analytic solution to integrating monic over a simplex.
    ! This formula is eq. 7.38 and 7.48 in Zienkiewicz and Taylor
    real :: simplex_answer

    simplex_answer=real(factorial(power))&
         /factorial(power+dim)

  end function simplex_answer

  function cube_answer()
    ! Analytic solution to integrating ((1-x)/2)**power over a hypercube.
    real :: cube_answer

    cube_answer=(2.0**dim)/(power+1)

  end function cube_answer

end subroutine test_quadrature

