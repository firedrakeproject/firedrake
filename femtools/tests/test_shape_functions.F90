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

subroutine test_shape_functions
  !!< Generic element test function.
  use quadrature
  use shape_functions_test
  use unittest_tools
  type(element_type) :: element
  type(quadrature_type) :: quad

  character(len=500) :: error_message, test_message
  integer :: dim, degree, vertices
  logical :: fail

  ! Rounding error tolerance.
  real, parameter :: eps=1E-11

  ! Test for simplices.
  do dim=1,3

     vertices=dim+1

     quad=make_quadrature(vertices, dim, degree=7)

     do degree=0,7

        element=make_element_shape(vertices=vertices, dim=dim, degree=degree,&
          quad=quad)

        do power=0,degree

           ! Shape function itself
           if (.not.(abs(shape_integrate(monic, element)&
                -simplex_answer(power, dim))<eps)) then
              write(error_message,'(e15.7)') &
                   shape_integrate(monic, element)&
                   &-simplex_answer(power, dim)
              fail=.true.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-simplex, ele&
                &ment degree ",degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))


           ! Derivative
           if (.not.(abs(shape_integrate_diff(monic, element,1)&
                -power*simplex_answer(power-1, dim))<eps)) then
              write(error_message,'(e15.7)') &
                   shape_integrate_diff(monic, element,1)&
                   &-power*simplex_answer(power-1, dim)
              fail=.true.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-simplex &
                &surface derivative, element degree ",&
                degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))


        end do

        call deallocate(element)

     end do

     call deallocate(quad)

  end do

  ! Test for hypercubes.
  do dim=2,3

     vertices=2**dim

     quad=make_quadrature(vertices, dim, degree=7)

     do degree=0,7

        element=make_element_shape(vertices=vertices, dim=dim, degree=degree,&
           quad=quad)

        do power=0,degree

           ! Shape function itself
           if (.not.(abs(shape_integrate(cube_monic, element)&
                -cube_answer(power, dim))<eps)) then
              write(error_message,'(e15.7)') &
                   shape_integrate(cube_monic, element)&
                   &-cube_answer(power, dim)
              write(error_message,'(2e15.7)') &
                   shape_integrate(cube_monic, element),&
                   &cube_answer(power, dim)
              fail=.true.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-cube, ele&
                &ment degree ",degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))


           ! Derivative
           if (.not.(abs(shape_integrate_diff(cube_monic, element,1)&
                -1*cube_danswer(power, dim))<eps)) then
              write(error_message,'(e15.7)') &
                   shape_integrate_diff(cube_monic, element,1)&
                   -1*cube_danswer(power, dim)
              fail=.true.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-cube deri&
                &vative, element degree ",degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))

        end do

        call deallocate(element)

     end do

     call deallocate(quad)

  end do

contains

  function simplex_answer(power, dim)
    ! Analytic solution to integrating monic over a simplex.
    ! This formula is eq. 7.38 and 7.48 in Zienkiewicz and Taylor
    real :: simplex_answer
    integer, intent(in) :: power, dim

    simplex_answer=real(factorial(power))&
         /factorial(power+dim)

  end function simplex_answer

  function cube_answer(power, dim)
    ! Analytic solution to integrating ((1-x)/2)**power over a hypercube.
    real :: cube_answer
    integer, intent(in) :: power, dim

    cube_answer=1./(power+1)

  end function cube_answer

  function cube_danswer(power, dim)
    ! Analytic solution to integrating diff(((1-x)/2)**power,x) over a
    ! hypercube.
    real :: cube_danswer
    integer, intent(in) :: power, dim

    if (power==0) then
       cube_danswer=0
    else
       cube_danswer=-1
    end if

  end function cube_danswer

  recursive function factorial(n) result (f)
    ! Calculate n!
    integer :: f
    integer, intent(in) :: n

    if (n==0) then
       f=1
    else if (n<0) then
       f=0
    else
       f=n*factorial(n-1)
    end if

  end function factorial

end subroutine test_shape_functions

