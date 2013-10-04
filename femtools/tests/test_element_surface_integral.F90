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

subroutine test_element_surface_integral
  use elements
  use quadrature
  use shape_functions_test
  use unittest_tools

  implicit none

  type(element_type) :: element
  type(quadrature_type) :: quad, quad_s

  character(len=500) :: error_message, test_message
  integer :: dim, degree, vertices, k
  logical :: fail
  ! Rounding error tolerance.
  real, parameter :: eps=1E-13

  dim=element%dim

  ! Test for simplices.
  do dim=2,2

     vertices=dim+1
     degree=0

     do degree=0,7

     quad=make_quadrature(vertices, dim, degree=7)
     quad_s=make_quadrature(vertices-1, dim-1, degree=degree)
     element=make_element_shape(vertices=vertices, dim=dim, degree=degree,&
          quad=quad, quad_s=quad_s)

        do power=0,degree

           ! surface
           if (.not.(abs(shape_integrate_surface(monic, element,1)&
                )<eps)) then
              write(error_message,'(e15.7)') &
                   shape_integrate_surface(monic, element,1)
              fail=.false.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-simplex &
                &surface, element degree ",&
                degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))

           do k=1,dim+1

              if (.not.(abs(shape_integrate_surface(monic, element,1,k)&
                   -line_answer(power, dim,k))<eps)) then
                 write(error_message,'(e15.7)') &
                      shape_integrate_surface(monic, element,1,k)&
                      -line_answer(power, dim,k)
                 fail=.false.
              else
                 error_message=""
                 fail=.false.
              end if

              write(test_message, '(4(a,i0),a)') "[",dim,"-simplex surface face,",&
                   k, " element degree ",&
                   degree," power ",power," ]"

              call report_test(trim(test_message), fail, .false.,&
                   & trim(error_message))

           end do

           ! surface derivative
           if (.not.(abs(shape_integrate_surface_diff(monic, element,1)&
                )<eps)) then
              write(error_message,'(e15.7)') &
                   shape_integrate_surface_diff(monic, element,1)
              fail=.false.
           else
              error_message=""
              fail=.false.
           end if

           write(test_message, '(3(a,i0),a)') "[",dim,"-simplex &
                &surface derivative, element degree ",&
                degree," power ",power," ]"

           call report_test(trim(test_message), fail, .false.,&
                & trim(error_message))

           do k=1,dim+1

              if (.not.(abs(shape_integrate_surface_diff(monic, element,1,k)&
                   -power*line_answer(power-1, dim,k))<eps)) then
                 write(error_message,'(e15.7)') &
                      shape_integrate_surface_diff(monic, element,1,k)&
                      -power*line_answer(power-1, dim,k)
                 fail=.false.
              else
                 error_message=""
                 fail=.false.
              end if

              write(test_message, '(4(a,i0),a)') "[",dim,"-simplex &
                   &surface derivative face,", k, " element degree ",&
                   degree," power ",power," ]"

              call report_test(trim(test_message), fail, .false.,&
                   & trim(error_message))

           end do


        end do

           call deallocate(quad)
           call deallocate(quad_s)
           call deallocate(element)

        end do


     end do

   contains

  function line_answer(power, dim, faces)
    ! Analytic solution to integrating monic on a line.
    real :: line_answer
    integer, intent(in) :: power, dim, faces

    if (faces==3) then
       line_answer=(1.0)/(power+1)
    else
       line_answer=-0.5*(1.0)/(power+1)
    end if


  end function line_answer

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

end subroutine test_element_surface_integral

