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

subroutine test_gm_quadrature
  use quadrature_test
  use unittest_tools
  implicit none

  integer :: dim, vertices
  logical :: fail
  integer :: degree, stat
  type(quadrature_type) :: quadrature
  character(len=254) :: error_message, test_message

  dim = 2
  vertices = 3

  do dim=1,3
    vertices = dim+1
    degree = 0
    degreeloop: do
      degree = degree + 1

      quadrature = make_quadrature(vertices, dim, degree=degree, family=FAMILY_GM, stat=stat)

      select case (stat)
      case (QUADRATURE_DEGREE_ERROR)
        exit degreeloop
      case (0)
        continue
      case default
        fail = .true.
        call report_test("[test_gm_quadrature]", fail, .false., "Making quadrature failed")
      end select

      degree = quadrature%degree

      do power=0,degree
        if(fnequals(quad_integrate(monic, quadrature), simplex_answer(), tol = 1.0e5 * epsilon(0.0))) then
          fail = .true.
          write(error_message, '(e15.7)') quad_integrate(monic, quadrature)-simplex_answer()
        else
          fail = .false.
          error_message = ""
        end if

        write(test_message, '(3(a,i0),a)') "[",dim,"-simplex, quad degree ",degree," power ",power," ]"
        call report_test(trim(test_message), fail, .false., trim(error_message))
      end do

      call deallocate(quadrature)
    end do degreeloop
  end do

  contains
    function simplex_answer()
      ! Analytic solution to integrating monic over a simplex.
      ! This formula is eq. 7.38 and 7.48 in Zienkiewicz and Taylor
      real :: simplex_answer
      integer :: i, j

      simplex_answer = 1.0
      do i=0,dim-1
        j = power + dim - i
        if (j <= 1) exit
        simplex_answer = simplex_answer * j
      end do
      simplex_answer = 1.0/simplex_answer

    end function simplex_answer
end subroutine test_gm_quadrature

