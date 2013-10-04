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
module polynomials
  !!< Module implementing real polynomials of one variable. Available
  !!< operations are addition, subtraction, multiplication and taking the
  !!< derivative.
  !!<
  !!< The polynomials can, of course, be evaluated at any point.
  !!<
  !!< Since polynomials have a pointer component, they must be explicitly
  !!< deallocated before they go out of scope or memory leaks will occur.
  !!<
  !!< The polynomial module also treats real vectors as
  !!< polynomials where the entries of the vector are read as the
  !!< coefficients of the polynomial. That is (/3.5, 1.0, 2.0/) is
  !!< 3.5*x**2+x+2.
  !!<
  !!< To allow polynomials to grow from the front of the coefs array, the
  !!< internal storage order of coefficients is the reverse of the external
  !!< storage order.
  !!<
  !!< To avoid memory leaks, all functions return real vectors which are then
  !!< turned into polynomials on assignment. Note however that at least one
  !!< operand of each operation must be a polynomial and not a vector acting
  !!< as a polynomial or the module won't know to do polynomial operations.
  !!<
  use futils
  use FLDebug

  implicit none

  type polynomial
     real, dimension(:), pointer :: coefs=>null()
     integer :: degree=-1
  end type polynomial

  interface assignment(=)
     module procedure assign_vec_poly, assign_poly_vec
  end interface

  interface operator(+)
     module procedure add_poly_poly, add_vec_poly, add_poly_vec
  end interface

  interface operator(-)
     module procedure subtract_poly_poly, subtract_vec_poly, &
          subtract_poly_vec
  end interface

  interface operator(-)
     module procedure unary_minus_poly
  end interface

  interface operator(*)
     module procedure mult_poly_poly, mult_poly_vec, mult_vec_poly, &
          mult_poly_scalar, mult_scalar_poly
  end interface

  interface operator(/)
     module procedure div_poly_scalar
  end interface

  interface ddx
     module procedure differentiate_poly, differentiate_vec
  end interface

  interface eval
     module procedure eval_poly_scalar, eval_poly_vector, &
          eval_vec_scalar, eval_vec_vector
  end interface

  interface deallocate
     module procedure deallocate_polynomial
  end interface

  private
  public :: polynomial, ddx, eval, deallocate, assignment(=), &
       operator(+), operator(-), operator(*), operator(/), poly2string,&
       & write_polynomial

contains

  subroutine assign_vec_poly(poly, vec)
    ! Assign a vector to a poly.
    type(polynomial), intent(inout) :: poly
    real, dimension(:), intent(in) :: vec

    call upsize(poly, size(vec)-1, preserve=.false.)

    poly%degree=size(vec)-1
    poly%coefs(:size(vec))=reverse(vec)

  end subroutine assign_vec_poly

  subroutine assign_poly_vec(vec, poly)
    ! Assign a vector to a poly.
    type(polynomial), intent(in) :: poly
    real, dimension(poly%degree+1), intent(out) :: vec

    vec=reverse(poly%coefs(:poly%degree+1))

  end subroutine assign_poly_vec

  function add_poly_poly(poly1, poly2) result (sum)
    ! Add two polynomials returning a vector.
    type(polynomial), intent(in) :: poly1, poly2
    real, dimension(max(poly1%degree,poly2%degree)+1) :: sum

    sum=0.0
    sum(:poly1%degree+1)=sum(:poly1%degree+1)&
         +poly1%coefs(:poly1%degree+1)
    sum(:poly2%degree+1)=sum(:poly2%degree+1)&
         +poly2%coefs(:poly2%degree+1)

    sum=reverse(sum)

  end function add_poly_poly

  function add_poly_vec(poly, vec) result (sum)
    ! Add two polynomials returning a vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vec
    real, dimension(max(poly%degree+1,size(vec))) :: sum

    sum=0.0
    sum(:poly%degree+1)=sum(:poly%degree+1)&
         +poly%coefs(:poly%degree+1)
    sum(:size(vec))=sum(:size(vec))+reverse(vec)

    sum=reverse(sum)

  end function add_poly_vec

  function add_vec_poly(vec, poly) result (sum)
    ! Add two polynomials returning a vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vec
    real, dimension(max(poly%degree+1,size(vec))) :: sum

    sum=0.0
    sum(:size(vec))=sum(:size(vec))+reverse(vec)
    sum(:poly%degree+1)=sum(:poly%degree+1)&
         +poly%coefs(:poly%degree+1)

    sum=reverse(sum)

  end function add_vec_poly

  function subtract_poly_poly(poly1, poly2) result (diff)
    ! Subtract two polynomials returning a vector.
    type(polynomial), intent(in) :: poly1, poly2
    real, dimension(max(poly1%degree,poly2%degree)+1) :: diff

    diff=0.0
    diff(:poly1%degree+1)=diff(:poly1%degree+1)&
         +poly1%coefs(:poly1%degree+1)
    diff(:poly2%degree+1)=diff(:poly2%degree+1)&
         -poly2%coefs(:poly2%degree+1)

    diff=reverse(diff)

  end function subtract_poly_poly

  function subtract_poly_vec(poly, vec) result (diff)
    ! Subtract two polynomials returning a vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vec
    real, dimension(max(poly%degree+1,size(vec))) :: diff

    diff=0.0
    diff(:poly%degree+1)=diff(:poly%degree+1)&
         +poly%coefs(:poly%degree+1)
    diff(:size(vec))=diff(:size(vec))-reverse(vec)

    diff=reverse(diff)

  end function subtract_poly_vec

  function subtract_vec_poly(vec, poly) result (diff)
    ! Subtract two polynomials returning a vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vec
    real, dimension(max(poly%degree+1,size(vec))) :: diff

    diff=0.0
    diff(:size(vec))=diff(:size(vec))+reverse(vec)
    diff(:poly%degree+1)=diff(:poly%degree+1)&
         -poly%coefs(:poly%degree+1)

    diff=reverse(diff)

  end function subtract_vec_poly

  function unary_minus_poly(poly) result(minus_poly)
    ! Calculate -1*poly. The result is a vector..
    type(polynomial), intent(in) :: poly
    real, dimension(poly%degree+1) :: minus_poly

    minus_poly=-reverse(poly%coefs(:poly%degree+1))

  end function unary_minus_poly

  function mult_poly_poly(poly1,poly2) result (product)
    ! Multiply two polynomials returning a vector.
    type(polynomial), intent(in) :: poly1, poly2
    real, dimension(poly1%degree+poly2%degree+1) :: product

    integer :: i

    ! In this algorithm, product is assembled in increasing power order and
    ! only reversed just before being returned.
    product=0.0

    do i=1, poly1%degree+1
       ! Standard long multiplication algorithm for polynomials.
       product(i:i+poly2%degree)=product(i:i+poly2%degree)+&
            poly1%coefs(i)*poly2%coefs(:poly2%degree+1)
    end do

    product=reverse(product)

  end function mult_poly_poly

  function mult_poly_vec(poly,vec) result (product)
    ! Multiply two polynomials returning a vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vec
    real, dimension(poly%degree+size(vec)) :: product

    integer :: i

    ! In this algorithm, product is assembled in increasing power order and
    ! only reversed just before being returned.
    product=0.0

    do i=1, size(vec)
       ! Standard long multiplication algorithm for polynomials.
       product(i:i+poly%degree)=product(i:i+poly%degree)+&
            vec(size(vec)+1-i)*poly%coefs(:poly%degree+1)
    end do

    product=reverse(product)

  end function mult_poly_vec

  function mult_vec_poly(vec,poly) result (product)
    ! Multiply two polynomials returning a vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vec
    real, dimension(poly%degree+size(vec)) :: product

    integer :: i

    ! In this algorithm, product is assembled in increasing power order and
    ! only reversed just before being returned.
    product=0.0

    do i=1, size(vec)
       ! Standard long multiplication algorithm for polynomials.
       product(i:i+poly%degree)=product(i:i+poly%degree)+&
            vec(size(vec)+1-i)*poly%coefs(:poly%degree+1)
    end do

    product=reverse(product)

  end function mult_vec_poly

  function mult_poly_scalar(poly, scalar) result (product)
    ! Multiply a polynomial by a scalar returning a vector.
    type(polynomial), intent(in) :: poly
    real, intent(in) :: scalar
    real, dimension(poly%degree+1) :: product

    product=poly
    product=product*scalar

  end function mult_poly_scalar

  function mult_scalar_poly(scalar, poly) result (product)
    ! Multiply a polynomial by a scalar returning a vector.
    type(polynomial), intent(in) :: poly
    real, intent(in) :: scalar
    real, dimension(poly%degree+1) :: product

    product=poly
    product=product*scalar

  end function mult_scalar_poly

  function div_poly_scalar(poly, scalar) result (quotient)
    ! Multiply a polynomial by a scalar returning a vector.
    type(polynomial), intent(in) :: poly
    real, intent(in) :: scalar
    real, dimension(poly%degree+1) :: quotient

    quotient=poly
    quotient=quotient/scalar

  end function div_poly_scalar

  function differentiate_poly(poly) result (diff)
    ! Differentiate a polynomial returning a vector.
    type(polynomial), intent(in) :: poly
    ! The derivative is always at least degree 0!
    real, dimension(max(poly%degree,1)) :: diff

    integer :: i

    ! This ensures that the degree 0 case is handled properly:
    diff=0.0

    forall (i=1:poly%degree)
       diff(i)=(poly%degree+1-i)*poly%coefs(poly%degree+2-i)
    end forall

  end function differentiate_poly

  function differentiate_vec(vec) result (diff)
    ! Differentiate a polynomial returning a vector.
    real, dimension(:), intent(in) :: vec
    real, dimension(size(vec)-1) :: diff

    integer :: i

    forall (i=1:size(vec)-1)
       diff(i)=(size(vec)-i)*vec(i)
    end forall

  end function differentiate_vec

  pure function eval_poly_scalar(poly, scalar) result (val)
    ! Evaluate poly(scalar) returning a scalar.
    type(polynomial), intent(in) :: poly
    real, intent(in) :: scalar
    real :: val

    integer :: i

    val=0.0

    do i=0,poly%degree

       val=val+poly%coefs(i+1)*scalar**i

    end do

  end function eval_poly_scalar

  pure function eval_poly_vector(poly, vector) result (val)
    ! Evaluate poly at each element of vector.
    type(polynomial), intent(in) :: poly
    real, dimension(:), intent(in) :: vector
    real, dimension(size(vector)) :: val

    integer :: i

    val=0.0

    do i=0,poly%degree

       val=val+poly%coefs(i+1)*vector**i

    end do

  end function eval_poly_vector

  pure function eval_vec_scalar(vec, scalar) result (val)
    ! Evaluate vec(scalar) interpreting vec as a vector and
    ! returning a scalar.
    real, dimension(:), intent(in) :: vec
    real, intent(in) :: scalar
    real :: val

    integer :: i

    val=0.0

    do i=0,size(vec)-1

       val=val+vec(size(vec)-i)*scalar**i

    end do

  end function eval_vec_scalar

  pure function eval_vec_vector(vec, vector) result (val)
    ! Evaluate vec interpreted  at each element of vector.
    real, dimension(:), intent(in) :: vec
    real, dimension(:), intent(in) :: vector
    real, dimension(size(vector)) :: val

    integer :: i

    val=0.0

    do i=0,size(vec)-1

       val=val+vec(size(vec)-i)*vector**i

    end do

  end function eval_vec_vector

  subroutine upsize(poly, degree, preserve)
    ! Ensure poly can handle entries of degree.
    ! This preserves existing data unless preserve is
    ! present and .false. in which case the polynomial is zeroed.
    type(polynomial), intent(inout) :: poly
    integer, intent(in) :: degree
    logical, intent(in), optional :: preserve

    real, dimension(:), pointer :: lcoefs

    if (associated(poly%coefs)) then
       if (size(poly%coefs)>=degree+1) then
          ! Nothing to do.
          if (present_and_false(preserve)) then
             poly%coefs=0.0
          end if
       end if
    end if

    lcoefs=>poly%coefs

    allocate(poly%coefs(degree+1))
    poly%coefs=0.0

    if (associated(lcoefs)) then
       ! Preserve existing data
       if (.not.present_and_false(preserve)) then
          poly%coefs(:poly%degree)=lcoefs
       end if

       deallocate(lcoefs)
    end if

  end subroutine upsize

  subroutine deallocate_polynomial(poly, stat)
    ! It's never necessary to allocate a polynomial since they
    ! automagically acquire the right size but it is necessary to
    ! deallocate them to prevent memory leaks.
    type(polynomial), intent(inout) :: poly
    integer, intent(out), optional :: stat

    integer :: lstat

    deallocate(poly%coefs, stat=lstat)

    poly%degree=-1

    if (present(stat)) then
       stat=lstat
    elseif (lstat/=0) then
       FLAbort("Failed to deallocate polynomial")
    end if

  end subroutine deallocate_polynomial

  pure function reverse(vec)
    ! This function reverses the elements of vec. This is useful because
    ! poly%coefs is in ascending order of power of x while vectors are
    ! taken to be in descending order.
    real, dimension(:), intent(in) :: vec
    real, dimension(size(vec)) :: reverse

    reverse=vec(size(vec):1:-1)

  end function reverse

  subroutine write_polynomial(unit, poly, format)
    !!< Output polynomial on unit. If format is present it specifies the
    !!< format of coefficients, otherwise f8.3 is used.
    integer, intent(in) :: unit
    type(polynomial), intent(in) :: poly
    character(len=*), intent(in), optional :: format
    character(len=poly%degree*20+20) :: string

    string=poly2string(poly, format)

    write(unit, "(a)") trim(string)

  end subroutine write_polynomial

  function poly2string(poly, format) result (string)
    !!< Produce a string representation of poly in which the coefficients
    !!< have the format given by format, if present, and f8.3 otherwise.
    type(polynomial), intent(in) :: poly
    character(len=*), intent(in), optional :: format
    character(len=poly%degree*20+20) :: string

    character(len=1000) :: outformat
    character(len=20) :: lformat
    integer :: i

    type real_integer
       !!< Local type with one real and one integer for io purposes.
       real :: r
       integer :: i
    end type real_integer

    type(real_integer), dimension(poly%degree-1) :: r_i

    if (present(format)) then
       lformat=format
    else
       lformat='f8.3'
    end if

    forall (i=1:poly%degree-1) r_i(i)%i=i+1

    r_i%r=poly%coefs(3:)

    ! Special case degree 0 and 1 polynomials
    select case(poly%degree)
    case (0)
       write(string, "("//lformat//")") poly%coefs(1)
    case (1)
       write(string, "("//lformat//",'x + ',"//lformat//")") reverse(poly%coefs)
    case default
       write(outformat, "(a,i0,a)") "(",poly%degree-1,"("//trim(lformat) &
        //",'x^',i0,' + ')"//trim(lformat)//",'x + ',"//trim(lformat)//")"

       write(string, outformat) r_i(size(r_i):1:-1), poly%coefs(2), poly%coefs(1)
    end select

  end function poly2string

end module polynomials
