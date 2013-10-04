#include "fdebug.h"
#define INLINE_MATMUL
module fetools
  !!< Module containing general tools for discretising Finite Element problems.

  use elements
  use transform_elements
  use fields_base
  implicit none

  !! X, Y and Z indices.
  integer, parameter :: X_=1,Y_=2,Z_=3
  !! U, V and W indices.
  integer, parameter :: U_=1,V_=2,W_=3

  !! Huge number useful for effectively zeroing out rows of a matrix.
  real, parameter :: INFINITY=huge(0.0)*epsilon(0.0)

  interface norm2
     module procedure norm2_element
  end interface

  interface integral_element
     module procedure integral_element_scalar, integral_element_vector, integral_element_scalars
  end interface

  private :: norm2_element
  private :: integral_element_scalar, integral_element_vector, integral_element_scalars

contains

  function shape_rhs(shape, detwei)
    !!<             /
    !!<   Calculate | shape detwei dV
    !!<             /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    type(element_type), intent(in) :: shape
    real, dimension(shape%ngi), intent(in) :: detwei

    real, dimension(shape%ndof) :: shape_rhs

    shape_rhs=matmul(shape%n, detwei)

  end function shape_rhs

  function shape_vector_rhs(shape,vector,detwei)
    !!<            /
    !!<  Calculate | shape vector detwei dV
    !!<            /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    type(element_type), intent(in) :: shape
    real, dimension(shape%ngi), intent(in) :: detwei
    !! vector is dim x ngi
    real, dimension(:,:), intent(in) :: vector

    real, dimension(size(vector,1),shape%ndof) :: shape_vector_rhs

    integer :: dim,i

    assert(size(vector,2)==shape%ngi)

    dim = size(vector,1)
    forall(i=1:dim)
       shape_vector_rhs(i,:)=matmul(shape%n, detwei * vector(i,:))
    end forall

  end function shape_vector_rhs

  function shape_tensor_rhs(shape,tensor,detwei)
    !!<            /
    !!<  Calculate | shape tensor detwei dV
    !!<            /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    type(element_type), intent(in) :: shape
    real, dimension(shape%ngi), intent(in) :: detwei
    !! tensor is dim1 x dim2 x ngi
    real, dimension(:,:,:), intent(in) :: tensor

    real, dimension(size(tensor,1), size(tensor, 2), shape%ndof) :: shape_tensor_rhs

    integer :: dim1,dim2,i, j

    assert(size(tensor,3)==shape%ngi)
    shape_tensor_rhs = 0.0

    dim1 = size(tensor,1)
    dim2 = size(tensor,2)
    forall(i=1:dim1)
      forall(j=1:dim2)
        shape_tensor_rhs(i,j,:)=matmul(shape%n, detwei * tensor(i,j,:))
      end forall
    end forall

  end function shape_tensor_rhs

  function shape_tensor_dot_vector_rhs(shape,tensor,vector,detwei)
    !!<            /
    !!<  Calculate | shape tensor detwei dV
    !!<            /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    type(element_type), intent(in) :: shape ! shape%n is loc x ngi
    real, dimension(shape%ngi), intent(in) :: detwei
    real, dimension(:,:,:), intent(in) :: tensor !dim1 x dim2 x ngi
    real, dimension(:,:) :: vector  !dim2 x ngi

    real, dimension(size(tensor,1), shape%ndof) :: shape_tensor_dot_vector_rhs

    integer :: dim,i, ngi, j

    assert(size(tensor,3)==shape%ngi)
    assert(size(tensor,2)==size(vector,1))

    dim = size(tensor,1)
    ngi = shape%ngi
    shape_tensor_dot_vector_rhs = 0.0
    do i=1,dim
      do j=1,ngi
        shape_tensor_dot_vector_rhs(i,:)=shape_tensor_dot_vector_rhs(i,:) + &
                                shape%n(:,j)*sum(tensor(i,:,j)*vector(:,j))*detwei(j)
      end do
    end do

  end function shape_tensor_dot_vector_rhs

  function dshape_dot_vector_rhs(dshape, vector, detwei)
    !!<             /
    !!<   Calculate | dshape detwei dV
    !!<             /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    real, dimension(:,:,:), intent(in) :: dshape
    !! vector is dim x ngi
    real, dimension(:,:), intent(in) :: vector
    real, dimension(:), intent(in) :: detwei

    real, dimension(size(dshape,1)) :: dshape_dot_vector_rhs
    integer :: ix,loc

    dshape_dot_vector_rhs=0.0
    loc=size(dshape,1)

    forall(ix=1:loc)
       dshape_dot_vector_rhs(ix)=sum(sum(dshape(ix,:,:)&
            *transpose(vector),2)*detwei,1)
    end forall

  end function dshape_dot_vector_rhs

  function dshape_dot_tensor_rhs(dshape,tensor,detwei)
    !!<            /
    !!<  Calculate | dshape_dxj tensor_ij dV
    !!<            /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    real, dimension(:,:,:), intent(in) :: dshape ! loc x ngi x dim2
    real, dimension(:), intent(in) :: detwei ! ngi
    real, dimension(:,:,:), intent(in) :: tensor ! dim1 x dim2 x ngi

    real, dimension(size(tensor,1),size(dshape,1)) :: dshape_dot_tensor_rhs

    integer :: dim,i,ngi,j

    assert(size(tensor,3)==size(detwei))
    assert(size(tensor,2)==size(dshape,3))
    dim = size(tensor,1)
    ngi = size(detwei)

    dshape_dot_tensor_rhs = 0.0
    do i = 1, dim
      do j = 1, ngi
        dshape_dot_tensor_rhs(i,:) = dshape_dot_tensor_rhs(i,:) +&
          matmul(tensor(i,:,j), transpose(dshape(:,j,:)))*detwei(j)
      end do
    end do

  end function dshape_dot_tensor_rhs

  function shape_shape(shape1, shape2, detwei)
    !!< For each node in each element shape1, shape2 calculate the
    !!< coefficient of the integral int(shape1shape2)dV.
    !!<
    !!< In effect, this calculates a mass matrix.
    type(element_type), intent(in) :: shape1, shape2
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape1%ngi), intent(in) :: detwei

    real, dimension(shape1%ndof,shape2%ndof) :: shape_shape

    integer :: iloc, jloc

    forall(iloc=1:shape1%ndof,jloc=1:shape2%ndof)
       ! Main mass matrix.
       shape_shape(iloc,jloc)=&
            dot_product(shape1%n(iloc,:)*shape2%n(jloc,:),detwei)
    end forall

  end function shape_shape

  function shape_shape_vector(shape1, shape2, detwei, vector)
    !!< For each node in each element shape1, shape2 calculate the
    !!< coefficient of the integral int(shape1shape2)vectordV.
    type(element_type), intent(in) :: shape1, shape2
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape1%ngi), intent(in) :: detwei
    !! dim x ngi list of vectors.
    real, dimension(:,:), intent(in) :: vector

    real, dimension(size(vector,1),shape1%ndof,shape2%ndof) ::&
         & shape_shape_vector

    integer :: iloc,jloc
    integer :: dim

    dim=size(vector,1)

    !      assert(size(vector,2)==shape1%ngi)

    forall(iloc=1:shape1%ndof,jloc=1:shape2%ndof)
       ! Main mass matrix.
       shape_shape_vector(:,iloc,jloc)=&
            matmul(vector*spread(shape1%n(iloc,:)*shape2%n(jloc,:),1,dim),detwei)
    end forall

  end function shape_shape_vector

  function shape_shape_tensor(shape1, shape2, detwei, tensor)
    !!< For each node in each element shape1, shape2 calculate the
    !!< coefficient of the integral int(shape1shape2)tensor dV.
    type(element_type), intent(in) :: shape1, shape2
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape1%ngi), intent(in) :: detwei
    real, dimension(:,:,:), intent(in) :: tensor

    real, dimension(size(tensor,1),size(tensor,2),shape1%ndof,shape2%ndof) ::&
         & shape_shape_tensor

    integer :: iloc,jloc,i,j

    assert(size(tensor,3)==shape1%ngi)

    forall(iloc=1:shape1%ndof,jloc=1:shape2%ndof,i=1:size(tensor,1),j=1:size(tensor,2))
       ! Main mass matrix.
       shape_shape_tensor(i,j,iloc,jloc)=&
            sum(shape1%n(iloc,:)*shape2%n(jloc,:)*tensor(i,j,:)*detwei)
    end forall

  end function shape_shape_tensor

  function shape_shape_vector_outer_vector(shape1, shape2, detwei,vector1,vector2)
    !!< For each node in each element shape1, shape2 calculate the
    !!< coefficient of the integral int(shape1shape2)vector outer vector dV.
    !!
    type(element_type), intent(in) :: shape1, shape2
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape1%ngi), intent(in) :: detwei
    real, dimension(:,:), intent(in) :: vector1
    real, dimension(:,:), intent(in) :: vector2

    real, dimension(size(vector1,1),size(vector2,1),shape1%ndof,shape2%ndof) ::&
         & shape_shape_vector_outer_vector

    integer :: iloc,jloc,i,j

    assert(size(vector1,2)==shape1%ngi)
    assert(size(vector2,2)==shape1%ngi)

    forall(iloc=1:shape1%ndof,jloc=1:shape2%ndof,i=1:size(vector1,1),j=1:size(vector2,1))
       ! Main mass matrix.
       shape_shape_vector_outer_vector(i,j,iloc,jloc)=&
            sum(shape1%n(iloc,:)*shape2%n(jloc,:)*vector1(i,:)* &
            vector2(j,:)*detwei)
    end forall

  end function shape_shape_vector_outer_vector

  function dshape_rhs(dshape, detwei)
    !!<            /
    !!<  Calculate | dshape detwei dV
    !!<            /
    !!<
    !!< Note that this is an integral of a single shape function. This is
    !!< primarily useful for evaluating righthand sides of equations where a
    !!< function has been evaluated at the quadrature points and incorporated
    !!< with detwei.
    real, dimension(:,:,:), intent(in) :: dshape !loc * ngi * dim
    real, dimension(size(dshape,2)), intent(in) :: detwei !ngi

    real, dimension(size(dshape,3),size(dshape,1)) :: dshape_rhs !dim * loc

    integer :: dim,i

    dim = size(dshape,3)

    forall(i=1:dim)
       dshape_rhs(i,:)=matmul(dshape(:,:,i),detwei)
    end forall

  end function dshape_rhs

  function shape_dshape(shape, dshape, detwei)
    !!< For each node in element shape and transformed gradient dshape,
    !!< calculate the coefficient of the integral int(shape dshape)dV.
    type(element_type), intent(in) :: shape
    !! The dimensions of dshape are:
    !!  (nodes, gauss points, dimensions)
    real, dimension(:,:,:), intent(in) :: dshape
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape%ngi), intent(in) :: detwei

    real, dimension(size(dshape,3), shape%ndof, size(dshape,1)) :: shape_dshape

    integer :: iloc,jloc
    integer :: dshape_loc, dim, idim

    dshape_loc=size(dshape,1)
    dim=size(dshape,3)

#ifdef INLINE_MATMUL
    forall(iloc=1:shape%ndof,jloc=1:dshape_loc,idim=1:dim)
       shape_dshape(idim,iloc,jloc)= sum(detwei * dshape(jloc,:,idim) * shape%n(iloc,:))
    end forall
#else
    forall(iloc=1:shape%ndof,jloc=1:dshape_loc)
       ! Main matrix.
       shape_dshape(1:dim,iloc,jloc)= &
            matmul(detwei,spread(shape%n(iloc,:),2,dim)*dshape(jloc,:,:))
    end forall
#endif
  end function shape_dshape

  function dshape_shape(dshape, shape, detwei)
    !!< For each node in element shape and transformed gradient dshape,
    !!< calculate the coefficient of the integral int(dshape shape)dV.
    type(element_type), intent(in) :: shape
    !! The dimensions of dshape are:
    !!  (nodes, gauss points, dimensions)
    real, dimension(:,:,:), intent(in) :: dshape
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape%ngi), intent(in) :: detwei

    real, dimension(size(dshape,3),size(dshape,1),shape%ndof) :: dshape_shape

    integer :: iloc,jloc
    integer :: dshape_loc, dim

    dshape_loc=size(dshape,1)
    dim=size(dshape,3)

    forall(iloc=1:dshape_loc,jloc=1:shape%ndof)
       ! Main matrix.
       dshape_shape(1:dim,iloc,jloc)= &
            matmul(detwei,dshape(iloc,:,1:dim)*spread(shape%n(jloc,:),2,dim))
    end forall

  end function dshape_shape

  function dshape_dot_dshape(dshape1, dshape2, detwei) result (R)
    !!<            /
    !!<  Evaluate: |(Grad N1)' dot (Grad N2) dV For shapes N1 and N2.
    !!<            /
    real, dimension(:,:,:), intent(in) :: dshape1, dshape2
    real, dimension(size(dshape1,2)) :: detwei

    real, dimension(size(dshape1,1),size(dshape2,1)) :: R

    integer :: iloc,jloc, gi
    integer :: loc1, loc2, ngi, dim

    loc1=size(dshape1,1)
    loc2=size(dshape2,1)
    ngi=size(dshape1,2)
    dim=size(dshape1,3)

    assert(loc1==loc2)

    R=0.0

    select case(dim)
      case(3)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * dshape2(jloc,gi,1)) &
                  & + (dshape1(iloc,gi,2) * dshape2(jloc,gi,2)) &
                  & + (dshape1(iloc,gi,3) * dshape2(jloc,gi,3)) &
                & ) * detwei(gi)
           end forall
        end do
      case(2)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * dshape2(jloc,gi,1)) &
                  & + (dshape1(iloc,gi,2) * dshape2(jloc,gi,2)) &
                & ) * detwei(gi)
           end forall
        end do
      case(1)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * dshape2(jloc,gi,1)) &
                & ) * detwei(gi)
           end forall
        end do
      case default
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                   +dot_product(dshape1(iloc,gi,:),dshape2(jloc,gi,:))&
                   *detwei(gi)
           end forall
        end do
    end select

  end function dshape_dot_dshape

  function shape_vector_outer_dshape(&
       shape,vector,dshape,detwei) result (tensor)
    !!< For each node in element shape and transformed gradient dshape,
    !!< calculate the coefficient of the integral
    !!<
    !!<  Q_ij = int(shape vector_i dshape_j)dV.
    type(element_type), intent(in) :: shape
    !! The dimensions of dshape are:
    !!  (nodes, gauss points, dimensions)
    real, dimension(:,:,:), intent(in) :: dshape
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape%ngi), intent(in) :: detwei
    real, dimension(:,:), intent(in) :: vector
    real, dimension(size(vector,1),size(dshape,3), &
         shape%ndof,size(dshape,1)) :: tensor

    integer :: iloc,jloc,i,j
    integer :: dshape_loc, dim1,dim2

    dshape_loc=size(dshape,1)
    dim1=size(dshape,3)
    dim2=size(vector,1)
    assert(size(vector,2)==shape%ngi)

    forall(iloc=1:shape%ndof,jloc=1:dshape_loc,i = 1:dim1,j = 1:dim2)
       ! Main matrix.
       tensor(i,j,iloc,jloc)= &
            sum(shape%n(iloc,:)*dshape(jloc,:,j)*detwei*vector(i,:))
    end forall

  end function shape_vector_outer_dshape

  function dshape_outer_vector_shape( &
       dshape,vector,shape,detwei) result (tensor)
    !!< For each node in element shape and transformed gradient dshape,
    !!< calculate the coefficient of the integral
    !!<
    !!<  Q_ij = int(shape vector_i dshape_j)dV.
    type(element_type), intent(in) :: shape
    !! The dimensions of dshape are:
    !!  (nodes, gauss points, dimensions)
    real, dimension(:,:,:), intent(in) :: dshape
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(shape%ngi), intent(in) :: detwei
    real, dimension(:,:), intent(in) :: vector
    real, dimension(size(dshape,3),size(vector,1), &
         shape%ndof,size(dshape,1)) :: tensor

    integer :: iloc,jloc,i,j
    integer :: dshape_loc, dim1,dim2

    dshape_loc=size(dshape,1)
    dim1=size(dshape,3)
    dim2=size(vector,1)
    assert(size(vector,2)==shape%ngi)

    forall(iloc=1:dshape_loc,jloc=1:shape%ndof,i = 1:dim1,j = 1:dim2)
       ! Main matrix.
       tensor(i,j,iloc,jloc)= &
            sum(dshape(iloc,:,i)*detwei*vector(j,:)*shape%n(jloc,:))
    end forall

  end function dshape_outer_vector_shape

  function dshape_outer_dshape(dshape1, dshape2, detwei) result (R)
    !!< For each node in each transformed gradient dshape1, dshape2,
    !!< calculate the coefficient of the integral int(dshape outer dshape)dV.

    !! The dimensions of dshape are:
    !!  (nodes, gauss points, dimensions)
    real, dimension(:,:,:), intent(in) :: dshape1,dshape2
    !! The gauss weights transformed by the coordinate transform
    !! from real to computational space.
    real, dimension(size(dshape1,2)) :: detwei
    real, dimension(size(dshape1,3),size(dshape2,3),size(dshape1,1),size(dshape2,1)) :: R

    integer :: iloc,jloc, i,j
    integer :: loc1, loc2, ngi, dim1, dim2

    loc1=size(dshape1,1)
    loc2=size(dshape2,1)
    ngi=size(dshape1,2)
    dim1=size(dshape1,3)
    dim2=size(dshape2,3)

    R=0.0

    forall(iloc=1:loc1,jloc=1:loc2,i=1:dim1,j=1:dim2)
       r(i,j,iloc,jloc)=r(i,j,iloc,jloc) &
            +sum(dshape1(iloc,:,i)*dshape2(jloc,:,j)*detwei)
    end forall

  end function dshape_outer_dshape

  function dshape_diagtensor_dshape(dshape1, tensor, dshape2, detwei) result (R)
    !!<
    !!< Evaluate: (Grad N1)' diag(T) (Grad N2) For shape N and tensor T.
    !!<
    real, dimension(:,:,:), intent(in) :: dshape1, dshape2
    real, dimension(size(dshape1,3),size(dshape1,3),size(dshape1,2)), intent(in) :: tensor
    real, dimension(size(dshape1,2)) :: detwei

    real, dimension(size(dshape1,1),size(dshape2,1)) :: R

    real, dimension(size(dshape1,3),size(dshape1,2)) :: diag_tensor
    integer :: iloc,jloc, gi
    integer :: loc1, loc2, ngi, dim

    loc1=size(dshape1,1)
    loc2=size(dshape2,1)
    ngi=size(dshape1,2)
    dim=size(dshape1,3)

    assert(loc1==loc2)

    R=0.0

    select case(dim)
      case(3)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * tensor(1,1,gi) * dshape2(jloc,gi,1)) &
                  & + (dshape1(iloc,gi,2) * tensor(2,2,gi) * dshape2(jloc,gi,2)) &
                  & + (dshape1(iloc,gi,3) * tensor(3,3,gi) * dshape2(jloc,gi,3)) &
                & ) * detwei(gi)
           end forall
        end do
      case(2)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * tensor(1,1,gi) * dshape2(jloc,gi,1)) &
                  & + (dshape1(iloc,gi,2) * tensor(2,2,gi) * dshape2(jloc,gi,2)) &
                & ) * detwei(gi)
           end forall
        end do
      case default
        diag_tensor = 0.0
        forall(iloc=1:dim)
          diag_tensor(iloc,:) = tensor(iloc,iloc,:)
        end forall
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                   +dot_product(dshape1(iloc,gi,:)*diag_tensor(:,gi),dshape2(jloc,gi,:))&
                   *detwei(gi)
           end forall
        end do
    end select

  end function dshape_diagtensor_dshape

  function dshape_vector_dshape(dshape1, vector, dshape2, detwei) result (R)
    !!<
    !!< Evaluate: (Grad N1)' V (Grad N2) For shape N and vector V.
    !!<
    real, dimension(:,:,:), intent(in) :: dshape1, dshape2
    real, dimension(size(dshape1,3),size(dshape1,2)), intent(in) :: vector
    real, dimension(size(dshape1,2)) :: detwei

    real, dimension(size(dshape1,1),size(dshape2,1)) :: R

    integer :: iloc,jloc, gi
    integer :: loc1, loc2, ngi, dim

    loc1=size(dshape1,1)
    loc2=size(dshape2,1)
    ngi=size(dshape1,2)
    dim=size(dshape1,3)

    assert(loc1==loc2)

    R=0.0

    select case(dim)
      case(3)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * vector(1,gi) * dshape2(jloc,gi,1)) &
                  & + (dshape1(iloc,gi,2) * vector(2,gi) * dshape2(jloc,gi,2)) &
                  & + (dshape1(iloc,gi,3) * vector(3,gi) * dshape2(jloc,gi,3)) &
                & ) * detwei(gi)
           end forall
        end do
      case(2)
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                & + ( &
                  & (dshape1(iloc,gi,1) * vector(1,gi) * dshape2(jloc,gi,1)) &
                  & + (dshape1(iloc,gi,2) * vector(2,gi) * dshape2(jloc,gi,2)) &
                & ) * detwei(gi)
           end forall
        end do
      case default
        do gi=1,ngi
           forall(iloc=1:loc1,jloc=1:loc2)
              r(iloc,jloc)=r(iloc,jloc) &
                   +dot_product(dshape1(iloc,gi,:)*vector(:,gi),dshape2(jloc,gi,:))&
                   *detwei(gi)
           end forall
        end do
    end select

  end function dshape_vector_dshape

  function dshape_tensor_dshape(dshape1, tensor, dshape2, detwei) result (R)
    !!<
    !!< Evaluate: (Grad N1)' T (Grad N2) For shape N and tensor T.
    !!<
    real, dimension(:,:,:), intent(in) :: dshape1, dshape2
    real, dimension(size(dshape1,3),size(dshape1,3),size(dshape1,2)), intent(in) :: tensor
    real, dimension(size(dshape1,2)) :: detwei

    real, dimension(size(dshape1,1),size(dshape2,1)) :: R

    integer :: iloc,jloc, gi
    integer :: loc1, loc2, ngi, dim

    loc1=size(dshape1,1)
    loc2=size(dshape2,1)
    ngi=size(dshape1,2)
    dim=size(dshape1,3)

    assert(loc1==loc2)

    R=0.0

    do gi=1,ngi
       forall(iloc=1:loc1,jloc=1:loc2)
          r(iloc,jloc)=r(iloc,jloc) &
               +dot_product(matmul(dshape1(iloc,gi,:), tensor(:,:,gi)),&
               &       dshape2(jloc,gi,:))*detwei(gi)
       end forall
    end do

  end function dshape_tensor_dshape

  function dshape_dot_vector_shape(dshape, vector, shape, detwei) result (R)
    !!<
    !!< Evaluate (Grad N1 dot vector) (N2)
    !!<
    real, dimension(:,:,:), intent(in) :: dshape
    real, dimension(size(dshape,3),size(dshape,2)), intent(in) :: vector
    type(element_type), intent(in) :: shape
    real, dimension(size(dshape,2)) :: detwei

    real, dimension(size(dshape,1),shape%ndof) :: R

    integer :: iloc,jloc
    integer :: dshape_loc

    dshape_loc=size(dshape,1)

    forall(iloc=1:dshape_loc,jloc=1:shape%ndof)
       R(iloc,jloc)= dot_product(sum(dshape(iloc,:,:)*transpose(vector),2) &
            *shape%n(jloc,:), detwei)
    end forall

  end function dshape_dot_vector_shape

  function dshape_dot_tensor_shape(dshape, tensor, shape, detwei) result (R)
    !!<          /
    !!< Evaluate | (Grad N1 dot tensor) (N2)
    !!<          /
    real, dimension(:,:,:), intent(in) :: dshape ! nloc1 x ngi x dim1
    real, dimension(:,:,:), intent(in) :: tensor ! dim1 x dim2 x ngi
    type(element_type), intent(in) :: shape
    real, dimension(size(dshape,2)) :: detwei

    real, dimension(size(tensor,2),size(dshape,1),shape%ndof) :: R

    integer :: iloc,jloc, idim2
    integer :: dshape_loc, dim2


    dshape_loc=size(dshape,1)
    dim2=size(tensor,2)

    forall(iloc=1:dshape_loc,jloc=1:shape%ndof,idim2=1:dim2)
      R(idim2,iloc,jloc)=dot_product(sum( dshape(iloc,:,:)* transpose(tensor(:,idim2,:)) ) &
            *shape%n(jloc,:), detwei)
    end forall


  end function dshape_dot_tensor_shape

  function shape_vector_dot_dshape(shape, vector, dshape, detwei) result (R)
    !!<
    !!< Evaluate (Grad N1 dot vector) (N2)
    !!<
   type(element_type), intent(in) :: shape
    real, dimension(:,:,:), intent(in) :: dshape
    real, dimension(size(dshape,3),size(dshape,2)), intent(in) :: vector
    real, dimension(size(dshape,2)) :: detwei

    real, dimension(shape%ndof, size(dshape,1)) :: R

    integer :: iloc,jloc
    integer :: dshape_loc, dim


    dshape_loc=size(dshape,1)
    dim=size(dshape,3)

    forall(iloc=1:shape%ndof,jloc=1:dshape_loc)
       R(iloc,jloc)= dot_product(shape%n(iloc,:) * &
            sum(dshape(jloc,:,:)*transpose(vector),2), detwei)
    end forall

  end function shape_vector_dot_dshape

  function shape_curl_shape_2d(shape, dshape, detwei) result (R)
    !!<            /
    !!<  Evaluate: |(N1)(Curl N2) dV For shapes N1 and N2.
    !!<            /
    !!< Note that curl is a dimension-specific operator so this version
    !!< only makes sense for 2D.
    type(element_type), intent(in) :: shape
    real, dimension(:,:,:), intent(in) :: dshape
    real, dimension(size(dshape,2)) :: detwei

    real, dimension(2,shape%ndof,size(dshape,1)) :: R

    integer :: iloc,jloc
    integer :: dshape_loc, dim

    dshape_loc=size(dshape,1)
    dim=size(dshape,3)

    assert(dim==2)

    forall(iloc=1:shape%ndof,jloc=1:dshape_loc)
       R(1,iloc,jloc)= dot_product(shape%n(iloc,:) * &
            dshape(jloc,:,2), detwei)

       R(2,iloc,jloc)= -dot_product(shape%n(iloc,:) * &
            dshape(jloc,:,1), detwei)
    end forall

  end function shape_curl_shape_2d

  function norm2_element(field, X, ele) result (norm)
    !!< Return the l2 norm of field on the given element.
    real :: norm
    ! Element values at the nodes.
    type(scalar_field), intent(in) :: field
    ! Shape of field elements.
    type(element_type), pointer :: field_shape
    ! The positions of the nodes in this element.
    type(vector_field), intent(in) :: X
    ! The number of the element to operate on.
    integer, intent(in) :: ele

    real, dimension(ele_ngi(field,ele)) :: detwei

    real, dimension(ele_loc(field,ele)) :: field_val

    field_val=ele_val(field, ele)

    field_shape=>ele_shape(field, ele)

    call transform_to_physical(X, ele, detwei=detwei)

    norm = dot_product(field_val, matmul(&
         &  shape_shape(field_shape, field_shape, detwei)&
         &                                               ,field_val))

  end function norm2_element

  function integral_element_scalar(field, X, ele) result&
       & (integral)
    !!< Return the integral of field over the given element.
    real :: integral
    ! Element values at the nodes.
    type(scalar_field), intent(in) :: field
    ! The positions of the nodes in this element.
    type(vector_field), intent(in) :: X
    ! The number of the current element
    integer, intent(in) :: ele

    real, dimension(ele_ngi(field, ele)) :: detwei

    call transform_to_physical(X, ele, detwei=detwei)

    integral=dot_product(ele_val_at_quad(field, ele), detwei)

  end function integral_element_scalar

  function integral_element_vector(field, X, ele) result&
       & (integral)
    !!< Return the integral of field over the given element_vector.
    ! Element values at the nodes.
    type(vector_field), intent(in) :: field
    ! The positions of the nodes in this element
    type(vector_field), intent(in) :: X
    ! The number of the current element
    integer, intent(in) :: ele

    real, dimension(field%dim) :: integral

    real, dimension(ele_ngi(field, ele)) :: detwei

    call transform_to_physical(X, ele, detwei=detwei)

    integral=matmul(matmul(ele_val(field, ele), field%mesh%shape%n), detwei)

  end function integral_element_vector

  function integral_element_scalars(fields, X, ele) result&
       & (integral)
    !!< Return the integral of the product of fields over the given element.
    real :: integral
    ! Element values at the nodes.
    type(scalar_field_pointer), dimension(:), intent(in) :: fields
    ! The positions of the nodes in this element.
    type(vector_field), intent(in) :: X
    ! The number of the current element
    integer, intent(in) :: ele

    integer :: s
    real, dimension(ele_ngi(fields(1)%ptr, ele)) :: detwei
    real, dimension(ele_ngi(fields(1)%ptr, ele)) :: product_ele_val_at_quad

    call transform_to_physical(X, ele, detwei=detwei)

    do s = 1,size(fields)

       assert(ele_ngi(X,ele) == ele_ngi(fields(s)%ptr,ele))

       if (s == 1) then

          product_ele_val_at_quad = ele_val_at_quad(fields(s)%ptr, ele)

       else

          product_ele_val_at_quad = product_ele_val_at_quad * &
                                    ele_val_at_quad(fields(s)%ptr, ele)

       end if

    end do

    integral=dot_product(product_ele_val_at_quad, detwei)

  end function integral_element_scalars

!!$  subroutine lump(mass)
!!$    !!< lump mass.
!!$    real, dimension(:,:), intent(inout) :: mass
!!$
!!$    integer :: i, j
!!$
!!$    ! Check that matrix is square.
!!$    ASSERT(size(mass,1)==size(mass,2))
!!$
!!$    forall(i=1:size(mass,1))
!!$       mass(i,i)=sum(mass(i,:))
!!$    end forall
!!$
!!$    forall(i=1:size(mass,1),j=1:size(mass,1), i/=j)
!!$       mass(i,j)=0.0
!!$    end forall
!!$
!!$  end subroutine lump

  function lumped(mass)
    !!< lumped mass.
    real, dimension(:,:), intent(in) :: mass
    real, dimension(size(mass,1),size(mass,2)) :: lumped

    integer :: i

    ! Check that matrix is square.
    ASSERT(size(mass,1)==size(mass,2))

    lumped=0

    forall(i=1:size(mass,1))
       lumped(i,i)=sum(mass(i,:))
    end forall

  end function lumped

end module fetools
