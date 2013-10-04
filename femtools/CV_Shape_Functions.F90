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
module cv_shape_functions
  !!< Generate shape functions for elements of arbitrary polynomial degree.
  use FLDebug
  use cv_faces, only: cv_faces_type
  use shape_functions
  implicit none

  private
  public :: make_cv_element_shape, make_cvbdy_element_shape

contains

    function make_cv_element_shape(cvfaces, parentshape, type, stat) result (shape)

      type(element_type) :: shape

      type(cv_faces_type), intent(in) :: cvfaces
      type(element_type), intent(in) :: parentshape
      integer, intent(in), optional :: type
      integer, intent(out), optional :: stat

      type(ele_numbering_type), pointer :: ele_num
      type(quadrature_type) :: quad
      type(element_type) :: tempshape

      integer :: i, j, k, gi
      integer :: ngi, coords, fdim, dim, loc, faces, nodes
      integer :: ltype

      real, dimension(:,:,:), allocatable :: dl

      if(present(stat)) stat = 0

      if(present(type)) then
        ltype = type
      else
        ltype = ELEMENT_CONTROLVOLUME_SURFACE
      end if

      ! some useful numbers
      ngi = cvfaces%shape%ngi*cvfaces%faces
                                  ! number of gauss points in parent element
      loc = cvfaces%vertices          ! vertices of parent element
      coords = cvfaces%coords    ! number of canonical coordinates in parent element
      faces = cvfaces%faces      ! number of faces
      fdim = cvfaces%dim-1       ! dimension of faces
      dim = cvfaces%dim          ! dimension of parent element

      ! Create the quadrature for the parent element
      call allocate(quad, loc, ngi, coords)
      quad%degree=cvfaces%shape%quadrature%degree
      quad%dim=dim

      allocate( dl(ngi, fdim, coords) )

      do i = 1, coords
        gi = 1
        do j = 1, faces
          ! work out the parent quadrature using a face shape function
          ! and the coordinates of the corners
          quad%l(gi:gi+cvfaces%shape%ngi-1,i) = &
                        matmul(cvfaces%corners(j,i,:), &
                                    cvfaces%shape%n(:,:))

          do k = 1, fdim
            ! at the same time may as well work out the transformation
            ! matrix between the parent and face coordinates
            dl(gi:gi+cvfaces%shape%ngi-1,k,i) = &
                        matmul(cvfaces%corners(j,i,:), &
                                    cvfaces%shape%dn(:,:,k))
          end do
          quad%weight(gi:gi+cvfaces%shape%ngi-1)= &
                        cvfaces%shape%quadrature%weight(:)

          gi = gi + cvfaces%shape%ngi
        end do
      end do

      select case(ltype)
      case(ELEMENT_CONTROLVOLUME_SURFACE)

        ! create an element based on the parent quadrature
        ! our final shape function will be almost identical but have a lower dimension of derivatives
        ! evaluated across the faces
        tempshape=make_element_shape(vertices=loc, dim=dim, &
                                      degree=parentshape%degree, quad=quad, &
                                      type=parentshape%type)

        ! start converting the lagrangian element into a control volume surface element
        ! another useful number
        nodes = tempshape%numbering%nodes
        ! Get the local numbering of our element
        ele_num=>find_element_numbering(loc, &
                  dim, parentshape%degree, type=parentshape%type)

        if (.not.associated(ele_num)) then
          if (present(stat)) then
              stat=1
              return
          else
              FLAbort('Element numbering unavailable')
          end if
        end if

        call allocate(element=shape, dim=dim, ndof=nodes, ngi=ngi, coords=coords, &
                      type=ELEMENT_CONTROLVOLUME_SURFACE)

        shape%numbering=>ele_num
        shape%cell=>parentshape%cell

        shape%quadrature=quad
        call incref(quad)

        shape%degree=parentshape%degree

        shape%n = tempshape%n

        ! construct the derivatives of the control volume surface shape function
        ! by performing a change of variables on the lagrangian derivatives
        do i = 1, nodes
          do j = 1, ngi
            ! here we throw away one of our co-ordinates if using simplex elements
            shape%dn(i,j,:) = matmul(dl(j,:,1:dim), tempshape%dn(i,j,:))
          end do
        end do

        call deallocate( tempshape )

      case(ELEMENT_CONTROLVOLUME_SURFACE_BODYDERIVATIVES)

        ! create a lagrangian element based on the parent quadrature
        shape=make_element_shape(vertices=loc, dim=dim, &
                                      degree=parentshape%degree, quad=quad, &
                                      type=parentshape%type)

      case default

        FLAbort ('Unsupported control volume element type')

      end select

      call deallocate( quad )
      deallocate( dl )

    end function make_cv_element_shape

    function make_cvbdy_element_shape(cvfaces, parentshape, type, stat) result (shape)

      type(element_type) :: shape

      type(cv_faces_type), intent(in), target :: cvfaces
      type(element_type), intent(in) :: parentshape
      integer, intent(in), optional :: type
      integer, intent(out), optional :: stat

      type(ele_numbering_type), pointer :: ele_num
      type(quadrature_type) :: quad
      type(element_type) :: tempshape

      integer :: i, j, k, gi
      integer :: ngi, coords, fdim, dim, loc, faces, nodes
      integer :: ltype

      real, dimension(:,:,:), allocatable :: dl

      if(present(stat)) stat = 0

      if(present(type)) then
        ltype = type
      else
        ltype = ELEMENT_CONTROLVOLUME_SURFACE
      end if

      ! some useful numbers
      ngi = cvfaces%shape%ngi*cvfaces%sfaces
                                  ! number of gauss points in parent element
      loc = cvfaces%svertices         ! vertices of parent element
      coords = cvfaces%scoords   ! number of canonical coordinates in parent element
      faces = cvfaces%sfaces     ! number of faces
      fdim = cvfaces%dim-1       ! dimension of faces
      dim = cvfaces%dim-1        ! dimension of parent element

      ! Create the quadrature for the parent element
      call allocate(quad, loc, ngi, coords)
      quad%degree=cvfaces%shape%quadrature%degree
      quad%dim=dim

      allocate( dl(ngi, fdim, coords) )

      do i = 1, coords
        gi = 1
        do j = 1, faces
          ! work out the parent quadrature using a face shape function
          ! and the coordinates of the corners
          quad%l(gi:gi+cvfaces%shape%ngi-1,i) = &
                        matmul(cvfaces%scorners(j,i,:), &
                                    cvfaces%shape%n(:,:))
          do k = 1, fdim
            ! at the same time may as well work out the transformation
            ! matrix between the parent and face coordinates
            dl(gi:gi+cvfaces%shape%ngi-1,k,i) = &
                        matmul(cvfaces%scorners(j,i,:), &
                                    cvfaces%shape%dn(:,:,k))
          end do
          quad%weight(gi:gi+cvfaces%shape%ngi-1)= &
                        cvfaces%shape%quadrature%weight(:)

          gi = gi + cvfaces%shape%ngi
        end do
      end do

      select case(ltype)
      case(ELEMENT_CONTROLVOLUME_SURFACE)

        ! create an element based on the parent quadrature
        ! our final shape function will be almost identical but have a lower dimension of derivatives
        ! evaluated across the faces
        tempshape=make_element_shape(vertices=loc, dim=dim, &
                                      degree=parentshape%degree, quad=quad, &
                                      type=parentshape%type)

        ! start converting the lagrangian element into a control volume surface element

        ! another useful number
        nodes = tempshape%numbering%nodes
        ! Get the local numbering of our element
        ele_num=>find_element_numbering(loc, &
                  dim, parentshape%degree, type=parentshape%type)

        if (.not.associated(ele_num)) then
          if (present(stat)) then
              stat=1
              return
          else
              FLAbort('Element numbering unavailable')
          end if
        end if

        call allocate(element=shape, dim=dim, ndof=nodes, ngi=ngi, coords=coords, &
                      type=ELEMENT_CONTROLVOLUMEBDY_SURFACE)

        shape%numbering=>ele_num
        shape%cell=>find_cell(dim, loc)

        shape%quadrature=quad
        call incref(quad)

        shape%degree=parentshape%degree

        shape%n = tempshape%n

        ! construct the derivatives of the control volume surface shape function
        ! by performing a change of variables on the lagrangian derivatives
        do i = 1, nodes
          do j = 1, ngi
            ! here we throw away one of our co-ordinates if using simplex elements
            shape%dn(i,j,:) = matmul(dl(j,:,1:dim), tempshape%dn(i,j,:))
          end do
        end do

        call deallocate( tempshape )

      case(ELEMENT_CONTROLVOLUME_SURFACE_BODYDERIVATIVES)

        ! create a lagrangian element based on the parent quadrature
        shape=make_element_shape(vertices=loc, dim=dim, &
                                      degree=parentshape%degree, quad=quad, &
                                      type=parentshape%type)

      case default

        FLAbort ('Unsupported control volume element type')

      end select

      call deallocate( quad )
      deallocate( dl )

    end function make_cvbdy_element_shape

end module cv_shape_functions
