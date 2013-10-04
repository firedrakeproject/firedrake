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
module cv_faces
  use FLDebug
  use shape_functions
  implicit none

  type cv_faces_type
    ! vertices = number of vertices, nodes = number of nodes, faces = number of faces
    ! degree = degree of polynomial, dim = dimensions of parent element
    integer :: vertices, nodes, faces, coords, degree, dim
    integer :: svertices, snodes, sfaces, scoords
    ! corners = volume coordinates of corners of faces
    ! faces x coords x face vertices
    real, dimension(:,:,:), pointer :: corners, scorners
    ! neiloc = relates faces to nodes and vice versa
    ! nodes x faces
    integer, dimension(:,:), pointer :: neiloc, sneiloc
    ! shape = shape function used in quadrature of faces
    ! 1 dimension lower than parent element
    type(element_type) :: shape
  end type cv_faces_type

  type corner_permutation_type
     integer, pointer, dimension(:,:) :: p
  end type corner_permutation_type

  type face_generator_type
     integer, dimension(:,:), pointer :: permutation
     real, dimension(:), pointer :: coords
     integer, dimension(2) :: nodes
  end type face_generator_type

  type face_corner_template
     !! A face is defined by a set of generators.
     type(face_generator_type), dimension(:), pointer :: generator
     integer :: dimension, vertices, ncorn, nodes
     integer :: degree, faces, coords
  end type face_corner_template

  integer, private, parameter :: CV_TET_MAX_DEGREE=2, CV_HEX_MAX_DEGREE=1, &
                        CV_TRI_MAX_DEGREE=2, CV_QUAD_MAX_DEGREE=1, CV_LINE_MAX_DEGREE=1

  type(corner_permutation_type), dimension(30), private, target, save :: cv_tet_face_permutations
  type(corner_permutation_type), dimension(12), private, target, save :: cv_tet_bdy_permutations
  type(corner_permutation_type), dimension(12), private, target, save :: cv_hex_face_permutations
  type(corner_permutation_type), dimension(4), private, target, save :: cv_hex_bdy_permutations
  type(corner_permutation_type), dimension(12), private, target, save :: cv_tri_face_permutations
  type(corner_permutation_type), dimension(4), private, target, save :: cv_quad_face_permutations
  type(corner_permutation_type), dimension(5), private, target, save :: cv_line_bdy_permutations
  type(corner_permutation_type), dimension(1), private, target, save :: cv_line_face_permutations
  type(corner_permutation_type), dimension(1), private, target, save :: cv_point_bdy_permutations

  type(face_corner_template), dimension(CV_TET_MAX_DEGREE), private, target, save :: cv_tet_face_temp
  type(face_corner_template), dimension(CV_TET_MAX_DEGREE), private, target, save :: cv_tet_bdy_temp
  type(face_corner_template), dimension(CV_HEX_MAX_DEGREE), private, target, save :: cv_hex_face_temp
  type(face_corner_template), dimension(CV_HEX_MAX_DEGREE), private, target, save :: cv_hex_bdy_temp
  type(face_corner_template), dimension(CV_TRI_MAX_DEGREE), private, target, save :: cv_tri_face_temp
  type(face_corner_template), dimension(CV_LINE_MAX_DEGREE), private, target, save :: cv_line_face_temp
  type(face_corner_template), dimension(CV_QUAD_MAX_DEGREE), private, target, save :: cv_quad_face_temp
  type(face_corner_template), dimension(max(CV_TRI_MAX_DEGREE,CV_QUAD_MAX_DEGREE)), private, target, save :: cv_line_bdy_temp
  type(face_corner_template), dimension(CV_LINE_MAX_DEGREE), private, target, save :: cv_point_bdy_temp

  logical, private, save :: initialised=.false.

  interface allocate
     module procedure allocate_cv_faces_type
  end interface

  interface deallocate
     module procedure deallocate_cv_faces_type
  end interface

  private
  public :: deallocate, find_cv_faces, cv_faces_type

contains

  function find_cv_faces(vertices, dimension, polydegree, quaddegree, quadngi) result (cvfaces)
    ! Return the element numbering type for an element in dimension
    ! dimensions with vertices vertices and degree polynomial bases.
    !
    ! If no suitable numbering is available, return a null pointer.
    type(face_corner_template), dimension(:), pointer :: cv_temp_list, cvbdy_temp_list
    type(face_corner_template), pointer :: cv_temp, cvbdy_temp
    type(cv_faces_type) :: cvfaces
    integer, intent(in) :: vertices, dimension, polydegree
    integer, intent(in), optional :: quaddegree, quadngi

    type(quadrature_type) :: face_quad

    if (.not.initialised) call locate_controlvolume_corners

    select case(dimension)
    case(1)
      select case (vertices)
      case (2)
          !Line segments
          if (polydegree>CV_LINE_MAX_DEGREE) then
            FLExit('Invalid control volume degree')
          else
            cv_temp_list=>cv_line_face_temp
            cvbdy_temp_list=> cv_point_bdy_temp
          end if
      case default

          FLExit('Invalid control volume type.')

      end select

    case(2)
      select case (vertices)
      case (3)
          !Triangles
          if (polydegree>CV_TRI_MAX_DEGREE) then
            FLExit('Invalid control volume degree.')
          else
            cv_temp_list=>cv_tri_face_temp
            cvbdy_temp_list=>cv_line_bdy_temp
          end if

      case (4)
          !Quads
          if (polydegree>CV_QUAD_MAX_DEGREE) then
            FLExit('Invalid control volume degree.')
          else
            cv_temp_list=>cv_quad_face_temp
            cvbdy_temp_list=>cv_line_bdy_temp
          end if

      case default

          FLExit('Invalid control volume type.')

      end select

    case(3)

      select case (vertices)
      case (4)
          !Tets
          if (polydegree>CV_TET_MAX_DEGREE) then
            FLExit('Invalid control volume degree.')
          else
            cv_temp_list=>cv_tet_face_temp
            cvbdy_temp_list=>cv_tet_bdy_temp
          end if

      case(8)
          !Hexes
          if (polydegree>CV_HEX_MAX_DEGREE) then
            FLExit('Invalid control volume degree.')
          else
            cv_temp_list=>cv_hex_face_temp
            cvbdy_temp_list=>cv_hex_bdy_temp
          end if

      case default

          FLExit('Invalid control volume type.')

      end select

    case default

      FLExit('Invalid control volume type.')

    end select

    cv_temp=>cv_temp_list(minloc(cv_temp_list%degree, dim=1,&
            mask=cv_temp_list%degree>=polydegree))
    cvbdy_temp=>cvbdy_temp_list(minloc(cvbdy_temp_list%degree, dim=1,&
            mask=cvbdy_temp_list%degree>=polydegree))

    ! Now we can start putting together the face info.
    call allocate(cvfaces, cv_temp%nodes, cvbdy_temp%nodes, cv_temp%faces, cvbdy_temp%faces, &
                   cv_temp%coords, cvbdy_temp%coords, &
                   cv_temp%ncorn)
    cvfaces%vertices=cv_temp%vertices
    cvfaces%nodes=cv_temp%nodes
    cvfaces%svertices=cvbdy_temp%vertices
    cvfaces%snodes=cvbdy_temp%nodes
    cvfaces%coords=cv_temp%coords
    cvfaces%scoords=cvbdy_temp%coords
    cvfaces%faces=cv_temp%faces
    cvfaces%sfaces=cvbdy_temp%faces
    cvfaces%dim=dimension
    cvfaces%degree=polydegree

    call expand_cv_faces_template(cvfaces, cv_temp, cvbdy_temp)

    if(present(quaddegree)) then
      face_quad=make_quadrature(vertices=size(cvfaces%corners, 3),dim=(cvfaces%dim-1), &
                                  degree=quaddegree)
    elseif(present(quadngi)) then
      face_quad=make_quadrature(vertices=size(cvfaces%corners, 3),dim=(cvfaces%dim-1), &
                                  ngi=quadngi)
    else
      ! code error
      FLAbort('Must specifiy either quaddegree or quadngi')
    end if

    cvfaces%shape=make_element_shape(vertices=size(cvfaces%corners, 3), dim=(cvfaces%dim-1), &
                                    degree=1, quad=face_quad, &
                                    type=ELEMENT_LAGRANGIAN)

    call deallocate(face_quad)

  end function find_cv_faces

  subroutine allocate_cv_faces_type(cvfaces, nodes, snodes, faces, sfaces, coords, scoords, &
                                    ncorn, stat)
    !!< Allocate memory for a quadrature type. Note that this is done
    !!< automatically in make_quadrature.
    type(cv_faces_type), intent(inout) :: cvfaces
    !! nodes is the number of nodes
    integer, intent(in) :: nodes, snodes, faces, sfaces, ncorn, coords, scoords
    !! Stat returns zero for successful completion and nonzero otherwise.
    integer, intent(out), optional :: stat

    integer :: lstat

    allocate(cvfaces%corners(faces,coords,ncorn), cvfaces%neiloc(nodes,faces), &
             cvfaces%scorners(sfaces,scoords,ncorn), cvfaces%sneiloc(snodes,sfaces), &
             stat=lstat)

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Error allocating cvfaces")
    end if

  end subroutine allocate_cv_faces_type

  subroutine deallocate_cv_faces_type(cvfaces,stat)
    !! The cvloc type to be deallocated.
    type(cv_faces_type), intent(inout) :: cvfaces
    !! Stat returns zero for successful completion and nonzero otherwise.
    integer, intent(out), optional :: stat

    integer :: lstat, tstat

    lstat=0
    tstat=0

    deallocate(cvfaces%corners, cvfaces%neiloc, &
               cvfaces%scorners, cvfaces%sneiloc, stat = tstat)
    lstat=max(tstat,lstat)
    call deallocate(cvfaces%shape, stat= tstat)
    lstat=max(tstat,lstat)

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Error deallocating cvfaces")
    end if

  end subroutine deallocate_cv_faces_type
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine locate_controlvolume_corners
    ! Fill the values in in element_numbering.

    ! make sure this is idempotent.
    if (initialised) return
    initialised=.true.

    call construct_cv_tet_face_permutations
    call construct_cv_tet_face_templates
    call construct_cv_tet_bdy_permutations
    call construct_cv_tet_bdy_templates

    call construct_cv_hex_face_permutations
    call construct_cv_hex_face_templates
    call construct_cv_hex_bdy_permutations
    call construct_cv_hex_bdy_templates

    call construct_cv_tri_face_permutations
    call construct_cv_tri_face_templates

    call construct_cv_quad_face_permutations
    call construct_cv_quad_face_templates

    call construct_cv_line_bdy_permutations
    call construct_cv_line_bdy_templates

    call construct_cv_line_face_permutations
    call construct_cv_line_face_templates
    call construct_cv_point_bdy_permutations
    call construct_cv_point_bdy_templates

  end subroutine locate_controlvolume_corners
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_tet_face_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(7) :: coords

    coords=0.0

    cv_tet_face_temp%dimension=3
    cv_tet_face_temp%vertices=4
    cv_tet_face_temp%ncorn=4
    cv_tet_face_temp%coords=4

    i=0

    !----------------------------------------------------------------------
    ! Linear tet
    i=i+1
    ! One generator per face.
    allocate(cv_tet_face_temp(i)%generator(6))

    cv_tet_face_temp(i)%faces=6
    cv_tet_face_temp(i)%degree=1
    cv_tet_face_temp(i)%nodes=4
    coords(1)=0.25
    coords(2)=0.333333333333333333333333333333333
    coords(3)=0.5
    cv_tet_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_tet_face_permutations(1), &
         nodes=(/1,2/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_tet_face_permutations(2), &
         nodes=(/2,3/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_tet_face_permutations(3), &
         nodes=(/1,3/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_tet_face_permutations(4), &
         nodes=(/1,4/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(5)=make_face_generator( &
         permutation=cv_tet_face_permutations(5), &
         nodes=(/2,4/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(6)=make_face_generator( &
         permutation=cv_tet_face_permutations(6), &
         nodes=(/3,4/), &
         coords=coords)

    !----------------------------------------------------------------------
    ! Quadratic tet
    i=i+1
    ! One generator per face.
    allocate(cv_tet_face_temp(i)%generator(24))

    cv_tet_face_temp(i)%faces=24
    cv_tet_face_temp(i)%degree=2
    cv_tet_face_temp(i)%nodes=10
    coords(1)=0.125
    coords(2)=0.166666666666666666666666666666666
    coords(3)=0.25
    coords(4)=0.333333333333333333333333333333333
    coords(5)=0.625
    coords(6)=0.666666666666666666666666666666666
    coords(7)=0.75
    cv_tet_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_tet_face_permutations(7), &
         nodes=(/1,2/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_tet_face_permutations(8), &
         nodes=(/1,4/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_tet_face_permutations(9), &
         nodes=(/1,7/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_tet_face_permutations(10), &
         nodes=(/2,3/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(5)=make_face_generator( &
         permutation=cv_tet_face_permutations(11), &
         nodes=(/3,5/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(6)=make_face_generator( &
         permutation=cv_tet_face_permutations(12), &
         nodes=(/3,8/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(7)=make_face_generator( &
         permutation=cv_tet_face_permutations(13), &
         nodes=(/4,6/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(8)=make_face_generator( &
         permutation=cv_tet_face_permutations(14), &
         nodes=(/5,6/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(9)=make_face_generator( &
         permutation=cv_tet_face_permutations(15), &
         nodes=(/6,9/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(10)=make_face_generator( &
         permutation=cv_tet_face_permutations(16), &
         nodes=(/7,10/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(11)=make_face_generator( &
         permutation=cv_tet_face_permutations(17), &
         nodes=(/8,10/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(12)=make_face_generator( &
         permutation=cv_tet_face_permutations(18), &
         nodes=(/9,10/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(13)=make_face_generator( &
         permutation=cv_tet_face_permutations(19), &
         nodes=(/2,7/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(14)=make_face_generator( &
         permutation=cv_tet_face_permutations(20), &
         nodes=(/2,8/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(15)=make_face_generator( &
         permutation=cv_tet_face_permutations(21), &
         nodes=(/7,8/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(16)=make_face_generator( &
         permutation=cv_tet_face_permutations(22), &
         nodes=(/5,8/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(17)=make_face_generator( &
         permutation=cv_tet_face_permutations(23), &
         nodes=(/5,9/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(18)=make_face_generator( &
         permutation=cv_tet_face_permutations(24), &
         nodes=(/8,9/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(19)=make_face_generator( &
         permutation=cv_tet_face_permutations(25), &
         nodes=(/4,7/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(20)=make_face_generator( &
         permutation=cv_tet_face_permutations(26), &
         nodes=(/4,9/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(21)=make_face_generator( &
         permutation=cv_tet_face_permutations(27), &
         nodes=(/7,9/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(22)=make_face_generator( &
         permutation=cv_tet_face_permutations(28), &
         nodes=(/2,4/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(23)=make_face_generator( &
         permutation=cv_tet_face_permutations(29), &
         nodes=(/2,5/), &
         coords=coords)
    cv_tet_face_temp(i)%generator(24)=make_face_generator( &
         permutation=cv_tet_face_permutations(30), &
         nodes=(/4,5/), &
         coords=coords)

  end subroutine construct_cv_tet_face_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_tri_face_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(5) :: coords

    coords=0.0

    cv_tri_face_temp%dimension=2
    cv_tri_face_temp%vertices=3
    cv_tri_face_temp%ncorn=2
    cv_tri_face_temp%coords=3

    i=0

    !----------------------------------------------------------------------
    ! Linear triangle
    i=i+1
    ! One generator per face.
    allocate(cv_tri_face_temp(i)%generator(3))

    cv_tri_face_temp(i)%faces=3
    cv_tri_face_temp(i)%degree=1
    cv_tri_face_temp(i)%nodes=3
    coords(1)=0.333333333333333333333333333333333
    coords(2)=0.5
    cv_tri_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_tri_face_permutations(1), &
         nodes=(/1,2/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_tri_face_permutations(2), &
         nodes=(/2,3/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_tri_face_permutations(3), &
         nodes=(/1,3/), &
         coords=coords)

    !----------------------------------------------------------------------
    ! Quadratic triangle
    i=i+1
    ! One generator per face.
    allocate(cv_tri_face_temp(i)%generator(9))

    cv_tri_face_temp(i)%faces=9
    cv_tri_face_temp(i)%degree=2
    cv_tri_face_temp(i)%nodes=6
    coords(1)=0.166666666666666666666666666666666
    coords(2)=0.25
    coords(3)=0.333333333333333333333333333333333
    coords(4)=0.666666666666666666666666666666666
    coords(5)=0.75
    cv_tri_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_tri_face_permutations(4), &
         nodes=(/1,2/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_tri_face_permutations(5), &
         nodes=(/1,4/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_tri_face_permutations(6), &
         nodes=(/2,3/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_tri_face_permutations(7), &
         nodes=(/3,5/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(5)=make_face_generator( &
         permutation=cv_tri_face_permutations(8), &
         nodes=(/5,6/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(6)=make_face_generator( &
         permutation=cv_tri_face_permutations(9), &
         nodes=(/4,6/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(7)=make_face_generator( &
         permutation=cv_tri_face_permutations(10), &
         nodes=(/2,4/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(8)=make_face_generator( &
         permutation=cv_tri_face_permutations(11), &
         nodes=(/2,5/), &
         coords=coords)
    cv_tri_face_temp(i)%generator(9)=make_face_generator( &
         permutation=cv_tri_face_permutations(12), &
         nodes=(/4,5/), &
         coords=coords)

  end subroutine construct_cv_tri_face_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_line_face_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(5) :: coords

    coords=0.0

    cv_line_face_temp%dimension=1
    cv_line_face_temp%vertices=2
    cv_line_face_temp%ncorn=1
    cv_line_face_temp%coords=2

    i=0

    !----------------------------------------------------------------------
    ! Linear line
    i=i+1
    ! One generator per face.
    allocate(cv_line_face_temp(i)%generator(1))

    cv_line_face_temp(i)%faces=1
    cv_line_face_temp(i)%degree=1
    cv_line_face_temp(i)%nodes=2
    coords(1)=0.5
    cv_line_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_line_face_permutations(1), &
         nodes=(/1,2/), &
         coords=coords)

  end subroutine construct_cv_line_face_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_hex_face_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(4) :: coords

    coords=0.0

    cv_hex_face_temp%dimension=3
    cv_hex_face_temp%vertices=8
    cv_hex_face_temp%ncorn=4
    cv_hex_face_temp%coords=3

    i=0

    !----------------------------------------------------------------------
    ! Tri-Linear hex
    i=i+1
    ! One generator per face.
    allocate(cv_hex_face_temp(i)%generator(12))

    cv_hex_face_temp(i)%faces=12
    cv_hex_face_temp(i)%degree=1
    cv_hex_face_temp(i)%nodes=8
    coords(1)=1.0
    cv_hex_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_hex_face_permutations(1), &
         nodes=(/1,3/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_hex_face_permutations(2), &
         nodes=(/2,4/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_hex_face_permutations(3), &
         nodes=(/1,2/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_hex_face_permutations(4), &
         nodes=(/3,4/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(5)=make_face_generator( &
         permutation=cv_hex_face_permutations(5), &
         nodes=(/1,5/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(6)=make_face_generator( &
         permutation=cv_hex_face_permutations(6), &
         nodes=(/2,6/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(7)=make_face_generator( &
         permutation=cv_hex_face_permutations(7), &
         nodes=(/3,7/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(8)=make_face_generator( &
         permutation=cv_hex_face_permutations(8), &
         nodes=(/4,8/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(9)=make_face_generator( &
         permutation=cv_hex_face_permutations(9), &
         nodes=(/5,7/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(10)=make_face_generator( &
         permutation=cv_hex_face_permutations(10), &
         nodes=(/6,8/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(11)=make_face_generator( &
         permutation=cv_hex_face_permutations(11), &
         nodes=(/5,6/), &
         coords=coords)
    cv_hex_face_temp(i)%generator(12)=make_face_generator( &
         permutation=cv_hex_face_permutations(12), &
         nodes=(/7,8/), &
         coords=coords)

  end subroutine construct_cv_hex_face_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_quad_face_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(4) :: coords

    coords=0.0

    cv_quad_face_temp%dimension=2
    cv_quad_face_temp%vertices=4
    cv_quad_face_temp%ncorn=2
    cv_quad_face_temp%coords=2

    i=0

    !----------------------------------------------------------------------
    ! Tri-Linear quad
    i=i+1
    ! One generator per face.
    allocate(cv_quad_face_temp(i)%generator(4))

    cv_quad_face_temp(i)%faces=4
    cv_quad_face_temp(i)%degree=1
    cv_quad_face_temp(i)%nodes=4
    coords(1)=1.0
    cv_quad_face_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_quad_face_permutations(1), &
         nodes=(/1,2/), &
         coords=coords)
    cv_quad_face_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_quad_face_permutations(2), &
         nodes=(/3,4/), &
         coords=coords)
    cv_quad_face_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_quad_face_permutations(3), &
         nodes=(/1,3/), &
         coords=coords)
    cv_quad_face_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_quad_face_permutations(4), &
         nodes=(/2,4/), &
         coords=coords)

  end subroutine construct_cv_quad_face_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_tet_bdy_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(7) :: coords

    coords=0.0

    cv_tet_bdy_temp%dimension=2
    cv_tet_bdy_temp%vertices=3
    cv_tet_bdy_temp%ncorn=4
    cv_tet_bdy_temp%coords=3

    i=0

    !----------------------------------------------------------------------
    ! Linear tet boundary
    i=i+1
    ! One generator per face.
    allocate(cv_tet_bdy_temp(i)%generator(3))

    cv_tet_bdy_temp(i)%faces=3
    cv_tet_bdy_temp(i)%degree=1
    cv_tet_bdy_temp(i)%nodes=3
    coords(1)=0.333333333333333333333333333333333
    coords(2)=0.5
    coords(3)=1.0
    cv_tet_bdy_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(1), &
         nodes=(/1,1/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(2), &
         nodes=(/2,2/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(3), &
         nodes=(/3,3/), &
         coords=coords)

    !----------------------------------------------------------------------
    ! Quadratic tet boundary
    i=i+1
    ! One generator per face.
    ! (Multiple faces per node to allow non-quadrilateral shapes)
    allocate(cv_tet_bdy_temp(i)%generator(9))

    cv_tet_bdy_temp(i)%faces=9 ! should only be 6 but 3 are fictitious to account for non-quadrilateral shapes
    cv_tet_bdy_temp(i)%degree=2
    cv_tet_bdy_temp(i)%nodes=6
    coords(1)=0.166666666666666666666666666666666
    coords(2)=0.25
    coords(3)=0.333333333333333333333333333333333
    coords(4)=0.5
    coords(5)=0.666666666666666666666666666666666
    coords(6)=0.75
    coords(7)=1.0
    cv_tet_bdy_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(4), &
         nodes=(/1,1/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(5), &
         nodes=(/2,2/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(6), &
         nodes=(/2,2/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(7), &
         nodes=(/3,3/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(5)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(8), &
         nodes=(/4,4/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(6)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(9), &
         nodes=(/4,4/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(7)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(10), &
         nodes=(/5,5/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(8)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(11), &
         nodes=(/5,5/), &
         coords=coords)
    cv_tet_bdy_temp(i)%generator(9)=make_face_generator( &
         permutation=cv_tet_bdy_permutations(12), &
         nodes=(/6,6/), &
         coords=coords)

  end subroutine construct_cv_tet_bdy_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_line_bdy_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(4) :: coords

    coords=0.0

    cv_line_bdy_temp%dimension=1
    cv_line_bdy_temp%vertices=2
    cv_line_bdy_temp%ncorn=2
    cv_line_bdy_temp%coords=2

    i=0

    !----------------------------------------------------------------------
    ! Linear line boundary
    i=i+1
    ! One generator per face.
    allocate(cv_line_bdy_temp(i)%generator(2))

    cv_line_bdy_temp(i)%faces=2
    cv_line_bdy_temp(i)%degree=1
    cv_line_bdy_temp(i)%nodes=2
    coords(1)=0.5
    coords(2)=1.0
    cv_line_bdy_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_line_bdy_permutations(1), &
         nodes=(/1,1/), &
         coords=coords)
    cv_line_bdy_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_line_bdy_permutations(2), &
         nodes=(/2,2/), &
         coords=coords)

    !----------------------------------------------------------------------
    ! Quadratic line boundary
    i=i+1
    ! One generator per face.
    allocate(cv_line_bdy_temp(i)%generator(3))

    cv_line_bdy_temp(i)%faces=3
    cv_line_bdy_temp(i)%degree=2
    cv_line_bdy_temp(i)%nodes=3
    coords(1)=0.25
    coords(2)=0.75
    coords(3)=1.0
    cv_line_bdy_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_line_bdy_permutations(3), &
         nodes=(/1,1/), &
         coords=coords)
    cv_line_bdy_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_line_bdy_permutations(4), &
         nodes=(/2,2/), &
         coords=coords)
    cv_line_bdy_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_line_bdy_permutations(5), &
         nodes=(/3,3/), &
         coords=coords)

  end subroutine construct_cv_line_bdy_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_point_bdy_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(4) :: coords

    coords=0.0

    cv_point_bdy_temp%dimension=0
    cv_point_bdy_temp%vertices=1
    cv_point_bdy_temp%ncorn=1
    cv_point_bdy_temp%coords=1

    i=0

    !----------------------------------------------------------------------
    ! Linear line boundary
    i=i+1
    ! One generator per face.
    allocate(cv_point_bdy_temp(i)%generator(1))

    cv_point_bdy_temp(i)%faces=1
    cv_point_bdy_temp(i)%degree=1
    cv_point_bdy_temp(i)%nodes=1
    coords(1)=1.0
    cv_point_bdy_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_point_bdy_permutations(1), &
         nodes=(/1,1/), &
         coords=coords)

  end subroutine construct_cv_point_bdy_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_hex_bdy_templates
    ! Construct list of available templates.
    integer :: i
    real, dimension(4) :: coords

    coords=0.0

    cv_hex_bdy_temp%dimension=2
    cv_hex_bdy_temp%vertices=4
    cv_hex_bdy_temp%ncorn=4
    cv_hex_bdy_temp%coords=2

    i=0

    !----------------------------------------------------------------------
    ! Tri-Linear hex boundary
    i=i+1
    ! One generator per face.
    allocate(cv_hex_bdy_temp(i)%generator(4))

    cv_hex_bdy_temp(i)%faces=4
    cv_hex_bdy_temp(i)%degree=1
    cv_hex_bdy_temp(i)%nodes=4
    coords(1)=1.0
    cv_hex_bdy_temp(i)%generator(1)=make_face_generator( &
         permutation=cv_hex_bdy_permutations(1), &
         nodes=(/1,1/), &
         coords=coords)
    cv_hex_bdy_temp(i)%generator(2)=make_face_generator( &
         permutation=cv_hex_bdy_permutations(2), &
         nodes=(/2,2/), &
         coords=coords)
    cv_hex_bdy_temp(i)%generator(3)=make_face_generator( &
         permutation=cv_hex_bdy_permutations(3), &
         nodes=(/3,3/), &
         coords=coords)
    cv_hex_bdy_temp(i)%generator(4)=make_face_generator( &
         permutation=cv_hex_bdy_permutations(4), &
         nodes=(/4,4/), &
         coords=coords)

  end subroutine construct_cv_hex_bdy_templates
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_tet_face_permutations

    ! linear faces

    allocate(cv_tet_face_permutations(1)%p(4,4))

    cv_tet_face_permutations(1)%p=reshape((/&
         3, 3, 0, 0, &
         2, 2, 2, 0, &
         2, 2, 0, 2, &
         1, 1, 1, 1/),(/4,4/))

    allocate(cv_tet_face_permutations(2)%p(4,4))

    cv_tet_face_permutations(2)%p=reshape((/&
         2, 2, 2, 0, &
         0, 3, 3, 0, &
         1, 1, 1, 1, &
         0, 2, 2, 2/),(/4,4/))

    allocate(cv_tet_face_permutations(3)%p(4,4))

    cv_tet_face_permutations(3)%p=reshape((/&
         2, 2, 2, 0, &
         3, 0, 3, 0, &
         1, 1, 1, 1, &
         2, 0, 2, 2/),(/4,4/))

    allocate(cv_tet_face_permutations(4)%p(4,4))

    cv_tet_face_permutations(4)%p=reshape((/&
         3, 0, 0, 3, &
         2, 2, 0, 2, &
         2, 0, 2, 2, &
         1, 1, 1, 1/),(/4,4/))

    allocate(cv_tet_face_permutations(5)%p(4,4))

    cv_tet_face_permutations(5)%p=reshape((/&
         2, 2, 0, 2, &
         0, 3, 0, 3, &
         1, 1, 1, 1, &
         0, 2, 2, 2/),(/4,4/))

    allocate(cv_tet_face_permutations(6)%p(4,4))

    cv_tet_face_permutations(6)%p=reshape((/&
         1, 1, 1, 1, &
         0, 2, 2, 2, &
         2, 0, 2, 2, &
         0, 0, 3, 3/),(/4,4/))

    ! end of linear faces

    ! quadratic faces

    allocate(cv_tet_face_permutations(7)%p(4,4))

    cv_tet_face_permutations(7)%p=reshape((/&
         7, 3, 0, 0, &
         6, 2, 2, 0, &
         6, 2, 0, 2, &
         5, 1, 1, 1/),(/4,4/))  ! 1,2

    allocate(cv_tet_face_permutations(8)%p(4,4))

    cv_tet_face_permutations(8)%p=reshape((/&
         6, 2, 2, 0, &
         7, 0, 3, 0, &
         5, 1, 1, 1, &
         6, 0, 2, 2/),(/4,4/))  ! 1,4

    allocate(cv_tet_face_permutations(9)%p(4,4))

    cv_tet_face_permutations(9)%p=reshape((/&
         7, 0, 0, 3, &
         6, 2, 0, 2, &
         6, 0, 2, 2, &
         5, 1, 1, 1/),(/4,4/))  ! 1,7

    allocate(cv_tet_face_permutations(10)%p(4,4))

    cv_tet_face_permutations(10)%p=reshape((/&
         3, 7, 0, 0, &
         2, 6, 2, 0, &
         2, 6, 0, 2, &
         1, 5, 1, 1/),(/4,4/))  ! 2,3

    allocate(cv_tet_face_permutations(11)%p(4,4))

    cv_tet_face_permutations(11)%p=reshape((/&
         2, 6, 2, 0, &
         0, 7, 3, 0, &
         1, 5, 1, 1, &
         0, 6, 2, 2/),(/4,4/))  ! 3,5

    allocate(cv_tet_face_permutations(12)%p(4,4))

    cv_tet_face_permutations(12)%p=reshape((/&
         0, 7, 0, 3, &
         2, 6, 0, 2, &
         0, 6, 2, 2, &
         1, 5, 1, 1/),(/4,4/))  ! 3,8

    allocate(cv_tet_face_permutations(13)%p(4,4))

    cv_tet_face_permutations(13)%p=reshape((/&
         3, 0, 7, 0, &
         2, 2, 6, 0, &
         2, 0, 6, 2, &
         1, 1, 5, 1/),(/4,4/))  ! 4,6

    allocate(cv_tet_face_permutations(14)%p(4,4))

    cv_tet_face_permutations(14)%p=reshape((/&
         2, 2, 6, 0, &
         0, 3, 7, 0, &
         1, 1, 5, 1, &
         0, 2, 6, 2/),(/4,4/))  ! 5,6

    allocate(cv_tet_face_permutations(15)%p(4,4))

    cv_tet_face_permutations(15)%p=reshape((/&
         0, 0, 7, 3, &
         2, 0, 6, 2, &
         0, 2, 6, 2, &
         1, 1, 5, 1/),(/4,4/))  ! 6,9

    allocate(cv_tet_face_permutations(16)%p(4,4))

    cv_tet_face_permutations(16)%p=reshape((/&
         3, 0, 0, 7, &
         2, 2, 0, 6, &
         2, 0, 2, 6, &
         1, 1, 1, 5/),(/4,4/))  ! 7,10

    allocate(cv_tet_face_permutations(17)%p(4,4))

    cv_tet_face_permutations(17)%p=reshape((/&
         2, 2, 0, 6, &
         0, 3, 0, 7, &
         1, 1, 1, 5, &
         0, 2, 2, 6/),(/4,4/))  ! 8,10

    allocate(cv_tet_face_permutations(18)%p(4,4))

    cv_tet_face_permutations(18)%p=reshape((/&
         0, 0, 3, 7, &
         2, 0, 2, 6, &
         0, 2, 2, 6, &
         1, 1, 1, 5/),(/4,4/))  ! 9,10

    allocate(cv_tet_face_permutations(19)%p(4,4))

    cv_tet_face_permutations(19)%p=reshape((/&
         6, 2, 0, 2, &
         4, 4, 0, 4, &
         5, 1, 1, 1, &
         3, 3, 3, 3/),(/4,4/))  ! 2,7

    allocate(cv_tet_face_permutations(20)%p(4,4))

    cv_tet_face_permutations(20)%p=reshape((/&
         4, 4, 0, 4, &
         2, 6, 0, 2, &
         3, 3, 3, 3, &
         1, 5, 1, 1/),(/4,4/))  ! 2,8

    allocate(cv_tet_face_permutations(21)%p(4,4))

    cv_tet_face_permutations(21)%p=reshape((/&
         2, 2, 0, 6, &
         4, 4, 0, 4, &
         1, 1, 1, 5, &
         3, 3, 3, 3/),(/4,4/))  ! 7,8

    allocate(cv_tet_face_permutations(22)%p(4,4))

    cv_tet_face_permutations(22)%p=reshape((/&
         0, 6, 2, 2, &
         0, 4, 4, 4, &
         1, 5, 1, 1, &
         3, 3, 3, 3/),(/4,4/))  ! 5,8

    allocate(cv_tet_face_permutations(23)%p(4,4))

    cv_tet_face_permutations(23)%p=reshape((/&
         0, 4, 4, 4, &
         0, 2, 6, 2, &
         3, 3, 3, 3, &
         1, 1, 5, 1/),(/4,4/))  ! 5,9

    allocate(cv_tet_face_permutations(24)%p(4,4))

    cv_tet_face_permutations(24)%p=reshape((/&
         0, 2, 2, 6, &
         0, 4, 4, 4, &
         1, 1, 1, 5, &
         3, 3, 3, 3/),(/4,4/))  ! 8,9

    allocate(cv_tet_face_permutations(25)%p(4,4))

    cv_tet_face_permutations(25)%p=reshape((/&
         6, 0, 2, 2, &
         4, 0, 4, 4, &
         5, 1, 1, 1, &
         3, 3, 3, 3/),(/4,4/))  ! 4,7

    allocate(cv_tet_face_permutations(26)%p(4,4))

    cv_tet_face_permutations(26)%p=reshape((/&
         4, 0, 4, 4, &
         2, 0, 6, 2, &
         3, 3, 3, 3, &
         1, 1, 5, 1/),(/4,4/))  ! 4,9

    allocate(cv_tet_face_permutations(27)%p(4,4))

    cv_tet_face_permutations(27)%p=reshape((/&
         2, 0, 2, 6, &
         4, 0, 4, 4, &
         1, 1, 1, 5, &
         3, 3, 3, 3/),(/4,4/))  ! 7,9

    allocate(cv_tet_face_permutations(28)%p(4,4))

    cv_tet_face_permutations(28)%p=reshape((/&
         6, 2, 2, 0, &
         4, 4, 4, 0, &
         5, 1, 1, 1, &
         3, 3, 3, 3/),(/4,4/))  ! 2,4

    allocate(cv_tet_face_permutations(29)%p(4,4))

    cv_tet_face_permutations(29)%p=reshape((/&
         4, 4, 4, 0, &
         2, 6, 2, 0, &
         3, 3, 3, 3, &
         1, 5, 1, 1/),(/4,4/))  ! 2,5

    allocate(cv_tet_face_permutations(30)%p(4,4))

    cv_tet_face_permutations(30)%p=reshape((/&
         2, 2, 6, 0, &
         4, 4, 4, 0, &
         1, 1, 5, 1, &
         3, 3, 3, 3/),(/4,4/))  ! 4,5

    ! end of quadratic faces

  end subroutine construct_cv_tet_face_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_tri_face_permutations

    ! linear faces

    allocate(cv_tri_face_permutations(1)%p(3,2))

    cv_tri_face_permutations(1)%p=reshape((/&
         1, 1, 1, &
         2, 2, 0/),(/3,2/))

    allocate(cv_tri_face_permutations(2)%p(3,2))

    cv_tri_face_permutations(2)%p=reshape((/&
         1, 1, 1, &
         0, 2, 2/),(/3,2/))

    allocate(cv_tri_face_permutations(3)%p(3,2))

    cv_tri_face_permutations(3)%p=reshape((/&
         1, 1, 1, &
         2, 0, 2/),(/3,2/))

    ! end of linear faces

    ! quadratic faces

    allocate(cv_tri_face_permutations(4)%p(3,2))

    cv_tri_face_permutations(4)%p=reshape((/&
         5, 2, 0, &
         4, 1, 1/),(/3,2/))

    allocate(cv_tri_face_permutations(5)%p(3,2))

    cv_tri_face_permutations(5)%p=reshape((/&
         4, 1, 1, &
         5, 0, 2/),(/3,2/))

    allocate(cv_tri_face_permutations(6)%p(3,2))

    cv_tri_face_permutations(6)%p=reshape((/&
         2, 5, 0, &
         1, 4, 1/),(/3,2/))

    allocate(cv_tri_face_permutations(7)%p(3,2))

    cv_tri_face_permutations(7)%p=reshape((/&
         1, 4, 1, &
         0, 5, 2/),(/3,2/))

    allocate(cv_tri_face_permutations(8)%p(3,2))

    cv_tri_face_permutations(8)%p=reshape((/&
         0, 2, 5, &
         1, 1, 4/),(/3,2/))

    allocate(cv_tri_face_permutations(9)%p(3,2))

    cv_tri_face_permutations(9)%p=reshape((/&
         2, 0, 5, &
         1, 1, 4/),(/3,2/))

    allocate(cv_tri_face_permutations(10)%p(3,2))

    cv_tri_face_permutations(10)%p=reshape((/&
         4, 1, 1, &
         3, 3, 3/),(/3,2/))

    allocate(cv_tri_face_permutations(11)%p(3,2))

    cv_tri_face_permutations(11)%p=reshape((/&
         3, 3, 3, &
         1, 4, 1/),(/3,2/))

    allocate(cv_tri_face_permutations(12)%p(3,2))

    cv_tri_face_permutations(12)%p=reshape((/&
         3, 3, 3, &
         1, 1, 4/),(/3,2/))

    ! end of quadratic faces

  end subroutine construct_cv_tri_face_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_line_face_permutations

    allocate(cv_line_face_permutations(1)%p(2,1))

    cv_line_face_permutations(1)%p=reshape((/&
         1, 1/),(/2,1/))

  end subroutine construct_cv_line_face_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_hex_face_permutations

    allocate(cv_hex_face_permutations(1)%p(3,4))

    cv_hex_face_permutations(1)%p=reshape((/&
         -1, 0, -1, &
         0, 0, -1, &
         -1, 0, 0, &
         0, 0, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(2)%p(3,4))

    cv_hex_face_permutations(2)%p=reshape((/&
         0, 0, -1, &
         1, 0, -1, &
         0, 0, 0, &
         1, 0, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(3)%p(3,4))

    cv_hex_face_permutations(3)%p=reshape((/&
         0, -1, -1, &
         0, 0, -1, &
         0, -1, 0, &
         0, 0, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(4)%p(3,4))

    cv_hex_face_permutations(4)%p=reshape((/&
         0, 0, -1, &
         0, 1, -1, &
         0, 0, 0, &
         0, 1, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(5)%p(3,4))

    cv_hex_face_permutations(5)%p=reshape((/&
         -1, -1, 0, &
         0, -1, 0, &
         -1, 0, 0, &
         0, 0, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(6)%p(3,4))

    cv_hex_face_permutations(6)%p=reshape((/&
         0, -1, 0, &
         1, -1, 0, &
         0, 0, 0, &
         1, 0, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(7)%p(3,4))

    cv_hex_face_permutations(7)%p=reshape((/&
         -1, 0, 0, &
         0, 0, 0, &
         -1, 1, 0, &
         0, 1, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(8)%p(3,4))

    cv_hex_face_permutations(8)%p=reshape((/&
         0, 0, 0, &
         1, 0, 0, &
         0, 1, 0, &
         1, 1, 0/),(/3,4/))

    allocate(cv_hex_face_permutations(9)%p(3,4))

    cv_hex_face_permutations(9)%p=reshape((/&
         -1, 0, 0, &
         0, 0, 0, &
         -1, 0, 1, &
         0, 0, 1/),(/3,4/))

    allocate(cv_hex_face_permutations(10)%p(3,4))

    cv_hex_face_permutations(10)%p=reshape((/&
         0, 0, 0, &
         1, 0, 0, &
         0, 0, 1, &
         1, 0, 1/),(/3,4/))

    allocate(cv_hex_face_permutations(11)%p(3,4))

    cv_hex_face_permutations(11)%p=reshape((/&
         0, -1, 0, &
         0, 0, 0, &
         0, -1, 1, &
         0, 0, 1/),(/3,4/))

    allocate(cv_hex_face_permutations(12)%p(3,4))

    cv_hex_face_permutations(12)%p=reshape((/&
         0, 0, 0, &
         0, 1, 0, &
         0, 0, 1, &
         0, 1, 1/),(/3,4/))

  end subroutine construct_cv_hex_face_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_quad_face_permutations

    allocate(cv_quad_face_permutations(1)%p(2,2))

    cv_quad_face_permutations(1)%p=reshape((/&
         -1, 0, &
         0, 0/),(/2,2/))

    allocate(cv_quad_face_permutations(2)%p(2,2))

    cv_quad_face_permutations(2)%p=reshape((/&
         0, 0, &
         1, 0/),(/2,2/))

    allocate(cv_quad_face_permutations(3)%p(2,2))

    cv_quad_face_permutations(3)%p=reshape((/&
         0, -1, &
         0, 0/),(/2,2/))

    allocate(cv_quad_face_permutations(4)%p(2,2))

    cv_quad_face_permutations(4)%p=reshape((/&
         0, 0, &
         0, 1/),(/2,2/))

  end subroutine construct_cv_quad_face_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_tet_bdy_permutations

    ! linear

    allocate(cv_tet_bdy_permutations(1)%p(3,4))

    cv_tet_bdy_permutations(1)%p=reshape((/&
         3, 0, 0, &
         2, 2, 0, &
         2, 0, 2, &
         1, 1, 1/),(/3,4/))

    allocate(cv_tet_bdy_permutations(2)%p(3,4))

    cv_tet_bdy_permutations(2)%p=reshape((/&
         2, 2, 0, &
         0, 3, 0, &
         1, 1, 1, &
         0, 2, 2/),(/3,4/))

    allocate(cv_tet_bdy_permutations(3)%p(3,4))

    cv_tet_bdy_permutations(3)%p=reshape((/&
         2, 0, 2, &
         1, 1, 1, &
         0, 0, 3, &
         0, 2, 2/),(/3,4/))

    ! end of linear

    ! quadratic

    allocate(cv_tet_bdy_permutations(4)%p(3,4))

    cv_tet_bdy_permutations(4)%p=reshape((/&
         7, 0, 0, &
         6, 2, 0, &
         6, 0, 2, &
         5, 1, 1/),(/3,4/))  !1

    allocate(cv_tet_bdy_permutations(5)%p(3,4))

    cv_tet_bdy_permutations(5)%p=reshape((/&
         5, 1, 1, &
         6, 2, 0, &
         3, 3, 3, &
         4, 4, 0/),(/3,4/))  !2

    allocate(cv_tet_bdy_permutations(6)%p(3,4))

    cv_tet_bdy_permutations(6)%p=reshape((/&
         3, 3, 3, &
         4, 4, 0, &
         1, 5, 1, &
         2, 6, 0/),(/3,4/))  !2

    allocate(cv_tet_bdy_permutations(7)%p(3,4))

    cv_tet_bdy_permutations(7)%p=reshape((/&
         0, 7, 0, &
         2, 6, 0, &
         0, 6, 2, &
         1, 5, 1/),(/3,4/))  !3

    allocate(cv_tet_bdy_permutations(8)%p(3,4))

    cv_tet_bdy_permutations(8)%p=reshape((/&
         5, 1, 1, &
         6, 0, 2, &
         3, 3, 3, &
         4, 0, 4/),(/3,4/))  !4

    allocate(cv_tet_bdy_permutations(9)%p(3,4))

    cv_tet_bdy_permutations(9)%p=reshape((/&
         3, 3, 3, &
         4, 0, 4, &
         1, 1, 5, &
         2, 0, 6/),(/3,4/))  !4

    allocate(cv_tet_bdy_permutations(10)%p(3,4))

    cv_tet_bdy_permutations(10)%p=reshape((/&
         1, 5, 1, &
         0, 6, 2, &
         3, 3, 3, &
         0, 4, 4/),(/3,4/))  !5

    allocate(cv_tet_bdy_permutations(11)%p(3,4))

    cv_tet_bdy_permutations(11)%p=reshape((/&
         3, 3, 3, &
         0, 4, 4, &
         1, 1, 5, &
         0, 2, 6/),(/3,4/))  !5

    allocate(cv_tet_bdy_permutations(12)%p(3,4))

    cv_tet_bdy_permutations(12)%p=reshape((/&
         0, 0, 7, &
         2, 0, 6, &
         0, 2, 6, &
         1, 1, 5/),(/3,4/))  !6

    ! end of quadratic

  end subroutine construct_cv_tet_bdy_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_line_bdy_permutations

    ! linear
    allocate(cv_line_bdy_permutations(1)%p(2,2))

    cv_line_bdy_permutations(1)%p=reshape((/&
         2, 0, &
         1, 1/),(/2,2/))

    allocate(cv_line_bdy_permutations(2)%p(2,2))

    cv_line_bdy_permutations(2)%p=reshape((/&
         1, 1, &
         0, 2/),(/2,2/))
    ! end of linear

    ! quadratic
    allocate(cv_line_bdy_permutations(3)%p(2,2))

    cv_line_bdy_permutations(3)%p=reshape((/&
         3, 0, &
         2, 1/),(/2,2/))

    allocate(cv_line_bdy_permutations(4)%p(2,2))

    cv_line_bdy_permutations(4)%p=reshape((/&
         2, 1, &
         1, 2/),(/2,2/))

    allocate(cv_line_bdy_permutations(5)%p(2,2))

    cv_line_bdy_permutations(5)%p=reshape((/&
         1, 2, &
         0, 3/),(/2,2/))
    ! end of quadratic

  end subroutine construct_cv_line_bdy_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_point_bdy_permutations

    allocate(cv_point_bdy_permutations(1)%p(1,1))

    cv_point_bdy_permutations(1)%p=reshape((/&
         1/),(/1,1/))

  end subroutine construct_cv_point_bdy_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  subroutine construct_cv_hex_bdy_permutations

    allocate(cv_hex_bdy_permutations(1)%p(2,4))

    cv_hex_bdy_permutations(1)%p=reshape((/&
         -1, -1, &
         0, -1, &
         -1, 0, &
         0, 0/),(/2,4/))

    allocate(cv_hex_bdy_permutations(2)%p(2,4))

    cv_hex_bdy_permutations(2)%p=reshape((/&
         0, -1, &
         1, -1, &
         0, 0, &
         1, 0/),(/2,4/))

    allocate(cv_hex_bdy_permutations(3)%p(2,4))

    cv_hex_bdy_permutations(3)%p=reshape((/&
         -1, 0, &
         0, 0, &
         -1, 1, &
         0, 1/),(/2,4/))

    allocate(cv_hex_bdy_permutations(4)%p(2,4))

    cv_hex_bdy_permutations(4)%p=reshape((/&
         0, 0, &
         1, 0, &
         0, 1, &
         1, 1/),(/2,4/))

  end subroutine construct_cv_hex_bdy_permutations
!< ------------------------------------------------- >!

!< ------------------------------------------------- >!
  function make_face_generator(permutation, nodes, coords) result (generator)
    !!< Function hiding the fact that generators are dynamically sized.
    type(face_generator_type) :: generator
    type(corner_permutation_type), intent(in) :: permutation
    integer, dimension(:), intent(in) :: nodes
    real, dimension(:), intent(in) :: coords

    generator%permutation=>permutation%p
    allocate(generator%coords(size(coords)))
    generator%nodes=nodes
    generator%coords=coords

  end function make_face_generator

  subroutine expand_cv_faces_template(cvfaces, cv_temp, cvbdy_temp)
    ! Expand the given template into the cvfaces provided.
    type(cv_faces_type), intent(inout) :: cvfaces
    type(face_corner_template), intent(in) :: cv_temp, cvbdy_temp

    integer :: i, j, k
    type(face_generator_type), pointer :: lgen

    cvfaces%corners=0.0
    cvfaces%neiloc=0

    do i=1,size(cv_temp%generator)
       lgen=>cv_temp%generator(i)

       ! Permute coordinates and insert into cvloc%corners
       forall(j=1:size(lgen%permutation,1), &
            k=1:size(lgen%permutation,2), &
            lgen%permutation(j,k)/=0)
          ! The permutation stores both the permutation order and (for
          ! quads and hexs) the sign of the coordinate.
            cvfaces%corners(i,j,k)=sign(lgen%coords(abs(lgen%permutation(j,k))),&
                  &                            real(lgen%permutation(j,k)))
       end forall

       cvfaces%neiloc(lgen%nodes(1),i)=lgen%nodes(2)
       cvfaces%neiloc(lgen%nodes(2),i)=lgen%nodes(1)

    end do

    cvfaces%scorners=0.0
    cvfaces%sneiloc=0

    do i=1,size(cvbdy_temp%generator)
       lgen=>cvbdy_temp%generator(i)

       ! Permute coordinates and insert into cvloc%corners
       forall(j=1:size(lgen%permutation,1), &
            k=1:size(lgen%permutation,2), &
            lgen%permutation(j,k)/=0)
          ! The permutation stores both the permutation order and (for
          ! quads and hexs) the sign of the coordinate.
          cvfaces%scorners(i,j,k)=sign(lgen%coords(abs(lgen%permutation(j,k))),&
               &                            real(lgen%permutation(j,k)))
       end forall

       cvfaces%sneiloc(lgen%nodes(1),i)=lgen%nodes(2)

    end do

    if ((cvfaces%vertices==4.and.cvfaces%dim==2)&
         .or.(cvfaces%vertices==8.and.cvfaces%dim==3)) then
       ! The rules for cvfaces and hex elements in the templates
       ! are written for local coordinates in the interval [-1,1], however
       ! we wish to use local coordinates in the interval [0,1]. This
       ! requires us to change coordinates.
       cvfaces%corners=0.5*(cvfaces%corners+1)
       cvfaces%scorners=0.5*(cvfaces%scorners+1)
    end if

  end subroutine expand_cv_faces_template

end module cv_faces
