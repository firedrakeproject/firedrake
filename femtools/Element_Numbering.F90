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
module element_numbering
  !!< Module containing local element numbering for the One True Numbering
  !!< Order.
  !!<
  !!< This is currently very tet-centric but is written to be
  !!< generalised.
  !!<
  !!< Conventions:
  !!<
  !!< The One True Numbering Order is the recursive Pascal's triangle order
  !!< as documented in the wiki.
  !!<
  !!< The bdys of a Tet have the same indices as the opposite vertex.
  !!<
  !!< This module currently implements tets of polynomial order 0 to 5.
  !!<
  !!< Nodes are subdivided into four disjoint sets: vertex nodes, those lying
  !!< on edges, those lying on faces and those interior to elements.
  use futils
  use FLDebug
  implicit none

  integer, parameter :: ELEMENT_LAGRANGIAN=1, ELEMENT_NONCONFORMING=2, ELEMENT_BUBBLE=3, &
                        ELEMENT_CONTROLVOLUMEBDY_SURFACE=4, ELEMENT_CONTROLVOLUME_SURFACE=5, &
                        ELEMENT_CONTROLVOLUME_SURFACE_BODYDERIVATIVES=6, &
                        ELEMENT_DISCONTINUOUS_LAGRANGIAN=7, ELEMENT_TRACE=8,&
                        ELEMENT_OTHER=9

  integer, parameter :: FAMILY_SIMPLEX=1, FAMILY_CUBE=2

  type ele_numbering_type
     ! Type to record element numbering details.
     ! Number of each sort of topological entity.
     integer :: faces, vertices, edges, facets
     ! How many nodes are there on each topological entity. The index in this
     ! array is the dimension of the object.
     integer, dimension(0:3) :: nodes_per
     integer :: degree ! Degree of polynomials.
     integer :: dimension ! 2D or 3D
     integer :: nodes
     integer :: type=ELEMENT_LAGRANGIAN
     integer :: family
     ! Map local count coordinates to local number.
     integer, dimension(:,:,:), pointer :: count2number
     ! Map local number to local count coordinates.
     integer, dimension(:,:), pointer :: number2count
     ! Count coordinate which is held constant for each element facet.
     integer, dimension(:), pointer :: facet_coord
     ! Value of that count coordinate on the element facet.
     integer, dimension(:), pointer :: facet_val
  end type ele_numbering_type

  integer, parameter :: TET_MAX_DEGREE=9, TET_BUBBLE_MAX_DEGREE=3
  integer, parameter :: TRI_MAX_DEGREE=32
  integer, parameter :: TRI_BUBBLE_MAX_DEGREE=2
  integer, parameter :: INTERVAL_MAX_DEGREE=32
  integer, parameter :: HEX_MAX_DEGREE=9, QUAD_MAX_DEGREE=9
  ! bubbles are restricted to prevent co-located nodes

  type(ele_numbering_type), dimension(0:TET_MAX_DEGREE), target, save ::&
       & tet_numbering
  type(ele_numbering_type), dimension(1:TET_BUBBLE_MAX_DEGREE), target, save ::&
       & tet_numbering_bubble
  type(ele_numbering_type), dimension(0:TRI_MAX_DEGREE), target, save ::&
       & tri_numbering
  type(ele_numbering_type), dimension(0:TRI_MAX_DEGREE), target, save ::&
       & tri_numbering_trace
  type(ele_numbering_type), dimension(0:TRI_MAX_DEGREE), target, save ::&
       & quad_numbering_trace
  type(ele_numbering_type), dimension(1:TRI_BUBBLE_MAX_DEGREE), target, save ::&
       & tri_numbering_bubble
  type(ele_numbering_type), target, save :: tri_numbering_nc
  type(ele_numbering_type), dimension(0:INTERVAL_MAX_DEGREE), target, &
       save :: interval_numbering
  type(ele_numbering_type), target, save :: interval_numbering_bubble
  type(ele_numbering_type), dimension(0:HEX_MAX_DEGREE), target, save ::&
       & hex_numbering
  type(ele_numbering_type), dimension(0:QUAD_MAX_DEGREE), target, save ::&
       & quad_numbering
  type(ele_numbering_type), dimension(0:0), target, save :: point_numbering

  ! Map from local node numbers to local edge numbers
  integer, dimension(4,4) :: LOCAL_EDGE_NUM=reshape(&
       (/0,1,2,4,&
       1,0,3,5,&
       2,3,0,6,&
       4,5,6,0/), (/4,4/))

  ! map from pair of local linear tet node numbers to local number of
  ! inbetween quadratic node
  integer, parameter, dimension(4,4) :: ilink2=reshape(&
      (/ 1, 2, 4, 7, &
         2, 3, 5, 8, &
         4, 5, 6, 9, &
         7, 8, 9, 10 /), (/ 4, 4 /) )

  ! map from pair of local linear tet node numbers to local number of
  ! for surface triangle
  ! ! THIS IS NOT IN THE ONE TRUE ORDERING - THIS SHOULD CHANGE ! !
  integer, parameter, dimension(3,3) :: silink2=reshape(&
      (/ 1, 4, 6, &
         4, 2, 5, &
         6, 5, 3 /), (/ 3, 3 /) )

  logical, private, save :: initialised=.false.

  interface local_coords
     module procedure ele_num_local_coords
  end interface

  interface local_vertices
     module procedure ele_num_local_vertices
  end interface

  interface vertex_num
     module procedure svertex_num, vvertex_num
  end interface

  interface facet_numbering
     module procedure numbering_facet_numbering
  end interface

!!$  interface edge_num
!!$     module procedure edge_num_int, edge_num_no_int
!!$  end interface

!!$  interface face_num
!!$     module procedure face_num_int, face_num_no_int
!!$  end interface

!!$  interface facet_local_num
!!$     module procedure facet_local_num_int, facet_local_num_no_int
!!$  end interface

  interface face_local_num
     module procedure face_local_num_int, face_local_num_no_int
  end interface

  interface operator(==)
     module procedure element_num_equal
  end interface

contains

  function find_element_numbering(vertices, dimension, degree, type) result (ele_num)
    ! Return the element numbering type for an element in dimension
    ! dimensions with vertices vertices and degree polynomial bases.
    !
    ! If no suitable numbering is available, return a null pointer.
    type(ele_numbering_type), pointer :: ele_num
    integer, intent(in) :: vertices, dimension, degree
    integer, intent(in), optional :: type

    integer :: ltype

    if (.not.initialised) call number_elements

    if (present(type)) then
       ltype=type
    else
       ltype=ELEMENT_LAGRANGIAN
    end if

    ! For element numbering, DG and CG Lagrangian are the same. The
    !  difference occurs in the element_type.
    if (ltype==ELEMENT_DISCONTINUOUS_LAGRANGIAN) then
       ltype=ELEMENT_LAGRANGIAN
    end if

    select case(ltype)
    case (ELEMENT_LAGRANGIAN)
       select case(dimension)
       case (0)
          select case(vertices)
          case(1)
             ! The point element always has degree 0
             ele_num=>point_numbering(0)
             return
          case default
             ele_num=>null()
             return
          end select

       case (1)
          select case(vertices)
          case(2)
             ! Intervals - the only possibility.
             if (degree>INTERVAL_MAX_DEGREE) then
                ele_num=>null()
                return
             else
                ele_num=>interval_numbering(degree)
                return
             end if

          case default
             ele_num=>null()
             return
          end select

       case(2)

          select case(vertices)
          case(3)
             !Triangles.

             if (degree>TRI_MAX_DEGREE) then
                ele_num=>null()
                return
             else
                ele_num=>tri_numbering(degree)
                return
             end if

          case (4)
             ! Quads

             if (degree>QUAD_MAX_DEGREE) then
                ele_num=>null()
                return
             else
                ele_num=>quad_numbering(degree)
                return
             end if

          case default
             ele_num=>null()
             return
          end select

       case(3)

          select case (vertices)
          case (4)
             !Tets

             if (degree>TET_MAX_DEGREE) then
                ele_num=>null()
                return
             else
                ele_num=>tet_numbering(degree)
                return
             end if

          case (8)
             ! Hexes

             if (degree>HEX_MAX_DEGREE) then
                ele_num=>null()
                return
             else
                ele_num=>hex_numbering(degree)
                return
             end if

          case default
             ele_num=>null()
             return
          end select

       case default
          ele_num=>null()
          return
       end select

    case (ELEMENT_NONCONFORMING)

       assert(vertices==3)
       assert(dimension==2)
       assert(degree==1)
       ele_num=>tri_numbering_nc

!    case (ELEMENT_NONCONFORMING_FACE)

!       assert(vertices==3)
!       assert(dimension==2)
!       assert(degree==1)
!       ele_num=>tri_numbering_nc

    case (ELEMENT_TRACE)

       if(dimension /= 2) then
          FLAbort('Trace elements only currently coded for 2D')
       end if
       select case (vertices)
       case (3)
          ele_num=>tri_numbering_trace(degree)
       case (4)
          ele_num=>quad_numbering_trace(degree)
       case default
          FLAbort('Vertex count not supported for trace elements')
       end select
    case (ELEMENT_BUBBLE)

       select case(dimension)
       case(1)

          select case(vertices)
          case(2)
             ! Intervals - the only possibility.
             if (degree/=1) then
                ele_num=>null()
                return
             else
                ele_num=>interval_numbering_bubble
                return
             end if

          case default
             ele_num=>null()
             return
          end select

       case(2)

          select case(vertices)
          case(3)
             !Triangles.

             if ((degree==0).or.(degree>TRI_BUBBLE_MAX_DEGREE)) then
                ele_num=>null()
                return
             else
                ele_num=>tri_numbering_bubble(degree)
                return
             end if

          case default
             ele_num=>null()
             return
          end select

       case(3)

          select case(vertices)
          case(4)
             !Tets.

             if ((degree==0).or.(degree>TET_BUBBLE_MAX_DEGREE)) then
                ele_num=>null()
                return
             else
                ele_num=>tet_numbering_bubble(degree)
                return
             end if

          case default
             ele_num=>null()
             return
          end select

       case default
          ele_num=>null()
          return
       end select

    case default

       FLAbort('Attempt to select an illegal element type.')

    end select

  end function find_element_numbering

  subroutine number_elements
    ! Fill the values in in element_numbering.

    ! make sure this is idempotent.
    if (initialised) return
    initialised=.true.

    call number_tets_lagrange
    call number_tets_bubble
    call number_triangles_lagrange
    call number_triangles_bubble
    call number_triangles_trace
    call number_triangles_nc
    call number_intervals_lagrange
    call number_intervals_bubble
    call number_point_lagrange
    call number_hexes_lagrange
    call number_quads_lagrange
    call number_quads_trace

  end subroutine number_elements

  subroutine number_tets_lagrange
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt
    integer, dimension(4) :: l
    type(ele_numbering_type), pointer :: ele

    ! Currently only tets are supported.
    tet_numbering%faces=4
    tet_numbering%vertices=4
    tet_numbering%edges=6
    tet_numbering%dimension=3
    tet_numbering%facets=4
    tet_numbering%family=FAMILY_SIMPLEX
    tet_numbering%type=ELEMENT_LAGRANGIAN

    degree_loop: do i=0,TET_MAX_DEGREE
       ele=>tet_numbering(i)
       ele%degree=i

       if (i>0) then
          ele%nodes_per(0)=1
          ele%nodes_per(1)=i-1
          ele%nodes_per(2)=tr(i-2)
          ele%nodes_per(3)=te(i-3)
       else
          ele%nodes_per(0:2)=0
          ele%nodes_per(3)=1
       end if

       ! Allocate mappings:
       allocate(ele%count2number(0:i,0:i,0:i))
       allocate(ele%number2count(ele%dimension+1,te(i+1)))
       allocate(ele%facet_coord(ele%faces))
       allocate(ele%facet_val(ele%faces))

       ele%nodes=te(i+1)

       ele%count2number=0
       ele%number2count=0

       l=0
       l(1)=i

       cnt=0

       number_loop: do

          cnt=cnt+1

          ele%count2number(l(1), l(2), l(3))=cnt
          ele%number2count(:,cnt)=l

          ! If the last index has reached the current degree then we are
          ! done.
          if (l(4)==i) exit number_loop

          ! Increment the index counter.
          l(2)=l(2)+1

          do j=2,3
             ! This comparison implements the decreasing dimension lengths
             ! as you move up the pyramid.
             if (l(j)>i-sum(l(j+1:))) then
                l(j)=0
                l(j+1)=l(j+1)+1
             end if
          end do

          l(1)=i-sum(l(2:))


       end do number_loop

       ! Sanity test
       if (te(i+1)/=cnt) then
          ewrite(3,*) 'Counting error', i, te(i+1), cnt
          stop
       end if

       ! Number faces.
       forall(j=1:ele%faces)
          ele%facet_coord(j)=j
       end forall
       ! In a tet all faces occur on planes of zero value for one local coord.
       ele%facet_val=0

    end do degree_loop


  end subroutine number_tets_lagrange

  subroutine number_tets_bubble
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt
    integer, dimension(4) :: l
    type(ele_numbering_type), pointer :: ele

    ! Currently only tets are supported.
    tet_numbering_bubble%faces=4
    tet_numbering_bubble%vertices=4
    tet_numbering_bubble%edges=6
    tet_numbering_bubble%dimension=3
    tet_numbering_bubble%facets=4
    tet_numbering_bubble%family=FAMILY_SIMPLEX
    tet_numbering_bubble%type=ELEMENT_BUBBLE

    degree_loop: do i=1,TET_BUBBLE_MAX_DEGREE
       ele=>tet_numbering_bubble(i)
       ele%degree=i

       ele%nodes_per(0)=1
       ele%nodes_per(1)=i-1
       ele%nodes_per(2)=tr(i-2)
       ele%nodes_per(3)=te(i-3)+1

       ! Allocate mappings:
       allocate(ele%count2number(0:i*(ele%dimension+1),0:i*(ele%dimension+1),0:i*(ele%dimension+1)))
       allocate(ele%number2count(ele%dimension+1,te(i+1)+1))
       allocate(ele%facet_coord(ele%faces))
       allocate(ele%facet_val(ele%faces))

       ele%nodes=te(i+1)+1
       ele%count2number=0
       ele%number2count=0

       l=0
       l(1)=i*(ele%dimension+1)

       cnt=0

       number_loop: do

          cnt=cnt+1

          ele%count2number(l(1), l(2), l(3))=cnt
          ele%number2count(:,cnt)=l

          ! If the last index has reached the current degree then we are
          ! done.
          if (l(4)==i*(ele%dimension+1)) exit number_loop

          ! Increment the index counter.
          l(2)=l(2)+ele%dimension+1

          do j=2,3
             ! This comparison implements the decreasing dimension lengths
             ! as you move up the pyramid.
             if (l(j)>i*(ele%dimension+1)-sum(l(j+1:))) then
                l(j)=0
                l(j+1)=l(j+1)+ele%dimension+1
             end if
          end do

          l(1)=i*(ele%dimension+1)-sum(l(2:))


       end do number_loop

       ! add in the bubble node
       l(1) = i
       l(2) = i
       l(3) = i
       l(4) = i
       cnt=cnt+1
       ele%count2number(l(1), l(2), l(3))=cnt
       ele%number2count(:,cnt)=l

       ! Sanity test
       if (te(i+1)+1/=cnt) then
          ewrite(-1,*) 'degree, nodes, cnt = ', i, te(i+1)+1, cnt
          FLAbort("Counting error")
       end if

       ! Number faces.
       forall(j=1:ele%faces)
          ele%facet_coord(j)=j
       end forall
       ! In a tet all faces occur on planes of zero value for one local coord.
       ele%facet_val=0

    end do degree_loop


  end subroutine number_tets_bubble

  subroutine number_triangles_lagrange
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt
    integer, dimension(3) :: l
    type(ele_numbering_type), pointer :: ele

    tri_numbering%faces=1
    tri_numbering%vertices=3
    tri_numbering%edges=3
    tri_numbering%dimension=2
    tri_numbering%facets=3
    tri_numbering%family=FAMILY_SIMPLEX
    tri_numbering%type=ELEMENT_LAGRANGIAN

    ! Degree 0 elements are a special case.
    ele=>tri_numbering(0)
    ele%degree=0

    degree_loop: do i=0,TRI_MAX_DEGREE
       ele=>tri_numbering(i)
       ele%degree=i

       if (i>0) then
          ele%nodes_per(0)=1
          ele%nodes_per(1)=i-1
          ele%nodes_per(2)=tr(i-2)
       else
          ele%nodes_per(0:1)=0
          ele%nodes_per(2)=1
       end if
       ele%nodes_per(3)=0

       ! Allocate mappings:
       allocate(ele%count2number(0:i,0:i,0:i))
       allocate(ele%number2count(ele%dimension+1,tr(i+1)))
       allocate(ele%facet_coord(ele%vertices))
       allocate(ele%facet_val(ele%vertices))

       ele%nodes=tr(i+1)
       ele%count2number=0
       ele%number2count=0

       l=0
       l(1)=i

       cnt=0

       number_loop: do

          cnt=cnt+1

          ele%count2number(l(1), l(2), l(3))=cnt
          ele%number2count(:,cnt)=l

          ! If the last index has reached the current degree then we are
          ! done.
          if (l(3)==i) exit number_loop

          ! Increment the index counter.
          l(2)=l(2)+1

          do j=2,2
             ! This comparison implements the decreasing dimension lengths
             ! as you move up the triangle.
             if (l(j)>i-sum(l(j+1:))) then
                l(j)=0
                l(j+1)=l(j+1)+1
             end if
          end do

          l(1)=i-sum(l(2:))


       end do number_loop

       ! Sanity test
       if (tr(i+1)/=cnt) then
          ewrite(3,*) 'Counting error', i, tr(i+1), cnt
          stop
       end if

       ! Number edges.
       forall(j=1:ele%vertices)
          ele%facet_coord(j)=j
       end forall
       ! In a triangle all faces occur on planes of zero value for one local coord.
       ele%facet_val=0
    end do degree_loop

  end subroutine number_triangles_lagrange

  subroutine number_triangles_bubble
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt
    integer, dimension(3) :: l
    type(ele_numbering_type), pointer :: ele

    tri_numbering_bubble%faces=1
    tri_numbering_bubble%vertices=3
    tri_numbering_bubble%edges=3
    tri_numbering_bubble%dimension=2
    tri_numbering_bubble%facets=3
    tri_numbering_bubble%family=FAMILY_SIMPLEX
    tri_numbering_bubble%type=ELEMENT_BUBBLE

    degree_loop: do i=1,TRI_BUBBLE_MAX_DEGREE
       ele=>tri_numbering_bubble(i)
       ele%degree=i

       ele%nodes_per(0)=1
       ele%nodes_per(1)=i-1
       ele%nodes_per(2)=tr(i-2)+1
       ele%nodes_per(3)=0

       ! Allocate mappings:
       allocate(ele%count2number(0:i*(ele%dimension+1),0:i*(ele%dimension+1),0:i*(ele%dimension+1)))
       allocate(ele%number2count(ele%dimension+1,tr(i+1)+1))
       allocate(ele%facet_coord(ele%vertices))
       allocate(ele%facet_val(ele%vertices))

       ele%nodes=tr(i+1)+1
       ele%count2number=0
       ele%number2count=0

       l=0
       l(1)=i*(ele%dimension+1)

       cnt=0

       number_loop: do

          cnt=cnt+1

          ele%count2number(l(1), l(2), l(3))=cnt
          ele%number2count(:,cnt)=l

          ! If the last index has reached the current degree then we are
          ! done.
          if (l(3)==i*(ele%dimension+1)) exit number_loop

          ! Increment the index counter.
          l(2)=l(2)+ele%dimension+1

          do j=2,2
             ! This comparison implements the decreasing dimension lengths
             ! as you move up the triangle.
             if (l(j)>i*(ele%dimension+1)-sum(l(j+1:))) then
                l(j)=0
                l(j+1)=l(j+1)+ele%dimension+1
             end if
          end do

          l(1)=i*(ele%dimension+1)-sum(l(2:))


       end do number_loop

      ! add in the bubble node
      l(1) = i
      l(2) = i
      l(3) = i
      cnt=cnt+1
      ele%count2number(l(1), l(2), l(3))=cnt
      ele%number2count(:,cnt)=l

       ! Sanity test
       if (tr(i+1)+1/=cnt) then
          ewrite(-1,*) 'degree, nodes, cnt = ', i, tr(i+1)+1, cnt
          FLAbort("Counting error")
       end if

       ! Number edges.
       forall(j=1:ele%vertices)
          ele%facet_coord(j)=j
       end forall
       ! In a triangle all faces occur on planes of zero value for one local coord.
       ele%facet_val=0
    end do degree_loop

  end subroutine number_triangles_bubble

  subroutine number_triangles_nc
    ! Fill the values in in element_numbering.
    integer :: j
    type(ele_numbering_type), pointer :: ele

    tri_numbering_nc%faces=1
    tri_numbering_nc%vertices=3
    tri_numbering_nc%edges=3
    tri_numbering_nc%dimension=2
    tri_numbering_nc%type=ELEMENT_NONCONFORMING
    tri_numbering_nc%facets=3
    tri_numbering_nc%family=FAMILY_SIMPLEX
    tri_numbering_nc%type=ELEMENT_NONCONFORMING

    ele=>tri_numbering_nc
    ele%degree=1

    ele%nodes_per(0)=0
    ele%nodes_per(1)=1
    ele%nodes_per(2)=0
    ele%nodes_per(3)=0

    ! Allocate mappings:
    allocate(ele%count2number(0:ele%degree,0:ele%degree,0:ele%degree))
    allocate(ele%number2count(ele%dimension+1,tr(ele%degree+1)))
    allocate(ele%facet_coord(ele%vertices))
    allocate(ele%facet_val(ele%vertices))

    ele%nodes=tr(ele%degree+1)
    ele%count2number=0
    ele%number2count=0

    ! NOTE THAT THE FOLLOWING MAPPING IS NOT 1:1 !!!!!
    ! These are the mappings for the shape function locations.

    ! This is the relationship between shape function and vertex numbers.
    ! (Only for this element type)
    !          3
    !         / \
    !        2   1
    !       /     \
    !      1---3---2
    ele%count2number(0,1,1)=1
    ele%count2number(1,0,1)=2
    ele%count2number(1,1,0)=3
    ! These are the mappings for the vertices.
    ele%count2number(1,0,0)=1
    ele%count2number(0,1,0)=2
    ele%count2number(0,0,1)=3

    ! NOTE THAT INVERSE MAPPINGS ARE ONLY DEFINED FOR SHAPE FUNCTIONS.
    ele%number2count(:,1)=(/0,1,1/)
    ele%number2count(:,2)=(/1,0,1/)
    ele%number2count(:,3)=(/1,1,0/)

    ! Number edges.
    forall(j=1:ele%vertices)
       ele%facet_coord(j)=j
    end forall
    ! In a triange all faces occur on planes of zero value for one local coord.
    ele%facet_val=0

  end subroutine number_triangles_nc

  subroutine number_triangles_trace
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt, ll
    integer, dimension(3) :: l
    type(ele_numbering_type), pointer :: ele

    tri_numbering_trace%faces=1
    tri_numbering_trace%vertices=0 ! This is set to zero because no DOFs
    ! located on vertices.
    tri_numbering_trace%edges=3
    tri_numbering_trace%dimension=2
    tri_numbering_trace%facets=3
    tri_numbering_trace%family=FAMILY_SIMPLEX
    tri_numbering_trace%type=ELEMENT_TRACE

    ! Degree 0 elements are a special case.
    ele=>tri_numbering_trace(0)
    ele%degree=0

    degree_loop: do i=0,TRI_MAX_DEGREE
       ele=>tri_numbering_trace(i)
       ele%degree=i

       ! Allocate mappings:

       ele%nodes=(ele%dimension+1)*(ele%degree+1) !faces x floc

       ! For trace elements, the first index is the facet number.
       allocate(ele%count2number(1:ele%dimension+1,0:i,0:i))
       allocate(ele%number2count(ele%dimension+1,ele%nodes))
       allocate(ele%facet_coord(3))
       allocate(ele%facet_val(3))

       ele%count2number=0
       ele%number2count=0

       l=0
       l(1)=ele%degree

       cnt=0

       facet_loop: do ll=1,ele%dimension+1

          l=0
          l(1)=ll
          number_loop: do j=0,ele%degree

             cnt=cnt+1
             l(2:3)=(/ele%degree-j,j/)

             ele%count2number(l(1), l(2), l(3))=cnt
             ele%number2count(:,cnt)=l

          end do number_loop
       end do facet_loop

       ! Sanity test
       if (ele%nodes/=cnt) then
          ewrite(3,*) 'Counting error', i, ele%nodes, cnt
          stop
       end if

       ! For trace elements, the first local_coordinate is the face number.
       ele%facet_coord=1
       forall(j=1:ele%facets)
          ! The first local coordinate labels the face.
          ele%facet_val(j)=j
       end forall

    end do degree_loop

  end subroutine number_triangles_trace

  subroutine number_quads_trace
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt, ll
    integer, dimension(3) :: l
    type(ele_numbering_type), pointer :: ele

    quad_numbering_trace%faces=1
    quad_numbering_trace%vertices=0
    quad_numbering_trace%edges=4
    quad_numbering_trace%dimension=2
    quad_numbering_trace%facets=4
    quad_numbering_trace%family=FAMILY_CUBE
    quad_numbering_trace%type=ELEMENT_TRACE

    ! Degree 0 elements are a special case.
    ele=>quad_numbering_trace(0)
    ele%degree=0

    degree_loop: do i=0,QUAD_MAX_DEGREE
       ele=>quad_numbering_trace(i)
       ele%degree=i

       ele%nodes_per(0)=0
       ele%nodes_per(1)=i+1
       ele%nodes_per(2)=0
       ele%nodes_per(3)=0

       ! Allocate mappings:

       ele%nodes= 2**ele%dimension * (ele%degree + 1) ! faces x floc

       ! For trace elements, the first index is the facet number.
       allocate(ele%count2number(1:2*ele%dimension,0:i,0:i))
       allocate(ele%number2count(ele%dimension+1,ele%nodes))
       allocate(ele%facet_coord(ele%facets))
       allocate(ele%facet_val(ele%facets))

       ele%count2number=0
       ele%number2count=0

       cnt=0

       facet_loop: do ll=1,2*ele%dimension

          l=0

          l(1)=ll
          number_loop: do j=0,ele%degree

             cnt=cnt+1
             l(2:3)=(/ele%degree-j,j/)

             ele%count2number(l(1), l(2), l(3))=cnt
             ele%number2count(:,cnt)=l

          end do number_loop
       end do facet_loop

       ! Sanity test
       if (ele%nodes/=cnt) then
          ewrite(0,*) 'Counting error', i, ele%nodes, cnt
          stop
       end if

       ! For trace elements, the first local_coordinate is the face number.
       ele%facet_coord=1
       forall(j=1:ele%facets)
          ! The first local coordinate labels the face.
          ele%facet_val(j)=j
       end forall
    end do degree_loop

  end subroutine number_quads_trace

  subroutine number_intervals_lagrange
    ! Fill the values in in element_numbering.
    integer :: i, j, cnt
    integer, dimension(2) :: l
    type(ele_numbering_type), pointer :: ele

    interval_numbering%faces=0
    interval_numbering%vertices=2
    interval_numbering%edges=1
    interval_numbering%dimension=1
    interval_numbering%facets=2
    interval_numbering%family=FAMILY_SIMPLEX
    interval_numbering%type=ELEMENT_LAGRANGIAN

    ! Degree 0 elements are a special case.
    ele=>interval_numbering(0)
    ele%degree=0

    degree_loop: do i=0,INTERVAL_MAX_DEGREE
       ele=>interval_numbering(i)
       ele%degree=i

       if (i>0) then
          ele%nodes_per(0)=1
          ele%nodes_per(1)=i-1
       else
          ele%nodes_per(0)=0
          ele%nodes_per(1)=1
       end if
       ele%nodes_per(2)=0
       ele%nodes_per(3)=0

       ! Allocate mappings:
       allocate(ele%count2number(0:i,0:i,0:0))
       allocate(ele%number2count(ele%dimension+1,i+1))
       allocate(ele%facet_coord(ele%vertices))
       allocate(ele%facet_val(ele%vertices))

       ele%nodes=i+1
       ele%count2number=0
       ele%number2count=0

       l=0
       l(1)=i

       cnt=0

       number_loop: do

          cnt=cnt+1

          ele%count2number(l(1), l(2), 0)=cnt
          ele%number2count(:,cnt)=l

          ! If the last index has reached the current degree then we are
          ! done.
          if (l(2)==i) exit number_loop

          ! Increment the index counter.
          l(2)=l(2)+1

          l(1)=i-l(2)

       end do number_loop

       ! Sanity test
       if (i+1/=cnt) then
          ewrite(3,*) 'Counting error', i, i+1, cnt
          stop
       end if

       ! Number edges.
       forall(j=1:ele%vertices)
          ele%facet_coord(j)=j
       end forall
       ! In an interval all faces occur on planes of zero value for one local coord.
       ele%facet_val=0

    end do degree_loop

  end subroutine number_intervals_lagrange

  subroutine number_intervals_bubble
    ! Fill the values in in element_numbering.
    integer :: j, cnt
    integer, dimension(2) :: l
    type(ele_numbering_type), pointer :: ele

    interval_numbering_bubble%faces=0
    interval_numbering_bubble%vertices=2
    interval_numbering_bubble%edges=1
    interval_numbering_bubble%dimension=1
    interval_numbering_bubble%facets=2
    interval_numbering_bubble%family=FAMILY_SIMPLEX
    interval_numbering_bubble%type=ELEMENT_BUBBLE

    ! we cannot exceed the bubble max degree of 1 because
    ! the count2number map becomes nonunique when two nodes
    ! are co-located
     ele=>interval_numbering_bubble
     ele%degree=1

     ele%nodes_per(0)=1
     ele%nodes_per(1)=1
     ele%nodes_per(2)=0
     ele%nodes_per(3)=0

     ! Allocate mappings:
     ! we need a lot of blank spaces here to make this
     ! mapping bijective!
     allocate(ele%count2number(0:(ele%dimension+1),0:(ele%dimension+1),0:0))
     allocate(ele%number2count(ele%dimension+1,3))
     allocate(ele%facet_coord(ele%vertices))
     allocate(ele%facet_val(ele%vertices))

     ele%nodes=3
     ele%count2number=0
     ele%number2count=0

     l=0
     l(1)=ele%dimension+1

     cnt=0

     number_loop: do
        ! this loop just takes care of the standard lagrangian element
        ! nodes (i.e. it intentionally excludes the bubble node)

        cnt=cnt+1

        ele%count2number(l(1), l(2), 0)=cnt
        ele%number2count(:,cnt)=l

        ! If the last index has reached the current degree then we are
        ! done.
        if (l(2)==(ele%dimension+1)) exit number_loop

        ! Increment the index counter.
        l(2)=l(2)+ele%dimension+1

        l(1)=ele%dimension+1-l(2)

     end do number_loop

     ! add in the bubble node
     l(1) = 1
     l(2) = 1
     cnt = cnt +1
     ele%count2number(l(1), l(2), 0)=cnt
     ele%number2count(:,cnt)=l

     ! Sanity test
     if (cnt/=3) then
        ewrite(-1,*) 'Counting error', 1, 3, cnt
        FLAbort("Counting error.")
     end if

     ! Number edges.
     forall(j=1:ele%vertices)
        ele%facet_coord(j)=j
     end forall
     ! In an interval all faces occur on planes of zero value for one local coord.
     ele%facet_val=0

  end subroutine number_intervals_bubble

  subroutine number_point_lagrange
    !!< The highly complex 1 point 0D element.
    type(ele_numbering_type), pointer :: ele

    point_numbering%faces=0
    point_numbering%vertices=1
    point_numbering%edges=0
    point_numbering%dimension=0
    point_numbering%facets=0
    point_numbering%family=FAMILY_SIMPLEX
    point_numbering%type=ELEMENT_LAGRANGIAN

    ! Degree 0 elements are a special case.
    ele=>point_numbering(0)
    ele%degree=0

    ele%nodes_per(0)=1
    ele%nodes_per(1)=0
    ele%nodes_per(2)=0
    ele%nodes_per(3)=0

    ! Allocate mappings:
    allocate(ele%count2number(0:0,0:0,0:0))
    allocate(ele%number2count(ele%dimension+1,1))
    allocate(ele%facet_coord(0))
    allocate(ele%facet_val(0))

    ele%nodes=1
    ele%count2number=1
    ele%number2count=0

  end subroutine number_point_lagrange

  subroutine number_hexes_lagrange
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt, l1, l2, l3
    type(ele_numbering_type), pointer :: ele

    ! Currently only hexes are supported.
    hex_numbering%faces=6
    hex_numbering%vertices=8
    hex_numbering%edges=12
    hex_numbering%dimension=3
    hex_numbering%facets=6
    hex_numbering%family=FAMILY_CUBE
    hex_numbering%type=ELEMENT_LAGRANGIAN

    ! Degree 0 elements are a special case.
    ele=>hex_numbering(0)
    ele%degree=0

    degree_loop: do i=0,HEX_MAX_DEGREE
       ele=>hex_numbering(i)
       ele%degree=i

       if (i>0) then
          ele%nodes_per(0)=1
          ele%nodes_per(1)=i-1
          ele%nodes_per(2)=ele%nodes_per(1)**2
          ele%nodes_per(3)=ele%nodes_per(1)**3
       else
          ele%nodes_per(0:2)=0
          ele%nodes_per(3)=1
       end if

       ! Allocate mappings:
       allocate(ele%count2number(0:i,0:i,0:i))
       allocate(ele%number2count(ele%dimension,(i+1)**3))
       allocate(ele%facet_coord(ele%faces))
       allocate(ele%facet_val(ele%faces))

       ele%nodes=(i+1)**3

       ele%count2number=0
       ele%number2count=0

       cnt=0

       do l1=0, ele%degree

          do l2=0, ele%degree

             do l3=0, ele%degree

                cnt=cnt+1

                ele%count2number(l1, l2, l3)=cnt
                ele%number2count(:,cnt)=(/l1, l2, l3/)

             end do

          end do

       end do

       ! Number faces.
       ele%facet_coord((/1,6/))=3
       ele%facet_coord((/2,5/))=2
       ele%facet_coord((/3,4/))=1
       ! In a hex all faces occur on planes of value -1/+1 for one local
       ! coord, that is resp. 0 and degree in count coordinates
       ele%facet_val((/4,5,6/))=0
       ele%facet_val((/1,2,3/))=ele%degree

    end do degree_loop

  end subroutine number_hexes_lagrange

  subroutine number_quads_lagrange
    ! Fill the values in in element_numbering.
    integer :: i,j, cnt, l1, l2
    type(ele_numbering_type), pointer :: ele

    ! Currently only quads are supported.
    quad_numbering%faces=1
    quad_numbering%vertices=4
    quad_numbering%edges=4
    quad_numbering%dimension=2
    quad_numbering%facets=4
    quad_numbering%family=FAMILY_CUBE
    quad_numbering%type=ELEMENT_LAGRANGIAN

    ! Degree 0 elements are a special case.
    ele=>quad_numbering(0)
    ele%degree=0

    degree_loop: do i=0,QUAD_MAX_DEGREE
       ele=>quad_numbering(i)
       ele%degree=i

       if (i>0) then
          ele%nodes_per(0)=1
          ele%nodes_per(1)=i-1
          ele%nodes_per(2)=ele%nodes_per(1)**2
       else
          ele%nodes_per(0:1)=0
          ele%nodes_per(2)=1
       end if
       ele%nodes_per(3)=0


       ! Allocate mappings:
       allocate(ele%count2number(0:i,0:i,0:0))
       allocate(ele%number2count(ele%dimension,(i+1)**2))
       allocate(ele%facet_coord(ele%vertices))
       allocate(ele%facet_val(ele%vertices))

       ele%nodes=(i+1)**2

       ele%count2number=0
       ele%number2count=0

       cnt=0

       do l2=0, ele%degree

          do l1=0, ele%degree

                cnt=cnt+1

                ele%count2number(l1, l2, 0)=cnt
                ele%number2count(:,cnt)=(/l1, l2/)

          end do

       end do

       ! Number faces.
       ele%facet_coord((/2,3/))=1
       ele%facet_coord((/1,4/))=2
       ! In a quad all faces occur on planes of value 1 or 0 for one local
       ! coord. That is resp. 0 and ele%degree in 'count' coordinates
       ele%facet_val((/3,4/))=0
       ele%facet_val((/1,2/))=ele%degree

    end do degree_loop

  end subroutine number_quads_lagrange

  pure function tr(n)
    ! Return the nth triangular number
    integer :: tr
    integer, intent(in) :: n

    tr=max(n,0)*(n+1)/2

  end function tr

  pure function te(n)
    ! Return the nth tetrahedral number
    integer :: te
    integer, intent(in) :: n

    te=max(n,0)*(n+1)*(n+2)/6

  end function te

  pure function inv_tr(m) result (n)
    ! Return n where m=tr(n). If m is not a triangular number, return -1
    integer :: n
    integer, intent(in) :: m

    integer :: i, tri

    i=0

    do
       i=i+1

       tri=tr(i)

       if (tri==m) then
          n=i
          return
       else if (tri>m) then
          ! m is not a triangular number.
          n=-1
          return
       end if

    end do

  end function inv_tr

  pure function inv_te(m) result (n)
    ! Return n where m=te(n). If m is not a tetrahedral number, return -1
    integer :: n
    integer, intent(in) :: m

    integer :: i, tei

    i=0

    do
       i=i+1

       tei=te(i)

       if (tei==m) then
          n=i
          return
       else if (tei>m) then
          ! m is not a tetrahedral number.
          n=-1
          return
       end if

    end do

  end function inv_te

  pure function element_num_equal(element_num1,element_num2)
    ! Return true if the two element_nums are equivalent.
    logical :: element_num_equal
    type(ele_numbering_type), intent(in) :: element_num1, element_num2

    element_num_equal = element_num1%faces==element_num2%faces &
         .and. element_num1%vertices==element_num2%vertices &
         .and. element_num1%edges==element_num2%edges &
         .and. element_num1%dimension==element_num2%dimension &
         .and. element_num1%nodes==element_num2%nodes &
         .and. element_num1%degree==element_num2%degree &
         .and. element_num1%facets==element_num2%facets

  end function element_num_equal

  function svertex_num(node, element, ele_num, stat)
    ! Given a global vertex node number and a vector of node numbers
    ! defining a tet or triangle, return the local node number of that vertex.
    !
    ! If the element numbering ele_num is present then the node number in
    ! that element is returned. Otherwise the node number for a linear element
    ! is returned.
    !
    ! If stat is present then it returns 1 if node is not in the
    ! element and 0 otherwise.
    integer :: svertex_num
    integer, intent(in) :: node
    integer, dimension(:), intent(in) :: element
    type(ele_numbering_type), intent(in), optional :: ele_num
    integer, intent(out), optional :: stat

    integer, dimension(4) :: l
    integer :: i, c

    if (present(stat)) then
       stat=0
    end if

    ! Find the vertex number on the tet.
    svertex_num=minloc(array=element, dim=1, mask=(node==element))

    if (svertex_num==0) then
       if (present(stat)) then
          stat=1
          return
       else
          FLAbort("Node is not part of an element in vertex_num.")
       end if
    end if

    if (present(ele_num)) then
       ! Special case: 0 order elements have no vertices.
       if (ele_num%degree==0) then
          svertex_num=0
          return
       end if

       ! Calculate the local count coordinate.
       select case(ele_num%type)
       case(ELEMENT_LAGRANGIAN)
          select case (ele_num%family)
          case (FAMILY_SIMPLEX)
             l=0
             l(svertex_num)=ele_num%degree
          case (FAMILY_CUBE)
             l=0
             c=1 ! coordinate counter
             do i=1, svertex_num
                do c=1, 2
                   if (l(c)==0) then
                      l(c)=ele_num%degree
                      exit
                   else
                      ! switch back to 0, continue with next binary digit
                      l(c)=0
                   end if
                end do
             end do
          case default
             FLAbort('Unknown element shape.')
          end select
       case(ELEMENT_BUBBLE)
         l=0
         l(svertex_num)=ele_num%degree*(ele_num%dimension+1)
       case(ELEMENT_TRACE)
         FLAbort("Trace elements do not have well defined vertices")
       case default
         FLAbort("Unknown element type")
       end select

       ! Look up the node number of the vertex.
       svertex_num=ele_num%count2number(l(1), l(2), l(3))
    end if

  end function svertex_num

  function vvertex_num(nodes, element, ele_num, stat)
    ! Given a vector of global vertex node numbers and a vector of node
    ! numbers defining a tet or triangle, return the local node number of those
    ! vertices.
    !
    ! If the element numbering ele_num is present then the node numbers in
    ! that element are returned. Otherwise the node numbers for a linear element
    ! are returned.
    !
    ! If stat is present then it returns 1 if node is not in the
    ! element and 0 otherwise.
    integer, dimension(:), intent(in) :: nodes
    integer, dimension(:), intent(in) :: element
    type(ele_numbering_type), intent(in), optional :: ele_num
    integer, intent(out), optional :: stat
    integer, dimension(size(nodes)) :: vvertex_num

    integer :: i

    if (present(stat)) then
       stat=0
    end if

    do i=1,size(nodes)
       vvertex_num(i)=vertex_num(nodes(i), element, ele_num, stat)
    end do

  end function vvertex_num

  function ele_num_local_vertices(ele_num)
    ! Given an element numbering, return the local node numbers of its
    ! vertices.
    type(ele_numbering_type), intent(in) :: ele_num
    integer, dimension(ele_num%vertices) :: ele_num_local_vertices

    integer, dimension(4) :: l
    integer :: i, c

    select case (ele_num%type)
    case (ELEMENT_LAGRANGIAN)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)

         ! Simplices
         do i=1,ele_num%vertices
           l=0
           l(i)=ele_num%degree

           ele_num_local_vertices(i)=ele_num%count2number(l(1),l(2),l(3))

         end do

      case (FAMILY_CUBE)

         ! Degree zero element only has one node which is non-zero on all vertices.
         if(ele_num%degree == 0) then
            ele_num_local_vertices = 1
            return
         end if

         l=0
         c=1 ! coordinate counter
         do i=1, ele_num%vertices
           ele_num_local_vertices( i )=ele_num%count2number(l(1), l(2), l(3))
           do c=1, ele_num%vertices
             if (l(c)==0) then
               l(c)=ele_num%degree
               exit
             else
               ! switch back to 0, continue with next binary digit
               l(c)=0
             end if
           end do
         end do
         assert(c==ele_num%dimension+1)

      case default

         FLAbort('Unknown element shape.')

      end select

    case (ELEMENT_BUBBLE)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)

         ! Simplices
         do i=1,ele_num%vertices
           l=0
           l(i)=ele_num%degree*(ele_num%dimension+1)

           ele_num_local_vertices(i)=ele_num%count2number(l(1),l(2),l(3))

         end do

      case default

         FLAbort('Unknown element shape.')

      end select

    case (ELEMENT_TRACE)
       continue ! Trace elements have no vertex DOFs.

    case default
      FLAbort("Unknown element type")
    end select

  end function ele_num_local_vertices

  !------------------------------------------------------------------------
  ! Extract element boundaries
  !------------------------------------------------------------------------

  pure function facet_num_length(ele_num,interior)
    !!< Determine the length of the vector returned by facet_num.
    type(ele_numbering_type), intent(in) :: ele_num
    logical, intent(in) :: interior
    integer :: facet_num_length

    select case (ele_num%dimension)
    case(1)
       if (interior) then
          facet_num_length=0
       else
          facet_num_length=1
       end if
    case (2)
       if (interior) then
          facet_num_length=ele_num%degree-1
       else
          facet_num_length=ele_num%degree+1
       end if
    case (3)
       facet_num_length=face_num_length(ele_num, interior)
    end select

  end function facet_num_length

  pure function face_num_length(ele_num, interior)
    ! Determine the length of the vector returned by face_num.
    integer :: face_num_length
    type(ele_numbering_type), intent(in) :: ele_num
    logical, intent(in) :: interior

    select case (ele_num%family)
    case (FAMILY_SIMPLEX)
       if (interior.and.ele_num%type/=ELEMENT_TRACE) then
          face_num_length=tr(ele_num%degree-2)
       else
          face_num_length=tr(ele_num%degree+1)
       end if
    case (FAMILY_CUBE)
       if (interior.and.ele_num%type/=ELEMENT_TRACE) then
          face_num_length=(ele_num%degree-1)**2
       else
          face_num_length=(ele_num%degree+1)**2
       end if
    case default
    ! can't flabort in a pure function, sorry
    !   FLAbort("Unknown element family.")
       face_num_length=-666
    end select

  end function face_num_length

  function numbering_facet_numbering(ele_num, facet) result (numbering)
    !!< Give the local nodes associated with face face in ele_num.
    !!<
    !!< This is totally generic to all element shapes. Cute huh?
    !!<
    !!< The underlying principles are that nodes on one element facet
    !!< share one fixed local coordinate and that nodes on a facet are
    !!< ordered in the same order as those nodes occur in the element.
    integer, intent(in) :: facet
    type(ele_numbering_type), intent(in) :: ele_num
    integer, dimension(facet_num_length(ele_num, .false.)) ::&
         & numbering

    integer :: i, k, l

    k=0

    do i=1,ele_num%nodes
       if (ele_num%number2count(ele_num%facet_coord(facet),i)==&
            & ele_num%facet_val(facet)) then
          ! We are on the face.
          k=k+1
          numbering(k)=i
       end if

    end do

    ASSERT(k==size(numbering))

!!$    select case (ele_num%family)
!!$    case (FAMILY_SIMPLEX)
!!$
!!$       if (mod(facet,2)==0) then
!!$          ! reverse ordering for even faces, so that orientation is
!!$          ! always positive with respect to the element:
!!$          if (ele_num%dimension==2) then
!!$             numbering=numbering(size(numbering):1:-1)
!!$          else if (ele_num%dimension==3) then
!!$             l=1
!!$             i=size(numbering)
!!$             do
!!$               numbering(i:i+l-1)=numbering(i+l-1:i:-1)
!!$                l=l+1
!!$                i=i-l
!!$                if (i<1) exit
!!$             end do
!!$          end if
!!$
!!$       end if
!!$
!!$    case (FAMILY_CUBE)
!!$
!!$       if (facet==1 .or. facet==4 .or. facet==6) then
!!$          ! reverse ordering so that orientation is
!!$          ! always positive with respect to the element:
!!$          numbering=numbering(size(numbering):1:-1)
!!$       end if
!!$
!!$    case default
!!$       FLAbort("Unknown element family.")
!!$    end select

  end function numbering_facet_numbering

  !------------------------------------------------------------------------
  ! Edge numbering routines.
  !------------------------------------------------------------------------

!!$  function edge_num_no_int(nodes, element, ele_num,  stat)
!!$    ! This function exists only to make interior in effect an optional
!!$    ! argument.
!!$    integer, dimension(2), intent(in) :: nodes
!!$    integer, dimension(:), intent(in) :: element
!!$    type(ele_numbering_type), intent(in) :: ele_num
!!$    integer, intent(out), optional :: stat
!!$
!!$    integer, dimension(edge_num_length(ele_num, interior=.false.)) ::&
!!$         & edge_num_no_int
!!$
!!$    edge_num_no_int = edge_num_int(nodes, element, ele_num, .false., stat)
!!$
!!$  end function edge_num_no_int
!!$
!!$  function edge_num_int(nodes, element, ele_num, interior, stat)
!!$    ! Given a pair of vertex node numbers and a vector of node numbers
!!$    ! defining a tet, hex, quad, or triangle, return the node numbers
!!$    ! of the edge elements along the edge from nodes(1) to nodes(2).
!!$    !
!!$    ! The numbers returned are those for the element numbering ele_num and
!!$    ! they are in order from nodes(1) to nodes(2).
!!$    !
!!$    ! If interior is present and true it indicates that vertices are to be
!!$    ! disregarded.
!!$    !
!!$    ! If stat is present then it returns 1 if either node is not in the
!!$    ! element and 0 otherwise.
!!$    integer, dimension(2), intent(in) :: nodes
!!$    integer, dimension(:), intent(in) :: element
!!$    type(ele_numbering_type), intent(in) :: ele_num
!!$    logical, intent(in) :: interior
!!$    integer, intent(out), optional :: stat
!!$
!!$    integer, dimension(edge_num_length(ele_num, interior)) :: edge_num_int
!!$
!!$    integer, dimension(2) :: lnodes
!!$
!!$    if (present(stat)) stat=0
!!$
!!$    ! Special case: degree 0 elements to not have edge nodes and elements
!!$    ! with degree <2 do not have interior edge nodes.
!!$    if (ele_num%type/=ELEMENT_TRACE .and.(ele_num%degree==0 .or. &
!!$         present_and_true(interior).and.ele_num%degree<2)) return
!!$
!!$    ! Find the vertex numbers on the tet.
!!$    lnodes(1)=minloc(array=element, dim=1, mask=(nodes(1)==element))
!!$    lnodes(2)=minloc(array=element, dim=1, mask=(nodes(2)==element))
!!$
!!$    if (any(nodes==0)) then
!!$       edge_num_int=0
!!$       if (present(stat)) then
!!$          stat=1
!!$          return
!!$       else
!!$          FLAbort("Nodes are not part of element in edge_num_int.")
!!$       end if
!!$    end if
!!$
!!$    edge_num_int=edge_local_num(lnodes, ele_num, interior)
!!$
!!$  end function edge_num_int

  function edge_local_num(nodes, ele_num)
    ! Given a pair of local vertex node numbers (ie in the range 1..4)
    ! return the local node numbers of the edge elements along
    ! the edge from nodes(1) to nodes(2) in order from nodes(1) to nodes(2).
    !
    ! The vertex nodes are disregarded.
    !
    ! If stat is present then it returns 1 if either node is not in the
    ! element and 0 otherwise.
    integer, dimension(2), intent(in) :: nodes
    type(ele_numbering_type), intent(in) :: ele_num

    integer, dimension(ele_num%nodes_per(1)) :: edge_local_num

    integer, dimension(4) :: l
    integer :: cnt, i, j, k, inc, inc_l
    ! Local edge vertices for trace elements.
    integer, dimension(2) :: ln
    ! array for locating face in quad
    integer, dimension(7) :: sum2face
    integer, dimension(1:ele_num%vertices) :: vertices

    select case (ele_num%type)
    case (ELEMENT_LAGRANGIAN)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)
          l=0
          l(nodes(1))=ele_num%degree
          cnt=0
          number_loop: do
             ! Skip end dofs.
             if (all(l(nodes)/=0 .and. l(nodes)/=ele_num%degree)) then
                cnt=cnt+1

                edge_local_num(cnt)=ele_num%count2number(l(1), l(2), l(3))
             end if

             ! Advance the index:
             l(nodes)=l(nodes)+(/-1,1/)

             ! Check for completion
             if (any(l<0)) exit number_loop

          end do number_loop
      case (FAMILY_CUBE)

          ! If a quad element has degree zero then the local
          if(ele_num%degree == 0) then
            edge_local_num = 1
            return
          end if
         ! Get local node numbers of vertices
         vertices=local_vertices(ele_num)
          l=0
          k=1 ! bit mask
          j=0
          do i=1,ele_num%dimension
             ! compute ith 'count' coordinate
             l(i) = iand(nodes(1)-1, k)/k

             ! increment to go from node 1 to node 2: 0, -1 or +1
             inc_l = iand(nodes(2)-1, k)/k-l(i)
             ! remember the coordinate in which node 1 and 2 differ:
             if (inc_l/=0)  then
                j = i
                inc = inc_l
             end if
             k=k*2
          end do

          if (j==0) then
            FLAbort("The same node appears more than once in edge_local_num.")
          end if

          ! instead of between 0 and 1, between 0 and degree
          l=l*ele_num%degree
          do i=1, ele_num%degree-1
             l(j)=l(j)+inc
             edge_local_num(i)=ele_num%count2number(l(1), l(2), l(3))
          end do

      case default
          FLAbort("Unknown element family.")
      end select

    case (ELEMENT_BUBBLE)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)
          l=0
          l(nodes(1))=ele_num%degree*(ele_num%dimension+1)
          cnt=0
          number_loop_b: do
             ! Skip spurious boundary cases.
             if (all(l(nodes)/=0 .and. l(nodes)/=(ele_num%degree*(ele_num%dimension+1)))) then
                cnt=cnt+1

                edge_local_num(cnt)=ele_num%count2number(l(1), l(2), l(3))
             end if

             ! Advance the index:
             l(nodes)=l(nodes)+(/-(ele_num%dimension+1),ele_num%dimension+1/)

             ! Check for completion
             if (any(l<0)) exit number_loop_b

          end do number_loop_b

      case default
          FLAbort("Unknown element family.")
      end select

   case (ELEMENT_TRACE)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)
         l=0

         do i=1,3
            if(.not.any(nodes==i)) l(1)=i
         end do
         assert(l(1)/=0)

         if (nodes(2)>nodes(1)) then
            ! Counting forward
            ln = (/2,3/)
         else
            ! Counting backwards
            ln = (/3,2/)
         end if

         l(ln(1))=ele_num%degree
         cnt=0
         trace_number_loop: do
            cnt=cnt+1

            edge_local_num(cnt)=ele_num%count2number(l(1), l(2), l(3))

            ! Advance the index:
            l(ln)=l(ln)+(/-1,1/)

            ! Check for completion
            if (any(l<0)) exit trace_number_loop

         end do trace_number_loop
         assert(cnt==size(edge_local_num))

      case (FAMILY_CUBE)
         l = 0
         !facet_coord = (0,0,1,1)
         !facet_val = (0,1,0,1)
         !numbering is
         !       1
         !   3      4
         !  3        2
         !   1      2
         !       4
         sum2face = (/0,0,4,3,0,2,1/)
         if(sum(nodes)>7) then
            FLAbort('bad vertex numbers')
         end if
         !first local coordinate is face number
         l(1) = sum2face(sum(nodes))
         if(l(1)==0) then
            FLAbort('bad vertex numbers')
         end if

         if (nodes(2)>nodes(1)) then
            ! Counting forward
            ln = (/2,3/)
         else
            ! Counting backwards
            ln = (/3,2/)
         end if

         l(ln(1))=ele_num%degree
         cnt=0
         trace_number_loop1: do
            cnt=cnt+1

            edge_local_num(cnt)=ele_num%count2number(l(1), l(2), l(3))

            ! Advance the index:
            l(ln)=l(ln)+(/-1,1/)

            ! Check for completion
            if (any(l<0)) exit trace_number_loop1

         end do trace_number_loop1
         assert(cnt==size(edge_local_num))
      case default
         FLAbort("Unknown element family.")
      end select

   case default
      FLAbort("Unknown element type")
   end select

  end function edge_local_num

  !------------------------------------------------------------------------
  ! Face numbering routines.
  !------------------------------------------------------------------------

!!$  function face_num_no_int(nodes, element, ele_num,  stat)
!!$    ! This function exists only to make interior in effect an optional
!!$    ! argument.
!!$    integer, dimension(3), intent(in) :: nodes
!!$    integer, dimension(:), intent(in) :: element
!!$    type(ele_numbering_type), intent(in) :: ele_num
!!$    integer, intent(out), optional :: stat
!!$
!!$    integer, dimension(face_num_length(ele_num, interior=.false.)) ::&
!!$         & face_num_no_int
!!$
!!$    face_num_no_int = face_num_int(nodes, element, ele_num, .false., stat)
!!$
!!$  end function face_num_no_int
!!$
!!$  function face_num_int(nodes, element, ele_num, interior, stat)
!!$    ! Given a triple of vertex node numbers and a 4-vector (or 8-vector
!!$    ! in case of hexes) of node numbers defining an element,
!!$    ! return the node numbers of the face elements on the
!!$    ! face defined by nodes.
!!$    !
!!$    ! The numbers returned are those for the element numbering ele_num and
!!$    ! they are in the order given by the order in nodes.
!!$    !
!!$    ! If stat is present then it returns 1 if any node is not in the
!!$    ! element and 0 otherwise.
!!$    integer, dimension(:), intent(in) :: nodes
!!$    integer, dimension(:), intent(in) :: element
!!$    type(ele_numbering_type), intent(in) :: ele_num
!!$    logical, intent(in) :: interior
!!$    integer, intent(out), optional :: stat
!!$
!!$    integer, dimension(face_num_length(ele_num,interior)) :: face_num_int
!!$
!!$    integer :: i
!!$    integer, dimension(size(nodes)) :: lnodes
!!$
!!$    if (present(stat)) stat=0
!!$
!!$    ! Special case: degree 0 elements to not have face nodes
!!$    if (ele_num%degree==0.and.ele_num%type/=ELEMENT_TRACE) return
!!$
!!$    do i=1, 3
!!$       lnodes(i)=minloc(array=element, dim=1, mask=(element==nodes(i)))
!!$    end do
!!$
!!$    ! Minloc returns 0 if all elements of mask are false.
!!$    if (any(lnodes==0)) then
!!$       if (present(stat)) then
!!$          stat=1
!!$          return
!!$       else
!!$          FLAbort("Nodes are not part of an element in face_num_int.")
!!$       end if
!!$    end if
!!$
!!$    face_num_int=face_local_num_int(lnodes, ele_num, interior)
!!$
!!$  end function face_num_int

  function face_local_num_no_int(nodes, ele_num)
    ! This function exists only to make interior in effect an optional
    ! argument.
    integer, dimension(:), intent(in) :: nodes
    type(ele_numbering_type), intent(in) :: ele_num
    !output variable
    integer, dimension(face_num_length(ele_num, interior=.false.)) ::&
         & face_local_num_no_int

    face_local_num_no_int = face_local_num_int(nodes, ele_num, .false.)

  end function face_local_num_no_int

  function face_local_num_int(nodes, ele_num, interior)
    ! Given a triple of local vertex node numbers (ie in the range 1-4,
    ! or in the range 1-8 in the case of hexes)
    ! return the node numbers of the face elements on the
    ! face defined by those nodes.
    !
    ! The numbers returned are those for the element numbering ele_num and
    ! they are in the order given by the order in nodes.
    integer, dimension(:), intent(in) :: nodes
    type(ele_numbering_type), intent(in) :: ele_num
    logical, intent(in) :: interior

    integer, dimension(face_num_length(ele_num,interior)) :: face_local_num_int

    integer, dimension(4) :: l
    integer :: i, j, k, cnt, j12, j13, inc12, inc13, tmp

    select case (ele_num%type)
    case (ELEMENT_LAGRANGIAN)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)
          l=0
          l(nodes(1))=ele_num%degree
          cnt=0

          number_loop: do
             ! Skip spurious facet cases.
             if ((.not.interior) .or. all(&
                  l(nodes)/=0 .and. l(nodes)/=ele_num%degree)) then
                cnt=cnt+1

                ! Do the actual numbering.
                face_local_num_int(cnt)=ele_num%count2number(l(1), l(2), l(3))
             end if

             ! Advance the index:
             l(nodes(2))=l(nodes(2))+1

             if (l(nodes(2))>ele_num%degree-sum(l(nodes(3:)))) then
                l(nodes(2))=0
                l(nodes(3))=l(nodes(3))+1
             end if

             l(nodes(1))=ele_num%degree-sum(l(nodes(2:)))

             ! Check for completion
             if (l(nodes(3))>ele_num%degree) exit number_loop

          end do number_loop

          ! Sanity test.
      !    assert(cnt==size(face_local_num_int))
      case (FAMILY_CUBE)

          ! this first loop works out the count coordinates
          ! of the first given node N, using the following formula:
          !   l(i)=iand( N-1, 2**(3-i) ) / 2**(3-i)
          ! also it finds out in which count coordinate node1 and node2
          ! differ and the increment in this coordinate needed to walk
          ! from node1 to node2, stored in j12 and inc12 resp.
          ! same for node1 and node3, stored in j13 and inc13

          l=0
          k=1 ! bit mask
          j12=0
          j13=0
          do i=ele_num%dimension, 1, -1
             ! compute ith 'count' coordinate
             l(i)=iand(nodes(1)-1, k)/k

             ! increment to go from node 1 to node 2: 0, -1 or +1
             tmp=iand(nodes(2)-1, k)/k-l(i)
             ! remember the coordinate in which node 1 and 2 differ:
             if (tmp /= 0) then
               j12=i
               inc12 = tmp
             end if

             ! increment to go from node 1 to node 3: 0, -1 or +1
             tmp=iand(nodes(3)-1, k)/k-l(i)
             ! remember the coordinate in which node 1 and 3 differ:
             if (tmp/=0) then
               j13=i
               inc13 = tmp
             end if
             k=k*2
          end do

          if (j12==0 .or. j13==0) then
            FLAbort("The same node appears more than once in edge_local_num.")
          end if

          ! Now find the nodes on the face by walking through the count
          ! numbers, starting at node1 and walking from node1 to node2
          ! in the inner loop, and from node1 to node3 in the outer loop

          ! instead of between 0 and 1, between 0 and degree
          l=l*ele_num%degree
          cnt=0
          if (interior) then
             ! leave out facet nodes
             do i=1, ele_num%degree-1
                l(j13)=l(j13)+inc13
                k=l(j12) ! save original value of node1
                do j=1, ele_num%degree-1
                   l(j12)=l(j12)+inc12
                   cnt=cnt+1
                   face_local_num_int(cnt)=ele_num%count2number(l(1), l(2), l(3))
                end do
                l(j12)=k
             end do
          else
             do i=0, ele_num%degree
                k=l(j12) ! save original value of node1
                do j=0, ele_num%degree
                   cnt=cnt+1
                   face_local_num_int(cnt)=ele_num%count2number(l(1), l(2), l(3))
                   l(j12)=l(j12)+inc12
                end do
                l(j12)=k
                l(j13)=l(j13)+inc13
             end do
          end if

      case default

          FLAbort("Unknown element family.")

      end select
    case (ELEMENT_BUBBLE)
      select case (ele_num%family)
      case (FAMILY_SIMPLEX)
          l=0
          l(nodes(1))=ele_num%degree*(ele_num%dimension+1)
          cnt=0

          number_loop_b: do
             ! Skip spurious facet cases.
             if ((.not.interior) .or. all(&
                  l(nodes)/=0 .and. l(nodes)/=(ele_num%degree*(ele_num%dimension+1)))) then
                cnt=cnt+1

                ! Do the actual numbering.
                face_local_num_int(cnt)=ele_num%count2number(l(1), l(2), l(3))
             end if

             ! Advance the index:
             l(nodes(2))=l(nodes(2))+ele_num%dimension+1

             if (l(nodes(2))>ele_num%degree*(ele_num%dimension+1)-sum(l(nodes(3:)))) then
                l(nodes(2))=0
                l(nodes(3))=l(nodes(3))+ele_num%dimension+1
             end if

             l(nodes(1))=ele_num%degree*(ele_num%dimension+1)-sum(l(nodes(2:)))

             ! Check for completion
             if (l(nodes(3))>(ele_num%degree*(ele_num%dimension+1))) exit number_loop_b

          end do number_loop_b

      case default

          FLAbort("Unknown element family.")

      end select
    case default
      FLAbort("Unknown element type.")
    end select

  end function face_local_num_int

  !-------------------------------------------------
  !facet numbering wrapper
  !=================================================

  function facet_local_dofs(vertices, ele_num) result (dofs)
    ! Return the dofs whose basis functions are not uniformly zero on the
    !  facet whose vertices are given.
    integer, dimension(:), intent(in) :: vertices
    type(ele_numbering_type), intent(in) :: ele_num
    integer, dimension((facet_num_length(ele_num, &
         interior = .false.))) :: dofs

    ! Directions to traverse the facet in.
    integer, dimension(ele_num%dimension-1, 4) :: delta
    ! Start point for traversal.
    integer, dimension(4) :: start
    ! Current point in traversal.
    integer, dimension(4) :: current

    ! Dofs associated with the vertices of this element.
    integer, dimension(ele_num%vertices) :: vertex_dofs

    integer :: i, j, k

    if (ele_num%degree==0) then
       ! In the special case of degree 0 elements, we are done.
       dofs=1
       return
    end if

    vertex_dofs=local_vertices(ele_num)

    ! In the special case of dimension 1 elements, the dof must be the
    ! vertex dof provided.
    if (ele_num%dimension==1) then
       dofs=vertex_dofs(vertices(1))
       return
    end if

    ! Start from first vertex provided
    start=ele_num%number2count(vertex_dofs(vertices(1)),:)

    ! Facets have codimension 1
    do i=1, ele_num%dimension-1
       delta(i,:)=ele_num%number2count(vertex_dofs(vertices(i+1)),:)&
            -start
    end do

    current=start
    k=0
    do j=1, ele_num%degree
       do i=1, ele_num%degree
          ! For simplex elements, only count coordinates summing to the
          !  element degree count.
          if (ele_num%family==FAMILY_SIMPLEX.and.sum(current)/=ele_num&
               &%degree) then
             cycle
          end if

          k=k+1
          dofs(k)=ele_num%count2number(current(1), current(2), current(3))
          current=current+delta(1,:)
       end do
       if (ele_num%dimension<2) exit
       current=start+j*delta(2,:)
    end do

    assert(k==size(dofs))

  end function facet_local_dofs


!!$  function facet_local_num_no_int(nodes, ele_num) &
!!$       result (facet_local_num)
!!$    integer, dimension(:), intent(in) :: nodes
!!$    type(ele_numbering_type), intent(in) :: ele_num
!!$    integer, dimension((facet_num_length(ele_num, &
!!$         interior = .false.))) :: facet_local_num
!!$
!!$     select case (ele_num%dimension)
!!$        case(1)
!!$           if (nodes(1)==1) then
!!$              facet_local_num=1
!!$           else if (nodes(1)==2) then
!!$              facet_local_num=ele_num%nodes
!!$           else
!!$              write(0,*) 'Error in facet_local_num_no_int'
!!$              stop
!!$           end if
!!$        case (2)
!!$           facet_local_num = &
!!$                edge_local_num(nodes, ele_num, interior = .false.)
!!$        case (3)
!!$           facet_local_num = &
!!$                face_local_num_no_int(nodes, ele_num)
!!$        case default
!!$           write(0,*) 'Error in facet_local_num_no_int'
!!$           stop
!!$     end select
!!$
!!$  end function facet_local_num_no_int
!!$
!!$  function facet_local_num_int(nodes, ele_num, interior) &
!!$       result (facet_local_num)
!!$    integer, dimension(:), intent(in) :: nodes
!!$    type(ele_numbering_type), intent(in) :: ele_num
!!$    logical, intent(in) :: interior
!!$    integer, dimension((facet_num_length(ele_num,interior))) &
!!$         :: facet_local_num
!!$
!!$     select case (ele_num%dimension)
!!$     case (1)
!!$        if (.not. interior) then
!!$           if (nodes(1)==1) then
!!$              facet_local_num=1
!!$           else if (nodes(1)==2) then
!!$              facet_local_num=ele_num%nodes
!!$           else
!!$              write(0,*) 'Error in facet_local_num_no_int'
!!$              stop
!!$           end if
!!$        end if
!!$        ! If interior then there is no facet.
!!$     case (2)
!!$        facet_local_num = &
!!$             edge_local_num(nodes, ele_num, interior)
!!$     case (3)
!!$        facet_local_num = &
!!$             face_local_num(nodes, ele_num, interior)
!!$     case default
!!$        write(0,*) 'Error in facet_local_num'
!!$        stop
!!$     end select
!!$
!!$  end function facet_local_num_int

  !------------------------------------------------------------------------
  ! Return all local nodes in the order determined by specified local vertices
  !------------------------------------------------------------------------

  function ele_local_num(nodes, ele_num)
    !!< Given the local vertex numbers (1-4 for tets) in a certain order,
    !!< return all local node numbers of ele_num in the corresponding order
    !! nodes are the specified vertices
    integer, dimension(:), intent(in) :: nodes
    type(ele_numbering_type), intent(in) :: ele_num
    integer, dimension(1:ele_num%nodes) :: ele_local_num

    ! count coordinate
    integer, dimension(3) :: cc
    integer :: i, j

    cc=(/ 0, 0, 0 /)

    if (ele_num%family/=FAMILY_SIMPLEX) then
      FLAbort("ele_local_num currently only works for simplices")
    end if

    if (ele_num%type==ELEMENT_TRACE) then
       FLAbort("ele_local_num doesn't know about trace elements yet")
    end if

    do i=1, ele_num%nodes

      do j=1, size(ele_num%number2count,1)
        if (nodes(j)<=3) then
          cc(nodes(j)) = ele_num%number2count(j, i)
        end if
      end do

      if (size(ele_num%number2count,1)<3) cc(3)=0
      ele_local_num(i) = ele_num%count2number(cc(1), cc(2), cc(3))

    end do

  end function ele_local_num

  !------------------------------------------------------------------------
  ! Local coordinate calculations.
  !------------------------------------------------------------------------

  function ele_num_local_coords(n, ele_num) result (coords)
    ! Work out the local coordinates of node n in ele_num.
    integer, intent(in) :: n
    type(ele_numbering_type), intent(in) :: ele_num
    real, dimension(size(ele_num%number2count, 1)) :: coords

    integer, dimension(size(ele_num%number2count, 1)) :: count_coords
    integer :: i
    integer, allocatable, dimension(:) :: boundary2element,&
         &boundary2count_component
    real, allocatable, dimension(:) :: boundary2local_coordinate

    select case(ele_num%type)
    case (ELEMENT_LAGRANGIAN)

       select case (ele_num%family)
       case (FAMILY_SIMPLEX)

          if (ele_num%degree>0) then
             coords=real(ele_num%number2count(:,n))/real(ele_num%degree)
          else
             ! Degree 0 elements have a single node in the centre of the
             ! element.
             coords=1.0/ele_num%vertices
          end if

       case (FAMILY_CUBE)

          if (ele_num%degree>0) then
             coords=real(ele_num%number2count(:,n))&
                  &                                /real(ele_num%degree)
          else
             ! Degree 0 elements have a single node in the centre of the
             ! element.
             coords=0.5
          end if

       case default

          FLAbort('Unknown element family.')

       end select

    case (ELEMENT_BUBBLE)

       select case (ele_num%family)
       case (FAMILY_SIMPLEX)

          if (ele_num%degree>0) then
             coords=real(ele_num%number2count(:,n))/real(ele_num%degree*(ele_num%dimension+1))
          else
             FLAbort('Illegal element degree')
          end if

       case default

          FLAbort('Unknown element family.')

       end select

    case (ELEMENT_NONCONFORMING)

       coords=real(ele_num%number2count(:,n))/2.0

    case (ELEMENT_TRACE)

       select case (ele_num%family)
       case (FAMILY_SIMPLEX)

          count_coords=ele_num%number2count(:,n)

          if (ele_num%degree>0) then
             do i=1,ele_num%dimension+1
                if (i<count_coords(1)) then
                   coords(i)=count_coords(i+1)/real(ele_num%degree)
                else if (i==count_coords(1)) then
                   coords(i)=0.0
                else
                   coords(i)=count_coords(i)/real(ele_num%degree)
                end if
             end do
          else
             ! Degree 0 elements have a single node in the centre of the
             ! face.
             coords=0.5
             coords(n) = 0.0
          end if
       case (FAMILY_CUBE)

          !for trace elements, the first count coordinate is the
          !boundary number.
          !The other coordinates are the count coordinates on the
          !boundary.

          if(ele_num%dimension==2) then
             !special case for quads because the boundary element
             !type is simplex, not cubes

             !the mapping from boundary count coordinates to
             !element count
             !coordinates is done in ascending component order
             coords = 0.
             count_coords=ele_num%number2count(:,n)

             !numbering is
             !       1
             !   3      4
             !  3        2
             !   1      2
             !       4

             !local coordinates are
             !  0,1 -- 1,1
             !   |      |
             !  0,0 -- 1,0

             allocate(boundary2element(4),boundary2count_component(4),&
                  &boundary2local_coordinate(4))
             !boundary2element(i) is the element count coordinate
             !component corresponding to boundary element component i
             boundary2element = (/1,2,2,1/)
             !boundary2count_component is the component of the
             !element local coordinates that is held constant on
             !this boundary
             boundary2count_component = (/2,1,1,2/)
             !boundary2local_coordinate is the value of the
             !local coordinate on that boundary
             boundary2local_coordinate = (/1.,1.,0.,0./)

             i = boundary2element(count_coords(1))
             if(ele_num%degree>0) then
                !count_coords(3) increases with increasing element count
                !coordinate component boundar2element(count_coords(1))
                coords(i)=count_coords(3)/real(ele_num%degree)
             else
                !special case for degree 0 trace space,
                !a single node in the middle of the face
                coords(i)=0.5
             end if

             i = boundary2count_component(count_coords(1))
             coords(i)=&
                  &boundary2local_coordinate(count_coords(1))

             deallocate(boundary2element,boundary2count_component,&
                  &boundary2local_coordinate)
          else
             FLAbort('Haven''t implemented the dimension yet.')
          end if

       case default

          FLAbort('Unknown element family.')

       end select

    case default

       FLAbort('Illegal element type.')

    end select

  end function ele_num_local_coords

end module element_numbering
