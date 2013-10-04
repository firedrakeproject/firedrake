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
module quadrature
  !!< This module implements quadrature of varying degrees for a number of
  !!< elements. Quadrature information is used to numerically evaluate
  !!< integrals over an element.
  use FLDebug
  use reference_counting
  use wandzura_quadrature
  use grundmann_moeller_quadrature
  use vector_tools
  implicit none

  private

  type permutation_type
     !!< A type encoding a series of permutations. This type is only used
     !!< internally in the quadrature module.

     !! Each column of P is a different permuation of points which can be
     !! used in quadrature on a given element shape.
     integer, pointer, dimension(:,:) :: p
  end type permutation_type

  type generator_type
     !!< The generator type is an encoding of a quadrature generator of the
     !!< type used in the encyclopedia of cubature. This type is only used
     !!< internally in the quadrature module.
     integer, dimension(:,:), pointer :: permutation
     real, dimension(:), pointer :: coords
     real :: weight
  end type generator_type

  type quadrature_template
     !!< A data type which defines a quadrature rule. These are only
     !!< directly used inside the quadrature module.

     !! A quadrature is defined by a set of generators.
     type(generator_type), dimension(:), pointer :: generator
     !! Dimension of the space we are in and the degree of accuracy of the
     !! quadrature.
     integer :: dim, degree
     !! Ngi is number of quadrature points. Vertices is number of vertices. These
     !! names are chosen for consistency with the rest of fluidity.
     integer :: ngi, vertices
  end type quadrature_template

  type quadrature_type
     !!< A data type which describes quadrature information. For most
     !!< developers, quadrature can be treated as an opaque data type which
     !!< will only be encountered when creating element_type variables to
     !!< represent shape functions.
     integer :: dim !! Dimension of the elements for which quadrature
     !!< is required.
     integer :: degree !! Degree of accuracy of quadrature.
     integer :: vertices !! Number of vertices of the element.
     integer :: ngi !! Number of quadrature points.
     real, pointer :: weight(:)=>null() !! Quadrature weights.
     real, pointer :: l(:,:)=>null() !! Locations of quadrature points.
     character(len=0) :: name !! Fake name for reference counting.
     !! Reference count to prevent memory leaks.
     type(refcount_type), pointer :: refcount=>null()
     integer :: family
  end type quadrature_type

  type(permutation_type), dimension(11), target, save :: tet_permutations
  type(permutation_type), dimension(6), target, save :: tri_permutations
  ! Cyclic permutations for triangles.
  type(permutation_type), dimension(6), target, save :: tri_cycles
  type(permutation_type), dimension(2), target, save :: interval_permutations
  type(permutation_type), dimension(7), target, save :: hex_permutations
  type(permutation_type), dimension(4), target, save :: quad_permutations
  type(permutation_type), dimension(1), target, save :: point_permutation

  type(quadrature_template), dimension(8), target, save, public :: tet_quads
  type(quadrature_template), dimension(8), target, save, public :: tri_quads
  type(quadrature_template), dimension(8), target, save, public :: interval_quads
  type(quadrature_template), dimension(6), target, save, public :: hex_quads
  type(quadrature_template), dimension(6), target, save, public :: quad_quads
  type(quadrature_template), dimension(1), target, save, public :: point_quad

  character(len=100), save, public :: quadrature_error_message=""

  !! Unsupported vertex count.
  integer, parameter, public :: QUADRATURE_VERTEX_ERROR=1
  !! Quadrature degree requested is not available.
  integer, parameter, public :: QUADRATURE_DEGREE_ERROR=2
  !! Elements with this number of dimensions are not available.
  integer, parameter, public :: QUADRATURE_DIMENSION_ERROR=3
  !! Unsupported number of quadrature points.
  integer, parameter, public :: QUADRATURE_NGI_ERROR=4
  !! Not enough arguments specified.
  integer, parameter, public :: QUADRATURE_ARGUMENT_ERROR=5

  logical, save, private :: initialised=.false.

  integer, parameter :: FAMILY_COOLS=0, FAMILY_WANDZURA=1, FAMILY_GM=2

  interface allocate
     module procedure allocate_quad
  end interface

  interface deallocate
     module procedure deallocate_quad
  end interface

  interface operator (==)
     module procedure quad_equal
  end interface

#include "Reference_count_interface_quadrature_type.F90"

  public make_quadrature, allocate, deallocate, quadrature_type,&
       & quadrature_template, construct_quadrature_templates, &
       & operator(==), incref, addref, decref, &
       & has_references, FAMILY_COOLS, FAMILY_WANDZURA, FAMILY_GM

contains

  !------------------------------------------------------------------------
  ! Procedures for creating and destroying quadrature data types.
  !------------------------------------------------------------------------

  function make_generator(permutation, weight, coords) result (generator)
    !!< Function hiding the fact that generators are dynamically sized.
    type(generator_type) :: generator
    type(permutation_type), intent(in) :: permutation
    real, intent(in) :: weight
    real, dimension(:), intent(in) :: coords

    generator%permutation=>permutation%p
    generator%weight=weight
    allocate(generator%coords(size(coords)))
    generator%coords=coords

  end function make_generator

  function make_quadrature(vertices, dim, degree, ngi, family, stat) result (quad)
    !!< Given information about a quadrature, return a quad type encoding
    !!< that quadrature.
    type(quadrature_type) :: quad
    !! Using vertices and dimension it is possible to determine what shape we are
    !! using. At this stage we assume that no-one will require elements in
    !! the shape of the tetragonal antiwedge!
    integer, intent(in) :: vertices, dim
    !! Ngi is the old way of specifying quadrature. This should really be
    !! done via degree. At least one of these must be specified. If both are
    !! specified then ngi is used.
    integer, intent(in), optional :: degree, ngi
    !! Which family of quadrature you'd like to use.
    integer, intent(in), optional :: family
    !! Status argument - zero for success non-zero otherwise.
    integer, intent(out), optional :: stat

    ! The set of quadrature templates for this shape of element.
    type(quadrature_template), dimension(:), pointer :: template_set
    ! The quadrature template we will use.
    type(quadrature_template), pointer :: template
    ! Number of local coordinates
    integer coords

    integer :: lfamily
    integer :: wandzura_rule_idx, wandzura_rule_degree, max_wandzura_rule, wandzura_order
    real, dimension(2, 3) :: wandzura_ref_tri
    real, dimension(3, 3) :: wandzura_ref_map
    real, dimension(:, :), allocatable :: tmp_coordinates
    integer :: gi

    integer :: gm_rule, gm_order, vertex
    real, dimension(:, :), allocatable :: gm_ref_simplex
    real, dimension(:, :), allocatable :: gm_ref_map

    ! Idempotent initialisation
    call construct_quadrature_templates

    if (present(stat)) stat=0

    if (present(family)) then
      lfamily = family
    else
      lfamily = FAMILY_COOLS
    end if

    if (lfamily == FAMILY_COOLS) then
      ! First isolate the set of templates applicable to this shape.
      select case(dim)
      case(0)
         select case(vertices)
            case(1)
               ! All zero dimensional elements are points.
               template_set=>point_quad
               coords=1
            case default
               FLAbort('Invalid quadrature')
            end select

      case(1)
         select case(vertices)
            case(2)
               ! All one dimensional elements are intervals.
               template_set=>interval_quads
               coords=2

            case default
               FLAbort('Invalid quadrature')
            end select

      case(2)
         select case(vertices)
         case(3) ! Triangles

            template_set=>tri_quads
            coords=3

         case(4) ! Quads

            template_set=>quad_quads
            coords=2

         case default
            ! Sanity test
            write (quadrature_error_message, '(a,i0,a)') &
                 "make_quadrature: ",vertices, " is an unsupported vertex count."

            if (present(stat)) then
               stat=QUADRATURE_VERTEX_ERROR
               return
            else
               FLAbort(quadrature_error_message)
            end if

         end select

      case(3)

         select case(vertices)
         case(4) ! Tets.

            template_set=>tet_quads
            coords=4

         case(8) ! Hexahedra

            template_set=>hex_quads
            coords=3

         case default
            ! Sanity test
            write (quadrature_error_message, '(a,i0,a)') &
                 "make_quadrature: ",vertices, " is an unsupported vertex count."

            if (present(stat)) then
               stat=QUADRATURE_VERTEX_ERROR
               return
            else
               FLAbort(quadrature_error_message)
            end if

         end select

      case default
         ! Sanity test
         write (quadrature_error_message, '(a,i0,a)') &
              "make_quadrature: ",dim, " is not a supported dimension."

         if (present(stat)) then
             stat=QUADRATURE_DIMENSION_ERROR
             return
         else
             FLAbort(quadrature_error_message)
         end if

      end select

      ! Now locate the appropriate template for this degree or number of
      ! quadrature points.`
      if (present(degree)) then
         ! Attempt to find a quadrature of at least required degree.
         if (all(template_set%degree<degree)) then
            write (quadrature_error_message, '(a,i0,a)') &
                 "make_quadrature: No quadrature of degree at least ", &
                 degree
            if (present(stat)) then
               stat=QUADRATURE_DEGREE_ERROR
               return
            else
               FLExit(quadrature_error_message)
            end if

         end if

         template=>template_set(minloc(template_set%degree, dim=1,&
              mask=template_set%degree>=degree))

      else if (present(ngi)) then
         ! Attempt to find a quadrature with the specified number of points.
         if (any(template_set%ngi==ngi)) then

            template=>template_set(minloc(template_set%ngi, dim=1,&
                 mask=template_set%ngi==ngi))

         else
            write (quadrature_error_message, '(a,i0,a)') &
                 "make_quadrature: No quadrature with ",ngi," points."
            if (present(stat)) then
               stat=QUADRATURE_NGI_ERROR
               return
            else
               FLExit(quadrature_error_message)
            end if
         end if

      else
         write (quadrature_error_message, '(a,i0,a)') &
              "make_quadrature: You must specify either degree or ngi."
         if (present(stat)) then
            stat=QUADRATURE_ARGUMENT_ERROR
            return
         else
            FLAbort(quadrature_error_message)
         end if

      end if

#if defined (DDEBUG)
      if (present(degree).and.(dim/=0)) then
         if (template%degree/=degree) then
            ewrite(0,*) "Warning:make_quadrature: degree ",degree, " requested&
                 & but ", template%degree, "available."
         end if
      end if
#endif

      ! Now we can start putting together the quad.
      call allocate(quad, vertices, template%ngi, coords)
      quad%degree=template%degree
      quad%dim=dim

      call expand_quadrature_template(quad, template)
    else if (lfamily == FAMILY_WANDZURA) then

      ! Make sure we're on triangles.
      if (dim /= 2 .or. vertices /= 3) then
        write (quadrature_error_message, '(a,i0,a)') &
          "make_quadrature: You can only specify Wandzura quadrature for triangles."
        if (present(stat)) then
          stat=QUADRATURE_ARGUMENT_ERROR
          return
        else
          FLExit(quadrature_error_message)
        end if
      end if

      ! OK. First let's figure out which rule we want to use.
      if (.not. present(degree)) then
        write (quadrature_error_message, '(a,i0,a)') &
          "make_quadrature: You can only specify degree if you want Wandzura quadrature."
        if (present(stat)) then
          stat=QUADRATURE_ARGUMENT_ERROR
          return
        else
          FLExit(quadrature_error_message)
        end if
      end if

      call wandzura_rule_num(max_wandzura_rule)
      do wandzura_rule_idx=1,max_wandzura_rule
        call wandzura_degree(wandzura_rule_idx, wandzura_rule_degree)
        if (wandzura_rule_degree >= degree) then
          exit
        end if
      end do

      if (wandzura_rule_degree < degree) then
        write (quadrature_error_message, '(a,i0,a)') &
          "make_quadrature: We can only supply degree ", wandzura_rule_degree, "with Wandzura quadrature. Sorry."
        if (present(stat)) then
          stat=QUADRATURE_DEGREE_ERROR
          return
        else
          FLExit(quadrature_error_message)
        end if
      end if

      ! OK. So now we know which Wandzura rule to use. Let's make it happen ..
      call wandzura_order_num(wandzura_rule_idx, wandzura_order)
      call allocate(quad, vertices, wandzura_order, coords=3)
      allocate(tmp_coordinates(2, wandzura_order))
      quad%degree = wandzura_rule_degree
      quad%dim = 2
      call wandzura_rule(wandzura_rule_idx, wandzura_order, tmp_coordinates, quad%weight)
      wandzura_ref_tri(:, 1) = (/0, 0/)
      wandzura_ref_tri(:, 2) = (/1, 0/)
      wandzura_ref_tri(:, 3) = (/0, 1/)
      call local_coords_matrix_positions(wandzura_ref_tri, wandzura_ref_map)
      do gi=1,wandzura_order
        quad%l(gi, 1:2) = tmp_coordinates(:, gi); quad%l(gi, 3) = 1.0
        quad%l(gi, :) = matmul(wandzura_ref_map, quad%l(gi, :))
      end do
    elseif (lfamily == FAMILY_GM) then
      ! Make sure we're on triangles.
      if (vertices /= dim+1) then
        write (quadrature_error_message, '(a,i0,a)') &
          "make_quadrature: You can only specify Grundmann-Moeller quadrature for simplices."
        if (present(stat)) then
          stat=QUADRATURE_ARGUMENT_ERROR
          return
        else
          FLExit(quadrature_error_message)
        end if
      end if

      ! OK. First let's figure out which rule we want to use.
      if (.not. present(degree)) then
        write (quadrature_error_message, '(a,i0,a)') &
          "make_quadrature: You can only specify degree if you want Grundmann-Moeller quadrature."
        if (present(stat)) then
          stat=QUADRATURE_ARGUMENT_ERROR
          return
        else
          FLExit(quadrature_error_message)
        end if
      end if

      if (degree >= 30) then
        write (quadrature_error_message, '(a,i0,a)') &
          "Grundmann-Moeller quadrature is only accurate up to about degree 30."
        if (present(stat)) then
          stat=QUADRATURE_DEGREE_ERROR
          return
        else
          FLExit(quadrature_error_message)
        end if
      end if

      if (modulo(degree, 2) == 0) then
        gm_rule = degree / 2
      else
        gm_rule = (degree-1)/2
      end if

      call gm_rule_size(gm_rule, dim, gm_order)
      call allocate(quad, vertices, gm_order, coords=vertices)
      allocate(tmp_coordinates(dim, gm_order))
      quad%degree = 2*gm_rule + 1
      quad%dim = dim

      call gm_rule_set(gm_rule, dim, gm_order, quad%weight, tmp_coordinates)

      allocate(gm_ref_simplex(dim, vertices))
      gm_ref_simplex(:, 1) = 0.0
      do vertex=1,dim
        gm_ref_simplex(:, vertex+1) = 0.0
        gm_ref_simplex(vertex, vertex+1) = 1.0
      end do
      allocate(gm_ref_map(vertices, vertices))

      call local_coords_matrix_positions(gm_ref_simplex, gm_ref_map)
      do gi=1,gm_order
        quad%l(gi, 1:dim) = tmp_coordinates(:, gi); quad%l(gi, dim+1) = 1.0
        quad%l(gi, :) = matmul(gm_ref_map, quad%l(gi, :))
      end do
      quad%weight = quad%weight / factorial(dim)
    else
      if (present(stat)) then
        stat=QUADRATURE_ARGUMENT_ERROR
      else
        FLAbort("Unknown family of quadrature")
      end if
    end if

    quad%family = lfamily

  end function make_quadrature

  subroutine allocate_quad(quad, vertices, ngi, coords, stat)
    !!< Allocate memory for a quadrature type. Note that this is done
    !!< automatically in make_quadrature.
    type(quadrature_type), intent(inout) :: quad
    !! Vertices is the number of vertices. Ngi is the number of quadrature
    !! points. Coords the number of local coords
    integer, intent(in) :: vertices, ngi, coords
    !! Stat returns zero for successful completion and nonzero otherwise.
    integer, intent(out), optional :: stat

    integer :: lstat

    allocate(quad%weight(ngi), quad%l(ngi,coords), stat=lstat)

    quad%vertices=vertices
    quad%ngi=ngi

    nullify(quad%refcount) ! Hack for gfortran component initialisation
    !                         bug.

    call addref(quad)

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Error allocating quad")
    end if

  end subroutine allocate_quad

  subroutine deallocate_quad(quad,stat)
    !!< Since quadrature types contain pointers it is necessary to
    !!< explicitly deallocate them.
    !! The quadrature type to be deallocated.
    type(quadrature_type), intent(inout) :: quad
    !! Stat returns zero for successful completion and nonzero otherwise.
    integer, intent(out), optional :: stat

    integer :: lstat

    call decref(quad)
    if (has_references(quad)) then
       ! There are still references to this quad so we don't deallocate.
       return
    end if

    deallocate(quad%weight,quad%l, stat=lstat)

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Error deallocating quad")
    end if

  end subroutine deallocate_quad

#include "Reference_count_quadrature_type.F90"

  pure function quad_equal(quad1,quad2)
    !!< Return true if the two quadratures are equivalent.
    logical :: quad_equal
    type(quadrature_type), intent(in) :: quad1, quad2

    quad_equal = quad1%dim==quad2%dim &
         .and. quad1%degree==quad2%degree &
         .and. quad1%vertices==quad2%vertices &
         .and. quad1%ngi==quad2%ngi

  end function quad_equal

  subroutine expand_quadrature_template(quad, template)
    ! Expand the given template into the quad provided.
    type(quadrature_type), intent(inout) :: quad
    type(quadrature_template), intent(in) :: template

    integer :: i, j, k, dk
    type(generator_type), pointer :: lgen

    quad%l=0.0
    dk=0

    do i=1,size(template%generator)
       lgen=>template%generator(i)

       ! Permute coordinates and insert into quad%l
       ! Note that for external compatibility, quad%l is transposed.
       forall(j=1:size(lgen%permutation,1), &
            k=1:size(lgen%permutation,2), &
            lgen%permutation(j,k)/=0)
          ! The permutation stores both the permutation order and (for
          ! quads and hexs) the sign of the coordinate.
          quad%l(k+dk,j)=sign(lgen%coords(abs(lgen%permutation(j,k))),&
               &                            real(lgen%permutation(j,k)))
       end forall

       ! Insert weights:
       quad%weight(dk+1:dk+size(lgen%permutation,2))=lgen%weight

       ! Move down k:
       dk=dk+size(lgen%permutation,2)

    end do

    if ((quad%vertices==4.and.quad%dim==2).or.(quad%vertices==8.and.quad&
         &%dim==3)) then
       ! The quadrature rules for quad and hex elements in the encyclopedia
       ! are written for local coordinates in the interval [-1,1], however
       ! we wish to use local coordinates in the interval [0,1]. This
       ! requires us to change coordinates but also to scale the weights.
       quad%l=0.5*(quad%l+1)
       quad%weight=quad%weight/2**quad%dim
    end if

  end subroutine expand_quadrature_template

  !------------------------------------------------------------------------
  ! Procedures for generating permutations and quadratures.
  !------------------------------------------------------------------------

  subroutine construct_quadrature_templates
    !!< Construct the generators of symmetric rules on the tet.
    !!< The order of generators follows that on the
    !!< Encyclopaedia of Cubature Formulas at:
    !!< http://www.cs.kuleuven.ac.be/~nines/research/ecf/ecf.html

    !Idempotency test.
    if (initialised) return
    initialised=.true.

    call construct_point_permutation

    call construct_point_quadrature

    call construct_interval_permutations

    call construct_interval_quadratures

    call construct_tri_permutations

    call construct_tri_cycles

    call construct_tri_quadratures

    call construct_tet_permutations

    call construct_tet_quadratures

    call construct_hex_permutations

    call construct_hex_quadratures

    call construct_quad_permutations

    call construct_quad_quadratures

  end subroutine construct_quadrature_templates

  subroutine construct_tet_quadratures
    ! Construct list of available quadratures.
    ! The references cited are listed on the Encyclopedia of Cubature
    ! Formulas.
    integer :: i
    real, dimension(4) :: coords

    tet_quads%dim=3
    tet_quads%vertices=4

    i=0

    !----------------------------------------------------------------------
    ! 1 point degree 1 quadrature.
    ! Citation: Str71
    i=i+1
    ! Only one generator.
    allocate(tet_quads(i)%generator(1))
    tet_quads(i)%ngi=1
    tet_quads(i)%degree=1

    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(7), &
         weight=0.166666666666666666666666666666666, &
         coords=(/0.25/))

    !----------------------------------------------------------------------
    ! 4 point degree 2 quadrature.
    ! Citation: str71
    i=i+1
    allocate(tet_quads(i)%generator(1))
    tet_quads(i)%ngi=4
    tet_quads(i)%degree=2

    coords(1)=0.138196601125010515179541316563436
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(8), &
         weight=0.041666666666666666666666666666666, &
         coords=coords(1:2))

    !----------------------------------------------------------------------
    ! 5 point degree 3 quadrature.
    ! Citation: str71
    i=i+1
    allocate(tet_quads(i)%generator(2))
    tet_quads(i)%ngi=5
    tet_quads(i)%degree=3

    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(7), &
         weight=-0.133333333333333333333333333333333, &
         coords=(/0.25/))

    coords(1)=0.166666666666666666666666666666666
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(2)=make_generator( &
         permutation=tet_permutations(8), &
         weight=0.075, &
         coords=coords(1:2))

    !----------------------------------------------------------------------
    ! 11 point degree 4 quadrature.
    ! Citation: kea86
    i=i+1
    allocate(tet_quads(i)%generator(3))
    tet_quads(i)%ngi=11
    tet_quads(i)%degree=4

    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(7), &
         weight=-0.013155555555555555555555555555555, &
         coords=(/0.25/))

    coords(1)=0.0714285714285714285714285714285714
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(2)=make_generator( &
         permutation=tet_permutations(8), &
         weight=7.62222222222222222222222222222222E-3, &
         coords=coords(1:2))

    coords(1)=0.399403576166799204996102147461640
    coords(2)=0.5-coords(1)
    tet_quads(i)%generator(3)=make_generator( &
         permutation=tet_permutations(9), &
         weight=0.0248888888888888888888888888888888, &
         coords=coords(1:2))

    !----------------------------------------------------------------------
    ! 14 point degree 5 quadrature.
    ! Citation gm78
    i=i+1
    allocate(tet_quads(i)%generator(3))
    tet_quads(i)%ngi=14
    tet_quads(i)%degree=5

    coords(1)=0.0927352503108912264023239137370306
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(8), &
         weight=0.0122488405193936582572850342477212, &
         coords=coords(1:2))

    coords(1)=0.310885919263300609797345733763457
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(2)=make_generator( &
         permutation=tet_permutations(8), &
         weight=0.0187813209530026417998642753888810, &
         coords=coords(1:2))

    coords(1)=0.454496295874350350508119473720660
    coords(2)=0.5-coords(1)
    tet_quads(i)%generator(3)=make_generator( &
         permutation=tet_permutations(9), &
         weight=7.09100346284691107301157135337624E-3, &
         coords=coords(1:2))

    !----------------------------------------------------------------------
    ! 24 point degree 6 quadrature.
    ! Citation kea86
    i=i+1
    allocate(tet_quads(i)%generator(4))
    tet_quads(i)%ngi=24
    tet_quads(i)%degree=6

    coords(1)=0.214602871259152029288839219386284
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(8), &
         weight=6.65379170969458201661510459291332E-3, &
         coords=coords(1:2))

    coords(1)=0.0406739585346113531155794489564100
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(2)=make_generator( &
         permutation=tet_permutations(8), &
         weight=1.67953517588677382466887290765614E-3, &
         coords=coords(1:2))

    coords(1)=0.322337890142275510343994470762492
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(3)=make_generator( &
         permutation=tet_permutations(8), &
         weight=9.22619692394245368252554630895433E-3, &
         coords=coords(1:2))

    coords(1)=0.0636610018750175252992355276057269
    coords(2)=0.269672331458315808034097805727606
    coords(3)=1.0-2.0*coords(1)-coords(2)
    tet_quads(i)%generator(4)=make_generator( &
         permutation=tet_permutations(10), &
         weight=8.03571428571428571428571428571428E-3, &
         coords=coords(1:3))

    !----------------------------------------------------------------------
    ! 31 point degree 7 quadrature.
    ! Citation kea86
    i=i+1
    allocate(tet_quads(i)%generator(6))
    tet_quads(i)%ngi=31
    tet_quads(i)%degree=7

    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(2), &
         weight=9.70017636684303350970017636684303E-4, &
         coords=(/0.5/))

    tet_quads(i)%generator(2)=make_generator( &
         permutation=tet_permutations(7), &
         weight=0.0182642234661088202912015685649462, &
         coords=(/0.25/))

    coords(1)=0.0782131923303180643739942508375545
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(3)=make_generator( &
         permutation=tet_permutations(8), &
         weight=0.0105999415244136869164138748545257, &
         coords=coords(1:2))

    coords(1)=0.121843216663905174652156372684818
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(4)=make_generator( &
         permutation=tet_permutations(8), &
         weight=-0.0625177401143318516914703474927900, &
         coords=coords(1:2))

    coords(1)=0.332539164446420624152923823157707
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(5)=make_generator( &
         permutation=tet_permutations(8), &
         weight=4.89142526307349938479576303671027E-3, &
         coords=coords(1:2))

    coords(1)=0.1
    coords(2)=0.2
    coords(3)=1.0-2.0*coords(1)-coords(2)
    tet_quads(i)%generator(6)=make_generator( &
         permutation=tet_permutations(10), &
         weight=0.0275573192239858906525573192239858, &
         coords=coords(1:3))

    !----------------------------------------------------------------------
    ! 43 point degree 8 quadrature.
    ! Citation bh90 bec92
    i=i+1
    allocate(tet_quads(i)%generator(7))
    tet_quads(i)%ngi=43
    tet_quads(i)%degree=8

    tet_quads(i)%generator(1)=make_generator( &
         permutation=tet_permutations(7), &
         weight=-0.0205001886586399158405865177642941, &
         coords=(/0.25/))

    coords(1)=0.206829931610673204083980900024961
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(2)=make_generator( &
         permutation=tet_permutations(8), &
         weight=0.0142503058228669012484397415358704, &
         coords=coords(1:2))

    coords(1)=0.0821035883105467230906058078714215
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(3)=make_generator( &
         permutation=tet_permutations(8), &
         weight=1.96703331313390098756280342445466E-3, &
         coords=coords(1:2))

    coords(1)=5.78195050519799725317663886414270E-3
    coords(2)=1.0-3.0*coords(1)
    tet_quads(i)%generator(4)=make_generator( &
         permutation=tet_permutations(8), &
         weight=1.69834109092887379837744566704016E-4, &
         coords=coords(1:2))

    coords(1)=0.0505327400188942244256245285579071
    coords(2)=0.5-coords(1)
    tet_quads(i)%generator(5)=make_generator( &
         permutation=tet_permutations(9), &
         weight=4.57968382446728180074351446297276E-3, &
         coords=coords(1:2))

    coords(1)=0.229066536116811139600408854554753
    coords(2)=0.0356395827885340437169173969506114
    coords(3)=1.0-2.0*coords(1)-coords(2)
    tet_quads(i)%generator(6)=make_generator( &
         permutation=tet_permutations(10), &
         weight=5.70448580868191850680255862783040E-3, &
         coords=coords(1:3))

    coords(1)=0.0366077495531974236787738546327104
    coords(2)=0.190486041934633455699433285315099
    coords(3)=1.0-2.0*coords(1)-coords(2)
    tet_quads(i)%generator(7)=make_generator( &
         permutation=tet_permutations(10), &
         weight=2.14051914116209259648335300092023E-3, &
         coords=coords(1:3))

  end subroutine construct_tet_quadratures

  subroutine construct_tri_quadratures
    ! Construct list of available quadratures.
    ! The references cited are listed on the Encyclopedia of Cubature
    ! Formulas.
    integer :: i
    real, dimension(3) :: coords

    tri_quads%dim=2
    tri_quads%vertices=3

    i=0

    !----------------------------------------------------------------------
    ! 1 point degree 1 quadrature.
    ! Citation: Str71
    i=i+1
    ! Only one generator.
    allocate(tri_quads(i)%generator(1))
    tri_quads(i)%ngi=1
    tri_quads(i)%degree=1

    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(4), &
         weight=0.5, &
         coords=(/0.333333333333333333333333333333333/))

    !----------------------------------------------------------------------
    ! 3 point degree 2 quadrature.
    ! Citation: Str71
    i=i+1
    ! Only one generator.
    allocate(tri_quads(i)%generator(1))
    tri_quads(i)%ngi=3
    tri_quads(i)%degree=2

    coords(1)=0.166666666666666666666666666666666
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.166666666666666666666666666666666, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 4 point degree 3 quadrature.
    ! Citation: Str71
    i=i+1

    allocate(tri_quads(i)%generator(2))
    tri_quads(i)%ngi=4
    tri_quads(i)%degree=3

    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(4), &
         weight=-0.28125, &
         coords=(/0.333333333333333333333333333333333/))

    coords(1)=0.2
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(2)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.260416666666666666666666666666666, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 6 point degree 4 quadrature.
    ! Citation: cow73 dun85 blg78 lj75 moa74 sf73
    i=i+1

    allocate(tri_quads(i)%generator(2))
    tri_quads(i)%ngi=6
    tri_quads(i)%degree=4

    coords(1)=0.0915762135097707434595714634022015
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0549758718276609338191631624501052, &
         coords=coords)

    coords(1)=0.445948490915964886318329253883051
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(2)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.111690794839005732847503504216561, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 7 point degree 5 quadrature.
    ! Citation: str71
    i=i+1

    allocate(tri_quads(i)%generator(3))
    tri_quads(i)%ngi=7
    tri_quads(i)%degree=5

    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(4), &
         weight=0.1125, &
         coords=(/0.333333333333333333333333333333333/))

    coords(1)=0.101286507323456338800987361915123
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(2)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0629695902724135762978419727500906, &
         coords=coords)

    coords(1)=0.470142064105115089770441209513447
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(3)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0661970763942530903688246939165759, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 12 point degree 6 quadrature.
    ! Citation: cow73 dun85 blg78 lj75 moa74 sf73
    i=i+1

    allocate(tri_quads(i)%generator(3))
    tri_quads(i)%ngi=12
    tri_quads(i)%degree=6

    coords(1)=0.0630890144915022283403316028708191
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0254224531851034084604684045534344, &
         coords=coords)

    coords(1)=0.249286745170910421291638553107019
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(2)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0583931378631896830126448056927897, &
         coords=coords)

    coords(1)=0.0531450498448169473532496716313981
    coords(2)=0.310352451033784405416607733956552
    coords(3)=1.0-coords(1)-coords(2)
    tri_quads(i)%generator(3)=make_generator( &
         permutation=tri_permutations(6), &
         weight=0.0414255378091867875967767282102212, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 12 point degree 7 quadrature.
    ! Citation: gat88
    i=i+1

    allocate(tri_quads(i)%generator(4))
    tri_quads(i)%ngi=12
    tri_quads(i)%degree=7

    coords(1)=0.0623822650944021181736830009963499
    coords(2)=0.0675178670739160854425571310508685
    coords(3)=1.0-coords(1)-coords(2)
    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_cycles(6), &
         weight=0.0265170281574362514287541804607391, &
         coords=coords)

    coords(1)=0.0552254566569266117374791902756449
    coords(2)=0.321502493851981822666307849199202
    coords(3)=1.0-coords(1)-coords(2)
    tri_quads(i)%generator(2)=make_generator( &
         permutation=tri_cycles(6), &
         weight=0.0438814087144460550367699031392875, &
         coords=coords)

    coords(1)=0.0343243029450971464696306424839376
    coords(2)=0.660949196186735657611980310197799
    coords(3)=1.0-coords(1)-coords(2)
    tri_quads(i)%generator(3)=make_generator( &
         permutation=tri_cycles(6), &
         weight=0.0287750427849815857384454969002185, &
         coords=coords)

    coords(1)=0.515842334353591779257463386826430
    coords(2)=0.277716166976391782569581871393723
    coords(3)=1.0-coords(1)-coords(2)
    tri_quads(i)%generator(4)=make_generator( &
         permutation=tri_cycles(6), &
         weight=0.0674931870098027744626970861664214, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 16 point degree 8 quadrature.
    ! Citation: lj75, dun85b, lg78
    i=i+1

    allocate(tri_quads(i)%generator(5))
    tri_quads(i)%ngi=16
    tri_quads(i)%degree=8

    tri_quads(i)%generator(1)=make_generator( &
         permutation=tri_permutations(4), &
         weight=0.0721578038388935841255455552445323, &
         coords=(/1.0/3.0/))

    coords(1)=0.170569307751760206622293501491464
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(2)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0516086852673591251408957751460645, &
         coords=coords(1:2))

    coords(1)=0.0505472283170309754584235505965989
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(3)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0162292488115990401554629641708902, &
         coords=coords(1:2))

    coords(1)=0.459292588292723156028815514494169
    coords(2)=1.0-2.0*coords(1)
    tri_quads(i)%generator(4)=make_generator( &
         permutation=tri_permutations(5), &
         weight=0.0475458171336423123969480521942921, &
         coords=coords(1:2))

    coords(1)=0.728492392955404281241000379176061
    coords(2)=0.263112829634638113421785786284643
    coords(3)=1.0-coords(1)-coords(2)
    tri_quads(i)%generator(5)=make_generator( &
         permutation=tri_permutations(6), &
         weight=0.0136151570872174971324223450369544, &
         coords=coords)

  end subroutine construct_tri_quadratures

  subroutine construct_interval_quadratures
    ! Construct list of available quadratures.
    ! Interval quadratures are based on a matlab script by Greg von Winkel
    integer :: i
    real, dimension(2) :: coords

    interval_quads%dim=1
    interval_quads%vertices=2

    i=0

    !----------------------------------------------------------------------
    ! 1 point degree 1 quadrature.
    i=i+1
    ! Only one generator.
    allocate(interval_quads(i)%generator(1))
    interval_quads(i)%ngi=1
    interval_quads(i)%degree=1

    coords(1)=0.5
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(1), &
         weight=1.0, &
         coords=coords)


    !----------------------------------------------------------------------
    ! 2 point degree 2 quadrature.
    i=i+1
    ! Only one generator.
    allocate(interval_quads(i)%generator(1))
    interval_quads(i)%ngi=2
    interval_quads(i)%degree=2

    coords(1)=0.788675134594813
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.5, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 3 point degree 3 quadrature.
    i=i+1

    allocate(interval_quads(i)%generator(2))
    interval_quads(i)%ngi=3
    interval_quads(i)%degree=3

    coords(1)=0.887298334620742
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.277777777777777, &
         coords=coords)

    coords(1)=0.5
    interval_quads(i)%generator(2)=make_generator( &
         permutation=interval_permutations(1), &
         weight=0.444444444444444, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 4 point degree 4 quadrature.
    i=i+1

    allocate(interval_quads(i)%generator(2))
    interval_quads(i)%ngi=4
    interval_quads(i)%degree=4

    coords(1)=0.9305681557970262
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.173927422568727, &
         coords=coords)

    coords(1)=0.6699905217924281
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(2)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.326072577431273, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 5 point degree 5 quadrature.
    i=i+1

    allocate(interval_quads(i)%generator(3))
    interval_quads(i)%ngi=5
    interval_quads(i)%degree=5

    coords(1)=0.9530899229693319
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.118463442528095, &
         coords=coords)

    coords(1)=0.7692346550528415
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(2)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.239314335249683, &
         coords=coords)

    coords(1)=0.5
    interval_quads(i)%generator(3)=make_generator( &
         permutation=interval_permutations(1), &
         weight=0.284444444444444, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 6 point degree 6 quadrature.
    i=i+1

    allocate(interval_quads(i)%generator(3))
    interval_quads(i)%ngi=6
    interval_quads(i)%degree=6

    coords(1)=0.9662347571015760
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.0856622461895852, &
         coords=coords)

    coords(1)=0.8306046932331322
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(2)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.1803807865240693, &
         coords=coords)

    coords(1)=0.6193095930415985
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(3)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.2339569672863455, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 7 point degree 7 quadrature.
    i=i+1

    allocate(interval_quads(i)%generator(4))
    interval_quads(i)%ngi=7
    interval_quads(i)%degree=7

    coords(1)=0.9745539561713792
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.0647424830844348, &
         coords=coords)

    coords(1)=0.8707655927996972
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(2)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.1398526957446384, &
         coords=coords)

    coords(1)=0.7029225756886985
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(3)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.1909150252525595, &
         coords=coords)

    coords(1)=0.5
    interval_quads(i)%generator(4)=make_generator( &
         permutation=interval_permutations(1), &
         weight=0.2089795918367347, &
         coords=coords)

    !----------------------------------------------------------------------
    ! 8 point degree 8 quadrature.
    i=i+1

    allocate(interval_quads(i)%generator(4))
    interval_quads(i)%ngi=8
    interval_quads(i)%degree=8

    coords(1)=0.9801449282487682
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(1)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.0506142681451884, &
         coords=coords)

    coords(1)=0.8983332387068134
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(2)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.1111905172266872, &
         coords=coords)

    coords(1)=0.7627662049581645
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(3)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.1568533229389437, &
         coords=coords)

    coords(1)=0.5917173212478249
    coords(2)=1.0-coords(1)
    interval_quads(i)%generator(4)=make_generator( &
         permutation=interval_permutations(2), &
         weight=0.1813418916891811, &
         coords=coords)

  end subroutine construct_interval_quadratures

  subroutine construct_point_quadrature
    !!< Construct the quadrature of the point according to a top secret
    !!< formula!

    point_quad%dim=0
    point_quad%vertices=1
    point_quad%ngi=1
    point_quad%degree=666
    allocate(point_quad(1)%generator(1))
    point_quad(1)%generator(1)=make_generator( &
         permutation=point_permutation(1), &
         weight=1.0, &
         coords=(/1.0/))


  end subroutine construct_point_quadrature

  subroutine construct_hex_quadratures
    ! Construct list of available quadratures.
    ! The references cited are listed on the Encyclopedia of Cubature
    ! Formulas.
    integer :: i

    hex_quads%dim=3
    hex_quads%vertices=8

    i=0

    !----------------------------------------------------------------------
    ! 1 point degree 1 quadrature.
    ! Citation: Str71
    i=i+1
    ! Only one generator.
    allocate(hex_quads(i)%generator(1))
    hex_quads(i)%ngi=1
    hex_quads(i)%degree=1

    hex_quads(i)%generator(1)=make_generator( &
         permutation=hex_permutations(1), &
         weight=8.0, &
         coords=(/0.0/))

    !----------------------------------------------------------------------
    ! 6 point degree 3 quadrature.
    ! Citation: Str71
    i=i+1
    allocate(hex_quads(i)%generator(1))
    hex_quads(i)%ngi=6
    hex_quads(i)%degree=3

    hex_quads(i)%generator(1)=make_generator( &
         permutation=hex_permutations(2), &
         weight=1.33333333333333333333333333333333, &
         coords=(/1.0/))

    !----------------------------------------------------------------------
    ! 8 point degree 3 quadrature.
    ! Citation: Gauss
    i=i+1
    allocate(hex_quads(i)%generator(1))
    hex_quads(i)%ngi=8
    hex_quads(i)%degree=3

    hex_quads(i)%generator(1)=make_generator( &
         permutation=hex_permutations(5), &
         weight=1.0, &
         coords=(/sqrt(3.0)/3.0/))

    !----------------------------------------------------------------------
    ! 14 point degree 5 quadrature.
    ! Citation: Str71
    i=i+1
    allocate(hex_quads(i)%generator(2))
    hex_quads(i)%ngi=14
    hex_quads(i)%degree=5

    hex_quads(i)%generator(1)=make_generator( &
         permutation=hex_permutations(2), &
         weight=0.886426592797783933518005540166204, &
         coords=(/0.795822425754221463264548820476135/))

    hex_quads(i)%generator(2)=make_generator( &
         permutation=hex_permutations(5), &
         weight=0.335180055401662049861495844875346, &
         coords=(/0.758786910639328146269034278112267/))

    !----------------------------------------------------------------------
    ! 27 point degree 5 quadrature.
    ! Citation: Gauss
    i=i+1
    allocate(hex_quads(i)%generator(4))
    hex_quads(i)%ngi=27
    hex_quads(i)%degree=5

    ! Origin
    hex_quads(i)%generator(1)=make_generator( &
         permutation=hex_permutations(1), &
         weight=(8./9.)**3, &
         coords=(/0.0/))

    ! 2 points on each axis
    hex_quads(i)%generator(2)=make_generator( &
         permutation=hex_permutations(2), &
         weight=(5./9.)*(8./9.)**2, &
         coords=(/sqrt(15.)/5./))

    ! Point for each edge.
    hex_quads(i)%generator(3)=make_generator( &
         permutation=hex_permutations(3), &
         weight=(8./9.)*(5./9.)**2, &
         coords=(/sqrt(15.)/5./))

    ! Corner points.
    hex_quads(i)%generator(4)=make_generator( &
         permutation=hex_permutations(5), &
         weight=(5./9.)**3, &
         coords=(/sqrt(15.)/5./))

    !----------------------------------------------------------------------
    ! 38 point degree 7 quadrature.
    ! Citation: KS 98
    i=i+1
    allocate(hex_quads(i)%generator(3))
    hex_quads(i)%ngi=38
    hex_quads(i)%degree=7

    hex_quads(i)%generator(1)=make_generator( &
         permutation=hex_permutations(2), &
         weight=0.295189738262622903181631100062774, &
         coords=(/0.901687807821291289082811566285950/))

    hex_quads(i)%generator(2)=make_generator( &
         permutation=hex_permutations(5), &
         weight=0.404055417266200582425904380777126, &
         coords=(/0.408372221499474674069588900002128/))

    hex_quads(i)%generator(3)=make_generator( &
         permutation=hex_permutations(6), &
         weight=0.124850759678944080062624098058597, &
         coords=(/0.859523090201054193116477875786220, &
         &        0.414735913727987720499709244748633/))


  end subroutine construct_hex_quadratures

  subroutine construct_quad_quadratures
    ! Construct list of available quadratures.
    ! The references cited are listed on the Encyclopedia of Cubature
    ! Formulas.
    integer :: i

    quad_quads%dim=2
    quad_quads%vertices=4

    i=0

    !----------------------------------------------------------------------
    ! 1 point degree 1 quadrature.
    ! Citation: Str71
    i=i+1
    ! Only one generator.
    allocate(quad_quads(i)%generator(1))
    quad_quads(i)%ngi=1
    quad_quads(i)%degree=1

    quad_quads(i)%generator(1)=make_generator( &
         permutation=quad_permutations(1), &
         weight=4.0, &
         coords=(/0.0/))

    !----------------------------------------------------------------------
    ! 4 point degree 3 quadrature.
    ! Citation: Gauss
    i=i+1
    allocate(quad_quads(i)%generator(1))
    quad_quads(i)%ngi=4
    quad_quads(i)%degree=3

    quad_quads(i)%generator(1)=make_generator( &
         permutation=quad_permutations(3), &
         weight=1.0, &
         coords=(/sqrt(3.0)/3.0/))

    !----------------------------------------------------------------------
    ! 8 point degree 5 quadrature.
    ! Citation: Str71
    i=i+1
    allocate(quad_quads(i)%generator(2))
    quad_quads(i)%ngi=8
    quad_quads(i)%degree=5

    quad_quads(i)%generator(1)=make_generator( &
         permutation=quad_permutations(2), &
         weight=0.816326530612244897959183673469387, &
         coords=(/0.683130051063973225548069245368070/))

    quad_quads(i)%generator(2)=make_generator( &
         permutation=quad_permutations(3), &
         weight=0.183673469387755102040816326530612, &
         coords=(/0.881917103688196863500538584546420/))

    !----------------------------------------------------------------------
    ! 9 point degree 5 quadrature.
    ! Citation: Gauss
    i=i+1
    allocate(quad_quads(i)%generator(4))
    quad_quads(i)%ngi=14
    quad_quads(i)%degree=5

    ! Origin
    quad_quads(i)%generator(1)=make_generator( &
         permutation=quad_permutations(1), &
         weight=(8./9.)**2, &
         coords=(/0.0/))

    ! 2 points on each axis
    quad_quads(i)%generator(2)=make_generator( &
         permutation=quad_permutations(2), &
         weight=(5./9.)*(8./9.), &
         coords=(/sqrt(15.)/5./))

    ! Corner points.
    quad_quads(i)%generator(3)=make_generator( &
         permutation=quad_permutations(3), &
         weight=(5./9.)**2, &
         coords=(/sqrt(15.)/5./))

    !----------------------------------------------------------------------
    ! 12 point degree 7 quadrature.
    ! Citation: Str71
    i=i+1
    allocate(quad_quads(i)%generator(3))
    quad_quads(i)%ngi=12
    quad_quads(i)%degree=7

    quad_quads(i)%generator(1)=make_generator( &
         permutation=quad_permutations(2), &
         weight=0.241975308641975308641975308641975, &
         coords=(/0.925820099772551461566566776583999/))

    quad_quads(i)%generator(2)=make_generator( &
         permutation=quad_permutations(3), &
         weight=0.520592916667394457139919432046731, &
         coords=(/0.380554433208315656379106359086394/))

    quad_quads(i)%generator(3)=make_generator( &
         permutation=quad_permutations(3), &
         weight=0.237431774690630234218105259311293, &
         coords=(/0.805979782918598743707856181350744/))

    !----------------------------------------------------------------------
    ! 20 point degree 9 quadrature.
    ! Citation: Str71
    i=i+1
    allocate(quad_quads(i)%generator(4))
    quad_quads(i)%ngi=20
    quad_quads(i)%degree=9

    quad_quads(i)%generator(1)=make_generator( &
         permutation=quad_permutations(2), &
         weight=0.0716134247098109667847339079718044, &
         coords=(/0.984539811942252392433000600300987/))

    quad_quads(i)%generator(2)=make_generator( &
         permutation=quad_permutations(2), &
         weight=0.454090352551545224132152403485726, &
         coords=(/0.488886342842372416227768621326681/))

    quad_quads(i)%generator(3)=make_generator( &
         permutation=quad_permutations(3), &
         weight=0.0427846154667780511691683400146727, &
         coords=(/0.939567287421521534134303076231667/))

    quad_quads(i)%generator(4)=make_generator( &
         permutation=quad_permutations(4), &
         weight=0.215755803635932878956972674263898, &
         coords=(/0.836710325023988974095346291152195, &
         &        0.507376773674613005277484034493916/))

  end subroutine construct_quad_quadratures

  subroutine construct_tet_permutations

    allocate(tet_permutations(1)%p(4,4))

    tet_permutations(1)%p=reshape((/&
         1, 0, 0, 0, &
         0, 1, 0, 0, &
         0, 0, 1, 0, &
         0, 0, 0, 1/),(/4,4/))

    allocate(tet_permutations(2)%p(4,6))

    tet_permutations(2)%p=reshape((/&
         1, 1, 0, 0, &
         1, 0, 1, 0, &
         1, 0, 0, 1, &
         0, 1, 1, 0, &
         0, 1, 0, 1, &
         0, 0, 1, 1/),(/4,6/))

    allocate(tet_permutations(3)%p(4,12))

    tet_permutations(3)%p=reshape((/&
         1, 2, 0, 0, &
         1, 0, 2, 0, &
         1, 0, 0, 2, &
         2, 1, 0, 0, &
         0, 1, 2, 0, &
         0, 1, 0, 2, &
         2, 0, 1, 0, &
         0, 2, 1, 0, &
         0, 0, 1, 2, &
         2, 0, 0, 1, &
         0, 2, 0, 1, &
         0, 0, 2, 1/),(/4,12/))

    allocate(tet_permutations(4)%p(4,4))

    tet_permutations(4)%p=reshape((/&
         1, 1, 1, 0, &
         1, 1, 0, 1, &
         1, 0, 1, 1, &
         0, 1, 1, 1/),(/4,4/))

    allocate(tet_permutations(5)%p(4,12))

    tet_permutations(5)%p=reshape((/&
         1, 1, 2, 0, &
         1, 1, 0, 2, &
         1, 2, 1, 0, &
         1, 0, 1, 2, &
         1, 2, 0, 1, &
         1, 0, 2, 1, &
         2, 1, 1, 0, &
         0, 1, 1, 2, &
         2, 1, 0, 1, &
         0, 1, 2, 1, &
         2, 0, 1, 1, &
         0, 2, 1, 1/),(/4,12/))

    allocate(tet_permutations(6)%p(4,24))

    tet_permutations(6)%p=reshape((/&
         1, 2, 3, 0, &
         1, 2, 0, 3, &
         1, 3, 2, 0, &
         1, 0, 2, 3, &
         1, 3, 0, 2, &
         1, 0, 3, 2, &
         2, 1, 3, 0, &
         2, 1, 0, 3, &
         3, 1, 2, 0, &
         0, 1, 2, 3, &
         3, 1, 0, 2, &
         0, 1, 3, 2, &
         2, 3, 1, 0, &
         2, 0, 1, 3, &
         3, 2, 1, 0, &
         0, 2, 1, 3, &
         3, 0, 1, 2, &
         0, 3, 1, 2, &
         2, 3, 0, 1, &
         2, 0, 3, 1, &
         3, 2, 0, 1, &
         0, 2, 3, 1, &
         3, 0, 2, 1, &
         0, 3, 2, 1/),(/4,24/))

    allocate(tet_permutations(7)%p(4,1))

    tet_permutations(7)%p(:,1)=(/1, 1, 1, 1/)

    allocate(tet_permutations(8)%p(4,4))

    tet_permutations(8)%p=reshape((/&
         1, 1, 1, 2, &
         1, 1, 2, 1, &
         1, 2, 1, 1, &
         2, 1, 1, 1/),(/4,4/))

    allocate(tet_permutations(9)%p(4,6))

    tet_permutations(9)%p=reshape((/&
         1, 1, 2, 2, &
         1, 2, 1, 2, &
         1, 2, 2, 1, &
         2, 1, 1, 2, &
         2, 1, 2, 1, &
         2, 2, 1, 1/),(/4,6/))

    allocate(tet_permutations(10)%p(4,12))

    tet_permutations(10)%p=reshape((/&
         1, 1, 2, 3, &
         1, 1, 3, 2, &
         1, 2, 1, 3, &
         1, 3, 1, 2, &
         1, 2, 3, 1, &
         1, 3, 2, 1, &
         2, 1, 1, 3, &
         3, 1, 1, 2, &
         2, 1, 3, 1, &
         3, 1, 2, 1, &
         2, 3, 1, 1, &
         3, 2, 1, 1/),(/4,12/))

    allocate(tet_permutations(11)%p(4,24))

    tet_permutations(11)%p=reshape((/&
         1, 2, 3, 4, &
         1, 2, 4, 3, &
         1, 3, 2, 4, &
         1, 4, 2, 3, &
         1, 3, 4, 2, &
         1, 4, 3, 2, &
         2, 1, 3, 4, &
         2, 1, 4, 3, &
         3, 1, 2, 4, &
         4, 1, 2, 3, &
         3, 1, 4, 2, &
         4, 1, 3, 2, &
         2, 3, 1, 4, &
         2, 4, 1, 3, &
         3, 2, 1, 4, &
         4, 2, 1, 3, &
         3, 4, 1, 2, &
         4, 3, 1, 2, &
         2, 3, 4, 1, &
         2, 4, 3, 1, &
         3, 2, 4, 1, &
         4, 2, 3, 1, &
         3, 4, 2, 1, &
         4, 3, 2, 1/),(/4,24/))

  end subroutine construct_tet_permutations

  subroutine construct_tri_permutations

    allocate(tri_permutations(1)%p(3,3))

    tri_permutations(1)%p=reshape((/&
         1, 0, 0, &
         0, 1, 0, &
         0, 0, 1/),(/3,3/))

    allocate(tri_permutations(2)%p(3,3))

    tri_permutations(2)%p=reshape((/&
         1, 1, 0, &
         1, 0, 1, &
         0, 1, 1/),(/3,3/))

    allocate(tri_permutations(3)%p(3,6))

    tri_permutations(3)%p=reshape((/&
         1, 2, 0, &
         1, 0, 2, &
         2, 1, 0, &
         0, 1, 2, &
         2, 0, 1, &
         0, 2, 1/),(/3,6/))

    allocate(tri_permutations(4)%p(3,1))

    tri_permutations(4)%p(:,1)=(/1,1,1/)

    allocate(tri_permutations(5)%p(3,3))

    tri_permutations(5)%p=reshape((/&
         1, 1, 2, &
         1, 2, 1, &
         2, 1, 1/),(/3,3/))

    allocate(tri_permutations(6)%p(3,6))

    tri_permutations(6)%p=reshape((/&
         1, 2, 3, &
         1, 3, 2, &
         2, 1, 3, &
         3, 1, 2, &
         2, 3, 1, &
         3, 2, 1/),(/3,6/))

  end subroutine construct_tri_permutations

  subroutine construct_tri_cycles
    ! Construct the cyclic permutations of a triangle.

    allocate(tri_cycles(1)%p(3,3))

    tri_cycles(1)%p=reshape((/&
         1, 0, 0, &
         0, 1, 0, &
         0, 0, 1/),(/3,3/))

    allocate(tri_cycles(2)%p(3,3))

    tri_cycles(2)%p=reshape((/&
         1, 1, 0, &
         1, 0, 1, &
         0, 1, 1/),(/3,3/))

    allocate(tri_cycles(3)%p(3,3))

    tri_cycles(3)%p=reshape((/&
         1, 2, 0, &
         0, 1, 2, &
         2, 0, 1/),(/3,3/))

    allocate(tri_cycles(4)%p(3,1))

    tri_cycles(4)%p(:,1)=(/1,1,1/)

    allocate(tri_cycles(5)%p(3,3))

    tri_cycles(5)%p=reshape((/&
         1, 1, 2, &
         1, 2, 1, &
         2, 1, 1/),(/3,3/))

    allocate(tri_cycles(6)%p(3,3))

    tri_cycles(6)%p=reshape((/&
         1, 2, 3, &
         3, 1, 2, &
         2, 3, 1/),(/3,3/))

  end subroutine construct_tri_cycles

  subroutine construct_point_permutation
    !! The trivial single permuration of the point.

    allocate(point_permutation(1)%p(1,1))

    point_permutation(1)%p=1

  end subroutine construct_point_permutation

  subroutine construct_interval_permutations

    allocate(interval_permutations(1)%p(2,1))

    interval_permutations(1)%p(:,1)=(/1,1/)

    allocate(interval_permutations(2)%p(2,2))

    interval_permutations(2)%p=reshape((/&
         1, 2, &
         2, 1/),(/2,2/))

  end subroutine construct_interval_permutations

  subroutine construct_hex_permutations

    allocate(hex_permutations(1)%p(3,1))

    hex_permutations(1)%p(:,1)=(/0,0,0/)

    allocate(hex_permutations(2)%p(3,6))

    hex_permutations(2)%p=reshape((/&
          1, 0, 0, &
          0, 1, 0, &
          0, 0, 1, &
         -1 ,0, 0, &
          0,-1, 0, &
          0, 0,-1/),(/3,6/))

    allocate(hex_permutations(3)%p(3,12))

    hex_permutations(3)%p=reshape((/&
          1, 1, 0, &
          1, 0, 1, &
          0, 1, 1, &
         -1, 1, 0, &
          1, 0,-1, &
          0,-1, 1, &
          1,-1, 0, &
         -1, 0, 1, &
          0, 1,-1, &
         -1,-1, 0, &
         -1, 0,-1, &
          0,-1,-1 /),(/3,12/))

    allocate(hex_permutations(4)%p(3,12))

    hex_permutations(4)%p=reshape((/&
          1, 2, 0, &
          1, 0, 2, &
          0, 1, 2, &
          2, 1, 0, &
          0, 2, 1, &
          2, 0, 1, &
         -1, 2, 0, &
         -1, 0, 2, &
          0,-1, 2, &
          2,-1, 0, &
          0, 2,-1, &
          2, 0,-1, &
          1,-2, 0, &
          1, 0,-2, &
          0, 1,-2, &
         -2, 1, 0, &
          0,-2, 1, &
         -2, 0, 1, &
         -1,-2, 0, &
         -1, 0,-2, &
          0,-1,-2, &
         -2,-1, 0, &
          0,-2,-1, &
         -2, 0,-1/),(/3,12/))

    allocate(hex_permutations(5)%p(3,8))

    hex_permutations(5)%p=reshape((/&
          1, 1, 1, &
          1, 1,-1, &
          1,-1, 1, &
          1,-1,-1, &
         -1, 1, 1, &
         -1, 1,-1, &
         -1,-1, 1, &
         -1,-1,-1/),(/3,8/))

    allocate(hex_permutations(6)%p(3,24))

    hex_permutations(6)%p=reshape((/&
          1, 1, 2, &
          1, 1,-2, &
          1,-1, 2, &
          1,-1,-2, &
         -1, 1, 2, &
         -1, 1,-2, &
         -1,-1, 2, &
         -1,-1,-2, &
          1, 2, 1, &
          1, 2,-1, &
          1,-2, 1, &
          1,-2,-1, &
         -1, 2, 1, &
         -1, 2,-1, &
         -1,-2, 1, &
         -1,-2,-1, &
          2, 1, 1, &
          2, 1,-1, &
          2,-1, 1, &
          2,-1,-1, &
         -2, 1, 1, &
         -2, 1,-1, &
         -2,-1, 1, &
         -2,-1,-1/),(/3,24/))

    allocate(hex_permutations(7)%p(3,48))

    hex_permutations(7)%p=reshape((/&
          1, 2, 3, &
          1, 2,-3, &
          1,-2, 3, &
          1,-2,-3, &
         -1, 2, 3, &
         -1, 2,-3, &
         -1,-2, 3, &
         -1,-2,-3, &
          1, 3, 2, &
          1, 3,-2, &
          1,-3, 2, &
          1,-3,-2, &
         -1, 3, 2, &
         -1, 3,-2, &
         -1,-3, 2, &
         -1,-3,-2, &
          2, 1, 3, &
          2, 1,-3, &
          2,-1, 3, &
          2,-1,-3, &
         -2, 1, 3, &
         -2, 1,-3, &
         -2,-1, 3, &
         -2,-1,-3, &
          3, 1, 2, &
          3, 1,-2, &
          3,-1, 2, &
          3,-1,-2, &
         -3, 1, 2, &
         -3, 1,-2, &
         -3,-1, 2, &
         -3,-1,-2, &
          2, 3, 1, &
          2, 3,-1, &
          2,-3, 1, &
          2,-3,-1, &
         -2, 3, 1, &
         -2, 3,-1, &
         -2,-3, 1, &
         -2,-3,-1, &
          3, 2, 1, &
          3, 2,-1, &
          3,-2, 1, &
          3,-2,-1, &
         -3, 2, 1, &
         -3, 2,-1, &
         -3,-2, 1, &
         -3,-2,-1/),(/3,48/))

  end subroutine construct_hex_permutations

  subroutine construct_quad_permutations

    allocate(quad_permutations(1)%p(2,1))

    quad_permutations(1)%p(:,1)=(/0,0/)

    allocate(quad_permutations(2)%p(2,4))

    quad_permutations(2)%p=reshape((/&
          1, 0, &
         -1, 0, &
          0, 1, &
          0,-1/),(/2,4/))

    allocate(quad_permutations(3)%p(2,4))

    quad_permutations(3)%p=reshape((/&
          1, 1, &
          1,-1, &
         -1, 1, &
         -1,-1/), (/2,4/))

    allocate(quad_permutations(4)%p(2,8))

    quad_permutations(4)%p=reshape((/&
          1, 2, &
          1,-2, &
         -1, 2, &
         -1,-2, &
          2, 1, &
          2,-1, &
         -2, 1, &
         -2,-1/),(/2,8/))

  end subroutine construct_quad_permutations

  subroutine local_coords_matrix_positions(positions, mat)
    ! dim x loc
    real, dimension(:, :), intent(in) :: positions
    real, dimension(size(positions, 2), size(positions, 2)), intent(out) :: mat

    mat(1:size(positions,1), :) = positions
    mat(size(positions,2), :) = 1.0

    call invert(mat)
  end subroutine local_coords_matrix_positions

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

end module quadrature
