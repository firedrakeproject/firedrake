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
module elements
  !!< This module provides derived types for finite elements and associated functions.
  use element_numbering
  use quadrature
  use FLDebug
  use polynomials
  use reference_counting
  use cell_numbering
  use quicksort
  use ieee_arithmetic, only: ieee_quiet_nan, ieee_value
  implicit none

  type dof_list
     !!< Container type for a list of vertices
     integer, dimension(:), allocatable :: dofs
  end type dof_list

  type element_type
     !!< Type to encode shape and quadrature information for an element.
     integer :: dim !! 2d or 3d?
     integer :: ndof !! Number of degrees of fredom (nodes).
     integer :: ngi !! Number of gauss points.
     integer :: degree !! Polynomial degree of element.
     integer :: type !! Identifier for elements eg. Lagrange
     !!                 See Element_Numbering for a list.
     !! Shape functions: n is for the primitive function, dn is for partial derivatives, dn_s is for partial derivatives on surfaces.
     !! n is ndof x ngi, dn is ndof x ngi x dim
     !! dn_s is ndof x ngi x face x dim
     real, pointer :: n(:,:)=>null(), dn(:,:,:)=>null()
     real, pointer :: n_s(:,:,:)=>null(), dn_s(:,:,:,:)=>null()
     !! Polynomials defining shape functions and their derivatives.
     type(polynomial), dimension(:,:), pointer :: spoly=>null(), dspoly=>null()
     !! Link back to the node numbering used for this element.
     type(ele_numbering_type), pointer :: numbering=>null()
     !! Link back to the quadrature used for this element.
     type(quadrature_type) :: quadrature
     type(quadrature_type), pointer :: surface_quadrature=>null()
     !! Pointer to constraints data for this element
     type(constraints_type), pointer :: constraints=>null()
     !! Reference count to prevent memory leaks.
     type(refcount_type), pointer :: refcount=>null()
     !! Dummy name to satisfy reference counting
     character(len=0) :: name
     !! Mapping from element entities to degrees of freedom
     type(dof_list), dimension(:,:), pointer :: entity2dofs=>null()
     !! Mapping from facets to degrees of freedom.
     type(dof_list), dimension(:), pointer :: facet2dofs=>null()
     !! Topological entity numbering
     type(cell_type), pointer :: cell
  end type element_type

  type constraints_type
     !!< A type to encode the constraints from the local Lagrange basis for
     !!< (Pn)^d vector-valued elements to another local basis, possibly for
     !!< a proper subspace. This new basis must have DOFs consisting
     !!< of either normal components on faces corresponding to a Lagrange
     !!< basis for the normal component when restricted to each face,
     !!< or coefficients of basis
     !!< functions with vanishing normal components on all faces.
     !! type of constraints
     integer :: type
     !! local dimension
     integer :: dim
     !! order of local Lagrange basis
     integer :: degree
     !! number of degrees of freedom for local Lagrange basis
     integer :: ndof
     !! number of face degrees of freedom for local Lagrange basis
     integer :: face_ndof
     !! Number of constraints
     integer :: n_constraints
     !! basis of functions that are orthogonal to the
     !! constrained vector space
     !! dimension n_constraints x ndof x dim
     real, pointer :: orthogonal(:,:,:)=> null()

     !! BELOW: Stuff for doing commuting projections
     !! Number of basis functions used on each face for projection
     integer :: n_face_basis
     !! basis of functions (relative to local Lagrange basis on face)
     !! used for projection on faces
     !! dimension n_face_basis x face_ndof
     real, pointer :: face_basis(:,:) => null()
     !! Number of basis functions for gradient for projection
     integer :: n_grad_basis
     !! basis of functions (relative to local Lagrange basis)
     !! used for projection onto their gradient
     !! dimension n_grad_basis x loc x dim
     real, pointer :: grad_basis(:,:,:) => null()
     !! Number of basis functions for curl for projection
     integer :: n_curl_basis
     !! basis of divergence-free functions (relative to local Lagrange
     !! basis) used for projection
     !! dimension n_curl_basis x loc x dim
     real, pointer :: curl_basis(:,:,:) => null()
  end type constraints_type

  integer, parameter :: CONSTRAINT_NONE =0, CONSTRAINT_BDFM = 1,&
       & CONSTRAINT_RT = 2, CONSTRAINT_BDM = 3

  interface allocate
     module procedure allocate_element, allocate_constraints_type
  end interface

  interface deallocate
     module procedure deallocate_element
     module procedure deallocate_constraints
  end interface

  interface local_coords
     module procedure element_local_coords
  end interface

  interface local_coord_count
     module procedure element_local_coord_count
  end interface

  interface local_vertices
     module procedure element_local_vertices
  end interface

  interface operator(==)
     module procedure element_equal
  end interface

  interface eval_shape
    module procedure eval_shape_node, eval_shape_all_nodes
  end interface

  interface eval_dshape
    module procedure eval_dshape_node, eval_dshape_all_nodes
  end interface

  interface cell_family
     module procedure cell_family_element
  end interface cell_family

  interface vertex_count
     module procedure element_vertex_count
  end interface vertex_count

  interface facet_count
     module procedure element_facet_count
  end interface facet_count

#include "Reference_count_interface_element_type.F90"

contains

  subroutine allocate_element(element, dim, ndof, ngi, coords, type, stat)
    !!< Allocate memory for an element_type.
    type(element_type), intent(inout) :: element
    !! Dim is the dimension of the element, ndof is number of nodes, ngi is
    !! number of gauss points.
    integer, intent(in) :: dim,ndof,ngi
    !! Number of local coordinates.
    integer, intent(in) :: coords
    !! Stat returns zero for success and nonzero otherwise.
    integer, intent(in), optional :: type
    integer, intent(out), optional :: stat
    !
    integer :: lstat

    if(present(type)) then
       element%type = type
    else
       element%type=ELEMENT_LAGRANGIAN
    end if

    select case(element%type)
    case(ELEMENT_LAGRANGIAN, ELEMENT_DISCONTINUOUS_LAGRANGIAN,&
         & ELEMENT_NONCONFORMING, ELEMENT_BUBBLE, ELEMENT_TRACE)

      allocate(element%n(ndof,ngi),element%dn(ndof,ngi,dim), &
          element%spoly(coords,ndof), element%dspoly(coords,ndof), stat=lstat)

    case(ELEMENT_CONTROLVOLUME_SURFACE)

      allocate(element%n(ndof,ngi),element%dn(ndof,ngi,dim-1), &
          stat=lstat)

      element%spoly=>null()
      element%dspoly=>null()

    case(ELEMENT_CONTROLVOLUMEBDY_SURFACE)

      allocate(element%n(ndof,ngi),element%dn(ndof,ngi,dim), &
          stat=lstat)

      element%spoly=>null()
      element%dspoly=>null()

    case default

      FLAbort("Attempt to select an illegal element type.")

    end select

    element%ndof=ndof
    element%ngi=ngi
    element%dim=dim


    nullify(element%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(element)

    nullify(element%n_s)
    nullify(element%dn_s)

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Unable to allocate element.")
    end if

  end subroutine allocate_element

  subroutine allocate_element_facets(element, dim, ndof,&
       facets, ngi_s, stat)
    !!< Allocate memory for the facet shape functions of an element_type.
    type(element_type), intent(inout) :: element
    !! Dim is the dimension of the element, ndof is number of nodes, ngi is
    !! number of gauss points.
    integer, intent(in) :: dim,ndof,facets,ngi_s
    integer, intent(out), optional :: stat

    integer :: lstat

    allocate(element%n_s(ndof,ngi_s,facets),element%dn_s(ndof,ngi_s,facets,dim),&
         stat=lstat)

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Unable to allocate element facets.")
    end if

  end subroutine allocate_element_facets

  subroutine allocate_constraints_type(constraint, element, type, stat)
    !!< Allocate memory for a constraints type
    type(element_type), intent(in) :: element
    type(constraints_type), intent(inout) :: constraint
    integer, intent(in) :: type !type of constraint
    !! Stat returns zero for success and nonzero otherwise.
    integer, intent(out), optional :: stat
    !
    integer :: lstat

    lstat = 0
    constraint%type = type
    constraint%dim = element%dim
    constraint%ndof = element%ndof
    constraint%degree = element%degree
    constraint%face_ndof = facet_num_length(element%numbering,.false.)

    select case (type)
    case (CONSTRAINT_BDFM)
       select case(cell_family(element))
       case (FAMILY_SIMPLEX)
          if(constraint%degree<3) then
             constraint%n_constraints = constraint%dim+1
             constraint%n_face_basis = constraint%degree
             constraint%n_grad_basis = &
                  constraint%degree*(constraint%degree+1)/2-1
             !The below formula definitely fails for degree=>3
             constraint%n_curl_basis = constraint%degree-1
          else
             FLAbort('High order not supported yet')
          end if
       case (FAMILY_CUBE)
          FLExit('Haven''t implemented BDFM1 on quads yet.')
       case default
          FLAbort('Illegal element family.')
       end select
    case (CONSTRAINT_RT)
       select case(cell_family(element))
       case (FAMILY_SIMPLEX)
          FLExit('Haven''t implemented RT0 on simplices yet.')
          if(constraint%degree<3) then
             constraint%n_constraints = constraint%dim+1
          else
             FLAbort('High order not supported yet')
          end if
       case (FAMILY_CUBE)
          if(constraint%degree==1) then
             constraint%n_face_basis = 0
             constraint%n_grad_basis = 0
             constraint%n_curl_basis = 0
             constraint%n_constraints = 2**(constraint%dim)
          else if(constraint%degree==2) then
             constraint%n_face_basis = 0
             constraint%n_grad_basis = 0
             constraint%n_curl_basis = 0
             constraint%n_constraints = 6
          else
             FLAbort('Higher order not supported yet')
          end if
       case default
          FLAbort('Illegal element family.')
       end select
    case (CONSTRAINT_BDM)
       constraint%n_constraints = 0
       select case(cell_family(element))
       case (FAMILY_SIMPLEX)
          if(constraint%degree>3) then
             FLExit('high order not supported')
          end if
          !The below formulas definitely fail for degree=>3
          constraint%n_face_basis = constraint%degree+1
          constraint%n_grad_basis = 0
          constraint%n_curl_basis = 0
       case default
          FLAbort('Unsupported element family.')
       end select
       case (CONSTRAINT_NONE)
          constraint%n_constraints = 0
    case default
       FLExit('Unknown constraint type')
    end select

    if(constraint%n_constraints>0) then
       allocate(&
            constraint%orthogonal(constraint%n_constraints,&
            constraint%ndof,constraint%dim),stat=lstat)
       if(lstat==0) then
          call make_constraints(constraint,cell_family(element))
       end if
    end if

    if(constraint%n_face_basis>0) then
       allocate(constraint%face_basis(constraint%n_face_basis,&
            constraint%face_ndof))
    end if
    if(constraint%n_grad_basis>0) then
       allocate(constraint%grad_basis(constraint%n_grad_basis,&
            constraint%ndof,constraint%dim))
    end if
    if(constraint%n_curl_basis>0) then
       allocate(constraint%curl_basis(constraint%n_curl_basis,&
            constraint%ndof,constraint%dim))
    end if

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Unable to allocate element.")
    end if

  end subroutine allocate_constraints_type

  subroutine deallocate_element(element, stat)
    type(element_type), intent(inout) :: element
    integer, intent(out), optional :: stat

    integer :: lstat, tstat
    integer :: i,j

    tstat = 0
    lstat = 0

    call decref(element)
    if (has_references(element)) then
       ! There are still references to this element so we don't deallocate.
       return
    end if

    call deallocate(element%quadrature)

    if(associated(element%spoly)) then
      do i=1,size(element%spoly,1)
        do j=1,size(element%spoly,2)
            call deallocate(element%spoly(i,j))
        end do
      end do
      deallocate(element%spoly, stat=tstat)
    end if
    lstat=max(lstat,tstat)

    if (associated(element%n_s)) deallocate(element%n_s,element%dn_s)

    if(associated(element%dspoly)) then
      do i=1,size(element%dspoly,1)
        do j=1,size(element%dspoly,2)
            call deallocate(element%dspoly(i,j))
        end do
      end do
      deallocate(element%dspoly, stat=tstat)
    end if
    lstat=max(lstat,tstat)

    if(associated(element%entity2dofs)) then
       deallocate(element%entity2dofs, stat=tstat)
       lstat=max(lstat,tstat)
    end if

    if(associated(element%facet2dofs)) then
       deallocate(element%facet2dofs, stat=tstat)
       lstat=max(lstat,tstat)
    end if

    deallocate(element%n,element%dn, stat=tstat)
    lstat=max(lstat,tstat)

    if(associated(element%constraints)) then
       call deallocate(element%constraints,stat=tstat)
       lstat = max(lstat,tstat)

       deallocate(element%constraints, stat=tstat)
       lstat = max(lstat,tstat)
    end if
    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Unable to deallocate element.")
    end if

  end subroutine deallocate_element

  subroutine deallocate_constraints(constraint, stat)
    type(constraints_type), intent(inout) :: constraint
    integer, intent(out), optional :: stat

    integer :: lstat, tstat

    lstat = 0
    tstat = 0

    if(associated(constraint%orthogonal)) then
       deallocate(constraint%orthogonal,stat=lstat)
    end if
    if(associated(constraint%face_basis)) then
       deallocate(constraint%face_basis,stat=tstat)
       lstat=max(tstat,lstat)
    end if
    if(associated(constraint%grad_basis)) then
       deallocate(constraint%grad_basis,stat=tstat)
       lstat=max(tstat,lstat)
    end if
    if(associated(constraint%curl_basis)) then
       deallocate(constraint%curl_basis,stat=tstat)
       lstat=max(tstat,lstat)
    end if

    if (present(stat)) then
       stat=lstat
    else if (lstat/=0) then
       FLAbort("Unable to deallocate constraints.")
    end if

  end subroutine deallocate_constraints

  function facet_dofs(element, facet)
    !!< Return a pointer to the list of local dofs on facet.
    type(element_type), intent(in) :: element
    integer, intent(in) :: facet
    integer, dimension(:), pointer :: facet_dofs

    facet_dofs=>element%facet2dofs(facet)%dofs

  end function facet_dofs

  function element_local_coords(n, element) result (coords)
    !!< Work out the local coordinates of node n in element. This is just a
    !!< wrapper function which allows local_coords to be called on an element
    !!< instead of on an element numbering.
    integer, intent(in) :: n
    type(element_type), intent(in) :: element
    real, dimension(size(element%numbering%number2count, 1)) :: coords

    coords=local_coords(n, element%numbering)

  end function element_local_coords

  pure function element_local_coord_count(element) result (n)
    !!< Return the number of local coordinates associated with element.
    integer :: n
    type(element_type), intent(in) :: element

    n=size(element%numbering%number2count, 1)

  end function element_local_coord_count

  function element_local_vertices(element) result (vertices)
    !!< Given an element, return the local node numbers of its
    !!< vertices.
    type(element_type), intent(in) :: element
    integer, dimension(vertex_count(element)) :: vertices

    vertices=local_vertices(element%numbering)

  end function element_local_vertices

  pure function element_equal(element1,element2)
    !!< Return true if the two elements are equivalent.
    logical :: element_equal
    type(element_type), intent(in) :: element1, element2

    element_equal = element1%dim==element2%dim &
         .and. element1%ndof==element2%ndof &
         .and. element1%ngi==element2%ngi &
         .and. element1%numbering==element2%numbering &
         .and. element1%quadrature==element2%quadrature

  end function element_equal

  subroutine extract_old_element(element, N, NLX, NLY, NLZ)
    !!< Extract the shape function values from an old element.
    type(element_type), intent(in) :: element
    real, dimension(element%ndof, element%ngi), intent(out) :: N, NLX, NLY
    real, dimension(element%ndof, element%ngi), intent(out), optional :: NLZ

    N=element%n
    NLX=element%dn(:,:,1)
    if (size(element%dn,3)>1) then
       NLY=element%dn(:,:,2)
    else
       NLY=0.0
    end if

    if (present(NLZ)) then
       if (size(element%dn,3)>2) then
          NLZ=element%dn(:,:,3)
       else
          NLZ=0.0
       end if
    end if


  end subroutine extract_old_element

  pure function eval_shape_node(shape, node,  l) result(eval_shape)
    ! Evaluate the shape function for node node local coordinates l
    real :: eval_shape
    type(element_type), intent(in) :: shape
    integer, intent(in) :: node
    real, dimension(size(shape%spoly,1)), intent(in) :: l

    integer :: i

    eval_shape=1.0

    do i=1,size(shape%spoly,1)

       ! Raw shape function
       eval_shape=eval_shape*eval(shape%spoly(i,node), l(i))

    end do

  end function eval_shape_node

  pure function eval_shape_all_nodes(shape, l) result(eval_shape)
    ! Evaluate the shape function for all locations at local coordinates l
    type(element_type), intent(in) :: shape
    real, dimension(size(shape%spoly,1)), intent(in) :: l
    real, dimension(shape%ndof) :: eval_shape

    integer :: i,j

    eval_shape=1.0

    do j=1,shape%ndof

      do i=1,size(shape%spoly,1)

        ! Raw shape function
        eval_shape(j)=eval_shape(j)*eval(shape%spoly(i,j), l(i))

      end do

    end do

  end function eval_shape_all_nodes

  pure function eval_dshape_node(shape, node,  l) result(eval_dshape)
    !!< Evaluate the derivatives of the shape function for location node at local
    !!< coordinates l
    type(element_type), intent(in) :: shape
    integer, intent(in) :: node
    real, dimension(:), intent(in) :: l
    real, dimension(shape%dim) :: eval_dshape

    select case(cell_family(shape))

    case (FAMILY_SIMPLEX)

       eval_dshape=eval_dshape_simplex(shape, node,  l)

    case (FAMILY_CUBE)

       eval_dshape=eval_dshape_cube(shape, node,  l)

    case default
       ! Invalid element family. Return a really big number to stuff things
       ! quickly.

       eval_dshape=huge(0.0)

    end select

  end function eval_dshape_node

  function eval_dshape_all_nodes(shape, l) result(eval_dshape)
    type(element_type), intent(in) :: shape
    real, dimension(:), intent(in) :: l
    real, dimension(shape%ndof, shape%dim) :: eval_dshape

    integer :: dof

    do dof=1,shape%ndof
      eval_dshape(dof, :) = eval_dshape_node(shape, dof, l)
    end do
  end function eval_dshape_all_nodes

  function eval_dshape_transformed(shape, l, invJ) result(transformed_dshape)
    type(element_type), intent(in) :: shape
    real, dimension(:), intent(in) :: l
    real, dimension(shape%dim, shape%dim), intent(in) :: invJ
    real, dimension(shape%ndof, shape%dim) :: transformed_dshape, untransformed_dshape

    integer :: dof

    do dof=1,shape%ndof
      untransformed_dshape(dof, :) = eval_dshape_node(shape, dof, l)
      transformed_dshape(dof, :) = matmul(invJ, untransformed_dshape(dof, :))
    end do
  end function eval_dshape_transformed

  function eval_volume_dshape_at_face_quad(shape, local_face_number, invJ) result(output)
    ! Compute the derivatives of the volume basis functions at the quadrature points
    ! of a given surface element. Useful for strain tensors and such

    ! If this segfaults on entry, it's probably because
    ! shape%surface_quadrature is unassociated. You need to augment the shape
    ! function with the quadrature information. See the drag calculation
    ! in MeshDiagnostics.F90 for an example (search for augmented_shape).
    type(element_type), intent(in) :: shape ! NOT the face shape! The volume shape!
    integer, intent(in) :: local_face_number ! which face are we on
    real, dimension(:, :, :), intent(in) :: invJ
    real, dimension(shape%ndof, shape%surface_quadrature%ngi, shape%dim) :: output
    integer :: dof, gi

    assert(associated(shape%dn_s))
    assert(size(invJ, 1) == shape%dim)
    assert(size(invJ, 2) == shape%dim)
    assert(size(invJ, 3) == shape%surface_quadrature%ngi)
    assert(shape%dim == size(shape%dn_s, 4))
    assert(shape%ndof == size(shape%dn_s, 1))
    assert(shape%surface_quadrature%ngi == size(shape%dn_s, 2))
    assert(local_face_number <= size(shape%dn_s, 3))
    assert(shape%dim == size(shape%dn_s, 4))

    ! You can probably do this with some fancy-pants tensor contraction.
    do dof=1,shape%ndof
      do gi=1,shape%surface_quadrature%ngi
        output(dof, gi, :) = matmul(invJ(:, :, gi), shape%dn_s(dof, gi, local_face_number, :))
      end do
    end do
  end function eval_volume_dshape_at_face_quad

  pure function eval_dshape_simplex(shape, dof,  l) result (eval_dshape)
    !!< Evaluate the derivatives of the shape function for dofation dof at local
    !!< coordinates l
    !!<
    !!< This version of the function applies to members of the simplex
    !!< family including the interval.
    type(element_type), intent(in) :: shape
    integer, intent(in) :: dof
    real, dimension(shape%dim+1), intent(in) :: l
    real, dimension(shape%dim) :: eval_dshape

    integer :: i,j
    ! Derivative of the dependent coordinate with respect to the other
    ! coordinates:
    real, dimension(shape%dim) :: dl4dl

    ! Find derivative of dependent coordinate
    dl4dl=diffl4(shape%cell%entity_counts(0), shape%dim)

    do i=1,shape%dim
       ! Directional derivatives.

       ! The derivative has to take into account the dependent
       ! coordinate. In 3D:
       !
       !  S=P1(L1)P2(L2)P3(L3)P4(L4)
       !
       !  dS        / dP1     dL4 dP4  \
       !  --- = P2P3| ---P4 + ---*---P1|
       !  dL1       \ dL1     dL1 dL4  /
       !

       ! Expression in brackets.
       eval_dshape(i)=eval(shape%dspoly(i,dof), l(i))&
            *eval(shape%spoly(shape%dim+1,dof),l(shape%dim+1))&
            + dl4dl(i)&
            *eval(shape%dspoly(shape%dim+1,dof), l(shape%dim+1)) &
            *eval(shape%spoly(i,dof),l(i))

       ! The other terms
       do j=1,shape%dim
          if (j==i) cycle

          eval_dshape(i)=eval_dshape(i)*eval(shape%spoly(j,dof), l(j))
       end do

    end do

  end function eval_dshape_simplex

  pure function eval_dshape_cube(shape, dof,  l) result (eval_dshape)
    !!< Evaluate the derivatives of the shape function for location dof at local
    !!< coordinates l
    !!<
    !!< This version of the function applies to members of the hypercube
    !!< family. Note that this does NOT include the interval.
    type(element_type), intent(in) :: shape
    integer, intent(in) :: dof
    real, dimension(shape%dim+1), intent(in) :: l
    real, dimension(shape%dim) :: eval_dshape

    integer :: i,j

    do i=1,shape%dim
       eval_dshape(i)=1.0
       ! Directional derivatives.
       do j=1,shape%dim
          if(i==j) then
            eval_dshape(i)=eval_dshape(i)*eval(shape%dspoly(j,dof), l(j))
          else
            eval_dshape(i)=eval_dshape(i)*eval(shape%spoly(j,dof), l(j))
          end if
       end do

    end do

  end function eval_dshape_cube

  pure function diffl4(vertices, dimension)
    ! Derivative of the dependent coordinate with respect to the other
    ! coordinates.
    integer, intent(in) :: vertices, dimension
    real, dimension(dimension) :: diffl4

    if (vertices==dimension+1) then
       ! Simplex. Dependent coordinate depends on all other coordinates.
       diffl4=-1.0

    else if (vertices==2**dimension) then
       ! Hypercube. The dependent coordinate is redundant.
       diffl4=0.0

    else if (vertices==6.and.dimension==3) then
       ! Wedge. First coordinate is independent.
       diffl4=(/0.0,-1.0,-1.0/)

    else
       ! No output permitted in a pure procedure so we return a big number to stuff
       ! things up quickly.
       diffl4=huge(0.0)
    end if

  end function diffl4

  subroutine make_constraints(constraint,family)
    type(constraints_type), intent(inout) :: constraint
    integer, intent(in) :: family
    !
    select case(family)
    case (FAMILY_SIMPLEX)
       select case(constraint%type)
       case (CONSTRAINT_BDM)
          !do nothing
       case (CONSTRAINT_BDFM)
          select case(constraint%dim)
          case (2)
             select case(constraint%degree)
             case (1)
                !BDFM0 is the same as RT0
                call make_constraints_rt0_triangle(constraint)
             case (2)
                call make_constraints_bdfm1_triangle(constraint)
             case default
                FLExit('Unknown constraints type')
             end select
          case default
             FLExit('Unsupported dimension')
          end select
       case (CONSTRAINT_RT)
          select case(constraint%dim)
          case (2)
             select case(constraint%degree)
             case (1)
                call make_constraints_rt0_triangle(constraint)
             case default
                FLExit('Unknown constraints type')
             end select
          case default
             FLExit('Unsupported dimension')
          end select
       case default
          FLExit('Unknown constraints type')
       end select
    case (FAMILY_CUBE)
       select case(constraint%type)
       case (CONSTRAINT_BDM)
          !do nothing
       case (CONSTRAINT_RT)
          select case(constraint%dim)
          case (2)
             select case(constraint%degree)
             case (1)
                call make_constraints_rt0_square(constraint)
             case (2)
                call make_constraints_rt1_square(constraint)
             case default
                FLExit('Unknown constraints type')
             end select
          case default
             FLExit('Unsupported dimension')
          end select
       case default
          FLExit('Unknown constraints type')
       end select
    case default
       FLExit('Unknown element numbering family')
    end select
  end subroutine make_constraints

  subroutine make_constraints_bdfm1_triangle(constraint)
    implicit none
    type(constraints_type), intent(inout) :: constraint
    real, dimension(3,2) :: n
    integer, dimension(3,3) :: face_loc
    integer :: dim1, face, floc
    real, dimension(3) :: c

    if(constraint%dim/=2) then
       FLExit('Only implemented for 2D so far')
    end if

    !BDFM1 constraint requires that normal components are linear.
    !This means that the normal components at the edge centres
    !need to be constrained to the average of the normal components
    !at each end of the edge.

    !DOFS    FACES
    ! 3
    ! 5 2    1 3
    ! 6 4 1   2

    !constraint equations are:
    ! (0.5 u_3 - u_5 + 0.5 u_6).n_1 = 0
    ! (0.5 u_1 - u_4 + 0.5 u_6).n_2 = 0
    ! (0.5 u_1 - u_2 + 0.5 u_3).n_3 = 0

    !face local nodes to element local nodes
    face_loc(1,:) = (/ 3,5,6 /)
    face_loc(2,:) = (/ 1,4,6 /)
    face_loc(3,:) = (/ 1,2,3 /)

    !normals
    n(1,:) = (/ -1., 0. /)
    n(2,:) = (/  0.,-1. /)
    n(3,:) = (/ 1./sqrt(2.),1./sqrt(2.) /)

    !coefficients in each face
    c = (/ 0.5,-1.,0.5 /)

    !constraint%orthogonal(i,loc,dim1) stores the coefficient
    !for basis function loc, dimension dim1 in equation i.

    constraint%orthogonal = 0.
    do face = 1, 3
       do floc = 1,3
          do dim1 = 1, 2
             constraint%orthogonal(face,face_loc(face,floc),dim1) = &
                  c(floc)*n(face,dim1)
          end do
       end do
    end do
    !! dimension n_constraints x loc x dim
  end subroutine make_constraints_bdfm1_triangle

  subroutine make_constraints_rt0_triangle(constraint)
    implicit none
    type(constraints_type), intent(inout) :: constraint
    real, dimension(3,2) :: n
    integer, dimension(3,2) :: face_loc
    integer :: dim1, face, floc, count
    real, dimension(2) :: c

    if(constraint%dim/=2) then
       FLExit('Only implemented for 2D so far')
    end if

    !RT0 constraint requires that normal components are constant.
    !This means that both the normal components at each end of the
    !edge need to have the same value.

    !DOFS    FACES
    ! 2
    !        1 3
    ! 3   1   2

    !constraint equations are:
    ! (u_2 - u_3).n_1 = 0
    ! (u_1 - u_3).n_2 = 0
    ! (u_1 - u_2).n_3 = 0

    !face local nodes to element local nodes
    face_loc(1,:) = (/ 2,3 /)
    face_loc(2,:) = (/ 1,3 /)
    face_loc(3,:) = (/ 1,2 /)

    !normals
    n(1,:) = (/ -1., 0. /)
    n(2,:) = (/  0.,-1. /)
    n(3,:) = (/ 1./sqrt(2.),1./sqrt(2.) /)

    !constraint coefficients
    c = (/ 1., -1. /)

    !constraint%orthogonal(i,loc,dim1) stores the coefficient
    !for basis function loc, dimension dim1 in equation i.

    constraint%orthogonal = 0.
    count = 0
    do face = 1, 3
       count = count + 1
       do floc = 1,2
          do dim1 = 1, 2
             constraint%orthogonal(count,face_loc(face,floc),dim1)&
                  = c(floc)*n(face,dim1)
          end do
       end do
    end do
    assert(count==3)
    !! dimension n_constraints x loc x dim
  end subroutine make_constraints_rt0_triangle

  subroutine make_constraints_rt0_square(constraint)
    implicit none
    type(constraints_type), intent(inout) :: constraint
    real, dimension(4,2) :: n
    integer, dimension(4,2) :: face_loc
    integer :: dim1, face, floc, count
    real, dimension(2) :: c

    if(constraint%dim/=2) then
       FLExit('Only implemented for 2D so far')
    end if

    !RT0 constraint requires that normal components are constant.
    !This means that both the normal components at each end of the
    !edge need to have the same value.

    !DOFS    FACES
    ! 3   4   1
    !        3 2
    ! 1   2   4

    !constraint equations are:
    ! (u_3 - u_4).n_1 = 0
    ! (u_2 - u_4).n_2 = 0
    ! (u_1 - u_3).n_3 = 0
    ! (u_1 - u_2).n_4 = 0

    !face local nodes to element local nodes
    face_loc(1,:) = (/ 3,4 /)
    face_loc(2,:) = (/ 2,4 /)
    face_loc(3,:) = (/ 1,3 /)
    face_loc(4,:) = (/ 1,2 /)

    !normals
    n(1,:) = (/  0.,  1. /)
    n(2,:) = (/  1.,  0. /)
    n(3,:) = (/ -1.,  0. /)
    n(4,:) = (/  0., -1. /)

    !constraint%orthogonal(i,loc,dim1) stores the coefficient
    !for basis function loc, dimension dim1 in equation i.

    !constraint coefficients
    c = (/ 1., -1. /)

    constraint%orthogonal = 0.
    count  = 0
    do face = 1, 4
       count = count + 1
       do floc = 1,2
          do dim1 = 1, 2
             constraint%orthogonal(count,face_loc(face,floc),dim1)&
                  = c(floc)*n(face,dim1)
          end do
       end do
    end do
    assert(count==4)
    !! dimension n_constraints x loc x dim
  end subroutine make_constraints_rt0_square

  subroutine make_constraints_rt1_square(constraint)
    implicit none
    type(constraints_type), intent(inout) :: constraint
    real, dimension(2,2) :: n
    integer, dimension(6,3) :: loc
    integer :: dim1, d, nloc, count
    real, dimension(3) :: c

    if(constraint%dim/=2) then
       FLExit('Only implemented for 2D so far')
    end if

    !RT1 constraint requires that normal components are linear.
    !This means that the normal components at the centres
    !need to be constrained to the average of the normal components
    !at each end.

    !DOFS    FACES
    ! 7 8 9    1
    ! 4 5 6   3 2
    ! 1 2 3    4

    !constraint equations are:
    ! (0.5u_1 - u_2 + 0.5u_3).n_1 = 0
    ! (0.5u_4 - u_5 + 0.5u_6).n_1 = 0
    ! (0.5u_7 - u_8 + 0.5u_9).n_1 = 0
    ! (0.5u_1 - u_4 + 0.5u_7).n_2 = 0
    ! (0.5u_2 - u_5 + 0.5u_8).n_2 = 0
    ! (0.5u_3 - u_6 + 0.5u_9).n_2 = 0

    loc(1,:) = (/ 1,2,3 /)
    loc(2,:) = (/ 4,5,6 /)
    loc(3,:) = (/ 7,8,9 /)
    loc(4,:) = (/ 1,4,7 /)
    loc(5,:) = (/ 2,5,8 /)
    loc(6,:) = (/ 3,6,9 /)

    !normals
    n(1,:) = (/ 0.,  1. /)
    n(2,:) = (/ 1.,  0. /)

    !constraint%orthogonal(i,loc,dim1) stores the coefficient
    !for basis function loc, dimension dim1 in equation i.

    !constraint coefficients
    c = (/ 0.5, -1., 0.5 /)

    constraint%orthogonal = 0.
    do count = 1, 6
       do nloc = 1, 3
          do dim1 = 1, 2
             d = (count-1)/3 + 1
             constraint%orthogonal(count,loc(count,nloc),dim1)&
                  = c(nloc)*n(d,dim1)
          end do
       end do
    end do
    !! dimension n_constraints x loc x dim
  end subroutine make_constraints_rt1_square

  subroutine nodalise_basis(shape)
    !Subroutine to transform bubble basis to a equivalent nodal one.
    type(element_type), intent(inout) :: shape
    !
    if(shape%type .ne. ELEMENT_BUBBLE) then
       FLAbort('Only applies to bubbles.')
    end if

    select case(shape%dim)
    case (2)
       select case (vertex_count(shape))
       case (3)
          select case( shape%ndof )
          case (7)
             call nodalise_basis_P2b()
          case default
             FLAbort('Element not supported.')
          end select
       case default
          FLAbort('Family not supported')
       end select
    case default
       FLAbort('Dimension not supported.')
    end select

  contains

    subroutine nodalise_basis_P2b()
      !
      integer :: i,j,loc
      real, dimension(7) :: N_vals
      N_vals = eval_shape(shape, (/1.0/3.0,1.0/3.0,1.0/3.0/))

      !! n is loc x ngi, dn is loc x ngi x dim
      shape%n(7,:) = shape%n(7,:)/N_vals(7)
      do loc = 1, 6
         shape%n(loc,:) = shape%n(loc,:) - N_vals(loc)*shape%n(7,:)
         shape%dn(loc,:,:) = shape%dn(loc,:,:) - &
              &N_vals(loc)*shape%dn(7,:,:)
      end do

      !spoly is now useless
      if(associated(shape%spoly)) then
         do i=1,size(shape%spoly,1)
            do j=1,size(shape%spoly,2)
               shape%spoly(i,j) = (/ieee_value(0.0,ieee_quiet_nan)/)
            end do
         end do
      end if

    end subroutine nodalise_basis_P2b

  end subroutine nodalise_basis

  function sorted_facet(shape, facet, vertices)
    ! Return the degrees of freedom on facet of shape reoriented according
    ! to the order of vertices, which is the ordered set of vertices on
    ! the facet.
    !
    ! The change of coordinates is only valid for hypercube elements.
    ! Happily, simplex facets will be in the correct order by construction
    ! due to the element numbering convention.
    type(element_type), intent(in) :: shape
    integer, intent(in) :: facet
    integer, dimension(:), intent(in) :: vertices

    integer, dimension(size(shape%facet2dofs(facet)%dofs)) :: sorted_facet

    ! Coordinates of facet vertices in element local coordinates.
    real, dimension(size(vertices),shape%dim) :: vertex_coords
    ! Matrix mapping from element local coordinates to facet local coordinates.
    real, dimension(shape%dim-1, shape%dim) :: A
    ! Local coordinates of facet dofs in facet space.
    real, dimension(size(shape%facet2dofs(facet)%dofs),shape%dim-1) :: dof_coords

    ! Local coordinate temporary to deal with simplex dependent coordinates.
    real, dimension(local_coord_count(shape)) :: tmp_local

    integer, dimension(size(shape%facet2dofs(facet)%dofs)) :: permutation, raw_facet

    integer :: i

    if (cell_family(shape)==FAMILY_SIMPLEX) then

       sorted_facet=shape%facet2dofs(facet)%dofs

    else if (cell_family(shape)==FAMILY_CUBE) then

       assert(all(sorted(vertices)==entity_vertices(shape%cell,[shape%dim-1,facet])))

       vertex_coords=shape%cell%vertex_coords(vertices,:)

       do i=1, shape%dim-1
          A(i,:)=vertex_coords(i,:)-vertex_coords(1,:)
       end do

       raw_facet=shape%facet2dofs(facet)%dofs

       do i=1, size(sorted_facet)
          tmp_local = local_coords(raw_facet(i), shape)
          dof_coords(i,:) = matmul(A, tmp_local(:shape%dim)-vertex_coords(1,:))
       end do

       call sort(dof_coords(:,shape%dim-1:1:-1), permutation)

       sorted_facet = raw_facet(permutation)

    else

       FLAbort("Unknown element family")

    end if

  end function sorted_facet

  pure function cell_family_element(element)
    !! Wrap the numbering%family to enable its removal.
    integer :: cell_family_element
    type(element_type), intent(in) :: element

    cell_family_element = element%numbering%family

  end function cell_family_element

  pure function element_vertex_count(element)
    integer :: element_vertex_count
    type(element_type), intent(in) :: element

    element_vertex_count = element%cell%entity_counts(0)

  end function element_vertex_count

  pure function element_facet_count(element)
    integer :: element_facet_count
    type(element_type), intent(in) :: element

    element_facet_count = facet_count(element%cell)

  end function element_facet_count

#include "Reference_count_element_type.F90"

end module elements
