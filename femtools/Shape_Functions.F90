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
module shape_functions
  !!< Generate shape functions for elements of arbitrary polynomial degree.
  use futils
  use FLDebug
  use polynomials
  use elements
  use element_numbering
  use ieee_arithmetic, only: ieee_quiet_nan, ieee_value

  implicit none

  private :: lagrange_polynomial, nonconforming_polynomial

  interface make_element_shape
     module procedure make_element_shape_from_element, make_element_shape
  end interface

contains

  function make_element_shape_from_element(model, vertices, dim, degree,&
       & quad, type, constraint_type_choice, &
       stat, quad_s)  result (shape)
    !!< This function enables element shapes to be derived from other
    !!< element shapes by specifying which attributes to change.
    type(element_type) :: shape
    type(element_type), intent(in) :: model
    !! Vertices is the number of vertices of the element, not the number of nodes!
    !! dim may be 1, 2, or 3.
    !! Element constraints
    integer, intent(in), optional :: constraint_type_choice
    !! Degree is the degree of the Lagrange polynomials.
    integer, intent(in), optional :: vertices, dim, degree
    type(quadrature_type), intent(in), target, optional :: quad
    integer, intent(in), optional :: type
    integer, intent(out), optional :: stat
    type(quadrature_type), intent(in), optional, target :: quad_s

    integer :: lvertices, ldim, ldegree
    type(quadrature_type) :: lquad
    type(quadrature_type), pointer :: lquad_s
    integer :: ltype

    if (present(vertices)) then
       lvertices=vertices
    else
       lvertices=model%cell%entity_counts(0)
    end if
    if (present(dim)) then
       ldim=dim
    else
       ldim=model%dim
    end if
    if(present(degree)) then
       ldegree=degree
    else
       ldegree=model%degree
    end if
    if(present(quad)) then
       lquad=quad
    else
       lquad=model%quadrature
    end if
    if(present(type)) then
       ltype=type
    else
       ltype=model%type
    end if
    if(present(quad_s)) then
       lquad_s=>quad_s
    else if (associated(model%surface_quadrature)) then
       lquad_s=>model%surface_quadrature
    else
       lquad_s=>null()
    end if

    if (associated(lquad_s)) then
       shape = make_element_shape(lvertices, ldim, ldegree, lquad, ltype,&
            stat, lquad_s, constraint_type_choice=constraint_type_choice)
    else
       shape = make_element_shape(lvertices, ldim, ldegree, lquad, ltype,&
            stat, constraint_type_choice=constraint_type_choice)
    end if

  end function make_element_shape_from_element

  function make_element_shape(vertices, dim, degree, quad, type,&
       stat, quad_s, constraint_type_choice)  result (element)
    !!< Generate the shape functions for an element. The result is a suitable
    !!< element_type.
    !!
    !!< At this stage only Lagrange family polynomial elements are supported.
    type(element_type) :: element
    !! Vertices is the number of vertices of the element, not the number of nodes!
    !! dim \in [1,2,3] is currently supported.
    !! Degree is the degree of the Lagrange polynomials.
    integer, intent(in) :: vertices, dim, degree
    type(quadrature_type), intent(in), target :: quad
    integer, intent(in), optional :: type
    integer, intent(out), optional :: stat
    integer, intent(in), optional :: constraint_type_choice
    type(quadrature_type), intent(in), optional, target :: quad_s
    real, pointer :: g(:)=> null()

    type(ele_numbering_type), pointer :: ele_num
    ! Count coordinates of each point
    integer, dimension(dim+1) :: counts
    integer :: i,j,k
    integer :: ltype, coords,surface_count
    real :: dx
    type(constraints_type), pointer :: constraint

    ! Check that the quadrature and the element shapes match.
    assert(quad%vertices==vertices)
    assert(quad%dim==dim)

    if (present(stat)) stat=0

    ! Get the local numbering of our element
    ele_num=>find_element_numbering(vertices, dim, degree, type)

    if (.not.associated(ele_num)) then
       if (present(stat)) then
          stat=1
          return
       else
          FLAbort('Element numbering unavailable.')
       end if
    end if

    ! The number of local coordinates depends on the element family.
    select case(ele_num%family)
    case (FAMILY_SIMPLEX)
       coords=dim+1
    case (FAMILY_CUBE)
       if(ele_num%type==ELEMENT_TRACE .and. dim==2) then
          !For trace elements the local coordinate is face number
          !then the local coordinates on the face
          !For quads, the face is an interval element which has
          !two local coordinates.
          coords=3
       else
          coords=dim
       end if
    case default
       FLAbort('Illegal element family.')
    end select

    element%cell => find_cell(dim, vertices)
    element%numbering=>ele_num
    element%quadrature=quad
    call incref(quad)

    call allocate(element, dim, ele_num%nodes, quad%ngi, coords, type)

    if (present(quad_s)) then
       if (dim==1) then
          FLAbort("Unsupported dimension count. Can only generate facet functions for elements that exist in 2 or 3 dimensions.")
       end if
       call allocate_element_facets(element, dim, ele_num%nodes,&
            facet_count(element%cell), quad_s%ngi)
    end if

    element%degree=degree
    element%n=0.0
    element%dn=0.0

    ! Construct shape for each node
    do i=1,element%ndof

       counts(1:coords)=ele_num%number2count(:,i)

       ! Construct appropriate polynomials.
       do j=1,coords
          select case(element%type)
          case(ELEMENT_LAGRANGIAN,ELEMENT_DISCONTINUOUS_LAGRANGIAN)
             if (degree == 0) then
                dx = 0.0
             else
                dx = 1.0/degree
             end if
             select case(ele_num%family)
             case (FAMILY_SIMPLEX)
                ! Raw polynomial.
                element%spoly(j,i)&
                     =lagrange_polynomial(counts(j), counts(j), dx)
             case(FAMILY_CUBE)
                element%spoly(j,i)&
                     =lagrange_polynomial(counts(j), degree, dx)
             end select

          case(ELEMENT_TRACE)
             element%spoly(j,i) = (/ieee_value(0.0,ieee_quiet_nan)/)

          case(ELEMENT_BUBBLE)
             if(i==element%ndof) then

               ! the last node is the bubble shape function
               element%spoly(j,i) = (/1.0, 0.0/)

             else
               select case(ele_num%family)
               case (FAMILY_SIMPLEX)
                  ! Raw polynomial.
                  element%spoly(j,i)&
                       =lagrange_polynomial(counts(j)/coords, counts(j)/coords, 1.0/degree)

               end select

             end if

          case(ELEMENT_NONCONFORMING)
             element%spoly(j,i)=nonconforming_polynomial(counts(j))

          case default

             FLAbort('An unsupported element type has been selected.')

          end select

          ! Derivative
          if(ele_num%type==ELEMENT_TRACE) then
             element%dspoly(j,i) = (/ieee_value(0.0,ieee_quiet_nan)/)
          else
             element%dspoly(j,i)=ddx(element%spoly(j,i))
          end if
       end do

       if(ele_num%type==ELEMENT_TRACE) then
          !No interior functions, hence NaNs
          element%n = ieee_value(0.0,ieee_quiet_nan)
          element%dn = ieee_value(0.0,ieee_quiet_nan)
          if(present(quad_s)) then
             FLAbort('Shouldn''t be happening')
          end if
       else
          ! Loop over all the quadrature points.
          do j=1,quad%ngi

             ! Raw shape function
             element%n(i,j)=eval_shape(element, i, quad%l(j,:))

             ! Directional derivatives.
             element%dn(i,j,:)=eval_dshape(element, i, quad%l(j,:))
          end do

          if (present(quad_s)) then
             element%surface_quadrature=>quad_s
             select case(element%type)
             case(FAMILY_SIMPLEX)
                allocate(g(dim+1))
                do k=1,dim+1
                   do j=1,quad_s%ngi
                      if (dim==2) then
                         g(mod(k+2,3)+1)=0.0
                         g(mod(k,3)+1)=quad_s%l(j,1)
                         g(mod(k+1,3)+1)=quad_s%l(j,2)
                      else if (dim==3) then
                         ! Not checked !!
                         g(mod(k+3,4)+1)=0.0
                         g(mod(k,4)+1)=quad_s%l(j,1)
                         g(mod(k+1,4)+1)=quad_s%l(j,2)
                         g(mod(k+2,4)+1)=quad_s%l(j,3)
                      end if
                      element%n_s(i,j,k)=eval_shape(element, i,g)
                      element%dn_s(i,j,k,:)=eval_dshape(element, i,g)
                   end do
                end do
                deallocate(g)
             end select
          end if
       end if
    end do

    if(present(constraint_type_choice)) then
       if(constraint_type_choice/=CONSTRAINT_NONE) then
          allocate(constraint)
          element%constraints=>constraint
          call allocate(element%constraints,element,constraint_type_choice)
       end if
    end if

    call create_entity_dofs(element)
    call create_facet_dofs(element)

  contains

    subroutine create_entity_dofs(element)
      ! Create lists of the dofs on each entity.
      type(element_type), intent(inout) :: element

      type(cell_type), pointer :: cell
      type(ele_numbering_type), pointer :: numbering

      integer, dimension(:), allocatable :: dofs
      integer, dimension(:), pointer :: vertices
      integer :: i, j, dof_len, facet_dim, cell_dim
      logical, dimension(:), allocatable :: cell_mask

      cell=>element%cell
      numbering=>element%numbering

      allocate(element%entity2dofs(0:ubound(cell%entities,1),size(cell%entities,2)))

      allocate(cell_mask(element%ndof))
      cell_mask=.true.

      ! DG elements have all their dofs associated with the interior.
      ! Trace elements have all their dofs associated with the facets.
      if (element%type/=ELEMENT_DISCONTINUOUS_LAGRANGIAN &
           .and. element%type/=ELEMENT_TRACE) then

         ! Vertices
         if (cell%entity_counts(0)>0.and.numbering%nodes_per(0)>0) then
            allocate(dofs(cell%entity_counts(0)))
            dofs=local_vertices(numbering)
            do i=1,size(dofs)
               allocate(element%entity2dofs(0,i)%dofs(1))
               element%entity2dofs(0,i)%dofs(1)=dofs(i)
            end do
            cell_mask(dofs)=.false.
            deallocate(dofs)
         end if

         if (cell%dimension>1.and.numbering%nodes_per(1)>0) then
            ! Edges
            dof_len=numbering%nodes_per(1)
            do i=1,cell%entity_counts(1)
               vertices=>entity_vertices(cell,[1,i])
               allocate(element%entity2dofs(1,i)%dofs(dof_len))
               element%entity2dofs(1,i)%dofs=&
                    edge_local_num(vertices, numbering)
               ! Sanity check dofs uniquely belong to one entity.
               assert(all(cell_mask(element%entity2dofs(1,i)%dofs)))
               cell_mask(element%entity2dofs(1,i)%dofs)=.false.
            end do
         end if

         if (cell%dimension>2.and.numbering%nodes_per(1)>0) then
            ! Faces
            dof_len=face_num_length(numbering, interior=.true.)
            do i=1,cell%entity_counts(2)
               vertices=>entity_vertices(cell,[2,i])
               allocate(element%entity2dofs(2,i)%dofs(dof_len))
               element%entity2dofs(2,i)%dofs=&
                    face_local_num(vertices, numbering, interior=.true.)
               ! Sanity check dofs uniquely belong to one entity.
               assert(all(cell_mask(element%entity2dofs(2,i)%dofs)))
               cell_mask(element%entity2dofs(2,i)%dofs)=.false.
            end do
         end if
      end if

      ! Facets for trace elements.
      if (element%type==ELEMENT_TRACE) then
         dof_len=facet_num_length(numbering, interior=.false.)
         facet_dim=cell%dimension-1
         do i=1,facet_count(cell)
            allocate(element%entity2dofs(facet_dim,i)%dofs(dof_len))
            element%entity2dofs(facet_dim,i)%dofs=&
                 facet_numbering(numbering, i)
            ! Sanity check dofs uniquely belong to one entity.
            assert(all(cell_mask(element%entity2dofs(facet_dim,i)%dofs)))
            cell_mask(element%entity2dofs(facet_dim,i)%dofs)=.false.
         end do
         assert(all(.not.cell_mask))
      end if


      ! Interior cell elements.
      if (any(cell_mask)) then
         cell_dim=cell%dimension
         assert(.not.allocated(element%entity2dofs(cell_dim,1)%dofs))
         allocate(element%entity2dofs(cell_dim,1)%dofs(count(cell_mask)))
         element%entity2dofs(cell_dim,1)%dofs=&
              pack([(i,i=1,element%ndof)],cell_mask)
      end if

      ! Ensure that all remaining entity2dof entries are zero.
      do i=0,ubound(cell%entities,1)
         do j=1,size(cell%entities,2)
            if (.not.allocated(element%entity2dofs(i,j)%dofs)) then
               allocate(element%entity2dofs(i,j)%dofs(0))
            end if
         end do
      end do

    end subroutine create_entity_dofs

    subroutine create_facet_dofs(element)
      ! Create lists of the dofs on each facet.
      type(element_type), intent(inout) :: element

      type(cell_type), pointer :: cell
      type(ele_numbering_type), pointer :: numbering

      integer :: facet_dim

      cell=>element%cell
      numbering=>element%numbering
      facet_dim=element%dim-1

      allocate(element%facet2dofs(facet_count(cell)))
      do i=1,facet_count(cell)
         allocate(element%facet2dofs(i)%dofs(&
              facet_num_length(numbering, interior=.false.)))
         element%facet2dofs(i)%dofs=&
              facet_numbering(numbering, i)
      end do

    end subroutine create_facet_dofs

  end function make_element_shape

  function lagrange_polynomial(n,degree,dx, origin) result (poly)
    ! nth equispaced lagrange polynomial of specified degree and point
    ! spacing dx.
    integer, intent(in) :: n, degree
    real, intent(in) :: dx
    type(polynomial) :: poly
    ! fixes location of n=0 location (0.0 if not specified)
    real, intent(in), optional :: origin

    real lorigin
    integer :: i

    ! This shouldn't be necessary but there appears to be a bug in initial
    ! component values in gfortran:
    poly%coefs=>null()
    poly%degree=-1

    if (present(origin)) then
       lorigin=origin
    else
       lorigin=0.0
    end if

    poly=(/1.0/)

    degreeloop: do i=0,degree
       if (i==n) cycle degreeloop

       poly=poly*(/1.0, -(lorigin+i*dx) /)

    end do degreeloop

    ! normalize to 1.0 in the n-th location
    poly=poly/eval(poly, lorigin+n*dx)

  end function lagrange_polynomial

  function nonconforming_polynomial(n) result (poly)
    ! nth P1 nonconforming polynomial.
    integer, intent(in) :: n
    type(polynomial) :: poly

    ! This shouldn't be necessary but there appears to be a bug in initial
    ! component values in gfortran:
    poly%coefs=>null()
    poly%degree=-1

    poly=(/1.0/)

    if (n==0) then

       ! polynomial is -2x+1
       poly=(/-2.0, 1.0/)

    end if

  end function nonconforming_polynomial

end module shape_functions
