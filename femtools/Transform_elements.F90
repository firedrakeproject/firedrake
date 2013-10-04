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
module transform_elements
  ! Module to calculate element transformations from local to physical
  ! coordinates.
  use quadrature
  use elements
  use vector_tools
  use parallel_tools, only: abort_if_in_parallel_region
  use fields_base
  use cv_faces, only: cv_faces_type
  use eventcounter
  use memory_diagnostics

  implicit none

  interface transform_to_physical
    module procedure transform_to_physical_full, transform_to_physical_detwei
  end interface

  interface transform_facet_to_physical
    module procedure transform_facet_to_physical_full, &
      transform_facet_to_physical_detwei
  end interface transform_facet_to_physical

  interface retrieve_cached_transform
     module procedure retrieve_cached_transform_full, &
          retrieve_cached_transform_det
  end interface

  interface retrieve_cached_face_transform
     module procedure retrieve_cached_face_transform_full
  end interface

  private
  public :: transform_to_physical, transform_facet_to_physical, &
            transform_cvsurf_to_physical, transform_cvsurf_facet_to_physical, &
            transform_horizontal_to_physical, &
            compute_jacobian, compute_inverse_jacobian, element_volume,&
            cache_transform_elements, deallocate_transform_cache, &
            prepopulate_transform_cache

  integer, parameter :: cyc3(1:5)=(/ 1, 2, 3, 1, 2 /)

  logical, save :: cache_transform_elements=.true.
  real, dimension(:,:,:), allocatable, save :: invJ_cache
  real, dimension(:,:,:), allocatable, save :: J_T_cache
  real, dimension(:), allocatable, save :: detJ_cache

  real, dimension(:,:), allocatable, save :: face_normal_cache
  real, dimension(:), allocatable, save :: face_detJ_cache
  ! Record which element is on the other side of the last n/2 elements.
  integer, dimension(:), allocatable, save :: face_cache


  ! The reference count id of the positions mesh being cached.
  integer, save :: position_id=-1
  integer, save :: last_mesh_movement=-1
  integer, save :: face_position_id=-1
  integer, save :: face_last_mesh_movement=-1

contains

  function retrieve_cached_transform_full(X, ele, J_local_T, invJ_local,&
       & detJ_local) result (cache_valid)
    !!< Determine whether the transform cache is valid for this operation.
    !!<
    !!< If caching is applicable and the cache is not ready, set up the
    !!< cache and then return true.
    type(vector_field), intent(in) :: X
    integer, intent(in) :: ele
    !! Local versions of the Jacobian matrix and its inverse. (dim x dim)
    real, dimension(:,:), intent(out) :: J_local_T, invJ_local
    !! Local version of the determinant of J
    real, intent(out) :: detJ_local

    logical :: cache_valid

    cache_valid=.true.

    if (X%refcount%id/=position_id) then
!       ewrite(2,*) "Reference count identity of X has changed."
       cache_valid=.false.
       if (X%name/="Coordinate") then
          !!< If Someone is not calling this on the main Coordinate field
          !!< then we're screwed anyway.
          return
       end if

    else if(eventcount(EVENT_MESH_MOVEMENT)/=last_mesh_movement) then
!       ewrite(2,*) "Mesh has moved."
       cache_valid=.false.

    end if

    if (.not.cache_valid) then
       call construct_cache(X)
       cache_valid=.true.
    end if

    J_local_T=J_T_cache(:, :, ele)
    invJ_local=invJ_cache(:, :, ele)
    detJ_local=detJ_cache(ele)

  end function retrieve_cached_transform_full

  function retrieve_cached_transform_det(X, ele, detJ_local) &
       result (cache_valid)
    !!< Determine whether the transform cache is valid for this operation.
    !!<
    !!< If caching is applicable and the cache is not ready, set up the
    !!< cache and then return true.
    type(vector_field), intent(in) :: X
    integer, intent(in) :: ele
    !! Local version of the determinant of J
    real, intent(out) :: detJ_local

    logical :: cache_valid

    cache_valid=.true.

    if (X%refcount%id/=position_id) then
       cache_valid=.false.
       if (X%name/="Coordinate") then
          !!< If Someone is not calling this on the main Coordinate field
          !!< then we're screwed anyway.
          return
       end if

!       ewrite(2,*) "Reference count identity of X has changed."
    else if(eventcount(EVENT_MESH_MOVEMENT)/=last_mesh_movement) then
!       ewrite(2,*) "Mesh has moved."
       cache_valid=.false.

    end if

    if (.not.cache_valid) then
       call construct_cache(X)
       cache_valid=.true.
    end if

    detJ_local=detJ_cache(ele)

  end function retrieve_cached_transform_det

  function prepopulate_transform_cache(X) result(cache_valid)
    !!< Prepopulate the caches for transform_to_physical and
    !!< transform_face_to_physical
    !!
    !!< If you're going to call transform_to_physical on a coordinate
    !!< field inside a threaded region, you need to call this on the
    !!< same field before entering the region.
    type(vector_field), intent(in) :: X
    logical :: cache_valid
    logical :: face_cache_valid
    cache_valid=.true.
    face_cache_valid=.true.
    ! Although the caches are thread safe, the code that assembles the
    ! caches is not so we want a simple way to construct them if
    ! appropriate before entering a threaded region.
    if (X%refcount%id /= position_id) then
       cache_valid=.false.
       if (X%name/="Coordinate") then
          !!< If Someone is not calling this on the main Coordinate field
          !!< then we're screwed anyway.
          return
       end if
    else if (eventcount(EVENT_MESH_MOVEMENT) /= last_mesh_movement) then
       cache_valid=.false.
    end if

    if (X%refcount%id /= face_position_id) then
       face_cache_valid=.false.
    else if (eventcount(EVENT_MESH_MOVEMENT) /= face_last_mesh_movement) then
       face_cache_valid = .false.
    end if

    if (.not.cache_valid) then
       call construct_cache(X)
       cache_valid=.true.
    end if

    if (.not.face_cache_valid) then
       call construct_face_cache(X)
       face_cache_valid=.true.
    end if

    cache_valid = cache_valid .and. face_cache_valid

  end function prepopulate_transform_cache

  subroutine construct_cache(X)
    !!< The cache is invalid so make a new one.
    type(vector_field), intent(in) :: X

    integer :: elements, ele, i, k
    !! Note that if X is not all linear simplices we are screwed.
    real, dimension(X%dim, ele_loc(X,1)) :: X_val
    type(element_type), pointer :: X_shape

!    ewrite(1,*) "Reconstructing element geometry cache."

    call abort_if_in_parallel_region

    position_id=X%refcount%id
    last_mesh_movement=eventcount(EVENT_MESH_MOVEMENT)

    if (allocated(invJ_cache)) then
#ifdef HAVE_MEMORY_STATS
       call register_deallocation("transform_cache", &
            "real", size(invJ_cache)+size(J_T_cache)+size(detJ_cache))
#endif
       deallocate(invJ_cache, J_T_cache, detJ_cache)
    end if

    elements=element_count(X)

    allocate(invJ_cache(X%dim,X%dim,elements), &
         J_T_cache(X%dim,X%dim,elements), &
         detJ_cache(elements))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("transform_cache", &
         "real", size(invJ_cache)+size(J_T_cache)+size(detJ_cache))
#endif

    x_shape=>ele_shape(X,1)

    do ele=1,elements
       X_val=ele_val(X, ele)
       !     |- dx  dx  dx  -|
       !     |  dL1 dL2 dL3  |
       !     |               |
       !     |  dy  dy  dy   |
       ! J = |  dL1 dL2 dL3  |
       !     |               |
       !     |  dz  dz  dz   |
       !     |- dL1 dL2 dL3 -|

       ! Form Jacobian.
       ! Since X is linear we need only do this at quadrature point 1.
       J_T_cache(:,:,ele)=matmul(X_val(:,:), x_shape%dn(:, 1, :))

       select case (X%dim)
       case(1)
          invJ_cache(:,:,ele)=1.0
       case(2)
          invJ_cache(:,:,ele)=reshape(&
               (/ J_T_cache(2,2,ele),-J_T_cache(1,2,ele),&
               & -J_T_cache(2,1,ele), J_T_cache(1,1,ele)/),(/2,2/))
       case(3)
          ! Calculate (scaled) inverse using recursive determinants.
          forall (i=1:3,k=1:3)
             invJ_cache(i, k, ele)= &
                  J_T_cache(cyc3(i+1),cyc3(k+1),ele)&
                  &          *J_T_cache(cyc3(i+2),cyc3(k+2),ele) &
                  -J_T_cache(cyc3(i+2),cyc3(k+1),ele)&
                  &          *J_T_cache(cyc3(i+1),cyc3(k+2),ele)
          end forall
       case default
          FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
       end select

       ! Form determinant by expanding minors.
       detJ_cache(ele)=dot_product(J_T_cache(:,1,ele),invJ_cache(:,1,ele))

       ! Scale inverse by determinant.
       invJ_cache(:,:,ele)=invJ_cache(:,:,ele)/detJ_cache(ele)

    end do

  end subroutine construct_cache

  function retrieve_cached_face_transform_full(X, face, &
       & normal_local, detJ_local) result (cache_valid)
    !!< Determine whether the transform cache is valid for this operation.
    !!<
    !!< If caching is applicable and the cache is not ready, set up the
    !!< cache and then return true.
    type(vector_field), intent(in) :: X
    integer, intent(in) :: face
    !! Face determinant
    real, intent(out) :: detJ_local
    !! Face normal
    real, dimension(X%dim), intent(out) :: normal_local

    logical :: cache_valid

    cache_valid=.true.

    if (X%refcount%id/=face_position_id) then
!       ewrite(2,*) "Reference count identity of X has changed."
       cache_valid=.false.
       if (X%name/="Coordinate") then
          !!< If Someone is not calling this on the main Coordinate field
          !!< then we're screwed anyway.
          return
       end if

    else if(eventcount(EVENT_MESH_MOVEMENT)/=face_last_mesh_movement) then
!       ewrite(2,*) "Mesh has moved."
       cache_valid=.false.

    end if

    if (.not.cache_valid) then
       call construct_face_cache(X)
       cache_valid=.true.
    end if

    detJ_local=face_detJ_cache(abs(face_cache(face)))
    normal_local=sign(1,face_cache(face))*face_normal_cache(:,abs(face_cache(face)))

  end function retrieve_cached_face_transform_full

  subroutine construct_face_cache(X)
    !!< The cache is invalid so make a new one.
    type(vector_field), intent(in) :: X

    integer :: elements, ele, i, current_face, face, face2, faces, n,&
         & unique_faces
    !! Note that if X is not all linear simplices we are screwed.
    real, dimension(X%dim, ele_loc(X,1)) :: X_val
    real, dimension(X%dim, face_loc(X,1)) :: X_f
    real, dimension(X%dim, X%dim-1) :: J
    type(element_type), pointer :: X_shape_f
    real :: detJ
    integer, dimension(:), pointer :: neigh

!    ewrite(1,*) "Reconstructing element geometry cache."

    call abort_if_in_parallel_region

    face_position_id=X%refcount%id
    face_last_mesh_movement=eventcount(EVENT_MESH_MOVEMENT)

    if (allocated(face_detJ_cache)) then
#ifdef HAVE_MEMORY_STATS
       call register_deallocation("transform_cache", "real", &
            & size(face_detJ_cache)+size(face_normal_cache))
       call register_deallocation("transform_cache", "integer", &
            & size(face_cache))
#endif
       deallocate(face_detJ_cache, face_normal_cache, face_cache)
    end if

    elements=element_count(X)
    faces=face_count(X)
    !! This counts 1/2 for each interior face and 1 for each surface face.
    unique_faces=unique_face_count(X%mesh)

    allocate(face_detJ_cache(unique_faces), &
         face_normal_cache(X%dim,unique_faces), &
         face_cache(faces))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("transform_cache", "real", &
         & size(face_detJ_cache)+size(face_normal_cache))
    call register_allocation("transform_cache", "integer", &
         & size(face_cache))
#endif

    current_face=0
    do ele=1,elements
       neigh=>ele_neigh(X, ele)
       X_val=ele_val(X,ele)

       do n=1,size(neigh)

          if (neigh(n)<0) then
             face=ele_face(X, ele, neigh(n))

             current_face=current_face+1
             face_cache(face)=current_face

          else

             face=ele_face(X, ele, neigh(n))
             face2=ele_face(X, neigh(n), ele)

             ! Only do this once for each face pairl
             if (face>face2) then
                cycle
             end if

             current_face=current_face+1

             face_cache(face)=current_face
             face_cache(face2)=-current_face

          end if


          X_f=face_val(X,face)
          X_shape_f=>face_shape(X,face)

          !     |- dx  dx  -|
          !     |  dL1 dL2  |
          !     |           |
          !     |  dy  dy   |
          ! J = |  dL1 dL2  |
          !     |           |
          !     |  dz  dz   |
          !     |- dL1 dL2 -|

          ! Form Jacobian.
          J=matmul(X_f(:,:), x_shape_f%dn(:, 1, :))

          detJ=0.0
          ! Calculate determinant.
          select case (X%dim)
          case(1)
             detJ=1.0
          case(2)
             detJ = sqrt(J(1,1)**2 + J(2,1)**2)
          case(3)
             do i=1,3
                detJ=detJ+ &
                     (J(cyc3(i+2),1)*J(cyc3(i+1),2)-J(cyc3(i+2),2)*J(cyc3(i+1),1))**2
             end do
             detJ=sqrt(detJ)
          case default
             FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
          end select

          ! Calculate normal.
          face_normal_cache(:,current_face)=normgi(X_val,X_f,J)
          face_detJ_cache(current_face)=detJ

       end do
    end do
    assert(current_face==unique_faces)

  end subroutine construct_face_cache

  function unique_face_count(mesh) result (face_count)
    !!< Count the number of geometrically unique faces in mesh,
    type(mesh_type), intent(in) :: mesh
    integer :: face_count

    integer :: ele
    integer, dimension(:), pointer :: neigh

    face_count=0

    do ele=1,element_count(mesh)
       neigh=>ele_neigh(mesh, ele)

       ! Count 1 for interior and 2 for surface.
       face_count=face_count+sum(merge(1,2,neigh>0))

    end do

    face_count=(face_count+1)/2

  end function unique_face_count

  subroutine deallocate_transform_cache

    if (allocated(invJ_cache)) then
#ifdef HAVE_MEMORY_STATS
       call register_deallocation("transform_cache", "real",&
            & size(invJ_cache)+size(J_T_cache)+size(detJ_cache))
#endif
       deallocate(invJ_cache, J_T_cache, detJ_cache)
    end if

    if (allocated(face_detJ_cache)) then
#ifdef HAVE_MEMORY_STATS
       call register_deallocation("transform_cache", "real", &
            & size(face_detJ_cache)+size(face_normal_cache))
       call register_deallocation("transform_cache", "integer", &
            & size(face_cache))
#endif
       deallocate(face_detJ_cache, face_normal_cache, face_cache)
    end if

    position_id=-1
    last_mesh_movement=-1
    face_position_id=-1
    face_last_mesh_movement=-1

  end subroutine deallocate_transform_cache

  subroutine transform_to_physical_full(X, ele, shape, dshape, detwei, J,&
       & invJ, detJ, x_shape)
    !!< Calculate the derivatives of a shape function shape in physical
    !!< space using the positions x. Calculate the transformed quadrature
    !!< weights as a side bonus.
    !!<
    !!< Do this by calculating the Jacobian of the transform and inverting it.

    !! X is the positions field.
    type(vector_field), intent(in) :: X
    !! The index of the current element
    integer :: ele
    !! Reference element of which the derivatives are to be transformed
    type(element_type), intent(in) :: shape
    !! Derivatives of this shape function transformed to physical space (loc x ngi x dim)
    real, dimension(:,:,:), intent(out) ::  dshape

    !! Quadrature weights for physical coordinates.
    real, dimension(:), intent(out), optional :: detwei(:)
    !! Jacobian matrix and its inverse at each quadrature point (dim x dim x x_shape%ngi)
    !! Facilitates access to this information externally
    real, dimension(:,:,:), intent(out), optional :: J, invJ
    !! Determinant of the Jacobian at each quadrature point (x_shape%ngi)
    !! Facilitates access to this information externally
    real, dimension(:), intent(out), optional :: detJ
    !! Shape function to use for the coordinate field. ONLY SUPPLY THIS IF
    !! YOU DO NOT WANT TO USE THE SHAPE FUNCTION IN THE COORDINATE FIELD.
    !! This is mostly useful for control volumes.
    type(element_type), intent(in), optional, target :: X_shape

    !! Column n of X is the position of the nth node. (dim x x_shape%ndof)
    !! only need position of n nodes since Jacobian is only calculated once
    real, dimension(X%dim,ele_loc(X,ele)) :: X_val
    !! Shape function to be used for coordinate interpolation.
    type(element_type), pointer :: lx_shape

    !! Local copy of element gradients. This is an attempt to induce a
    !! prefetch.
    real, dimension(size(shape%dn,1), size(shape%dn,3)) :: dn_local

    !! Local versions of the Jacobian matrix and its inverse. (dim x dim)
    real, dimension(X%dim, X%dim) :: J_local_T, invJ_local
    !! Local version of the determinant of J
    real :: detJ_local

    integer :: gi, i, k, dim
    logical :: x_nonlinear, m_nonlinear, cache_valid

    if (present(X_shape)) then
       lx_shape=>X_shape
    else
       lx_shape=>ele_shape(X,ele)
    end if

    ! Optimisation checks. Optimisations apply to linear elements.
    x_nonlinear= .not.(lx_shape%degree==1 .and. cell_family(lx_shape)==FAMILY_SIMPLEX)
    m_nonlinear= .not.(shape%degree==1 &
         .and. cell_family(shape)==FAMILY_SIMPLEX &
         .and. any(shape%type==[ELEMENT_LAGRANGIAN,ELEMENT_DISCONTINUOUS_LAGRANGIAN]))

    dim=X%dim

#ifdef DDEBUG
    if (present(detwei)) then
       assert(size(detwei)==lx_shape%ngi)
    end if
    !if (present(dshape)) then
       assert(size(dshape,1)==shape%ndof)
       assert(size(dshape,2)==shape%ngi)
       assert(size(dshape,3)==dim)
    !end if
#endif

    if ((.not.x_nonlinear).and.cache_transform_elements) then
       cache_valid=retrieve_cached_transform(X, ele, J_local_T, invJ_local,&
            & detJ_local)

       if (cache_valid) then
          if (m_nonlinear) then
             do gi=1,lx_shape%ngi
                dshape(:,gi,:)&
                     =matmul(shape%dn(:,gi,:),transpose(invJ_local))
             end do
          else
             dn_local=matmul(shape%dn(:,1,:),transpose(invJ_local))
             forall(gi=1:lx_shape%ngi)
                dshape(:,gi,:)=dn_local
             end forall
          end if

          if (present(J)) then
             J_local_T=transpose(J_local_T)
             forall(gi=1:lx_shape%ngi)
                J(:,:,gi)=J_local_T
             end forall
          end if
          if (present(invJ)) then
             forall(gi=1:lx_shape%ngi)
                invJ(:,:,gi)=invJ_local
             end forall
          end if
          if(present(detJ)) then
             detJ=detJ_local
          end if
          if (present(detwei)) then
             detwei=abs(detJ_local)*lx_shape%quadrature%weight
          end if

          return
       end if
    else
      cache_valid = .false.
    end if

    X_val=ele_val(X, ele)

    ! Loop over quadrature points.
    quad_loop: do gi=1,lx_shape%ngi

       if ((x_nonlinear.or.gi==1).and..not.cache_valid) then
          ! For linear space elements only calculate Jacobian once.

          !     |- dx  dx  dx  -|
          !     |  dL1 dL2 dL3  |
          !     |               |
          !     |  dy  dy  dy   |
          ! J = |  dL1 dL2 dL3  |
          !     |               |
          !     |  dz  dz  dz   |
          !     |- dL1 dL2 dL3 -|

          ! Form Jacobian.
          J_local_T=matmul(X_val(:,:), lx_shape%dn(:, gi, :))

          select case (dim)
          case(1)
             invJ_local=1.0
          case(2)
             invJ_local=reshape((/ J_local_T(2,2),-J_local_T(1,2),&
                  &               -J_local_T(2,1), J_local_T(1,1)/),(/2,2/))
          case(3)
             ! Calculate (scaled) inverse using recursive determinants.
             forall (i=1:3,k=1:3)
                invJ_local(i, k)= &
                     J_local_T(cyc3(i+1),cyc3(k+1))*J_local_T(cyc3(i+2),cyc3(k+2)) &
                  -J_local_T(cyc3(i+2),cyc3(k+1))*J_local_T(cyc3(i+1),cyc3(k+2))
             end forall
          case default
             FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
          end select

          ! Form determinant by expanding minors.
          detJ_local=dot_product(J_local_T(:,1),invJ_local(:,1))

          ! Scale inverse by determinant.
          invJ_local=invJ_local/detJ_local

       end if

       ! Evaluate derivatives in physical space.
       ! If both space and the derivatives are linear then we only need
       ! to do this once.
       if (x_nonlinear.or.m_nonlinear.or.gi==1) then
          do i=1,shape%ndof
             dshape(i,gi,:)=matmul(invJ_local, shape%dn(i,gi,:))
          end do
       else
          dshape(:,gi,:)=dshape(:,1,:)
       end if

       ! Calculate transformed quadrature weights.
       if (present(detwei)) then
          detwei(gi)=abs(detJ_local)*lx_shape%quadrature%weight(gi)
       end if

       ! Copy the Jacobian and related variables to externally accessible memory
       if (present(J)) then
          if (x_nonlinear.or.gi==1) then
             J(:,:,gi)    = transpose(J_local_T(:,:))
          else
             J(:,:,gi)    = J(:,:,1)
          end if
       end if
       if (present(invJ))  invJ(:,:,gi) = invJ_local(:,:)
       if (present(detJ))  detJ(gi)     = detJ_local

    end do quad_loop

  end subroutine transform_to_physical_full

  subroutine transform_to_physical_detwei(X, ele, detwei)
    !!< Fast version of transform_to_physical that only calculates detwei

    !! Coordinate field
    type(vector_field), intent(in) :: X
    !! Current element
    integer :: ele
    !! Quadrature weights for physical coordinates.
    real, dimension(:), intent(out):: detwei(:)

    !! Shape function used for coordinate interpolation
    type(element_type), pointer :: x_shape
    !! Column n of X is the position of the nth node. (dim x x_shape%ndof)
    !! only need position of n nodes since Jacobian is only calculated once
    real, dimension(X%dim,ele_loc(X,ele)) :: X_val

    real :: J(X%dim, mesh_dim(X)), det
    integer :: gi, dim, ldim
    logical :: x_nonlinear, cache_valid

    x_shape=>ele_shape(X, ele)

    ! Optimisation checks. Optimisations apply to linear elements.
    x_nonlinear= .not.(x_shape%degree==1 .and. cell_family(x_shape)==FAMILY_SIMPLEX)

    dim=X%dim ! dimension of space (n/o real coordinates)
    ldim=size(x_shape%dn,3) ! dimension of element (n/o local coordinates)
    if (dim==ldim) then

       if ((.not.x_nonlinear).and.cache_transform_elements) then
          cache_valid=retrieve_cached_transform(X, ele, det)

          if (cache_valid) then
             detwei=abs(det)*x_shape%quadrature%weight
             return
          end if

       end if

!#ifdef DDEBUG
!       if (ele==1) then
!          ewrite(2,*) "Element geometry cache not used."
!       end if
!#endif

       X_val=ele_val(X, ele)

       select case (dim)
       case (1)
         do gi=1, x_shape%ngi
           J(1,1)=dot_product(X_val(1,:), x_shape%dn(:,gi,1))
           detwei(gi)=abs(J(1,1))*x_shape%quadrature%weight(gi)
         end do
       case (2)
         do gi=1, x_shape%ngi
            if (x_nonlinear.or.gi==1) then
               ! the Jacobian is the transpose of this
               J=matmul(X_val(:,:), x_shape%dn(:, gi, :))
               ! but that doesn't matter for determinant:
               det=abs(J(1,1)*J(2,2)-J(1,2)*J(2,1))
            end if
           detwei(gi)=det*x_shape%quadrature%weight(gi)
         end do
       case (3)
         do gi=1, x_shape%ngi
            if (x_nonlinear.or.gi==1) then
               ! the Jacobian is the transpose of this
               J=matmul(X_val(:,:), x_shape%dn(:, gi, :))
               ! but that doesn't matter for determinant:
               det=abs( &
                    J(1,1)*(J(2,2)*J(3,3)-J(2,3)*J(3,2)) &
                    -J(1,2)*(J(2,1)*J(3,3)-J(2,3)*J(3,1)) &
                    +J(1,3)*(J(2,1)*J(3,2)-J(2,2)*J(3,1)) &
                    )
            end if
            detwei(gi)= det *x_shape%quadrature%weight(gi)
         end do
       case default
          FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
       end select
    else if (ldim<dim) then

       X_val=ele_val(X, ele)

       ! lower dimensional element (ldim) embedded in higher dimensional space (dim)
       select case (ldim)
       case (1)
          ! 1-dim element embedded in 'dim'-dimensional space:
          do gi=1, x_shape%ngi
             if (x_nonlinear.or.gi==1) then
                ! J is 'dim'-dimensional vector:
                J(:,1)=matmul(X_val(:,:), x_shape%dn(:,gi,1))
                ! length of that
                det=norm2(J(:,1))
             end if
             ! length of that times quad. weight
             detwei(gi)=det*x_shape%quadrature%weight(gi)
          end do
       case (2)
          ! 2-dim element embedded in 'dim'-dimensional space:
          do gi=1, x_shape%ngi
             if (x_nonlinear.or.gi==1) then
                ! J is 2 columns of 2 'dim'-dimensional vectors:
                J=matmul(X_val(:,:), x_shape%dn(:,gi,:))
                ! outer product
                det=abs( &
                     J(2,1)*J(3,2)-J(3,1)*J(2,2) &
                     -J(3,1)*J(1,2)+J(1,1)*J(3,2) &
                     +J(1,1)*J(2,2)-J(2,1)*J(1,2))
             end if
             ! outer product times quad. weight
             detwei(gi)=det *x_shape%quadrature%weight(gi)
          end do
       end select

    else
       FLAbort("Don't know how to compute higher-dimensional elements in a lower-dimensional space.")

    end if

  end subroutine transform_to_physical_detwei

  subroutine compute_inverse_jacobian(X, x_shape, invJ, detwei, detJ)
    !!< Fast version of transform_to_physical that only calculates detwei and invJ

    !! Column n of X is the position of the nth node. (dim x x_shape%ndof)
    !! only need position of n nodes since Jacobian is only calculated once
    real, dimension(:,:), intent(in) :: X
    !! Shape function used for coordinate interpolation
    type(element_type), intent(in) :: x_shape

    !! Inverse of the jacobian matrix at each quadrature point (dim x dim x x_shape%ngi)
    !! Facilitates access to this information externally
    real, dimension(size(X,1),size(X,1),x_shape%ngi), intent(out) :: invJ
    !! Quadrature weights for physical coordinates.
    real, dimension(:), optional, intent(out) :: detwei (:)
    !! Determinant of the Jacobian at each quadrature point (x_shape%ngi)
    !! Facilitates access to this information externally
    real, dimension(:), intent(out), optional :: detJ

    !! Local versions of the Jacobian matrix and its inverse. (dim x dim)
    real, dimension(size(X,1),size(X,1)) :: J_local
    !! Local version of the determinant of J
    real :: detJ_local(x_shape%ngi)

    integer gi, i, k, dim, compute_ngi

    dim=size(X,1)

    assert(size(X,2)==x_shape%ndof)
    if (present(detwei)) then
      assert(size(detwei)==x_shape%ngi)
    end if
    if (present(detJ)) then
      assert(size(detJ)==x_shape%ngi)
    end if

    if (.not.(x_shape%degree==1 .and. cell_family(x_shape)==FAMILY_SIMPLEX)) then
      ! for non-linear compute on all gauss points
      compute_ngi=x_shape%ngi
    else
      ! for linear: compute only the first and copy the rest
      compute_ngi=1
    end if

    select case (dim)
    case(1)

      do gi=1,compute_ngi
        J_local(1,1)=dot_product(X(1,:), x_shape%dn(:, gi, 1))
        detJ_local(gi)=J_local(1,1)
        invJ(1,1,gi)=1.0/detJ_local(gi)
      end do
      ! copy the rest
      do gi=compute_ngi+1, x_shape%ngi
        detJ_local(gi) = detJ_local(1)
        invJ(1,1,gi)=invJ(1,1,1)
      end do

    case(2)

      do gi=1, compute_ngi

        J_local=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
        ! Form determinant by expanding minors.
        detJ_local(gi)=J_local(1,1)*J_local(2,2)-J_local(2,1)*J_local(1,2)
        invJ(:,:,gi)=reshape((/ J_local(2,2),-J_local(2,1), &
           &               -J_local(1,2), J_local(1,1)/),(/2,2/)) &
                      / detJ_local(gi)
      end do
      ! copy the rest
      do gi=compute_ngi+1, x_shape%ngi
        detJ_local(gi) = detJ_local(1)
        invJ(:,:,gi)=invJ(:,:,1)
      end do

    case(3)

      do gi=1,x_shape%ngi

        J_local=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
        ! Calculate (scaled) inverse using recursive determinants.
        forall (i=1:dim,k=1:dim)
          invJ(k,i,gi)=J_local(cyc3(i+1),cyc3(k+1))*J_local(cyc3(i+2),cyc3(k+2)) &
              -J_local(cyc3(i+2),cyc3(k+1))*J_local(cyc3(i+1),cyc3(k+2))
        end forall
        ! Form determinant by expanding minors.
        detJ_local(gi)=dot_product(J_local(1,:), invJ(:,1,gi))

        ! Scale inverse by determinant.
        invJ(:,:,gi)=InvJ(:,:,gi)/detJ_local(gi)

      end do

      ! copy the rest
      do gi=compute_ngi+1, x_shape%ngi
        detJ_local(gi) = detJ_local(1)
        invJ(:,:,gi)=invJ(:,:,1)
      end do

    case default
      FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
    end select

    if (present(detJ)) then
      detJ = detJ_local
    end if

    if (present(detwei)) then
      detwei = abs(detJ_local)*x_shape%quadrature%weight
    end if

  end subroutine compute_inverse_jacobian

  subroutine compute_jacobian(X, x_shape, J, detwei, detJ)
    !!< Fast version of transform_to_physical that only calculates detwei and J

    !! Column n of X is the position of the nth node. (dim x x_shape%ndof)
    !! only need position of n nodes since Jacobian is only calculated once
    real, dimension(:,:), intent(in) :: X
    !! Shape function used for coordinate interpolation
    type(element_type), intent(in) :: x_shape

    !! Jacobian matrix at each quadrature point (dim x dim x x_shape%ngi)
    !! Facilitates access to this information externally
    real, dimension(x_shape%dim,size(X,1),x_shape%ngi), intent(out) :: J
    !! Quadrature weights for physical coordinates.
    real, dimension(:), optional, intent(out) :: detwei (:)
    !! Determinant of the Jacobian at each quadrature point (x_shape%ngi)
    !! Facilitates access to this information externally
    real, dimension(:), intent(out), optional :: detJ

    !! Local version of the determinant of J
    real :: detJ_local(x_shape%ngi)

    integer gi, dim, ldim, compute_ngi

    dim=size(X,1) ! dimension of space
    ldim=x_shape%dim ! dimension of element

    assert(size(X,2)==x_shape%ndof)
    if (present(detwei)) then
      assert(size(detwei)==x_shape%ngi)
    end if
    if (present(detJ)) then
      assert(size(detJ)==x_shape%ngi)
    end if

    if (.not.(x_shape%degree==1 .and. cell_family(x_shape)==FAMILY_SIMPLEX)) then
      ! for non-linear compute on all gauss points
      compute_ngi=x_shape%ngi
    else
      ! for linear: compute only the first and copy the rest
      compute_ngi=1
    end if

    select case (dim)
    case(1)

      do gi=1,compute_ngi
        J(1,1,gi)=dot_product(X(1,:), x_shape%dn(:, gi, 1))
        detJ_local(gi)=J(1,1,gi)
      end do
      ! copy the rest
      do gi=compute_ngi+1, x_shape%ngi
        J(1,1,gi)=J(1,1,1)
        detJ_local(gi)=detJ_local(1)
      end do

    case(2)

      select case(ldim)
      case(1)
         do gi=1,compute_ngi
            J(:,:,gi)=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
            detJ_local(gi)=sum(sqrt(abs(J(:,:,gi))))
         end do
      case(2)
         do gi=1, compute_ngi
            J(:,:,gi)=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
            ! Form determinant by expanding minors.
            detJ_local(gi)=J(1,1,gi)*J(2,2,gi)-J(2,1,gi)*J(1,2,gi)
         end do
      case default
         FLAbort("oh dear, dimension of element > spatial dimension")
      end select

      ! copy the rest
      do gi=compute_ngi+1, x_shape%ngi
        J(:,:,gi)=J(:,:,1)
        detJ_local(gi)=detJ_local(1)
      end do

   case(3)

      select case(ldim)
      case(1)
         do gi=1,compute_ngi
            J(:,:,gi)=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
            detJ_local(gi)=sqrt(sum(J(:, :, gi)**2))
         end do
      case(2)
         do gi=1,compute_ngi
            J(:,:,gi)=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
            detJ_local(gi)=sqrt((J(1,2,gi)*J(2,3,gi)-J(1,3,gi)*J(2,2,gi))**2+ &
                           (J(1,3,gi)*J(2,1,gi)-J(1,1,gi)*J(2,3,gi))**2+ &
                           (J(1,1,gi)*J(2,2,gi)-J(1,2,gi)*J(2,1,gi))**2)

         end do
      case(3)
         do gi=1,compute_ngi
            J(:,:,gi)=transpose(matmul(X(:,:), x_shape%dn(:, gi, :)))
            detJ_local(gi)=det_3(J(:,:,gi))
         end do
      case default
         FLAbort("oh dear, dimension of element > spatial dimension")
      end select

      ! copy the rest
      do gi=compute_ngi+1, x_shape%ngi
        J(:,:,gi)=J(:,:,1)
        detJ_local(gi)=detJ_local(1)
      end do

    case default
      FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
    end select

    if (present(detJ)) then
      detJ=detJ_local
    end if

    if (present(detwei)) then
      detwei = abs(detJ_local)*x_shape%quadrature%weight
    end if

  end subroutine compute_jacobian

  subroutine transform_facet_to_physical_full(X, face, detwei_f, normal)

    ! Coordinate transformations for facet integrals.
    ! Calculate the transformed quadrature
    ! weights as a side bonus.
    !
    ! For facet integrals, we also need to know the facet outward
    ! pointing normal.
    !
    ! In this case it is only the determinant of the Jacobian which is
    ! required.

    ! Column n of X is the position of the nth node of the adjacent element
    ! (this is only used to work out the orientation of the boundary)
    type(vector_field), intent(in) :: X
    ! The face to transform.
    integer, intent(in) :: face
    ! Quadrature weights for physical coordinates for integration over the boundary.
    real, dimension(:), intent(out), optional :: detwei_f
    ! Outward normal vector. (dim x x_shape_f%ngi)
    real, dimension(:,:), intent(out) :: normal


    ! Column n of X_f is the position of the nth node on the facet.
    real, dimension(X%dim,face_loc(X,face)) :: X_f
    ! Column n of X_f is the position of the nth node on the facet.
    real, dimension(X%dim,ele_loc(X,face_ele(X,face))) :: X_val
    ! shape function coordinate interpolation on the boundary
    type(element_type), pointer :: x_shape_f


    ! Jacobian matrix and its inverse.
    real, dimension(X%dim,mesh_dim(X)-1) :: J
    ! Determinant of J
    real :: detJ
    ! Whether the cache can be used
    logical :: cache_valid


    integer :: gi, i, compute_ngi

    x_shape_f=>face_shape(X,face)

#ifdef DDEBUG
    assert(size(normal,1)==X%dim)
#endif
#ifdef DDEBUG
    if (present(detwei_f)) then
       assert(size(detwei_f)==x_shape_f%ngi)
    end if
#endif

    if (.not.(x_shape_f%degree==1 .and. cell_family(x_shape_f)==FAMILY_SIMPLEX)) then
      ! for non-linear compute on all gauss points
      compute_ngi=x_shape_f%ngi
      cache_valid=.false.
    else
      ! for linear: compute only the first and copy the rest
      if (cache_transform_elements) then
         cache_valid=retrieve_cached_face_transform(X, face, normal(:,1),&
              & detJ)
      else
         cache_valid=.false.
      end if
      if (cache_valid) then
         compute_ngi=0
      else
         compute_ngi=1
      end if

    end if

    if (.not.cache_valid) then
       X_val=ele_val(X, face_ele(X,face))
       X_f=face_val(X, face)
    end if

    ! Loop over quadrature points.
    quad_loop: do gi=1, compute_ngi

       !     |- dx  dx  -|
       !     |  dL1 dL2  |
       !     |           |
       !     |  dy  dy   |
       ! J = |  dL1 dL2  |
       !     |           |
       !     |  dz  dz   |
       !     |- dL1 dL2 -|

       ! Form Jacobian.
       J=matmul(X_f(:,:), x_shape_f%dn(:, gi, :))

       detJ=0.0
       ! Calculate determinant.
       select case (mesh_dim(X))
       case(1)
          detJ=1.0
       case(2)
          select case (X%dim)
          case(2)
             detJ = sqrt(J(1,1)**2 + J(2,1)**2)
          case(3)
             detJ = sqrt(sum(J(:,1)**2))
          case default
             FLAbort("Unsupported dimension specified")
          end select
       case(3)
          select case (X%dim)
          case(3)
             do i=1,3
                detJ=detJ+ &
                     (J(cyc3(i+2),1)*J(cyc3(i+1),2)-J(cyc3(i+2),2)*J(cyc3(i&
                     &+1),1))**2
             end do
             detJ=sqrt(detJ)
          case default
             FLAbort("Unsupported dimension specified")
          end select
       case default
          FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
       end select

       ! Calculate transformed quadrature weights.
       if(present(detwei_f)) then
          detwei_f(gi)=detJ*x_shape_f%quadrature%weight(gi)
       end if
       ! Calculate normal.
       normal(:,gi)=normgi(X_val,X_f,J)

    end do quad_loop

    ! copy the value at gi==1 to the rest of the gauss points
    if(present(detwei_f)) then
      do gi=compute_ngi+1, x_shape_f%ngi
         ! uses detJ from above
         detwei_f=detJ*x_shape_f%quadrature%weight
      end do
    end if

    do gi=compute_ngi+1, x_shape_f%ngi
       normal(:,gi)=normal(:,1)
    end do

  end subroutine transform_facet_to_physical_full

  function NORMGI(X, X_f, J)
    ! Calculate the normal at a given quadrature point,
    real, dimension(:,:), intent(in) :: J
    real, dimension(size(J,1)) :: normgi
    ! Element and normal node locations respectively.
    real, dimension (:,:), intent(in) :: X, X_f
    ! Facet Jacobian.

    ! Outward pointing not necessarily normal vector.
    real, dimension(3) :: outv

    integer :: ldim

    ldim = size(J,1)

    ! Outv is the vector from the element centroid to the facet centroid.
    outv(1:ldim) = sum(X_f,2)/size(X_f,2)-sum(X,2)/size(X,2)

    select case (ldim)
    case(1)
       normgi = 1.0
    case (2)
       normgi = (/ -J(2,1), J(1,1) /)
    case (3)
       normgi=cross_product(J(:,1),J(:,2))
    case default
       FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
    end select

    ! Set correct orientation.
    normgi=normgi*dot_product(normgi, outv(1:ldim) )

    ! normalise
    normgi=normgi/sqrt(sum(normgi**2))

  contains

    function cross_product(vector1,vector2) result (prod)
      real, dimension(3) :: prod
      real, dimension(3), intent(in) :: vector1, vector2

      integer :: i

      forall(i=1:3)
         prod(i)=vector1(cyc3(i+1))*vector2(cyc3(i+2))&
              -vector1(cyc3(i+2))*vector2(cyc3(i+1))
      end forall

    end function cross_product

  end function NORMGI

  subroutine transform_facet_to_physical_detwei(X, face, detwei_f)

    ! Coordinate transformed quadrature weights for facet integrals.
    !
    type(vector_field), intent(in) :: X
    ! The face to transform.
    integer, intent(in) :: face
    ! Quadrature weights for physical coordinates for integration over the boundary.
    real, dimension(:), intent(out), optional :: detwei_f

    ! Column n of X_f is the position of the nth node on the facet.
    real, dimension(X%dim,face_loc(X,face)) :: X_f
    ! Column n of X_f is the position of the nth node on the facet.
    real, dimension(X%dim,ele_loc(X,face_ele(X,face))) :: X_val
    ! shape function coordinate interpolation on the boundary
    type(element_type), pointer :: x_shape_f


    ! Jacobian matrix and its inverse.
    real, dimension(X%dim,X%dim-1) :: J
    ! Determinant of J
    real :: detJ
    ! Whether the cache can be used
    logical :: cache_valid
    ! Outward normal vector. This is a dummy.
    real, dimension(X%dim) :: lnormal


    integer :: gi, i, compute_ngi

    x_shape_f=>face_shape(X,face)

#ifdef DDEBUG
    if (present(detwei_f)) then
       assert(size(detwei_f)==x_shape_f%ngi)
    end if
#endif

    if (.not.(x_shape_f%degree==1 .and. cell_family(x_shape_f)==FAMILY_SIMPLEX)) then
      ! for non-linear compute on all gauss points
      compute_ngi=x_shape_f%ngi
      cache_valid=.false.
    else
      ! for linear: compute only the first and copy the rest
      if (cache_transform_elements) then
         cache_valid=retrieve_cached_face_transform(X, face, lnormal(:),&
              & detJ)
      else
         cache_valid=.false.
      end if
      if (cache_valid) then
         compute_ngi=0
      else
         compute_ngi=1
      end if

    end if

    if (.not.cache_valid) then
       X_val=ele_val(X, face_ele(X,face))
       X_f=face_val(X, face)
    end if

    ! Loop over quadrature points.
    quad_loop: do gi=1, compute_ngi

       !     |- dx  dx  -|
       !     |  dL1 dL2  |
       !     |           |
       !     |  dy  dy   |
       ! J = |  dL1 dL2  |
       !     |           |
       !     |  dz  dz   |
       !     |- dL1 dL2 -|

       ! Form Jacobian.
       J=matmul(X_f(:,:), x_shape_f%dn(:, gi, :))

       detJ=0.0
       ! Calculate determinant.
       select case (mesh_dim(X))
       case(1)
          detJ=1.0
       case(2)
          select case (X%dim)
          case(2)
             detJ = sqrt(J(1,1)**2 + J(2,1)**2)
          case(3)
             detJ = sqrt(sum(J(:,1)**2))
          case default
             FLAbort("Unsupported dimension specified")
          end select
       case(3)
          select case (X%dim)
          case(3)
             do i=1,3
                detJ=detJ+ &
                     (J(cyc3(i+2),1)*J(cyc3(i+1),2)-J(cyc3(i+2),2)*J(cyc3(i&
                     &+1),1))**2
             end do
             detJ=sqrt(detJ)
          case default
             FLAbort("Unsupported dimension specified")
          end select
       case default
          FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
       end select

       ! Calculate transformed quadrature weights.
       if(present(detwei_f)) then
          detwei_f(gi)=detJ*x_shape_f%quadrature%weight(gi)
       end if

    end do quad_loop

    ! copy the value at gi==1 to the rest of the gauss points
    if(present(detwei_f)) then
      do gi=compute_ngi+1, x_shape_f%ngi
         ! uses detJ from above
         detwei_f=detJ*x_shape_f%quadrature%weight
      end do
    end if

  end subroutine transform_facet_to_physical_detwei

  subroutine transform_cvsurf_to_physical(X, x_shape, detwei, normal, cvfaces)
    !!< Coordinate transformations for control volume surface integrals.
    !!< Calculates the quadrature weights and unorientated face normals as a side bonus.

    ! Column n of X is the position of the nth node on the facet.
    real, dimension(:,:), intent(in) :: X
    ! Reference coordinate control volume surface element:
    type(element_type), intent(in) :: x_shape
    ! Quadrature weights for physical coordinates.
    real, dimension(:), intent(out) :: detwei
    ! face normals - not necessarily correctly orientated
    real, dimension(:,:), intent(out) :: normal
    ! control volume face information - allows optimisation
    type(cv_faces_type), intent(in) :: cvfaces

    ! Jacobian matrix
    real, dimension(size(X,1),x_shape%cell%dimension) :: J

    ! Determinant of J
    real :: detJ

    integer :: gi, i, dim, ggi, face
    logical :: x_nonlinear

    dim=size(X,1)

    assert(size(detwei)==x_shape%ngi)

    ! Optimisation checks. Optimisations apply to linear elements.
    x_nonlinear= .not.(x_shape%degree==1.and.cell_family(x_shape)==FAMILY_SIMPLEX)

    face_loop: do face = 1, cvfaces%faces

      quad_loop: do gi = 1, cvfaces%shape%ngi

        ! global gauss pt index
        ggi = (face-1)*cvfaces%shape%ngi + gi

        ! assemble the jacobian...
        ! this needs to be done at every gauss point if space is nonlinear
        ! but if space is linear then it only needs to be done at the
        ! first gauss point of each cv face
        if(x_nonlinear.or.(gi==1)) then

          !     |- dx  dx  -|
          !     |  dL1 dL2  |
          !     |           |
          !     |  dy  dy   |
          ! J = |  dL1 dL2  |
          !     |           |
          !     |  dz  dz   |
          !     |- dL1 dL2 -|

          ! Form Jacobian.
          J=matmul(X(:,:), x_shape%dn(:, ggi, :))

          detJ=0.0
          ! Calculate determinant.
          select case (dim)
          case(1)
              detJ=1.0
          case(2)
              detJ = sqrt(J(1,1)**2 + J(2,1)**2)
          case(3)
              do i=1,3
                detJ=detJ+ &
                      (J(cyc3(i+2),1)*J(cyc3(i+1),2)-J(cyc3(i+2),2)*J(cyc3(i+1),1))**2
              end do
              detJ=sqrt(detJ)
          case default
              FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
          end select
        end if

        ! Calculate transformed quadrature weights.
        detwei(ggi)=detJ*x_shape%quadrature%weight(ggi)

        ! find the normal...
        ! this needs to be done at every gauss point if space is nonlinear
        ! otherwise it only needs to be done for the first gauss point on each
        ! cv face - other faces have their normals set to that of the first gauss
        ! point of the same face
        if(x_nonlinear.or.(gi==1)) then
          normal(:,ggi)=normgi(J)
        else
          normal(:,ggi)=normal(:,(face-1)*cvfaces%shape%ngi+1)
        end if

      end do quad_loop

    end do face_loop

    contains

    function normgi(J)
      ! Calculate the normal at a given quadrature point,
      ! Control volume surface Jacobian.
      real, dimension(:,:), intent(in) :: J
      real, dimension(size(J,1)) :: normgi

      select case (dim)
      case(1)
         normgi = 1.0
      case (2)
         normgi = (/ -J(2,1), J(1,1) /)
      case (3)
         normgi=cross_product(J(:,1),J(:,2))
      case default
         FLAbort('Unsupported dimension selected.')
      end select

    end function normgi

    pure function cross_product(vector1,vector2)
      real, dimension(3) :: cross_product
      real, dimension(3), intent(in) :: vector1, vector2

      integer :: i

      forall(i=1:3)
         cross_product(i)=vector1(cyc3(i+1))*vector2(cyc3(i+2))&
                         -vector1(cyc3(i+2))*vector2(cyc3(i+1))
      end forall

    end function cross_product

  end subroutine transform_cvsurf_to_physical

  subroutine transform_cvsurf_facet_to_physical(X, X_f, x_shape_f, &
       normal, detwei)

    ! Coordinate transformations for facet integrals around control volumes.
    ! Calculate the transformed quadrature
    ! weights as a side bonus.
    !
    ! For facet integrals, we also need to know the facet outward
    ! pointing normal.
    !
    ! In this case it is only the determinant of the Jacobian which is
    ! required.

    ! Coordinate facet element:
    type(element_type), intent(in) :: x_shape_f
    ! Column n of X is the position of the nth node.
    real, dimension(:,:), intent(in) :: X
    ! Column n of X_f is the position of the nth node on the facet.
    real, dimension(:,:), intent(in) :: X_f
    ! Quadrature weights for physical coordinates.
    real, dimension(:), intent(out), optional :: detwei
    ! Outward normal vector. (dim x x_shape_f%ngi)
    real, dimension(:,:), intent(out) :: normal

    ! Jacobian matrix and its inverse.
    real, dimension(size(X,1),x_shape_f%cell%dimension) :: J
    ! Determinant of J
    real :: detJ

    integer :: gi, i, dim

    logical :: x_nonlinear

    dim=size(X,1)
    assert(size(X_f,1)==dim)
    assert(size(X_f,2)==x_shape_f%ndof)

    assert(size(normal,1)==dim)

    if(present(detwei)) then
      assert(size(detwei)==x_shape_f%ngi)
    end if

    ! Optimisation checks. Optimisations apply to linear space elements.
    x_nonlinear= .not.(x_shape_f%degree==1.and.cell_family(x_shape_f)==FAMILY_SIMPLEX)

    ! Loop over quadrature points.
    quad_loop: do gi=1,x_shape_f%ngi

       if (x_nonlinear.or.gi==1) then
          ! For linear space elements only calculate Jacobian once.
        !     |- dx  dx  -|
        !     |  dL1 dL2  |
        !     |           |
        !     |  dy  dy   |
        ! J = |  dL1 dL2  |
        !     |           |
        !     |  dz  dz   |
        !     |- dL1 dL2 -|

        ! Form Jacobian.
        J=matmul(X_f(:,:), x_shape_f%dn(:, gi, :))

          detJ=0.0
          ! Calculate determinant.
          select case (dim)
          case(1)
              detJ=1.0
          case(2)
              detJ = sqrt(J(1,1)**2 + J(2,1)**2)
          case(3)
              do i=1,3
                detJ=detJ+ &
                      (J(cyc3(i+2),1)*J(cyc3(i+1),2)-J(cyc3(i+2),2)*J(cyc3(i+1),1))**2
              end do
              detJ=sqrt(detJ)
          case default
              FLAbort("Unsupported dimension specified.  Universe is 3 dimensional (sorry Albert).")
          end select
       end if

       ! Calculate transformed quadrature weights.
       if(present(detwei)) then
         detwei(gi)=detJ*x_shape_f%quadrature%weight(gi)
       end if

       if (x_nonlinear.or.gi==1) then
          ! Calculate normal.
          normal(:,gi)=normgi(X,X_f,J)
       else
          normal(:,gi)=normal(:,1)
       end if

    end do quad_loop

  contains

    function normgi(X, X_f, J)
      ! Calculate the normal at a given quadrature point,
      real, dimension(:,:), intent(in) :: J
      real, dimension(size(J,1)) :: normgi
      ! Element and normal node locations respectively.
      real, dimension (:,:), intent(in) :: X, X_f
      ! Facet Jacobian.

      ! Outward pointing not necessarily normal vector.
      real, dimension(3) :: outv

      integer :: ldim

      ldim = size(J,1)

      ! Outv is the vector from the element centroid to the facet centroid.
      outv(1:ldim) = sum(X_f,2)/size(X_f,2)-sum(X,2)/size(X,2)

      select case (dim)
      case(1)
         normgi = 1.0
      case (2)
         normgi = (/ -J(2,1), J(1,1) /)
      case (3)
         normgi=cross_product(J(:,1),J(:,2))
      case default
         FLAbort('Unsupported dimension selected.')
      end select

      ! Set correct orientation.
      normgi=normgi*dot_product(normgi, outv(1:ldim) )

      ! normalise
      normgi=normgi/sqrt(sum(normgi**2))

    end function normgi

    function cross_product(vector1,vector2) result (prod)
      real, dimension(3) :: prod
      real, dimension(3), intent(in) :: vector1, vector2

      integer :: i

      forall(i=1:3)
         prod(i)=vector1(cyc3(i+1))*vector2(cyc3(i+2))&
              -vector1(cyc3(i+2))*vector2(cyc3(i+1))
      end forall

    end function cross_product

  end subroutine transform_cvsurf_facet_to_physical

  subroutine transform_horizontal_to_physical(X_f, X_face_shape, vertical_normal, &
    m_f, dm_hor, detwei_hor)
    !!< Given the 'dim+1'-dimensional coordinates of a 'dim'-dimensional face
    !!< and its shape function on that face, return the inverse Jacobian
    !!< associated with the transformation between the local 'dim' coordinates
    !!< on the face augmented with an auxiliary local vertical coordinate,
    !!< and the 'dim+1' physical coordinates.
    !!< This can be used to transform derivatives of fields defined on the face
    !!< to a horizontal derivative.
    !!< Also returned is detwei_hor which can be used to perform an integration
    !!< of fields defined on the face integrated over the face projected in
    !!< the horizontal plane.
    !! NOTE: in the following nloc are the number of nodes, and ngi
    !! the number of gausspoints on the FACE
    !! positions of the nodes on the face (dim+1 x nloc)
    real, dimension(:,:):: X_f
    !! element shape used to interpolate these positions
    type(element_type), intent(in):: X_face_shape
    !! vertical normal vector at the gauss points of the face (dim+1 x ngi)
    real, dimension(:,:):: vertical_normal
    !! element shape of field on the face, you wish to transform
    type(element_type), optional, intent(in):: m_f
    !! transformed derivatives (nloc x ngi x dim+1):
    real, dimension(:,:,:), optional, intent(out):: dm_hor
    !! integration weights at gausspoint for horizontal integration (ngi):
    real, dimension(:), optional, intent(out):: detwei_hor

    real, dimension(size(X_f,1),size(X_f,1)):: J, invJ
    real det
    logical x_nonlinear
    integer i, gi, dim, cdim

    dim=X_face_shape%dim
    cdim=size(X_f,1)

    ! make sure everything is the right size:
    assert(cdim==dim+1)
    assert(size(vertical_normal,1)==cdim)
    assert(X_face_shape%quadrature%ngi==size(vertical_normal,2))
    if (present(dm_hor)) then
      assert(size(X_f,2)==size(dm_hor,1))
      assert(size(vertical_normal,2)==size(dm_hor,2))
      assert(m_f%quadrature%ngi==size(dm_hor,2))
      assert(size(dm_hor,3)==cdim)
    end if
    if (present(detwei_hor)) then
      assert(size(vertical_normal,2)==size(detwei_hor))
    end if

    ! Optimisation checks. Optimisations apply to linear elements.
    x_nonlinear= .not. (X_face_shape%degree==1 .and. cell_family(X_face_shape)==FAMILY_SIMPLEX)

    do gi=1, X_face_shape%ngi

       ! in 3 dimensions:
       !     |- dx  dx  dx  -|
       !     |  dL1 dL2 dL3  |
       !     |               |
       !     |  dy  dy  dy   |
       ! J = |  dL1 dL2 dL3  |
       !     |               |
       !     |  dz  dz  dz   |
       !     |- dL1 dL2 dL3 -|
       ! where L1 and L2 are the 2 local coordinates of the face
       ! and L3 is an auxillary vertical local coordinate.
       if (gi==1 .or. x_nonlinear) then
          ! we do follow the definition of J as above
          ! (as opposed to tranform_to_physical where J is defined as its transpose)
          J(:,1:dim)=matmul(X_f, x_face_shape%dn(:, gi, :))
          ! make extra local coordinate (L_3) in the vertical direction
          J(:,cdim)=vertical_normal(:,gi)

          if (cdim==3) then
             ! Cross product gives area spanned by local coordinate unit vectors
             ! times the surface normal. Taking dot_product with vertical normal
             ! then gives the area in the projected in the horizontal direction.
             det=abs(dot_product( J(:,3), cross_product( J(:,1), J(:,2) ) ))
          else
             ! cross product of the local coordinate unit vector and
             ! the vertical normal gives projection of the unit vector
             ! in horizontal direction.
             det=abs(J(1,1)*J(2,2)-J(1,2)*J(2,1))
          end if

       end if

       if (present(detwei_hor)) then
          detwei_hor(gi)=det*X_face_shape%quadrature%weight(gi)
       end if

       if (present(m_f)) then

          assert(present(dm_hor))

          invJ=inverse(J)

          do i=1, size(dm_hor,1)
             dm_hor(i,gi,:)=matmul( m_f%dn(i,gi,:), invJ(1:dim,:) )
             ! assume no change of in-field in vertical direction (L_3)
             ! thereby effectively taking horizontal derivative
          end do
       end if

    end do

  end subroutine transform_horizontal_to_physical

  function element_volume(position, ele)
    !!< Return the volume of element in the positions field.
    real :: element_volume
    type(vector_field), intent(in) :: position
    integer, intent(in) :: ele

    real, dimension(ele_ngi(position, ele)) :: detwei

    call transform_to_physical_detwei(position, ele, detwei)

    element_volume=sum(detwei)

  end function element_volume

end module transform_elements
