!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineeringp
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
module fields_manipulation
use elements
use element_set
use embed_python
use data_structures
use fields_data_types
use fields_base
use fields_allocates
use halo_data_types
use halos_allocates
use halos_base
use halos_debug
use halos_numbering
use halos_ownership
use halos_repair
use quicksort
use parallel_tools
use vector_tools
use iso_c_binding
use global_parameters, only: malloc

implicit none

#ifdef __INTEL_COMPILER
intrinsic sizeof
#define c_sizeof sizeof
#endif

  private

  public :: addto, set_from_function, set, set_all, &
    & set_from_python_function, remap_field, remap_field_to_surface, &
    & set_to_submesh, set_from_submesh, scale, bound, invert, &
    & absolute_value, inner_product, cross_prod, clone_header
  public :: piecewise_constant_field, piecewise_constant_mesh
  public :: renumber_positions, renumber_positions_trailing_receives, &
    & renumber_positions_elements, &
    & renumber_positions_elements_trailing_receives, reorder_element_numbering
  public :: get_patch_ele, get_patch_node, patch_type
  public :: set_ele_nodes, normalise, tensor_second_invariant
  public :: remap_to_subdomain, remap_to_full_domain
  public :: get_coordinates_remapped_to_surface, get_remapped_coordinates

  integer, parameter, public :: REMAP_ERR_DISCONTINUOUS_CONTINUOUS = 1, &
                                REMAP_ERR_HIGHER_LOWER_CONTINUOUS  = 2, &
                                REMAP_ERR_UNPERIODIC_PERIODIC      = 3, &
                                REMAP_ERR_BUBBLE_LAGRANGE          = 4

  interface addto
     module procedure scalar_field_vaddto, scalar_field_addto, &
          vector_field_addto, vector_field_vaddto_dim, tensor_field_addto, &
          tensor_field_vaddto, tensor_field_vaddto_single, tensor_field_vaddto_dim, &
          vector_field_vaddto_vec, scalar_field_addto_scalar, vector_field_addto_vector, &
          scalar_field_addto_field, vector_field_addto_field, vector_field_addto_dim, &
          vector_field_addto_field_dim, tensor_field_addto_field_dim_dim, &
          tensor_field_addto_dim, tensor_field_addto_tensor_field, &
          real_addto_real, vector_field_addto_field_scale_field
  end interface

  interface set_from_function
     module procedure set_from_function_scalar, set_from_function_vector,&
          & set_from_function_tensor
  end interface

  interface set
    module procedure set_scalar_field_node, set_scalar_field, &
                   & set_vector_field_node, set_vector_field, &
                   & set_vector_field_node_dim, set_vector_field_dim, &
                   & set_tensor_field_node, set_tensor_field, &
                   & set_scalar_field_nodes, set_scalar_field_constant_nodes, &
                   & set_tensor_field_node_dim, &
                   & set_vector_field_nodes, &
                   & set_vector_field_nodes_dim, &
                   & set_tensor_field_nodes, &
                   & set_scalar_field_field, &
                   & set_scalar_field_from_vector_field, &
                   & set_vector_field_field, &
                   & set_vector_field_field_dim, &
                   & set_tensor_field_field, &
                   & set_tensor_field_scalar_field, &
                   & set_tensor_field_diag_vector_field, &
                   & set_scalar_field_theta, set_vector_field_theta, &
                   & set_vector_field_vfield_dim, &
                   & set_tensor_field_theta
  end interface

  interface set_all
     module procedure set_vector_field_arr, set_vector_field_arr_dim, &
          & set_scalar_field_arr, set_tensor_field_arr, &
          & set_tensor_field_arr_dim
  end interface

  interface set_from_python_function
     module procedure set_from_python_function_scalar,&
          & set_from_python_function_vector, &
          & set_from_python_function_tensor
  end interface

  interface test_remap_validity
     module procedure test_remap_validity_scalar, test_remap_validity_vector, &
                      test_remap_validity_tensor, test_remap_validity_generic
  end interface

  interface remap_field
     module procedure remap_scalar_field, remap_vector_field, remap_tensor_field, &
                    & remap_scalar_field_specific, remap_vector_field_specific
  end interface

  interface remap_field_to_surface
     module procedure remap_scalar_field_to_surface, remap_vector_field_to_surface
  end interface

  interface set_to_submesh
    module procedure set_to_submesh_scalar, set_to_submesh_vector
  end interface

  interface set_from_submesh
    module procedure set_from_submesh_scalar, set_from_submesh_vector
  end interface

  interface scale
     module procedure scalar_scale, vector_scale, tensor_scale, &
          scalar_scale_scalar_field, &
          vector_scale_scalar_field, &
          tensor_scale_scalar_field, &
          vector_scale_vector_field
  end interface

  interface bound
    module procedure bound_scalar_field, bound_scalar_field_field, bound_vector_field, bound_tensor_field
  end interface

  interface invert
     module procedure invert_scalar_field, invert_vector_field, &
      invert_scalar_field_inplace, invert_vector_field_inplace
  end interface

  interface absolute_value
     module procedure absolute_value_scalar_field
  end interface

  interface inner_product
     module procedure inner_product_array_field, inner_product_field_array, &
        inner_product_field_field
  end interface inner_product

  !  This is named cross_prod rather than cross_product to avoid a name
  !  clash with various cross_product functions (this one is a subroutine).
  interface cross_prod
     module procedure cross_product_vector
  end interface cross_prod

  interface clone_header
    module procedure clone_header_scalar, clone_header_vector, clone_header_tensor
  end interface clone_header

  interface normalise
    module procedure normalise_scalar, normalise_vector
  end interface

  interface remap_to_subdomain
    module procedure remap_to_subdomain_scalar, remap_to_subdomain_vector, remap_to_subdomain_tensor
  end interface

  interface remap_to_full_domain
    module procedure remap_to_full_domain_scalar, remap_to_full_domain_vector, remap_to_full_domain_tensor
  end interface


  type patch_type
    !!< This is a type that represents a patch of elements around a given node.

    ! Really this isn't necessary, as it's just an array, but
    ! I think encapsulation is good.

    !! The number of elements around the node
    integer :: count
    !! The array of element indices surrounding the node
    integer, dimension(:), pointer :: elements
  end type patch_type


  contains

  subroutine tensor_second_invariant(t_field,second_invariant)
      !!< This routine computes the second invariant of an infield tensor field t_field.
      !!< Note - currently assumes that tensor field t_field is symmetric.
      type(tensor_field), intent(in):: t_field
      type(scalar_field), intent(inout) :: second_invariant

      type(tensor_field) :: t_field_local

      integer :: node, dim1, dim2
      real :: val

      ! Remap t_field to second invariant mesh if required:
      call allocate(t_field_local, second_invariant%mesh, "LocalTensorField")
      call remap_field(t_field, t_field_local)

      do node = 1, node_count(second_invariant)
         val = 0.
         do dim1 = 1, t_field_local%dim(1)
            do dim2 = 1, t_field_local%dim(2)
               val = val + node_val(t_field_local,dim1,dim2,node)**2
            end do
         end do
         call set(second_invariant,node,sqrt(val/2.))
      end do

      call deallocate(t_field_local)

  end subroutine tensor_second_invariant

  subroutine scalar_field_vaddto(field, node_numbers, val)
    !!< Add val to the field%val(node_numbers) for a vector of
    !!< node_numbers.
    !!<
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(size(node_numbers)), intent(in) :: val

    integer :: j

    assert(field%field_type==FIELD_TYPE_NORMAL)
    ! Note that this has to be a do loop in case i contains repeated
    ! indices.
    do j=1,size(node_numbers)
       field%val(node_numbers(j))=field%val(node_numbers(j))+val(j)
    end do

  end subroutine scalar_field_vaddto

  subroutine scalar_field_addto(field, node_number, val)
    !!< Add val to the field%val(node_number).
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    integer, intent(in) :: node_number
    real, intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    field%val(node_number)=field%val(node_number)+val

  end subroutine scalar_field_addto

  subroutine scalar_field_addto_scalar(field, val)
    !!< Add val to field%val
    !!< Works for both constant and space varying fields
    type(scalar_field), intent(inout) :: field
    real, intent(in) :: val

    assert(field%field_type/=FIELD_TYPE_PYTHON)
    field%val=field%val+val

  end subroutine scalar_field_addto_scalar

  subroutine vector_field_addto_vector(field, val)
    !!< Add val to field%val
    !!< Works for both constant and space varying fields
    type(vector_field), intent(inout) :: field
    real, dimension(field%dim), intent(in) :: val

    integer :: i

    assert(field%field_type/=FIELD_TYPE_PYTHON)
    do i = 1, field%dim
      field%val(i,:)=field%val(i,:)+val(i)
    end do

  end subroutine vector_field_addto_vector

  subroutine vector_field_addto(field, node_number, val)
    !!< Add val to the field%val(node_number).
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, intent(in) :: node_number
    real, dimension(field%dim), intent(in) :: val

    integer :: j

    assert(field%field_type==FIELD_TYPE_NORMAL)

    do j=1,field%dim
       field%val(j,node_number)=field%val(j,node_number)+val(j)
    end do

  end subroutine vector_field_addto

  subroutine vector_field_addto_dim(field, dim, node_number, val)
    !!< Add val to the field%val(node_number) only for the specified dim
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, intent(in) :: dim, node_number
    real, intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(dim,node_number)=field%val(dim,node_number)+val

  end subroutine vector_field_addto_dim

  subroutine vector_field_vaddto_dim(field, dim, node_numbers, val)
    !!< Add val to dimension dim of the field%val(node_numbers) for a
    !!< vector of node_numbers.
    !!<
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    integer, intent(in) :: dim
    real, dimension(size(node_numbers)), intent(in) :: val

    integer :: j

    assert(field%field_type==FIELD_TYPE_NORMAL)
    do j=1,size(node_numbers)
       field%val(dim,node_numbers(j))&
            =field%val(dim,node_numbers(j))+val(j)
    end do

  end subroutine vector_field_vaddto_dim

  subroutine vector_field_vaddto_vec(field, node_numbers, val)
    !!< Add val(:, node) to field for each node in node_numbers.
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(:, :), intent(in) :: val

    integer :: i
    assert(size(val, 1) == field%dim)
    assert(size(val, 2) == size(node_numbers))

    assert(field%field_type==FIELD_TYPE_NORMAL)
    do i=1,size(node_numbers)
      call addto(field, node_numbers(i), val(:, i))
    end do

  end subroutine vector_field_vaddto_vec

  subroutine tensor_field_addto(field, node_number, val)
    !!< Add val to the field%val(i).
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, intent(in) :: node_number
    real, dimension(:,:), intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    field%val(:,:,node_number)=field%val(:,:,node_number)+val

  end subroutine tensor_field_addto

  subroutine tensor_field_vaddto(field, node_numbers, val)
    !!< Add val(:,:,i) to field%val(:,:,i) for vector of node_numbers.
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(:, :, :), intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    field%val(:, :, node_numbers) = field%val(:, :, node_numbers) + val

  end subroutine tensor_field_vaddto

  subroutine tensor_field_vaddto_single(field, node_numbers, val)
    !!< Add val(:,:,node_numbers) to field%val(:,:,node_numbers) for vector
    !!< of node_numbers.
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(:, :), intent(in) :: val

    integer :: j

    assert(field%field_type==FIELD_TYPE_NORMAL)
    do j=1,size(node_numbers)
      field%val(:, :, node_numbers(j)) &
           = field%val(:, :, node_numbers(j)) + val
    end do

  end subroutine tensor_field_vaddto_single

  subroutine tensor_field_vaddto_dim(field, dim1, dim2, node_numbers, val)
    !!< Add val(node_numbers) to field%val(dim1,dim2,node_numbers) for
    !!< vector of node_numbers.
    !!<
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, intent(in) :: dim1, dim2
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(:), intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    field%val(dim1, dim2, node_numbers) &
         = field%val(dim1, dim2, node_numbers) + val

  end subroutine tensor_field_vaddto_dim

  subroutine tensor_field_addto_dim(field, dim1, dim2, node_number, val)
    !!< Add val(node_number) to field%val(dim1,dim2,node_number) for a single node_number.
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, intent(in) :: dim1, dim2
    integer, intent(in) :: node_number
    real, intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    field%val(dim1, dim2, node_number) = field%val(dim1, dim2, node_number) + val

  end subroutine tensor_field_addto_dim

  subroutine scalar_field_addto_field(field1, field2, scale)
    !!< Compute field1=field1+[scale*]field2.
    !!< Works for constant and space varying fields.
    !!< if field1%mesh/=field2%mesh map field2 to field1%mesh first
    !!< (same restrictions apply as mentioned in remap_field() )
    type(scalar_field), intent(inout) :: field1
    type(scalar_field), intent(in) :: field2
    real, intent(in), optional :: scale

    type(scalar_field) lfield2

    assert(field1%field_type/=FIELD_TYPE_PYTHON .and. field2%field_type/=FIELD_TYPE_PYTHON)

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lfield2, field1%mesh)
       call remap_field(field2, lfield2)
    else
       lfield2=field2
    end if

    if (field1%field_type==field2%field_type) then
       if (present(scale)) then
          field1%val=field1%val+scale*lfield2%val
       else
          field1%val=field1%val+lfield2%val
       end if
    else if (field1%field_type==FIELD_TYPE_NORMAL) then

       assert(field2%field_type==FIELD_TYPE_CONSTANT)
       if (present(scale)) then
          field1%val=field1%val+scale*field2%val(1)
       else
          field1%val=field1%val+field2%val(1)
       end if

    else

       FLAbort("Illegal addition for given field types.")

    end if

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call deallocate(lfield2)
    end if

  end subroutine scalar_field_addto_field

  subroutine vector_field_addto_field(field1, field2, scale)
    !!< Compute field1=field1+[scale*]field2.
    !!< Works for constant and space varying fields.
    !!< if field1%mesh/=field2%mesh map field2 to field1%mesh first
    !!< (same restrictions apply as mentioned in remap_field() )
    type(vector_field), intent(inout) :: field1
    type(vector_field), intent(in) :: field2
    real, intent(in), optional :: scale

    integer :: i

    type(vector_field) lfield2

    assert(field1%field_type/=FIELD_TYPE_PYTHON .and. field2%field_type/=FIELD_TYPE_PYTHON)
    assert(field1%dim==field2%dim)

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lfield2, field1%dim, field1%mesh)
       call remap_field(field2, lfield2)
    else
       lfield2=field2
    end if

    if (field1%field_type==field2%field_type) then
       if (present(scale)) then
          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+scale*lfield2%val(i,:)
          end do
       else
          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+lfield2%val(i,:)
          end do
       end if
    else if (field1%field_type==FIELD_TYPE_NORMAL) then

       assert(field2%field_type==FIELD_TYPE_CONSTANT)
       if (present(scale)) then
          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+scale*field2%val(i,1)
          end do
       else
          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+field2%val(i,1)
          end do
       end if
    else

       FLAbort("Illegal addition for given field types.")

    end if

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
      call deallocate(lfield2)
    end if

  end subroutine vector_field_addto_field

  subroutine vector_field_addto_field_scale_field(field1, field2, scale)
    !!< Compute field1=field1+[scale*]field2.
    !!< In this version of the routine, scale is a scalar field.
    !!< Works for constant and space varying fields.
    !!< if field1%mesh/=field2%mesh map field2 to field1%mesh first
    !!< (same restrictions apply as mentioned in remap_field() )
    type(vector_field), intent(inout) :: field1
    type(vector_field), intent(in) :: field2
    type(scalar_field), intent(in) :: scale

    integer :: i

    type(vector_field) :: lfield2
    type(scalar_field) :: lscale

    assert(field1%field_type/=FIELD_TYPE_PYTHON .and. field2%field_type/=FIELD_TYPE_PYTHON)
    assert(field1%dim==field2%dim)
    assert(lscale%field_type/=FIELD_TYPE_PYTHON)

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lfield2, field1%dim, field1%mesh)
       call remap_field(field2, lfield2)
    else
       lfield2=field2
       call incref(lfield2)
    end if

    if (.not. field1%mesh==scale%mesh .and. .not. scale%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lscale, field1%mesh)
       call remap_field(scale, lscale)
    else
       lscale=scale
       call incref(scale)
    end if

    if (field1%field_type==FIELD_TYPE_CONSTANT) then

       if ((lfield2%field_type==FIELD_TYPE_CONSTANT) .and. &
            (lscale%field_type==FIELD_TYPE_CONSTANT)) then

          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+lscale%val*lfield2%val(i,:)
          end do

       else

          FLAbort("Illegal addition for given field types.")

       end if

    else
       ! field1 is not constant.

       if ((lfield2%field_type==FIELD_TYPE_CONSTANT) .and. &
            (lscale%field_type==FIELD_TYPE_CONSTANT)) then

          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+lscale%val(1)*lfield2%val(i,1)
          end do

       else if ((lfield2%field_type==FIELD_TYPE_NORMAL) .and. &
            (lscale%field_type==FIELD_TYPE_CONSTANT)) then

          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+lscale%val(1)*lfield2%val(i,:)
          end do

       else if ((lfield2%field_type==FIELD_TYPE_CONSTANT) .and. &
            (lscale%field_type==FIELD_TYPE_NORMAL)) then

          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+lscale%val*lfield2%val(i,1)
          end do

       else if ((lfield2%field_type==FIELD_TYPE_NORMAL) .and. &
            (lscale%field_type==FIELD_TYPE_NORMAL)) then

          do i=1,field1%dim
             field1%val(i,:)=field1%val(i,:)+lscale%val*lfield2%val(i,:)
          end do

       else

          FLAbort("Illegal addition for given field types.")

       end if

    end if

    call deallocate(lfield2)
    call deallocate(lscale)

  end subroutine vector_field_addto_field_scale_field

  subroutine vector_field_addto_field_dim(field1, dim, field2, scale)
    !!< Compute field1(dim)=field1(dim)+scale*field2.
    !!< Works for constant and space varying fields.
    type(vector_field), intent(inout) :: field1
    integer, intent(in) :: dim
    type(scalar_field), intent(in) :: field2
    real, intent(in), optional :: scale

    type(scalar_field) lfield2

    assert(field1%field_type/=FIELD_TYPE_PYTHON .and. field2%field_type/=FIELD_TYPE_PYTHON)
    ! only allow addition to non-constant field1 or
    ! addition of constant field1 and constant field2
    assert(field1%field_type==FIELD_TYPE_NORMAL .or. field2%field_type==FIELD_TYPE_CONSTANT)

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lfield2, field1%mesh)
       call remap_field(field2, lfield2)
    else
       lfield2=field2
    end if

    if (field1%field_type==field2%field_type) then
       if (present(scale)) then
          field1%val(dim,:)=field1%val(dim,:)+scale*lfield2%val
       else
          field1%val(dim,:)=field1%val(dim,:)+lfield2%val
       end if
    else if (field1%field_type==FIELD_TYPE_NORMAL) then

       assert(field2%field_type==FIELD_TYPE_CONSTANT)
       if (present(scale)) then
          field1%val(dim,:)=field1%val(dim,:)+scale*field2%val(1)
       else
          field1%val(dim,:)=field1%val(dim,:)+field2%val(1)
       end if
    else

       FLAbort("Illegal addition for given field types.")

    end if

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call deallocate(lfield2)
    end if

  end subroutine vector_field_addto_field_dim

  subroutine tensor_field_addto_field_dim_dim(field1, dim1, dim2, field2, scale)
    !!< Compute field1(dim1,dim2)=field1(dim1,dim2)+scale*field2.
    !!< Works for constant and space varying fields.
    type(tensor_field), intent(inout) :: field1
    integer, intent(in) :: dim1, dim2
    type(scalar_field), intent(in) :: field2
    real, intent(in), optional :: scale

    type(scalar_field) lfield2

    assert(field1%field_type/=FIELD_TYPE_PYTHON .and. field2%field_type/=FIELD_TYPE_PYTHON)
    ! only allow addition to non-constant field1 or
    ! addition of constant field1 and constant field2
    assert(field1%field_type==FIELD_TYPE_NORMAL .or. field2%field_type==FIELD_TYPE_CONSTANT)

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lfield2, field1%mesh)
       call remap_field(field2, lfield2)
    else
       lfield2=field2
    end if

    if (field1%field_type==field2%field_type) then
       if (present(scale)) then
          field1%val(dim1,dim2,:)=field1%val(dim1,dim2,:)+scale*lfield2%val
       else
          field1%val(dim1,dim2,:)=field1%val(dim1,dim2,:)+lfield2%val
       end if
    else if (field1%field_type==FIELD_TYPE_NORMAL) then

       assert(field2%field_type==FIELD_TYPE_CONSTANT)
       if (present(scale)) then
          field1%val(dim1,dim2,:)=field1%val(dim1,dim2,:)+scale*field2%val(1)
       else
          field1%val(dim1,dim2,:)=field1%val(dim1,dim2,:)+field2%val(1)
       end if
    else

       FLAbort("Illegal addition for given field types.")

    end if

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call deallocate(lfield2)
    end if

  end subroutine tensor_field_addto_field_dim_dim

  subroutine tensor_field_addto_tensor_field(field1, field2, scale)
    !!< Compute field1(dim1,dim2)=field1(dim1,dim2)+scale*field2.
    !!< Works for constant and space varying fields.
    type(tensor_field), intent(inout) :: field1
    type(tensor_field), intent(in) :: field2
    real, intent(in), optional :: scale
    integer :: i

    type(tensor_field) lfield2

    assert(field1%field_type/=FIELD_TYPE_PYTHON .and. field2%field_type/=FIELD_TYPE_PYTHON)
    ! only allow addition to non-constant field1 or
    ! addition of constant field1 and constant field2
    assert(field1%field_type==FIELD_TYPE_NORMAL .or. field2%field_type==FIELD_TYPE_CONSTANT)

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call allocate(lfield2, field1%mesh)
       call remap_field(field2, lfield2)
    else
       lfield2=field2
    end if

    if (field1%field_type==field2%field_type) then
       if (present(scale)) then
          field1%val(:,:,:)=field1%val(:, :,:) +scale*lfield2%val(:, :, :)
       else
          field1%val=field1%val+lfield2%val
       end if
    else if (field1%field_type==FIELD_TYPE_NORMAL) then

       assert(field2%field_type==FIELD_TYPE_CONSTANT)
       if (present(scale)) then
          forall(i=1:size(field1%val, 3))
            field1%val(:, :, i)=field1%val(:, :, i)+scale*field2%val(:, :, 1)
          end forall
       else
          forall(i=1:size(field1%val, 3))
            field1%val(:, :, i)=field1%val(:, :, i)+field2%val(:, :, 1)
          end forall
       end if
    else

       FLAbort("Illegal addition for given field types.")

    end if

    if (.not. field1%mesh==field2%mesh .and. .not. field2%field_type==FIELD_TYPE_CONSTANT) then
       call deallocate(lfield2)
    end if

  end subroutine tensor_field_addto_tensor_field

  subroutine real_addto_real(arr, idx, val)
    ! Real recognize real, dunn. Fo' life.

    real, dimension(:), intent(inout) :: arr
    integer, dimension(:), intent(in) :: idx
    real, dimension(size(idx)), intent(in) :: val

    arr(idx) = arr(idx) + val
  end subroutine real_addto_real

  subroutine set_scalar_field_field(out_field, in_field)
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(scalar_field), intent(inout) :: out_field
    type(scalar_field), intent(in) :: in_field

    assert(mesh_compatible(out_field%mesh, in_field%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
    assert(out_field%field_type==FIELD_TYPE_NORMAL .or. in_field%field_type==FIELD_TYPE_CONSTANT)

    select case (in_field%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val=in_field%val
    case (FIELD_TYPE_CONSTANT)
       out_field%val=in_field%val(1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_scalar_field_field

  subroutine set_scalar_field_from_vector_field(out_field, in_field, dim)
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(scalar_field), intent(inout) :: out_field
    type(vector_field), intent(in) :: in_field
    integer, intent(in) :: dim

    assert(mesh_compatible(out_field%mesh, in_field%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
    assert(out_field%field_type==FIELD_TYPE_NORMAL .or. in_field%field_type==FIELD_TYPE_CONSTANT)
    assert(dim>=1 .and. dim<=in_field%dim)

    select case (in_field%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val=in_field%val(dim,:)
    case (FIELD_TYPE_CONSTANT)
       out_field%val=in_field%val(dim,1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_scalar_field_from_vector_field

  subroutine set_scalar_field_node(field, node_number, val)
    !!< Set the scalar field at the specified node
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    integer, intent(in) :: node_number
    real, intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(node_number) = val

  end subroutine set_scalar_field_node

  subroutine set_scalar_field_nodes(field, node_numbers, val)
    !!< Set the scalar field at the specified node_numbers
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(:), intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    assert(size(node_numbers)==size(val))

    field%val(node_numbers) = val

  end subroutine set_scalar_field_nodes

  subroutine set_scalar_field_constant_nodes(field, node_numbers, val)
    !!< Set the scalar field at the specified node_numbers
    !!< to a constant value
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(node_numbers) = val

  end subroutine set_scalar_field_constant_nodes

  subroutine set_scalar_field(field, val)
    !!< Set the scalar field with a constant value
    !!< Works for constant and space varying fields.
    type(scalar_field), intent(inout) :: field
    real, intent(in) :: val

    assert(field%field_type/=FIELD_TYPE_PYTHON)

    field%val = val

  end subroutine set_scalar_field

  subroutine set_scalar_field_arr(field, val)
    !!< Set the scalar field at all nodes at once
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    real, dimension(:), intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val = val

  end subroutine set_scalar_field_arr

  subroutine set_vector_field_node(field, node, val)
    !!< Set the vector field at the specified node
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, intent(in) :: node
    real, intent(in), dimension(:) :: val
    integer :: i

    assert(field%field_type==FIELD_TYPE_NORMAL)

    do i=1,field%dim
      field%val(i,node) = val(i)
    end do

  end subroutine set_vector_field_node

  subroutine set_scalar_field_theta(out_field, in_field_new, in_field_old, theta)
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(scalar_field), intent(inout) :: out_field
    type(scalar_field), intent(in) :: in_field_new, in_field_old
    real, intent(in) :: theta

    assert(mesh_compatible(out_field%mesh, in_field_new%mesh))
    assert(mesh_compatible(out_field%mesh, in_field_old%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
#ifndef NDEBUG
    if(.not.(out_field%field_type==FIELD_TYPE_NORMAL .or. &
       (in_field_new%field_type==FIELD_TYPE_CONSTANT .and. &
       in_field_old%field_type==FIELD_TYPE_CONSTANT))) then
       ewrite(-1,*) "Incompatible field types in set()"
       FLAbort("evilness unleashed")
    end if
#endif

    select case (in_field_new%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val=theta*in_field_new%val + (1.-theta)*in_field_old%val
    case (FIELD_TYPE_CONSTANT)
       out_field%val=theta*in_field_new%val(1) + (1.-theta)*in_field_old%val(1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_scalar_field_theta

  subroutine set_vector_field_node_dim(field, dim, node, val)
    !!< Set the vector field at the specified node
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, intent(in) :: node
    real, intent(in) :: val
    integer, intent(in):: dim

    assert(field%field_type==FIELD_TYPE_NORMAL)
    assert(dim>=1 .and. dim<=field%dim)

    field%val(dim,node) = val

  end subroutine set_vector_field_node_dim

  subroutine set_vector_field_nodes(field, node_numbers, val)
    !!< Set the vector field at the specified nodes
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    !! values to set ( dimension x #nodes)
    real, intent(in), dimension(:,:) :: val
    integer :: i

    assert(field%field_type==FIELD_TYPE_NORMAL)

    do i=1,field%dim
      field%val(i,node_numbers) = val(i, :)
    end do

  end subroutine set_vector_field_nodes

  subroutine set_vector_field_nodes_dim(field, dim, node_numbers, val)
    !!< Set the vector field at the specified nodes
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    !! values to set
    real, intent(in), dimension(:) :: val
    integer, intent(in) :: dim

    assert(field%field_type==FIELD_TYPE_NORMAL)
    assert(dim>=1 .and. dim<=field%dim)

    field%val(dim,node_numbers) = val

  end subroutine set_vector_field_nodes_dim

  subroutine set_vector_field(field, val)
    !!< Set the vector field with a constant value
    !!< Works for constant and space varying fields.
    type(vector_field), intent(inout) :: field
    real, intent(in), dimension(:) :: val
    integer :: i

    assert(field%field_type/=FIELD_TYPE_PYTHON)

    do i=1,field%dim
      field%val(i,:) = val(i)
    end do

  end subroutine set_vector_field

  subroutine set_vector_field_dim(field, dim, val)
    !!< Set the vector field with a constant value
    !!< Works for constant and space varying fields.
    type(vector_field), intent(inout) :: field
    real, intent(in):: val
    integer, intent(in):: dim

    assert(field%field_type/=FIELD_TYPE_PYTHON)
    assert(dim>=1 .and. dim<=field%dim)

    field%val(dim,:) = val

  end subroutine set_vector_field_dim

  subroutine set_vector_field_arr(field, val)
    !!< Set the vector field with an array for all nodes at once
    type(vector_field), intent(inout) :: field
    real, intent(in), dimension(:, :) :: val
    integer :: i

    assert(field%field_type == FIELD_TYPE_NORMAL)

    do i=1,field%dim
      field%val(i,:) = val(i, :)
    end do

  end subroutine set_vector_field_arr

  subroutine set_vector_field_arr_dim(field, dim, val)
    !!< Set the vector field with an array for all nodes at once
    type(vector_field), intent(inout) :: field
    real, intent(in), dimension(:) :: val
    integer, intent(in):: dim

    assert(field%field_type == FIELD_TYPE_NORMAL)
    assert(dim>=1 .and. dim<=field%dim)

    field%val(dim,:) = val

  end subroutine set_vector_field_arr_dim

  subroutine set_vector_field_field(out_field, in_field )
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(vector_field), intent(inout) :: out_field
    type(vector_field), intent(in) :: in_field

    integer :: dim

    assert(mesh_compatible(out_field%mesh, in_field%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
    assert(out_field%field_type==FIELD_TYPE_NORMAL .or. in_field%field_type==FIELD_TYPE_CONSTANT)
    assert(in_field%dim==out_field%dim)

    select case (in_field%field_type)
    case (FIELD_TYPE_NORMAL)
       do dim=1,in_field%dim
          out_field%val(dim,:)=in_field%val(dim,:)
       end do
    case (FIELD_TYPE_CONSTANT)
       do dim=1,in_field%dim
          out_field%val(dim,:)=in_field%val(dim,1)
       end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_vector_field_field

  subroutine set_vector_field_theta(out_field, in_field_new, in_field_old, theta)
    !!< Set theta*in_field_new + (1.-theta)*in_field_old to out_field. This will only work if the fields have
    !!< the same mesh.
    type(vector_field), intent(inout) :: out_field
    type(vector_field), intent(in) :: in_field_new, in_field_old
    real, intent(in) :: theta

    integer :: dim

    assert(mesh_compatible(out_field%mesh, in_field_new%mesh))
    assert(mesh_compatible(out_field%mesh, in_field_old%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
#ifndef NDEBUG
    if(.not.(out_field%field_type==FIELD_TYPE_NORMAL .or. &
       (in_field_new%field_type==FIELD_TYPE_CONSTANT .and. &
        in_field_old%field_type==FIELD_TYPE_CONSTANT))) then
       ewrite(-1,*) "Incompatible field types in set()"
        FLAbort("Evilness");
    end if
#endif
    assert(in_field_new%dim==out_field%dim)
    assert(in_field_old%dim==out_field%dim)

    select case (in_field_new%field_type)
    case (FIELD_TYPE_NORMAL)
      do dim = 1, out_field%dim
        out_field%val(dim,:)=theta*in_field_new%val(dim,:) + (1.-theta)*in_field_old%val(dim,:)
      end do
    case (FIELD_TYPE_CONSTANT)
      do dim = 1, out_field%dim
        out_field%val(dim,:)=theta*in_field_new%val(dim,1) + (1.-theta)*in_field_old%val(dim,1)
      end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_vector_field_theta

  subroutine set_vector_field_field_dim(out_field, dim, in_field)
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(vector_field), intent(inout) :: out_field
    type(scalar_field), intent(in) :: in_field
    integer, intent(in):: dim

    assert(mesh_compatible(out_field%mesh, in_field%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
    assert(out_field%field_type==FIELD_TYPE_NORMAL.or.in_field%field_type==FIELD_TYPE_CONSTANT)
    assert(dim>=1 .and. dim<=out_field%dim)

    select case (in_field%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val(dim,:)=in_field%val
    case (FIELD_TYPE_CONSTANT)
       out_field%val(dim,:)=in_field%val(1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_vector_field_field_dim

  subroutine set_vector_field_vfield_dim(out_field, dim, in_field)
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(vector_field), intent(inout) :: out_field
    type(vector_field), intent(in) :: in_field
    integer, intent(in):: dim

    assert(mesh_compatible(out_field%mesh, in_field%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
    assert(out_field%field_type==FIELD_TYPE_NORMAL.or.in_field%field_type==FIELD_TYPE_CONSTANT)
    assert(dim>=1 .and. dim<=out_field%dim .and. dim<=in_field%dim)

    select case (in_field%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val(dim,:)=in_field%val(dim,:)
    case (FIELD_TYPE_CONSTANT)
       out_field%val(dim,:)=in_field%val(dim,1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_vector_field_vfield_dim

  subroutine set_tensor_field_field(out_field, in_field )
    !!< Set in_field to out_field. This will only work if the fields have
    !!< the same mesh.
    type(tensor_field), intent(inout) :: out_field
    type(Tensor_field), intent(in) :: in_field

    integer i

    assert(mesh_compatible(out_field%mesh, in_field%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
    assert(out_field%field_type==FIELD_TYPE_NORMAL.or.in_field%field_type==FIELD_TYPE_CONSTANT)
    assert(all(in_field%dim==out_field%dim))

    select case (in_field%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val=in_field%val
    case (FIELD_TYPE_CONSTANT)
       do i=1, size(out_field%val,3)
          out_field%val(:,:,i)=in_field%val(:,:,1)
       end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_tensor_field_field

  subroutine set_tensor_field_theta(out_field, in_field_new, in_field_old, theta)
    !!< Set theta*in_field_new + (1.-theta)*in_field_old to out_field. This will only work if the fields have
    !!< the same mesh.
    type(tensor_field), intent(inout) :: out_field
    type(tensor_field), intent(in) :: in_field_new, in_field_old
    real, intent(in) :: theta

    integer i

    assert(mesh_compatible(out_field%mesh, in_field_new%mesh))
    assert(mesh_compatible(out_field%mesh, in_field_old%mesh))
    assert(out_field%field_type/=FIELD_TYPE_PYTHON)
#ifndef NDEBUG
    if(.not.(out_field%field_type==FIELD_TYPE_NORMAL .or. &
       (in_field_new%field_type==FIELD_TYPE_CONSTANT .and. &
        in_field_old%field_type==FIELD_TYPE_CONSTANT))) then
       ewrite(-1,*) "Incompatible field types in set()"
        FLAbort("Evil")
    end if
#endif
    assert(all(in_field_new%dim==out_field%dim))
    assert(all(in_field_old%dim==out_field%dim))

    select case (in_field_new%field_type)
    case (FIELD_TYPE_NORMAL)
       out_field%val=theta*in_field_new%val + (1.-theta)*in_field_old%val
    case (FIELD_TYPE_CONSTANT)
      do i = 1, size(out_field%val, 3)
        out_field%val(:,:,i)=theta*in_field_new%val(:,:,1) + (1.-theta)*in_field_old%val(:,:,1)
      end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_tensor_field_theta

  subroutine set_tensor_field_scalar_field(tensor, i, j, scalar, symmetric, scale)
    !!< Set the i,j^th component of tensor to be scalar.
    type(tensor_field), intent(inout) :: tensor
    integer, intent(in) :: i, j
    type(scalar_field), intent(in) :: scalar
    logical, intent(in), optional :: symmetric
    real, intent(in), optional :: scale

    real :: lscale

    assert(tensor%mesh%refcount%id==scalar%mesh%refcount%id)
    assert(tensor%field_type/=FIELD_TYPE_PYTHON)
    assert(tensor%field_type==FIELD_TYPE_NORMAL .or. scalar%field_type==FIELD_TYPE_CONSTANT)

    if (present(scale)) then
       lscale=scale
    else
       lscale=1.0
    end if

    select case (scalar%field_type)
    case (FIELD_TYPE_NORMAL)
       tensor%val(i, j, :) = scalar%val*lscale
       if (present_and_true(symmetric)) then
         tensor%val(j, i, :) = scalar%val*lscale
       end if
    case (FIELD_TYPE_CONSTANT)
       tensor%val(i, j, :) = scalar%val(1)*lscale

       if (present_and_true(symmetric)) then
         tensor%val(j, i, :) = scalar%val(1)*lscale
       end if
    case default
       ! someone could implement scalar field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_tensor_field_scalar_field

  subroutine set_tensor_field_diag_vector_field(tensor, vector, scale)
    !!< Set the diagonal components of tensor to be vector.
    type(tensor_field), intent(inout) :: tensor
    type(vector_field), intent(in) :: vector
    real, intent(in), optional :: scale

    integer :: i
    real :: lscale

    assert(tensor%mesh%refcount%id==vector%mesh%refcount%id)
    assert(tensor%field_type/=FIELD_TYPE_PYTHON)
    assert(tensor%field_type==FIELD_TYPE_NORMAL .or. vector%field_type==FIELD_TYPE_CONSTANT)
    assert(minval(tensor%dim)==vector%dim)

    if (present(scale)) then
       lscale=scale
    else
       lscale=1.0
    end if

    select case (vector%field_type)
    case (FIELD_TYPE_NORMAL)
       do i = 1, minval(tensor%dim)
          tensor%val(i, i, :) = vector%val(i,:)*lscale
       end do
    case (FIELD_TYPE_CONSTANT)
       do i = 1, minval(tensor%dim)
          tensor%val(i, i, :) = vector%val(i,1)*lscale
       end do
    case default
       ! someone could implement scalar field type python
       FLAbort("Illegal in_field field type in set()")
    end select

  end subroutine set_tensor_field_diag_vector_field

  subroutine set_tensor_field_node(field, node, val)
    !!< Set the tensor field at the specified node
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, intent(in) :: node
    real, intent(in), dimension(:, :) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(:, :, node) = val

  end subroutine set_tensor_field_node

  subroutine set_tensor_field_node_dim(field, dim1, dim2, node, val)
    !!< Set the tensor field at the specified node
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, intent(in) :: dim1, dim2, node
    real, intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)
    assert(dim1>=1 .and. dim1<=field%dim(1))
    assert(dim2>=1 .and. dim2<=field%dim(2))

    field%val(dim1, dim2, node) = val

  end subroutine set_tensor_field_node_dim

  subroutine set_tensor_field_nodes(field, node_numbers, val)
    !!< Set the tensor field at the specified nodes
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, intent(in), dimension(:, :, :) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(:, :, node_numbers) = val

  end subroutine set_tensor_field_nodes

  subroutine set_tensor_field(field, val)
    !!< Sets tensor with constant value
    !!< Works for constant and space varying fields.
    type(tensor_field), intent(inout) :: field
    real, intent(in), dimension(:, :) :: val
    integer :: i

    assert(field%field_type/=FIELD_TYPE_PYTHON)

    do i=1,size(field%val, 3)
      field%val(:, :, i) = val
    end do

  end subroutine set_tensor_field

  subroutine set_tensor_field_arr(field, val)
    !!< Set the tensor field at all nodes at once
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    real, dimension(:,:,:), intent(in) :: val

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val = val

  end subroutine set_tensor_field_arr

  subroutine set_tensor_field_arr_dim(field, dim1, dim2, val)
    !!< Set the tensor field at all nodes at once
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    real, dimension(:), intent(in) :: val
    integer, intent(in):: dim1, dim2

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(dim1, dim2, :) = val

  end subroutine set_tensor_field_arr_dim

  subroutine set_from_python_function_scalar(field, func, position, time)
    !!< Set the values at the nodes of field using the python function
    !!< specified in the string func. The position field is used to
    !!< determine the locations of the nodes.
    type(scalar_field), intent(inout) :: field
    !! Func may contain any python at all but the following function must
    !! be defined:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    type(vector_field), intent(in), target :: position
    real, intent(in) :: time

    type(vector_field) :: lposition
    real, dimension(:), pointer :: x, y, z
    real, dimension(0), target :: zero
    integer :: stat, dim

    dim=position%dim

    x=>zero
    y=>zero
    z=>zero
    if (field%mesh==position%mesh) then
       x=>position%val(1,:)

       if (dim>1) then
          y=>position%val(2,:)

          if (dim>2) then
             z=>position%val(3,:)
          end if
       end if
    else
       ! Remap position first.
       lposition = get_remapped_coordinates(position, field%mesh)
       ! we've just allowed remapping from a higher order to a lower order continuous field as this should be valid for
       ! coordinates
       ! also allowed to remap from unperiodic to periodic... hopefully the python function used will also be periodic!

       x=>lposition%val(1,:)

       if (dim>1) then
          y=>lposition%val(2,:)

          if (dim>2) then
             z=>lposition%val(3,:)
          end if
       end if
    end if

    call set_scalar_field_from_python(func, len(func), dim,&
            & node_count(field), x, y, z, time, field%val, stat)

    if (stat/=0) then
      ewrite(-1, *) "Python error while setting field: "//trim(field%name)
      ewrite(-1, *) "Python string was:"
      ewrite(-1, *) trim(func)
      FLExit("Dying")
    end if

    if (has_references(lposition)) then
       call deallocate(lposition)
    end if

  end subroutine set_from_python_function_scalar

  subroutine set_from_python_function_vector(field, func, position, time)
    !!< Set the values at the nodes of field using the python function
    !!< specified in the string func. The position field is used to
    !!< determine the locations of the nodes.
    type(vector_field), intent(inout) :: field
    !! Func may contain any python at all but the following function must
    !! be defiled:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    type(vector_field), intent(in), target :: position
    real, intent(in) :: time

    type(vector_field) :: lposition
    real, dimension(:), pointer :: x, y, z, fx, fy, fz
    real, dimension(0), target :: zero
    integer :: stat, dim

    dim=position%dim

    if (mesh_dim(field)/=mesh_dim(position)) then
       ewrite(0,'(a,i0)') "Vector field "//trim(field%name)//" has mesh dimension ",mesh_dim(field)
       ewrite(0,'(a,i0)') "Position field "//trim(position%name)//" has mesh dimension ",mesh_dim(position)
       FLExit("This is inconsistent")
    end if

    x=>zero
    y=>zero
    z=>zero
    if (field%mesh==position%mesh) then
       x=>position%val(1,:)

       if (dim>1) then
          y=>position%val(2,:)

          if (dim>2) then
             z=>position%val(3,:)
          end if
       end if
    else
       ! Remap position first.
       lposition = get_remapped_coordinates(position, field%mesh)
       ! we've just allowed remapping from a higher order to a lower order continuous field as this should be valid for
       ! coordinates
       ! also allowed to remap from unperiodic to periodic... hopefully the python function used will also be periodic!

       x=>lposition%val(1,:)

       if (dim>1) then
          y=>lposition%val(2,:)

          if (dim>2) then
             z=>lposition%val(3,:)
          end if
       end if
    end if

    fx=>zero
    fy=>zero
    fz=>zero

    fx=>field%val(1,:)
    if (field%dim>1) then
       fy=>field%val(2,:)

       if (field%dim>2) then
          fz=>field%val(3,:)
       end if
    end if


    call set_vector_field_from_python(func, len_trim(func), dim,&
            & node_count(field), x, y, z, time, field%dim, &
            & fx, fy, fz, stat)

    if (stat/=0) then
      ewrite(-1, *) "Python error while setting field: "//trim(field%name)
      ewrite(-1, *) "Python string was:"
      ewrite(-1, *) trim(func)
      FLExit("Dying")
    end if

    if (has_references(lposition)) then
       call deallocate(lposition)
    end if

  end subroutine set_from_python_function_vector

  subroutine set_from_python_function_tensor(field, func, position, time)
    !!< Set the values at the nodes of field using the python function
    !!< specified in the string func. The position field is used to
    !!< determine the locations of the nodes.
    type(tensor_field), intent(inout) :: field
    !! Func may contain any python at all but the following function must
    !! be defined:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    type(vector_field), intent(in), target :: position
    real, intent(in) :: time

    type(vector_field) :: lposition
    real, dimension(:), pointer :: x, y, z
    real, dimension(0), target :: zero
    integer :: stat, dim

    dim=position%dim

    x=>zero
    y=>zero
    z=>zero
    if (field%mesh==position%mesh) then
       x=>position%val(1,:)

       if (dim>1) then
          y=>position%val(2,:)

          if (dim>2) then
             z=>position%val(3,:)
          end if
       end if
    else
       ! Remap position first.
       lposition = get_remapped_coordinates(position, field%mesh)
       ! we've just allowed remapping from a higher order to a lower order continuous field as this should be valid for
       ! coordinates
       ! also allowed to remap from unperiodic to periodic... hopefully the python function used will also be periodic!

       x=>lposition%val(1,:)

       if (dim>1) then
          y=>lposition%val(2,:)

          if (dim>2) then
             z=>lposition%val(3,:)
          end if
       end if
    end if

    call set_tensor_field_from_python(func, len(func), dim,&
            & node_count(field), x, y, z, time, field%dim, &
            field%val, stat)

    if (stat/=0) then
      ewrite(-1, *) "Python error while setting field: "//trim(field%name)
      ewrite(-1, *) "Python string was:"
      ewrite(-1, *) trim(func)
      FLExit("Dying")
    end if

    if (has_references(lposition)) then
       call deallocate(lposition)
    end if

  end subroutine set_from_python_function_tensor

  subroutine set_from_function_scalar(field, func, position)
    !!< Set the values in field using func applied to the position field.
    !!< Func should be a function which takes a real position vector and
    !!< returns a scalar real value.
    type(scalar_field), intent(inout) :: field
    type(vector_field), intent(in) :: position
    interface
       function func(X)
         real :: func
         real, dimension(:), intent(in) :: X
       end function func
    end interface

    type(vector_field) :: lpos
    integer :: i

    if (field%field_type /= FIELD_TYPE_NORMAL) then
      FLAbort("You can only set a normal field from a function!")
    end if

    call allocate(lpos, position%dim, field%mesh, "Local Position")

    call remap_field(position, lpos)

    do i=1,node_count(field)
       field%val(i)=func(node_val(lpos, i))
    end do

    call deallocate(lpos)

  end subroutine set_from_function_scalar

  subroutine set_from_function_vector(field, func, position)
    !!< Set the values in field using func applied to the position field.
    !!< Func should be a function which takes a real position vector and
    !!< returns a vector real value of the same dimension as the position
    !!< field.
    type(vector_field), intent(inout) :: field
    type(vector_field), intent(in) :: position
    interface
       function func(X)
         real, dimension(:), intent(in) :: X
         real, dimension(size(X)) :: func
       end function func
    end interface

    type(vector_field) :: lpos
    integer :: i

    if (field%field_type /= FIELD_TYPE_NORMAL) then
      FLAbort("You can only set a normal field from a function!")
    end if

    call allocate(lpos, position%dim, field%mesh, "Local Position")

    call remap_field(position, lpos)

    call zero(field)

    do i=1,node_count(field)
       call addto(field, i, func(node_val(lpos, i)))
    end do

    call deallocate(lpos)

  end subroutine set_from_function_vector

  subroutine set_from_function_tensor(field, func, position)
    !!< Set the values in field using func applied to the position field.
    !!< Func should be a function which takes a real position vector and
    !!< returns a tensor real value of the same dimension as the position
    !!< field.
    type(tensor_field), intent(inout) :: field
    type(vector_field), intent(in) :: position
    interface
       function func(X)
         real, dimension(:), intent(in) :: X
         real, dimension(size(X), size(X)) :: func
       end function func
    end interface

    type(vector_field) :: lpos
    integer :: i

    if (field%field_type /= FIELD_TYPE_NORMAL) then
      FLAbort("You can only set a normal field from a function!")
    end if

    call allocate(lpos, position%dim, field%mesh, "Local Position")

    call remap_field(position, lpos)

    call zero(field)

    do i=1,node_count(field)
       call addto(field, i, func(node_val(lpos, i)))
    end do

    call deallocate(lpos)

  end subroutine set_from_function_tensor

  ! ------------------------------------------------------------------------
  ! Mapping of fields between different meshes
  ! ------------------------------------------------------------------------

  subroutine test_remap_validity_scalar(from_field, to_field, stat)
    type(scalar_field), intent(in):: from_field, to_field
    integer, intent(out), optional:: stat

    if(present(stat)) stat = 0

    call test_remap_validity_generic(trim(from_field%name), trim(to_field%name), &
                                     continuity(from_field), continuity(to_field), &
                                     element_degree(from_field, 1), element_degree(to_field, 1), &
                                     mesh_periodic(from_field), mesh_periodic(to_field), &
                                     from_field%mesh%shape%type, to_field%mesh%shape%type, &
                                     stat)

  end subroutine test_remap_validity_scalar

  subroutine test_remap_validity_vector(from_field, to_field, stat)
    type(vector_field), intent(in):: from_field, to_field
    integer, intent(out), optional:: stat

    if(present(stat)) stat = 0

    call test_remap_validity_generic(trim(from_field%name), trim(to_field%name), &
                                     continuity(from_field), continuity(to_field), &
                                     element_degree(from_field, 1), element_degree(to_field, 1), &
                                     mesh_periodic(from_field), mesh_periodic(to_field), &
                                     from_field%mesh%shape%type, to_field%mesh%shape%type, &
                                     stat)

  end subroutine test_remap_validity_vector

  subroutine test_remap_validity_tensor(from_field, to_field, stat)
    type(tensor_field), intent(in):: from_field, to_field
    integer, intent(out), optional:: stat

    if(present(stat)) stat = 0

    call test_remap_validity_generic(trim(from_field%name), trim(to_field%name), &
                                     continuity(from_field), continuity(to_field), &
                                     element_degree(from_field, 1), element_degree(to_field, 1), &
                                     mesh_periodic(from_field), mesh_periodic(to_field), &
                                     from_field%mesh%shape%type, to_field%mesh%shape%type, &
                                     stat)

  end subroutine test_remap_validity_tensor

  subroutine test_remap_validity_generic(from_name, to_name, &
                                         from_continuity, to_continuity, &
                                         from_degree, to_degree, &
                                         from_periodic, to_periodic, &
                                         from_type, to_type, &
                                         stat)
    character(len=*), intent(in):: from_name, to_name
    integer, intent(in):: from_continuity, to_continuity
    integer, intent(in):: from_degree, to_degree
    logical, intent(in):: from_periodic, to_periodic
    integer, intent(in):: from_type, to_type
    integer, intent(out), optional:: stat

    if(present(stat)) stat = 0

    if((from_continuity<0).and.(.not.(to_continuity<0))) then
      if(present(stat)) then
        stat = REMAP_ERR_DISCONTINUOUS_CONTINUOUS
      else
        ewrite(-1,*) "Remapping from field "//trim(from_name)//" to field "//trim(to_name)//"."
        FLAbort("Trying to remap from discontinuous to continuous field.")
      end if
    end if

    ! this test currently assumes that the shape function degree is constant over the mesh
    if((.not.(from_continuity<0)).and.(.not.(to_continuity<0))&
        .and.(from_degree>to_degree)) then
      if(present(stat)) then
        stat = REMAP_ERR_HIGHER_LOWER_CONTINUOUS
      else
        ewrite(-1,*) "Remapping from field "//trim(from_name)//" to field "//trim(to_name)//"."
        FLAbort("Trying to remap from higher order to lower order continuous field")
      end if
    end if

    if((.not.(from_continuity<0)).and.(.not.(to_continuity<0))&
        .and.(.not.from_periodic).and.(to_periodic)) then
      if(present(stat)) then
        stat = REMAP_ERR_UNPERIODIC_PERIODIC
      else
        ewrite(-1,*) "Remapping from field "//trim(from_name)//" to field "//trim(to_name)//"."
        FLAbort("Trying to remap from an unperiodic to a periodic continuous field")
      end if
    end if

    if((from_type==ELEMENT_BUBBLE).and.&
       (to_type==ELEMENT_LAGRANGIAN)) then
      if(present(stat)) then
        stat = REMAP_ERR_BUBBLE_LAGRANGE
      else
        ewrite(-1,*) "Remapping from field "//trim(from_name)//" to field "//trim(to_name)//"."
        FLAbort("Trying to remap from a bubble to a lagrange field")
      end if
    end if

  end subroutine test_remap_validity_generic

  subroutine remap_scalar_field(from_field, to_field, stat)
    !!< Remap the components of from_field onto the locations of to_field.
    !!< This is used to change the element type of a field.
    !!<
    !!< This will not validly map a discontinuous field to a continuous
    !!< field.
    type(scalar_field), intent(in) :: from_field
    type(scalar_field), intent(inout) :: to_field
    integer, intent(out), optional :: stat

    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof) :: locweight

    integer :: fromloc, toloc, ele
    integer, dimension(:), pointer :: from_ele, to_ele

    if(present(stat)) stat = 0

    if(from_field%mesh==to_field%mesh) then

      call set(to_field, from_field)

    else

      select case(from_field%field_type)
      case(FIELD_TYPE_NORMAL)

        call test_remap_validity(from_field, to_field, stat=stat)

        ! First construct remapping weights.
        do toloc=1,size(locweight,1)
          do fromloc=1,size(locweight,2)
              locweight(toloc,fromloc)=eval_shape(from_field%mesh%shape, fromloc, &
                  local_coords(toloc, to_field%mesh%shape))
          end do
        end do

        ! Now loop over the elements.
        do ele=1,element_count(from_field)
          from_ele=>ele_nodes(from_field, ele)
          to_ele=>ele_nodes(to_field, ele)

          to_field%val(to_ele)=matmul(locweight,from_field%val(from_ele))

        end do

      case(FIELD_TYPE_CONSTANT)
        to_field%val = from_field%val(1)
      end select

    end if

  end subroutine remap_scalar_field

  subroutine remap_scalar_field_specific(from_field, to_field, elements, output, locweight, stat)
    !!< Remap the components of from_field onto the locations of to_field.
    !!< This is used to change the element type of a field.
    !!<
    !!< This will not validly map a discontinuous field to a continuous
    !!< field.
    !!< This only does certain elements, and can optionally take in a precomputed locweight.

    type(scalar_field), intent(in) :: from_field
    type(scalar_field), intent(inout) :: to_field
    integer, dimension(:), intent(in) :: elements
    real, dimension(size(elements), to_field%mesh%shape%ndof), intent(out) :: output
    integer, intent(out), optional:: stat

    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof), optional :: locweight
    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof) :: llocweight

    integer :: fromloc, toloc, ele, i

    if(present(stat)) stat = 0

    if (from_field%field_type == FIELD_TYPE_CONSTANT) then
      output = from_field%val(1)
      return
    end if

    call test_remap_validity(from_field, to_field, stat=stat)

    if (.not. present(locweight)) then
      ! First construct remapping weights.
      do toloc=1,size(llocweight,1)
         do fromloc=1,size(llocweight,2)
            llocweight(toloc,fromloc)=eval_shape(from_field%mesh%shape, fromloc, &
                 local_coords(toloc, to_field%mesh%shape))
         end do
      end do
    else
      llocweight = locweight
    end if

      ! Now loop over the elements.
    do i=1,size(elements)
      ele = elements(i)
      output(i, :)=matmul(llocweight,ele_val(from_field, ele))
    end do
  end subroutine remap_scalar_field_specific

  subroutine remap_vector_field(from_field, to_field, stat)
    !!< Remap the components of from_field onto the locations of to_field.
    !!< This is used to change the element type of a field.
    !!<
    !!< The result will only be valid if to_field is DG.
    type(vector_field), intent(in) :: from_field
    type(vector_field), intent(inout) :: to_field
    integer, intent(out), optional :: stat

    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof) :: locweight

    integer :: fromloc, toloc, ele, i
    integer, dimension(:), pointer :: from_ele, to_ele

    if(present(stat)) stat = 0

    assert(to_field%dim>=from_field%dim)

    if (mesh_dim(from_field)/=mesh_dim(to_field)) then
       ewrite (0,*)"Remapping "//trim(from_field%name)//" to "&
            &//trim(to_field%name)
       ewrite (0,'(a,i0)')"Mesh dimension of "//trim(from_field%name)//&
            " is ", mesh_dim(from_field)
       ewrite (0,'(a,i0)')"Mesh dimension of "//trim(to_field%name)//&
            " is ", mesh_dim(to_field)
       FLExit("Mesh dimensions inconsistent")
    end if

    if(from_field%mesh==to_field%mesh) then

      call set(to_field, from_field)

    else

      select case(from_field%field_type)
      case(FIELD_TYPE_NORMAL)

        call test_remap_validity(from_field, to_field, stat=stat)

        ! First construct remapping weights.
        do toloc=1,size(locweight,1)
          do fromloc=1,size(locweight,2)
              locweight(toloc,fromloc)=eval_shape(from_field%mesh%shape, fromloc, &
                  local_coords(toloc, to_field%mesh%shape))
          end do
        end do

        ! Now loop over the elements.
        do ele=1,element_count(from_field)
          from_ele=>ele_nodes(from_field, ele)
          to_ele=>ele_nodes(to_field, ele)

          do i=1,from_field%dim
              to_field%val(i,to_ele)= &
                  matmul(locweight,from_field%val(i,from_ele))
          end do

        end do

      case(FIELD_TYPE_CONSTANT)
        do i=1,from_field%dim
          to_field%val(i,:) = from_field%val(i,1)
        end do
      end select

    end if

    ! Zero any left-over dimensions
    do ele=from_field%dim+1,to_field%dim
      to_field%val(i,:)=0.0
    end do

  end subroutine remap_vector_field

  subroutine remap_vector_field_specific(from_field, to_field, elements, output, locweight, stat)
    !!< Remap the components of from_field onto the locations of to_field.
    !!< This is used to change the element type of a field.
    !!<
    !!< The result will only be valid if to_field is DG.
    type(vector_field), intent(in) :: from_field
    type(vector_field), intent(inout) :: to_field
    integer, dimension(:), intent(in) :: elements
    real, dimension(size(elements), to_field%dim, to_field%mesh%shape%ndof), intent(out) :: output
    integer, intent(out), optional:: stat

    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof), optional :: locweight
    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof) :: llocweight

    integer :: fromloc, toloc, ele, i, j

    if(present(stat)) stat = 0

    assert(to_field%dim>=from_field%dim)

    output = 0.0

    select case(from_field%field_type)
    case(FIELD_TYPE_CONSTANT)
      do i=1,from_field%dim
        output(:, i, :) = from_field%val(i,1)
      end do
      return
    end select

    call test_remap_validity(from_field, to_field, stat=stat)

    if (.not. present(locweight)) then
      ! First construct remapping weights.
      do toloc=1,size(llocweight,1)
         do fromloc=1,size(llocweight,2)
            llocweight(toloc,fromloc)=eval_shape(from_field%mesh%shape, fromloc, &
                 local_coords(toloc, to_field%mesh%shape))
         end do
      end do
    else
      llocweight = locweight
    end if

    ! Now loop over the elements.
    do j=1,size(elements)
      ele = elements(j)
      do i=1,from_field%dim
        output(j, i, :) = matmul(llocweight,ele_val(from_field, i, ele))
      end do
    end do
  end subroutine remap_vector_field_specific

  subroutine remap_tensor_field(from_field, to_field, stat)
    !!< Remap the components of from_field onto the locations of to_field.
    !!< This is used to change the element type of a field.
    !!<
    !!< The result will only be valid if to_field is DG.
    type(tensor_field), intent(in) :: from_field
    type(tensor_field), intent(inout) :: to_field
    integer, intent(inout), optional :: stat

    real, dimension(to_field%mesh%shape%ndof, from_field%mesh%shape%ndof) :: locweight

    integer :: fromloc, toloc, ele, i, j
    integer, dimension(:), pointer :: from_ele, to_ele

    if(present(stat)) stat = 0

    assert(all(to_field%dim>=from_field%dim))

    if(from_field%mesh==to_field%mesh) then

      call set(to_field, from_field)

    else

      select case(from_field%field_type)
      case(FIELD_TYPE_NORMAL)

        call test_remap_validity(from_field, to_field, stat=stat)

        ! First construct remapping weights.
        do toloc=1,size(locweight,1)
          do fromloc=1,size(locweight,2)
              locweight(toloc,fromloc)=eval_shape(from_field%mesh%shape, fromloc, &
                  local_coords(toloc, to_field%mesh%shape))
          end do
        end do

        ! Now loop over the elements.
        do ele=1,element_count(from_field)
          from_ele=>ele_nodes(from_field, ele)
          to_ele=>ele_nodes(to_field, ele)

          do i=1,from_field%dim(1)
            do j=1,from_field%dim(2)
              to_field%val(i, j, to_ele) = matmul(locweight, from_field%val(i, j, from_ele))
            end do
          end do

        end do
      case(FIELD_TYPE_CONSTANT)
        do i=1,size(to_field%val, 3)
          to_field%val(:, :, i) = from_field%val(:, :, 1)
        end do
      end select

    end if

  end subroutine remap_tensor_field

  subroutine remap_scalar_field_to_surface(from_field, to_field, surface_element_list, stat)
    !!< Remap the values of from_field onto the surface_field to_field, which is defined
    !!< on the faces given by surface_element_list.
    !!< This also deals with remapping between different orders.
    type(scalar_field), intent(in):: from_field
    type(scalar_field), intent(inout):: to_field
    integer, dimension(:), intent(in):: surface_element_list
    integer, intent(out), optional:: stat

    real, dimension(ele_loc(to_field,1), face_loc(from_field,1)) :: locweight
    type(element_type), pointer:: from_shape, to_shape
    real, dimension(face_loc(from_field,1)) :: from_val
    integer, dimension(:), pointer :: to_nodes
    integer toloc, fromloc, ele, face

    if (present(stat)) stat = 0

    select case(from_field%field_type)
    case(FIELD_TYPE_NORMAL)

      call test_remap_validity(from_field, to_field, stat=stat)

      ! the remapping happens from a face of from_field which is at the same
      ! time an element of to_field
      from_shape => face_shape(from_field, 1)
      to_shape => ele_shape(to_field, 1)
      ! First construct remapping weights.
      do toloc=1,size(locweight,1)
         do fromloc=1,size(locweight,2)
            locweight(toloc,fromloc)=eval_shape(from_shape, fromloc, &
                 local_coords(toloc, to_shape))
         end do
      end do

      ! Now loop over the surface elements.
      do ele=1, size(surface_element_list)
         ! element ele is a face in the mesh of from_field:
         face=surface_element_list(ele)

         to_nodes => ele_nodes(to_field, ele)

         from_val = face_val(from_field, face)

         to_field%val(to_nodes)=matmul(locweight,from_val)

      end do

    case(FIELD_TYPE_CONSTANT)

      to_field%val = from_field%val(1)

    end select

  end subroutine remap_scalar_field_to_surface

  subroutine remap_vector_field_to_surface(from_field, to_field, surface_element_list, stat)
    !!< Remap the values of from_field onto the surface_field to_field, which is defined
    !!< on the faces given by surface_element_list.
    !!< This also deals with remapping between different orders.
    type(vector_field), intent(in):: from_field
    type(vector_field), intent(inout):: to_field
    integer, dimension(:), intent(in):: surface_element_list
    integer, intent(out), optional:: stat

    real, dimension(ele_loc(to_field,1), face_loc(from_field,1)) :: locweight
    type(element_type), pointer:: from_shape, to_shape
    real, dimension(from_field%dim, face_loc(from_field,1)) :: from_val
    integer, dimension(:), pointer :: to_nodes
    integer toloc, fromloc, ele, face, i

    if(present(stat)) stat = 0

    assert(to_field%dim>=from_field%dim)

    select case(from_field%field_type)
    case(FIELD_TYPE_NORMAL)

      call test_remap_validity(from_field, to_field, stat=stat)

      ! the remapping happens from a face of from_field which is at the same
      ! time an element of to_field
      from_shape => face_shape(from_field, 1)
      to_shape => ele_shape(to_field, 1)
      ! First construct remapping weights.
      do toloc=1,size(locweight,1)
         do fromloc=1,size(locweight,2)
            locweight(toloc,fromloc)=eval_shape(from_shape, fromloc, &
                 local_coords(toloc, to_shape))
         end do
      end do

      ! Now loop over the surface elements.
      do ele=1, size(surface_element_list)
         ! element ele is a face in the mesh of from_field:
         face=surface_element_list(ele)

         to_nodes => ele_nodes(to_field, ele)

         from_val = face_val(from_field, face)

         do i=1, to_field%dim
           to_field%val(i,to_nodes)=matmul(locweight,from_val(i, :))
         end do

      end do

    case(FIELD_TYPE_CONSTANT)
      do i=1, from_field%dim
        to_field%val(i,:) = from_field%val(i,1)
      end do
    end select

    ! Zero any left-over dimensions
    do ele=from_field%dim+1, to_field%dim
       to_field%val(i,:)=0.0
    end do

  end subroutine remap_vector_field_to_surface

  function piecewise_constant_mesh(in_mesh, name, with_faces) result(new_mesh)
    !!< From a given mesh, return a scalar field
    !!< allocated on the mesh that's topologically the same
    !!< but has piecewise constant basis functions.
    !!< This is for the definition of elementwise quantities.
    type(mesh_type), intent(in) :: in_mesh
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: with_faces

    type(mesh_type) :: new_mesh
    type(element_type) :: shape, old_shape

    old_shape = in_mesh%shape

    shape = make_element_shape(vertices=old_shape%ndof, dim=old_shape%dim, degree=0, quad=old_shape%quadrature)
    new_mesh = make_mesh(model=in_mesh, shape=shape, continuity=-1, &
         name=name, with_faces=with_faces)
    call deallocate(shape)

  end function piecewise_constant_mesh

  function piecewise_constant_field(in_mesh, name) result(field)
    !!< From a given mesh, return a scalar field
    !!< allocated on the mesh that's topologically the same
    !!< but has piecewise constant basis functions.
    !!< This is for the definition of elementwise quantities.
    type(mesh_type), intent(in) :: in_mesh
    type(mesh_type) :: new_mesh
    type(element_type) :: shape, old_shape
    type(scalar_field) :: field
    character(len=*), intent(in) :: name

    old_shape = in_mesh%shape

    shape = make_element_shape(vertices=old_shape%ndof, dim=old_shape%dim, degree=0, quad=old_shape%quadrature)
    new_mesh = make_mesh(model=in_mesh, shape=shape, continuity=-1)
    call allocate(field, new_mesh, name)
    call zero(field)
    call deallocate(shape)
    call deallocate(new_mesh)

  end function piecewise_constant_field

  subroutine scalar_scale(field, factor)
    !!< Multiply scalar field with factor
    type(scalar_field), intent(inout) :: field
    real, intent(in) :: factor

    assert(field%field_type/=FIELD_TYPE_PYTHON)

    field%val = field%val * factor

  end subroutine scalar_scale

  subroutine vector_scale(field, factor, dim)
    !!< Multiply vector field with factor
    type(vector_field), intent(inout) :: field
    real, intent(in) :: factor
    integer, intent(in), optional :: dim

    integer :: i

    assert(field%field_type/=FIELD_TYPE_PYTHON)

    if (present(dim)) then
      field%val(dim,:) = field%val(dim,:) * factor
    else
      do i=1,field%dim
        field%val(i,:) = field%val(i,:) * factor
      end do
    end if

  end subroutine vector_scale

  subroutine tensor_scale(field, factor)
    !!< Multiply tensor field with factor
    type(tensor_field), intent(inout) :: field
    real, intent(in) :: factor

    assert(field%field_type/=FIELD_TYPE_PYTHON)

    field%val = field%val * factor

  end subroutine tensor_scale

  subroutine scalar_scale_scalar_field(field, sfield)
    !!< Multiply scalar field with sfield. This will only work if the
    !!< fields have the same mesh.
    !!< NOTE that the integral of the resulting field by a weighted sum over its values in gauss points
    !!< will not be as accurate as multiplying the fields at each gauss point seperately
    !!< and then summing over these.
    type(scalar_field), intent(inout) :: field
    type(scalar_field), intent(in) :: sfield

    assert(field%mesh%refcount%id==sfield%mesh%refcount%id)
    assert(field%field_type/=FIELD_TYPE_PYTHON)
    assert(field%field_type==FIELD_TYPE_NORMAL .or. sfield%field_type==FIELD_TYPE_CONSTANT)

    select case (sfield%field_type)
    case (FIELD_TYPE_NORMAL)
       field%val = field%val * sfield%val
    case (FIELD_TYPE_CONSTANT)
       field%val = field%val * sfield%val(1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in scale()")
    end select

  end subroutine scalar_scale_scalar_field

  subroutine vector_scale_scalar_field(field, sfield)
    !!< Multiply vector field with scalar field. This will only work if the
    !!< fields have the same mesh.
    !!< NOTE that the integral of the resulting field by a weighted sum over its values in gauss points
    !!< will not be as accurate as multiplying the fields at each gauss point seperately
    !!< and then summing over these.
    type(vector_field), intent(inout) :: field
    type(scalar_field), intent(in) :: sfield

    integer :: i

    assert(field%mesh%refcount%id==sfield%mesh%refcount%id)
    assert(field%field_type/=FIELD_TYPE_PYTHON)
    assert(field%field_type==FIELD_TYPE_NORMAL .or. sfield%field_type==FIELD_TYPE_CONSTANT)

    select case (sfield%field_type)
    case (FIELD_TYPE_NORMAL)
       do i=1,field%dim
          field%val(i,:) = field%val(i,:) * sfield%val
       end do
    case (FIELD_TYPE_CONSTANT)
       do i=1,field%dim
          field%val(i,:) = field%val(i,:) * sfield%val(1)
       end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in scale()")
    end select

  end subroutine vector_scale_scalar_field

  subroutine tensor_scale_scalar_field(field, sfield)
    !!< Multiply tensor field with scalar field. This will only work if the
    !!< fields have the same mesh.
    !!< NOTE that the integral of the resulting field by a weighted sum over its values in gauss points
    !!< will not be as accurate as multiplying the fields at each gauss point seperately
    !!< and then summing over these.
    type(tensor_field), intent(inout) :: field
    type(scalar_field), intent(in) :: sfield

    integer :: i, j

    assert(field%mesh%refcount%id==sfield%mesh%refcount%id)
    assert(field%field_type/=FIELD_TYPE_PYTHON)
    assert(field%field_type==FIELD_TYPE_NORMAL .or. sfield%field_type==FIELD_TYPE_CONSTANT)

    select case (sfield%field_type)
    case (FIELD_TYPE_NORMAL)
       do i=1,field%dim(1)
          do j=1,field%dim(2)
             field%val(i,j,:) = field%val(i,j,:) * sfield%val
          end do
       end do
    case (FIELD_TYPE_CONSTANT)
       field%val(:,:,1) = field%val(:,:,1) * sfield%val(1)
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in scale()")
    end select

  end subroutine tensor_scale_scalar_field

  subroutine vector_scale_vector_field(field, vfield)
    !!< Multiply vector field with vector field. This will only work if the
    !!< fields have the same mesh.
    !!< NOTE that the integral of the resulting field by a weighted sum over its values in gauss points
    !!< will not be as accurate as multiplying the fields at each gauss point seperately
    !!< and then summing over these.
    type(vector_field), intent(inout) :: field
    type(vector_field), intent(in) :: vfield

    integer :: i

    assert(field%mesh%refcount%id==vfield%mesh%refcount%id)
    assert(field%field_type/=FIELD_TYPE_PYTHON)
    assert(field%field_type==FIELD_TYPE_NORMAL .or. vfield%field_type==FIELD_TYPE_CONSTANT)

    select case (vfield%field_type)
    case (FIELD_TYPE_NORMAL)
       do i=1,field%dim
          field%val(i,:) = field%val(i,:) * vfield%val(i,:)
       end do
    case (FIELD_TYPE_CONSTANT)
       do i=1,field%dim
          field%val(i,:) = field%val(i,:) * vfield%val(i,1)
       end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in scale()")
    end select

  end subroutine vector_scale_vector_field

  subroutine bound_scalar_field(field, lower_bound, upper_bound)
    !!< Bound a field by the lower and upper bounds supplied
    type(scalar_field), intent(inout) :: field
    real, intent(in) :: lower_bound, upper_bound

    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i = 1, node_count(field)
        field%val(i) = min(max(field%val(i), lower_bound), upper_bound)
      end do
    case(FIELD_TYPE_CONSTANT)
      field%val(1) = min(max(field%val(1), lower_bound), upper_bound)
    case default
      FLAbort("Illegal field type in bound()")
    end select

  end subroutine bound_scalar_field

  subroutine bound_scalar_field_field(field, lower_bound, upper_bound)
    !!< Bound a field by the lower and upper bounds supplied
    type(scalar_field), intent(inout) :: field
    type(scalar_field), intent(in), optional :: lower_bound, upper_bound

    integer :: i

    if(present(lower_bound)) then
      assert(field%mesh==lower_bound%mesh)
      assert(lower_bound%field_type==FIELD_TYPE_NORMAL) ! The case lower_bound=FIELD_TYPE_CONSTANT should be implemented
    end if
    if(present(upper_bound)) then
      assert(field%mesh==upper_bound%mesh)
      assert(upper_bound%field_type==FIELD_TYPE_NORMAL) ! The case upper_bound=FIELD_TYPE_CONSTANT should be implemented
    end if
    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
        if (present(lower_bound)) then
            do i = 1, node_count(field)
                field%val(i) = max(field%val(i), lower_bound%val(i))
            end do
        end if
        if (present(upper_bound)) then
            do i = 1, node_count(field)
                field%val(i) = min(field%val(i), upper_bound%val(i))
            end do
        end if
    case default
      FLAbort("Illegal field type in bound()")
    end select

  end subroutine bound_scalar_field_field


  subroutine bound_vector_field(field, lower_bound, upper_bound)
    !!< Bound a field by the lower and upper bounds supplied
    type(vector_field), intent(inout) :: field
    real, intent(in) :: lower_bound, upper_bound

    integer :: i, j

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i = 1, field%dim
        do j = 1, node_count(field)
          field%val(i,j) = min(max(field%val(i,j), lower_bound), upper_bound)
        end do
      end do
    case(FIELD_TYPE_CONSTANT)
      do i = 1, field%dim
        field%val(i,1) = min(max(field%val(i,1), lower_bound), upper_bound)
      end do
    case default
      FLAbort("Illegal field type in bound()")
    end select

  end subroutine bound_vector_field

  subroutine bound_tensor_field(field, lower_bound, upper_bound)
    !!< Bound a field by the lower and upper bounds supplied
    type(tensor_field), intent(inout) :: field
    real, intent(in) :: lower_bound, upper_bound

    integer :: i, j, k

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i = 1, field%dim(1)
        do j = 1, field%dim(2)
          do k = 1, node_count(field)
            field%val(i, j, k) = min(max(field%val(i, j, k), lower_bound), upper_bound)
          end do
        end do
      end do
    case(FIELD_TYPE_CONSTANT)
      do i = 1, field%dim(1)
        do j = 1, field%dim(2)
          field%val(i, j, 1) = min(max(field%val(i, j, 1), lower_bound), upper_bound)
        end do
      end do
    case default
      FLAbort("Illegal field type in bound()")
    end select

  end subroutine bound_tensor_field

  subroutine normalise_scalar(field)
    type(scalar_field), intent(inout) :: field

    integer :: i
    real :: tolerance

    tolerance = tiny(0.0)

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i = 1, node_count(field)
        call set(field, i, node_val(field, i)/(max(tolerance, abs(node_val(field, i)))))
      end do
    case(FIELD_TYPE_CONSTANT)
      field%val(1) = field%val(1)/(max(tolerance, abs(node_val(field, 1))))
    case default
      FLAbort("Illegal field type in normalise()")
    end select

  end subroutine normalise_scalar

  subroutine normalise_vector(field)
    type(vector_field), intent(inout) :: field

    integer :: i
    real :: tolerance

    tolerance = tiny(0.0)

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i = 1, node_count(field)
        call set(field, i, node_val(field, i)/(max(tolerance, norm2(node_val(field, i)))))
      end do
    case(FIELD_TYPE_CONSTANT)
      call set(field, 1, node_val(field, 1)/(max(tolerance, norm2(node_val(field, 1)))))
    case default
      FLAbort("Illegal field type in normalise()")
    end select

  end subroutine normalise_vector

  subroutine invert_scalar_field_inplace(field, tolerance)
  !!< Computes 1/field for a scalar field
    type(scalar_field), intent(inout):: field
    real, intent(in), optional :: tolerance

    call invert_scalar_field(field, field, tolerance)

  end subroutine invert_scalar_field_inplace

  subroutine invert_scalar_field(in_field, out_field, tolerance)
  !!< Computes 1/field for a scalar field
    type(scalar_field), intent(in):: in_field
    type(scalar_field), intent(inout):: out_field
    real, intent(in), optional :: tolerance

    integer :: i

    assert(out_field%field_type==FIELD_TYPE_NORMAL .or. out_field%field_type==FIELD_TYPE_CONSTANT)
    assert(out_field%mesh==in_field%mesh)
    if (in_field%field_type==out_field%field_type) then
      if(present(tolerance)) then
        do i = 1, size(out_field%val)
          out_field%val(i) = 1/sign(max(tolerance, abs(in_field%val(i))), in_field%val(i))
        end do
      else
        out_field%val=1/in_field%val
      end if
    else if (in_field%field_type==FIELD_TYPE_CONSTANT) then
      if(present(tolerance)) then
        out_field%val = 1/sign(max(tolerance, abs(in_field%val(1))), in_field%val(1))
      else
        out_field%val=1/in_field%val(1)
      end if
    else
      FLAbort("Calling invert_scalar_field with wrong field type")
    end if

  end subroutine invert_scalar_field

  subroutine invert_vector_field_inplace(field, tolerance)
  !!< Computes 1/field for a vector field
    type(vector_field), intent(inout):: field
    real, intent(in), optional :: tolerance

    call invert_vector_field(field, field, tolerance)

  end subroutine invert_vector_field_inplace

  subroutine invert_vector_field(in_field, out_field, tolerance)
  !!< Computes 1/field for a vector field
    type(vector_field), intent(in):: in_field
    type(vector_field), intent(inout):: out_field
    real, intent(in), optional :: tolerance

    integer :: i, j

    assert(out_field%field_type==FIELD_TYPE_NORMAL .or. out_field%field_type==FIELD_TYPE_CONSTANT)
    assert(in_field%dim==in_field%dim)
    do i = 1, out_field%dim
      if (in_field%field_type==out_field%field_type) then
        if(present(tolerance)) then
          do j = 1, size(out_field%val(i,:))
            out_field%val(i,j) = 1/sign(max(tolerance, abs(in_field%val(i,j))), in_field%val(i,j))
          end do
        else
          out_field%val(i,:)=1/in_field%val(i,:)
        end if
      else if (in_field%field_type==FIELD_TYPE_CONSTANT) then
        if(present(tolerance)) then
          out_field%val(i,:)=1/sign(max(tolerance, abs(in_field%val(i,1))), in_field%val(i,1))
        else
          out_field%val(i,:)=1/in_field%val(i,1)
        end if
      else
        FLAbort("Calling invert_vector_field with wrong field type")
      end if
    end do

  end subroutine invert_vector_field

  subroutine absolute_value_scalar_field(field)
  !!< Computes abs(field) for a scalar field
    type(scalar_field), intent(inout) :: field

    field%val = abs(field%val)

  end subroutine absolute_value_scalar_field

  subroutine cross_product_vector(a, b, c)
    !!< Computes the node-wise outer product a=b x c
    !!< NOTE that the integral of the resulting field by a weighted sum over its values in gauss points
    !!< will not be as accurate as multiplying the fields at each gauss point seperately
    !!< and then summing over these.
    type(vector_field), intent(inout) :: a
    type(vector_field), intent(in) :: b, c

    type(vector_field) tmp_b, tmp_c
    integer, dimension(3), parameter:: perm1=(/ 2,3,1 /), perm2=(/ 3,1,2 /)
    integer i

    assert(a%field_type/=FIELD_TYPE_PYTHON)
    assert(a%field_type==FIELD_TYPE_NORMAL .or. b%field_type==FIELD_TYPE_CONSTANT)
    assert(a%field_type==FIELD_TYPE_NORMAL .or. c%field_type==FIELD_TYPE_CONSTANT)
    assert(a%dim==b%dim)
    assert(a%dim==c%dim)

    if (a%mesh==c%mesh .and. c%field_type/=FIELD_TYPE_CONSTANT) then
       tmp_c=c
    else
       call allocate(tmp_c, c%dim, a%mesh, name='cross_product_vector_tmp_c')
       call remap_field(c, tmp_c)
    end if

    select case (b%field_type)
    case (FIELD_TYPE_NORMAL)

       if (a%mesh==b%mesh) then
          tmp_b=b
       else
          call allocate(tmp_b, b%dim, a%mesh, name='cross_product_vector_tmp_b')
          call remap_field(b, tmp_b)
       end if

       select case (c%field_type)
       case (FIELD_TYPE_NORMAL)
          do i=1, a%dim
            a%val(i,:)=tmp_b%val( perm1(i),: ) * tmp_c%val( perm2(i),: )- &
               tmp_b%val( perm2(i),: ) * tmp_c%val( perm1(i),: )
          end do
       case (FIELD_TYPE_CONSTANT)
          do i=1, a%dim
            a%val(i,:)=tmp_b%val( perm1(i),: ) * tmp_c%val( perm2(i),1 )- &
               tmp_b%val( perm2(i),: ) * tmp_c%val( perm1(i),1 )
          end do
       case default
          ! someone could implement in_field type python
          FLAbort("Illegal in_field field type in cross_product()")
       end select

       if (.not. a%mesh==b%mesh) then
          call deallocate(tmp_b)
       end if

    case (FIELD_TYPE_CONSTANT)

       select case (c%field_type)
       case (FIELD_TYPE_NORMAL)
          do i=1, a%dim
            a%val(i,:)=b%val( perm1(i),1 ) * tmp_c%val( perm2(i),: )- &
               b%val( perm2(i),1 ) * tmp_c%val( perm1(i),: )
          end do
       case (FIELD_TYPE_CONSTANT)
          do i=1, a%dim
            a%val(i,:)=b%val( perm1(i),1 ) * tmp_c%val( perm2(i),1 )- &
               b%val( perm2(i),1 ) * tmp_c%val( perm1(i),1 )
          end do
       case default
          ! someone could implement b type python
          FLAbort("Illegal in_field field type in cross_product()")
       end select

    case default

       ! someone could implement c field type python
       FLAbort("Illegal in_field field type in cross_product()")

    end select

    if (.not. a%mesh==c%mesh .or. c%field_type==FIELD_TYPE_CONSTANT) then
       call deallocate(tmp_c)
    end if

  end subroutine cross_product_vector

  subroutine inner_product_field_field(a, b, c)
    !!< Computes the node-wise inner/dot product a=b . c
    !!< This version takes two scalar fields. NOTE that if a and b and c
    !!< have the same polynomial degree you will loose accuracy. In many
    !!< cases you have to calculate this at the gauss points instead.
    type(scalar_field), intent(inout) :: a
    type(vector_field), intent(in) :: b,c

    type(vector_field) tmp_b, tmp_c
    integer i

    assert(a%field_type/=FIELD_TYPE_PYTHON)
    assert(a%field_type==FIELD_TYPE_NORMAL .or. b%field_type==FIELD_TYPE_CONSTANT)
    assert(a%field_type==FIELD_TYPE_NORMAL .or. c%field_type==FIELD_TYPE_CONSTANT)
    assert(b%dim==c%dim)

    if (a%mesh==c%mesh .and. c%field_type/=FIELD_TYPE_CONSTANT) then
       tmp_c=c
    else
       call allocate(tmp_c, c%dim, a%mesh, name='inner_product_vector_tmp_c')
       call remap_field(c, tmp_c)
    end if

    select case (b%field_type)
    case (FIELD_TYPE_NORMAL)

       if (a%mesh==b%mesh) then
          tmp_b=b
       else
          call allocate(tmp_b, b%dim, a%mesh, name='cross_product_vector_tmp_b')
          call remap_field(b, tmp_b)
       end if

       select case (c%field_type)
       case (FIELD_TYPE_NORMAL)
          a%val=tmp_b%val(1,:)*tmp_c%val(1,:)
          do i=2, c%dim
             a%val=a%val+tmp_b%val(i,:)*tmp_c%val(i,:)
          end do
       case (FIELD_TYPE_CONSTANT)
          a%val=tmp_b%val(1,:)*c%val(1,1)
          do i=2, c%dim
             a%val=a%val+tmp_b%val(i,:)*c%val(i,1)
          end do
       case default
          ! someone could implement in_field type python
          FLAbort("Illegal in_field field type in inner_product()")
       end select

    case (FIELD_TYPE_CONSTANT)

       select case (c%field_type)
       case (FIELD_TYPE_NORMAL)
          a%val=b%val(1,1)*tmp_c%val(1,:)
          do i=2, c%dim
             a%val=a%val+b%val(i,1)*tmp_c%val(i,:)
          end do
       case (FIELD_TYPE_CONSTANT)
          a%val=b%val(1,1)*c%val(1,1)
          do i=2, c%dim
             a%val=a%val+b%val(i,1)*c%val(i,1)
          end do
       case default
          ! someone could implement in_field type python
          FLAbort("Illegal in_field field type in inner_product()")
       end select

    case default

       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in inner_product()")

    end select

    if (.not. c%mesh==tmp_c%mesh) then
       call deallocate(tmp_c)
    end if

  end subroutine inner_product_field_field

  subroutine inner_product_array_field(a, b, c)
    !!< Computes the node-wise inner/dot product a=b . c
    type(scalar_field), intent(inout) :: a
    real, dimension(:), intent(in) :: b
    type(vector_field), intent(in) :: c

    integer i

    assert(a%mesh%refcount%id==c%mesh%refcount%id)
    assert(a%field_type/=FIELD_TYPE_PYTHON)
    assert(a%field_type==FIELD_TYPE_NORMAL .or. c%field_type==FIELD_TYPE_CONSTANT)
    assert(size(b)==c%dim)

    select case (c%field_type)
    case (FIELD_TYPE_NORMAL)
       a%val=b(1)*c%val(1,:)
       do i=2, c%dim
         a%val=a%val+b(i)*c%val(i,:)
       end do
    case (FIELD_TYPE_CONSTANT)
       a%val=b(1)*c%val(1,1)
       do i=2, c%dim
         a%val=a%val+b(i)*c%val(i,1)
       end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in inner_product()")
    end select

  end subroutine inner_product_array_field

  subroutine inner_product_field_array(a, b, c)
    !!< Computes the node-wise inner/dot product a=b . c
    type(scalar_field), intent(inout) :: a
    type(vector_field), intent(in) :: b
    real, dimension(:), intent(in) :: c

    integer i

    assert(a%mesh%refcount%id==b%mesh%refcount%id)
    assert(a%field_type/=FIELD_TYPE_PYTHON)
    assert(a%field_type==FIELD_TYPE_NORMAL .or. b%field_type==FIELD_TYPE_CONSTANT)
    assert(size(c)==b%dim)

    select case (b%field_type)
    case (FIELD_TYPE_NORMAL)
       a%val=c(1)*b%val(1,:)
       do i=2, b%dim
         a%val=a%val+c(i)*b%val(i,:)
       end do
    case (FIELD_TYPE_CONSTANT)
       a%val=c(1)*b%val(1,1)
       do i=2, b%dim
         a%val=a%val+c(i)*b%val(i,1)
       end do
    case default
       ! someone could implement in_field type python
       FLAbort("Illegal in_field field type in inner_product()")
    end select

  end subroutine inner_product_field_array

  function get_patch_ele(mesh, node, level) result(patch)
    !!< This function takes in a node and returns a patch_type containing
    !!< information about the elements around this node.
    integer, intent(in) :: node
    type(mesh_type), intent(inout) :: mesh
    integer, optional, intent(in) :: level ! how many elements deep do you want the patch
    type(patch_type) :: patch

    integer :: i, j, k, l, llevel, ele, setsize
    integer, dimension(:), pointer :: ele_node_list
    type(csr_sparsity), pointer :: nelist

    if (present(level)) then
      llevel = level
    else
      llevel = 1
    end if

    nelist => extract_nelist(mesh)

    ! Compute level-1 patch.
    do j=nelist%findrm(node), nelist%findrm(node+1) - 1
      ele = nelist%colm(j)
      call eleset_add(ele)
    end do

    ! Compute any other levels.
    ! There's an obvious optimisation here for l > 2, but in
    ! practice I don't use l > 2, so I couldn't be bothered coding it.
    ! (The optimisation being don't check elements you've already checked)
    do l=2,llevel
      call eleset_get_size(setsize)
      do i=1,setsize
        call eleset_get_ele(i, ele)
        ele_node_list => ele_nodes(mesh, ele) ! get the nodes in that element
        do j=1,size(ele_node_list) ! loop over those nodes
          do k=nelist%findrm(ele_node_list(j)),nelist%findrm(ele_node_list(j)+1)-1 ! loop over their elements
            call eleset_add(nelist%colm(k)) ! add
          end do
        end do
      end do
    end do

    call eleset_get_size(patch%count)
    allocate(patch%elements(patch%count))
    call eleset_fetch_list(patch%elements)

  end function get_patch_ele

  function get_patch_node(mesh, node, level, min_nodes) result(patch)
    !!< This function takes in a node and returns a patch_type containing
    !!< information about the nodes around this node.
    integer, intent(in) :: node
    type(mesh_type), intent(inout) :: mesh
    integer, optional, intent(in) :: level ! how many elements deep do you want the patch
    integer, optional, intent(in) :: min_nodes ! how many nodes must be in the patch
    type(patch_type) :: patch

    integer :: i, j, k, l, llevel, nnode, nnnode, ele, setsize
    integer, dimension(:), pointer :: ele_node_list
    type(csr_sparsity), pointer :: nelist

    if (present(level)) then
      llevel = level
    else
      llevel = 1
    end if

    nelist => extract_nelist(mesh)

    ! Compute level-1 patch.
    do j=nelist%findrm(node), nelist%findrm(node+1) - 1
      ele = nelist%colm(j)
      ele_node_list => ele_nodes(mesh, ele)
      do k=1,size(ele_node_list)
        nnode = ele_node_list(k)
        call eleset_add(nnode)
      end do
    end do

    ! Compute any other levels.
    ! There's an obvious optimisation here for l > 2, but in
    ! practice I don't use l > 2, so I couldn't be bothered coding it.
    ! (The optimisation being don't check elements you've already checked)
    l = 0
    do
      ! Let's decide whether
      ! to exit or not.
      l = l + 1
      if (present(min_nodes)) then
        call eleset_get_size(setsize)
        if (setsize > min_nodes .and. l >= llevel) then
          exit
        end if
      else
        if (l >= llevel) then
          exit
        end if
      end if

      do i=1,setsize
        call eleset_get_ele(i, nnode)
        do j=nelist%findrm(nnode),nelist%findrm(nnode+1)-1 ! loop over their elements
          ele = nelist%colm(j)
          ele_node_list => ele_nodes(mesh, ele) ! loop over this elements' nodes
          do k=1,size(ele_node_list)
            nnnode = ele_node_list(k)
            call eleset_add(nnnode) ! add
          end do
        end do
      end do
    end do

    call eleset_get_size(patch%count)
    allocate(patch%elements(patch%count))
    call eleset_fetch_list(patch%elements)

  end function get_patch_node

  function clone_header_scalar(field) result(out_field)
    type(scalar_field), intent(in) :: field
    type(scalar_field) :: out_field

    out_field = field
    nullify(out_field%val)
  end function clone_header_scalar

  function clone_header_vector(field) result(out_field)
    type(vector_field), intent(in) :: field
    type(vector_field) :: out_field

    out_field = field
    nullify(out_field%val)

  end function clone_header_vector

  function clone_header_tensor(field) result(out_field)
    type(tensor_field), intent(in) :: field
    type(tensor_field) :: out_field

    out_field = field
    nullify(out_field%val)
  end function clone_header_tensor

  subroutine set_to_submesh_scalar(from_field, to_field)
    !!< Set the nodal values of a field on a higher order mesh to a field on its submesh.
    type(scalar_field), intent(in) :: from_field
    type(scalar_field), intent(inout) :: to_field

    integer :: vertices, from_ele, to_ele, l_ele
    integer, dimension(:,:), allocatable :: permutation
    real, dimension(:), allocatable :: from_vals
    integer, dimension(:), pointer :: to_nodes

    ewrite(1,*) 'entering set_to_submesh_scalar'

    assert(to_field%mesh%shape%degree==1)

    vertices = from_field%mesh%shape%quadrature%vertices

    select case(cell_family(from_field%mesh%shape))
    case(FAMILY_SIMPLEX)

      select case(from_field%mesh%shape%degree)
      case(2)

        select case(vertices)
        case(3) ! triangle
          assert(to_field%mesh%elements==4*from_field%mesh%elements)

          allocate(permutation(4,3))
          ! here we assume that the one true node ordering is used
          permutation = reshape((/1, 2, 2, 4, &
                                  2, 3, 4, 5, &
                                  4, 5, 5, 6/), (/4,3/))
        case(4) ! tet
          assert(to_field%mesh%elements==8*from_field%mesh%elements)

          allocate(permutation(8,4))
          ! here we assume that the one true node ordering is used
          ! also we arbitrarily select a diagonal (between 5 and 7) through the central octahedron
          permutation = reshape((/1, 2, 4,  7, 2, 2, 4, 5, &
                                  2, 3, 5,  8, 4, 5, 5, 7, &
                                  4, 5, 6,  9, 5, 7, 7, 8, &
                                  7, 8, 9, 10, 7, 8, 9, 9/), (/8,4/))
        case default
          FLAbort("unrecognised vertex count")
        end select
      case(1)
        !nothing to be done really

        select case(vertices)
        case(3) ! triangle
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,3))
          permutation = reshape((/1, 2, 3/), (/1,3/))
        case(4) ! tet
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,4))
          permutation = reshape((/1, 2, 3, 4/), (/1,4/))
        case default
          FLAbort("unrecognised vertex count")
        end select

      case default
        FLAbort("set_to_submesh_scalar only works for quadratic or lower elements")
      end select

    case default
      FLExit("set_to_submesh_scalar only works for simplex elements")
    end select

    allocate(from_vals(from_field%mesh%shape%ndof))

    to_ele = 0
    do from_ele = 1, element_count(from_field)
      from_vals=ele_val(from_field, from_ele)

      do l_ele = 1, size(permutation,1)
        to_ele = to_ele+1
        to_nodes=>ele_nodes(to_field, to_ele)
        call set(to_field, to_nodes, from_vals(permutation(l_ele,:)))
      end do

    end do

  end subroutine set_to_submesh_scalar

  subroutine set_to_submesh_vector(from_field, to_field)
    !!< Set the nodal values of a field on a higher order mesh to a field on its submesh.
    type(vector_field), intent(in) :: from_field
    type(vector_field), intent(inout) :: to_field

    integer :: vertices, from_ele, to_ele, l_ele
    integer, dimension(:,:), allocatable :: permutation
    real, dimension(:,:), allocatable :: from_vals
    integer, dimension(:), pointer :: to_nodes

    ewrite(1,*) 'entering set_to_submesh_vector'

    assert(to_field%mesh%shape%degree==1)

    vertices = from_field%mesh%shape%quadrature%vertices

    select case(cell_family(from_field%mesh%shape))
    case(FAMILY_SIMPLEX)

      select case(from_field%mesh%shape%degree)
      case(2)

        select case(vertices)
        case(3) ! triangle
          assert(to_field%mesh%elements==4*from_field%mesh%elements)

          allocate(permutation(4,3))
          ! here we assume that the one true node ordering is used
          permutation = reshape((/1, 2, 2, 4, &
                                  2, 3, 4, 5, &
                                  4, 5, 5, 6/), (/4,3/))
        case(4) ! tet
          assert(to_field%mesh%elements==8*from_field%mesh%elements)

          allocate(permutation(8,4))
          ! here we assume that the one true node ordering is used
          ! also we arbitrarily select a diagonal (between 5 and 7) through the central octahedron
          permutation = reshape((/1, 2, 4,  7, 2, 2, 4, 5, &
                                  2, 3, 5,  8, 4, 5, 5, 7, &
                                  4, 5, 6,  9, 5, 7, 7, 8, &
                                  7, 8, 9, 10, 7, 8, 9, 9/), (/8,4/))
        case default
          FLAbort("unrecognised vertex count")
        end select
      case(1)
        !nothing to be done really

        select case(vertices)
        case(3) ! triangle
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,3))
          permutation = reshape((/1, 2, 3/), (/1,3/))
        case(4) ! tet
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,4))
          permutation = reshape((/1, 2, 3, 4/), (/1,4/))
        case default
          FLAbort("unrecognised vertex count")
        end select

      case default
        FLAbort("set_to_submesh_vector only works for quadratic or lower elements")
      end select

    case default
      FLExit("set_to_submesh_vector only works for simplex elements")
    end select

    allocate(from_vals(from_field%dim, from_field%mesh%shape%ndof))

    to_ele = 0
    do from_ele = 1, element_count(from_field)
      from_vals=ele_val(from_field, from_ele)

      do l_ele = 1, size(permutation,1)
        to_ele = to_ele+1
        to_nodes=>ele_nodes(to_field, to_ele)
        call set(to_field, to_nodes, from_vals(:, permutation(l_ele,:)))
      end do

    end do

  end subroutine set_to_submesh_vector

  subroutine set_from_submesh_scalar(from_field, to_field)
    !!< Set the nodal values of a field on a lower order submesh to a field on its parent mesh.
    type(scalar_field), intent(in) :: from_field
    type(scalar_field), intent(inout) :: to_field

    integer :: vertices, from_ele, to_ele, l_ele
    integer, dimension(:,:), allocatable :: permutation
    real, dimension(:), allocatable :: from_vals
    integer, dimension(:), pointer :: to_nodes

    ewrite(1,*) 'entering set_from_submesh_scalar'

    assert(from_field%mesh%shape%degree==1)

    vertices = to_field%mesh%shape%quadrature%vertices

    select case(cell_family(to_field%mesh%shape))
    case(FAMILY_SIMPLEX)

      select case(to_field%mesh%shape%degree)
      case(2)

        select case(vertices)
        case(3) ! triangle
          assert(4*to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(4,3))
          ! here we assume that the one true node ordering is used
          permutation = reshape((/1, 2, 2, 4, &
                                  2, 3, 4, 5, &
                                  4, 5, 5, 6/), (/4,3/))
        case(4) ! tet
          assert(8*to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(8,4))
          ! here we assume that the one true node ordering is used
          ! also we arbitrarily select a diagonal (between 5 and 7) through the central octahedron
          permutation = reshape((/1, 2, 4,  7, 2, 2, 4, 5, &
                                  2, 3, 5,  8, 4, 5, 5, 7, &
                                  4, 5, 6,  9, 5, 7, 7, 8, &
                                  7, 8, 9, 10, 7, 8, 9, 9/), (/8,4/))
        case default
          FLAbort("unrecognised vertex count")
        end select
      case(1)
        !nothing to be done really

        select case(vertices)
        case(3) ! triangle
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,3))
          permutation = reshape((/1, 2, 3/), (/1,3/))
        case(4) ! tet
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,4))
          permutation = reshape((/1, 2, 3, 4/), (/1,4/))
        case default
          FLExit("unrecognised vertex count")
        end select

      case default
        FLAbort("set_from_submesh_vector only works for quadratic or lower elements")
      end select

    case default
      FLAbort("set_from_submesh_vector only works for simplex elements")
    end select

    allocate(from_vals(from_field%mesh%shape%ndof))

    from_ele = 0
    do to_ele = 1, element_count(to_field)
      to_nodes=>ele_nodes(to_field, to_ele)

      do l_ele = 1, size(permutation,1)

        from_ele = from_ele + 1
        from_vals=ele_val(from_field, from_ele)

        call set(to_field, to_nodes(permutation(l_ele,:)), from_vals)

      end do

    end do

  end subroutine set_from_submesh_scalar

  subroutine set_from_submesh_vector(from_field, to_field)
    !!< Set the nodal values of a field on a lower order submesh to a field on its parent mesh.
    type(vector_field), intent(in) :: from_field
    type(vector_field), intent(inout) :: to_field

    integer :: vertices, from_ele, to_ele, l_ele
    integer, dimension(:,:), allocatable :: permutation
    real, dimension(:,:), allocatable :: from_vals
    integer, dimension(:), pointer :: to_nodes

    ewrite(1,*) 'entering set_from_submesh_vector'

    assert(from_field%mesh%shape%degree==1)

    vertices = to_field%mesh%shape%quadrature%vertices

    select case(cell_family(to_field%mesh%shape))
    case(FAMILY_SIMPLEX)

      select case(to_field%mesh%shape%degree)
      case(2)

        select case(vertices)
        case(3) ! triangle
          assert(4*to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(4,3))
          ! here we assume that the one true node ordering is used
          permutation = reshape((/1, 2, 2, 4, &
                                  2, 3, 4, 5, &
                                  4, 5, 5, 6/), (/4,3/))
        case(4) ! tet
          assert(8*to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(8,4))
          ! here we assume that the one true node ordering is used
          ! also we arbitrarily select a diagonal (between 5 and 7) through the central octahedron
          permutation = reshape((/1, 2, 4,  7, 2, 2, 4, 5, &
                                  2, 3, 5,  8, 4, 5, 5, 7, &
                                  4, 5, 6,  9, 5, 7, 7, 8, &
                                  7, 8, 9, 10, 7, 8, 9, 9/), (/8,4/))
        case default
          FLAbort("unrecognised vertex count")
        end select
      case(1)
        !nothing to be done really

        select case(vertices)
        case(3) ! triangle
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,3))
          permutation = reshape((/1, 2, 3/), (/1,3/))
        case(4) ! tet
          assert(to_field%mesh%elements==from_field%mesh%elements)

          allocate(permutation(1,4))
          permutation = reshape((/1, 2, 3, 4/), (/1,4/))
        case default
          FLExit("unrecognised vertex count")
        end select

      case default
        FLAbort("set_from_submesh_vector only works for quadratic or lower elements")
      end select

    case default
      FLAbort("set_from_submesh_vector only works for simplex elements")
    end select

    allocate(from_vals(from_field%dim, from_field%mesh%shape%ndof))

    from_ele = 0
    do to_ele = 1, element_count(to_field)
      to_nodes=>ele_nodes(to_field, to_ele)

      do l_ele = 1, size(permutation,1)

        from_ele = from_ele + 1
        from_vals=ele_val(from_field, from_ele)

        call set(to_field, to_nodes(permutation(l_ele,:)), from_vals)

      end do

    end do

  end subroutine set_from_submesh_vector

  subroutine set_ele_nodes(mesh, ele, nodes)
    type(mesh_type), intent(inout) :: mesh
    integer, intent(in) :: ele
    integer, dimension(:), intent(in) :: nodes

    assert(size(nodes) == ele_loc(mesh, ele))

    mesh%ndglno(mesh%shape%ndof*(ele-1)+1:&
                &mesh%shape%ndof*ele) = nodes
  end subroutine set_ele_nodes

  subroutine renumber_positions_trailing_receives(positions, permutation)
    type(vector_field), intent(inout) :: positions
    integer, dimension(:), intent(out), optional :: permutation

    integer :: i, j, nhalos, nonods
    integer, dimension(:), allocatable :: inverse_permutation, receive_node, &
      & renumber_permutation
    type(vector_field) :: positions_renumbered

    ewrite(1, *) "In renumber_positions_trailing_receives"
    assert(positions%refcount%count == 1)

    nhalos = halo_count(positions)
    if(nhalos == 0) return

    nonods = node_count(positions)

    allocate(receive_node(nonods))
    allocate(renumber_permutation(nonods))
    allocate(inverse_permutation(nonods))
    receive_node = 0
    do i = nhalos, 1, -1
      do j = 1, halo_proc_count(positions%mesh%halos(i))
        receive_node(halo_receives(positions%mesh%halos(i), j)) = i
      end do
    end do
    call qsort(receive_node, renumber_permutation)
    do i=1 , size(renumber_permutation)
      inverse_permutation(renumber_permutation(i))=i
    end do

    call renumber_positions(positions, inverse_permutation, positions_renumbered, &
      & node_halo_ordering_scheme = HALO_ORDER_TRAILING_RECEIVES)

    call deallocate(positions)
    positions = positions_renumbered

    if(present(permutation)) then
      assert(size(permutation)==nonods)
      permutation=inverse_permutation
    end if

    deallocate(receive_node)
    deallocate(renumber_permutation)
    deallocate(inverse_permutation)

#ifdef DDEBUG
    do i = 1, nhalos
      !if (.not. has_references(positions%mesh%halos(i))) cycle
      assert(trailing_receives_consistent(positions%mesh%halos(i)))
    end do
#endif

    ewrite(1, *) "Exiting renumber_positions_trailing_receives"

  end subroutine renumber_positions_trailing_receives

  subroutine renumber_positions(input_positions, permutation, output_positions, node_halo_ordering_scheme)
    type(vector_field), intent(in) :: input_positions
    type(vector_field), intent(out) :: output_positions
    integer, dimension(:), intent(in) :: permutation
    !! As we're reordering nodes it is assumed that output node halos should
    !! use a general ordering scheme, unless explicitly overridden via this
    !! argument
    integer, optional, intent(in) :: node_halo_ordering_scheme

    type(mesh_type) :: output_mesh
    integer :: ele, node, halo_num, lnode_halo_ordering_scheme, proc
    type(halo_type), pointer :: input_halo, output_halo

    ewrite(1, *) "In renumber_positions"

    assert(size(permutation) == node_count(input_positions))

    if(present(node_halo_ordering_scheme)) then
      lnode_halo_ordering_scheme = node_halo_ordering_scheme
    else
      lnode_halo_ordering_scheme = HALO_ORDER_GENERAL
    end if

    call allocate(output_mesh, node_count(input_positions), ele_count(input_positions), &
               &  input_positions%mesh%shape, trim(input_positions%mesh%name))

    do ele=1,ele_count(input_positions)
      call set_ele_nodes(output_mesh, ele, permutation(ele_nodes(input_positions, ele)))
    end do

    if(associated(input_positions%mesh%columns)) then
      allocate(output_mesh%columns(node_count(input_positions)))
      do node=1,node_count(input_positions)
        output_mesh%columns(permutation(node)) = input_positions%mesh%columns(node)
      end do
    end if

    output_mesh%periodic = input_positions%mesh%periodic
    if (associated(input_positions%mesh%region_ids)) then
      call allocate_region_ids(output_mesh, size(input_positions%mesh%region_ids))
      output_mesh%region_ids = input_positions%mesh%region_ids
    end if

    ! Now here comes the damnable face information

    if (associated(input_positions%mesh%faces)) then
      allocate(output_mesh%faces)
      allocate(output_mesh%faces%shape)
      output_mesh%faces%shape = input_positions%mesh%faces%shape
      call incref(output_mesh%faces%shape)
      call incref(output_mesh%faces%shape%quadrature)
      output_mesh%faces%face_list = input_positions%mesh%faces%face_list
      call incref(output_mesh%faces%face_list)
      allocate(output_mesh%faces%face_lno(size(input_positions%mesh%faces%face_lno)))
      output_mesh%faces%face_lno = input_positions%mesh%faces%face_lno
      output_mesh%faces%surface_mesh = input_positions%mesh%faces%surface_mesh
      call incref(output_mesh%faces%surface_mesh)
      allocate(output_mesh%faces%surface_node_list(size(input_positions%mesh%faces%surface_node_list)))
      output_mesh%faces%surface_node_list = permutation(input_positions%mesh%faces%surface_node_list)
      allocate(output_mesh%faces%face_element_list(size(input_positions%mesh%faces%face_element_list)))
      output_mesh%faces%face_element_list = input_positions%mesh%faces%face_element_list
      allocate(output_mesh%faces%local_face_number(size(input_positions%mesh%faces%local_face_number)))
      output_mesh%faces%local_face_number = input_positions%mesh%faces%local_face_number

      output_mesh%faces%boundary_ids_c = malloc(size(input_positions%mesh%faces%boundary_ids) *&
           & c_sizeof(1_c_int))
      call c_f_pointer(output_mesh%faces%boundary_ids_c, output_mesh%faces%boundary_ids,&
           & [size(input_positions%mesh%faces%boundary_ids)])
      output_mesh%faces%boundary_ids = input_positions%mesh%faces%boundary_ids
      if(associated(input_positions%mesh%faces%coplanar_ids)) then
        allocate(output_mesh%faces%coplanar_ids(size(input_positions%mesh%faces%coplanar_ids)))
        output_mesh%faces%coplanar_ids = input_positions%mesh%faces%coplanar_ids
      end if
      if (associated(input_positions%mesh%faces%dg_surface_mesh)) then
        allocate(output_mesh%faces%dg_surface_mesh)
        output_mesh%faces%dg_surface_mesh = input_positions%mesh%faces%dg_surface_mesh
        call incref(output_mesh%faces%dg_surface_mesh)
      end if
    end if

    output_mesh%option_path = input_positions%mesh%option_path
    output_mesh%continuity = input_positions%mesh%continuity

    ! Now for the positions
    call allocate(output_positions, input_positions%dim, output_mesh, trim(input_positions%name))
    do node=1,node_count(output_positions)
      call set(output_positions, permutation(node), node_val(input_positions, node))
    end do
    call deallocate(output_mesh)

    ! Node halos
    allocate(output_positions%mesh%halos(halo_count(input_positions)))
    do halo_num = 1, halo_count(input_positions)
      if (.not. has_references(input_positions%mesh%halos(halo_num))) cycle
      input_halo => input_positions%mesh%halos(halo_num)
      output_halo => output_positions%mesh%halos(halo_num)
      call allocate(output_halo, input_halo)
      call set_halo_ordering_scheme(output_halo, lnode_halo_ordering_scheme)
      do proc = 1, halo_proc_count(input_halo)
        call set_halo_sends(output_halo, proc, permutation(halo_sends(input_halo, proc)))
        call set_halo_receives(output_halo, proc, permutation(halo_receives(input_halo, proc)))
      end do

      ! Create caches
      call create_ownership(output_halo)
      call create_global_to_universal_numbering(output_halo)
      assert(has_global_to_universal_numbering(output_halo))
    end do
    ! Element halos
    allocate(output_positions%mesh%element_halos(element_halo_count(input_positions)))
    do halo_num = 1, element_halo_count(input_positions)
      if (.not. has_references(input_positions%mesh%element_halos(halo_num))) cycle
      output_positions%mesh%element_halos(halo_num) = input_positions%mesh%element_halos(halo_num)
      call incref(output_positions%mesh%element_halos(halo_num))
    end do

    call refresh_topology(output_positions)

    output_positions%option_path = input_positions%option_path

    ewrite(1, *) "Exiting renumber_positions"

  end subroutine renumber_positions

  subroutine renumber_positions_elements(input_positions, permutation, output_positions, element_halo_ordering_scheme)
    type(vector_field), intent(in) :: input_positions
    type(vector_field), intent(out) :: output_positions
    integer, dimension(:), intent(in) :: permutation
    !! As we're reordering nodes it is assumed that output halos should
    !! use a general ordering scheme, unless explicitly overridden via this
    !! argument
    integer, optional, intent(in) :: element_halo_ordering_scheme

    type(mesh_type) :: output_mesh
    integer :: ele, node, halo_num, lelement_halo_ordering_scheme, proc
    type(halo_type), pointer :: input_halo, output_halo
    integer, dimension(:), allocatable :: sndgln

    ewrite(1, *) "In renumber_positions_elements"

    assert(size(permutation) == ele_count(input_positions))

    if(present(element_halo_ordering_scheme)) then
      lelement_halo_ordering_scheme = element_halo_ordering_scheme
    else
      lelement_halo_ordering_scheme = HALO_ORDER_GENERAL
    end if

    call allocate(output_mesh, node_count(input_positions), ele_count(input_positions), &
               &  input_positions%mesh%shape, trim(input_positions%mesh%name))

    do ele=1,ele_count(input_positions)
      call set_ele_nodes(output_mesh, permutation(ele), ele_nodes(input_positions, ele))
    end do

    if(associated(input_positions%mesh%columns)) then
      allocate(output_mesh%columns(node_count(input_positions)))
      output_mesh%columns = input_positions%mesh%columns
    end if

    output_mesh%periodic = input_positions%mesh%periodic
    if (associated(input_positions%mesh%region_ids)) then
      call allocate_region_ids(output_mesh, size(input_positions%mesh%region_ids))
      output_mesh%region_ids = input_positions%mesh%region_ids
    end if

    ! Now here comes the damnable face information

    if (associated(input_positions%mesh%faces)) then
      allocate(sndgln(surface_element_count(input_positions) * face_loc(input_positions, 1)))
      call getsndgln(input_positions%mesh, sndgln)
      call add_faces(output_mesh, sndgln=sndgln, element_owner=permutation(input_positions%mesh%faces%face_element_list(1:surface_element_count(input_positions))))
      deallocate(sndgln)
      output_mesh%faces%boundary_ids = input_positions%mesh%faces%boundary_ids
      if (associated(input_positions%mesh%faces%coplanar_ids)) then
        allocate(output_mesh%faces%coplanar_ids(size(input_positions%mesh%faces%coplanar_ids)))
        output_mesh%faces%coplanar_ids = input_positions%mesh%faces%coplanar_ids
      end if
    end if

    output_mesh%option_path = input_positions%mesh%option_path
    output_mesh%continuity = input_positions%mesh%continuity

    ! Now for the positions
    call allocate(output_positions, input_positions%dim, output_mesh, trim(input_positions%name))
    do node=1,node_count(output_positions)
      call set(output_positions, node, node_val(input_positions, node))
    end do
    call deallocate(output_mesh)

    ! Node halos
    allocate(output_positions%mesh%halos(halo_count(input_positions)))
    do halo_num = 1, halo_count(input_positions)
      if (.not. has_references(input_positions%mesh%halos(halo_num))) cycle
      output_positions%mesh%halos(halo_num) = input_positions%mesh%halos(halo_num)
      call incref(output_positions%mesh%halos(halo_num))
    end do

    ! Element halos
    allocate(output_positions%mesh%element_halos(element_halo_count(input_positions)))
    do halo_num = 1, element_halo_count(input_positions)
      if (.not. has_references(input_positions%mesh%element_halos(halo_num))) cycle
      input_halo => input_positions%mesh%element_halos(halo_num)
      output_halo => output_positions%mesh%element_halos(halo_num)
      call allocate(output_halo, input_halo)
      call set_halo_ordering_scheme(output_halo, lelement_halo_ordering_scheme)
      do proc = 1, halo_proc_count(input_halo)
        call set_halo_sends(output_halo, proc, permutation(halo_sends(input_halo, proc)))
        call set_halo_receives(output_halo, proc, permutation(halo_receives(input_halo, proc)))
      end do

      ! Create caches
      call create_ownership(output_halo)
      call create_global_to_universal_numbering(output_halo)
      assert(has_global_to_universal_numbering(output_halo))
    end do

    output_positions%option_path = input_positions%option_path

    call refresh_topology(output_positions)
    ewrite(1, *) "Exiting renumber_positions_elements"

  end subroutine renumber_positions_elements

  subroutine renumber_positions_elements_trailing_receives(positions, permutation)
    type(vector_field), intent(inout) :: positions
    integer, dimension(:), intent(out), optional :: permutation

    integer :: i, j, nhalos, elmcnt
    integer, dimension(:), allocatable :: inverse_permutation, receive_node, &
      & renumber_permutation
    type(vector_field) :: positions_renumbered

    ewrite(1, *) "In renumber_positions_elements_trailing_receives"

    assert(positions%refcount%count == 1)

    nhalos = element_halo_count(positions)
    if(nhalos == 0) return

    elmcnt = ele_count(positions)

    allocate(receive_node(elmcnt))
    allocate(renumber_permutation(elmcnt))
    allocate(inverse_permutation(elmcnt))
    receive_node = 0
    do i = nhalos, 1, -1
      do j = 1, halo_proc_count(positions%mesh%element_halos(i))
        receive_node(halo_receives(positions%mesh%element_halos(i), j)) = i
      end do
    end do
    call qsort(receive_node, renumber_permutation)
    do i=1,size(renumber_permutation)
      inverse_permutation(renumber_permutation(i)) = i
    end do

    call renumber_positions_elements(positions, inverse_permutation, positions_renumbered, &
      & element_halo_ordering_scheme = HALO_ORDER_TRAILING_RECEIVES)

    call deallocate(positions)
    positions = positions_renumbered

    if (present(permutation)) then
      assert(size(permutation) == elmcnt)
      permutation = inverse_permutation
    end if

    deallocate(receive_node)
    deallocate(renumber_permutation)
    deallocate(inverse_permutation)

#ifdef DDEBUG
    do i = 1, nhalos
      assert(trailing_receives_consistent(positions%mesh%element_halos(i)))
    end do
#endif

    ewrite(1, *) "Exiting renumber_positions_elements_trailing_receives"

  end subroutine renumber_positions_elements_trailing_receives

  subroutine reorder_element_numbering(positions, use_unns)
    !!< On return from adaptivity, the element node list for halo elements
    !!< contains arbitrary reorderings. This routine reorders the element
    !!< node lists so that they are consistent accross all processes.

    type(vector_field), target, intent(inout) :: positions
    !! Supply this to override unn caches on the positions field. Useful for
    !! reordering before caches have been generated.
    type(integer_set), dimension(:), intent(in), optional :: use_unns

    integer :: tmp, ele, nhalos
    ! Note that this is invalid for mixed geometry meshes, but adaptivity
    ! doesn't support those anyway!
    integer, dimension(ele_loc(positions,1)) :: unns, unns_order
    integer, dimension(:), allocatable :: sndgln
    integer, dimension(:), pointer :: nodes
    type(mesh_type), pointer :: mesh

    mesh => positions%mesh
    if(has_faces(mesh)) then
      allocate(sndgln(face_loc(mesh, 1) * surface_element_count(mesh)))
      call getsndgln(mesh, sndgln)
    end if

    nhalos = halo_count(mesh)
    if((nhalos == 0).and.(.not.present(use_unns))) then
      FLAbort("Need halos or unns to reorder the mesh.")
    end if

    do ele = 1, element_count(mesh)
      nodes => ele_nodes(mesh, ele)

      if(present(use_unns)) then
        unns = set2vector(use_unns(ele))
      else
        ! Get the universal numbers from the largest available halo
        unns = halo_universal_numbers(mesh%halos(nhalos), nodes)
      end if

      call qsort(unns, unns_order)
      call apply_permutation(nodes, unns_order)

    end do

    ! Now we have the nodes in a known order. However, some elements may
    ! be inverted. This is only an issue in 3D.

    if(mesh_dim(mesh) == 3) then
      do ele = 1, element_count(mesh)
        nodes => ele_nodes(mesh, ele)
        if(simplex_volume(positions, ele) < 0.0) then
          tmp = nodes(1)
          nodes(1) = nodes(2)
          nodes(2) = tmp
        end if
      end do
    end if

    call remove_eelist(mesh)
    if(has_faces(mesh)) then
      call update_faces(mesh, sndgln)
      deallocate(sndgln)
    end if

  contains

    subroutine update_faces(mesh, sndgln)
      type(mesh_type), intent(inout) :: mesh
      integer, dimension(face_loc(mesh, 1) * surface_element_count(mesh)), intent(in) :: sndgln

      integer, dimension(surface_element_count(mesh)) :: boundary_ids
      integer, dimension(:), allocatable :: coplanar_ids, element_owners

      assert(has_faces(mesh))

      boundary_ids = mesh%faces%boundary_ids
      if(associated(mesh%faces%coplanar_ids)) then
        allocate(coplanar_ids(surface_element_count(mesh)))
        coplanar_ids = mesh%faces%coplanar_ids
      end if

      allocate(element_owners((surface_element_count(mesh))))
      element_owners = mesh%faces%face_element_list(1:surface_element_count(mesh))

      call deallocate_faces(mesh)
      call add_faces(mesh, sndgln = sndgln, element_owner=element_owners)
      mesh%faces%boundary_ids = boundary_ids
      if(allocated(coplanar_ids)) then
        allocate(mesh%faces%coplanar_ids(size(coplanar_ids)))
        mesh%faces%coplanar_ids = coplanar_ids
        deallocate(coplanar_ids)
      end if
      deallocate(element_owners)

    end subroutine update_faces

  end subroutine reorder_element_numbering

  subroutine remap_to_subdomain_scalar(parent_field,sub_field)
    !!< remaps scalar fields from full domain to sub_domain:
    type(scalar_field), intent(in) :: parent_field
    type(scalar_field), intent(inout) :: sub_field
    integer, dimension(:), pointer :: node_map

    assert(associated(sub_field%mesh%subdomain_mesh%node_list))
    node_map => sub_field%mesh%subdomain_mesh%node_list

    if(parent_field%field_type == FIELD_TYPE_CONSTANT) then
       call set(sub_field,node_val(parent_field,1))
    else
       call set_all(sub_field, node_val(parent_field,node_map))
    end if

  end subroutine remap_to_subdomain_scalar

  subroutine remap_to_subdomain_vector(parent_field,sub_field)

    type(vector_field), intent(in) :: parent_field
    type(vector_field), intent(inout) :: sub_field
    integer, dimension(:), pointer :: node_map

    assert(associated(sub_field%mesh%subdomain_mesh%node_list))
    node_map => sub_field%mesh%subdomain_mesh%node_list

    if(parent_field%field_type == FIELD_TYPE_CONSTANT) then
       call set(sub_field,node_val(parent_field,1))
    else
       call set_all(sub_field, node_val(parent_field,node_map))
    end if

  end subroutine remap_to_subdomain_vector

  subroutine remap_to_subdomain_tensor(parent_field,sub_field)

    type(tensor_field), intent(in) :: parent_field
    type(tensor_field), intent(inout) :: sub_field
    integer, dimension(:), pointer :: node_map

    assert(associated(sub_field%mesh%subdomain_mesh%node_list))
    node_map => sub_field%mesh%subdomain_mesh%node_list

    if(parent_field%field_type == FIELD_TYPE_CONSTANT) then
       call set(sub_field,node_val(parent_field,1))
    else
       call set_all(sub_field, node_val(parent_field,node_map))
    end if

  end subroutine remap_to_subdomain_tensor

  subroutine remap_to_full_domain_scalar(sub_field,parent_field)
    !!< remaps scalar fields from sub_domain to full_domain:
    type(scalar_field), intent(in) :: sub_field
    type(scalar_field), intent(inout) :: parent_field
    integer, dimension(:), pointer :: node_map
    integer :: inode

    assert(associated(sub_field%mesh%subdomain_mesh%node_list))
    node_map => sub_field%mesh%subdomain_mesh%node_list

    if(parent_field%field_type == FIELD_TYPE_CONSTANT) then
       call set(parent_field,node_val(sub_field,1))
    else
       do inode = 1, size(node_map)
          call set(parent_field, node_map(inode), node_val(sub_field,inode))
       end do
    end if

  end subroutine remap_to_full_domain_scalar

  subroutine remap_to_full_domain_vector(sub_field,parent_field)
    type(vector_field), intent(in) :: sub_field
    type(vector_field), intent(inout) :: parent_field
    integer, dimension(:), pointer :: node_map
    integer :: inode

    assert(associated(sub_field%mesh%subdomain_mesh%node_list))
    node_map => sub_field%mesh%subdomain_mesh%node_list

    if(parent_field%field_type == FIELD_TYPE_CONSTANT) then
       call set(parent_field,node_val(sub_field,1))
    else
       do inode = 1, size(node_map)
          call set(parent_field, node_map(inode), node_val(sub_field,inode))
       end do
    end if

  end subroutine remap_to_full_domain_vector

  subroutine remap_to_full_domain_tensor(sub_field,parent_field)
    type(tensor_field), intent(in) :: sub_field
    type(tensor_field), intent(inout) :: parent_field
    integer, dimension(:), pointer :: node_map
    integer :: inode

    assert(associated(sub_field%mesh%subdomain_mesh%node_list))
    node_map => sub_field%mesh%subdomain_mesh%node_list

    if(parent_field%field_type == FIELD_TYPE_CONSTANT) then
       call set(parent_field,node_val(sub_field,1))
    else
       do inode = 1, size(node_map)
          call set(parent_field, node_map(inode), node_val(sub_field,inode))
       end do
    end if

  end subroutine remap_to_full_domain_tensor

  function get_remapped_coordinates(positions, mesh) result(remapped_positions)
    type(vector_field), intent(in):: positions
    type(mesh_type), intent(inout):: mesh
    type(vector_field):: remapped_positions

    integer:: stat

    call allocate(remapped_positions, positions%dim, mesh, "RemappedCoordinates")
    call remap_field(positions, remapped_positions, stat=stat)
    ! we allow stat==REMAP_ERR_UNPERIODIC_PERIODIC, to create periodic surface positions with coordinates
    ! at the periodic boundary having a value that is only determined upto a random number of periodic mappings
    if(stat==REMAP_ERR_DISCONTINUOUS_CONTINUOUS) then
      ewrite(-1,*) 'Remapping of the coordinates just threw an error because'
      ewrite(-1,*) 'the input coordinates are discontinuous and you are trying'
      ewrite(-1,*) 'to remap them to a continuous field.'
      FLAbort("Why are your coordinates discontinuous?")
    else if ((stat/=0).and. &
             (stat/=REMAP_ERR_UNPERIODIC_PERIODIC).and. &
             (stat/=REMAP_ERR_BUBBLE_LAGRANGE).and. &
             (stat/=REMAP_ERR_HIGHER_LOWER_CONTINUOUS)) then
      FLAbort('Unknown error when remapping coordinates')
    end if

  end function get_remapped_coordinates

  function get_coordinates_remapped_to_surface(positions, surface_mesh, surface_element_list) result(surface_positions)
    type(vector_field), intent(in):: positions
    type(mesh_type), intent(inout):: surface_mesh
    integer, dimension(:), intent(in):: surface_element_list
    type(vector_field):: surface_positions

    integer:: stat

    call allocate(surface_positions, positions%dim, surface_mesh, "RemappedSurfaceCoordinates")
    call remap_field_to_surface(positions, surface_positions, surface_element_list, stat=stat)
    ! we allow stat==REMAP_ERR_UNPERIODIC_PERIODIC, to create periodic surface positions with coordinates
    ! at the periodic boundary having a value that is only determined upto a random number of periodic mappings
    if(stat==REMAP_ERR_DISCONTINUOUS_CONTINUOUS) then
      ewrite(-1,*) 'Remapping of the coordinates just threw an error because'
      ewrite(-1,*) 'the input coordinates are discontinuous and you are trying'
      ewrite(-1,*) 'to remap them to a continuous field.'
      FLAbort("Why are your coordinates discontinuous?")
    else if ((stat/=0).and. &
             (stat/=REMAP_ERR_UNPERIODIC_PERIODIC).and. &
             (stat/=REMAP_ERR_BUBBLE_LAGRANGE).and. &
             (stat/=REMAP_ERR_HIGHER_LOWER_CONTINUOUS)) then
      FLAbort('Unknown error in mapping coordinates from mesh to surface')
    end if

  end function get_coordinates_remapped_to_surface

end module fields_manipulation

