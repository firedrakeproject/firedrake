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
module fields_calculations

use elements
use fields_allocates
use fields_data_types
use fields_base
use fields_manipulation
use fetools
use parallel_fields
use parallel_tools
use vector_tools
use supermesh_construction
use intersection_finder_module
use linked_lists

implicit none

  interface mean
     module procedure mean_scalar
  end interface

  interface maxval
     module procedure maxval_scalar
  end interface

  interface minval
     module procedure minval_scalar
  end interface

  interface sum
     module procedure sum_scalar
  end interface

  interface norm2
     module procedure norm2_scalar
  end interface

  interface field_stats
     module procedure field_stats_scalar, field_stats_vector, field_stats_tensor
  end interface

  interface field_cv_stats
     module procedure field_cv_stats_scalar
  end interface

  interface field_con_stats
     module procedure field_con_stats_scalar, field_con_stats_vector
  end interface

  interface field_integral
     module procedure integral_scalar, integral_vector
  end interface

  interface fields_integral
     module procedure integral_scalars
  end interface

  interface function_val_at_quad
     module procedure function_val_at_quad_scalar, function_val_at_quad_vector
  end interface

  interface dot_product
    module procedure dot_product_scalar, dot_product_vector
  end interface dot_product

  interface outer_product
    module procedure outer_product_vector
 end interface

  interface norm2_difference
    module procedure norm2_difference_single, norm2_difference_multiple
  end interface

  integer, parameter, public :: CONVERGENCE_INFINITY_NORM=0, CONVERGENCE_L2_NORM=1, CONVERGENCE_CV_L2_NORM=2

  contains

  function magnitude(field)
    !!< Return a scalar field which is the magnitude of the vector field.
    type(scalar_field) :: magnitude
    type(vector_field), intent(inout) :: field

    integer :: node

    call allocate(magnitude, field%mesh,trim(field%name)//"Magnitude")

    do node=1,node_count(field)
       magnitude%val(node)=norm2(node_val(field, node))
    end do

  end function magnitude

  function magnitude_tensor(field)
    !!< Return a scalar field which is the magnitude of the tensor field.
    type(scalar_field) :: magnitude_tensor
    type(tensor_field), intent(inout) :: field

    integer :: node

    call allocate(magnitude_tensor, field%mesh,trim(field%name)//"Magnitude")

    do node=1,node_count(field)
       magnitude_tensor%val(node)=norm2(node_val(field, node))
    end do

  end function magnitude_tensor

  pure function mean_scalar(field) result (mean)
    !!< Return the mean value of a field
    real :: mean
    type(scalar_field), intent(in) :: field

    mean = sum(field%val)/size(field%val)
  end function mean_scalar

  pure function maxval_scalar(field) result (max)
    !!< Return the maximum value in a field.
    real :: max
    type(scalar_field), intent(in) :: field

    max=maxval(field%val)

  end function maxval_scalar

  pure function minval_scalar(field) result (min)
    !!< Return the maximum value in a field.
    real :: min
    type(scalar_field), intent(in) :: field

    min=minval(field%val)

  end function minval_scalar

  pure function sum_scalar(field) result (sumval)
    !!< Return the sum of the values of a field
    real :: sumval
    type(scalar_field), intent(in) :: field

    sumval = sum(field%val)
  end function sum_scalar

  function norm2_scalar(field, X) result (norm)
    !!< Return the L2 norm of field:
    !!<   /
    !!<  (| |field|^2 dV)^(1/2)
    !!<   /
    real :: norm
    type(scalar_field), intent(in) :: field
    !! The positions field associated with field.
    type(vector_field), intent(in) :: X

    integer :: ele

    norm=0

    do ele=1, element_count(field)
      if(element_owned(field, ele)) then
        norm=norm+norm2(field, X, ele)
      end if
    end do

    call allsum(norm)

    norm = sqrt(norm)

  end function norm2_scalar

  function norm2_scalar_cv(field, cv_mass) result (norm)
    !!< Return the L2 norm of CV field:
    !!<   /
    !!<   | |field|^2 dV
    !!<   /
    real :: norm
    type(scalar_field), intent(in) :: field
    type(scalar_field), intent(in) :: cv_mass

    assert(node_count(field)==node_count(cv_mass))

    norm = dot_product(cv_mass%val, field%val**2)

    call allsum(norm)

    norm = sqrt(norm)

  end function norm2_scalar_cv

  function integral_scalar(field, X) result (integral)
    !!< Integrate field over its mesh.
    real :: integral
    type(scalar_field), intent(in) :: field
    !! The positions field associated with field.
    type(vector_field), intent(in) :: X

    integer :: ele

    integral=0

    do ele=1, element_count(field)
      if(element_owned(field, ele)) then
        integral=integral &
            +integral_element(field, X, ele)
      end if
    end do

    call allsum(integral)

  end function integral_scalar

  function integral_vector(field, X) result (integral)
    !!< Integrate field over its mesh.
    type(vector_field), intent(in) :: field
    real, dimension(field%dim) :: integral
    !! The positions field associated with field.
    type(vector_field), intent(in) :: X

    ! Note: this is much slower than it needs to be because
    ! it does the integration twice. Don't use for anything
    ! important!

    integer :: ele, i

    integral=0

    do ele=1, element_count(field)
       if(element_owned(field, ele)) then
          integral=integral &
               +integral_element(field, X, ele)
       end if
    end do

    do i=1,field%dim
      call allsum(integral(i))
    end do

  end function integral_vector

  function integral_scalar_cv(field, cv_mass) result (integral)
    !!< Integrate CV field over its mesh.
    real :: integral
    type(scalar_field), intent(in) :: field
    type(scalar_field), intent(in) :: cv_mass

    assert(node_count(field)==node_count(cv_mass))

    integral = dot_product(cv_mass%val, field%val)

    call allsum(integral)

  end function integral_scalar_cv

  function integral_scalars(fields, X, region_ids) result (integral)
    !!< Integrate the product of fields assuming the same coordinate mesh X.
    !!< If region ids is present then only integrate these associated regions
    !!< else integrate the whole domain
    real :: integral
    type(scalar_field_pointer), dimension(:), intent(in) :: fields
    !! The positions field associated with the fields.
    type(vector_field), intent(in) :: X
    integer, dimension(:), intent(in), optional :: region_ids

    integer :: ele
    integer :: s
    integer :: id
    integer :: ele_id
    logical :: found_id

    integral=0

    ! Ideally there needs to be an assertion that the fields are associated
    ! with the same positions mesh X

    ! assert that each scalar field has the same number of elements
    ! and the same dim as positions mesh X
    do s = 1,size(fields)

      assert(ele_count(X) == ele_count(fields(s)%ptr))

      assert(mesh_dim(X) == mesh_dim(fields(s)%ptr))

    end do

    ! if region_ids is present assert that it has something
    if (present(region_ids)) then

      assert(size(region_ids) > 0)

    end if

    velement_loop: do ele=1, element_count(fields(1)%ptr)

      if(element_owned(fields(1)%ptr, ele)) then

        ! if present only conisder input region_ids
        region_id_present: if (present(region_ids)) then

          ! initialise flag for whether this volume element ele should be considered
          found_id = .false.

          ! find the positions X field ele region id
          ele_id = ele_region_id(X,ele)

          region_id_loop: do id = 1,size(region_ids)

            check_id: if (ele_id == region_ids(id)) then

              found_id = .true.

              exit region_id_loop

            end if check_id

          end do region_id_loop

          ! if not found an id match then cycle the volume element loop
          if (.not. found_id) cycle velement_loop

        end if region_id_present

        integral=integral + integral_element(fields, X, ele)

      end if

    end do velement_loop

    call allsum(integral)

  end function integral_scalars

  function mesh_integral(X) result (integral)
    !!< Integrate mesh volume.
    real :: integral
    !! The positions field.
    type(vector_field), intent(in) :: X

    integer :: ele
    real, dimension(X%mesh%shape%ndof) :: ones

    integral=0

    ones = 1.0

    do ele=1, element_count(X)
       integral=integral &
            +element_volume(X, ele)
    end do
  end function mesh_integral

  subroutine field_stats_scalar(field, X, min, max, norm2, integral)
    !!< Return scalar statistical informaion about field.
    type(scalar_field) :: field
    !! Positions field associated with field
    type(vector_field), optional :: X
    !! Minimum value in the field.
    real, intent(out), optional :: min
    !! Maximum value in the field.
    real, intent(out), optional :: max
    !! L2 norm of the field. This requires positions to be specified as
    !! well.
    real, intent(out), optional :: norm2
    !! Integral of the field. This requires positions to be specified as
    !! well.
    real, intent(out), optional :: integral

    if (present(min)) then
       min=minval(field%val)
       call allmin(min)
    end if

    if (present(max)) then
       max=maxval(field%val)
       call allmax(max)
    end if

    if (present(X).and.present(norm2)) then

       norm2=norm2_scalar(field, X)

    elseif (present(norm2)) then
       FLAbort("Cannot evaluate L2 norm without providing positions field")
    end if

    if (present(X).and.present(integral)) then

       integral=integral_scalar(field, X)

    elseif (present(integral)) then
       FLAbort("Cannot evaluate integral without providing positions field")
    end if

  end subroutine field_stats_scalar

  subroutine field_cv_stats_scalar(field, cv_mass, norm2, integral)
    !!< Return scalar statistical informaion about field.
    type(scalar_field), intent(in) :: field
    type(scalar_field), intent(in) :: cv_mass
    !! L2 norm of the field. This requires positions to be specified as
    !! well.
    real, intent(out), optional :: norm2
    !! Integral of the field. This requires positions to be specified as
    !! well.
    real, intent(out), optional :: integral

    if (present(norm2)) then

       norm2=norm2_scalar_cv(field, cv_mass)

    end if

    if (present(integral)) then

       integral=integral_scalar_cv(field, cv_mass)

    end if

  end subroutine field_cv_stats_scalar

  subroutine field_stats_vector(field, X, min, max, norm2)
    !!< Return scalar statistical information about field. For a vector
    !!< field the statistics are calculated on the magnitude of the field.
    type(vector_field) :: field
    !! Positions field assocated with field
    type(vector_field), optional :: X
    !! Minimum value in the field.
    real, intent(out), optional :: min
    !! Maximum value in the field.
    real, intent(out), optional :: max
    !! L2 norm of the field. This requires positions to be specified as
    !! well.
    real, intent(out), optional :: norm2

    type(scalar_field) :: mag

    mag=magnitude(field)

    call field_stats(mag, X, min, max, norm2)

    call deallocate(mag)

  end subroutine field_stats_vector

  subroutine field_stats_tensor(field, X, min, max, norm2)
    !!< Return scalar statistical information about field. For a tensor
    !!< field the statistics are calculated on the magnitude of the field.
    type(tensor_field) :: field
    !! Positions field assocated with field
    type(vector_field), optional :: X
    !! Minimum value in the field.
    real, intent(out), optional :: min
    !! Maximum value in the field.
    real, intent(out), optional :: max
    !! L2 norm of the field. This requires positions to be specified as
    !! well.
    real, intent(out), optional :: norm2

    type(scalar_field) :: mag

    mag=magnitude_tensor(field)

    call field_stats(mag, X, min, max, norm2)

    call deallocate(mag)

  end subroutine field_stats_tensor

  subroutine field_con_stats_scalar(field, nlfield, error, &
                                    norm, coordinates, cv_mass)
    !!< Return scalar convergence informaion about field.
    type(scalar_field), intent(inout) :: field, nlfield
    !! error in the field.
    real, intent(out) :: error
    !! what norm are we working out
    integer, intent(in), optional :: norm
    type(vector_field), intent(in), optional :: coordinates
    type(scalar_field), intent(in), optional :: cv_mass

    type(scalar_field) :: difference
    integer :: l_norm

    if(present(norm)) then
      l_norm = norm
    else
      l_norm = CONVERGENCE_INFINITY_NORM
    end if

    assert(field%mesh==nlfield%mesh)

    call allocate(difference, field%mesh, "Difference")
    call set(difference, field)
    call addto(difference, nlfield, -1.0)
    call absolute_value(difference)

    select case(l_norm)
    case(CONVERGENCE_INFINITY_NORM)
      error = maxval(difference%val)
      call allmax(error)
    case(CONVERGENCE_L2_NORM)
      call field_stats(difference, X=coordinates, norm2=error)
    case(CONVERGENCE_CV_L2_NORM)
      if (present(cv_mass)) then
        call field_cv_stats(difference, cv_mass=cv_mass, norm2=error)
      else
        FLAbort('Require cv_mass to calculate field_cv_stats')
      end if
    case default
      FLAbort("Unknown norm for convergence statistics.")
    end select

    call deallocate(difference)

  end subroutine field_con_stats_scalar

  subroutine field_con_stats_vector(field, nlfield, error, &
                                    norm, coordinates)
    !!< Return scalar convergence information about field. For a vector
    !!< field the statistics are calculated on the magnitude of the field.
    type(vector_field) :: field, nlfield
    !! error in the field.
    real, intent(out) :: error
    integer, intent(in), optional :: norm
    type(vector_field), intent(in), optional :: coordinates

    type(scalar_field) :: mag, nlmag

    mag=magnitude(field)
    nlmag=magnitude(nlfield)

    call field_con_stats(mag, nlmag, error, &
                         norm, coordinates)

    call deallocate(mag)
    call deallocate(nlmag)

  end subroutine field_con_stats_vector

  subroutine divergence_field_stats(field, X, field_min, field_max, field_norm2, field_integral)
    !!< Return scalar statistical informaion about the divergence of field.
    type(vector_field) :: field
    !! Positions field associated with field
    type(vector_field) :: X
    !! Minimum value in the field.
    real, intent(out) :: field_min
    !! Maximum value in the field.
    real, intent(out) :: field_max
    !! L2 norm of the field. This requires positions to be specified as
    !! well.
    real, intent(out) :: field_norm2
    !! Integral of the field. This requires positions to be specified as
    !! well.
    real, intent(out) :: field_integral

    integer :: ele
    real :: ele_min, ele_max, ele_norm2, ele_integral

    field_min = huge(0.0)
    field_max = -huge(0.0)
    field_norm2 = 0.0
    field_integral = 0.0

    do ele = 1, ele_count(field)
      call divergence_field_stats_element(ele, field, X, ele_min, ele_max, ele_norm2, ele_integral)
      field_min = min(field_min, ele_min)
      field_max = max(field_max, ele_max)
      field_norm2 = field_norm2 + ele_norm2
      field_integral = field_integral + ele_integral
    end do

    call allmin(field_min)
    call allmax(field_max)
    call allsum(field_norm2)
    field_norm2 = sqrt(field_norm2)
    call allsum(field_integral)

    contains

    subroutine divergence_field_stats_element(ele, field, X, ele_min, ele_max, ele_norm2, ele_integral)
      integer, intent(in) :: ele
      type(vector_field), intent(in) :: field, X
      real, intent(inout) :: ele_min, ele_max, ele_norm2, ele_integral

      real, dimension(ele_loc(field, ele), ele_ngi(field, ele), mesh_dim(field)) :: df_t
      real, dimension(ele_ngi(field, ele)) :: detwei, field_div_at_quad

      call transform_to_physical(X, ele, &
           & ele_shape(field, ele), dshape = df_t, detwei = detwei)

      field_div_at_quad = ele_div_at_quad(field, ele, df_t)

      ele_min      = minval(field_div_at_quad)
      ele_max      = maxval(field_div_at_quad)
      ele_norm2    = dot_product(field_div_at_quad*field_div_at_quad, detwei)
      ele_integral = dot_product(field_div_at_quad, detwei)

    end subroutine divergence_field_stats_element

  end subroutine divergence_field_stats

  function distance(positions, p, q) result(dist)
    !!< Return the euclidean distance between nodes p and q.
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: p, q
    real :: dist
    integer :: i

    dist = 0.0
    do i=1,positions%dim
      dist = dist + (node_val(positions, i, p) - node_val(positions, i, q))**2
    end do

    dist = sqrt(dist)

  end function distance

  subroutine trace(tensor, output)
    type(tensor_field), intent(in) :: tensor
    type(scalar_field), intent(inout) :: output

    integer :: i, j
    real, dimension(tensor%dim(1), tensor%dim(2)) :: val
    real :: x

    do i=1,node_count(tensor)
      x = 0.0

      val = node_val(tensor, i)
      do j=1,minval(tensor%dim)
        x = x + val(j, j)
      end do

      call set(output, i, x)
    end do
  end subroutine trace

  function dot_product_scalar(fieldA, fieldB) result(val)
    type(scalar_field), intent(in) :: fieldA, fieldB
    real :: val
    integer :: i

    if (.not. associated(fieldA%mesh%refcount, fieldB%mesh%refcount)) then
      ewrite(-1,*) "Hello! dot_product_scalar here."
      ewrite(-1,*) "I couldn't be bothered remapping the fields,"
      ewrite(-1,*) "even though this is perfectly possible."
      ewrite(-1,*) "Code this up to remap the fields to continue!"
      FLAbort("Programmmer laziness detected")
    end if

    if (fieldA%field_type == FIELD_TYPE_NORMAL .and. fieldB%field_type == FIELD_TYPE_NORMAL) then
      val = dot_product(fieldA%val, fieldB%val)
    else if (fieldA%field_type == FIELD_TYPE_CONSTANT .and. fieldB%field_type == FIELD_TYPE_CONSTANT) then
      val = fieldA%val(1) * fieldB%val(1) * node_count(fieldA)
    else
      val = 0.0
      do i=1,node_count(fieldA)
        val = val + node_val(fieldA, i) * node_val(fieldB, i)
      end do
    end if
  end function dot_product_scalar

  function dot_product_vector(fieldA, fieldB) result(val)
    type(vector_field), intent(in) :: fieldA, fieldB
    real, dimension(fieldA%dim) :: val
    integer :: i, d

    assert(fieldA%dim==fieldB%dim)

    if (.not. associated(fieldA%mesh%refcount, fieldB%mesh%refcount)) then
      ewrite(-1,*) "Hello! dot_product_vector here."
      ewrite(-1,*) "I couldn't be bothered remapping the fields,"
      ewrite(-1,*) "even though this is perfectly possible."
      ewrite(-1,*) "Code this up to remap the fields to continue!"
      FLAbort("Programmmer laziness detected")
    end if

    if (fieldA%field_type == FIELD_TYPE_NORMAL .and. fieldB%field_type == FIELD_TYPE_NORMAL) then
       do d=1,fieldA%dim
          val(d) = dot_product(fieldA%val(d,:), fieldB%val(d,:))
       end do
    else if (fieldA%field_type == FIELD_TYPE_CONSTANT .and. fieldB%field_type == FIELD_TYPE_CONSTANT) then
       do d=1,fieldA%dim
          val(d) = fieldA%val(d,1) * fieldB%val(d,1) * node_count(fieldA)
       end do
    else
       val = 0.0
       do i=1,node_count(fieldA)
          val = val + node_val(fieldA, i) * node_val(fieldB, i)
       end do
    end if
  end function dot_product_vector

  function outer_product_vector(fieldA, fieldB) result(val)
    type(vector_field), intent(in) :: fieldA, fieldB
    real, dimension(fieldA%dim,fieldB%dim) :: val
    integer :: i, d1,d2
    real, dimension(fieldA%dim) :: tmpA
    real, dimension(fieldB%dim) :: tmpB

    if (.not. associated(fieldA%mesh%refcount, fieldB%mesh%refcount)) then
      ewrite(-1,*) "Hello! outer_product_vector here."
      ewrite(-1,*) "I couldn't be bothered remapping the fields,"
      ewrite(-1,*) "even though this is perfectly possible."
      ewrite(-1,*) "Code this up to remap the fields to continue!"
      FLAbort("Programmmer laziness detected")
    end if

    if (fieldA%field_type == FIELD_TYPE_NORMAL .and. fieldB%field_type == FIELD_TYPE_NORMAL) then
       do d1=1,fieldA%dim
          do d2=1,fieldB%dim
             val(d1,d2) = dot_product(fieldA%val(d1,:), fieldB%val(d2,:))
          end do
       end do
    else if (fieldA%field_type == FIELD_TYPE_CONSTANT .and. fieldB%field_type == FIELD_TYPE_CONSTANT) then
       do d1=1,fieldA%dim
          do d2=1,fieldB%dim
             val(d1,d2) = fieldA%val(d1,1) * fieldB%val(d2,1) * node_count(fieldA)
          end do
       end do
    else
       val = 0.0
       do i=1,node_count(fieldA)
          tmpA=node_val(fieldA, i)
          tmpB=node_val(fieldB, i)
          do d1=1,fieldA%dim
             do d2=1,fieldB%dim
                val(d1,d2) = val(d1,d2) + tmpA(d1) * tmpB(d2)
             end do
          end do
       end do
    end if
  end function outer_product_vector

  function function_val_at_quad_scalar(fxn, positions, ele)
    interface
      function fxn(pos)
        real, dimension(:), intent(in) :: pos
        real :: fxn
      end function
    end interface
    type(vector_field), intent(in) :: positions
    integer :: ele, ngi

    real, dimension(positions%dim, ele_ngi(positions, ele)) :: pos
    real, dimension(ele_ngi(positions, ele)) :: function_val_at_quad_scalar

    pos = ele_val_at_quad(positions, ele)
    do ngi=1,ele_ngi(positions, ele)
      function_val_at_quad_scalar(ngi) = fxn(pos(:, ngi))
    end do
  end function function_val_at_quad_scalar

  function function_val_at_quad_vector(fxn, positions, ele)
    interface
      function fxn(pos)
        real, dimension(:), intent(in) :: pos
        real, dimension(size(pos)) :: fxn
      end function
    end interface
    type(vector_field), intent(in) :: positions
    integer :: ele, ngi

    real, dimension(positions%dim, ele_ngi(positions, ele)) :: pos
    real, dimension(positions%dim, ele_ngi(positions, ele)) :: function_val_at_quad_vector

    pos = ele_val_at_quad(positions, ele)
    do ngi=1,ele_ngi(positions, ele)
      function_val_at_quad_vector(:, ngi) = fxn(pos(:, ngi))
    end do
  end function function_val_at_quad_vector

  function norm2_difference_single(fieldA, positionsA, fieldB, positionsB) result(norm)
    !! Return ||fieldA - fieldB||_2.
    !! Since positionsA and positionsB are different, we need to supermesh!
    !! If positionsA and positionsB are the same, don't use this:
    !! it will be much slower than necessary.
    type(scalar_field), intent(in) :: fieldA, fieldB
    type(vector_field), intent(in) :: positionsA, positionsB
    real :: norm
    type(ilist), dimension(ele_count(positionsB)) :: map_BA
    integer :: ele_A, ele_B

    type(quadrature_type) :: supermesh_quad
    type(element_type) :: supermesh_positions_shape, supermesh_fields_shape

    type(vector_field) :: supermesh
    real :: ele_error

    real, dimension(ele_loc(positionsB, 1), ele_loc(positionsB, 1)) :: inversion_matrix_B
    real, dimension(ele_loc(positionsB, 1), ele_loc(positionsB, 1), ele_count(positionsA)) :: inversion_matrices_A
    integer :: dim, max_degree

    norm = 0.0
    dim = mesh_dim(positionsB)
    call intersector_set_dimension(dim)

    max_degree = max(element_degree(fieldA, 1), element_degree(fieldB, 1))
    supermesh_quad = make_quadrature(vertices=ele_loc(positionsB, 1), dim=dim, &
                                   & degree=2*max_degree)
    supermesh_positions_shape = make_element_shape(vertices=ele_loc(positionsB, 1), dim=dim, degree=1, quad=supermesh_quad)
    supermesh_fields_shape = make_element_shape(vertices=ele_loc(positionsB, 1), dim=dim, degree=max_degree, quad=supermesh_quad)
    map_BA = intersection_finder(positionsB, positionsA)

    do ele_A=1,ele_count(positionsA)
      call local_coords_matrix(positionsA, ele_A, inversion_matrices_A(:, :, ele_A))
    end do

    do ele_B=1,ele_count(positionsB)
      call local_coords_matrix(positionsB, ele_B, inversion_matrix_B)
      ! Construct the supermesh associated with ele_B.
      call construct_supermesh(positionsB, ele_B, positionsA, map_BA(ele_B), supermesh_positions_shape, supermesh)

      ! Interpolate fieldA onto the supermesh.
      ! Interpolate fieldB onto the supermesh.
      ! Compute the l2norm**2 of the difference.
      call compute_projection_error(fieldA, positionsA, supermesh_fields_shape, ele_val(fieldB, ele_B), positionsB, ele_B, supermesh, &
                                    inversion_matrices_A, inversion_matrix_B, ele_error)

      norm = norm + ele_error
      call deallocate(supermesh)
    end do

    norm = sqrt(norm)

    call deallocate(supermesh_quad)
    call deallocate(supermesh_positions_shape)
    call deallocate(supermesh_fields_shape)
    do ele_B=1,ele_count(positionsB)
      call deallocate(map_BA(ele_B))
    end do

  end function norm2_difference_single

  function norm2_difference_multiple(fieldA, positionsA, fieldB, positionsB) result(norm)
    !! Return ||fieldA - fieldB||_2.
    !! Since positionsA and positionsB are different, we need to supermesh!
    !! If positionsA and positionsB are the same, don't use this:
    !! it will be much slower than necessary.
    type(scalar_field), dimension(:), intent(in) :: fieldA, fieldB
    type(vector_field), intent(in) :: positionsA, positionsB
    real, dimension(size(fieldA)) :: norm
    type(ilist), dimension(ele_count(positionsB)) :: map_BA
    integer :: ele_A, ele_B

    type(quadrature_type) :: supermesh_quad
    type(element_type) :: supermesh_positions_shape, supermesh_fields_shape

    type(vector_field) :: supermesh
    real :: ele_error

    real, dimension(ele_loc(positionsB, 1), ele_loc(positionsB, 1)) :: inversion_matrix_B
    real, dimension(ele_loc(positionsB, 1), ele_loc(positionsB, 1), ele_count(positionsA)) :: inversion_matrices_A
    integer :: dim, max_degree, field, field_count

    field_count = size(fieldA)
    assert(size(fieldB) == field_count)
    norm = 0.0
    dim = mesh_dim(positionsB)
    call intersector_set_dimension(dim)

    max_degree = 0
    do field=1,field_count
      max_degree = max(max_degree, max(element_degree(fieldA(field), 1), element_degree(fieldB(field), 1)))
    end do
    supermesh_quad = make_quadrature(vertices=ele_loc(positionsB, 1), dim=dim, &
                                   & degree=2*max_degree)
    supermesh_positions_shape = make_element_shape(vertices=ele_loc(positionsB, 1), dim=dim, degree=1, quad=supermesh_quad)
    supermesh_fields_shape = make_element_shape(vertices=ele_loc(positionsB, 1), dim=dim, degree=max_degree, quad=supermesh_quad)
    map_BA = intersection_finder(positionsB, positionsA)

    do ele_A=1,ele_count(positionsA)
      call local_coords_matrix(positionsA, ele_A, inversion_matrices_A(:, :, ele_A))
    end do

    do ele_B=1,ele_count(positionsB)
      call local_coords_matrix(positionsB, ele_B, inversion_matrix_B)
      ! Construct the supermesh associated with ele_B.
      call construct_supermesh(positionsB, ele_B, positionsA, map_BA(ele_B), supermesh_positions_shape, supermesh)

      ! Interpolate fieldA onto the supermesh.
      ! Interpolate fieldB onto the supermesh.
      ! Compute the l2norm**2 of the difference.
      do field=1,field_count
        call compute_projection_error(fieldA(field), positionsA, supermesh_fields_shape, ele_val(fieldB(field), ele_B), positionsB, ele_B, supermesh, &
                                      inversion_matrices_A, inversion_matrix_B, ele_error)
        norm(field) = norm(field) + ele_error
      end do

      call deallocate(supermesh)
    end do

    norm = sqrt(norm)

    call deallocate(supermesh_quad)
    call deallocate(supermesh_positions_shape)
    call deallocate(supermesh_fields_shape)
    do ele_B=1,ele_count(positionsB)
      call deallocate(map_BA(ele_B))
    end do

  end function norm2_difference_multiple

  function merge_meshes(meshes, name)
    !! merges a set of disjoint meshes, elements and nodes
    !! are consecutively numbered following the order of the input meshes
    type(mesh_type), dimension(:), intent(in):: meshes
    character(len=*), intent(in), optional:: name
    type(mesh_type):: merge_meshes

    integer:: nodes, elements, ndglno_count, i

    elements=0
    do i=1, size(meshes)
      elements=elements+element_count(meshes(i))
    end do

    nodes=0
    do i=1, size(meshes)
      nodes=nodes+node_count(meshes(i))
    end do

    call allocate(merge_meshes, nodes, elements, meshes(1)%shape, &
      name=name)

    nodes=0
    ndglno_count=0
    do i=1, size(meshes)
      merge_meshes%ndglno(ndglno_count+1:ndglno_count+size(meshes(i)%ndglno)) = &
        meshes(i)%ndglno+nodes
      ndglno_count=ndglno_count+size(meshes(i)%ndglno)
      nodes=nodes+node_count(meshes(i))
    end do

  end function merge_meshes

end module fields_calculations
