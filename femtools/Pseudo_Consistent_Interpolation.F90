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

module pseudo_consistent_interpolation

  use data_structures
  use field_options
  use fields
  use fldebug
  use node_ownership
  use state_module

  implicit none

  private

  public :: pseudo_consistent_interpolate

  interface pseudo_consistent_interpolate
    module procedure pseudo_consistent_interpolate_state, &
      & pseudo_consistent_interpolate_states
  end interface pseudo_consistent_interpolate

  !! The tolerance used for boundary node detection
  real, parameter :: ownership_tolerance = 1.0e3 * epsilon(0.0)

contains

  subroutine pseudo_consistent_interpolate_state(old_state, new_state)
    type(state_type), intent(in) :: old_state
    type(state_type), intent(inout) :: new_state

    type(integer_set), dimension(:), allocatable :: map

    type(vector_field), pointer :: old_position
    type(vector_field) :: new_position

    type(mesh_type), pointer :: old_mesh, new_mesh
    integer :: new_node
    integer :: ele
    integer :: field_count_s
    integer :: field_s
    integer :: field_count_v
    integer :: field_v
    integer :: field_count_t
    integer :: field_t
    integer, dimension(:), pointer :: node_list
    integer :: i, j
    real :: val_s
    real, dimension(mesh_dim(old_state%meshes(1)%ptr)) :: val_v
    real, dimension(mesh_dim(old_state%meshes(1)%ptr), mesh_dim(old_state%meshes(1)%ptr)) :: val_t
    real, dimension(:), allocatable :: local_coord, shape_fns

    type(scalar_field), dimension(:), allocatable, target :: old_fields_s
    type(scalar_field), dimension(:), allocatable, target :: new_fields_s

    type(vector_field), dimension(:), allocatable, target :: old_fields_v
    type(vector_field), dimension(:), allocatable, target :: new_fields_v

    type(tensor_field), dimension(:), allocatable, target :: old_fields_t
    type(tensor_field), dimension(:), allocatable, target :: new_fields_t

    integer :: elei, neles

    ewrite(1, *) "In pseudo_consistent_interpolate_state"

    field_count_s = scalar_field_count(old_state)
    field_count_v = vector_field_count(old_state)
    field_count_t = tensor_field_count(old_state)

    if(field_count_s > 0) then
      allocate(old_fields_s(field_count_s))
      allocate(new_fields_s(field_count_s))
    end if
    if(field_count_v > 0) then
      allocate(old_fields_v(field_count_v))
      allocate(new_fields_v(field_count_v))
    end if
    if(field_count_t > 0) then
      allocate(old_fields_t(field_count_t))
      allocate(new_fields_t(field_count_t))
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Scalar fields
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if(field_count_s > 0) then
      ! Construct the list of new_fields to be modified

      do field_s=1,field_count_s
        old_fields_s(field_s) = extract_scalar_field(old_state, field_s)
        new_fields_s(field_s) = extract_scalar_field(new_state, field_s)
        ! Zero the new fields.
        call zero(new_fields_s(field_s))
      end do
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Vector fields
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    j=1
    do i=1, field_count_v
      old_fields_v(j) = extract_vector_field(old_state, i)
      ! skip coordinate fields
      if (.not. (old_fields_v(j)%name=="Coordinate" .or. &
         old_fields_v(j)%name==trim(old_fields_v(j)%mesh%name)//"Coordinate")) then

         new_fields_v(j) = extract_vector_field(new_state, i)
         ! Zero the new fields.
         call zero(new_fields_v(j))
         j=j+1
      end if
    end do
    field_count_v = j - 1

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Tensor fields
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if(field_count_t > 0) then
      ! Construct the list of new_fields to be modified

      do field_t=1,field_count_t
        old_fields_t(field_t) = extract_tensor_field(old_state, field_t)
        new_fields_t(field_t) = extract_tensor_field(new_state, field_t)
        ! Zero the new fields.
        call zero(new_fields_t(field_t))
      end do
    end if

    if(field_count_s > 0) then
      old_mesh => old_fields_s(1)%mesh
      new_mesh => new_fields_s(1)%mesh
    else if(field_count_v > 0) then
      old_mesh => old_fields_v(1)%mesh
      new_mesh => new_fields_v(1)%mesh
    else if(field_count_t > 0) then
      old_mesh => old_fields_t(1)%mesh
      new_mesh => new_fields_t(1)%mesh
    else
      return
    end if

    if(continuity(new_mesh) == 0) then
      ewrite(0, *) "For mesh " // trim(new_mesh%name)
      ewrite(0, *) "Warning: Pseudo consistent interpolation applied for fields on a continuous mesh"
    end if

    old_position => extract_vector_field(old_state, "Coordinate")
    new_position=get_coordinate_field(new_state, new_mesh)

    allocate(local_coord(mesh_dim(new_position) + 1))
    allocate(shape_fns(ele_loc(old_mesh, 1)))

    allocate(map(node_count(new_mesh)))
    call find_node_ownership(old_position, new_position, map, ownership_tolerance = ownership_tolerance)

    ! Loop over the nodes of the new mesh.

    do new_node=1,node_count(new_mesh)

      neles = key_count(map(new_node))
      assert(neles > 0)
      do elei = 1, neles
        ! In what element of the old mesh does the new node lie?
        ! Find the local coordinates of the point in that element,
        ! and evaluate all the shape functions at that point
        ele = fetch(map(new_node), elei)

        node_list => ele_nodes(old_mesh, ele)
        local_coord = local_coords(old_position, ele, node_val(new_position, new_node))
        do i=1,ele_loc(old_mesh, ele)
          shape_fns(i) = eval_shape(ele_shape(old_mesh, ele), i, local_coord)
        end do

        do field_s=1,field_count_s
          ! At each node of the old element, evaluate val * shape_fn
          val_s = 0.0
          do i=1,ele_loc(old_mesh, ele)
            val_s = val_s + node_val(old_fields_s(field_s), node_list(i)) * shape_fns(i)
          end do
          call addto(new_fields_s(field_s), new_node, val_s / float(neles))
        end do

        do field_v=1,field_count_v
          ! At each node of the old element, evaluate val * shape_fn
          val_v = 0.0
          do i=1,ele_loc(old_mesh, ele)
            val_v = val_v + node_val(old_fields_v(field_v), node_list(i)) * shape_fns(i)
          end do
          call addto(new_fields_v(field_v), new_node, val_v / float(neles))
        end do

        do field_t=1,field_count_t
          ! At each node of the old element, evaluate val * shape_fn
          val_t = 0.0
          do i=1,ele_loc(old_mesh, ele)
            val_t = val_t + node_val(old_fields_t(field_t), node_list(i)) * shape_fns(i)
          end do
          call addto(new_fields_t(field_t), new_node, val_t / float(neles))
        end do
      end do
    end do

    call deallocate(map)
    deallocate(map)
    deallocate(local_coord)
    deallocate(shape_fns)

    call deallocate(new_position)

    do field_s = 1, field_count_s
      call halo_update(new_fields_s(field_s))
    end do
    do field_v = 1, field_count_v
      call halo_update(new_fields_v(field_v))
    end do
    do field_t = 1, field_count_t
      call halo_update(new_fields_t(field_t))
    end do

    ewrite(1, *) "Exiting pseudo_consistent_interpolate_state"

  end subroutine pseudo_consistent_interpolate_state

  subroutine pseudo_consistent_interpolate_states(old_states, new_states)
    type(state_type), dimension(:), intent(in) :: old_states
    type(state_type), dimension(size(old_states)), intent(inout) :: new_states

    integer :: i, j
    type(mesh_type), pointer:: old_mesh, new_mesh
    type(state_type) :: old_mesh_state
    type(vector_field), pointer :: old_positions

    do i = 1, size(new_states)
      do j = 1, mesh_count(new_states(i))
        new_mesh => extract_mesh(new_states(i), j)
        old_mesh => extract_mesh(old_states(i), new_mesh%name)
        call select_state_by_mesh(old_states(i), new_mesh%name, old_mesh_state)

        old_positions => extract_vector_field(old_states(i), "Coordiante")
        call insert(old_mesh_state, old_positions, old_positions%name)

        call pseudo_consistent_interpolate(old_mesh_state, new_states(i))

        call deallocate(old_mesh_state)
      end do
    end do

  end subroutine pseudo_consistent_interpolate_states

end module pseudo_consistent_interpolation
