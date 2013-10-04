#include "fdebug.h"

module interpolation_module
  use fields
  use field_derivatives
  use state_module
  use sparse_tools
  use supermesh_construction
  use futils
  use transform_elements
  use meshdiagnostics
  use field_options
  use node_ownership
  implicit none

  interface linear_interpolation
    module procedure linear_interpolation_state, linear_interpolation_scalar, &
      & linear_interpolation_scalars, linear_interpolation_vector, linear_interpolation_vectors, &
      & linear_interpolation_tensors, linear_interpolation_tensor
  end interface

  contains

  function get_element_mapping(old_position, new_position, different_domains) result(map)
    !!< Return the elements in the mesh of old_position
    !!< that contains the nodes of the mesh of new_position
    type(vector_field), intent(inout) :: old_position
    type(vector_field), intent(in) :: new_position
    logical, optional, intent(in) :: different_domains

    integer, dimension(node_count(new_position)) :: map

    ! Thanks, James!
    call find_node_ownership(old_position, new_position, map)

    if(.not. present_and_true(different_domains)) then
      if(.not. all(map > 0)) then
        FLAbort("Failed to find element mapping")
      end if
    end if

  end function get_element_mapping

  subroutine linear_interpolation_scalar(old_field, old_position, new_field, new_position, map)
    type(scalar_field), intent(in) :: old_field
    type(vector_field), intent(in) :: old_position
    type(scalar_field), intent(inout) :: new_field
    type(vector_field), intent(in) :: new_position
    integer, dimension(:), optional, intent(in) :: map

    type(state_type) :: old_state, new_state

    call insert(old_state, old_field, old_field%name)
    call insert(new_state, new_field, old_field%name)

    call insert(old_state, old_position, "Coordinate")
    call insert(new_state, new_position, "Coordinate")

    call insert(old_state, old_field%mesh, "Mesh")
    call insert(new_state, new_field%mesh, "Mesh")

    call linear_interpolation_state(old_state, new_state, map = map)
    call deallocate(old_state)
    call deallocate(new_state)

  end subroutine linear_interpolation_scalar

  subroutine linear_interpolation_scalars(old_fields, old_position, new_fields, new_position, map)
    type(scalar_field), dimension(:), intent(in) :: old_fields
    type(vector_field), intent(in) :: old_position
    type(scalar_field), dimension(:), intent(inout) :: new_fields
    type(vector_field), intent(in) :: new_position
    integer, dimension(:), optional, intent(in) :: map

    type(state_type) :: old_state, new_state
    integer :: field, field_count

    field_count = size(old_fields)

    do field=1,field_count
      call insert(old_state, old_fields(field), old_fields(field)%name)
      call insert(new_state, new_fields(field), old_fields(field)%name)
    end do

    call insert(old_state, old_position, "Coordinate")
    call insert(new_state, new_position, "Coordinate")

    call insert(old_state, old_fields(1)%mesh, "Mesh")
    call insert(new_state, new_fields(1)%mesh, "Mesh")

    call linear_interpolation_state(old_state, new_state, map = map)
    call deallocate(old_state)
    call deallocate(new_state)

  end subroutine linear_interpolation_scalars

  subroutine linear_interpolation_vector(old_field, old_position, new_field, new_position, map)
    type(vector_field), intent(in) :: old_field
    type(vector_field), intent(in) :: old_position
    type(vector_field), intent(inout) :: new_field
    type(vector_field), intent(in) :: new_position
    integer, dimension(:), optional, intent(in) :: map

    type(state_type) :: old_state, new_state

    call insert(old_state, old_field, old_field%name)
    call insert(new_state, new_field, new_field%name)

    call insert(old_state, old_position, "Coordinate")
    call insert(new_state, new_position, "Coordinate")

    call insert(old_state, old_field%mesh, "Mesh")
    call insert(new_state, new_field%mesh, "Mesh")

    call linear_interpolation_state(old_state, new_state, map = map)
    call deallocate(old_state)
    call deallocate(new_state)

  end subroutine linear_interpolation_vector

  subroutine linear_interpolation_vectors(old_fields, old_position, new_fields, new_position)
    type(vector_field), dimension(:), intent(in) :: old_fields
    type(vector_field), intent(in) :: old_position
    type(vector_field), dimension(:), intent(inout) :: new_fields
    type(vector_field), intent(in) :: new_position

    type(state_type) :: old_state, new_state
    integer :: field, field_count

    field_count = size(old_fields)

    do field=1,field_count
      call insert(old_state, old_fields(field), old_fields(field)%name)
      call insert(new_state, new_fields(field), old_fields(field)%name)
    end do

    call insert(old_state, old_position, "Coordinate")
    call insert(new_state, new_position, "Coordinate")

    call insert(old_state, old_fields(1)%mesh, "Mesh")
    call insert(new_state, new_fields(1)%mesh, "Mesh")

    call linear_interpolation_state(old_state, new_state)
    call deallocate(old_state)
    call deallocate(new_state)

  end subroutine linear_interpolation_vectors

  subroutine linear_interpolation_tensors(old_fields, old_position, new_fields, new_position)
    type(tensor_field), dimension(:), intent(in) :: old_fields
    type(vector_field), intent(in) :: old_position
    type(tensor_field), dimension(:), intent(inout) :: new_fields
    type(vector_field), intent(in) :: new_position

    type(state_type) :: old_state, new_state
    integer :: field, field_count

    field_count = size(old_fields)

    do field=1,field_count
      call insert(old_state, old_fields(field), old_fields(field)%name)
      call insert(new_state, new_fields(field), old_fields(field)%name)
    end do

    call insert(old_state, old_position, "Coordinate")
    call insert(new_state, new_position, "Coordinate")

    call insert(old_state, old_fields(1)%mesh, "Mesh")
    call insert(new_state, new_fields(1)%mesh, "Mesh")

    call linear_interpolation_state(old_state, new_state)
    call deallocate(old_state)
    call deallocate(new_state)

  end subroutine linear_interpolation_tensors

  subroutine linear_interpolation_tensor(old_field, old_position, new_field, new_position)
    type(tensor_field), intent(in) :: old_field
    type(vector_field), intent(in) :: old_position
    type(tensor_field), intent(inout) :: new_field
    type(vector_field), intent(in) :: new_position

    type(tensor_field), dimension(1) :: new_field_array, old_field_array

    new_field_array(1) = new_field
    old_field_array(1) = old_field

    call linear_interpolation(old_field_array, old_position, new_field_array, new_position)

  end subroutine linear_interpolation_tensor

  subroutine linear_interpolation_state(old_state, new_state, map, different_domains)
    !!< Interpolate the fields defined on the old_fields mesh
    !!< onto the new_fields mesh.
    !!< This routine assumes new_state has all been allocated,
    !!< ON THE SAME MESH; it also assumes old_state has all been
    !!< allocated on the same mesh. Call it multiple times for
    !!< multiple meshes.

    type(state_type), intent(in), target :: old_state, new_state
    integer, dimension(:), intent(in), optional, target :: map
    integer, dimension(:), pointer :: lmap
    logical, intent(in), optional :: different_domains

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
    real, dimension(:), allocatable :: val_v
    real, dimension(:,:), allocatable :: val_t
    real, dimension(:), allocatable :: local_coord, shape_fns

    type(scalar_field), dimension(:), allocatable, target :: old_fields_s
    type(scalar_field), dimension(:), allocatable, target :: new_fields_s

    type(vector_field), dimension(:), allocatable, target :: old_fields_v
    type(vector_field), dimension(:), allocatable, target :: new_fields_v

    type(tensor_field), dimension(:), allocatable, target :: old_fields_t
    type(tensor_field), dimension(:), allocatable, target :: new_fields_t

    if (associated(old_state%scalar_fields)) then
      allocate(old_fields_s(size(old_state%scalar_fields)))
      allocate(new_fields_s(size(new_state%scalar_fields)))
    end if
    if (associated(old_state%vector_fields)) then
      allocate(old_fields_v(size(old_state%vector_fields)))
      allocate(new_fields_v(size(new_state%vector_fields)))
    end if
    if (associated(old_state%tensor_fields)) then
      allocate(old_fields_t(size(old_state%tensor_fields)))
      allocate(new_fields_t(size(new_state%tensor_fields)))
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Scalar fields
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (associated(old_state%scalar_fields)) then
      field_count_s = size(old_state%scalar_fields)

      ! Construct the list of new_fields to be modified

      do field_s=1,field_count_s
        old_fields_s(field_s) = extract_scalar_field(old_state, trim(old_state%scalar_names(field_s)))
        new_fields_s(field_s) = extract_scalar_field(new_state, trim(old_state%scalar_names(field_s)))
      end do

      ! Zero the new fields.

      if (.not. present_and_true(different_domains)) then
        do field_s=1,field_count_s
          call zero(new_fields_s(field_s))
        end do
      end if
    else
      field_count_s = 0
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Vector fields
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    j=1
    do i=1, vector_field_count(old_state)
      old_fields_v(j) = extract_vector_field(old_state, i)
      ! skip coordinate fields
      if (.not. (old_fields_v(j)%name=="Coordinate" .or. &
         old_fields_v(j)%name==trim(old_fields_v(j)%mesh%name)//"Coordinate")) then

         new_fields_v(j) = extract_vector_field(new_state, old_state%vector_names(i))
         if (.not. present_and_true(different_domains)) then
           call zero(new_fields_v(j))
         end if
         j=j+1

      end if
    end do
    field_count_v=j-1

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Tensor fields
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (associated(old_state%tensor_fields)) then
      field_count_t = size(old_state%tensor_fields)

      ! Construct the list of new_fields to be modified

      do field_t=1,field_count_t
        old_fields_t(field_t) = extract_tensor_field(old_state, trim(old_state%tensor_names(field_t)))
        new_fields_t(field_t) = extract_tensor_field(new_state, trim(old_state%tensor_names(field_t)))
      end do

      ! Zero the new fields.

      if (.not. present_and_true(different_domains)) then
        do field_t=1,field_count_t
          call zero(new_fields_t(field_t))
        end do
      end if
    else
      field_count_t = 0
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

    old_position => extract_vector_field(old_state, "Coordinate")
    new_position=get_coordinate_field(new_state, new_mesh)

    allocate(local_coord(old_position%dim + 1))
    allocate(shape_fns(ele_loc(old_mesh, 1)))

    if(field_count_v>0) then
      allocate(val_v(old_state%vector_fields(1)%ptr%dim))
    end if
    if(field_count_t>0) then
      allocate(val_t(old_state%tensor_fields(1)%ptr%dim(1), old_state%tensor_fields(1)%ptr%dim(2)))
    end if

    if (present(map)) then
      assert(node_count(new_mesh) == size(map))
      lmap => map
    else
      allocate(lmap(node_count(new_mesh)))
      lmap = get_element_mapping(old_position, new_position, different_domains = different_domains)
    end if

    ! Loop over the nodes of the new mesh.

    do new_node=1,node_count(new_mesh)

      ! In what element of the old mesh does the new node lie?
      ! Find the local coordinates of the point in that element,
      ! and evaluate all the shape functions at that point
      ele = lmap(new_node)

      if (ele < 0) then
        assert(present_and_true(different_domains))
        cycle
      end if

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
        call set(new_fields_s(field_s), new_node, val_s)
      end do

      do field_v=1,field_count_v
        ! At each node of the old element, evaluate val * shape_fn
        val_v = 0.0
        do i=1,ele_loc(old_mesh, ele)
          val_v = val_v + node_val(old_fields_v(field_v), node_list(i)) * shape_fns(i)
        end do
        call set(new_fields_v(field_v), new_node, val_v)
      end do

      do field_t=1,field_count_t
        ! At each node of the old element, evaluate val * shape_fn
        val_t = 0.0
        do i=1,ele_loc(old_mesh, ele)
          val_t = val_t + node_val(old_fields_t(field_t), node_list(i)) * shape_fns(i)
        end do
        call set(new_fields_t(field_t), new_node, val_t)
      end do
    end do

    if (.not. present(map)) then
      deallocate(lmap)
    end if
    deallocate(local_coord)
    deallocate(shape_fns)

    call deallocate(new_position)

  end subroutine linear_interpolation_state

  subroutine linear_interpolate_states(from_states, target_states, map)
  type(state_type), dimension(:), intent(inout):: from_states
  type(state_type), dimension(:), intent(inout):: target_states
  integer, dimension(:), intent(in), optional :: map

    type(state_type) meshj_state
    type(vector_field), pointer:: from_positions
    type(mesh_type), pointer:: from_meshj, target_meshj
    integer i, j

    do i=1, size(target_states)
      do j=1, mesh_count(target_states(i))
        target_meshj => extract_mesh(target_states(i), j)
        from_meshj => extract_mesh(from_states(i), j)
        ! select fields that are on meshj only
        call select_state_by_mesh(from_states(i), trim(from_meshj%name), meshj_state)

        ! insert coordinate field in selection
        ! (possibly on other mesh, will be remapped in linear_interpolation_state)
        from_positions => extract_vector_field(from_states(i), "Coordinate")
        call insert(meshj_state, from_positions, name=from_positions%name)

        call linear_interpolation_state(meshj_state, target_states(i), &
          map)
        call deallocate(meshj_state)
      end do
    end do

  end subroutine linear_interpolate_states

end module interpolation_module
