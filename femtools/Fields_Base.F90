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

module fields_base
  !!< This module contains abstracted field types which carry shape and
  !!< connectivity with them.
  use shape_functions, only: element_type
  use tensors
  use fields_data_types
  use reference_counting
  use global_parameters, only: FIELD_NAME_LEN, current_debug_level, current_time
  use elements
  use element_numbering
  use embed_python
  use sparse_tools
  use vector_tools, only: solve, invert, norm2, cross_product
  implicit none

  interface ele_nodes
     module procedure ele_nodes_scalar, ele_nodes_vector, ele_nodes_tensor,&
          & ele_nodes_mesh
  end interface

  interface face_local_nodes
     module procedure face_local_nodes_mesh,face_local_nodes_scalar,&
          & face_local_nodes_vector, face_local_nodes_tensor
  end interface

  interface face_global_nodes
     module procedure face_global_nodes_mesh, face_global_nodes_vector,&
          & face_global_nodes_scalar, face_global_nodes_tensor
  end interface

  interface ele_neigh
     module procedure ele_neigh_mesh, ele_neigh_scalar, ele_neigh_vector, &
          & ele_neigh_tensor, ele_neigh_i_mesh, ele_neigh_i_scalar, &
          & ele_neigh_i_vector, ele_neigh_i_tensor
  end interface

  interface ele_faces
     module procedure ele_faces_mesh, ele_faces_vector, ele_faces_scalar, &
          & ele_faces_tensor
  end interface

  interface node_neigh
     module procedure node_neigh_mesh, node_neigh_vector, node_neigh_scalar, &
          & node_neigh_tensor
  end interface

  interface face_neigh
     module procedure face_neigh_mesh, face_neigh_scalar, face_neigh_vector, &
          & face_neigh_tensor
  end interface

  interface ele_face
     module procedure ele_face_mesh, ele_face_scalar, ele_face_vector,&
          & ele_face_tensor
  end interface

  interface face_ele
     module procedure face_ele_mesh, face_ele_scalar, face_ele_vector,&
          & face_ele_tensor
  end interface

  interface local_face_number
     module procedure local_face_number_mesh, local_face_number_scalar, &
          & local_face_number_vector, local_face_number_tensor
  end interface local_face_number

  interface ele_face_count
     module procedure ele_face_count_mesh, ele_face_count_scalar,&
          & ele_face_count_vector, ele_face_count_tensor
  end interface

  interface cell_family
    module procedure cell_family_mesh, &
      & cell_family_scalar, cell_family_vector, &
      & cell_family_tensor
  end interface cell_family

  interface finite_element_type
    module procedure element_type_shape, element_type_mesh, &
      & element_type_scalar, element_type_vector, &
      & element_type_tensor
  end interface finite_element_type

  interface ele_loc
     module procedure ele_loc_scalar, ele_loc_vector, ele_loc_tensor,&
          & ele_loc_mesh
  end interface

  interface ele_vertices
     module procedure ele_vertices_scalar, ele_vertices_vector, ele_vertices_tensor,&
          & ele_vertices_mesh
  end interface

  interface face_vertices
     module procedure face_vertices_scalar, face_vertices_vector, face_vertices_tensor,&
          & face_vertices_mesh, face_vertices_shape
  end interface

  interface ele_ngi
     module procedure ele_ngi_scalar, ele_ngi_vector, ele_ngi_tensor,&
          & ele_ngi_mesh
  end interface

  interface face_loc
     module procedure face_loc_scalar, face_loc_vector, face_loc_tensor,&
          & face_loc_mesh
  end interface

  interface ele_and_faces_loc
     module procedure ele_and_faces_loc_scalar, ele_and_faces_loc_vector,&
          & ele_and_faces_loc_tensor, ele_and_faces_loc_mesh
  end interface

  interface face_ngi
     module procedure face_ngi_scalar, face_ngi_vector, face_ngi_tensor,&
          & face_ngi_mesh
  end interface

  interface ele_shape
     module procedure ele_shape_scalar, ele_shape_vector, ele_shape_tensor,&
          & ele_shape_mesh
  end interface

  interface ele_n_constraints
     module procedure ele_n_constraints_vector
  end interface

  interface face_shape
     module procedure face_shape_scalar, face_shape_vector,&
          & face_shape_tensor, face_shape_mesh
  end interface

  interface ele_val
     module procedure ele_val_scalar, ele_val_vector, ele_val_vector_dim,&
          & ele_val_tensor, ele_val_tensor_dim_dim
  end interface

  interface face_val
     module procedure face_val_scalar, face_val_vector, face_val_tensor,&
          & face_val_vector_dim, face_val_tensor_dim_dim
  end interface

  interface ele_val_at_quad
     module procedure ele_val_at_quad_scalar, ele_val_at_quad_vector,&
          & ele_val_at_quad_tensor, ele_val_at_shape_quad_scalar, &
          & ele_val_at_shape_quad_vector, ele_val_at_shape_quad_tensor, &
          & ele_val_at_quad_vector_dim, ele_val_at_quad_tensor_dim_dim
  end interface

  interface face_val_at_quad
     module procedure face_val_at_quad_scalar, face_val_at_quad_vector, &
          & face_val_at_quad_tensor, face_val_at_quad_vector_dim,&
          & face_val_at_quad_tensor_dim_dim, face_val_at_shape_quad_scalar,&
          & face_val_at_shape_quad_vector, face_val_at_shape_quad_tensor
  end interface

  interface ele_grad_at_quad
     module procedure ele_grad_at_quad_scalar, ele_grad_at_quad_vector
  end interface

  interface node_val
     module procedure node_val_scalar, node_val_vector, node_val_tensor, &
          & node_val_scalar_v, node_val_vector_v, node_val_vector_dim_v,&
          & node_val_tensor_v, node_val_vector_dim, node_val_tensor_dim_dim, &
          & node_val_tensor_dim_dim_v
  end interface

  interface node_count
     module procedure node_count_scalar, node_count_vector,&
          & node_count_tensor, node_count_mesh
  end interface

  interface node_ele
     module procedure node_ele_mesh, node_ele_scalar, node_ele_vector,&
          & node_ele_tensor
  end interface

  interface element_count
     module procedure element_count_scalar, element_count_vector,&
          & element_count_tensor, element_count_mesh
  end interface

  interface ele_count
     module procedure element_count_scalar, element_count_vector,&
          & element_count_tensor, element_count_mesh
  end interface

  interface face_count
    module procedure face_count_scalar, face_count_vector, &
          & face_count_tensor, face_count_mesh
  end interface

  interface surface_element_count
    module procedure surface_element_count_scalar, surface_element_count_vector, &
          & surface_element_count_tensor, surface_element_count_mesh
  end interface

  interface surface_element_id
    module procedure surface_element_id_scalar, surface_element_id_vector, &
      surface_element_id_mesh
  end interface

  interface ele_region_id
    module procedure ele_region_id_mesh, ele_region_id_scalar, &
      & ele_region_id_vector, ele_region_id_tensor
  end interface

  interface ele_region_ids
    module procedure ele_region_ids_mesh, ele_region_ids_scalar, &
      & ele_region_ids_vector, ele_region_ids_tensor
  end interface

  interface continuity
     module procedure continuity_scalar, continuity_vector,&
          & continuity_tensor, continuity_mesh
  end interface

  interface element_degree
     module procedure element_degree_mesh, element_degree_scalar, &
          & element_degree_vector, element_degree_tensor
  end interface

  interface has_faces
     module procedure has_faces_mesh
  end interface

  interface mesh_dim
     module procedure mesh_dim_mesh, mesh_dim_scalar, mesh_dim_vector,&
          & mesh_dim_tensor
  end interface

  interface mesh_periodic
     module procedure mesh_periodic_mesh, mesh_periodic_scalar, mesh_periodic_vector, &
          & mesh_periodic_tensor
  end interface

  interface has_internal_boundaries
     module procedure mesh_has_internal_boundaries
  end interface has_internal_boundaries

  interface extract_scalar_field ! extract_scalar_field is already used in State.F90
     module procedure extract_scalar_field_from_vector_field, extract_scalar_field_from_tensor_field
  end interface

  interface field2file
     module procedure field2file_scalar, field2file_vector
  end interface

  interface halo_count
    module procedure halo_count_mesh, halo_count_scalar, halo_count_vector, &
      & halo_count_tensor
  end interface halo_count

  interface element_halo_count
    module procedure element_halo_count_mesh, element_halo_count_scalar, &
      & element_halo_count_vector, element_halo_count_tensor
  end interface element_halo_count

  interface operator(==)
     module procedure mesh_equal
  end interface

  interface local_coords
    module procedure local_coords_interpolation, &
          local_coords_interpolation_all, local_coords_mesh, &
          local_coords_scalar, local_coords_vector, local_coords_tensor
  end interface

  interface field_val
    module procedure field_val_scalar, field_val_vector
  end interface field_val

  interface eval_field
    module procedure eval_field_scalar, eval_field_vector, eval_field_tensor
  end interface eval_field

  interface face_eval_field
    module procedure face_eval_field_scalar, face_eval_field_vector, face_eval_field_tensor, &
       face_eval_field_vector_dim, face_eval_field_tensor_dim_dim
  end interface face_eval_field

  interface set_from_python_function
     module procedure set_values_from_python_scalar, set_values_from_python_scalar_pos, &
       set_values_from_python_vector, set_values_from_python_vector_pos, &
       set_values_from_python_vector_field
  end interface

  interface tetvol
    module procedure tetvol_old
  end interface tetvol

  interface face_opposite
    module procedure face_opposite_mesh, face_opposite_scalar, face_opposite_vector, &
      face_opposite_tensor
  end interface

  interface mesh_compatible
    module procedure mesh_compatible, mesh_positions_compatible
  end interface

  interface print_mesh_incompatibility
    module procedure print_mesh_incompatibility, print_mesh_positions_incompatibility
  end interface

  interface refresh_topology
     module procedure refresh_topology_mesh, &
          refresh_topology_vector_field
  end interface refresh_topology

  interface write_minmax
    module procedure write_minmax_scalar, write_minmax_vector, write_minmax_tensor
  end interface

contains

  pure function mesh_dim_mesh(mesh) result (mesh_dim)
    ! Return the dimensionality of the mesh.
    integer :: mesh_dim
    type(mesh_type), intent(in) :: mesh

    mesh_dim=mesh%shape%dim

  end function mesh_dim_mesh

  pure function mesh_dim_scalar(field) result (mesh_dim)
    ! Return the dimensionality of the field.
    integer :: mesh_dim
    type(scalar_field), intent(in) :: field

    mesh_dim=field%mesh%shape%dim

  end function mesh_dim_scalar

  pure function mesh_dim_vector(field) result (mesh_dim)
    ! Return the dimensionality of the field.
    integer :: mesh_dim
    type(vector_field), intent(in) :: field

    mesh_dim=field%mesh%shape%dim

  end function mesh_dim_vector

  pure function mesh_dim_tensor(field) result (mesh_dim)
    ! Return the dimensionality of the field.
    integer :: mesh_dim
    type(tensor_field), intent(in) :: field

    mesh_dim=field%mesh%shape%dim

  end function mesh_dim_tensor

  pure function mesh_periodic_mesh(mesh) result (mesh_periodic)
    ! Return the periodic flag of the mesh
    logical :: mesh_periodic
    type(mesh_type), intent(in) :: mesh

    mesh_periodic=mesh%periodic

  end function mesh_periodic_mesh

  pure function mesh_periodic_scalar(field) result (mesh_periodic)
    ! Return the periodic flag of the mesh
    logical :: mesh_periodic
    type(scalar_field), intent(in) :: field

    mesh_periodic=field%mesh%periodic

  end function mesh_periodic_scalar

  pure function mesh_periodic_vector(field) result (mesh_periodic)
    ! Return the periodic flag of the mesh
    logical :: mesh_periodic
    type(vector_field), intent(in) :: field

    mesh_periodic=field%mesh%periodic

  end function mesh_periodic_vector

  pure function mesh_periodic_tensor(field) result (mesh_periodic)
    ! Return the periodic flag of the mesh
    logical :: mesh_periodic
    type(tensor_field), intent(in) :: field

    mesh_periodic=field%mesh%periodic

  end function mesh_periodic_tensor

  pure function mesh_has_internal_boundaries(mesh)
    !!< Return whether the mesh has internal boundaries
    logical :: mesh_has_internal_boundaries
    type(mesh_type), intent(in) :: mesh

    if (associated(mesh%faces)) then
      mesh_has_internal_boundaries = mesh%faces%has_internal_boundaries
    else
      mesh_has_internal_boundaries = .false.
    end if

  end function mesh_has_internal_boundaries

  pure function node_count_mesh(mesh) result (node_count)
    ! Return the number of nodes in a mesh.
    integer :: node_count
    type(mesh_type), intent(in) :: mesh

    node_count=mesh%nodes

  end function node_count_mesh

  pure function node_count_scalar(field) result (node_count)
    ! Return the number of nodes in a field.
    integer :: node_count
    type(scalar_field),intent(in) :: field

    node_count=node_count_mesh(field%mesh)

  end function node_count_scalar

  pure function node_count_vector(field) result (node_count)
    ! Return the number of nodes in a field.
    integer :: node_count
    type(vector_field),intent(in) :: field

    node_count=node_count_mesh(field%mesh)

  end function node_count_vector

  pure function node_count_tensor(field) result (node_count)
    ! Return the number of nodes in a field.
    integer :: node_count
    type(tensor_field),intent(in) :: field

    node_count=node_count_mesh(field%mesh)

  end function node_count_tensor

  function node_ele_mesh(mesh, node) result (node_ele)
    ! Return the element to which node belongs in mesh. This is only a sane
    ! thing to do for dg meshes: for cg it is undefined as nodes may belong
    ! to multiple elements.
    integer :: node_ele
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: node

    assert(mesh%continuity<0)
    assert(node>=1)
    assert(node<=mesh%nodes)

    ! Note that this will have to change for mixed element meshes.
    node_ele=1+(node-1)/ele_loc(mesh,1)

  end function node_ele_mesh

  function node_ele_scalar(field, node) result (node_ele)
    ! Return the element to which node belongs in field. This is only a sane
    ! thing to do for dg meshes: for cg it is undefined as nodes may belong
    ! to multiple elements.
    integer :: node_ele
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: node

    node_ele=node_ele_mesh(field%mesh, node)

  end function node_ele_scalar

  function node_ele_vector(field, node) result (node_ele)
    ! Return the element to which node belongs in field. This is only a sane
    ! thing to do for dg meshes: for cg it is undefined as nodes may belong
    ! to multiple elements.
    integer :: node_ele
    type(vector_field), intent(in) :: field
    integer, intent(in) :: node

    node_ele=node_ele_mesh(field%mesh, node)

  end function node_ele_vector

  function node_ele_tensor(field, node) result (node_ele)
    ! Return the element to which node belongs in field. This is only a sane
    ! thing to do for dg meshes: for cg it is undefined as nodes may belong
    ! to multiple elements.
    integer :: node_ele
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: node

    node_ele=node_ele_mesh(field%mesh, node)

  end function node_ele_tensor

  pure function element_count_mesh(mesh) result (element_count)
    ! Return the number of nodes in a mesh.
    integer :: element_count
    type(mesh_type),intent(in) :: mesh

    element_count=mesh%elements

  end function element_count_mesh

  pure function surface_element_count_scalar(field) result (element_count)
    ! Return the number of surface elements in a field.
    integer :: element_count
    type(scalar_field),intent(in) :: field

    if (associated(field%mesh%faces)) then
      element_count=size(field%mesh%faces%boundary_ids)
    else
      element_count=0
    end if

  end function surface_element_count_scalar

  pure function surface_element_count_vector(field) result (element_count)
    ! Return the number of surface elements in a field.
    integer :: element_count
    type(vector_field),intent(in) :: field

    if (associated(field%mesh%faces)) then
      element_count=size(field%mesh%faces%boundary_ids)
    else
      element_count=0
    end if

  end function surface_element_count_vector

  pure function surface_element_count_tensor(field) result (element_count)
    ! Return the number of surface elements in a field.
    integer :: element_count
    type(tensor_field),intent(in) :: field

    if (associated(field%mesh%faces)) then
      element_count=size(field%mesh%faces%boundary_ids)
    else
      element_count=0
    end if

  end function surface_element_count_tensor

  pure function surface_element_count_mesh(mesh) result (element_count)
    ! Return the number of surface elements in a mesh.
    integer :: element_count
    type(mesh_type),intent(in) :: mesh

    if (associated(mesh%faces)) then
      element_count=size(mesh%faces%boundary_ids)
    else
      element_count=0
    end if

  end function surface_element_count_mesh

  pure function face_count_scalar(field) result (face_count)
    ! Return the number of faces in a mesh.
    integer :: face_count
    type(scalar_field),intent(in) :: field

    if (associated(field%mesh%faces)) then
      face_count=size(field%mesh%faces%face_element_list)
    else
      face_count=0
    end if

  end function face_count_scalar

  pure function face_count_vector(field) result (face_count)
    ! Return the number of faces in a mesh.
    integer :: face_count
    type(vector_field),intent(in) :: field

    if (associated(field%mesh%faces)) then
      face_count=size(field%mesh%faces%face_element_list)
    else
      face_count=0
    end if

  end function face_count_vector

  pure function face_count_tensor(field) result (face_count)
    ! Return the number of faces in a mesh.
    integer :: face_count
    type(tensor_field),intent(in) :: field

    if (associated(field%mesh%faces)) then
      face_count=size(field%mesh%faces%face_element_list)
    else
      face_count=0
    end if

  end function face_count_tensor

  pure function face_count_mesh(mesh) result (face_count)
    ! Return the number of faces in a mesh.
    integer :: face_count
    type(mesh_type),intent(in) :: mesh

    if (associated(mesh%faces)) then
      face_count=size(mesh%faces%face_element_list)
    else
      face_count=0
    end if

  end function face_count_mesh

  pure function element_count_scalar(field) result (element_count)
    ! Return the number of elements in a field.
    integer :: element_count
    type(scalar_field),intent(in) :: field

    element_count=field%mesh%elements

  end function element_count_scalar

  pure function element_count_vector(field) result (element_count)
    ! Return the number of elements in a field.
    integer :: element_count
    type(vector_field),intent(in) :: field

    element_count=field%mesh%elements

  end function element_count_vector

  pure function element_count_tensor(field) result (element_count)
    ! Return the number of elements in a field.
    integer :: element_count
    type(tensor_field),intent(in) :: field

    element_count=field%mesh%elements

  end function element_count_tensor

  pure function surface_element_id_mesh(mesh, ele) result (id)
    !!< Return the boundary id of the given surface element
    type(mesh_type), intent(in):: mesh
    integer, intent(in):: ele
    integer id

    ! sorry can't assert in pure
    !assert(associated(mesh%faces))
    !assert(ele>0 .and. ele<size(mesh%faces%boundary_ids))

    id=mesh%faces%boundary_ids(ele)

  end function surface_element_id_mesh

  pure function surface_element_id_scalar(field, ele) result (id)
    !!< Return the boundary id of the given surface element
    type(scalar_field), intent(in):: field
    integer, intent(in):: ele
    integer id

    ! sorry can't assert in pure
    !assert(associated(field%mesh%faces))
    !assert(ele>0 .and. ele<size(field%mesh%faces%boundary_ids))

    id=field%mesh%faces%boundary_ids(ele)

  end function surface_element_id_scalar

  pure function surface_element_id_vector(field, ele) result (id)
    !!< Return the boundary id of the given surface element
    type(vector_field), intent(in):: field
    integer, intent(in):: ele
    integer id

    ! sorry can't assert in pure
    !assert(associated(field%mesh%faces))
    !assert(ele>0 .and. ele<size(field%mesh%faces%boundary_ids))

    id=field%mesh%faces%boundary_ids(ele)

  end function surface_element_id_vector

  pure function ele_region_id_mesh(mesh, ele) result(id)
    !!< Return the region id of an element

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele

    integer :: id

    id = mesh%region_ids(ele)

  end function ele_region_id_mesh

  pure function ele_region_id_scalar(field, ele) result(id)
    !!< Return the region id of an element

    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: id

    id = ele_region_id(field%mesh, ele)

  end function ele_region_id_scalar

  pure function ele_region_id_vector(field, ele) result(id)
    !!< Return the region id of an element

    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: id

    id = ele_region_id(field%mesh, ele)

  end function ele_region_id_vector

  pure function ele_region_id_tensor(field, ele) result(id)
    !!< Return the region id of an element

    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: id

    id = ele_region_id(field%mesh, ele)

  end function ele_region_id_tensor

  function ele_region_ids_mesh(mesh) result(ids)
    !!< Return the region ids of all elements

    type(mesh_type), target, intent(in) :: mesh

    integer, dimension(:), pointer :: ids

    ids => mesh%region_ids

  end function ele_region_ids_mesh

  function ele_region_ids_scalar(field) result(ids)
    !!< Return the region ids of all elements

    type(scalar_field), target, intent(in) :: field

    integer, dimension(:), pointer :: ids

    ids => ele_region_ids(field%mesh)

  end function ele_region_ids_scalar

  function ele_region_ids_vector(field) result(ids)
    !!< Return the region ids of all elements

    type(vector_field), target, intent(in) :: field

    integer, dimension(:), pointer :: ids

    ids => ele_region_ids(field%mesh)

  end function ele_region_ids_vector

  function ele_region_ids_tensor(field) result(ids)
    !!< Return the region ids of all elements

    type(tensor_field), target, intent(in) :: field

    integer, dimension(:), pointer :: ids

    ids => ele_region_ids(field%mesh)

  end function ele_region_ids_tensor

  function mesh_connectivity(mesh) result (ndglno)
    !!< Assuming that the input mesh is at least C0, return the connectivity
    !!< of the mesh.
    type(mesh_type), intent(in) :: mesh
    integer, dimension(mesh%elements*mesh%shape%cell%entity_counts(0)) ::&
         & ndglno

    integer, dimension(mesh%shape%cell%entity_counts(0)) :: vertices
    integer :: i, nodes

    integer, dimension(:), allocatable :: map
    integer, dimension(:), pointer :: e_nodes

    vertices=local_vertices(mesh%shape%numbering)

    do i=1,mesh%elements
       e_nodes => ele_nodes(mesh, i)
       ndglno((i-1)*size(vertices)+1:i*size(vertices)) = e_nodes(vertices)
    end do

    allocate(map(maxval(ndglno)))

    map=0
    nodes=0

    do i=1,size(ndglno)
       if (map(ndglno(i))==0) then
          nodes=nodes+1
          map(ndglno(i))=nodes
       end if

       ndglno(i)=map(ndglno(i))
    end do

  end function mesh_connectivity

  pure function mesh_equal(mesh1, mesh2)
    !!< Test for the equality of two meshes. This is not totally safe since
    !!< we do not compare all of ndglno. We assume that two meshes are equal
    !!< if they have the same element, continuity, node count and element
    !!< count. This should be sufficient in almost all circumstances.
    logical :: mesh_equal
    type(mesh_type), intent(in) :: mesh1, mesh2

    if (associated(mesh1%refcount) .and. associated(mesh2%refcount)) then
       mesh_equal= (mesh1%refcount%id==mesh2%refcount%id)
    else
       mesh_equal= .false.
    end if

  end function mesh_equal

  function mesh_compatible(test_mesh, reference_mesh) result(pass)
    !!< Tests if a field on test_mesh is suitable for initialising a field
    !!< on reference_mesh.

    type(mesh_type), intent(in) :: test_mesh
    type(mesh_type), intent(in) :: reference_mesh

    logical :: pass

    pass = node_count(test_mesh) == node_count(reference_mesh) .and. &
      &ele_count(test_mesh) == ele_count(reference_mesh) .and. &
      &continuity(test_mesh) == continuity(reference_mesh)

  end function mesh_compatible

  function mesh_positions_compatible(test_positions, reference_positions) result(pass)
    !!< Tests if two meshes (including its positions) are the same

    type(vector_field), intent(in) :: test_positions, reference_positions
    logical :: pass

    real:: L
    integer:: i

    ! first test if the topological meshes are the same
    if (.not. mesh_compatible(test_positions%mesh, reference_positions%mesh)) then
      pass=.false.
      return
    end if

    do i=1, reference_positions%dim
      L=maxval(reference_positions%val(i,:))-minval(reference_positions%val(i,:))
      if (maxval(abs(test_positions%val(i,:)-reference_positions%val(i,:)))>1e-9*L) then
        pass=.false.
        return
      end if
    end do

    pass=.true.

  end function mesh_positions_compatible

  subroutine print_mesh_incompatibility(debug_level, test_mesh, reference_mesh)
    !!< Tests if a field on test_mesh is suitable for initialising a field
    !!< on reference_mesh, and prints a descriptive message.

    integer, intent(in) :: debug_level
    type(mesh_type), intent(in) :: test_mesh
    type(mesh_type), intent(in) :: reference_mesh

    if(node_count(test_mesh) /= node_count(reference_mesh)) then
      ewrite(debug_level, *) "Node counts do not match"
    end if
    if(ele_count(test_mesh) /= ele_count(reference_mesh)) then
      ewrite(debug_level, *) "Element counts do not match"
    end if
    if(continuity(test_mesh) /= continuity(reference_mesh)) then
      ewrite(debug_level, *) "Continuities do not match"
    end if

  end subroutine print_mesh_incompatibility

  subroutine print_mesh_positions_incompatibility(debug_level, test_positions, reference_positions)
    !!< Tests if two meshes are the same (including its positions) and prints a descriptive message.

    integer, intent(in) :: debug_level
    type(vector_field), intent(in) :: test_positions, reference_positions

    if (.not. mesh_compatible(test_positions%mesh, reference_positions%mesh)) then
      call print_mesh_incompatibility(debug_level, test_positions%mesh, reference_positions%mesh)
    else if (.not. mesh_compatible(test_positions, reference_positions)) then
      ewrite(debug_level, *) "Node positions do not match"
    end if

  end subroutine print_mesh_positions_incompatibility

  function ele_faces_mesh(mesh, ele_number) result (ele_faces)
    !!< Return a pointer to a vector containing the face numbers of the
    !!< faces adjacent to ele_number.
    integer, dimension(:), pointer :: ele_faces
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    ele_faces =>row_ival_ptr(mesh%faces%face_list, ele_number)

  end function ele_faces_mesh

  function ele_faces_scalar(field, ele_number) result (ele_faces)
    !!< Return a pointer to a vector containing the face numbers of the
    !!< faces adjacent to ele_number.
    integer, dimension(:), pointer :: ele_faces
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_faces=>ele_faces_mesh(field%mesh, ele_number)

  end function ele_faces_scalar

  function ele_faces_vector(field, ele_number) result (ele_faces)
    !!< Return a pointer to a vector containing the face numbers of the
    !!< faces adjacent to ele_number.
    integer, dimension(:), pointer :: ele_faces
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_faces=>ele_faces_mesh(field%mesh, ele_number)

  end function ele_faces_vector

  function ele_faces_tensor(field, ele_number) result (ele_faces)
    !!< Return a pointer to a vector containing the face numbers of the
    !!< faces adjacent to ele_number.
    integer, dimension(:), pointer :: ele_faces
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_faces=>ele_faces_mesh(field%mesh, ele_number)

  end function ele_faces_tensor

  function ele_neigh_mesh(mesh, ele_number) result (ele_neigh)
    !!< Return a pointer to a vector containing the element numbers of the
    !!< elements adjacent to ele_number.
    integer, dimension(:), pointer :: ele_neigh
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    ele_neigh=>row_m_ptr(mesh%faces%face_list, ele_number)

  end function ele_neigh_mesh

  function ele_neigh_scalar(field, ele_number) result (ele_neigh)
    !!< Return a pointer to a vector containing the element numbers of the
    !!< elements adjacent to ele_number.
    integer, dimension(:), pointer :: ele_neigh
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_neigh=>ele_neigh_mesh(field%mesh, ele_number)

  end function ele_neigh_scalar

  function ele_neigh_vector(field, ele_number) result (ele_neigh)
    !!< Return a pointer to a vector containing the element numbers of the
    !!< elements adjacent to ele_number.
    integer, dimension(:), pointer :: ele_neigh
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_neigh=>ele_neigh_mesh(field%mesh, ele_number)

  end function ele_neigh_vector

  function ele_neigh_tensor(field, ele_number) result (ele_neigh)
    !!< Return a pointer to a vector containing the element numbers of the
    !!< elements adjacent to ele_number.
    integer, dimension(:), pointer :: ele_neigh
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_neigh=>ele_neigh_mesh(field%mesh, ele_number)

  end function ele_neigh_tensor

  function ele_neigh_i_mesh(mesh, ele_number, neigh_number) result&
       & (ele_neigh)
    !!< Return the neigh_numberth neighbour of ele_number.
    integer :: ele_neigh
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele_number, neigh_number

    integer, dimension(:), pointer :: neighlist

    neighlist=>ele_neigh_mesh(mesh, ele_number)

    ele_neigh=neighlist(neigh_number)

  end function ele_neigh_i_mesh

  function ele_neigh_i_scalar(field, ele_number, neigh_number) result&
       & (ele_neigh)
    !!< Return the neigh_numberth neighbour of ele_number.
    integer :: ele_neigh
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele_number, neigh_number

    ele_neigh=ele_neigh_i_mesh(field%mesh, ele_number, neigh_number)

  end function ele_neigh_i_scalar

  function ele_neigh_i_vector(field, ele_number, neigh_number) result&
       & (ele_neigh)
    !!< Return the neigh_numberth neighbour of ele_number.
    integer :: ele_neigh
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele_number, neigh_number

    ele_neigh=ele_neigh_i_mesh(field%mesh, ele_number, neigh_number)

  end function ele_neigh_i_vector

  function ele_neigh_i_tensor(field, ele_number, neigh_number) result&
       & (ele_neigh)
    !!< Return the neigh_numberth neighbour of ele_number.
    integer :: ele_neigh
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele_number, neigh_number

    ele_neigh=ele_neigh_i_mesh(field%mesh, ele_number, neigh_number)

  end function ele_neigh_i_tensor

  function node_neigh_mesh(mesh, node_number) result (node_neigh)
    !!< Return a pointer to a vector containing the element numbers of the
    !!< elements containing node_number.
    integer, dimension(:), pointer :: node_neigh
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: node_number

    assert(associated(mesh%adj_lists))
    if (.not. associated(mesh%adj_lists%nelist)) then
      ewrite(-1,*) "nelist not initialised. I could allocate it myself,"
      ewrite(-1,*) "but you're probably calling this a lot."
      ewrite(-1,*) " call add_nelist(mesh) first"
      FLAbort("Call add_nelist(mesh) before calling node_neigh.")
    end if

    node_neigh => row_m_ptr(mesh%adj_lists%nelist, node_number)
  end function node_neigh_mesh

  function node_neigh_scalar(field, node_number) result (node_neigh)
    integer, dimension(:), pointer :: node_neigh
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: node_number

    node_neigh => node_neigh_mesh(field%mesh, node_number)
  end function node_neigh_scalar

  function node_neigh_vector(field, node_number) result (node_neigh)
    integer, dimension(:), pointer :: node_neigh
    type(vector_field),intent(in) :: field
    integer, intent(in) :: node_number

    node_neigh => node_neigh_mesh(field%mesh, node_number)
  end function node_neigh_vector

  function node_neigh_tensor(field, node_number) result (node_neigh)
    integer, dimension(:), pointer :: node_neigh
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: node_number

    node_neigh => node_neigh_mesh(field%mesh, node_number)
  end function node_neigh_tensor

  ! Returns the neighbouring face of a given face. The incoming face number is returned if no neighbour face exists.
  function face_neigh_mesh(mesh, face) result(face_neigh)
   integer, intent(in) :: face
   type(mesh_type), intent(in) :: mesh
   integer :: face_neigh
   integer :: ele, ele2, i
   integer, dimension(:), pointer :: ele_neighs

   ele=face_ele(mesh, face)
   ele_neighs=>ele_neigh(mesh, ele)

   face_neigh=face
   ! Search for a neighbour which shares the same face
   do i=1, size(ele_neighs)
        ele2=ele_neighs(i)
        if (ele2.le.0) then
             continue
        elseif (ele_face(mesh, ele, ele2)==face) then
            face_neigh=ele_face(mesh, ele2, ele)
            exit
        end if
   end do
 end function face_neigh_mesh

  ! Returns the neighbouring face of a given face. The incoming face number is returned if no neighbour face exists.
  function face_neigh_scalar(field, face) result(face_neigh)
   integer, intent(in) :: face
   type(scalar_field), intent(in) :: field
   integer :: face_neigh
   integer :: ele, ele2, i
   integer, dimension(:), pointer :: ele_neighs

   ele=face_ele(field%mesh, face)
   ele_neighs=>ele_neigh(field%mesh, ele)

   face_neigh=face
   ! Search for a neighbour which shares the same face
   do i=1, size(ele_neighs)
        ele2=ele_neighs(i)
        if (ele2.le.0) then
             continue
        elseif (ele_face(field%mesh, ele, ele2)==face) then
            face_neigh=ele_face(field%mesh, ele2, ele)
            exit
        end if
   end do
 end function face_neigh_scalar

  ! Returns the neighbouring face of a given face. The incoming face number is returned if no neighbour face exists.
  function face_neigh_vector(field, face) result(face_neigh)
   integer, intent(in) :: face
   type(vector_field), intent(in) :: field
   integer :: face_neigh
   integer :: ele, ele2, i
   integer, dimension(:), pointer :: ele_neighs

   ele=face_ele(field%mesh, face)
   ele_neighs=>ele_neigh(field%mesh, ele)

   face_neigh=face
   ! Search for a neighbour which shares the same face
   do i=1, size(ele_neighs)
        ele2=ele_neighs(i)
        if (ele2.le.0) then
             continue
        elseif (ele_face(field%mesh, ele, ele2)==face) then
            face_neigh=ele_face(field%mesh, ele2, ele)
            exit
        end if
   end do
 end function face_neigh_vector

  ! Returns the neighbouring face of a given face. The incoming face number is returned if no neighbour face exists.
  function face_neigh_tensor(field, face) result(face_neigh)
   integer, intent(in) :: face
   type(tensor_field), intent(in) :: field
   integer :: face_neigh
   integer :: ele, ele2, i
   integer, dimension(:), pointer :: ele_neighs

   ele=face_ele(field%mesh, face)
   ele_neighs=>ele_neigh(field%mesh, ele)

   face_neigh=face
   ! Search for a neighbour which shares the same face
   do i=1, size(ele_neighs)
        ele2=ele_neighs(i)
        if (ele2.le.0) then
             continue
        elseif (ele_face(field%mesh, ele, ele2)==face) then
            face_neigh=ele_face(field%mesh, ele2, ele)
            exit
        end if
   end do
 end function face_neigh_tensor

  pure function ele_face_count_mesh(mesh, ele_number) result (face_count)
    !!< Return the number of faces associated with ele_number.
    integer :: face_count
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele_number

    type(element_type), pointer :: shape

    face_count=facet_count(mesh%shape)

  end function ele_face_count_mesh

  pure function ele_face_count_scalar(field, ele_number) result (face_count)
    integer :: face_count
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele_number

    face_count=ele_face_count_mesh(field%mesh, ele_number)

  end function ele_face_count_scalar

  pure function ele_face_count_vector(field, ele_number) result (face_count)
    integer :: face_count
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele_number

    face_count=ele_face_count_mesh(field%mesh, ele_number)

  end function ele_face_count_vector

  pure function ele_face_count_tensor(field, ele_number) result (face_count)
    integer :: face_count
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele_number

    face_count=ele_face_count_mesh(field%mesh, ele_number)

  end function ele_face_count_tensor

  function ele_face_mesh(mesh, ele_number1, ele_number2) result (ele_face)
    ! Return the face in ele1 adjacent to ele2.
    integer ele_face
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele_number1, ele_number2

    ele_face=ival(mesh%faces%face_list, ele_number1, ele_number2)

  end function ele_face_mesh

  function ele_face_scalar(field, ele_number1, ele_number2) result (ele_face)
    ! Return the face in ele1 adjacent to ele2.
    integer ele_face
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele_number1, ele_number2

    ele_face=ele_face_mesh(field%mesh, ele_number1, ele_number2)

  end function ele_face_scalar

  function ele_face_vector(field, ele_number1, ele_number2) result (ele_face)
    ! Return the face in ele1 adjacent to ele2.
    integer ele_face
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele_number1, ele_number2

    ele_face=ele_face_mesh(field%mesh, ele_number1, ele_number2)

  end function ele_face_vector

  function ele_face_tensor(field, ele_number1, ele_number2) result (ele_face)
    ! Return the face in ele1 adjacent to ele2.
    integer ele_face
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele_number1, ele_number2

    ele_face=ele_face_mesh(field%mesh, ele_number1, ele_number2)

  end function ele_face_tensor

  elemental function face_ele_mesh(mesh, face_number) result (face_ele)
    ! Return the index of the element of which face is a part.
    integer :: face_ele
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: face_number

    face_ele=mesh%faces%face_element_list(face_number)

  end function face_ele_mesh

  pure function cell_family_mesh(mesh, ele) result(family)
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele

    integer :: family

    family = cell_family(mesh%shape)

  end function cell_family_mesh

  pure function cell_family_scalar(field, ele) result(family)
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: family

    family = cell_family(field%mesh%shape)

  end function cell_family_scalar

  pure function cell_family_vector(field, ele) result(family)
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: family

    family = cell_family(field%mesh%shape)

  end function cell_family_vector

  pure function cell_family_tensor(field, ele) result(family)
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: family

    family = cell_family(field%mesh%shape)

  end function cell_family_tensor

  pure function element_type_shape(shape) result(family)
    type(element_type), intent(in) :: shape

    integer :: family

    family = shape%type

  end function element_type_shape

  pure function element_type_mesh(mesh, ele) result(family)
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele

    integer :: family

    family = mesh%shape%type

  end function element_type_mesh

  pure function element_type_scalar(field, ele) result(family)
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: family

    family = field%mesh%shape%type

  end function element_type_scalar

  pure function element_type_vector(field, ele) result(family)
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: family

    family = field%mesh%shape%type

  end function element_type_vector

  pure function element_type_tensor(field, ele) result(family)
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele

    integer :: family

    family = field%mesh%shape%type

  end function element_type_tensor

  pure function ele_n_constraints_vector(vfield, ele_number) result&
       & (ele_n_constraints)
    integer :: ele_n_constraints
    type(vector_field), intent(in) :: vfield
    integer, intent(in) :: ele_number

    ele_n_constraints = vfield%mesh%shape%constraints%n_constraints

  end function ele_n_constraints_vector

  pure function ele_loc_mesh(mesh, ele_number) result (ele_loc)
    ! Return the number of nodes of element ele_number.
    integer :: ele_loc
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    ele_loc=mesh%shape%ndof

  end function ele_loc_mesh

  pure function ele_loc_scalar(field, ele_number) result (ele_loc)
    ! Return the number of nodes of element ele_number.
    integer :: ele_loc
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_loc=field%mesh%shape%ndof

  end function ele_loc_scalar

  pure function ele_loc_vector(field, ele_number) result (ele_loc)
    ! Return the number of nodes of element ele_number.
    integer :: ele_loc
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_loc=field%mesh%shape%ndof

  end function ele_loc_vector

  pure function ele_loc_tensor(field, ele_number) result (ele_loc)
    ! Return the number of nodes of element ele_number.
    integer :: ele_loc
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_loc=field%mesh%shape%ndof

  end function ele_loc_tensor

  pure function ele_vertices_mesh(mesh, ele_number) result (ele_vertices)
    ! Return the number of vertices of element ele_number.
    integer :: ele_vertices
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    ele_vertices=mesh%shape%quadrature%vertices

  end function ele_vertices_mesh

  pure function ele_vertices_scalar(field, ele_number) result (ele_vertices)
    ! Return the number of vertices of element ele_number.
    integer :: ele_vertices
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_vertices=field%mesh%shape%quadrature%vertices

  end function ele_vertices_scalar

  pure function ele_vertices_vector(field, ele_number) result (ele_vertices)
    ! Return the number of vertices of element ele_number.
    integer :: ele_vertices
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_vertices=field%mesh%shape%quadrature%vertices

  end function ele_vertices_vector

  pure function ele_vertices_tensor(field, ele_number) result (ele_vertices)
    ! Return the number of vertices of element ele_number.
    integer :: ele_vertices
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_vertices=field%mesh%shape%quadrature%vertices

  end function ele_vertices_tensor

  pure function face_vertices_mesh(mesh, face_number) result (face_vertices)
    ! Return the number of vertices of face face_number.
    integer :: face_vertices
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: face_number

    face_vertices=mesh%faces%shape%quadrature%vertices

  end function face_vertices_mesh

  pure function face_vertices_scalar(field, face_number) result (face_vertices)
    ! Return the number of vertices of face face_number.
    integer :: face_vertices
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_vertices=field%mesh%faces%shape%quadrature%vertices

  end function face_vertices_scalar

  pure function face_vertices_vector(field, face_number) result (face_vertices)
    ! Return the number of vertices of face face_number.
    integer :: face_vertices
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_vertices=field%mesh%faces%shape%quadrature%vertices

  end function face_vertices_vector

  pure function face_vertices_tensor(field, face_number) result (face_vertices)
    ! Return the number of vertices of face face_number.
    integer :: face_vertices
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_vertices=field%mesh%faces%shape%quadrature%vertices

  end function face_vertices_tensor

  pure function face_loc_mesh(mesh, face_number) result (face_loc)
    ! Return the number of nodes of face face_number.
    integer :: face_loc
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: face_number

    face_loc=mesh%faces%shape%ndof

  end function face_loc_mesh

  pure function face_loc_scalar(field, face_number) result (face_loc)
    ! Return the number of nodes of face face_number.
    integer :: face_loc
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_loc=field%mesh%faces%shape%ndof

  end function face_loc_scalar

  pure function face_loc_vector(field, face_number) result (face_loc)
    ! Return the number of nodes of face face_number.
    integer :: face_loc
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_loc=field%mesh%faces%shape%ndof

  end function face_loc_vector

  pure function face_loc_tensor(field, face_number) result (face_loc)
    ! Return the number of nodes of face face_number.
    integer :: face_loc
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_loc=field%mesh%faces%shape%ndof

  end function face_loc_tensor

  pure function ele_and_faces_loc_mesh(mesh, ele_number) result&
       & (loc)
    ! Return the number of nodes of facement ele_number.
    integer :: loc
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    if (mesh%continuity<0) then
       loc=mesh%shape%ndof + facet_count(mesh%shape) &
            * mesh%faces%shape%ndof
    else
       ! For a continuous mesh the face nodes are among the element nodes.
       loc=mesh%shape%ndof
    end if

  end function ele_and_faces_loc_mesh

  pure function ele_and_faces_loc_scalar(field, ele_number) result&
       & (loc)
    ! Return the number of nodes of facement ele_number.
    integer :: loc
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    loc=ele_and_faces_loc_mesh(field%mesh, ele_number)

  end function ele_and_faces_loc_scalar

  pure function ele_and_faces_loc_vector(field, ele_number) result&
       & (loc)
    ! Return the number of nodes of facement ele_number.
    integer :: loc
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    loc=ele_and_faces_loc_mesh(field%mesh, ele_number)

  end function ele_and_faces_loc_vector

  pure function ele_and_faces_loc_tensor(field, ele_number) result&
       & (loc)
    ! Return the number of nodes of facement ele_number.
    integer :: loc
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    loc=ele_and_faces_loc_mesh(field%mesh, ele_number)

  end function ele_and_faces_loc_tensor

  pure function ele_ngi_mesh(mesh, ele_number) result (ele_ngi)
    ! Return the number of nodes of element ele_number.
    integer :: ele_ngi
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    ele_ngi=mesh%shape%ngi

  end function ele_ngi_mesh

  pure function ele_ngi_scalar(field, ele_number) result (ele_ngi)
    ! Return the number of nodes of element ele_number.
    integer :: ele_ngi
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_ngi=field%mesh%shape%ngi

  end function ele_ngi_scalar

  pure function ele_ngi_vector(field, ele_number) result (ele_ngi)
    ! Return the number of nodes of element ele_number.
    integer :: ele_ngi
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_ngi=field%mesh%shape%ngi

  end function ele_ngi_vector

  pure function ele_ngi_tensor(field, ele_number) result (ele_ngi)
    ! Return the number of nodes of element ele_number.
    integer :: ele_ngi
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_ngi=field%mesh%shape%ngi

  end function ele_ngi_tensor

  pure function face_ngi_mesh(mesh, face_number) result (face_ngi)
    ! Return the number of nodes of element face_number.
    integer :: face_ngi
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: face_number

    face_ngi=mesh%faces%shape%ngi

  end function face_ngi_mesh

  pure function face_ngi_scalar(field, face_number) result (face_ngi)
    ! Return the number of nodes of element face_number.
    integer :: face_ngi
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_ngi=field%mesh%faces%shape%ngi

  end function face_ngi_scalar

  pure function face_ngi_vector(field, face_number) result (face_ngi)
    ! Return the number of nodes of element face_number.
    integer :: face_ngi
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_ngi=field%mesh%faces%shape%ngi

  end function face_ngi_vector

  pure function face_ngi_tensor(field, face_number) result (face_ngi)
    ! Return the number of nodes of element face_number.
    integer :: face_ngi
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_ngi=field%mesh%faces%shape%ngi

  end function face_ngi_tensor

  elemental function face_ele_scalar(field, face_number) result (face_ele)
    ! Return the index of the element of which face is a part.
    integer :: face_ele
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: face_number

    face_ele=field%mesh%faces%face_element_list(face_number)

  end function face_ele_scalar

  elemental function face_ele_vector(field, face_number) result (face_ele)
    ! Return the index of the element of which face is a part.
    integer :: face_ele
    type(vector_field), intent(in) :: field
    integer, intent(in) :: face_number

    face_ele=field%mesh%faces%face_element_list(face_number)

  end function face_ele_vector

  elemental function face_ele_tensor(field, face_number) result (face_ele)
    ! Return the index of the element of which face is a part.
    integer :: face_ele
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: face_number

    face_ele=field%mesh%faces%face_element_list(face_number)

  end function face_ele_tensor

  function local_face_number_mesh(mesh, global_face_number, stat) result (local_face_number)
    ! Return the local face number of the given global face number in element ele_number
    integer :: local_face_number
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: global_face_number
    integer, intent(inout), optional :: stat

    integer :: ele_number, i
    integer, dimension(:), pointer :: element_faces

    local_face_number = 0

    ele_number = face_ele(mesh, global_face_number)

    element_faces => ele_faces(mesh, ele_number)
    do i = 1, size(element_faces)
      if(element_faces(i) == global_face_number) then
        local_face_number = i
        exit
      end if
    end do

    if(local_face_number==0) then
      if(present(stat)) then
        stat = 1
      else
        FLAbort("Failed to find local face number.")
      end if
    else
      if(present(stat)) stat = 0
    end if

  end function local_face_number_mesh

  function local_face_number_scalar(field, global_face_number, stat) result (local_face_number)
    ! Return the local face number of the given global face number in element ele_number
    integer :: local_face_number
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: global_face_number
    integer, intent(inout), optional :: stat

    local_face_number = local_face_number_mesh(field%mesh, global_face_number, stat)

  end function local_face_number_scalar

  function local_face_number_vector(field, global_face_number, stat) result (local_face_number)
    ! Return the local face number of the given global face number in element ele_number
    integer :: local_face_number
    type(vector_field), intent(in) :: field
    integer, intent(in) :: global_face_number
    integer, intent(inout), optional :: stat

    local_face_number = local_face_number_mesh(field%mesh, global_face_number, stat)

  end function local_face_number_vector

  function local_face_number_tensor(field, global_face_number, stat) result (local_face_number)
    ! Return the local face number of the given global face number in element ele_number
    integer :: local_face_number
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: global_face_number
    integer, intent(inout), optional :: stat

    local_face_number = local_face_number_mesh(field%mesh, global_face_number, stat)

  end function local_face_number_tensor

  function ele_nodes_mesh(mesh, ele_number) result (ele_nodes)
    ! Return a pointer to a vector containing the global node numbers of
    ! element ele_number in mesh.
    integer, dimension(:), pointer :: ele_nodes
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: ele_number

    ele_nodes=>mesh%ndglno(mesh%shape%ndof*(ele_number-1)+1:&
         &mesh%shape%ndof*ele_number)

  end function ele_nodes_mesh

  function ele_nodes_scalar(field, ele_number) result (ele_nodes)
    ! Return a pointer to a vector containing the global node numbers of
    ! element ele_number in field.
    integer, dimension(:), pointer :: ele_nodes
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_nodes=>ele_nodes_mesh(field%mesh, ele_number)

  end function ele_nodes_scalar

  function ele_nodes_vector(field, ele_number) result (ele_nodes)
    ! Return a pointer to a vector containing the global node numbers of
    ! element ele_number in field.
    integer, dimension(:), pointer :: ele_nodes
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_nodes=>ele_nodes_mesh(field%mesh, ele_number)

  end function ele_nodes_vector

  function ele_nodes_tensor(field, ele_number) result (ele_nodes)
    ! Return a pointer to a tensor containing the global node numbers of
    ! element ele_number in field.
    integer, dimension(:), pointer :: ele_nodes
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number

    ele_nodes=>ele_nodes_mesh(field%mesh, ele_number)

  end function ele_nodes_tensor

 function face_local_nodes_mesh(mesh, face_number) result (face_nodes)
    ! Return a pointer to a vector containing the local node numbers of
    ! facet face_number in mesh.
    integer, dimension(:), pointer :: face_nodes
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: face_number

    ! This just reduces notational clutter.
    type(mesh_faces), pointer :: faces

    faces=>mesh%faces

    face_nodes=>faces%face_lno(faces%shape%ndof*(face_number-1)+1:&
         &faces%shape%ndof*face_number)

  end function face_local_nodes_mesh

  function face_local_nodes_scalar(field, face_number) result (face_nodes)
    !!< Return a vector containing the local node numbers of
    !!< facet face_number in field.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number
    integer, dimension(face_loc(field, face_number)) :: face_nodes

    face_nodes=face_local_nodes_mesh(field%mesh, face_number)

  end function face_local_nodes_scalar

  function face_local_nodes_vector(field, face_number) result (face_nodes)
    !!< Return a vector containing the local node numbers of
    !!< facet face_number in field.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    integer, dimension(face_loc(field, face_number)) :: face_nodes

    face_nodes=face_local_nodes_mesh(field%mesh, face_number)

  end function face_local_nodes_vector

  function face_local_nodes_tensor(field, face_number) result (face_nodes)
    !!< Return a vector containing the local node numbers of
    !!< facet face_number in field.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number
    integer, dimension(face_loc(field, face_number)) :: face_nodes

    face_nodes=face_local_nodes_mesh(field%mesh, face_number)

  end function face_local_nodes_tensor

  function face_global_nodes_mesh(mesh, face_number) result (face_nodes)
    !!< Return a vector containing the global node numbers of
    !!< facet face_number in mesh.
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: face_number
    integer, dimension(face_loc(mesh, face_number)) :: face_nodes

    integer, dimension(:), pointer :: ele

    assert(has_faces(mesh))

    ele=>ele_nodes(mesh, face_ele(mesh,face_number))

    face_nodes=ele(face_local_nodes_mesh(mesh, face_number))

  end function face_global_nodes_mesh

  function face_global_nodes_scalar(field, face_number) result (face_nodes)
    !!< Return a vector containing the global node numbers of
    !!< facet face_number in field.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number
    integer, dimension(face_loc(field, face_number)) :: face_nodes

    face_nodes=face_global_nodes_mesh(field%mesh, face_number)

  end function face_global_nodes_scalar

  function face_global_nodes_vector(field, face_number) result (face_nodes)
    !!< Return a vector containing the global node numbers of
    !!< facet face_number in field.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    integer, dimension(face_loc(field, face_number)) :: face_nodes

    face_nodes=face_global_nodes_mesh(field%mesh, face_number)

  end function face_global_nodes_vector

  function face_global_nodes_tensor(field, face_number) result (face_nodes)
    !!< Return a vector containing the global node numbers of
    !!< facet face_number in field.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number
    integer, dimension(face_loc(field, face_number)) :: face_nodes

    face_nodes=face_global_nodes_mesh(field%mesh, face_number)

  end function face_global_nodes_tensor

  function ele_shape_mesh(mesh, ele_number) result (ele_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: ele_shape
    type(mesh_type),intent(in), target :: mesh
    integer, intent(in) :: ele_number

    ele_shape=>mesh%shape

  end function ele_shape_mesh

  function ele_shape_scalar(field, ele_number) result (ele_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: ele_shape
    type(scalar_field),intent(in), target :: field
    integer, intent(in) :: ele_number

    ele_shape=>field%mesh%shape

  end function ele_shape_scalar

  function ele_shape_vector(field, ele_number) result (ele_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: ele_shape
    type(vector_field),intent(in), target :: field
    integer, intent(in) :: ele_number

    ele_shape=>field%mesh%shape

  end function ele_shape_vector

  function ele_shape_tensor(field, ele_number) result (ele_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: ele_shape
    type(tensor_field),intent(in), target :: field
    integer, intent(in) :: ele_number

    ele_shape=>field%mesh%shape

  end function ele_shape_tensor

  function face_shape_mesh(mesh, face_number) result (face_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: face_shape
    type(mesh_type),intent(in) :: mesh
    integer, intent(in) :: face_number

    face_shape=>mesh%faces%shape

  end function face_shape_mesh

  function face_shape_scalar(field, face_number) result (face_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: face_shape
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_shape=>field%mesh%faces%shape

  end function face_shape_scalar

  function face_shape_vector(field, face_number) result (face_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: face_shape
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_shape=>field%mesh%faces%shape

  end function face_shape_vector

  function face_shape_tensor(field, face_number) result (face_shape)
    ! Return a pointer to the shape of element ele_number.
    type(element_type), pointer :: face_shape
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number

    face_shape=>field%mesh%faces%shape

  end function face_shape_tensor

  function ele_val_scalar(field, ele_number) result (ele_val_out)
    ! Return the values of field at the nodes of ele_number.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%mesh%shape%ndof) :: ele_val_out
    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      ele_val_out=field%val(ele_nodes(field,ele_number))
    case(FIELD_TYPE_CONSTANT)
      ele_val_out=field%val(1)
    case(FIELD_TYPE_PYTHON)
       call val_python
    end select

  contains

    subroutine val_python
      !!< This subroutine only exists to remove the following stack variables
      !!< from the main routine.
      real, dimension(field%py_dim, field%mesh%shape%ndof) :: pos
      real, dimension(field%py_dim, field%py_positions_shape%ndof) :: tmp_pos

      if (.not. field%py_positions_same_mesh) then
        tmp_pos = ele_val(field%py_positions, ele_number)
        do i=1,field%py_dim
          pos(i, :) = matmul(field%py_locweight, tmp_pos(i, :))
        end do
      else
        do i=1,field%py_dim
          pos(i, :) = field%py_positions%val(i,ele_nodes(field%py_positions%mesh, ele_number))
        end do
      end if
      call set_from_python_function(ele_val_out, trim(field%py_func), pos, &
        time=current_time)

    end subroutine val_python

  end function ele_val_scalar

  function ele_val_vector(field, ele_number) result (ele_val)
    ! Return the values of field at the nodes of ele_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%dim, field%mesh%shape%ndof) :: ele_val

    integer :: i
    integer, dimension(:), pointer :: nodes

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      nodes => ele_nodes(field, ele_number)
      do i=1,field%dim
         ele_val(i, :) = field%val(i,nodes)
      end do
    case(FIELD_TYPE_CONSTANT)
      do i=1,field%dim
         ele_val(i,:)=field%val(i,1)
      end do
    end select


  end function ele_val_vector

  function ele_val_vector_dim(field, dim, ele_number) result (ele_val)
    ! Return the values of dimension dim of field at the nodes of ele_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%mesh%shape%ndof) :: ele_val
    integer, intent(in) :: dim

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      ele_val=field%val(dim,ele_nodes(field,ele_number))
    case(FIELD_TYPE_CONSTANT)
      ele_val=field%val(dim,1)
    end select

  end function ele_val_vector_dim

  function ele_val_tensor(field, ele_number) result (ele_val)
    ! Return the values of field at the nodes of ele_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%dim(1), field%dim(2), field%mesh%shape%ndof) :: ele_val

    integer, dimension(:), pointer :: nodes
    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      nodes=>ele_nodes(field,ele_number)
      ele_val=field%val(:,:,nodes)
    case(FIELD_TYPE_CONSTANT)
      do i=1,size(ele_val, 3)
        ele_val(:, :, i)=field%val(:, :, 1)
      end do
    end select

  end function ele_val_tensor

  function ele_val_tensor_dim_dim(field, dim1, dim2, ele_number) result (ele_val)
    ! Return the values of field at the nodes of ele_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: dim1, dim2
    integer, intent(in) :: ele_number
    real, dimension(field%mesh%shape%ndof) :: ele_val

    integer, dimension(:), pointer :: nodes

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      nodes=>ele_nodes(field,ele_number)
      ele_val=field%val(dim1,dim2,nodes)
    case(FIELD_TYPE_CONSTANT)
       ele_val=field%val(dim1, dim2, 1)
    end select

  end function ele_val_tensor_dim_dim

  function face_val_scalar(field, face_number) result (face_val)
    ! Return the values of field at the nodes of face_number.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(face_loc(field, face_number)) :: face_val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      face_val=field%val(face_global_nodes(field,face_number))
    case(FIELD_TYPE_CONSTANT)
      face_val=field%val(1)
    end select

  end function face_val_scalar

  function face_val_vector(field, face_number) result (face_val)
    ! Return the values of field at the nodes of face_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(field%dim, face_loc(field, face_number)) :: face_val

    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      face_val=field%val(:,face_global_nodes(field,face_number))
    case(FIELD_TYPE_CONSTANT)
      do i=1,field%dim
         face_val(i,:)=field%val(i,1)
      end do
    end select

  end function face_val_vector

  function face_val_vector_dim(field, dim, face_number) result (face_val)
    !!< Return the values of dimension dim of field at the nodes of face_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(face_loc(field, face_number)) :: face_val
    integer :: dim

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      face_val=field%val(dim,face_global_nodes(field,face_number))
    case(FIELD_TYPE_CONSTANT)
      face_val=field%val(dim,1)
    end select

  end function face_val_vector_dim

  function face_val_tensor(field, face_number) result (face_val)
    ! Return the values of field at the nodes of face_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(field%dim(1), field%dim(2), face_loc(field, face_number)) ::&
         & face_val

    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
       face_val=field%val(:,:,face_global_nodes(field,face_number))

    case(FIELD_TYPE_CONSTANT)
       forall(i=1:face_loc(field, face_number))
          face_val(:,:,i)=field%val(:,:,1)
       end forall
    end select

  end function face_val_tensor

  function face_val_tensor_dim_dim(field, dim1, dim2, face_number) result&
       & (face_val)
    !!< Return the values of dimension dim of field at the nodes of face_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: dim1, dim2
    integer, intent(in) :: face_number
    real, dimension(face_loc(field, face_number)) :: face_val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      face_val=field%val(dim1,dim2,face_global_nodes(field,face_number))
    case(FIELD_TYPE_CONSTANT)
      face_val=field%val(dim1,dim2,1)
    end select

  end function face_val_tensor_dim_dim

  function ele_val_at_quad_scalar(field, ele_number) result (quad_val)
    ! Return the values of field at the quadrature points of ele_number.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%mesh%shape%ngi) :: quad_val

    type(element_type), pointer :: shape

    shape=>ele_shape(field,ele_number)
    quad_val=matmul(ele_val(field, ele_number), shape%n)

  end function ele_val_at_quad_scalar

  function ele_val_at_quad_vector(field, ele_number) result (quad_val)
    ! Return the values of field at the quadrature points of ele_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%dim, field%mesh%shape%ngi) :: quad_val

    type(element_type), pointer :: shape

    shape=>ele_shape(field,ele_number)

    quad_val=matmul(ele_val(field, ele_number), shape%n)

  end function ele_val_at_quad_vector

  function ele_val_at_quad_vector_dim(field, ele_number, dim) result (quad_val)
    ! Return the values of field at the quadrature points of ele_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%mesh%shape%ngi) :: quad_val
    integer, intent(in):: dim

    type(element_type), pointer :: shape

    shape=>ele_shape(field,ele_number)

    quad_val=matmul(ele_val(field, dim, ele_number), shape%n)

  end function ele_val_at_quad_vector_dim

  function ele_val_at_quad_tensor(field, ele_number) result (quad_val)
    ! Return the values of field at the quadrature points of ele_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(field%dim(1), field%dim(2), field%mesh%shape%ngi) :: quad_val

    type(element_type), pointer :: shape

    shape=>ele_shape(field,ele_number)

    quad_val=tensormul(ele_val(field, ele_number), shape%n)

  end function ele_val_at_quad_tensor

  function ele_val_at_quad_tensor_dim_dim(field, i, j, ele_number) result (quad_val)
    ! Return the values of field at the quadrature points of ele_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number
    integer, intent(in) :: i, j
    real, dimension(field%mesh%shape%ngi) :: quad_val

    type(element_type), pointer :: shape

    shape=>ele_shape(field,ele_number)

    quad_val=matmul(ele_val(field, i, j, ele_number), shape%n)
  end function ele_val_at_quad_tensor_dim_dim

  function ele_val_at_shape_quad_scalar(field, ele_number, shape) result (quad_val)
    ! Return the values of field at the quadrature points of shape in ele_number.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number
    type(element_type), intent(in) :: shape
    real, dimension(shape%ngi) :: quad_val

    type(element_type), pointer :: meshshape

    meshshape=>ele_shape(field, ele_number)
    assert(meshshape%ndof==shape%ndof)

    quad_val=matmul(ele_val(field, ele_number), shape%n)

  end function ele_val_at_shape_quad_scalar

  function ele_val_at_shape_quad_vector(field, ele_number, shape) result (quad_val)
    ! Return the values of field at the quadrature points of shape in ele_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    type(element_type), intent(in) :: shape
    real, dimension(field%dim, shape%ngi) :: quad_val

    type(element_type), pointer :: meshshape

    meshshape=>ele_shape(field, ele_number)
    assert(meshshape%ndof==shape%ndof)

    quad_val=matmul(ele_val(field, ele_number), shape%n)

  end function ele_val_at_shape_quad_vector

  function ele_val_at_shape_quad_tensor(field, ele_number, shape) result (quad_val)
    ! Return the values of field at the quadrature points of shape in ele_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number
    type(element_type), intent(in) :: shape
    real, dimension(field%dim(1), field%dim(2), shape%ngi) :: quad_val

    type(element_type), pointer :: meshshape

    meshshape=>ele_shape(field, ele_number)
    assert(meshshape%ndof==shape%ndof)

    quad_val=tensormul(ele_val(field, ele_number), shape%n)

  end function ele_val_at_shape_quad_tensor

  function face_val_at_quad_scalar(field, face_number) result (quad_val)
    ! Return the values of field at the quadrature points of face_number.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension( face_ngi(field, face_number)) :: quad_val

    type(element_type), pointer :: shape

    shape=>face_shape(field, face_number)

    quad_val=matmul(face_val(field, face_number), shape%n)

  end function face_val_at_quad_scalar

  function face_val_at_quad_vector(field, face_number) result (quad_val)
    ! Return the values of field at the quadrature points of face_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(field%dim, face_ngi(field, face_number)) :: quad_val

    type(element_type), pointer :: shape

    shape=>face_shape(field,face_number)

    quad_val=matmul(face_val(field, face_number), shape%n)

  end function face_val_at_quad_vector

  function face_val_at_quad_vector_dim(field, face_number, dim) result (quad_val)
    ! Return the values of field at the quadrature points of face_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(face_ngi(field, face_number)) :: quad_val
    integer, intent(in) :: dim

    type(element_type), pointer :: shape

    shape=>face_shape(field,face_number)

    quad_val=matmul(face_val(field, dim, face_number), shape%n)

  end function face_val_at_quad_vector_dim

  function face_val_at_quad_tensor(field, face_number) result (quad_val)
    ! Return the values of field at the quadrature points of face_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(field%dim(1), field%dim(2), face_ngi(field, face_number)) :: quad_val

    type(element_type), pointer :: shape

    shape=>face_shape(field,face_number)

    quad_val=tensormul(face_val(field, face_number), shape%n)

  end function face_val_at_quad_tensor

  function face_val_at_quad_tensor_dim_dim(field, face_number, dim1, dim2) result (quad_val)
    ! Return the values of field at the quadrature points of face_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number
    real, dimension(face_ngi(field, face_number)) :: quad_val
    integer, intent(in) :: dim1, dim2

    type(element_type), pointer :: shape

    shape=>face_shape(field,face_number)

    quad_val=matmul(face_val(field, dim1, dim2, face_number), shape%n)

  end function face_val_at_quad_tensor_dim_dim

  function face_val_at_shape_quad_scalar(field, face_number, shape) result (quad_val)
    ! Return the values of field at the quadrature points of shape in face_number.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: face_number
    type(element_type), intent(in) :: shape
    real, dimension(shape%ngi) :: quad_val

    type(element_type), pointer :: meshshape

    meshshape=>face_shape(field, face_number)
    assert(meshshape%ndof==shape%ndof)

    quad_val=matmul(face_val(field, face_number), shape%n)

  end function face_val_at_shape_quad_scalar

  function face_val_at_shape_quad_vector(field, face_number, shape) result (quad_val)
    ! Return the values of field at the quadrature points of shape in face_number.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: face_number
    type(element_type), intent(in) :: shape
    real, dimension(field%dim, shape%ngi) :: quad_val

    type(element_type), pointer :: meshshape

    meshshape=>face_shape(field,face_number)
    assert(meshshape%ndof==shape%ndof)

    quad_val=matmul(face_val(field, face_number), shape%n)

  end function face_val_at_shape_quad_vector

  function face_val_at_shape_quad_tensor(field, face_number, shape) result (quad_val)
    ! Return the values of field at the quadrature points of face_number.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: face_number
    type(element_type), intent(in) :: shape
    real, dimension(mesh_dim(field), mesh_dim(field), shape%ngi) :: quad_val

    type(element_type), pointer :: meshshape

    meshshape=>face_shape(field, face_number)
    assert(meshshape%ndof==shape%ndof)

    quad_val=tensormul(face_val(field, face_number), shape%n)

  end function face_val_at_shape_quad_tensor

  function ele_grad_at_quad_scalar(field, ele_number, dn) result (quad_grad)
    ! Return the grad of field at the quadrature points of
    ! ele_number. dn is the transformed element gradient.
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field,ele_number), &
         &          ele_ngi(field,ele_number),&
         &          mesh_dim(field)), intent(in) :: dn
    real, dimension(mesh_dim(field), field%mesh%shape%ngi) :: quad_grad

    integer :: i

    do i=1, mesh_dim(field)
       quad_grad(i,:)=matmul(ele_val(field, ele_number),dn(:,:,i))
    end do

  end function ele_grad_at_quad_scalar

  function ele_grad_at_quad_vector(field, ele_number, dn) result (quad_grad)
    ! Return the grad of field at the quadrature points of
    ! ele_number. dn is the transformed element gradient.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field,ele_number), &
         &          ele_ngi(field,ele_number),&
         &          mesh_dim(field)), intent(in) :: dn
    real, dimension(mesh_dim(field), field%dim, &
         &          field%mesh%shape%ngi)        :: quad_grad

    integer :: i, j

    do i=1, mesh_dim(field)
       do j=1, mesh_dim(field)
          quad_grad(i,j,:)=matmul(ele_val(field, j, ele_number),dn(:,:,i))
       end do
    end do

  end function ele_grad_at_quad_vector

  function ele_div_at_quad_tensor(field, ele_number, dn) result (quad_div)
    ! Return the grad of field (dtensor_{ij}/dx_{j}) at the quadrature points of
    ! ele_number. dn is the transformed element gradient.
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field,ele_number), &
         &          ele_ngi(field,ele_number),&
         &          mesh_dim(field)), intent(in) :: dn
    real, dimension(mesh_dim(field), field%mesh%shape%ngi) :: quad_div

    integer :: i, j
    real, dimension(field%dim(1), field%dim(2), ele_loc(field, ele_number)) :: tensor

    tensor = ele_val(field, ele_number)
    quad_div = 0.0

    do i=1,mesh_dim(field)
      do j=1,mesh_dim(field)
        quad_div(i,:) = quad_div(i,:) + matmul(tensor(i,j,:), dn(:,:,j))
      end do
    end do

  end function ele_div_at_quad_tensor

  function ele_div_at_quad(field, ele_number, dn) result (quad_div)
    ! Return the divergence of field at the quadrature points of
    ! ele_number. dn is the transformed element gradient.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field,ele_number), &
         &          ele_ngi(field,ele_number),&
         &          field%dim),                   intent(in) :: dn
    real, dimension(field%mesh%shape%ngi) :: quad_div

    integer :: i

    quad_div=0.0

    do i=1,field%dim
       quad_div=quad_div+matmul(ele_val(field, i, ele_number),dn(:,:,i))
    end do

  end function ele_div_at_quad

  function ele_2d_curl_at_quad(field, ele_number, dn) result(quad_curl)
    ! Return the 2D curl of field at the quadrature points of ele_number. dn is
    ! the transformed element gradient.

    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field, ele_number), ele_ngi(field, ele_number), field%dim), intent(in) :: dn

    real, dimension(ele_ngi(field, ele_number)) :: quad_curl

    assert(field%dim == 2)

    quad_curl = matmul(ele_val(field, 2, ele_number), dn(:, :, 1)) - matmul(ele_val(field, 1, ele_number), dn(:, :, 2))

  end function ele_2d_curl_at_quad

  function ele_curl_at_quad(field, ele_number, dn) result (quad_curl)
    ! Return the 3D curl of field at the quadrature points of
    ! ele_number. dn is the transformed element gradient.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field,ele_number), &
                  & ele_ngi(field,ele_number), &
                  & field%dim), intent(in) :: dn
    real, dimension(3, ele_ngi(field, ele_number)) :: quad_curl

    integer :: i

    assert(field%dim == 3)

    do i = 1, field%dim
       quad_curl(i, :) = &
         & matmul(ele_val(field, rot3(i, 2), ele_number), &
         & dn(:, :, rot3(i, 1))) - &
         & matmul(ele_val(field, rot3(i, 1), ele_number), &
         & dn(:, :, rot3(i, 2)))
    end do

  contains

    function rot3(i, di)
      !! Rotate i di places in (1,2,3)

      integer, intent(in) :: i, di
      integer :: rot3

      rot3 = mod(i + di - 1, 3) + 1

    end function rot3

  end function ele_curl_at_quad

  function ele_jacobian_at_quad(field, ele_number, dn) result (quad_J)
    ! Return the Jacobian matrix of field at the quadrature points of
    ! ele_number. dn is the transformed element gradient.
    type(vector_field),intent(in) :: field
    integer, intent(in) :: ele_number
    real, dimension(ele_loc(field,ele_number), &
         &          ele_ngi(field,ele_number),&
         &          field%dim),                   intent(in) :: dn
    real, dimension(field%dim, field%dim, field%mesh%shape%ngi) :: quad_J

    integer :: i, j

    quad_J=0.0

    do i=1, field%dim
       do j=1, field%dim
          quad_J(i,j,:)=matmul(ele_val(field, ele_number, i),dn(:,:,j))
       end do
    end do

  end function ele_jacobian_at_quad

  function node_val_scalar(field, node_number) result (val)
    ! Return the value of field at node node_number
    type(scalar_field),intent(in) :: field
    integer, intent(in) :: node_number
    real :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val=field%val(node_number)
    case(FIELD_TYPE_CONSTANT)
      val=field%val(1)
    case(FIELD_TYPE_PYTHON)
       call val_python
    end select

  contains

    subroutine val_python
      !!< This subroutine isolates the following stack variables from the
      !!< main code path.
      real, dimension(field%py_dim, 1) :: pos
      real, dimension(field%py_dim, field%py_positions_shape%ndof) :: tmp_pos
      real, dimension(field%py_dim, field%mesh%shape%ndof) :: tmp_pos_2
      real, dimension(1) :: tmp_val
      integer :: i, ele, loccoord

      assert(associated(field%mesh%adj_lists))
      if (.not. associated(field%mesh%adj_lists%nelist)) then
        ewrite(-1,*) "nelist not initialised. I could allocate it myself,"
        ewrite(-1,*) "but you're probably calling this a lot."
        ewrite(-1,*) " call add_nelist(mesh) first"
        FLAbort("Call add_nelist(mesh) before calling val_python.")
      end if

      if (.not. field%py_positions_same_mesh) then
        ele = field%mesh%adj_lists%nelist%colm(field%mesh%adj_lists%nelist%findrm(node_number))
        loccoord = local_coords(field%mesh, ele, node_number)
        tmp_pos = ele_val(field%py_positions, ele)
        do i=1,field%py_dim
          tmp_pos_2(i, :) = matmul(field%py_locweight, tmp_pos(i, :))
        end do
        pos(:, 1) = tmp_pos_2(:, loccoord)
      else
        do i=1,field%py_dim
          pos(i, 1) = field%py_positions%val(i,node_number)
        end do
      end if

      call set_from_python_function(tmp_val, trim(field%py_func), pos, &
        time=current_time)
      val = tmp_val(1)

    end subroutine val_python

  end function node_val_scalar

  pure function node_val_vector(field, node_number) result (val)
    ! Return the value of field at node node_number
    type(vector_field),intent(in) :: field
    integer, intent(in) :: node_number
    real, dimension(field%dim) :: val

    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i=1,field%dim
         val(i)=field%val(i,node_number)
      end do
    case(FIELD_TYPE_CONSTANT)
      do i=1,field%dim
        val(i)=field%val(i,1)
      end do
    end select

  end function node_val_vector

  pure function node_val_tensor(field, node_number) result (val)
    ! Return the value of field at node node_number
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: node_number
    real, dimension(field%dim(1), field%dim(2)) :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val=field%val(:,:,node_number)
    case(FIELD_TYPE_CONSTANT)
      val=field%val(:,:,1)
    end select

  end function node_val_tensor

  pure function node_val_tensor_dim_dim(field, dim1, dim2, node_number) result (val)
    ! Return the value of field at node node_number
    type(tensor_field),intent(in) :: field
    integer, intent(in) :: node_number
    integer, intent(in) :: dim1, dim2
    real :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val=field%val(dim1,dim2,node_number)
    case(FIELD_TYPE_CONSTANT)
      val=field%val(dim1,dim2,1)
    end select

  end function node_val_tensor_dim_dim

  pure function node_val_tensor_dim_dim_v(field, dim1, dim2, node_numbers) result (val)
    ! Return the value of field at nodes node_numbers
    type(tensor_field),intent(in) :: field
    integer, dimension(:), intent(in) :: node_numbers
    integer, intent(in) :: dim1, dim2
    real, dimension(size(node_numbers)) :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val=field%val(dim1,dim2,node_numbers)
    case(FIELD_TYPE_CONSTANT)
      val=field%val(dim1,dim2,1)
    end select

  end function node_val_tensor_dim_dim_v

  pure function node_val_scalar_v(field, node_numbers) result (val)
    ! Return the value of field at node node_numbers
    type(scalar_field),intent(in) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(size(node_numbers)) :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val=field%val(node_numbers)
    case(FIELD_TYPE_CONSTANT)
      val=field%val(1)
    end select

  end function node_val_scalar_v

  pure function node_val_vector_v(field, node_numbers) result (val)
    ! Return the value of field at node node_numbers
    type(vector_field),intent(in) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(field%dim, size(node_numbers)) :: val

    integer :: i

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      do i=1,field%dim
         val(i,:)=field%val(i,node_numbers)
      end do
    case(FIELD_TYPE_CONSTANT)
      do i=1,field%dim
         val(i,:)=field%val(i,1)
      end do
    end select

  end function node_val_vector_v

  pure function node_val_vector_dim(field, dim, node_number) &
       result (val)
    ! Return the value of field at node node_numbers
    type(vector_field),intent(in) :: field
    integer, intent(in) :: node_number
    integer, intent(in) :: dim
    real :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val = field%val(dim,node_number)
    case(FIELD_TYPE_CONSTANT)
      val = field%val(dim,1)
    end select

  end function node_val_vector_dim

  pure function node_val_vector_dim_v(field, dim, node_numbers) &
       result (val)
    ! Return the value of field at node node_numbers
    type(vector_field),intent(in) :: field
    integer, dimension(:), intent(in) :: node_numbers
    integer, intent(in) :: dim
    real, dimension(size(node_numbers)) :: val

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val(:)=field%val(dim,node_numbers)
    case(FIELD_TYPE_CONSTANT)
      val=field%val(dim,1)
    end select

  end function node_val_vector_dim_v

  pure function node_val_tensor_v(field, node_numbers) result (val)
    ! Return the value of field at node node_numbers
    type(tensor_field),intent(in) :: field
    integer, dimension(:), intent(in) :: node_numbers
    real, dimension(field%dim(1), field%dim(2), size(node_numbers)) :: val
    integer :: i


    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val=field%val(:,:,node_numbers)
    case(FIELD_TYPE_CONSTANT)
      do i=1,size(node_numbers)
        val(:, :, i)=field%val(:, :, 1)
      end do
    end select

  end function node_val_tensor_v

  function continuity_mesh(mesh) result (continuity)
    ! Return the degree of continuity of mesh.
    integer :: continuity
    type(mesh_type), intent(in) :: mesh

    continuity=mesh%continuity

  end function continuity_mesh

  function continuity_scalar(field) result (continuity)
    ! Return the degree of continuity of mesh.
    integer :: continuity
    type(scalar_field), intent(in) :: field

    continuity=field%mesh%continuity

  end function continuity_scalar

  function continuity_vector(field) result (continuity)
    ! Return the degree of continuity of mesh.
    integer :: continuity
    type(vector_field), intent(in) :: field

    continuity=field%mesh%continuity

  end function continuity_vector

  function continuity_tensor(field) result (continuity)
    ! Return the degree of continuity of mesh.
    integer :: continuity
    type(tensor_field), intent(in) :: field

    continuity=field%mesh%continuity

  end function continuity_tensor

  function element_degree_mesh(mesh, ele_number) result (element_degree)
    ! Return the polynomial degree of the shape function for this element of the mesh.
    integer :: element_degree
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele_number

    element_degree=mesh%shape%degree

  end function element_degree_mesh

  function element_degree_scalar(field, ele_number) result (element_degree)
    ! Return the polynomial degree of the shape function for this element of the field.
    integer :: element_degree
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele_number

    element_degree=field%mesh%shape%degree

  end function element_degree_scalar

  function element_degree_vector(field, ele_number) result (element_degree)
    ! Return the polynomial degree of the shape function for this element of the field.
    integer :: element_degree
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele_number

    element_degree=field%mesh%shape%degree

  end function element_degree_vector

  function element_degree_tensor(field, ele_number) result (element_degree)
    ! Return the polynomial degree of the shape function for this element of the field.
    integer :: element_degree
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele_number

    element_degree=field%mesh%shape%degree

  end function element_degree_tensor

  function has_faces_mesh(mesh) result (has_faces)
    ! Check whether the faces component of mesh has been calculated.
    logical :: has_faces
    type(mesh_type), intent(in) :: mesh

    has_faces=associated(mesh%faces)

  end function has_faces_mesh

  function extract_scalar_field_from_vector_field(vfield, dim, stat) result(sfield)
  !!< This function gives you a way to treat a vector field
  !!< as a union of scalar fields.

    type(vector_field), intent(in), target :: vfield
    integer, intent(in) :: dim
    type(scalar_field) :: sfield
    integer, intent(out), optional :: stat

    if (present(stat)) then
      stat = 0
      if (dim > vfield%dim) then
        stat = 1
        return
      end if
    end if
    assert(dim .le. vfield%dim)

    ! Note that the reference count is not incremented as this is a
    ! borrowed field reference.
    sfield%mesh = vfield%mesh
    sfield%val  => vfield%val(dim,:)
    sfield%val_stride = vfield%dim
    sfield%option_path = vfield%option_path
    sfield%field_type = vfield%field_type
    write(sfield%name, '(a, i0)') trim(vfield%name) // "%", dim

    ! FIXME: make these the same as the vector field
    sfield%py_dim = mesh_dim(vfield%mesh)
    sfield%py_positions_shape => vfield%mesh%shape

    sfield%refcount => vfield%refcount

  end function extract_scalar_field_from_vector_field

  function extract_scalar_field_from_tensor_field(tfield, dim1, dim2, stat) result(sfield)
  !!< This function gives you a way to treat a tensor field
  !!< as a union of scalar fields.

    type(tensor_field), intent(in) :: tfield
    integer, intent(in) :: dim1, dim2
    type(scalar_field) :: sfield
    integer, intent(out), optional :: stat

    if (present(stat)) then
      stat = 0
      if (dim1 > tfield%dim(1) .or. dim2 > tfield%dim(2)) then
        stat = 1
        return
      end if
    end if
    assert(dim1 .le. tfield%dim(1))
    assert(dim2 .le. tfield%dim(2))

    ! Note that the reference count is not incremented as this is a
    ! borrowed field reference.
    sfield%mesh = tfield%mesh
    sfield%val  => tfield%val(dim1, dim2, :)
    sfield%val_stride = tfield%dim(1) * tfield%dim(2)
    sfield%field_type = tfield%field_type
    write(sfield%name, '(a, 2i0)') trim(tfield%name) // "%", (dim1-1) * tfield%dim + dim2

    sfield%refcount => tfield%refcount
  end function extract_scalar_field_from_tensor_field

  subroutine field2file_scalar(filename, field)
    !!< Write the field values to filename.
    character(len=*), intent(in) :: filename
    type(scalar_field), intent(in) :: field

    integer :: unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    write(unit, "(g22.8e4)") field%val

    close(unit)

  end subroutine field2file_scalar

  subroutine field2file_vector(filename, field)
    !!< Write the field values to filename.
    character(len=*), intent(in) :: filename
    type(vector_field), intent(in) :: field

    integer :: unit, d

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    do d=1,field%dim
       write(unit, "(g22.8e4)") field%val(d,:)
    end do

    close(unit)

  end subroutine field2file_vector

  pure function halo_count_mesh(mesh) result(count)
    type(mesh_type), intent(in) :: mesh

    integer :: count

    if(.not. associated(mesh%halos)) then
      count = 0
    else
      count = size(mesh%halos)
    end if

  end function halo_count_mesh

  pure function halo_count_scalar(s_field) result(count)
    type(scalar_field), intent(in) :: s_field

    integer :: count

    count = halo_count(s_field%mesh)

  end function halo_count_scalar

  pure function halo_count_vector(v_field) result(count)
    type(vector_field), intent(in) :: v_field

    integer :: count

    count = halo_count(v_field%mesh)

  end function halo_count_vector

  pure function halo_count_tensor(t_field) result(count)
    type(tensor_field), intent(in) :: t_field

    integer :: count

    count = halo_count(t_field%mesh)

  end function halo_count_tensor

  pure function element_halo_count_mesh(mesh) result(count)
    type(mesh_type), intent(in) :: mesh

    integer :: count

    if(.not. associated(mesh%element_halos)) then
      count = 0
    else
      count = size(mesh%element_halos)
    end if

  end function element_halo_count_mesh

  pure function element_halo_count_scalar(s_field) result(count)
    type(scalar_field), intent(in) :: s_field

    integer :: count

    count = element_halo_count(s_field%mesh)

  end function element_halo_count_scalar

  pure function element_halo_count_vector(v_field) result(count)
    type(vector_field), intent(in) :: v_field

    integer :: count

    count = element_halo_count(v_field%mesh)

  end function element_halo_count_vector

  pure function element_halo_count_tensor(t_field) result(count)
    type(tensor_field), intent(in) :: t_field

    integer :: count

    count = element_halo_count(t_field%mesh)

  end function element_halo_count_tensor

  function face_vertices_shape(shape) result(vert)
    type(element_type), intent(in) :: shape
    integer :: vert

    select case(shape%type)
    case(ELEMENT_LAGRANGIAN, ELEMENT_BUBBLE, ELEMENT_TRACE, ELEMENT_DISCONTINUOUS_LAGRANGIAN)
      select case(cell_family(shape))
      case (FAMILY_SIMPLEX)
        vert = shape%dim
      case (FAMILY_CUBE)
        vert = 2**(shape%dim-1)
      case default
        FLAbort("Unknown element family.")
      end select
    case default
      FLAbort("Unknown element type.")
    end select

  end function face_vertices_shape

  function local_coords_interpolation(position_field, ele, position) result(local_coords)
    !!< Given a position field, this returns the local coordinates of
    !!< position with respect to element "ele".
    !!<
    !!< This assumes the position field is linear. For higher order
    !!< only the coordinates of the vertices are considered
    type(vector_field), intent(in) :: position_field
    integer, intent(in) :: ele
    real, dimension(:), intent(in) :: position
    real, dimension(size(position) + 1) :: local_coords
    real, dimension(mesh_dim(position_field) + 1, size(position) + 1) :: matrix
    real, dimension(mesh_dim(position_field), size(position) + 1) :: tmp_matrix
    integer, dimension(position_field%mesh%shape%cell%entity_counts(0)):: vertices
    integer, dimension(:), pointer:: nodes
    integer :: dim

    dim = size(position)

    assert(dim == mesh_dim(position_field))
    assert(cell_family(position_field%mesh%shape)==FAMILY_SIMPLEX)
    assert(position_field%mesh%shape%type==ELEMENT_LAGRANGIAN)

    local_coords(1:dim) = position
    local_coords(dim+1) = 1.0

    if (position_field%mesh%shape%degree==1) then
      tmp_matrix = ele_val(position_field, ele)
    else
      nodes => ele_nodes(position_field, ele)
      vertices=local_vertices(position_field%mesh%shape%numbering)
      tmp_matrix = node_val(position_field, nodes(vertices) )
    end if

    matrix(1:dim, :) = tmp_matrix
    matrix(dim+1, :) = 1.0

    call solve(matrix, local_coords)

  end function local_coords_interpolation

  function local_coords_interpolation_all(position_field, ele, position) result(local_coords)
    !!< Given a position field, this returns the local coordinates of a number
    !!< of positions which respect to element "ele".
    !!<
    !!< This assumes the positions field is linear. For higher order
    !!< only the coordinates of the vertices are considered
    type(vector_field), intent(in) :: position_field
    integer, intent(in) :: ele
    real, dimension(:, :), intent(in) :: position

    real, dimension(size(position,1)+1, size(position, 2)) :: local_coords

    real, dimension(size(position,1)+1, size(position,1)+1) :: inversion_matrix

    assert(size(position, 1) == position_field%dim)
    assert(cell_family(position_field%mesh%shape)==FAMILY_SIMPLEX)
    assert(position_field%mesh%shape%type==ELEMENT_LAGRANGIAN)

    call local_coords_matrix(position_field, ele, inversion_matrix)
    local_coords(1:position_field%dim, :) = position
    local_coords(position_field%dim + 1, :) = 1.0
    local_coords = matmul(inversion_matrix, local_coords)

  end function local_coords_interpolation_all

  subroutine local_coords_matrix(positions, ele, mat)
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele
    real, dimension(:,:), intent(out) :: mat

    integer, dimension(positions%mesh%shape%cell%entity_counts(0)):: vertices
    integer, dimension(:), pointer:: nodes

    assert( size(mat,1)==mesh_dim(positions)+1 )
    assert( size(mat,1)==size(mat,2) )

    if (positions%mesh%shape%degree==1) then
      mat(1:positions%dim, :) = ele_val(positions, ele)
    else
      nodes => ele_nodes(positions, ele)
      vertices=local_vertices(positions%mesh%shape%numbering)
      mat(1:positions%dim, :) = node_val(positions, nodes(vertices) )
    end if
    mat(positions%dim + 1, :) = 1.0

    call invert(mat)

  end subroutine local_coords_matrix

  function local_coords_scalar(field, ele, node, stat) result(local_coord)
    !!< returns the local node number within a given element ele of the global
    !!< node number node
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: ele, node
    integer, intent(inout), optional :: stat
    integer :: local_coord

    local_coord = local_coords(field%mesh, ele, node, stat)

  end function local_coords_scalar

  function local_coords_vector(field, ele, node, stat) result(local_coord)
    !!< returns the local node number within a given element ele of the global
    !!< node number node
    type(vector_field), intent(in) :: field
    integer, intent(in) :: ele, node
    integer, intent(inout), optional :: stat
    integer :: local_coord

    local_coord = local_coords(field%mesh, ele, node, stat)

  end function local_coords_vector

  function local_coords_tensor(field, ele, node, stat) result(local_coord)
    !!< returns the local node number within a given element ele of the global
    !!< node number node
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: ele, node
    integer, intent(inout), optional :: stat
    integer :: local_coord

    local_coord = local_coords(field%mesh, ele, node, stat)

  end function local_coords_tensor

  function local_coords_mesh(mesh, ele, node, stat) result(local_coord)
    !!< returns the local node number within a given element ele of the global
    !!< node number node
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele, node
    integer, intent(inout), optional :: stat
    integer :: local_coord
    integer, dimension(:), pointer :: nodes
    integer :: i

    if(present(stat)) stat = 0

    local_coord = 0

    nodes => ele_nodes(mesh, ele)
    do i=1,size(nodes)
      if (nodes(i) == node) then
        local_coord = i
        return
      end if
    end do

    if(present(stat)) then
      stat = 1
    else
      FLAbort("Failed to find local coordinate.")
    end if

  end function local_coords_mesh

  function field_val_scalar(field) result(val)
    type(scalar_field), intent(in) :: field
    real, dimension(:), pointer :: val

    val => null()

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val => field%val
      return
    case(FIELD_TYPE_CONSTANT)
      FLAbort("Trying to pass around the value space of a constant field")
    case(FIELD_TYPE_PYTHON)
      FLAbort("Trying to pass around the value space of a pythonic field")
    end select
  end function field_val_scalar

  function field_val_vector(field, dim) result(val)
    type(vector_field), intent(in) :: field
    integer, intent(in) :: dim
    real, dimension(:), pointer :: val

    val => null()

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
      val => field%val(dim,:)
      return
    case(FIELD_TYPE_CONSTANT)
      FLAbort("Trying to pass around the value space of a constant field")
    case(FIELD_TYPE_PYTHON)
      FLAbort("Trying to pass around the value space of a pythonic field")
    end select
  end function field_val_vector

  function eval_field_scalar(ele, s_field, local_coord) result(val)
    !!< Evaluate the scalar field s_field at element local coordinate
    !!< local_coord of element ele.

    integer, intent(in) :: ele
    type(scalar_field), intent(in) :: s_field
    real, dimension(:), intent(in) :: local_coord

    real :: val

    integer :: i
    real, dimension(ele_loc(s_field, ele)) :: n
    type(element_type), pointer :: shape

    shape => ele_shape(s_field, ele)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(s_field, ele) == FAMILY_SIMPLEX).and.&
           (finite_element_type(s_field, ele) == ELEMENT_LAGRANGIAN)) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    val = dot_product(ele_val(s_field, ele), n)

  end function eval_field_scalar

  function eval_field_vector(ele, v_field, local_coord) result(val)
    !!< Evaluate the vector field v_field at element local coordinate
    !!< local_coord of element ele.

    integer, intent(in) :: ele
    type(vector_field), intent(in) :: v_field
    real, dimension(:), intent(in) :: local_coord

    real, dimension(v_field%dim) :: val

    integer :: i
    real, dimension(ele_loc(v_field, ele)) :: n
    type(element_type), pointer :: shape

    shape => ele_shape(v_field, ele)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(v_field, ele) == FAMILY_SIMPLEX).and.&
           (finite_element_type(v_field, ele) == ELEMENT_LAGRANGIAN)) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    do i = 1, size(val)
      val(i) = dot_product(ele_val(v_field, i, ele), n)
    end do

  end function eval_field_vector

  function eval_field_tensor(ele, t_field, local_coord) result(val)
    !!< Evaluate the tensor field t_field at element local coordinate
    !!< local_coord of element ele.

    integer, intent(in) :: ele
    type(tensor_field), intent(in) :: t_field
    real, dimension(:), intent(in) :: local_coord

    real, dimension(t_field%dim(1), t_field%dim(2)) :: val

    integer :: i, j
    real, dimension(ele_loc(t_field, ele)) :: n
    type(element_type), pointer :: shape

    shape => ele_shape(t_field, ele)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(t_field, ele) == FAMILY_SIMPLEX).and.&
           (finite_element_type(t_field, ele) == ELEMENT_LAGRANGIAN)) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    do i = 1, size(val, 1)
      do j = 1, size(val, 2)
        val(i, j) = dot_product(ele_val(t_field, i, j, ele), n)
      end do
    end do

  end function eval_field_tensor

  function face_eval_field_scalar(face, s_field, local_coord) result(val)
    !!< Evaluate the scalar field s_field at face local coordinate
    !!< local_coord of the facet face.

    integer, intent(in) :: face
    type(scalar_field), intent(in) :: s_field
    real, dimension(:), intent(in) :: local_coord

    real :: val

    integer :: i
    real, dimension(face_loc(s_field, face)) :: n
    type(element_type), pointer :: shape

    shape => face_shape(s_field, face)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(shape) == FAMILY_SIMPLEX).and.&
           ((finite_element_type(shape) == ELEMENT_LAGRANGIAN).or.(finite_element_type(shape) == ELEMENT_BUBBLE))) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    val = dot_product(face_val(s_field, face), n)

  end function face_eval_field_scalar

  function face_eval_field_vector(face, v_field, local_coord) result(val)
    !!< Evaluate the vector field v_field at face local coordinate
    !!< local_coord of facet face.

    integer, intent(in) :: face
    type(vector_field), intent(in) :: v_field
    real, dimension(:), intent(in) :: local_coord

    real, dimension(v_field%dim) :: val

    integer :: i
    real, dimension(face_loc(v_field, face)) :: n
    type(element_type), pointer :: shape

    shape => face_shape(v_field, face)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(shape) == FAMILY_SIMPLEX).and.&
           ((finite_element_type(shape) == ELEMENT_LAGRANGIAN).or.(finite_element_type(shape) == ELEMENT_BUBBLE))) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    do i = 1, size(val)
      val(i) = dot_product(face_val(v_field, i, face), n)
    end do

  end function face_eval_field_vector

  function face_eval_field_vector_dim(face, v_field, dim, local_coord) result(val)
    !!< Evaluate the vector field v_field at face local coordinate
    !!< local_coord of facet face.

    integer, intent(in) :: face
    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: dim
    real, dimension(:), intent(in) :: local_coord

    real :: val

    integer :: i
    real, dimension(face_loc(v_field, face)) :: n
    type(element_type), pointer :: shape

    shape => face_shape(v_field, face)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(shape) == FAMILY_SIMPLEX).and.&
           ((finite_element_type(shape) == ELEMENT_LAGRANGIAN).or.(finite_element_type(shape) == ELEMENT_BUBBLE))) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    val = dot_product(face_val(v_field, dim, face), n)

  end function face_eval_field_vector_dim

  function face_eval_field_tensor(face, t_field, local_coord) result(val)
    !!< Evaluate the tensor field t_field at face local coordinate
    !!< local_coord of facet face.

    integer, intent(in) :: face
    type(tensor_field), intent(in) :: t_field
    real, dimension(:), intent(in) :: local_coord

    real, dimension(t_field%dim(1), t_field%dim(2)) :: val

    integer :: i, j
    real, dimension(face_loc(t_field, face)) :: n
    type(element_type), pointer :: shape

    shape => face_shape(t_field, face)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(shape) == FAMILY_SIMPLEX).and.&
           ((finite_element_type(shape) == ELEMENT_LAGRANGIAN).or.(finite_element_type(shape) == ELEMENT_BUBBLE))) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    do i = 1, size(val, 1)
      do j = 1, size(val, 2)
        val(i, j) = dot_product(face_val(t_field, i, j, face), n)
      end do
    end do

  end function face_eval_field_tensor

  function face_eval_field_tensor_dim_dim(face, t_field, dim1, dim2, local_coord) result(val)
    !!< Evaluate the tensor field t_field at face local coordinate
    !!< local_coord of facet face.

    integer, intent(in) :: face
    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: dim1, dim2
    real, dimension(:), intent(in) :: local_coord

    real :: val

    integer :: i, j
    real, dimension(face_loc(t_field, face)) :: n
    type(element_type), pointer :: shape

    shape => face_shape(t_field, face)

    select case(shape%degree)
      case(0)
        n = 1.0
      case(1)
        if((cell_family(shape) == FAMILY_SIMPLEX).and.&
           ((finite_element_type(shape) == ELEMENT_LAGRANGIAN).or.(finite_element_type(shape) == ELEMENT_BUBBLE))) then
          n = local_coord
        else
          do i = 1, size(n)
            n(i) = eval_shape(shape, i, local_coord)
          end do
        end if
      case default
        do i = 1, size(n)
          n(i) = eval_shape(shape, i, local_coord)
        end do
    end select

    val = dot_product(face_val(t_field, dim1, dim2, face), n)

  end function face_eval_field_tensor_dim_dim

  subroutine getsndgln(mesh, sndgln)
  !! get legacy surface mesh ndglno that uses node numbering of the full mesh
  type(mesh_type), intent(in):: mesh
  integer, dimension(:), intent(out):: sndgln

    integer sele, snloc, stotel

    assert(associated(mesh%faces))

    stotel=surface_element_count(mesh)
    snloc=face_loc(mesh, 1)

    assert(size(sndgln)==stotel*snloc)

    do sele=1, stotel
      sndgln( (sele-1)*snloc+1:sele*snloc )=face_global_nodes(mesh, sele)
    end do

  end subroutine getsndgln

  ! ------------------------------------------------------------------------
  ! Point wise python evalutiation
  ! ------------------------------------------------------------------------
  ! these could go in a separate module

  subroutine set_values_from_python_scalar(values, func, x, y, z, time)
    !!< Given a list of positions and a time, evaluate the python function
    !!< specified in the string func at those points.
    real, dimension(:), intent(inout) :: values
    !! Func may contain any python at all but the following function must
    !! be defiled:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    real, dimension(size(values)), target :: x
    real, dimension(size(values)), optional, target :: y
    real, dimension(size(values)), optional, target :: z
    real :: time

    real, dimension(:), pointer :: lx, ly, lz
    real, dimension(0), target :: zero
    integer :: stat, dim

    lx=>x
    ly=>zero
    lz=>zero
    dim=1
    if (present(y)) then
       ly=>y
       dim=2
       if (present(z)) then
          lz=>z
          dim=3
       end if
    end if

    call set_scalar_field_from_python(func, len(func), dim,&
            & size(values), lx, ly, lz, time, values, stat)

    if (stat/=0) then
      ewrite(-1, *) "Python error, Python string was:"
      ewrite(-1, *) trim(func)
      FLAbort("Dying")
    end if

  end subroutine set_values_from_python_scalar

  subroutine set_values_from_python_scalar_pos(values, func, pos, time)
    !!< Given a list of positions and a time, evaluate the python function
    !!< specified in the string func at those points.
    real, dimension(:), intent(inout) :: values
    !! Func may contain any python at all but the following function must
    !! be defiled:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    real, dimension(:, :), intent(in), target :: pos
    real :: time

    real, dimension(:), pointer :: lx, ly, lz
    real, dimension(0), target :: zero
    integer :: stat, dim

    dim=size(pos, 1)
    select case(dim)
    case(1)
      lx=>pos(1, :)
      ly=>zero
      lz=>zero
    case(2)
      lx=>pos(1, :)
      ly=>pos(2, :)
      lz=>zero
    case(3)
      lx=>pos(1, :)
      ly=>pos(2, :)
      lz=>pos(3, :)
    end select

    call set_scalar_field_from_python(func, len(func), dim,&
            & size(values), lx, ly, lz, time, values, stat)

    if (stat/=0) then
      ewrite(-1, *) "Python error, Python string was:"
      ewrite(-1, *) trim(func)
      FLAbort("Dying")
    end if
  end subroutine set_values_from_python_scalar_pos

  subroutine set_values_from_python_vector(values, func, x, y, z, time)
    !!< Given a list of positions and a time, evaluate the python function
    !!< specified in the string func at those points.
    real, dimension(:,:), target, intent(inout) :: values
    !! Func may contain any python at all but the following function must
    !! be defiled:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    real, dimension(size(values,2)), target :: x
    real, dimension(size(values,2)), optional, target :: y
    real, dimension(size(values,2)), optional, target :: z
    real :: time

    real, dimension(:), pointer :: lx, ly, lz
    real, dimension(:), pointer :: lvx,lvy,lvz
    real, dimension(0), target :: zero
    integer :: stat, dim

    lx=>x
    ly=>zero
    lz=>zero
    dim=1
    if (present(y)) then
       ly=>y
       dim=2
       if (present(z)) then
          lz=>z
          dim=3
       end if
    end if

    lvx=>values(1,:)
    lvy=>zero
    lvz=>zero
    if(size(values,1)>1) then
       lvy=>values(2,:)
       if(size(values,1)>2) then
          lvz => values(3,:)
       end if
    end if
    call set_vector_field_from_python(func, len_trim(func), dim,&
            & size(values,2), lx, ly, lz, time,size(values,1), &
            lvx,lvy,lvz, stat)

    if (stat/=0) then
      ewrite(-1, *) "Python error, Python string was:"
      ewrite(-1, *) trim(func)
      FLAbort("Dying")
    end if

  end subroutine set_values_from_python_vector

  subroutine set_values_from_python_vector_pos(values, func, pos, time)
    !!< Given a list of positions and a time, evaluate the python function
    !!< specified in the string func at those points.
    real, dimension(:,:), intent(inout) :: values
    !! Func may contain any python at all but the following function must
    !! be defiled:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    real, dimension(:, :), intent(in), target :: pos
    real, intent(in) :: time

    integer :: dim

    dim=size(pos, 1)
    select case(dim)
    case(1)
      call set_values_from_python_vector(values, func, pos(1,:), time=time)
    case(2)
      call set_values_from_python_vector(values, func, pos(1,:), pos(2,:), time=time)
    case(3)
      call set_values_from_python_vector(values, func, pos(1,:), pos(2,:), pos(3,:), time=time)
    end select

  end subroutine set_values_from_python_vector_pos

  subroutine set_values_from_python_vector_field(values, func, vfield, time)
    !!< Given a list of positions and a time, evaluate the python function
    !!< specified in the string func at those points.
    real, dimension(:,:), intent(inout) :: values
    !! Func may contain any python at all but the following function must
    !! be defiled:
    !!  def val(X, t)
    !! where X is a tuple containing the position of a point and t is the
    !! time. The result must be a float.
    character(len=*), intent(in) :: func
    type(vector_field), intent(in) :: vfield
    real, intent(in) :: time

    integer :: dim

    dim=vfield%dim
    select case(dim)
    case(1)
      call set_values_from_python_vector(values, func, vfield%val(1,:), time=time)
    case(2)
      call set_values_from_python_vector(values, func, vfield%val(1,:), vfield%val(2,:), time=time)
    case(3)
      call set_values_from_python_vector(values, func, vfield%val(1,:), vfield%val(2,:), vfield%val(3,:), time=time)
    end select

  end subroutine set_values_from_python_vector_field

  ! ------------------------------------------------------------------------
  ! Geometric element volume routines. These really ought to go somewhere
  ! else but tend to cause dependency problems when they do.
  ! ------------------------------------------------------------------------

  function tetvol_new(positions, ele) result(t)
    real :: t
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele
    real, dimension(positions%dim, ele_loc(positions, ele)) :: pos

    pos = ele_val(positions, ele)
    t = tetvol_old(pos(1, :), pos(2, :), pos(3, :))

  end function tetvol_new

  real function tetvol_old( x, y, z )

    real x(4), y(4), z(4)
    real vol, x12, x13, x14, y12, y13, y14, z12, z13, z14
    !
    x12 = x(2) - x(1)
    x13 = x(3) - x(1)
    x14 = x(4) - x(1)
    y12 = y(2) - y(1)
    y13 = y(3) - y(1)
    y14 = y(4) - y(1)
    z12 = z(2) - z(1)
    z13 = z(3) - z(1)
    z14 = z(4) - z(1)
    !
    vol = x12*( y13*z14 - y14*z13 )  &
         + x13*( y14*z12 - y12*z14 ) &
         + x14*( y12*z13 - y13*z12 )
    !
    tetvol_old = vol/6
    !
    return
  end function tetvol_old

  function triarea(positions, ele) result(t)
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele
    real :: t
    real, dimension(positions%dim, positions%mesh%shape%ndof) :: pos
    real :: xA, xB, yA, yB, xC, yC

    pos = ele_val(positions, ele)
    if (positions%dim == 2) then
      xA = pos(1, 1); xB = pos(1, 2); xC = pos(1, 3)
      yA = pos(2, 1); yB = pos(2, 2); yC = pos(2, 3)
      t = abs((xB*yA-xA*yB)+(xC*yB-xB*yC)+(xA*yC-xC*yA))/2
    elseif (positions%dim == 3) then
      ! http://mathworld.wolfram.com/TriangleArea.html, (19)
      t = 0.5 * norm2(cross_product(pos(:, 2) - pos(:, 1), pos(:, 1) - pos(:, 3)))
    else
      FLAbort("Only 2 or 3 dimensions supported, sorry")
    end if
  end function triarea

  function tetvol_1d(positions, ele) result(t)
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele

    real :: t

    integer, dimension(:), pointer :: element_nodes => null()

    assert(positions%dim == 1)

    element_nodes => ele_nodes(positions, ele)

    assert(size(element_nodes) == 2)
    t = abs(node_val(positions, 1, element_nodes(2)) - node_val(positions, 1, element_nodes(1)))

  end function tetvol_1d

  function simplex_volume(positions, ele) result(t)
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele
    real :: t

    select case(mesh_dim(positions))
      case(3)
        t = tetvol_new(positions, ele)
      case(2)
        t = triarea(positions, ele)
      case(1)
        t = tetvol_1d(positions, ele)
    case default
      FLAbort("Invalid dimension")
    end select

  end function simplex_volume

  function face_opposite_mesh(mesh, face) result (opp_face)
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: face

    integer :: parent_ele, opp_ele, opp_face
    integer, dimension(:), pointer :: neighbours

    parent_ele = face_ele(mesh, face)
    neighbours => ele_neigh(mesh, parent_ele)
    opp_ele = neighbours(local_face_number(mesh, face))
    if (opp_ele > 0) then
      opp_face = ele_face(mesh, opp_ele, parent_ele)
    else
      opp_face = -1
    end if
  end function face_opposite_mesh

  function face_opposite_scalar(sfield, face) result (opp_face)
    type(scalar_field), intent(in) :: sfield
    integer, intent(in) :: face

    integer :: opp_face

    opp_face = face_opposite_mesh(sfield%mesh, face)
  end function face_opposite_scalar

  function face_opposite_vector(vfield, face) result (opp_face)
    type(vector_field), intent(in) :: vfield
    integer, intent(in) :: face

    integer :: opp_face

    opp_face = face_opposite_mesh(vfield%mesh, face)
  end function face_opposite_vector

  function face_opposite_tensor(tfield, face) result (opp_face)
    type(tensor_field), intent(in) :: tfield
    integer, intent(in) :: face

    integer :: opp_face

    opp_face = face_opposite_mesh(tfield%mesh, face)
  end function face_opposite_tensor

  function mesh_topology(mesh) result (topology)
    ! Return a pointer to the topology mesh for mesh.
    type(mesh_type), intent(in), target :: mesh
    type(mesh_type), pointer :: topology

    assert(associated(mesh%topology))
    assert(mesh%topology%shape%degree == 1)
    topology => mesh%topology
  end function mesh_topology

  subroutine refresh_topology_mesh(mesh)
    ! When new structures are added to meshes which are topologies,
    ! the topology mesh descriptor becomes invalid.  This updates it.
    type(mesh_type), intent(inout) :: mesh
    if (mesh%refcount%id == mesh%topology%refcount%id) then
       mesh%topology = mesh
    end if
  end subroutine refresh_topology_mesh

  subroutine refresh_topology_vector_field(field)
    ! When new structures are added to meshes which are topologies,
    ! the topology mesh descriptor becomes invalid.  This updates it.
    type(vector_field), intent(inout) :: field

    call refresh_topology(field%mesh)

  end subroutine refresh_topology_vector_field

  subroutine write_minmax_scalar(sfield, field_expression)
    ! the scalar field to print its min and max of
    type(scalar_field), intent(in):: sfield
    ! the actual field in the code
    character(len=*), intent(in):: field_expression

    ewrite(2,*) 'Min, max of '//trim(field_expression)//' "'// &
       trim(sfield%name)//'" = ',minval(sfield%val), maxval(sfield%val)

  end subroutine write_minmax_scalar

  subroutine write_minmax_vector(vfield, field_expression)
    ! the vector field to print its min and max of
    type(vector_field), intent(in):: vfield
    ! the actual field in the code
    character(len=*), intent(in):: field_expression

    integer:: i

    do i=1, vfield%dim
      ewrite(2,*) 'Min, max of '//trim(field_expression)//' "'// &
         trim(vfield%name)//'%'//int2str(i)//'" = ', &
         minval(vfield%val(i,:)), maxval(vfield%val(i,:))
    end do

  end subroutine write_minmax_vector

  subroutine write_minmax_tensor(tfield, field_expression)
    ! the tensor field to print its min and max of
    type(tensor_field), intent(in):: tfield
    ! the actual field in the code
    character(len=*), intent(in):: field_expression

    integer:: i, j

    do i=1, tfield%dim(1)
      do j=1, tfield%dim(2)
        ewrite(2,*) 'Min, max of '//trim(field_expression)//' "'// &
          trim(tfield%name)//'%'//int2str(i)//','//int2str(j)// &
          '" = ', minval(tfield%val(i,j,:)), maxval(tfield%val(i,j,:))
      end do
    end do

  end subroutine write_minmax_tensor

end module fields_base
