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
module fields_halos
!!< This module contains code that depends on both fields and halos
use quadrature
use elements
use sparse_tools
use fields
use parallel_fields
use state_module
use halos
use data_structures
use quicksort
implicit none

private

public:: make_mesh_unperiodic, verify_consistent_local_element_numbering,&
     & order_elements

contains

  function make_mesh_unperiodic(model, my_physical_boundary_ids, aliased_boundary_ids, periodic_mapping_python, name, all_periodic_bc_ids, aliased_to_new_node_number) &
       result (new_positions)
    !!< Produce a mesh based on an old mesh but with periodic boundary conditions
    type(vector_field) :: new_positions

    type(vector_field), intent(in) :: model
    integer, dimension(:), intent(in) :: my_physical_boundary_ids, aliased_boundary_ids
    character(len=*), intent(in) :: periodic_mapping_python
    character(len=*), intent(in):: name
    type(integer_set), intent(in) :: all_periodic_bc_ids ! all boundary ids from all periodic BCs

    type(integer_hash_table), intent(out) :: aliased_to_new_node_number
    type(mesh_type):: mesh
    real, dimension(:,:), allocatable:: aliased_positions, physical_positions
    integer:: mapped_node_count, aliased_node, physical_node
    integer:: i, j, ele, sid, key, output
    integer, dimension(node_count(model)) :: is_periodic

    ! build a map from aliased node number to physical node number
    ! thus also counting the number mapped nodes
    call allocate( aliased_to_new_node_number )
    mapped_node_count = 0
    do i = 1, surface_element_count(model)
      sid = surface_element_id(model, i)
      if (any(aliased_boundary_ids==sid)) then
        call copy_aliased_nodes_face(i)
      end if
    end do

    ! before we use it, we need to use the model halos
    ! to ensure that everyone agrees on what is a periodic node
    ! to be split and what isn't!
    if (isparallel()) then
      assert(associated(model%mesh%halos))
      is_periodic = 0
      do i=1,key_count(aliased_to_new_node_number)
        call fetch_pair(aliased_to_new_node_number, i, key, output)
        is_periodic(key) = 1
      end do
      call halo_update(model%mesh%halos(2), is_periodic)

      do i=1,node_count(model)
        if (is_periodic(i) == 1) then
          if (.not. has_key(aliased_to_new_node_number, i)) then
            ! we didn't know about this one and need to generate a new local node number for it
            mapped_node_count = mapped_node_count + 1
            call insert(aliased_to_new_node_number, i, node_count(model)+mapped_node_count)
          end if
        else if (is_periodic(i) == 0) then
          if (has_key(aliased_to_new_node_number, i)) then
             write(0,*) halo_universal_number(model%mesh%halos(2),i)
            FLAbort("I thought it was periodic, but the owner says otherwise ... ")
          end if
        end if
      end do

#ifdef DDEBUG
      is_periodic = 0
      do i=1,key_count(aliased_to_new_node_number)
        call fetch_pair(aliased_to_new_node_number, i, key, output)
        is_periodic(key) = 1
      end do
      assert(halo_verifies(model%mesh%halos(2), is_periodic))
#endif
    end if

    ! we now have info to allocate the new mesh
    call allocate(mesh, node_count(model)+mapped_node_count, element_count(model), &
      model%mesh%shape, name=name)
    mesh%ndglno=model%mesh%ndglno

    ! now for the new_positions, first copy all positions of the model (including aliased nodes)
    call allocate(new_positions, model%dim, mesh, name=trim(name)//"Coordinate")
    allocate( aliased_positions(1:model%dim, mapped_node_count), &
      physical_positions(1:model%dim, mapped_node_count ) )
    do j=1, model%dim
       new_positions%val(j,1:node_count(model))=model%val(j,:)
    end do

    ! copy aliased positions into an array
    do i=1, mapped_node_count
      call fetch_pair(aliased_to_new_node_number, i, aliased_node, physical_node)
      aliased_positions(:, i)=node_val(model, aliased_node)
    end do

    ! apply the python map
    call set_from_python_function(physical_positions, &
            periodic_mapping_python, aliased_positions, &
            time=0.0)

    ! copy the physical node positions
    do i=1, mapped_node_count
      call fetch_pair(aliased_to_new_node_number, i, aliased_node, physical_node)
      do j=1, model%dim
         new_positions%val(j,physical_node)=physical_positions(j,i)
      end do
    end do

    ! now fix the elements
    do i = 1, surface_element_count(model)
      sid = surface_element_id(model, i)
      if (any(sid == my_physical_boundary_ids)) then
        ele=face_ele(model, i)
        call make_mesh_unperiodic_fix_ele(mesh, model%mesh, &
           aliased_to_new_node_number, all_periodic_bc_ids, ele)
      end if
    end do

    call deallocate( mesh )
    deallocate( aliased_positions, physical_positions )

  contains

    subroutine copy_aliased_nodes_face(face)
      integer, intent(in):: face

      integer, dimension(face_loc(model, face)):: aliased_nodes
      integer:: j

      aliased_nodes = face_global_nodes(model, face)
      do j = 1, size(aliased_nodes)
        if (.not. has_key(aliased_to_new_node_number, aliased_nodes(j))) then
          mapped_node_count = mapped_node_count + 1
          call insert(aliased_to_new_node_number, aliased_nodes(j), node_count(model)+mapped_node_count)
        end if
      end do

    end subroutine copy_aliased_nodes_face

  end function make_mesh_unperiodic

  recursive subroutine make_mesh_unperiodic_fix_ele(mesh, model, &
    aliased_to_new_node_number, boundary_ids_set, ele)
    ! For an element on the physical side of a periodic boundary,
    ! change all nodes from aliased to physical. This is recursively
    ! called for all neighbouring elements. Neighbours are found using
    ! the element-element list of the model, where we don't cross any
    ! facets with a physical boundary id - thus staying on this side of
    ! the boundary. Also as soon as an element without any aliased nodes
    ! is encountered the recursion stops
    ! so that we don't propagate into the interior of the mesh and don't fix
    ! elements twice. This assumes elements with aliased nodes and elements
    ! with physical nodes are not directly adjacent.
    type(mesh_type), intent(inout):: mesh
    type(mesh_type), intent(in):: model
    type(integer_hash_table), intent(in):: aliased_to_new_node_number
    type(integer_set), intent(in):: boundary_ids_set
    integer, intent(in):: ele

    integer, dimension(:), pointer:: nodes, neigh, faces
    integer:: j, sid
    logical:: changed

    changed=.false. ! have we changed this element

    nodes => ele_nodes(mesh, ele)
    do j = 1, size(nodes)
      if (has_key(aliased_to_new_node_number, nodes(j))) then
        nodes(j)=fetch(aliased_to_new_node_number, nodes(j))
        changed=.true.
      end if
    end do

    ! no aliased nodes found, we can stop the recursion
    if (.not. changed) return

    ! recursively "fix" our neighbours
    neigh => ele_neigh(model, ele)
    faces => ele_faces(model, ele)
    do j=1, size(neigh)
      if (neigh(j)>0) then
        ! found a neighbour

        ! check if we're crossing a physical boundary
        if (faces(j)<=surface_element_count(model)) then
          sid = surface_element_id(model, faces(j))
          if (has_value(boundary_ids_set, sid)) cycle
        end if

        ! otherwise go fix it
        call make_mesh_unperiodic_fix_ele(mesh, model, &
           aliased_to_new_node_number, boundary_ids_set, neigh(j))
      end if
    end do

  end subroutine make_mesh_unperiodic_fix_ele

  function verify_consistent_local_element_numbering(mesh) result (pass)
    !!< Checks that the local element ordering is consistent between the owner
    !!< of the element and all other processes that see it.
    type(mesh_type), intent(in):: mesh
    logical :: pass

    integer, dimension(:), allocatable:: eleunn, eleunn2
    integer:: nloc

    if (.not. associated(mesh%element_halos)) then
      FLAbort("Element halos not allocated in verify_local_element_numbering")
    end if

    nloc=ele_loc(mesh,1)
    assert(nloc*element_count(mesh)==size(mesh%ndglno))

    allocate( eleunn(1:size(mesh%ndglno)), eleunn2(1:size(mesh%ndglno)) )
    eleunn = halo_universal_numbers(mesh%halos(2),mesh%ndglno)
    eleunn2 = eleunn
    call halo_update(mesh%element_halos(2), eleunn, block_size=nloc)

    pass = all(eleunn==eleunn2)

  end function verify_consistent_local_element_numbering

  subroutine order_elements(numbering, positions, meshes)
    !!< Given a universal numbering field, renumber the local elements of
    !!< its mesh into this order. This should only be called for linear meshes:
    !!< derive the other meshes from the linear one.
    !!<
    !!< In parallel, the nodes will be renumbered into a consistent
    !!< trailing receives order.
    !!<
    !!< If meshes is present, also reorder the local elements in each of
    !!< its meshes. Meshes should likewise be linear: the intended use case
    !!< is that numbering is on the periodic mesh while the meshes are the
    !!< non-periodic (or possibly periodic on fewer sides) alternatives.
    type(scalar_field), intent(inout), target :: numbering
    type(vector_field), intent(inout), optional :: positions

    type(mesh_type), dimension(:), intent(inout), target, optional :: meshes

    integer :: ele, m, i, face, ndof, facet_ndof, new_ele
    integer, dimension(:), pointer :: nodes, facet_row, neigh_row
    real, dimension(:), allocatable :: unn
    integer, dimension(:), allocatable :: perm, ndglno, element_owner,&
         & boundary_id, surface_facets
    integer, dimension(2) :: tmpent
    type(cell_type), pointer :: cell
    type(mesh_type), pointer :: mesh
    type(mesh_type) :: p0_mesh
    type(mesh_type), target :: new_mesh
    integer, dimension(:), allocatable :: renumber

    assert(numbering%mesh%shape%degree==1)

    facet_ndof=face_loc(numbering,1)
    ndof=ele_loc(numbering,1)
    allocate(unn(ndof))
    allocate(perm(ndof))
    cell=>numbering%mesh%shape%cell

    ! First record the information required to rebuild faces.
    allocate(surface_facets(facet_ndof*surface_element_count(numbering%mesh)))
    do face=1,surface_element_count(numbering%mesh)
       surface_facets((face-1)*facet_ndof+1:face*facet_ndof)&
            &=face_global_nodes(numbering,face)
    end do
    if (numbering%mesh%faces%has_internal_boundaries) then
       allocate(element_owner(surface_element_count(numbering%mesh)))
       element_owner=numbering%mesh%faces&
            &%face_element_list(1:surface_element_count(numbering%mesh))
    end if
    allocate(boundary_id(surface_element_count(numbering%mesh)))
    boundary_id=numbering%mesh%faces%boundary_ids

    ! Next re-order the vertices in each element.
    do ele=1,element_count(numbering%mesh)
       nodes=>ele_nodes(numbering,ele)
       unn=ele_val(numbering, ele)

       ! Establish the node permutation.
       perm=vertex_permutation(unn, cell)

       nodes=nodes(perm)

       ! Re-order any additional meshes.
       if (present(meshes)) then
          do m=1, size(meshes)
             mesh=>meshes(m)
             nodes=>ele_nodes(mesh, ele)
             nodes=nodes(perm)
          end do
       end if
    end do

    ! Now re-order the elements.
    mesh=>numbering%mesh
    p0_mesh=piecewise_constant_mesh(mesh,"ReorderMesh", with_faces=.false.)
    allocate(ndglno(size(mesh%ndglno)))
    ndglno=mesh%ndglno
    do ele=1,element_count(mesh)
       new_ele=p0_mesh%ndglno(ele)

       mesh%ndglno((new_ele-1)*ndof+1:new_ele*ndof)&
            =ndglno((ele-1)*ndof+1:ele*ndof)
    end do

    if (present(meshes)) then
       do m=1, size(meshes)
          mesh=>meshes(m)
          ndglno=mesh%ndglno
          do ele=1,element_count(mesh)
             new_ele=p0_mesh%ndglno(ele)

             mesh%ndglno((new_ele-1)*ndof+1:new_ele*ndof)&
                  =ndglno((ele-1)*ndof+1:ele*ndof)
          end do
       end do
    end if
    if (allocated(element_owner)) then
       ! The element_owner list is contains old element numbers, it needs
       !  the new ones
       do i = 1, size(element_owner)
          element_owner(i) = p0_mesh%ndglno(element_owner(i))
       end do
    end if


    ! Now renumber the nodes into halo consistent order.
    mesh=>numbering%mesh
    new_mesh = make_mesh(model=mesh, shape=mesh%shape,&
         continuity=mesh%continuity, name=trim(mesh%name)//'new',&
         with_faces=.false.)
    new_mesh%element_classes = p0_mesh%node_classes
    call deallocate(p0_mesh)
    allocate(renumber(mesh%nodes))
    do i = 1, mesh%elements * ndof
       renumber(mesh%ndglno(i)) = new_mesh%ndglno(i)
    end do

    ! Renumber the surface facets according to new node numbers.
    surface_facets = renumber(surface_facets)
    ! Renumber the universal numbering field according to new node numbers.
    numbering%val(renumber) = numbering%val
    if (present(positions)) then
       ! Renumber the positions according to the new node numbers.
       positions%val(:,renumber) = positions%val
    end if

    deallocate(renumber)

    new_mesh%name = numbering%mesh%name
    new_mesh%option_path = numbering%mesh%option_path
    ! grab references that old_mesh had
    do i = 1, mesh%refcount%count - new_mesh%refcount%count
       call incref(new_mesh)
    end do
    ! drop references from old_mesh
    do i = 1, mesh%refcount%count-1
       call decref(mesh)
    end do
    call deallocate(numbering%mesh)
    ! Note that as a result of the following line, mesh now points to new_mesh.
    numbering%mesh = new_mesh
    new_mesh%topology=new_mesh
    ! The faces are now invalid so re-establish them.
    call deallocate_faces(mesh)
    ! Ordering is significant in the NElist and EElist.
    call remove_nelist(mesh)
    call add_nelist(mesh)
    call remove_eelist(mesh)
    call add_eelist(mesh)
    if (allocated(element_owner)) then
       call add_faces(mesh, &
            &               sndgln=surface_facets, &
            &               boundary_ids=boundary_id, &
            &               element_owner=element_owner)
    else
       call add_faces(mesh, &
            &               sndgln=surface_facets, &
            &               boundary_ids=boundary_id)
    end if
    ! Having trashed the faces, we need to re-establish the face uid.
    call add_surface_uid(mesh)

    ! Since we have renumbered the elements, the element halo is now
    !  invalid.
    if (associated(mesh%element_halos)) then
       if (present(meshes)) then
          do m=1, size(meshes)
             mesh=>meshes(m)
             call deallocate(mesh%element_halos)
          end do
       end if
       call deallocate(mesh%element_halos)
       call nullify(mesh%element_halos)
       call derive_element_halo_from_node_halo(mesh, &
            & ordering_scheme = HALO_ORDER_TRAILING_RECEIVES, create_caches = .true.)
       call refresh_topology(mesh)
    end if

    if (present(meshes)) then
       do m=1, size(meshes)
          mesh=>meshes(m)
          mesh%topology=mesh
          call deallocate_faces(mesh)
          call remove_nelist(mesh)
          call add_nelist(mesh)
          call remove_eelist(mesh)
          call add_eelist(mesh)
          call add_faces(mesh, model=numbering%mesh)
          if (associated(mesh%element_halos)) then
             do i=1,size(mesh%element_halos)
                mesh%element_halos(i)=numbering%mesh%element_halos(i)
                call incref(mesh%element_halos(i))
             end do
          end if
       end do
    end if

  contains

    function vertex_permutation(unn, cell) result (perm)
      ! Canonical numbers of the element vertices.
      real, dimension(:), intent(in) :: unn
      type(cell_type), intent(in) :: cell
      integer, dimension(size(unn)) :: perm

      integer, dimension(cell%dimension) :: adjacent_vertices, tmp
      real, dimension(cell%entity_counts(0), cell%dimension) :: vertex_coords
      real, dimension(cell%dimension, cell%dimension) :: A
      integer :: i, k, v0
      integer, dimension(2) :: edge

      if (cell%type==CELL_QUAD.or.cell%type==CELL_HEX) then
         ! The first vertex is the vertex of minimal canonical number.
         v0=minloc(unn,1)

         ! Vertices adjacent to the initial vertex.
         k=0
         do i=1,cell%entity_counts(1)
            edge=entity_vertices(cell,[1,i])
            if (any(edge==v0)) then
               k=k+1
               if (edge(1)==v0) then
                  adjacent_vertices(k)=edge(2)
               else
                  adjacent_vertices(k)=edge(1)
               end if
            end if
         end do

         ! Sort the adjacent vertices by canonical number.
         call qsort(unn(adjacent_vertices), tmp)
         adjacent_vertices=adjacent_vertices(tmp)

         ! Calculate change of coordinates matrix.
         do i=1,cell%dimension
            A(i,:)=cell%vertex_coords(adjacent_vertices(i),:)&
                 -cell%vertex_coords(v0,:)
         end do

         ! Calculate vertex coordinates in new coordinate system.
         do i=1,size(unn)
            vertex_coords(i,:)=matmul(A, cell%vertex_coords(i,:)&
                 -cell%vertex_coords(v0,:))
         end do

         ! Calculate permutation of vertices into new coordinate order.
         call sort(vertex_coords(:,cell%dimension:1:-1), perm)

      else
         call qsort(unn, perm)
      end if

    end function vertex_permutation

  end subroutine order_elements


end module fields_halos
