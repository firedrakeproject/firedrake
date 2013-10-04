!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
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
module global_numbering
  ! **********************************************************************
  ! Module to construct the global node numbering map for elements of a
  ! given degree.
  use adjacency_lists
  use elements
  use sparse_tools
  use fldebug
  use halo_data_types
  use halos_allocates
  use halos_base
  use halos_debug
  use halos_numbering
  use halos_ownership
  use parallel_tools
  use integer_set_module
  use linked_lists
  use mpi_interfaces
  use fields_base
  use memory_diagnostics
  use quicksort

  implicit none

  private

  public :: make_global_numbering_new, make_global_numbering_dg


contains

  subroutine make_global_numbering_new(mesh, dofs_per_column)
    ! Construct a new global numbering for mesh using the ordering from
    !  topology.
    type(mesh_type), intent(inout), target :: mesh
    integer, intent(in), dimension(0:), optional :: dofs_per_column

    type(csr_sparsity) :: facet_list, edge_list
    type(csr_matrix) :: facet_numbers, edge_numbers
    integer, dimension(0:mesh_dim(mesh)) :: entity_counts, dofs_per
    type(cell_type), pointer :: cell
    type(element_type), pointer :: element
    type(mesh_type), pointer :: topology
    ! The universal identifier. Required for parallel.
    type(scalar_field), pointer :: uid
    integer :: max_vertices, total_dofs, start, finish, d
    logical :: have_facets, have_halos, extrusion
    integer, dimension(0:mesh_dim(mesh)) :: dofs_per_stack

    type(integer_set), dimension(:,:), allocatable :: entity_send_targets
    integer, dimension(:), allocatable :: entity_owner
    integer, dimension(:), allocatable :: entity_receive_level

    ! Array to enable all the entities to be sorted together according to
    !  halo levels and universal numbers.
    integer, dimension(:,:), allocatable :: entity_sort_list
    ! Order in which entities should be visited
    integer, dimension(:), allocatable :: visit_order
    ! Order in which entities should be visited for halo ordering purposes.
    integer, dimension(:), allocatable :: halo_visit_order
    ! The entity order in visit order places core entities before sent
    !  entities before level 1 halo entities before level 2 halo entities.
    ! In contrast, the halo_visit_order places all entities owned by a
    !  given processor adjacently and the order is guaranteed consistent
    !  between processors. This is used to achieve consistent halo ordering.

    ! Dofs associated with each entity.
    integer, dimension(:), allocatable :: entity_dof_starts

    ! We also produce dof numberings using the halo visit order.
    integer, dimension(:), allocatable :: halo_entity_dof_starts

    ! Mapping from dofs to halo dofs.
    integer, dimension(:), allocatable :: dof_to_halo_dof

    element=>mesh%shape
    cell=>element%cell
    topology=>mesh%topology
    uid=>mesh%uid

    have_facets=associated(topology%faces)
    have_halos=associated(topology%halos)

    edge_list=make_edge_list(topology)
    if (have_facets) then
       facet_list=topology%faces%face_list%sparsity
    end if

    entity_counts=count_topological_entities(topology, edge_list, facet_list)
    max_vertices=size(entity_vertices(cell,[cell%dimension,1]))
    allocate(entity_sort_list(sum(entity_counts),0:max_vertices))
    entity_sort_list=-666
    allocate(visit_order(sum(entity_counts)))
    allocate(halo_visit_order(sum(entity_counts)))
    allocate(entity_dof_starts(sum(entity_counts)))
    allocate(halo_entity_dof_starts(sum(entity_counts)))

    ewrite(1, *)'Making GN for ', trim(mesh%name)
    ! Number the topological entities.
    call number_topology
    call calculate_dofs_per_entity
    ! Extract halo information for each topological entity.
    call create_topology_halos

    ! In the case of extruded meshes change the numbering
    if (present(dofs_per_column)) then
        extrusion=.True.
        dofs_per_stack=dofs_per_column
    else
        extrusion=.False.
        dofs_per_stack=dofs_per
    end if

    ! Save core/non-core/l1/l2 distinction in mesh node_classes
    if (have_halos) then

       mesh%node_classes = 0
       finish = 0
       do d=0, mesh_dim(mesh)
          start = finish + 1
          finish = finish + entity_counts(d)
          ! Count core nodes (primary sort key -2)
          mesh%node_classes(CORE_ENTITY) = mesh%node_classes(CORE_ENTITY) + &
               & count(entity_sort_list(start:finish, 0) == -2)*dofs_per_stack(d)
          ! Count non-core nodes (primary sort key -1)
          mesh%node_classes(NON_CORE_ENTITY) = &
               & mesh%node_classes(NON_CORE_ENTITY) + &
               & count(entity_sort_list(start:finish, 0) == -1)*dofs_per_stack(d)
          ! L1 halo (primary sort key \in [0, COMM_SIZE])
          mesh%node_classes(EXEC_HALO_ENTITY) = mesh%node_classes(EXEC_HALO_ENTITY) + &
               & count(entity_sort_list(start:finish, 0) >= 0 .and. &
               & entity_sort_list(start:finish, 0) <= getnprocs())*dofs_per_stack(d)
          ! L2 halo (primary sort key \in [COMM_SIZE + 1, \infty))
          mesh%node_classes(NON_EXEC_HALO_ENTITY) = mesh%node_classes(NON_EXEC_HALO_ENTITY) + &
               & count(entity_sort_list(start:finish, 0) > getnprocs())*dofs_per_stack(d)
       end do
    end if

    ! Determine the order in which topological entities should be numbered.
    call entity_order
    ! Associate dof numbers with each topological entity
    call topology_dofs
    ! Transpose dof numbers into the mesh object
    call populate_ndglno

    ! Non-parallel case, only know the node count now.
    if (.not.have_halos) then
       mesh%node_classes = 0
       mesh%node_classes(CORE_ENTITY) = node_count(mesh)
    end if

    ! Firedrake and op2 expect these to be the cumulative sum, so do it here
    do d=NON_CORE_ENTITY, NON_EXEC_HALO_ENTITY
       mesh%node_classes(d) = mesh%node_classes(d) + mesh%node_classes(d-1)
    end do
    if (have_halos) then
       call create_halos(mesh, topology%halos)
    end if
    if(have_facets) then
       if (mesh_dim(mesh)>1) call deallocate(facet_numbers)
    end if
    if (mesh_dim(mesh)>2) call deallocate(edge_numbers)
    call deallocate(edge_list)
    call deallocate_topology_halos

  contains

    function entity_dofs(dim, entity, dofs_per) result (dofs)
      ! Global dofs associated with local entity
      integer, intent(in) :: dim, entity
      integer, dimension(0:) :: dofs_per
      integer, dimension(dofs_per(dim)) :: dofs

      integer :: i,d,e

      dofs=entity_dof_starts(entity)&
           +[(i, i=1,dofs_per(dim))]

    end function entity_dofs

    function halo_entity_dofs(dim, entity, dofs_per) result (dofs)
      ! Halo-consistent dofs associated with local entity
      integer, intent(in) :: dim, entity
      integer, dimension(0:) :: dofs_per
      integer, dimension(dofs_per(dim)) :: dofs

      integer :: i,d,e

      dofs=halo_entity_dof_starts(entity)&
           +[(i, i=1,dofs_per(dim))]

    end function halo_entity_dofs

    subroutine calculate_dofs_per_entity
      ! Note that this will fail for meshes where not every topological
      !  entity of a given type has the same number of dofs. EG. wedge
      !  elements.
      integer :: i

      do i=0,mesh_dim(mesh)
         if (allocated(element%entity2dofs(i,1)%dofs)) then
            dofs_per(i)=size(element%entity2dofs(i,1)%dofs)
         else
            dofs_per(i)=0
         end if
      end do

    end subroutine calculate_dofs_per_entity

    subroutine number_topology
      integer :: ele1, ele2, vertex1, vertex2
      integer :: entity,n
      integer, dimension(:), pointer :: neigh

      ! No need to number vertices: it comes from the topology.
      entity=entity_counts(0)

      ! Number the edges
      if (mesh_dim(mesh)>2) then
         ! If mesh_dim(mesh)==2 then this is subsumed in the facets.
         call allocate(edge_numbers, edge_list, type=CSR_INTEGER)
         call zero(edge_numbers)

         do vertex1=1, node_count(topology)
            neigh=>row_m_ptr(edge_list, vertex1)
            do n=1, size(neigh)
               vertex2=neigh(n)
               if (vertex2>vertex1) then
                  entity=entity+1
                  call set(edge_numbers,vertex1,vertex2,entity)
                  call set(edge_numbers,vertex2,vertex1,entity)
               end if
            end do
         end do
      end if

      ! Number the facets
      if (mesh_dim(mesh)>1.and.have_facets) then
         call allocate(facet_numbers, facet_list, type=CSR_INTEGER)
         call zero(facet_numbers)

         do ele1=1, element_count(mesh)
            neigh=>ele_neigh(topology,ele1)
            do n=1, size(neigh)
               ele2=neigh(n)
               if (ele2<0) then
                  ! Exterior facet
                  entity=entity+1
                  call set(facet_numbers,ele1,ele2,entity)
               else if (ele2>ele1) then
                  entity=entity+1
                  call set(facet_numbers,ele1,ele2,entity)
                  call set(facet_numbers,ele2,ele1,entity)
               end if
            end do
         end do
      end if

      ! There's nothing to do for cells.
      if (mesh_dim(mesh)>0) then
         entity=entity+entity_counts(mesh_dim(mesh))
      end if

#ifdef DDEBUG
      if (.not.have_facets.and.mesh_dim(mesh)==2) then
         entity=entity+entity_counts(1)
      end if
      assert(sum(entity_counts)==entity)
#endif

    end subroutine number_topology

    subroutine create_topology_halos
      ! Armed with vertex, edge and facet numberings, we can work out who
      !  owns each of these objects and to which halos they may belong.
      !
      ! This routine also establishes the entity_sort_list, and so must be
      !  called even if there are no halos.
      integer :: ele, d, e, v, entity, cell_entity, rank
      integer, dimension(2) :: edge
      integer, dimension(:), pointer :: vertices, facets
      logical, dimension(:), allocatable :: level1

      ewrite(1,*) "Creating topology halos"

      allocate(entity_owner(sum(entity_counts)))
      rank=getprocno()

      if (have_halos) then
         ! Hard coded 2-level halo assumption.
         allocate(entity_send_targets(sum(entity_counts), 2))
         allocate(entity_receive_level(sum(entity_counts)))
         call get_node_owners(topology%halos(2), entity_owner(:entity_counts(0)))

         ! Level 1 flag for vertices. Used to determine if an entity has any
         !  level 1 vertices and is hence level 2.
         allocate(level1(entity_counts(0)))
         level1=.False.

         entity_receive_level=0
      else
         entity_owner=rank
      end if

      ! Establish entity ownership and the level 1 halo.
      do ele=1,element_count(topology)
         vertices=>ele_nodes(topology, ele)

         if (have_facets.and.mesh_dim(mesh)>1) facets=>row_ival_ptr(facet_numbers,ele)

         do d=0, mesh_dim(mesh)
            do e=1,cell%entity_counts(d)
               if (d==0) then
                  entity=vertices(cell%entities(d,e)%vertices(1))
               else if (d==cell%dimension) then
                  entity=sum(entity_counts(:d-1))+ele
               else if (d==cell%dimension-1) then
                  if (have_facets) entity=facets(e)
               else if (d==1) then
                  ! This case only gets hit for the edges of 3d elements.

                  ! The edge consists of the global dofs in the topology.
                  edge=vertices(entity_vertices(cell,[1,e]))
                  ! Now look up the corresponding edge number.
                  entity=ival(edge_numbers,edge(1),edge(2))
               end if

               ! Conduct voting for entity ownership.
               entity_owner(entity)&
                    &=maxval(entity_owner(&
                    &         vertices(cell%entities(d,e)%vertices)))

               ! Set up the sort order lists. The primary key is the owner,
               !  the following sort keys are the UIDs of the vertices in
               !   ascending order.
               if (entity_sort_list(entity, 0) == -666) then
                  if (entity_owner(entity)==rank) then
                     ! Default to core until proven level 1 halo.
                     entity_sort_list(entity,0)=-2
                  else
                     ! Default to level 2 halo until proven level 1 halo.
                     entity_sort_list(entity,0)=entity_owner(entity)&
                           + getnprocs() + 1
                  end if
                  entity_sort_list(entity,1:size(cell%entities(d,e)%vertices))&
                       =sorted(int(node_val(uid,vertices(cell%entities(d,e)%vertices))))
               end if

               ! Don't proceed to calculate halos if we don't have halos.
               if (.not.have_halos) cycle

               ! A level 1 halo cell has vertices owned by different
               !  processors. Note that level1 in this context means level1
               !   for any processor, not just us.
               if (all(entity_owner(vertices)==entity_owner(vertices(1)))) cycle

               ! Check if this is level 1 WRT us.
               if (any(entity_owner(vertices)==rank)) then
                  level1(vertices)=.True.

                  if (entity_owner(entity)/=rank) then
                     entity_sort_list(entity,0)=entity_owner(entity)
                  end if
               end if

               ! If the entity is foreign, check if this element causes it
               !  to have a level 1 receive level.
               if (entity_owner(entity)/=rank) then
                  if (any(entity_owner(vertices)==rank)) &
                     entity_receive_level(entity)=1
               end if

               ! If there are any foreign vertices, then insert all
               !  vertex owners into the send targets.
               if (any(entity_owner(vertices)/=rank)) then
                  call insert (entity_send_targets(entity,1), entity_owner(vertices))
               end if

            end do
         end do
      end do

      if (.not.have_halos) return

      ! Next, loop over elements again to establish the level 2 halo.
      do ele=1,element_count(topology)
         vertices=>ele_nodes(topology, ele)

         ! An element can only possibly be level 2 if one of its vertices
         !  is level 1.
         if (.not.any(level1(vertices))) cycle

         if (have_facets.and.mesh_dim(mesh)>1) facets=>row_ival_ptr(facet_numbers,ele)

         ! Special case for the cell
         cell_entity=sum(entity_counts(:cell%dimension-1))+ele
         do v=1,size(vertices)
            call insert(entity_send_targets(cell_entity,2), &
                 entity_send_targets(vertices(v),1))
         end do
         ! Set the receive level 2 if it isn't already 1.
         if (entity_owner(cell_entity)/=rank&
              .and.entity_receive_level(cell_entity)==0) then
               entity_receive_level(cell_entity)=2
         end if

         if (entity_owner(cell_entity)==rank) then
            entity_sort_list(cell_entity,0)=-1
         end if


         do d=0, mesh_dim(mesh)-1
            do e=1,cell%entity_counts(d)
               if (d==0) then
                  entity=vertices(e)
               else if (d==cell%dimension-1) then
                  if (have_facets) entity=facets(e)
               else if (d==1) then
                  ! This case only gets hit for the edges of 3d elements.

                  ! The edge consists of the global dofs in the topology.
                  edge=vertices(entity_vertices(cell,[1,e]))
                  ! Now look up the corresponding edge number.
                  entity=ival(edge_numbers,edge(1),edge(2))
               end if

               ! Set the receive level 2 if it isn't already 1.
               if (entity_owner(entity)/=rank&
                    .and.entity_receive_level(entity)==0) then
                  entity_receive_level(entity)=2
               end if

               ! Insert the cell send list into all the other send lists.
               call insert(entity_send_targets(entity,2), &
                    entity_send_targets(cell_entity,2))

               if (entity_owner(entity)==rank) then
                  entity_sort_list(entity,0)=-1
               end if
            end do
         end do
      end do

      ! There are corner cases of visible entities which are outside the
      !  level 2 halo. We give them a receive level of 3.
      do entity = 1, size(entity_owner)
         if (entity_owner(entity)/=rank .and. entity_receive_level(entity)==0) then
            entity_receive_level(entity)=3
         end if
      end do

      deallocate(level1)


    end subroutine create_topology_halos

    subroutine deallocate_topology_halos

      deallocate(entity_owner)

      if (.not.have_halos) return
      call deallocate(entity_send_targets)

      deallocate(entity_receive_level, entity_send_targets)

    end subroutine deallocate_topology_halos

    subroutine entity_order
      ! Form an orderly queue of topological entities. With halos, this
      !  ensures that all non-owned entities follow all owned entities.
      integer :: i

      call sort(entity_sort_list, visit_order)

      if (have_halos) then
         ! Remove the distinction between core and non-core, and halo
         !  levels 1 and 2.
         where (entity_sort_list(:,0) == -2)
            entity_sort_list(:,0) = -1
         elsewhere (entity_sort_list(:,0) > getnprocs())
            entity_sort_list(:,0) = &
                 entity_sort_list(:,0) - (getnprocs() + 1)
         end where
         call sort(entity_sort_list, halo_visit_order)
      else
         ! In the serial case, we just have one order.
         halo_visit_order = visit_order
      end if

    end subroutine entity_order

    subroutine topology_dofs
      ! For each topological entity, calculate the dofs which will lie on
      !  it.
      integer :: i, dof, halo_dof

      dof=0
      halo_dof=0
      do i=1,size(visit_order)
         entity_dof_starts(visit_order(i))=dof
         dof=dof+dofs_per_stack(entity_dim(visit_order(i)))

         halo_entity_dof_starts(halo_visit_order(i))=halo_dof
         halo_dof=halo_dof+dofs_per_stack(entity_dim(halo_visit_order(i)))
      end do

      assert(halo_dof == dof)
      total_dofs = dof

    end subroutine topology_dofs

    function entity_dim(entity)
      integer, intent(in) :: entity
      integer :: entity_dim

      integer d

      do d=0,ubound(entity_counts,1)
         if (entity<=sum(entity_counts(0:d))) then
            entity_dim=d
            return
         end if
      end do

      FLAbort("illegal entity")

    end function entity_dim

    subroutine populate_ndglno
      ! Having established what the numbering should be, actually populate
      !  the mesh with dof numbers.
      integer :: ele, d, e, entity
      integer, dimension(2) :: edge
      integer, dimension(:), pointer :: ele_dofs, topo_dofs, facets

      ewrite(1,*) "Populating ndglno for ", trim(mesh%name)

      allocate(dof_to_halo_dof(total_dofs))
      dof_to_halo_dof = -666

      do ele=1,element_count(mesh)
         ele_dofs=>ele_nodes(mesh, ele)
         topo_dofs=>ele_nodes(topology, ele)
         if (have_facets.and.mesh_dim(mesh)>1) facets=>row_ival_ptr(facet_numbers,ele)
#ifdef DDEBUG
         ele_dofs=0
#endif

         do d=0,mesh_dim(mesh)
            do e=1,cell%entity_counts(d)
               if (d==0) then
                  entity=topo_dofs(e)
               else if (d==cell%dimension) then
                  entity=sum(entity_counts(:d-1))+ele
               else if (d==cell%dimension-1) then
                  if (have_facets) entity=facets(e)
               else if (d==1) then
                  ! This case only gets hit for the edges of 3d elements.

                  ! The edge consists of the global dofs in the topology.
                  edge=topo_dofs(entity_vertices(cell,[1,e]))
                  ! Now look up the corresponding edge number.
                  entity=ival(edge_numbers,edge(1),edge(2))
               end if

               ele_dofs(element%entity2dofs(d,e)%dofs)=&
                    entity_dofs(d, entity, dofs_per)

               dof_to_halo_dof(entity_dofs(d, entity, dofs_per_stack)) &
                    = halo_entity_dofs(d, entity, dofs_per_stack)

            end do
         end do

         assert(all(ele_dofs>0))

      end do

      assert(all(dof_to_halo_dof>0))

      if (extrusion) then
        mesh%nodes=total_dofs
      else
        if (size(mesh%ndglno)>0) then
          mesh%nodes=maxval(mesh%ndglno)
        else
          mesh%nodes=0
        end if
      end if
    end subroutine populate_ndglno

    subroutine create_halos(mesh, thalos)
      type(mesh_type), intent(inout) :: mesh
      type(halo_type), dimension(:), intent(in) :: thalos

      type(integer_set), dimension(:,:), allocatable :: sends, receives
      integer :: local_dofs, data_type, entity, h, hh, k, p, rank

      allocate(sends(size(thalos), size(thalos(1)%sends)), &
           receives(size(thalos), size(thalos(1)%receives)))
      call allocate(sends)
      call allocate(receives)

      allocate(mesh%halos(size(thalos)))

      rank=getprocno()
      local_dofs=0

      if (size(mesh%shape%entity2dofs(cell%dimension,1)%dofs)==mesh%shape%ndof) then
         data_type=HALO_TYPE_DG_NODE
      else
         data_type=HALO_TYPE_CG_NODE
      end if

      ! Order doesn't matter at this stage.
      do entity=1,size(visit_order)

         if (entity_owner(entity)==rank) then
            ! One of ours
            local_dofs=local_dofs+dofs_per_stack(entity_dim(entity))

            do h=1,size(mesh%halos)
               do k=1,key_count(entity_send_targets(entity,h))
                  ! Don't self-send
                  if (fetch(entity_send_targets(entity,h),k)==rank) cycle

                  call insert(sends(h,fetch(entity_send_targets(entity,h),k)),&
                       & entity_dofs(entity_dim(entity),entity, dofs_per_stack))

               end do
            end do
         else
            ! Someone else's.

            assert(entity_receive_level(entity)>0)
            ! Insert into all receieve halos greater than or equal to the
            ! receive level.
            do h=size(mesh%halos),entity_receive_level(entity),-1

               call insert(receives(h,entity_owner(entity)),&
                    entity_dofs(entity_dim(entity),entity, dofs_per_stack))

            end do
         end if
      end do

      do h=1,size(mesh%halos)
         if (data_type==HALO_TYPE_CG_NODE) then
            hh=h
         else
            ! DG case: all halos are level 2.
            hh=2
         end if

         call allocate(mesh%halos(h), &
              nsends=key_count(sends(hh,:)), &
              nreceives=key_count(receives(hh,:)), &
              nowned_nodes=local_dofs, &
              data_type=data_type,&
              communicator=thalos(1)%communicator)

         write(mesh%halos(h)%name,'(a,i0,a)') trim(mesh%name)//"Level",h,"Halo"

         ! Since halo dof order follows UID order, sorting is safe.
         do p=1,size(mesh%halos(h)%sends)
            mesh%halos(h)%sends(p)%ptr=sorted(set2vector(sends(hh,p)), &
                 & key = dof_to_halo_dof(set2vector(sends(hh,p))))
         end do
         do p=1,size(mesh%halos(h)%receives)
            mesh%halos(h)%receives(p)%ptr=sorted(set2vector(receives(hh,p))&
                 &, key = dof_to_halo_dof(set2vector(receives(hh,p))))
         end do


         call create_global_to_universal_numbering(mesh%halos(h))
         call create_ownership(mesh%halos(h))
      end do

      call deallocate(sends)
      call deallocate(receives)

      call refresh_topology(mesh)

    end subroutine create_halos

  end subroutine make_global_numbering_new

  function count_topological_entities(topology, edge_list, facet_list) result&
       & (entities)
    ! Calculate the number of entities of each dimension in topology.
    type(mesh_type), intent(in) :: topology
    type(csr_sparsity), intent(in) :: edge_list, facet_list

    integer, dimension(0:mesh_dim(topology)) :: entities

    integer :: dim

    dim=mesh_dim(topology)

    entities(0)=node_count(topology)
    entities(mesh_dim(topology))=element_count(topology)

    if (dim>1) then
       entities(1)=entries(edge_list)/2
    end if
    if (dim>2) then
       ! (all_facets + exterior_facets)/2
       entities(2)=(entries(facet_list)+count(facet_list%colm<0))/2
    end if

  end function count_topological_entities

  subroutine make_global_numbering_DG(new_nonods, new_ndglno, Totele,&
       & element, element_halos, new_halos)
    ! Construct a global node numbering for the solution variables in a
    ! Discontinuous Galerkin simulation. This is trivial.
    !
    ! Note that this code is broken for mixed element meshes.
    integer, intent(in) :: totele
    type(element_type), intent(in) :: element

    integer, dimension(:), intent(out) :: new_ndglno
    integer, intent(out) :: new_nonods
    type(halo_type), dimension(:), intent(in), optional :: element_halos
    type(halo_type), dimension(:), intent(out), optional :: new_halos

    integer :: i

    new_nonods=totele*element%ndof

    forall (i=1:new_nonods)
       new_ndglno(i)=i
    end forall

    if (.not.present(element_halos)) return
    assert(present(new_halos))
    assert(size(element_halos)==size(new_halos))

    do i=1,size(new_halos)
       call make_halo_dg(element, element_halos(i), new_halos(i))
    end do

  contains


    subroutine make_halo_dg(element, element_halo, new_halo)
      !!< This routine constructs a node halo given an element halo.
      type(element_type), intent(in) :: element
      type(halo_type), intent(in) :: element_halo
      type(halo_type), intent(out) :: new_halo

      integer, dimension(size(element_halo%sends)) :: nsends
      integer, dimension(size(element_halo%receives)) :: nreceives

      integer :: i,j,k, nloc

      nloc=element%ndof

      do i=1, size(nsends)
         nsends(i)=nloc*size(element_halo%sends(i)%ptr)
      end do
      do i=1, size(nreceives)
         nreceives(i)=nloc*size(element_halo%receives(i)%ptr)
      end do

      call allocate(new_halo, &
           nsends, &
           nreceives, &
                                !! Query what is the naming convention for halos.
           name=trim(halo_name(element_halo)) // "DG", &
           nprocs=element_halo%nprocs, &
           nowned_nodes=element_halo%nowned_nodes*element%ndof, &
           data_type=HALO_TYPE_DG_NODE, &
           ordering_scheme=halo_ordering_scheme(element_halo))

      do i=1, size(nsends)
         do j=1,size(element_halo%sends(i)%ptr)

            new_halo%sends(i)%ptr((j-1)*nloc+1:j*nloc)&
                 =(element_halo%sends(i)%ptr(j)-1)*nloc + (/(k,k=1,nloc)/)

         end do
      end do

      do i=1, size(nreceives)
         do j=1,size(element_halo%receives(i)%ptr)

            new_halo%receives(i)%ptr((j-1)*nloc+1:j*nloc)&
                 =(element_halo%receives(i)%ptr(j)-1)*nloc + (/(k,k=1,nloc)/)

         end do
      end do

      call create_global_to_universal_numbering(new_halo)
      call create_ownership(new_halo)

    end subroutine make_halo_dg


  end subroutine make_global_numbering_DG

end module global_numbering
