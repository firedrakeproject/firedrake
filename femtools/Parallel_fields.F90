!    Copyright (C) 2007 Imperial College London and others.
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

module parallel_fields
  !!< This module exists to separate out parallel operations on fields

  use fields_allocates
  use fields_base
  use fields_data_types
  use parallel_tools
  use halos_communications
  use halos_ownership
  use halos_numbering
  use halo_data_types
  use halos_base
  use fields_manipulation
  use mpi_interfaces

  implicit none

  private

  public :: halo_communicator, element_owned, element_neighbour_owned, &
       & element_owner, node_owned, assemble_ele, &
       & surface_element_owned, nowned_nodes
  ! Apparently ifort has a problem with the generic name node_owned
  public :: node_owned_mesh, zero_non_owned
  public :: universal_number, universal_number_field, periodic_universal_ID_field

  interface node_owned
    module procedure node_owned_mesh, node_owned_scalar, node_owned_vector, &
      & node_owned_tensor
  end interface node_owned

  interface node_owner
    module procedure node_owner_mesh, node_owner_scalar, &
      & node_owner_vector, node_owner_tensor
  end interface node_owner

  interface element_owned
    module procedure element_owned_mesh, element_owned_scalar, &
      & element_owned_vector, element_owned_tensor
  end interface element_owned

  interface element_neighbour_owned
     module procedure element_neighbour_owned_mesh, &
          & element_neighbour_owned_scalar, element_neighbour_owned_vector, &
          & element_neighbour_owned_tensor
  end interface element_neighbour_owned

  interface element_owner
    module procedure element_owner_mesh, element_owner_scalar, &
      & element_owner_vector, element_owner_tensor
  end interface element_owner

  interface assemble_ele
    module procedure assemble_ele_mesh, assemble_ele_scalar, &
      & assemble_ele_vector, assemble_ele_tensor
  end interface assemble_ele

  interface surface_element_owned
    module procedure surface_element_owned_mesh, surface_element_owned_scalar, &
      & surface_element_owned_vector, surface_element_owned_tensor
  end interface surface_element_owned

  interface zero_non_owned
     module procedure zero_non_owned_scalar, zero_non_owned_vector
  end interface

  interface halo_communicator
    module procedure halo_communicator_mesh, halo_communicator_scalar, &
      & halo_communicator_vector, halo_communicator_tensor
  end interface halo_communicator

  interface nowned_nodes
    module procedure nowned_nodes_mesh, nowned_nodes_scalar, &
      & nowned_nodes_vector, nowned_nodes_tensor
  end interface nowned_nodes

  interface universal_number
     module procedure mesh_universal_number, mesh_universal_number_vector
  end interface universal_number

contains

  function universal_number_field(mesh) result (unn)
    !!< Return a field containing the universal numbering of mesh.
    type(mesh_type), intent(inout) :: mesh

    type(scalar_field) :: unn
    integer :: node

    call allocate(unn, mesh, trim(mesh%name)//"UniversalNumbering")

    do node=1,node_count(mesh)
       call set(unn, node, real(universal_number(mesh, node)))
    end do

  end function universal_number_field

  function mesh_universal_number(mesh, node) result (unn)
    !!< Return the universal node number of node in mesh.
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: node

    integer :: unn

    integer :: nhalos

    nhalos=halo_count(mesh)
    if (nhalos>0) then
       unn=halo_universal_number(mesh%halos(nhalos), node)
    else
       unn=node
    end if

  end function mesh_universal_number

  function mesh_universal_number_vector(mesh, node) result (unn)
    !!< Return the universal node number of node in mesh.
    type(mesh_type), intent(in) :: mesh
    integer, dimension(:), intent(in) :: node

    integer, dimension(size(node)) :: unn

    integer :: nhalos

    nhalos=halo_count(mesh)
    if (nhalos>0) then
       unn=halo_universal_number(mesh%halos(nhalos), node)
    else
       unn=node
    end if

  end function mesh_universal_number_vector

  function periodic_universal_ID_field(uid_in, periodic_mesh) result (uid)
    ! Given a (non-periodic) universal ID field, return the periodic
    !  version applicable to periodic_mesh. Note that this is the universal
    !  ID which is not necessarily contiguous and definitely not contiguous
    !  per processor.
    type(scalar_field), intent(in) :: uid_in
    type(mesh_type), intent(inout) :: periodic_mesh

    type(scalar_field) :: uid

    integer :: ele

    call allocate(uid, periodic_mesh, trim(periodic_mesh%name)//"UniversalNumbering")

    do ele=1,element_count(periodic_mesh)
       call set(uid, ele_nodes(uid, ele), ele_val(uid_in,ele))
    end do

  end function periodic_universal_ID_field

  function halo_communicator_mesh(mesh) result(communicator)
    !!< Return the halo communicator for this mesh. Returns the halo
    !!< communicator off of the max level node halo.

    type(mesh_type), intent(in) :: mesh

    integer :: communicator

    integer :: nhalos

    nhalos = halo_count(mesh)
    if(nhalos > 0) then
      communicator = halo_communicator(mesh%halos(nhalos))
    else
#ifdef HAVE_MPI
      communicator = MPI_COMM_FEMTOOLS
#else
      communicator = -1
#endif
    end if

  end function halo_communicator_mesh

  function halo_communicator_scalar(s_field) result(communicator)
    !!< Return the halo communicator for this field. Returns the halo
    !!< communicator off of the max level node halo.

    type(scalar_field), intent(in) :: s_field

    integer :: communicator

    communicator = halo_communicator(s_field%mesh)

  end function halo_communicator_scalar

  function halo_communicator_vector(v_field) result(communicator)
    !!< Return the halo communicator for this mesh. Returns the halo
    !!< communicator off of the max level node halo.

    type(vector_field), intent(in) :: v_field

    integer :: communicator

    communicator = halo_communicator(v_field%mesh)

  end function halo_communicator_vector

  function halo_communicator_tensor(t_field) result(communicator)
    !!< Return the halo communicator for this mesh. Returns the halo
    !!< communicator off of the max level node halo.

    type(tensor_field), intent(in) :: t_field

    integer :: communicator

    communicator = halo_communicator(t_field%mesh)

  end function halo_communicator_tensor

  function node_owner_mesh(mesh, node_number) result(owner)
  !!< Return number of processor that owns the supplied node in the
  !given mesh

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: node_number

    integer :: nhalos
    integer :: owner

    assert(node_number > 0)
    assert(node_number <= ele_count(mesh))

    nhalos = halo_count(mesh)

    if(nhalos == 0) then
      owner=getprocno()
    else
      owner = halo_node_owner(mesh%halos(nhalos), node_number)
    end if

  end function node_owner_mesh

  function node_owner_scalar(s_field, node_number) result(owner)
    !!< Return the processor that owns the supplied node in the mesh of the given scalar
    !!< field

    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: node_number

    integer :: owner

    owner = node_owner(s_field%mesh, node_number)

  end function node_owner_scalar

  function node_owner_vector(v_field, node_number) result(owner)
    !!< Return the processor that owns the supplied node in the mesh of the given vector
    !!< field

    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: node_number

    integer :: owner

    owner = node_owner(v_field%mesh, node_number)

  end function node_owner_vector

  function node_owner_tensor(t_field, node_number) result(owner)
    !!< Return the processor that owns the supplied node in the mesh of the given tensor
    !!< field

    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: node_number

    integer :: owner

    owner = node_owner(t_field%mesh, node_number)

  end function node_owner_tensor

  function node_owned_mesh(mesh, node_number) result(owned)
    !!< Return whether the supplied node in the given mesh is owned by this
    !!< process

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: node_number

    logical :: owned

    assert(node_number > 0)
    assert(node_number <= node_count(mesh))

    if(isparallel()) then
       ! For ownership it doesn't matter if we use depth 1 or 2.
       assert(associated(mesh%halos))
       owned = node_owned(mesh%halos(1), node_number)
    else
      owned = .true.
    end if

  end function node_owned_mesh

  function node_owned_scalar(s_field, node_number) result(owned)
    !!< Return whether the supplied node in the given field's mesh is owned by this
    !!< process

    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: node_number

    logical :: owned

    owned = node_owned(s_field%mesh, node_number)

  end function node_owned_scalar

  function node_owned_vector(v_field, node_number) result(owned)
    !!< Return whether the supplied node in the given field's mesh is owned by this
    !!< process

    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: node_number
    logical :: owned

    owned = node_owned(v_field%mesh, node_number)

  end function node_owned_vector

  function node_owned_tensor(t_field, node_number) result(owned)
    !!< Return whether the supplied node in the given field's mesh is owned by this
    !!< process

    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: node_number

    logical :: owned

    owned = node_owned(t_field%mesh, node_number)

  end function node_owned_tensor

  function element_owned_mesh(mesh, element_number) result(owned)
    !!< Return whether the supplied element in the given mesh is owned by this
    !!< process

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: element_number

    integer :: nhalos
    logical :: owned

    assert(element_number > 0)
    assert(element_number <= ele_count(mesh))

    nhalos = element_halo_count(mesh)
    if(nhalos > 0) then
      owned = node_owned(mesh%element_halos(nhalos), element_number)
    else
       owned = .true.
    end if

  end function element_owned_mesh

  function element_owned_scalar(s_field, element_number) result(owned)
    !!< Return whether the supplied element in the mesh of the given scalar
    !!< field is owned by this process

    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: element_number

    logical :: owned

    owned = element_owned(s_field%mesh, element_number)

  end function element_owned_scalar

  function element_owned_vector(v_field, element_number) result(owned)
    !!< Return whether the supplied element in the mesh of the given scalar
    !!< field is owned by this process

    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: element_number

    logical :: owned

    owned = element_owned(v_field%mesh, element_number)

  end function element_owned_vector

  function element_owned_tensor(t_field, element_number) result(owned)
    !!< Return whether the supplied element in the mesh of the given tensor
    !!< field is owned by this process

    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: element_number

    logical :: owned

    owned = element_owned(t_field%mesh, element_number)

  end function element_owned_tensor

  function element_neighbour_owned_mesh(mesh, element_number) result(owned)
    !!< Return .true. if ELEMENT_NUMBER has a neighbour in MESH that
    !!< is owned by this process otherwise .false.

    !! Note, you cannot use this function to compute if
    !! ELEMENT_NUMBER is owned.  Imagine if this is the only owned
    !! element on a process, then none of the neighbours will be owned.
    !! You can use this function to compute whether ELEMENT_NUMBER is
    !! in the L1 element halo.
    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: element_number
    logical :: owned
    integer, dimension(:), pointer :: neighbours
    integer :: i, n_neigh

    neighbours => ele_neigh(mesh, element_number)
    n_neigh = size(neighbours)
    owned = .false.
    do i = 1, n_neigh
       ! If element_number is in the halo, then some of the neighbour
       ! data might not be available, in which case neighbours(i) can
       ! be invalid (missing data are marked by negative values).
       if ( neighbours(i) <= 0 ) cycle
       if ( element_owned(mesh, neighbours(i)) ) then
          owned = .true.
          return
       end if
    end do
  end function element_neighbour_owned_mesh

  function element_neighbour_owned_scalar(field, element_number) result(owned)
    !!< Return .true. if ELEMENT_NUMBER has a neighbour in FIELD that
    !!< is owned by this process otherwise .false.

    !! Note, you cannot use this function to compute if
    !! ELEMENT_NUMBER is owned.  Imagine if this is the only owned
    !! element on a process, then none of the neighbours will be owned.
    !! You can use this function to compute whether ELEMENT_NUMBER is
    !! in the L1 element halo.
    type(scalar_field), intent(in) :: field
    integer, intent(in) :: element_number
    logical :: owned

    owned = element_neighbour_owned(field%mesh, element_number)
  end function element_neighbour_owned_scalar

  function element_neighbour_owned_vector(field, element_number) result(owned)
    !!< Return .true. if ELEMENT_NUMBER has a neighbour in FIELD that
    !!< is owned by this process otherwise .false.

    !! Note, you cannot use this function to compute if
    !! ELEMENT_NUMBER is owned.  Imagine if this is the only owned
    !! element on a process, then none of the neighbours will be owned.
    !! You can use this function to compute whether ELEMENT_NUMBER is
    !! in the L1 element halo.
    type(vector_field), intent(in) :: field
    integer, intent(in) :: element_number
    logical :: owned

    owned = element_neighbour_owned(field%mesh, element_number)
  end function element_neighbour_owned_vector

  function element_neighbour_owned_tensor(field, element_number) result(owned)
    !!< Return .true. if ELEMENT_NUMBER has a neighbour in FIELD that
    !!< is owned by this process otherwise .false.

    !! Note, you cannot use this function to compute if
    !! ELEMENT_NUMBER is owned.  Imagine if this is the only owned
    !! element on a process, then none of the neighbours will be owned.
    !! You can use this function to compute whether ELEMENT_NUMBER is
    !! in the L1 element halo.
    type(tensor_field), intent(in) :: field
    integer, intent(in) :: element_number
    logical :: owned

    owned = element_neighbour_owned(field%mesh, element_number)
  end function element_neighbour_owned_tensor

  function element_owner_mesh(mesh, element_number) result(owner)
  !!< Return number of processor that owns the supplied element in the
  !given mesh

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: element_number

    integer :: nhalos
    integer :: owner

    assert(element_number > 0)
    assert(element_number <= ele_count(mesh))

    nhalos = element_halo_count(mesh)

    if(nhalos == 0) then
      owner=getprocno()
    else
      owner = halo_node_owner(mesh%element_halos(nhalos), element_number)
    end if

  end function element_owner_mesh

  function element_owner_scalar(s_field, element_number) result(owner)
    !!< Return the processor that owns the supplied element in the mesh of the given scalar
    !!< field

    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: element_number

    integer :: owner

    owner = element_owner(s_field%mesh, element_number)

  end function element_owner_scalar

  function element_owner_vector(v_field, element_number) result(owner)
    !!< Return the processor that owns the supplied element in the mesh of the given vector
    !!< field

    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: element_number

    integer :: owner

    owner = element_owner(v_field%mesh, element_number)

  end function element_owner_vector

  function element_owner_tensor(t_field, element_number) result(owner)
    !!< Return the processor that owns the supplied element in the mesh of the given tensor
    !!< field

    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: element_number

    integer :: owner

    owner = element_owner(t_field%mesh, element_number)

  end function element_owner_tensor

  function assemble_ele_mesh(mesh, ele) result(assemble)
    !!< Return whether the supplied element for the supplied mesh should be
    !!< assembled. An element need not be assembled if it has no owned nodes.

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: ele

    logical :: assemble

    select case(continuity(mesh))
      case(0)
        if(associated(mesh%halos)) then
          assemble = any(nodes_owned(mesh%halos(1), ele_nodes(mesh, ele)))
        else
          assemble = .true.
        end if
      case(-1)
        if(associated(mesh%element_halos)) then
          assemble = element_owned(mesh, ele)
        else
          assemble = .true.
        end if
      case default
        ewrite(-1, "(a,i0)") "For mesh continuity", mesh%continuity
        FLAbort("Unrecognised mesh continuity")
    end select

  end function assemble_ele_mesh

  function assemble_ele_scalar(s_field, ele) result(assemble)
    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: ele

    logical :: assemble

    assemble = assemble_ele(s_field%mesh, ele)

  end function assemble_ele_scalar

  function assemble_ele_vector(v_field, ele) result(assemble)
    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: ele

    logical :: assemble

    assemble = assemble_ele(v_field%mesh, ele)

  end function assemble_ele_vector

  function assemble_ele_tensor(t_field, ele) result(assemble)
    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: ele

    logical :: assemble

    assemble = assemble_ele(t_field%mesh, ele)

  end function assemble_ele_tensor

  function surface_element_owned_mesh(mesh, face_number) result(owned)
    !!< Return if the supplied surface element in the given mesh is owned by
    !!< this process.

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: face_number

    logical :: owned

    owned = element_owned(mesh, face_ele(mesh, face_number))

  end function surface_element_owned_mesh

  function surface_element_owned_scalar(s_field, face_number) result(owned)
    !!< Return if the supplied surface element in the mesh of the given field is
    !!< owned by this process.

    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: face_number

    logical :: owned

    owned = surface_element_owned(s_field%mesh, face_number)

  end function surface_element_owned_scalar

  function surface_element_owned_vector(v_field, face_number) result(owned)
    !!< Return if the supplied surface element in the mesh of the given field is
    !!< owned by this process.

    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: face_number

    logical :: owned

    owned = surface_element_owned(v_field%mesh, face_number)

  end function surface_element_owned_vector

  function surface_element_owned_tensor(t_field, face_number) result(owned)
    !!< Return if the supplied surface element in the mesh of the given field is
    !!< owned by this process.

    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: face_number

    logical :: owned

    owned = surface_element_owned(t_field%mesh, face_number)

  end function surface_element_owned_tensor

  subroutine zero_non_owned_scalar(field)
    !!< Zero all of the entries of field which do not correspond to
    !!< that are owned by this process.
    !!<
    !!< This is useful for where dirty halo data has poluted a field.
    type(scalar_field), intent(inout) :: field

    integer :: i
    real :: zero

    zero=0.0

    if (.not.isparallel()) return

    if (halo_ordering_scheme(field%mesh%halos(1))&
         &==HALO_ORDER_TRAILING_RECEIVES) then
       do i = halo_nowned_nodes(field%mesh%halos(1))+1,&
            node_count(field)
          call set(field, i, zero)
       end do
    else
       do i = 1, node_count(field)
          if (.not.node_owned(field, i)) then
             call set(field, i, zero)
          end if
       end do
    end if

  end subroutine zero_non_owned_scalar

  subroutine zero_non_owned_vector(field)
    !!< Zero all of the entries of field which do not correspond to
    !!< that are owned by this process.
    !!<
    !!< This is useful for where dirty halo data has poluted a vector field.
    type(vector_field), intent(inout) :: field

    integer :: i
    real, dimension(field%dim) :: zero

    zero=0.0

    if (.not.isparallel()) return

    if (halo_ordering_scheme(field%mesh%halos(1))&
         &==HALO_ORDER_TRAILING_RECEIVES) then
       do i = halo_nowned_nodes(field%mesh%halos(1))+1,&
            node_count(field)
          call set(field, i, zero)
       end do
    else
       do i = 1, node_count(field)
          if (.not.node_owned(field, i)) then
             call set(field, i, zero)
          end if
       end do
    end if

  end subroutine zero_non_owned_vector

  pure function nowned_nodes_mesh(mesh) result(nodes)
    type(mesh_type), intent(in) :: mesh

    integer :: nodes

    integer :: nhalos

    nhalos = halo_count(mesh)
    if(nhalos > 0) then
      nodes = halo_nowned_nodes(mesh%halos(nhalos))
    else
      nodes = node_count(mesh)
    end if

  end function nowned_nodes_mesh

  pure function nowned_nodes_scalar(s_field) result(nodes)
    type(scalar_field), intent(in) :: s_field

    integer :: nodes

    nodes = nowned_nodes(s_field%mesh)

  end function nowned_nodes_scalar

  pure function nowned_nodes_vector(v_field) result(nodes)
    type(vector_field), intent(in) :: v_field

    integer :: nodes

    nodes = nowned_nodes(v_field%mesh)

  end function nowned_nodes_vector

  pure function nowned_nodes_tensor(t_field) result(nodes)
    type(tensor_field), intent(in) :: t_field

    integer :: nodes

    nodes = nowned_nodes(t_field%mesh)

  end function nowned_nodes_tensor

end module parallel_fields
