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
!    License as published by the Free Software Foundation; either
!    version 2.1 of the License, or (at your option) any later version.
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

module halos_derivation

  use data_structures
  use fields_data_types
  use fields_allocates
  use fields_manipulation
  use fields_base
  use fldebug
  use halo_data_types
  use halos_allocates
  use halos_base
  use halos_debug
  use halos_numbering
  use halos_ownership
  use halos_repair
  use halos_communications
  use mpi_interfaces
  use parallel_tools
  use sparse_tools
  use linked_lists
  use parallel_tools

  implicit none

  private

  public :: derive_l1_from_l2_halo, derive_element_halo_from_node_halo, &
    & derive_maximal_surface_element_halo, derive_nonperiodic_halos_from_periodic_halos, derive_sub_halo
  public :: invert_comms_sizes, ele_owner, combine_halos, create_combined_numbering_trailing_receives

  interface derive_l1_from_l2_halo
    module procedure derive_l1_from_l2_halo_mesh
  end interface

  interface derive_element_halos_from_l2_halo
    module procedure derive_element_halos_from_l2_halo_mesh, &
      & derive_element_halos_from_l2_halo_halo
  end interface derive_element_halos_from_l2_halo

contains

  subroutine derive_l1_from_l2_halo_mesh(mesh, ordering_scheme, create_caches)
    !!< For the supplied mesh, generate a level 1 node halo from the level 2
    !!< node halo

    type(mesh_type), intent(inout) :: mesh
    !! By default the l1 halo will inherit it's ordering scheme from the l2
    !! halo. Supply this to override.
    integer, optional, intent(in) :: ordering_scheme
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

    assert(halo_count(mesh) >= 2)
    assert(.not. has_references(mesh%halos(1)))
    assert(has_references(mesh%halos(2)))

    mesh%halos(1) = derive_l1_from_l2_halo_halo(mesh, mesh%halos(2), &
      & ordering_scheme = ordering_scheme, create_caches = create_caches)

  end subroutine derive_l1_from_l2_halo_mesh

  function derive_l1_from_l2_halo_halo(mesh, l2_halo, ordering_scheme, create_caches) result(l1_halo)
    !!< Given a level 2 node halo for the supplied mesh, strip it down to form
    !!< a level 1 node halo

    type(mesh_type), intent(in) :: mesh
    type(halo_type), intent(in) :: l2_halo
    !! By default the l1 halo will inherit it's ordering scheme from the l2
    !! halo. Supply this to override.
    integer, optional, intent(in) :: ordering_scheme
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

    type(halo_type) :: l1_halo

    integer :: i, j, k, l, lordering_scheme, nprocs, proc
    integer, dimension(:), allocatable :: receive_paint
    integer, dimension(:), pointer :: neigh
    type(csr_sparsity), pointer :: nnlist
    type(integer_set), dimension(:), allocatable :: send_paint, sends, receives

    assert(continuity(mesh) == 0)
    assert(halo_data_type(l2_halo) == HALO_TYPE_CG_NODE)
    assert(.not. serial_storage_halo(l2_halo))

    if(present(ordering_scheme)) then
      lordering_scheme = ordering_scheme
    else
      lordering_scheme = halo_ordering_scheme(l2_halo)
    end if

    nprocs = halo_proc_count(l2_halo)

    ! Paint the sends and receives
    allocate(send_paint(node_count(mesh)))
    call allocate(send_paint)
    allocate(receive_paint(node_count(mesh)))
    receive_paint = 0
    do i = 1, nprocs
      do j = 1, halo_send_count(l2_halo, i)
        call insert(send_paint(halo_send(l2_halo, i, j)), i)
      end do
      receive_paint(halo_receives(l2_halo, i)) = i
    end do

    ! Walk the nnlist from the sends and receives in l2. If we hit a l2 send
    ! one adjacency from a l2 receive, then we have a l1 send. If we hit a l2
    ! receive for process n one adjacency from a l2 send for process n, then we
    ! have a l1 receive for process n.
    nnlist => extract_nnlist(mesh)
    allocate(sends(nprocs))
    call allocate(sends)
    allocate(receives(nprocs))
    call allocate(receives)
    do i = 1, nprocs
      do j = 1, halo_send_count(l2_halo, i)
        neigh => row_m_ptr(nnlist, halo_send(l2_halo, i, j))
        do k = 1, size(neigh)
          proc = receive_paint(neigh(k))
          if(proc > 0) then
            call insert(receives(proc), neigh(k))
          end if
        end do
      end do

      do j = 1, halo_receive_count(l2_halo, i)
        neigh => row_m_ptr(nnlist, halo_receive(l2_halo, i, j))
        do k = 1, size(neigh)
          do l = 1, key_count(send_paint(neigh(k)))
            proc = fetch(send_paint(neigh(k)), l)
            if(proc == i) then
              call insert(sends(proc), neigh(k))
            end if
          end do
        end do
      end do
    end do
    call deallocate(send_paint)
    deallocate(send_paint)
    deallocate(receive_paint)

    ! Generate the l1 halo from the sends and receives sets

    call allocate(l1_halo, &
      & nsends = key_count(sends), &
      & nreceives = key_count(receives), &
      & name = trim(mesh%name) // "Level1Halo", &
      & communicator = halo_communicator(l2_halo), &
      & nowned_nodes = halo_nowned_nodes(l2_halo), &
      & data_type = HALO_TYPE_CG_NODE, &
      & ordering_scheme = lordering_scheme)

    assert(valid_halo_node_counts(l1_halo))

    do i = 1, nprocs
      call set_halo_sends(l1_halo, i, set2vector(sends(i)))
      call deallocate(sends(i))

      call set_halo_receives(l1_halo, i, set2vector(receives(i)))
      call deallocate(receives(i))
    end do
    deallocate(sends)
    deallocate(receives)

    call reorder_l1_from_l2_halo(l1_halo, l2_halo, sorted_l1_halo = .true.)

#ifdef DDEBUG
    if(halo_ordering_scheme(l1_halo) == HALO_ORDER_TRAILING_RECEIVES) then
      assert(trailing_receives_consistent(l1_halo))
    end if
#endif

    if(.not. present_and_false(create_caches)) then
      ! Create caches
      call create_global_to_universal_numbering(l1_halo)
      call create_ownership(l1_halo)
    end if

  end function derive_l1_from_l2_halo_halo

  subroutine derive_element_halo_from_node_halo(mesh, ordering_scheme, create_caches)
    !!< For the supplied mesh, generate element halos from the level 2 node halo

    type(mesh_type), intent(inout) :: mesh
    !! By default the element halos will have a HALO_ORDER_GENERAL ordering
    !! scheme. Supply this to override.
    integer, optional, intent(in) :: ordering_scheme
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

#ifdef DDEBUG
    integer :: i

    assert(halo_count(mesh) > 0)
    assert(has_references(mesh%halos(halo_count(mesh))))
    assert(element_halo_count(mesh) <= 2)
    do i = 1, element_halo_count(mesh)
      assert(.not. has_references(mesh%element_halos(i)))
    end do
#endif

    mesh%element_halos(1) = derive_maximal_element_halo(mesh, mesh%halos(halo_count(mesh)), &
      & ordering_scheme = ordering_scheme, create_caches = create_caches)
    if(element_halo_count(mesh) > 1) then
      mesh%element_halos(2) = mesh%element_halos(1)
      call incref(mesh%element_halos(2))
    end if

  end subroutine derive_element_halo_from_node_halo

  function derive_maximal_element_halo(mesh, node_halo, ordering_scheme, create_caches) result(element_halo)
    !!< Given a node halo for the supplied mesh, derive the maximal element halo

    type(mesh_type), intent(in) :: mesh
    type(halo_type), intent(in) :: node_halo
    !! By default the maximal element halo will have a HALO_ORDER_GENERAL
    !! ordering scheme. Supply this to override.
    integer, optional, intent(in) :: ordering_scheme
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

    type(halo_type) :: element_halo

    integer :: communicator, i, lordering_scheme, nowned_eles, nprocs, &
      & owner, procno
    type(integer_set), dimension(:), allocatable :: receives

    ewrite(1, *) "In derive_maximal_element_halo"

    assert(continuity(mesh) == 0)
    assert(halo_data_type(node_halo) == HALO_TYPE_CG_NODE)
    assert(.not. serial_storage_halo(node_halo))

    if(present(ordering_scheme)) then
      lordering_scheme = ordering_scheme
    else
      lordering_scheme = HALO_ORDER_GENERAL
    end if

    communicator = halo_communicator(node_halo)
    nprocs = halo_proc_count(node_halo)
    procno = getprocno(communicator = communicator)

    ! Step 1: Generate the maximal set of receives

    allocate(receives(nprocs))
    call allocate(receives)
    do i = 1, ele_count(mesh)
      owner = ele_owner(i, mesh, node_halo)
      if(owner /= procno) call insert(receives(owner), i)
    end do
    ewrite(2, *) "Maximal receive elements: ", sum(key_count(receives))

    nowned_eles = ele_count(mesh) - sum(key_count(receives))
    ewrite(2, *) "Owned elements: ", nowned_eles

    ! Step 2: Allocate the halo and set the receives

    call allocate(element_halo, &
      & nsends = spread(0, 1, nprocs), &
      & nreceives = key_count(receives), &
      & name = trim(mesh%name) // "MaximalElementHalo", &
      & communicator = communicator, &
      & nowned_nodes = nowned_eles, &
      & data_type = HALO_TYPE_ELEMENT, &
      & ordering_scheme = lordering_scheme)

    do i = 1, nprocs
      call set_halo_receives(element_halo, i, set2vector(receives(i)))
      call deallocate(receives(i))
    end do
    deallocate(receives)

    ! Step 3: Invert the receives to form the sends
    call invert_element_halo_receives(mesh, node_halo, element_halo)

#ifdef DDEBUG
    if(halo_ordering_scheme(element_halo) == HALO_ORDER_TRAILING_RECEIVES) then
      assert(trailing_receives_consistent(element_halo))
    end if
#endif

    if(.not. present_and_false(create_caches)) then
      call create_global_to_universal_numbering(element_halo)
      call create_ownership(element_halo)
    end if

    ewrite(1, *) "Exiting derive_maximal_element_halo"

  end function derive_maximal_element_halo

  function derive_maximal_surface_element_halo(mesh, element_halo, ordering_scheme, create_caches) result(selement_halo)
    !!< Given an element halo for the supplied mesh, derive the maximal surface
    !!< element halo

    type(mesh_type), intent(inout) :: mesh
    type(halo_type), intent(in) :: element_halo
    !! By default the maximal surface element halo will have a
    !! HALO_ORDER_GENERAL ordering scheme. Supply this to override.
    integer, optional, intent(in) :: ordering_scheme
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

    type(halo_type) :: selement_halo

    integer :: communicator, i, lordering_scheme, nowned_eles, nprocs, &
      & owner, procno
    type(integer_set), dimension(:), allocatable :: receives

    ewrite(1, *) "In derive_maximal_surface_element_halo"

    assert(continuity(mesh) == 0)
    assert(halo_data_type(element_halo) == HALO_TYPE_ELEMENT)
    assert(.not. serial_storage_halo(element_halo))

    if(present(ordering_scheme)) then
      lordering_scheme = ordering_scheme
    else
      lordering_scheme = HALO_ORDER_GENERAL
    end if

    communicator = halo_communicator(element_halo)
    nprocs = halo_proc_count(element_halo)
    procno = getprocno(communicator = communicator)

    ! Step 1: Generate the maximal set of receives

    allocate(receives(nprocs))
    call allocate(receives)
    do i = 1, surface_element_count(mesh)
      owner = halo_node_owner(element_halo, face_ele(mesh, i))
      if(owner /= procno) call insert(receives(owner), i)
    end do
    ewrite(2, *) "Maximal receive elements: ", sum(key_count(receives))

    nowned_eles = surface_element_count(mesh) - sum(key_count(receives))
    ewrite(2, *) "Owned elements: ", nowned_eles

    ! Step 2: Allocate the halo and set the receives

    call allocate(selement_halo, &
      & nsends = spread(0, 1, nprocs), &
      & nreceives = key_count(receives), &
      & name = trim(mesh%name) // "MaximalSurfaceElementHalo", &
      & communicator = communicator, &
      & nowned_nodes = nowned_eles, &
      & data_type = HALO_TYPE_ELEMENT, &
      & ordering_scheme = lordering_scheme)

    do i = 1, nprocs
      call set_halo_receives(selement_halo, i, set2vector(receives(i)))
      call deallocate(receives(i))
    end do
    deallocate(receives)

    ! Step 3: Invert the receives to form the sends
    call invert_surface_element_halo_receives(mesh, element_halo, selement_halo)

#ifdef DDEBUG
    if(halo_ordering_scheme(selement_halo) == HALO_ORDER_TRAILING_RECEIVES) then
      assert(trailing_receives_consistent(selement_halo))
    end if
#endif

    if(.not. present_and_false(create_caches)) then
      call create_global_to_universal_numbering(selement_halo)
      call create_ownership(selement_halo)
    end if

    ewrite(1, *) "Exiting derive_maximal_surface_element_halo"

  end function derive_maximal_surface_element_halo

  subroutine derive_element_halos_from_l2_halo_mesh(mesh, create_caches)
    !!< For the supplied mesh, generate element halos from the level 2 node
    !!< halo

    type(mesh_type), intent(inout) :: mesh
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

#ifdef DDEBUG
    assert(halo_count(mesh) >= 2)
    assert(has_references(mesh%halos(2)))
    assert(element_halo_count(mesh) >= 2)
    assert(.not. has_references(mesh%element_halos(1)))
    assert(.not. has_references(mesh%element_halos(2)))
#endif

    call derive_element_halos_from_l2_halo(mesh, mesh%halos(2), mesh%element_halos(:2), &
      & create_caches = create_caches)

  end subroutine derive_element_halos_from_l2_halo_mesh

  subroutine derive_element_halos_from_l2_halo_halo(mesh, l2_halo, element_halos, ordering_scheme, create_caches)
    !!< Given a level 2 node halo for the supplied mesh, derive element halos.
    !!< We do this by:
    !!<  1. Determining which elements we don't own, and who does own them, to
    !!<     generate a maximum set of receives
    !!<  2. Walking the eelist from the maximum set of receives to form the l1
    !!<     and l2 sends
    !!<  3. Walking the eelist from the l1 sends to form the l1 and l2 receives

    type(mesh_type), intent(in) :: mesh
    type(halo_type), intent(in) :: l2_halo
    type(halo_type), dimension(2), intent(out) :: element_halos
    !! By default the element halos will have HALO_ORDER_GENERAL
    !! ordering schemes. Supply this to override.
    integer, optional, intent(in) :: ordering_scheme
    !! If present and .false., do not create halo caches
    logical, optional, intent(in) :: create_caches

    integer :: communicator, ele1, ele2, ele3, i, j, k, l, lordering_scheme, &
      & nowned_eles, nprocs, owner, procno
    integer, dimension(:), pointer :: neigh1, neigh2
    type(csr_sparsity), pointer :: eelist
    type(integer_set), dimension(:), allocatable :: l1_receives, l1_sends, &
      & l2_receives, l2_sends

    ewrite(1, *) "In derive_element_halos_from_l2_halo_halo"

    assert(continuity(mesh) == 0)
    assert(halo_data_type(l2_halo) == HALO_TYPE_CG_NODE)
    assert(.not. serial_storage_halo(l2_halo))

    if(present(ordering_scheme)) then
      lordering_scheme = ordering_scheme
    else
      lordering_scheme = HALO_ORDER_GENERAL
    end if

    communicator = halo_communicator(l2_halo)
    nprocs = halo_proc_count(l2_halo)
    procno = getprocno(communicator = communicator)

    ! Step 1: Generate the ownership boundary (stored as the maximum set of
    ! receives)

    allocate(l2_receives(nprocs))
    call allocate(l2_receives)
    do i = 1, ele_count(mesh)
      owner = ele_owner(i, mesh, l2_halo)
      if(owner /= procno) call insert(l2_receives(owner), i)
    end do

    ! l2_receives is now a superset of the l2 element halo receives
    ! This gives us the number of owned elements:
    nowned_eles = ele_count(mesh) - sum(key_count(l2_receives))
    ewrite(2, *) "Owned elements: ", nowned_eles

    ! Step 2: Walk the eelist from the element ownership boundary to generate
    ! the send elements

    eelist => extract_eelist(mesh)

    allocate(l1_sends(nprocs))
    call allocate(l1_sends)
    allocate(l2_sends(nprocs))
    call allocate(l2_sends)
    do i = 1, nprocs
      do j = 1, key_count(l2_receives(i))
        ele1 = fetch(l2_receives(i), j)
        neigh1 => row_m_ptr(eelist, ele1)
        do k = 1, size(neigh1)
          ele2 = neigh1(k)
          if(ele2 <= 0) cycle
          if(ele_owner(ele2, mesh, l2_halo) /= procno) cycle

          ! This is a level 1 send element
          call insert(l1_sends(i), ele2)
          call insert(l2_sends(i), ele2)

          neigh2 => row_m_ptr(eelist, ele2)
          do l = 1, size(neigh2)
            ele3 = neigh2(l)
            if(ele3 <= 0) cycle
            if(ele_owner(ele3, mesh, l2_halo) /= procno) cycle

            ! This is a level 2 send element
            call insert(l2_sends(i), ele3)
          end do
        end do
      end do
    end do
    call deallocate(l2_receives)
    ewrite(2, *) "Level 1 send elements: ", sum(key_count(l1_sends))
    ewrite(2, *) "Level 2 send elements: ", sum(key_count(l2_sends))

    ! Step 3: Walk the eelist from the l1 send elements to generate the receive
    ! elements

    allocate(l1_receives(nprocs))
    call allocate(l1_receives)
    call allocate(l2_receives)
    do i = 1, nprocs
      do j = 1, key_count(l1_sends(i))
        ele1 = fetch(l1_sends(i), j)
        neigh1 => row_m_ptr(eelist, ele1)
        do k = 1, size(neigh1)
          ele2 = neigh1(k)
          if(ele2 <= 0) cycle
          if(ele_owner(ele2, mesh, l2_halo) /= i) cycle

          ! This is a level 1 receive element
          call insert(l1_receives(i), ele2)
          call insert(l2_receives(i), ele2)

          neigh2 => row_m_ptr(eelist, ele2)
          do l = 1, size(neigh2)
            ele3 = neigh2(l)
            if(ele3 <= 0) cycle
            if(ele_owner(ele3, mesh, l2_halo) /= i) cycle

            ! This is a level 2 receive element
            call insert(l2_receives(i), ele3)
          end do
        end do
      end do
    end do
    ewrite(2, *) "Level 1 receive elements: ", key_count(l1_receives)
    ewrite(2, *) "Level 2 receive elements: ", key_count(l2_receives)

    ! Step 4: Generate the element halos from the sends and receives sets

    call allocate(element_halos(1), &
      & nsends = key_count(l1_sends), &
      & nreceives = key_count(l1_receives), &
      & name = trim(mesh%name) // "Level1ElementHalo", &
      & communicator = communicator, &
      & nowned_nodes = nowned_eles, &
      & data_type = HALO_TYPE_ELEMENT, &
      & ordering_scheme = lordering_scheme)

    call allocate(element_halos(2), &
      & nsends = key_count(l2_sends), &
      & nreceives = key_count(l2_receives), &
      & name = trim(mesh%name) // "Level2ElementHalo", &
      & communicator = communicator, &
      & nowned_nodes = nowned_eles, &
      & data_type = HALO_TYPE_ELEMENT, &
      & ordering_scheme = lordering_scheme)

    assert(valid_halo_node_counts(element_halos(1)))
    assert(valid_halo_node_counts(element_halos(2)))

    do i = 1, nprocs
      call set_halo_sends(element_halos(1), i, set2vector(l1_sends(i)))
      call deallocate(l1_sends(i))

      call set_halo_sends(element_halos(2), i, set2vector(l2_sends(i)))
      call deallocate(l2_sends(i))

      call set_halo_receives(element_halos(1), i, set2vector(l1_receives(i)))
      call deallocate(l1_receives(i))

      call set_halo_receives(element_halos(2), i, set2vector(l2_receives(i)))
      call deallocate(l2_receives(i))
    end do
    deallocate(l1_sends)
    deallocate(l1_receives)
    deallocate(l2_sends)
    deallocate(l2_receives)

    call reorder_element_halo(element_halos(1), l2_halo, mesh)
    call reorder_element_halo(element_halos(2), l2_halo, mesh)

    if(.not. present_and_false(create_caches)) then
      ! Create caches
      call create_global_to_universal_numbering(element_halos(1))
      call create_global_to_universal_numbering(element_halos(2))
      call create_ownership(element_halos(1))
      call create_ownership(element_halos(2))
    end if

    ewrite(1, *) "Exiting derive_element_halos_from_l2_halo_halo"

  end subroutine derive_element_halos_from_l2_halo_halo

  function ele_owner(ele, mesh, node_halo)
    !!< Use the node halo node_halo, associated with mesh mesh, to determine a
    !!< universally unique owner for element ele in mesh

    integer, intent(in) :: ele
    type(mesh_type), intent(in) :: mesh
    type(halo_type), intent(in) :: node_halo

    integer :: ele_owner

    ele_owner = maxval(halo_node_owners(node_halo, ele_nodes(mesh, ele)))
    assert(ele_owner > 0)

  end function ele_owner

  function invert_comms_sizes(knowns_sizes, communicator) result(unknowns_sizes)
    !!< Given a set of knowns with size knowns_sizes (e.g. halo receive node
    !!< counts), form the inverse (the halo send node counts)

    integer, dimension(:), intent(in) :: knowns_sizes
    integer, optional, intent(in) :: communicator

    integer, dimension(size(knowns_sizes)) :: unknowns_sizes

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if
    assert(size(knowns_sizes) == getnprocs(communicator = lcommunicator))

    call mpi_alltoall(knowns_sizes, 1, getpinteger(), unknowns_sizes, 1, getpinteger(), communicator, ierr)
    assert(ierr == MPI_SUCCESS)
#else
    FLAbort("invert_comms_sizes cannot be called without MPI support")
#endif

  end function invert_comms_sizes

  subroutine invert_element_halo_receives(mesh, node_halo, element_halo)
    !!< Invert an element halo receives to form the element halo sends, using
    !!< the unn cache of a node halo

    type(mesh_type), intent(in) :: mesh
    type(halo_type), intent(in) :: node_halo
    type(halo_type), intent(inout) :: element_halo

#ifdef HAVE_MPI
    ! No mixed mesh support here
    integer :: loc
    integer :: communicator, ele, i, ierr, j, nprocs, rank
    integer, dimension(ele_loc(mesh, 1)) :: nodes
    integer, dimension(:), allocatable :: nreceives, nsends, requests, statuses
    type(integer_hash_table) :: gnns
    type(integer_vector), dimension(:), allocatable :: receives_uenlist, sends_uenlist
    integer tag

    assert(continuity(mesh) == 0)
    assert(halo_data_type(node_halo) == HALO_TYPE_CG_NODE)
    assert(halo_data_type(element_halo) == HALO_TYPE_ELEMENT)
    assert(has_global_to_universal_numbering(node_halo))
    assert(.not. has_global_to_universal_numbering(element_halo))

    communicator = halo_communicator(node_halo)
    nprocs = halo_proc_count(node_halo)
    rank = getrank(communicator)

    allocate(nsends(nprocs))
    allocate(nreceives(nprocs))
    call halo_receive_counts(element_halo, nreceives)
    nsends = invert_comms_sizes(nreceives, communicator = communicator)
    deallocate(nreceives)
    call reallocate(element_halo, nsends = nsends)
    deallocate(nsends)

    assert(valid_halo_node_counts(element_halo))

    loc = mesh%shape%ndof

    allocate(receives_uenlist(nprocs))
    do i = 1, nprocs
      allocate(receives_uenlist(i)%ptr(halo_receive_count(element_halo, i)*loc))
      do j = 1, halo_receive_count(element_halo, i)
        receives_uenlist(i)%ptr((j - 1) * loc + 1:j * loc) = halo_universal_numbers(node_halo, ele_nodes(mesh, halo_receive(element_halo, i, j)))
      end do
    end do

    ! Set up non-blocking communications
    allocate(sends_uenlist(nprocs))
    allocate(requests(nprocs * 2))
    requests = MPI_REQUEST_NULL
    tag = next_mpi_tag()

    do i = 1, nprocs
      allocate(sends_uenlist(i)%ptr(halo_send_count(element_halo, i)*loc))

      ! Non-blocking sends
      if(size(receives_uenlist(i)%ptr) > 0) then
        call mpi_isend(receives_uenlist(i)%ptr, size(receives_uenlist(i)%ptr), getpinteger(), &
             i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      ! Non-blocking receives
      if(size(sends_uenlist(i)%ptr) > 0) then
        call mpi_irecv(sends_uenlist(i)%ptr, size(sends_uenlist(i)%ptr), getpinteger(), &
             i - 1, tag, communicator, requests(i + nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(statuses)
    deallocate(requests)
    do i = 1, nprocs
      deallocate(receives_uenlist(i)%ptr)
    end do
    deallocate(receives_uenlist)

    call get_universal_numbering_inverse(node_halo, gnns)
    call add_nelist(mesh)
    do i = 1, nprocs
      do j = 1, halo_send_count(element_halo, i)
        nodes = fetch(gnns, sends_uenlist(i)%ptr((j - 1) * loc + 1:j * loc))
        ele = nodes_ele(mesh, nodes)
        call set_halo_send(element_halo, i, j, ele)
      end do
      deallocate(sends_uenlist(i)%ptr)
    end do
    call deallocate(gnns)
    deallocate(sends_uenlist)
#else
    FLAbort("invert_element_halo_receives cannot be called without MPI support")
#endif

  end subroutine invert_element_halo_receives

  subroutine invert_surface_element_halo_receives(mesh, element_halo, selement_halo)
    !!< Invert surface element halo receives to form the surface element halo
    !!< sends, using the unn cache of an element halo

    type(mesh_type), intent(in) :: mesh
    type(halo_type), intent(in) :: element_halo
    type(halo_type), intent(inout) :: selement_halo

#ifdef HAVE_MPI
    integer :: communicator, ele, face, i, ierr, j, lface, nprocs, rank
    integer, dimension(:), allocatable :: nreceives, nsends, requests, statuses
    integer, dimension(:), pointer :: faces
    type(integer_hash_table) :: gnns
    type(integer_vector), dimension(:), allocatable :: receives_uenlist, sends_uenlist
    integer tag

    assert(continuity(mesh) == 0)
    assert(halo_data_type(element_halo) == HALO_TYPE_ELEMENT)
    assert(halo_data_type(selement_halo) == HALO_TYPE_ELEMENT)
    assert(has_global_to_universal_numbering(element_halo))
    assert(.not. has_global_to_universal_numbering(selement_halo))

    communicator = halo_communicator(element_halo)
    nprocs = halo_proc_count(element_halo)
    rank = getrank(communicator)

    allocate(nsends(nprocs))
    allocate(nreceives(nprocs))
    call halo_receive_counts(selement_halo, nreceives)
    nsends = invert_comms_sizes(nreceives, communicator = communicator)
    deallocate(nreceives)
    call reallocate(selement_halo, nsends = nsends)
    deallocate(nsends)

    assert(valid_halo_node_counts(selement_halo))

    allocate(receives_uenlist(nprocs))
    do i = 1, nprocs
      allocate(receives_uenlist(i)%ptr(halo_receive_count(selement_halo, i)*2))
      do j = 1, halo_receive_count(selement_halo, i)
        face = halo_receive(selement_halo, i, j)
        receives_uenlist(i)%ptr((j - 1) * 2 + 1) = halo_universal_number(element_halo, face_ele(mesh, face))
        receives_uenlist(i)%ptr((j - 1) * 2 + 2) = local_face_number(mesh, face)
      end do
    end do

    ! Set up non-blocking communications
    allocate(sends_uenlist(nprocs))
    allocate(requests(nprocs * 2))
    requests = MPI_REQUEST_NULL
    tag = next_mpi_tag()

    do i = 1, nprocs
      allocate(sends_uenlist(i)%ptr(halo_send_count(selement_halo, i)*2))

      ! Non-blocking sends
      if(size(receives_uenlist(i)%ptr) > 0) then
        call mpi_isend(receives_uenlist(i)%ptr, size(receives_uenlist(i)%ptr), getpinteger(), &
             i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      ! Non-blocking receives
      if(size(sends_uenlist(i)%ptr) > 0) then
        call mpi_irecv(sends_uenlist(i)%ptr, size(sends_uenlist(i)%ptr), getpinteger(), &
             i - 1, tag, communicator, requests(i + nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(statuses)
    deallocate(requests)
    do i = 1, nprocs
      deallocate(receives_uenlist(i)%ptr)
    end do
    deallocate(receives_uenlist)

    call get_universal_numbering_inverse(element_halo, gnns)
    do i = 1, nprocs
      do j = 1, halo_send_count(selement_halo, i)
        ele = fetch(gnns, sends_uenlist(i)%ptr((j - 1) * 2 + 1))
        lface = sends_uenlist(i)%ptr((j - 1) * 2 + 2)
        faces => ele_faces(mesh, ele)
        face = faces(lface)
        assert(face > 0)
        call set_halo_send(selement_halo, i, j, face)
      end do
      deallocate(sends_uenlist(i)%ptr)
    end do
    call deallocate(gnns)
    deallocate(sends_uenlist)
#else
    FLAbort("invert_surface_element_halo_receives cannot be called without MPI support")
#endif

  end subroutine invert_surface_element_halo_receives

  function nodes_ele(mesh, nodes) result(ele)
    !!< Inverse of ele_nodes

    type(mesh_type), intent(in) :: mesh
    integer, dimension(:), intent(in) :: nodes

    integer :: ele

    integer :: i, j
    integer, dimension(:), pointer :: element_nodes, eles

    assert(size(nodes) > 0)
    assert(associated(mesh%adj_lists))
    assert(associated(mesh%adj_lists%nelist))
    eles => row_m_ptr(mesh%adj_lists%nelist, nodes(1))
    eles_loop: do i = 1, size(eles)
      element_nodes => ele_nodes(mesh, eles(i))
      if(size(element_nodes) /= size(nodes)) cycle eles_loop
      do j = 1, size(nodes)
        if(.not. any(element_nodes == nodes(j))) cycle eles_loop
      end do

      ele = eles(i)
      return
    end do eles_loop

    ewrite(-1, *) "For nodes ", nodes
    FLAbort("Failed to find element")

  end function nodes_ele

  function invert_comms_global(halo, knowns) result(unknowns)
    !!< Given a set of knowns global numbers (e.g. halo receive nodes)
    !!< for the supplied halo, form the inverse (the halo send nodes)

    type(halo_type), intent(in) :: halo
    type(integer_set), dimension(:), intent(in) :: knowns

    type(integer_set), dimension(size(knowns)) :: unknowns

    integer :: i, j
    type(integer_hash_table) :: gnns
    type(integer_set), dimension(size(knowns)) :: unn_knowns, unn_unknowns

    call allocate(unn_knowns)
    do i = 1, size(knowns)
      do j = 1, key_count(knowns(i))
        call insert(unn_knowns(i), halo_universal_number(halo, fetch(knowns(i), j)))
      end do
    end do

    unn_unknowns = invert_comms(unn_knowns, communicator = halo_communicator(halo))
    call deallocate(unn_knowns)

    call get_universal_numbering_inverse(halo, gnns)

    call allocate(unknowns)
    do i = 1, size(unn_unknowns)
      do j = 1, key_count(unn_unknowns(i))
        call insert(unknowns(i), fetch(gnns, fetch(unn_unknowns(i), j)))
      end do
      call deallocate(unn_unknowns(i))
    end do
    call deallocate(gnns)

  end function invert_comms_global

  function invert_comms(knowns, communicator) result(unknowns)
#ifdef HAVE_ZOLTAN
    use zoltan
#endif

    type(integer_set), dimension(:), intent(in) :: knowns
    type(integer_set), dimension(size(knowns)) :: unknowns ! we'll actually know them by the end, but however
    integer, intent(in), optional :: communicator

#ifdef HAVE_ZOLTAN
    type(zoltan_struct), pointer :: zz
    integer(zoltan_int) :: ierr

    integer :: num_known, num_unknown
    integer, dimension(:), pointer :: known_global_ids, known_local_ids, known_procs, known_to_part
    integer, dimension(:), pointer :: unknown_global_ids, unknown_local_ids, unknown_procs, unknown_to_part

    integer :: i, head

    ! Set up zoltan structure
    if (present(communicator)) then
      zz => Zoltan_Create(communicator)
    else
      zz => Zoltan_Create(MPI_COMM_FEMTOOLS)
    end if

    ! Convert the sets to zoltan's input (a bunch of arrays)
    unknown_global_ids => null()
    unknown_local_ids  => null()
    unknown_procs      => null()
    unknown_to_part    => null()

    num_known = sum(key_count(knowns))
    allocate(known_global_ids(num_known))
    allocate(known_local_ids(num_known))
    allocate(known_procs(num_known))
    known_to_part => null()
    known_local_ids = -666 ! in my reading of the zoltan docs, you shouldn't need this, but however

    ! Set known_global_ids and known_procs
    head = 1
    do i=1,size(knowns)
      known_global_ids(head:head + key_count(knowns(i)) - 1) = set2vector(knowns(i))
      known_procs(head:head + key_count(knowns(i)) - 1) = i - 1
    end do

    ! Call Zoltan_Compute_Destinations
    ierr = Zoltan_Compute_Destinations(zz, &
     & num_known, known_global_ids, known_local_ids, known_procs, &
     & num_unknown, unknown_global_ids, unknown_local_ids, unknown_procs)
    assert(ierr == ZOLTAN_OK)

    ! Set unknowns
    call allocate(unknowns)
    do i=1,num_unknown
      call insert(unknowns(unknown_procs(i) + 1), unknown_global_ids(i))
    end do

    ! Deallocate

    ierr = Zoltan_LB_Free_Part(unknown_global_ids, unknown_local_ids, unknown_procs, unknown_to_part)
    assert(ierr == ZOLTAN_OK)

    deallocate(known_procs)
    deallocate(known_local_ids)
    deallocate(known_global_ids)
    call Zoltan_Destroy(zz)
#else
    call allocate(unknowns)  ! Keep the compiler quiet
    FLAbort("invert_comms called without zoltan support")
#endif

  end function invert_comms

  subroutine derive_nonperiodic_halos_from_periodic_halos(new_positions, model, aliased_to_new_node_number)
    type(vector_field), intent(inout) :: new_positions
    type(vector_field), intent(in) :: model
    type(integer_hash_table), intent(in) :: aliased_to_new_node_number
    type(ilist), dimension(:), allocatable :: sends, receives
    integer :: proc, i, new_nowned_nodes
#ifdef DDEBUG
    integer, dimension(node_count(model)) :: map_verification
    integer :: key, output
#endif

    assert(halo_count(model) == 2)
    assert(element_halo_count(model) == 2)
    assert(halo_valid_for_communication(model%mesh%element_halos(1)))
    assert(halo_valid_for_communication(model%mesh%element_halos(2)))
    assert(halo_valid_for_communication(model%mesh%halos(1)))
    assert(halo_valid_for_communication(model%mesh%halos(2)))
#ifdef DDEBUG
    map_verification = 0
    do i=1,key_count(aliased_to_new_node_number)
      call fetch_pair(aliased_to_new_node_number, i, key, output)
      map_verification(key) = 1
    end do
    assert(halo_verifies(model%mesh%halos(2), map_verification))

    assert(halo_verifies(model%mesh%halos(2), model))
#endif

    ! element halos are easy, just a copy, na ja?
    allocate(new_positions%mesh%element_halos(2))
    new_positions%mesh%element_halos(1) = model%mesh%element_halos(1); call incref(new_positions%mesh%element_halos(1))
    new_positions%mesh%element_halos(2) = model%mesh%element_halos(2); call incref(new_positions%mesh%element_halos(2))

    ! nodal halo: let's compute the l2 nodal halo, then derive the l1
    ! from it
    allocate(new_positions%mesh%halos(2))

    allocate(sends(halo_proc_count(model%mesh%halos(2))))
    allocate(receives(halo_proc_count(model%mesh%halos(2))))
    do proc=1,halo_proc_count(model%mesh%halos(2))

      do i=1,halo_send_count(model%mesh%halos(2), proc)
        call insert(sends(proc), halo_send(model%mesh%halos(2), proc, i))
      end do
      do i=1,halo_receive_count(model%mesh%halos(2), proc)
        call insert(receives(proc), halo_receive(model%mesh%halos(2), proc, i))
      end do

      do i=1,halo_send_count(model%mesh%halos(2), proc)
        if (has_key(aliased_to_new_node_number, halo_send(model%mesh%halos(2), proc, i))) then
          call insert(sends(proc), fetch(aliased_to_new_node_number, halo_send(model%mesh%halos(2), proc, i)))
        end if
      end do
      do i=1,halo_receive_count(model%mesh%halos(2), proc)
        if (has_key(aliased_to_new_node_number, halo_receive(model%mesh%halos(2), proc, i))) then
          call insert(receives(proc), fetch(aliased_to_new_node_number, halo_receive(model%mesh%halos(2), proc, i)))
        end if
      end do
    end do

    new_nowned_nodes = halo_nowned_nodes(model%mesh%halos(2))
    do i=1,node_count(model)
      if (.not. node_owned(model%mesh%halos(2), i)) cycle
      if (has_key(aliased_to_new_node_number, i)) then
        new_nowned_nodes = new_nowned_nodes + 1
      end if
    end do

    call allocate(new_positions%mesh%halos(2), &
                  nsends = sends%length, &
                  nreceives = receives%length, &
                  name = trim(new_positions%mesh%name) // "Level2Halo", &
                  communicator = halo_communicator(model%mesh%halos(2)), &
                  nowned_nodes = new_nowned_nodes, &
                  data_type = halo_data_type(model%mesh%halos(2)), &
                  ordering_scheme = HALO_ORDER_GENERAL)

    do proc=1,halo_proc_count(model%mesh%halos(2))
      call set_halo_sends(new_positions%mesh%halos(2), proc, list2vector(sends(proc)))
      call deallocate(sends(proc))
      call set_halo_receives(new_positions%mesh%halos(2), proc, list2vector(receives(proc)))
      call deallocate(receives(proc))
    end do

    deallocate(sends)
    deallocate(receives)

    assert(halo_valid_for_communication(new_positions%mesh%halos(2)))
    assert(halo_verifies(new_positions%mesh%halos(2), new_positions))

    ! Ideally we would derive, but it fails if the mesh has nodes not associated with
    ! any elements
    !call derive_l1_from_l2_halo_mesh(new_positions%mesh, ordering_scheme = HALO_ORDER_GENERAL, create_caches = .false.)
    new_positions%mesh%halos(1) = new_positions%mesh%halos(2)
    call incref(new_positions%mesh%halos(1))

    assert(halo_valid_for_communication(new_positions%mesh%halos(1)))
    call renumber_positions_trailing_receives(new_positions)
    assert(halo_valid_for_communication(new_positions%mesh%halos(1)))
    assert(halo_valid_for_communication(new_positions%mesh%halos(2)))

    call refresh_topology(new_positions%mesh)

  end subroutine derive_nonperiodic_halos_from_periodic_halos

  function derive_sub_halo(halo, sub_nodes) result (sub_halo)
    !!< Derive a halo that is valid for a problem defined on a subset of the nodes
    !!< of the original problem. A new local numbering (determined by the order
    !!< of the 'sub_nodes' array) of these nodes is formed that is used in the sub-problem.
    !!< Each process decides for itself which nodes of the full mesh will take part in
    !!< the sub-problem, both for owned and receive nodes.

    !!< It is allowed for nodes that are considered to be part of the sub-problem
    !!< by their owner, to be left out on other processes that see this node (receivers). This means
    !!< each receiving process needs to tell its sender which nodes it is interested in for the sub-problem.

    !!< Usage cases:
    !!< - solving equations on only part of a mesh. The subproblem is typically defined by a subset of the
    !!< elements. It is then possible that a node adjacent to an element that forms part of the sub-problem,
    !!< is seen (received) by another process which doesn't know about this element. For that process it
    !!< wouldn't make sense to add it to the sub-problem, as it would not be part of any element in the
    !!< sub-problem that it sees. It is therefore allowed in the sub_halo, to leave this node out as a recv
    !!< node even though it is actually part of the global sub-problem that is solved for.
    !!< - a similar situation arised in the case of solving on (parts of) the surface mesh. A receive node
    !!< may only touch the surface mesh, without any surface elements being known in this process.

    type(halo_type), intent(in):: halo
    integer, dimension(:), intent(in):: sub_nodes
    type(halo_type) :: sub_halo

    type(integer_hash_table):: inverse_map
    type(integer_set):: sub_receives_indices_set
    type(integer_vector),  dimension(:), allocatable:: sub_receives_indices, sub_sends_indices
    integer, dimension(:), allocatable:: full_sends, full_receives
    integer, dimension(:), allocatable:: full_sends_start_indices, full_receives_start_indices
    integer, dimension(:), allocatable:: full_nsends, full_nreceives
    integer, dimension(:), allocatable:: sub_nsends, sub_nreceives
    integer, dimension(:), allocatable:: send_requests
    integer, dimension(:), allocatable:: statuses
    integer, dimension(MPI_STATUS_SIZE):: status
    integer:: nprocs, tag, communicator, ierr
    integer:: start, full_node, sub_node, no_sends, no_recvs, nowned_nodes
    integer:: i, iproc

    ewrite(1,*) "deriving halo for sub-problem"

    nprocs = halo_proc_count(halo)
    tag = next_mpi_tag()
    communicator = halo_communicator(halo)

    ! create inverse map from the full problem to the sub_nodes numbering:
    call invert_set(sub_nodes, inverse_map)

    allocate(full_sends(halo_all_sends_count(halo)))
    allocate(full_receives(halo_all_receives_count(halo)))

    full_sends = 0
    full_receives = 0

    allocate(full_sends_start_indices(nprocs))
    allocate(full_receives_start_indices(nprocs))

    full_sends_start_indices = 0
    full_receives_start_indices = 0

    allocate(full_nsends(nprocs))
    allocate(full_nreceives(nprocs))

    full_nsends = 0
    full_nreceives = 0

    allocate(sub_receives_indices(1:nprocs))
    allocate(sub_sends_indices(1:nprocs))

    allocate(sub_nsends(nprocs))
    allocate(sub_nreceives(nprocs))

    sub_nsends = 0
    sub_nreceives = 0

    call extract_all_halo_sends(halo, full_sends, nsends=full_nsends, start_indices=full_sends_start_indices)
    call extract_all_halo_receives(halo, full_receives, nreceives=full_nreceives, start_indices=full_receives_start_indices)

    ! we only send to processors of which we have receive nodes
    allocate(send_requests(1:count(full_nreceives>0)))
    no_sends = 0

    do iproc = 1, nprocs
      if(full_nreceives(iproc) > 0 ) then

        ! a set of indices in the array of all receive nodes received from iproc, corresponding to the nodes
        ! we want to keep
        call allocate(sub_receives_indices_set)

        ! start+i gives the location in full_receives of the i-th receive node from processor iproc
        start = full_receives_start_indices(iproc)-1

        ! collect receive nodes we want to keep
        do i = 1, full_nreceives(iproc)
           if (has_key(inverse_map, full_receives(start+i))) then
              ! we store i as this is the index in the array of all receive nodes received from iproc
              ! this is what we'll send to iproc, as it will know which send node it corresponds to
              call insert(sub_receives_indices_set, i)
           end if
        end do

        sub_nreceives(iproc) = key_count(sub_receives_indices_set)

        ! convert to array so we can send it off
        allocate( sub_receives_indices(iproc)%ptr(sub_nreceives(iproc)) )
        sub_receives_indices(iproc)%ptr = set2vector(sub_receives_indices_set)
        call deallocate(sub_receives_indices_set)

        no_sends=no_sends+1
        ! Non-blocking send to the sender iproc to tell it which nodes we want to receive
        call mpi_isend(sub_receives_indices(iproc)%ptr, sub_nreceives(iproc), MPI_INTEGER, iproc-1, tag, &
          communicator, send_requests(no_sends), ierr)

      end if

    end do ! iproc

    ! number of messages expected:
    ! only expect a request from processes we have send nodes for
    no_recvs = count(full_nsends>0)
    assert( no_sends==count(full_nreceives>0) )

    ! receive all requests and store in sub_sends_indices

    do i = 1, no_recvs

      ! check for pending messages, only returns when a message is ready to receive:
      call mpi_probe(MPI_ANY_SOURCE, tag, communicator, status, ierr)
      assert(ierr == MPI_SUCCESS)

      ! who did we receive from?
      iproc = status(MPI_SOURCE) + 1

      ! query the size and set up the recv buffer
      call mpi_get_count(status, MPI_INTEGER, sub_nsends(iproc), ierr)
      assert(ierr == MPI_SUCCESS)
      allocate(sub_sends_indices(iproc)%ptr(sub_nsends(iproc)))

      ! because of the mpi_probe, this blocking send is guaranteed to finish
      call mpi_recv(sub_sends_indices(iproc)%ptr, sub_nsends(iproc), MPI_INTEGER, iproc-1, tag, &
        communicator, status, ierr)
      assert(ierr == MPI_SUCCESS)

      assert( full_nsends(iproc)>0 ) ! should only be receiving from processes we are sending to
      assert( sub_nsends(iproc)<=full_nsends(iproc) ) ! should request less than the complete number of sends
      assert( maxval(sub_sends_indices(iproc)%ptr)<=full_nsends(iproc) ) ! requested indices should be within the full list of sends

    end do

    ! we now have all the information to allocate the new sub_halo
    call allocate(sub_halo, sub_nsends, sub_nreceives, name=trim(halo%name) // "SubHalo", &
            & communicator = communicator, data_type = halo%data_type)

    ! now count the number of owned nodes in sub_nodes and store it on the halo
    nowned_nodes = 0
    do i = 1, size(sub_nodes)
      if (node_owned(halo, sub_nodes(i))) then
        nowned_nodes = nowned_nodes + 1
      end if
    end do
    call set_halo_nowned_nodes(sub_halo, nowned_nodes)

    ! store the requested nodes in the send lists of the sub_halo
    do iproc = 1, nprocs
      if(full_nsends(iproc) > 0 ) then
        ! start+i gives the location in full_sends of the i-th node sent to processor iproc
        start = full_sends_start_indices(iproc)-1
        ! loop over the requested nodes one by one
        do i=1, sub_nsends(iproc)
          ! requested node in full numbering:
          full_node = full_sends(start+sub_sends_indices(iproc)%ptr(i))
          ! this will fail if the receiver has requested a node, that is not in our 'sub_nodes' array
          sub_node = fetch(inverse_map, full_node)
          call set_halo_send(sub_halo, iproc, i, sub_node)
        end do
        deallocate(sub_sends_indices(iproc)%ptr)
      end if
    end do

    ! we have to wait here till all isends have finished, so we can safely deallocate the sub_receives_indices
    allocate(statuses(MPI_STATUS_SIZE*no_sends))
    call mpi_waitall(no_sends, send_requests, statuses, ierr)
    deallocate(statuses)

    ! store the receive nodes we've requested in the receive lists of the sub_halo
    do iproc = 1, nprocs
      if(full_nreceives(iproc) > 0 ) then
        ! start+i gives the location in full_receives of the i-th node received from processor iproc
        start = full_receives_start_indices(iproc)-1
        ! loop over the requested nodes one by one
        do i=1, sub_nreceives(iproc)
          ! requested node in full numbering:
          full_node = full_receives(start+sub_receives_indices(iproc)%ptr(i))
          ! this shouldn't fail because of the has_key check where sub_receives_indices was assembled
          sub_node = fetch(inverse_map, full_node)
          call set_halo_receive(sub_halo, iproc, i, sub_node)
        end do
        deallocate(sub_receives_indices(iproc)%ptr)
      end if
    end do

    call deallocate(inverse_map)
    deallocate(sub_nreceives)
    deallocate(sub_nsends)

  end function derive_sub_halo

  function combine_halos(halos, node_maps, name) result (halo_out)
    !!< Combine two or more halos of a number of subproblems into a single
    !!< halo, where the numbering of the dofs (nodes) of each subproblem is
    !!< combined into one single contiguous numbering of the dofs in the combined
    !!< system. This is useful to combine the linear solve of two or more coupled
    !!< problem into a single system (which can then be passed to petsc_solve e.g.)
    !!< The numbering of the combined system should be provided by node_maps(:)
    !!< which for each of the sub-problems, provides a map from the sub-problem
    !!< dof-numbering to the dof-numbering of the combined system. If you need
    !!< the combined halo to be in trailing receive order, you need to choose
    !!< the new numbering accordingly for which the function
    !!< create_combined_numbering_trailing_receives() below can be used.
    type(halo_type) :: halo_out
    type(halo_type), dimension(:), intent(in):: halos
    type(integer_vector), dimension(:), intent(in):: node_maps
    character(len=*), intent(in):: name ! name for the halo

    integer, dimension(halo_proc_count(halos(1))):: nsends, nreceives, combined_nsends, combined_nreceives, &
      send_count, receive_count
    integer:: data_type, communicator, nprocs
    integer:: combined_nowned_nodes
    integer:: ihalo, jproc, k, new_node_no

    communicator = halo_communicator(halos(1))
    data_type = halo_data_type(halos(1))
    nprocs = halo_proc_count(halos(1))
#ifdef DDEBUG
    do ihalo=2, size(halos)
      assert(communicator==halo_communicator(halos(ihalo)))
      assert(data_type==halo_data_type(halos(ihalo)))
      assert(nprocs==halo_proc_count(halos(ihalo)))
    end do
#endif
    assert( size(node_maps)==size(halos) )


    combined_nsends = 0
    combined_nreceives =0
    combined_nowned_nodes =0

    do ihalo=1, size(halos)
      call halo_send_counts(halos(ihalo), nsends)
      combined_nsends = combined_nsends + nsends
      call halo_receive_counts(halos(ihalo), nreceives)
      combined_nreceives = combined_nreceives + nreceives
      combined_nowned_nodes = combined_nowned_nodes + halo_nowned_nodes(halos(ihalo))
    end do

    call allocate(halo_out, combined_nsends, combined_nreceives, name=name, &
      communicator=communicator, data_type=data_type, nowned_nodes=combined_nowned_nodes)

    send_count = 0
    receive_count = 0

    do ihalo=1, size(halos)

      do jproc=1, nprocs

        do k=1, halo_send_count(halos(ihalo), jproc)
          new_node_no = node_maps(ihalo)%ptr(halo_send(halos(ihalo), jproc, k))
          call set_halo_send(halo_out, jproc, send_count(jproc)+k, new_node_no)
        end do
        send_count(jproc) = send_count(jproc) + halo_send_count(halos(ihalo), jproc)

        do k=1, halo_receive_count(halos(ihalo), jproc)
          new_node_no = node_maps(ihalo)%ptr(halo_receive(halos(ihalo), jproc, k))
          call set_halo_receive(halo_out, jproc, receive_count(jproc)+k, new_node_no)
        end do
        receive_count(jproc) = receive_count(jproc) + halo_receive_count(halos(ihalo), jproc)
      end do

    end do

    assert( all(receive_count==combined_nreceives) )
    assert( all(send_count==combined_nsends) )

  end function combine_halos

  function create_combined_numbering_trailing_receives(halos1, halos2) result (node_maps)
    !!< Combine the node numbering of one of more subproblems into a single
    !!< contiguous numbering in such a way that we fulfill all requirements
    !!< of trailing_receives_consistent(). This means we first number all
    !!< owned nodes (first of subsystem 1, then of subsystem2, etc.), followed
    !!< by the receive nodes. We assume here that halos1 and halos2 are based
    !!< on the same numbering of the subsystems, and that the receive
    !!< nodes of halos1 are a subset of those in halos2.
    !!< In order to adhere to the following requirement:
    !!<   max_halo_receive_node(halo) == node_count(halo)
    !!< for both halos1 and halos2, we need to then number the receive nodes
    !!< of halos1 first (for all subsystems), followed by any receive nodes
    !!< that are only in halos2.
    type(halo_type), dimension(:), intent(in):: halos1, halos2
    !! we return an array of node_maps that maps the numbering of each subsystem
    !! to that of the combined system
    type(integer_vector), dimension(size(halos1)):: node_maps

    integer, dimension(:), allocatable :: all_recvs
    integer :: new_node
    integer :: i, j

    ! some rudimentary checks to see the halos adhere to the above
    assert(size(halos1)==size(halos2))
    do i=1, size(halos1)
      assert(halo_nowned_nodes(halos1(i))==halo_nowned_nodes(halos2(i)))
      assert(node_count(halos1(i))<=node_count(halos2(i)))
    end do

    do i=1, size(node_maps)
      allocate(node_maps(i)%ptr(node_count(halos2(i))))
      node_maps(i)%ptr = 0
    end do

    new_node = 1 ! keeps track of the next new node number in the combined numbering

    ! owned nodes
    do i=1, size(halos1)
      do j=1, halo_nowned_nodes(halos1(i))
        node_maps(i)%ptr(j) = new_node
        new_node = new_node + 1
      end do
    end do

    ! receive nodes of halos1
    do i=1, size(halos1)
      allocate(all_recvs(1:halo_all_receives_count(halos1(i))))
      call extract_all_halo_receives(halos1(i), all_recvs)
      do j=1, size(all_recvs)
        node_maps(i)%ptr(all_recvs(j)) = new_node
        new_node = new_node + 1
      end do
      deallocate(all_recvs)
    end do

    ! receive nodes of halos2
    do i=1, size(halos2)
      allocate(all_recvs(1:halo_all_receives_count(halos2(i))))
      call extract_all_halo_receives(halos2(i), all_recvs)
      do j=1, size(all_recvs)
        if (node_maps(i)%ptr(all_recvs(j))==0) then
          ! only include those that weren't recv nodes of halos1 already
          node_maps(i)%ptr(all_recvs(j)) = new_node
          new_node = new_node + 1
        end if
      end do
      deallocate(all_recvs)
    end do

  end function create_combined_numbering_trailing_receives

end module halos_derivation
