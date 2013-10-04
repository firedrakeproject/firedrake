#include "fdebug.h"

module halos_ownership

  use fldebug
  use futils
  use halo_data_types
  use halos_allocates
  use halos_base
  use halos_debug
  use halos_numbering
  use parallel_tools
  use quicksort

  implicit none

  private

  public :: create_ownership, has_ownership, halo_node_owner, &
    & halo_node_owners, node_owned, nodes_owned, get_node_owners, &
    & get_owned_nodes, halo_universal_node_owners

  interface node_owned
     module procedure node_owned_halo
  end interface

  interface nodes_owned
    module procedure nodes_owned_halo
  end interface nodes_owned

contains

  subroutine create_ownership(halo)
    !!< Establish the node ownership on the supplied halo, and cache it on the
    !!< halo

    type(halo_type), intent(inout) :: halo

    if(has_ownership(halo)) return

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        call create_ownership_order_general(halo)
      case(HALO_ORDER_TRAILING_RECEIVES)
        call create_ownership_order_trailing_receives(halo)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine create_ownership

  subroutine create_ownership_order_general(halo)
    type(halo_type), intent(inout) :: halo

    integer :: i

    allocate(halo%owners(node_count(halo)))

    halo%owners = getprocno(communicator = halo_communicator(halo))
    do i = 1, halo_proc_count(halo)
      halo%owners(halo_receives(halo, i)) = i
    end do

  end subroutine create_ownership_order_general

  subroutine create_ownership_order_trailing_receives(halo)
    type(halo_type), intent(inout) :: halo

    integer :: i, nowned_nodes

    assert(trailing_receives_consistent(halo))
    nowned_nodes = halo_nowned_nodes(halo)

    allocate(halo%owners(max_halo_node(halo)-halo%nowned_nodes))
    do i = 1, halo_proc_count(halo)
      assert(all(halo_receives(halo, i) >= nowned_nodes))
      assert(all(halo_receives(halo, i) <= nowned_nodes + size(halo%owners)))
      halo%owners(halo_receives(halo, i) - nowned_nodes) = i
    end do

  end subroutine create_ownership_order_trailing_receives

  pure function has_ownership(halo)
    !!< Return whether the supplied halo has node ownership data

    type(halo_type), intent(in) :: halo

    logical :: has_ownership

    has_ownership = associated(halo%owners)

  end function has_ownership

  function halo_node_owner(halo, node, permit_extended) result(node_owner)
    !!< Return the node owner for the supplied node on the supplied halo

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: node
    !! If present and .true. and the node is not contained in the supplied halo,
    !! return a negative owning process
    logical, optional, intent(in) :: permit_extended

    integer :: node_owner

    assert(node > 0)
    assert(has_ownership(halo))

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        node_owner = halo_node_owner_order_general(halo, node, permit_extended = permit_extended)
      case(HALO_ORDER_TRAILING_RECEIVES)
        node_owner = halo_node_owner_order_trailing_receives(halo, node, permit_extended = permit_extended)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end function halo_node_owner

  function halo_node_owner_order_general(halo, node, permit_extended) result(node_owner)
    type(halo_type), intent(in) :: halo
    integer, intent(in) :: node
    !! If present and .true. and the node is not contained in the supplied halo,
    !! return a negative owning process
    logical, optional, intent(in) :: permit_extended

    integer :: node_owner

    if(present_and_true(permit_extended)) then
      if(node <= size(halo%owners)) then
        node_owner = halo%owners(node)
      else
        node_owner = -1
      end if
    else
      assert(node <= size(halo%owners))
      node_owner = halo%owners(node)
    end if

  end function halo_node_owner_order_general

  function halo_node_owner_order_trailing_receives(halo, node, permit_extended) result(node_owner)
    type(halo_type), intent(in) :: halo
    integer, intent(in) :: node
    !! If present and .true. and the node is not contained in the supplied halo,
    !! return a negative owning process
    logical, optional, intent(in) :: permit_extended

    integer :: node_owner, nowned_nodes

    nowned_nodes = halo_nowned_nodes(halo)
    if(node <= nowned_nodes) then
      node_owner = getprocno(halo_communicator(halo))
    else if(present_and_true(permit_extended)) then
      if(node - nowned_nodes <= size(halo%owners)) then
        node_owner = halo%owners(node - nowned_nodes)
      else
        node_owner = -1
      end if
    else
      assert(node - nowned_nodes <= size(halo%owners))
      node_owner = halo%owners(node - nowned_nodes)
    end if

  end function halo_node_owner_order_trailing_receives

  function halo_node_owners(halo, nodes, permit_extended) result(node_owners)
    !!< Return the node owners for the supplied nodes on the supplied halo

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(in) :: nodes
    !! If present and .true. and a node is not contained in the supplied halo,
    !! return a negative owning process for that node
    logical, optional, intent(in) :: permit_extended

    integer, dimension(size(nodes)) :: node_owners

    integer :: i

    do i = 1, size(nodes)
      node_owners(i) = halo_node_owner(halo, nodes(i), permit_extended)
    end do

  end function halo_node_owners

  function node_owned_halo(halo, node)
    !!< Return whether this process owns the supplied node on the supplied halo

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: node

    logical :: node_owned_halo

    assert(node > 0)

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        node_owned_halo = halo_node_owner(halo, node, permit_extended = .true.) == getprocno(communicator = halo_communicator(halo))
      case(HALO_ORDER_TRAILING_RECEIVES)
        node_owned_halo = node <= halo_nowned_nodes(halo)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end function node_owned_halo

  function nodes_owned_halo(halo, nodes) result(owned)
    !!< Return whether this process owns the supplied nodes on the supplied halo

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(in) :: nodes

    logical, dimension(size(nodes)) :: owned

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        owned = halo_node_owners(halo, nodes, permit_extended = .true.) == getprocno(communicator = halo_communicator(halo))
      case(HALO_ORDER_TRAILING_RECEIVES)
        owned = nodes <= halo_nowned_nodes(halo)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end function nodes_owned_halo

  subroutine get_node_owners(halo, owners)
    !!< For the supplied halo, retreive the complete node ownership list

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(out) :: owners

    assert(has_ownership(halo))

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        assert(size(owners) == size(halo%owners))

        owners = halo%owners
      case(HALO_ORDER_TRAILING_RECEIVES)
        assert(size(owners) == halo_nowned_nodes(halo) + size(halo%receives_gnn_to_unn))

        owners(:halo_nowned_nodes(halo)) = getprocno(halo_communicator(halo))
        owners(halo_nowned_nodes(halo) + 1:) = halo%owners
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine get_node_owners

  subroutine get_owned_nodes(halo, owned_nodes)
    !!< Retrieve the owned nodes for the supplied halo

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(out) :: owned_nodes

    integer :: i, index

    assert(size(owned_nodes) == halo_nowned_nodes(halo))

    select case(halo_ordering_scheme(halo))
      case(HALO_ORDER_GENERAL)
        index = 1
        do i = 1, size(halo%owners)
          if(node_owned(halo, i)) then
            assert(index <= size(owned_nodes))
            owned_nodes(index) = i
            index = index + 1
          end if
        end do
        assert(index == size(owned_nodes) + 1)
      case(HALO_ORDER_TRAILING_RECEIVES)
        owned_nodes = (/(i, i = 1, halo_nowned_nodes(halo))/)
      case default
        FLAbort("Unrecognised halo ordering scheme")
    end select

  end subroutine get_owned_nodes

  function halo_universal_node_owners(halo, unns) result(node_owners)
    !!< Return the node owners for the supplied universal nodes on the supplied
    !!< halo.

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(in) :: unns

    integer, dimension(size(unns)) :: node_owners

    integer :: i, nprocs, proc
    integer, dimension(:), allocatable :: permutation

    assert(has_global_to_universal_numbering(halo))
    assert(associated(halo%owned_nodes_unn_base))
    assert(size(halo%owned_nodes_unn_base) == nprocs + 1)

    nprocs = halo_proc_count(halo)
    assert(all(unns > 0) .and. all(unns < halo%owned_nodes_unn_base(nprocs + 1)))

    ! We could add an optimisation for when the caller promises to supply
    ! sorted unns
    allocate(permutation(size(unns)))
    call qsort(unns, permutation)

#ifdef DDEBUG
    node_owners = -1
#endif

    proc = 1
    assert(nprocs > 0)
    unns_loop: do i = 1, size(unns)
      do while(unns(permutation(i)) > halo%owned_nodes_unn_base(proc))
        proc = proc + 1
        if(proc > nprocs) exit unns_loop
      end do
      node_owners(permutation(i)) = proc
    end do unns_loop

    deallocate(permutation)

#ifdef DDEBUG
    assert(all(node_owners > 0))
#endif

  end function halo_universal_node_owners

end module halos_ownership
