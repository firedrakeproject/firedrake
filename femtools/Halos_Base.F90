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

module halos_base

  use fldebug
  use halo_data_types
  use mpi_interfaces
  use quicksort
  use parallel_tools

  implicit none

  private

  public :: zero, halo_name, set_halo_name, halo_data_type, &
    & set_halo_data_type, halo_ordering_scheme, set_halo_ordering_scheme, &
    & has_nowned_nodes, halo_nowned_nodes, set_halo_nowned_nodes, halo_communicator, &
    & set_halo_communicator, halo_proc_count, halo_send_count, &
    & halo_receive_count, halo_unique_receive_count, halo_send, halo_receive, &
    & set_halo_send, set_halo_receive, halo_send_counts, halo_receive_counts, &
    & halo_sends, halo_receives, set_halo_sends, set_halo_receives, &
    & halo_all_sends_count, halo_all_receives_count, &
    & halo_all_unique_receives_count, extract_all_halo_sends, &
    & extract_all_halo_receives, set_all_halo_sends, set_all_halo_receives, &
    & min_halo_send_node, min_halo_receive_node, min_halo_node, &
    & max_halo_send_node, max_halo_receive_node, max_halo_node,&
    & node_count, serial_storage_halo

  interface zero
    module procedure zero_halo
  end interface zero

  interface halo_communicator
    module procedure halo_communicator_halo
  end interface halo_communicator

  interface node_count
    module procedure node_count_halo
  end interface node_count

  interface serial_storage_halo
    module procedure serial_storage_halo_single, serial_storage_halo_multiple
  end interface serial_storage_halo

contains

  subroutine zero_halo(halo)
    !!< Zero the sends and receives information for the supplied halo

    type(halo_type), intent(inout) :: halo

    integer :: i

    assert(associated(halo%sends))
    assert(associated(halo%receives))
    do i = 1, halo_proc_count(halo)
      assert(associated(halo%sends(i)%ptr))
      assert(associated(halo%receives(i)%ptr))
      halo%sends(i)%ptr = 0
      halo%receives(i)%ptr = 0
    end do

  end subroutine zero_halo

  pure function halo_name(halo)
    !!< Retrieve the name of the supplied halo

    type(halo_type), intent(in) :: halo

    character(len = len_trim(halo%name)) :: halo_name

    halo_name = halo%name

  end function halo_name

  subroutine set_halo_name(halo, name)
    !!< Set the name of the supplied halo

    type(halo_type), intent(inout) :: halo
    character(len = *), intent(in) :: name

    halo%name = trim(name)

  end subroutine set_halo_name

  pure function halo_data_type(halo)
    !!< Return the data type of the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: halo_data_type

    halo_data_type = halo%data_type

  end function halo_data_type

  subroutine set_halo_data_type(halo, data_type)
    !!< Set the data type of the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: data_type

#ifdef DDEBUG
    logical:: data_type_valid

    data_type_valid=any(data_type == (/HALO_TYPE_CG_NODE, &
         HALO_TYPE_DG_NODE, &
         HALO_TYPE_ELEMENT/))

    assert(data_type_valid)
#endif

    halo%data_type = data_type

  end subroutine set_halo_data_type

  pure function halo_ordering_scheme(halo)
    !!< Return the ordering scheme of the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: halo_ordering_scheme

    halo_ordering_scheme = halo%ordering_scheme

  end function halo_ordering_scheme

  subroutine set_halo_ordering_scheme(halo, ordering_scheme)
    !!< Set the ordering scheme of the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: ordering_scheme

    assert(any(ordering_scheme == (/HALO_ORDER_GENERAL, HALO_ORDER_TRAILING_RECEIVES/)))
    halo%ordering_scheme = ordering_scheme

  end subroutine set_halo_ordering_scheme

  pure function has_nowned_nodes(halo)
    !!< Return whether the supplied halo has a number of owned nodes set

    type(halo_type), intent(in) :: halo

    logical :: has_nowned_nodes

    has_nowned_nodes = halo%nowned_nodes >= 0

  end function has_nowned_nodes

  pure function halo_nowned_nodes(halo)
    !!< Retrieve the number of owned nodes for the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: halo_nowned_nodes

    halo_nowned_nodes = halo%nowned_nodes

  end function halo_nowned_nodes

  subroutine set_halo_nowned_nodes(halo, nowned_nodes)
    !!< Set the number of owned nodes for the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: nowned_nodes

    assert(nowned_nodes >= 0)

    halo%nowned_nodes = nowned_nodes

  end subroutine set_halo_nowned_nodes

  pure function node_count_halo(halo) result(node_count)
    !!< Retrieve the total number of nodes in the supplied halo.

    type(halo_type), intent(in) :: halo

    integer :: node_count

    node_count = halo_nowned_nodes(halo) &
         + halo_all_receives_count(halo)

  end function node_count_halo

  pure function halo_communicator_halo(halo) result(communicator)
    !!< Extract the halo MPI communicator

    type(halo_type), intent(in) :: halo

    integer :: communicator

    communicator = halo%communicator

  end function halo_communicator_halo

  subroutine set_halo_communicator(halo, communicator)
    !!< Set the halo MPI communicator

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: communicator

    assert(valid_communicator(communicator))
    assert(getnprocs(communicator = communicator) == halo_proc_count(halo))
    halo%communicator = communicator

  end subroutine set_halo_communicator

  pure function halo_proc_count(halo)
    !!< Retrieve the number of processes in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: halo_proc_count

    halo_proc_count = halo%nprocs

  end function halo_proc_count

  pure function halo_send_count(halo, process)
    !!< Retrieve the number of send nodes in the supplied halo for the supplied
    !!< process

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process

    integer :: halo_send_count

    !assert(process > 0)
    !assert(process <= halo_proc_count(halo))

    !assert(associated(halo%sends))
    !assert(associated(halo%sends(process)%ptr))

    halo_send_count = size(halo%sends(process)%ptr)

  end function halo_send_count

  pure function halo_receive_count(halo, process)
    !!< Retrieve the number of receive nodes in the supplied halo for the
    !!< supplied process

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process

    integer :: halo_receive_count

    !assert(process > 0)
    !assert(process <= halo_proc_count(halo))

    !assert(associated(halo%receives))
    !assert(associated(halo%receives(process)%ptr))

    halo_receive_count = size(halo%receives(process)%ptr)

  end function halo_receive_count

  function halo_unique_receive_count(halo, process)
    !!< Retrieve the number of unique receives nodes in the supplied halo for
    !!< the supplied process

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process

    integer :: halo_unique_receive_count

    halo_unique_receive_count = count_unique(halo_receives(halo, process))

  end function halo_unique_receive_count

  function halo_send(halo, process, index)
    !!< Retrieve a send node from the supplied halo

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process
    integer, intent(in) :: index

    integer :: halo_send

    assert(process > 0)
    assert(process <= halo_proc_count(halo))
    assert(index > 0)
    assert(index <= halo_send_count(halo, process))

    assert(associated(halo%sends))
    assert(associated(halo%sends(process)%ptr))

    halo_send = halo%sends(process)%ptr(index)

  end function halo_send

  function halo_receive(halo, process, index)
    !!< Retrieve a receive node from the supplied halo

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process
    integer, intent(in) :: index

    integer :: halo_receive

    assert(process > 0)
    assert(process <= halo_proc_count(halo))
    assert(index > 0)
    assert(index <= halo_receive_count(halo, process))

    assert(associated(halo%receives))
    assert(associated(halo%receives(process)%ptr))

    halo_receive = halo%receives(process)%ptr(index)

  end function halo_receive

  subroutine set_halo_send(halo, process, index, node)
    !!< Set a send node for the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: process
    integer, intent(in) :: index
    integer, intent(in) :: node

    assert(process > 0)
    assert(process <= halo_proc_count(halo))
    assert(index > 0)
    assert(index <= halo_send_count(halo, process))

    assert(associated(halo%sends))
    assert(associated(halo%sends(process)%ptr))

    halo%sends(process)%ptr(index) = node

  end subroutine set_halo_send

  subroutine set_halo_receive(halo, process, index, node)
    !!< Set a receive node for the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: process
    integer, intent(in) :: index
    integer, intent(in) :: node

    assert(process > 0)
    assert(process <= halo_proc_count(halo))
    assert(index > 0)
    assert(index <= halo_receive_count(halo, process))

    assert(associated(halo%receives))
    assert(associated(halo%receives(process)%ptr))

    halo%receives(process)%ptr(index) = node

  end subroutine set_halo_receive

  subroutine halo_send_counts(halo, nsends)
    !!< Retrieve the number of sends nodes for all process

    type(halo_type), intent(in) :: halo
    integer, dimension(halo_proc_count(halo)), intent(out) :: nsends

    integer :: i

    do i = 1, halo_proc_count(halo)
      nsends(i) = halo_send_count(halo, i)
    end do

  end subroutine halo_send_counts

  subroutine halo_receive_counts(halo, nreceives)
    !!< Retrieve the number of receives nodes for all process

    type(halo_type), intent(in) :: halo
    integer, dimension(halo_proc_count(halo)), intent(out) :: nreceives

    integer :: i

    do i = 1, halo_proc_count(halo)
      nreceives(i) = halo_receive_count(halo, i)
    end do

  end subroutine halo_receive_counts

  function halo_sends(halo, process)
    !!< Retrieve all send nodes for the supplied process from the supplied halo

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process

    integer, dimension(:), pointer :: halo_sends

    assert(process > 0)
    assert(process <= halo_proc_count(halo))

    assert(associated(halo%sends))
    assert(associated(halo%sends(process)%ptr))

    halo_sends => halo%sends(process)%ptr

  end function halo_sends

  function halo_receives(halo, process)
    !!< Retrieve all receive nodes for the supplied process from the supplied
    !!< halo

    type(halo_type), intent(in) :: halo
    integer, intent(in) :: process

    integer, dimension(:), pointer :: halo_receives

    assert(process > 0)
    assert(process <= halo_proc_count(halo))

    assert(associated(halo%receives))
    assert(associated(halo%receives(process)%ptr))

    halo_receives => halo%receives(process)%ptr

  end function halo_receives

  subroutine set_halo_sends(halo, process, sends)
    !!< Set all send nodes for the supplied process for the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: process
    integer, dimension(halo_send_count(halo, process)), intent(in) :: sends

    assert(process > 0)
    assert(process <= halo_proc_count(halo))

    assert(associated(halo%sends))
    assert(associated(halo%sends(process)%ptr))

    halo%sends(process)%ptr = sends

  end subroutine set_halo_sends

  subroutine set_halo_receives(halo, process, receives)
    !!< Set all receive nodes for the supplied process for the supplied halo

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: process
    integer, dimension(halo_receive_count(halo, process)), intent(in) :: receives

    assert(process > 0)
    assert(process <= halo_proc_count(halo))

    assert(associated(halo%receives))
    assert(associated(halo%receives(process)%ptr))

    halo%receives(process)%ptr = receives

  end subroutine set_halo_receives

  pure function halo_all_sends_count(halo)
    !!< Count the total number of sends in the supplied halo.

    type(halo_type), intent(in) :: halo

    integer :: halo_all_sends_count

    integer :: i

    !assert(associated(halo%receives))

    halo_all_sends_count = 0
    do i = 1, halo_proc_count(halo)
      !assert(associated(halo%sends(i)%ptr))
      halo_all_sends_count = halo_all_sends_count + size(halo%sends(i)%ptr)
    end do

  end function halo_all_sends_count

  pure function halo_all_receives_count(halo)
    !!< Count the total number of receives in the supplied halo.

    type(halo_type), intent(in) :: halo

    integer :: halo_all_receives_count

    integer :: i

    !assert(associated(halo%receives))

    halo_all_receives_count = 0
    do i = 1, halo_proc_count(halo)
      !assert(associated(halo%receives(i)%ptr))
      halo_all_receives_count = halo_all_receives_count + size(halo%receives(i)%ptr)
    end do

  end function halo_all_receives_count

  function halo_all_unique_receives_count(halo)
    !!< Count the total number of unique receives in the supplied halo.

    type(halo_type), intent(in) :: halo

    integer :: halo_all_unique_receives_count

    integer, dimension(:), allocatable :: receives

    allocate(receives(halo_all_receives_count(halo)))

    call extract_all_halo_receives(halo, receives)
    halo_all_unique_receives_count = count_unique(receives)

    deallocate(receives)

  end function halo_all_unique_receives_count

  subroutine extract_all_halo_sends(halo, sends, nsends, start_indices)
    !!< Extract all sends from the supplied halo and assemble them onto a
    !!< single vector.

    type(halo_type), intent(in) :: halo
    integer, dimension(halo_all_sends_count(halo)), intent(out) :: sends
    integer, dimension(halo_proc_count(halo)), optional, intent(out) :: nsends
    integer, dimension(halo_proc_count(halo)), optional, intent(out) :: start_indices

    integer :: i, index, nprocs

    nprocs = halo_proc_count(halo)

    assert(associated(halo%sends))

    index = 1
    do i = 1, nprocs
      assert(associated(halo%sends(i)%ptr))

      sends(index:index + size(halo%sends(i)%ptr) - 1) = halo%sends(i)%ptr

      if(present(nsends)) nsends(i) = size(halo%sends(i)%ptr)
      if(present(start_indices)) start_indices(i) = index

      index = index + size(halo%sends(i)%ptr)
    end do
    assert(index == size(sends) + 1)

  end subroutine extract_all_halo_sends

  subroutine extract_all_halo_receives(halo, receives, nreceives, start_indices)
    !!< Extract all receives from the supplied halo and assemble them onto a
    !!< single vector.

    type(halo_type), intent(in) :: halo
    integer, dimension(halo_all_receives_count(halo)), intent(out) :: receives
    integer, dimension(halo_proc_count(halo)), optional, intent(out) :: nreceives
    integer, dimension(halo_proc_count(halo)), optional, intent(out) :: start_indices

    integer :: i, index, nprocs

    nprocs = halo_proc_count(halo)

    assert(associated(halo%receives))

    index = 1
    do i = 1, nprocs
      assert(associated(halo%receives(i)%ptr))

      receives(index:index + size(halo%receives(i)%ptr) - 1) = halo%receives(i)%ptr

      if(present(nreceives)) nreceives(i) = size(halo%receives(i)%ptr)
      if(present(start_indices)) start_indices(i) = index

      index = index + size(halo%receives(i)%ptr)
    end do
    assert(index == size(receives) + 1)

  end subroutine extract_all_halo_receives

  subroutine set_all_halo_sends(halo, sends)
    !!< Set all sends from the supplied vector.

    type(halo_type), intent(inout) :: halo
    integer, dimension(:), intent(in) :: sends

    integer :: i, index, nprocs

    assert(size(sends) == halo_all_sends_count(halo))

    nprocs = halo_proc_count(halo)

    assert(associated(halo%sends))

    index = 1
    do i = 1, nprocs
      call set_halo_sends(halo, i, sends(index:index + halo_send_count(halo, i) - 1))
      index = index + halo_send_count(halo, i)
    end do
    assert(index == size(sends) + 1)

  end subroutine set_all_halo_sends

  subroutine set_all_halo_receives(halo, receives)
    !!< Set all receives from the supplied vector.

    type(halo_type), intent(inout) :: halo
    integer, dimension(:), intent(in) :: receives

    integer :: i, index, nprocs

    assert(size(receives) == halo_all_receives_count(halo))

    nprocs = halo_proc_count(halo)

    assert(associated(halo%receives))

    index = 1
    do i = 1, nprocs
      call set_halo_receives(halo, i, receives(index:index + halo_receive_count(halo, i) - 1))
      index = index + halo_receive_count(halo, i)
    end do
    assert(index == size(receives) + 1)

  end subroutine set_all_halo_receives

  function min_halo_send_node(halo) result(min_node)
    !!< Return the minimum send node stored in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: min_node

    integer :: i

    min_node = huge(0)
    do i = 1, halo_proc_count(halo)
      min_node = min(min_node, minval(halo_sends(halo, i)))
    end do

  end function min_halo_send_node

  function min_halo_receive_node(halo) result(min_node)
    !!< Return the minimum receive node stored in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: min_node

    integer :: i

    min_node = huge(0)
    do i = 1, halo_proc_count(halo)
      min_node = min(min_node, minval(halo_receives(halo, i)))
    end do

  end function min_halo_receive_node

  function min_halo_node(halo) result(min_node)
    !!< Return the minimum node stored in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: min_node

    min_node = min_halo_send_node(halo)
    min_node = min(min_node, min_halo_receive_node(halo))

  end function min_halo_node

  pure function max_halo_send_node(halo) result(max_node)
    !!< Return the maximum send node stored in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: max_node

    integer :: i

    max_node = -huge(0)
    do i = 1, halo_proc_count(halo)
      max_node = max(max_node, maxval(halo%sends(i)%ptr))
    end do

  end function max_halo_send_node

  pure function max_halo_receive_node(halo) result(max_node)
    !!< Return the maximum receive node stored in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: max_node

    integer :: i

    max_node = -huge(0)
    do i = 1, halo_proc_count(halo)
      max_node = max(max_node, maxval(halo%receives(i)%ptr))
    end do

  end function max_halo_receive_node

  pure function max_halo_node(halo) result(max_node)
    !!< Return the maximum node stored in the supplied halo

    type(halo_type), intent(in) :: halo

    integer :: max_node

    max_node = max(halo%nowned_nodes, max_halo_send_node(halo))
    max_node = max(max_node, max_halo_receive_node(halo))

  end function max_halo_node

  function serial_storage_halo_single(halo) result(serial)
    !!< Return whether this halo is used to store parallel data in serial. This
    !!< should be used (rather than a .not. isparallel()) for future proofing.

    type(halo_type), intent(in) :: halo

    logical :: serial

#ifdef HAVE_MPI
    serial = getnprocs(communicator = MPI_COMM_FEMTOOLS) == 1
#else
    serial = .true.
#endif

  end function serial_storage_halo_single

  function serial_storage_halo_multiple(halos) result(serial)
    !!< Return whether these halos are used to store parallel data in serial. This
    !!< should be used (rather than a .not. isparallel()) for future proofing.

    type(halo_type), dimension(:), intent(in) :: halos

    integer :: i
    logical, dimension(size(halos)) :: serial

    do i = 1, size(halos)
      serial(i) = serial_storage_halo(halos(i))
    end do

  end function serial_storage_halo_multiple

end module halos_base
