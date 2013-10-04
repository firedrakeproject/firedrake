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

module halos_registration

  use fields_allocates
  use fields_base
  use fields_data_types
  use fields_manipulation
  use fldebug
  use futils
  use halo_data_types
  use halos_allocates
  use halos_base
  use halos_debug
  use halos_communications
  use halos_derivation
  use halos_ownership
  use halos_numbering
  use mpi_interfaces
  use parallel_tools
  use data_structures

  implicit none

  private

  public :: read_halos, write_halos, generate_substate_halos, verify_halos
  public :: extract_raw_halo_data, form_halo_from_raw_data

  interface
    subroutine chalo_reader_reset()
    end subroutine chalo_reader_reset

    function chalo_reader_set_input(filename, filename_len, process, nprocs)
      implicit none
      integer, intent(in) :: filename_len
      character(len = filename_len) :: filename
      integer, intent(in) :: process
      integer, intent(in) :: nprocs
      integer :: chalo_reader_set_input
    end function chalo_reader_set_input

    subroutine chalo_reader_query_output(level, nprocs, nsends, nreceives)
      implicit none
      integer, intent(in) :: level
      integer, intent(in) :: nprocs
      integer, dimension(nprocs), intent(out) :: nsends
      integer, dimension(nprocs), intent(out) :: nreceives
    end subroutine chalo_reader_query_output

    subroutine chalo_reader_get_output(level, nprocs, nsends, nreceives, &
      & npnodes, send, recv)
      implicit none
      integer, intent(in) :: level
      integer, intent(in) :: nprocs
      integer, dimension(nprocs), intent(in) :: nsends
      integer, dimension(nprocs), intent(in) :: nreceives
      integer, intent(out) :: npnodes
      integer, dimension(sum(nsends)), intent(out) :: send
      integer, dimension(sum(nreceives)), intent(out) :: recv
    end subroutine chalo_reader_get_output

    subroutine chalo_writer_reset()
    end subroutine chalo_writer_reset

    subroutine chalo_writer_initialise(process, nprocs)
      implicit none
      integer, intent(in) :: process
      integer, intent(in) :: nprocs
    end subroutine chalo_writer_initialise

    subroutine chalo_writer_set_input(level, nprocs, nsends, nreceives, &
      & npnodes, send, recv)
      implicit none
      integer, intent(in) :: level
      integer, intent(in) :: nprocs
      integer, dimension(nprocs), intent(in) :: nsends
      integer, dimension(nprocs), intent(in) :: nreceives
      integer, intent(in) :: npnodes
      integer, dimension(sum(nsends)), intent(in) :: send
      integer, dimension(sum(nreceives)), intent(in) :: recv
    end subroutine chalo_writer_set_input

    function chalo_writer_write(filename, filename_len)
      implicit none
      integer, intent(in) :: filename_len
      character(len = filename_len) :: filename
      integer :: chalo_writer_write
    end function chalo_writer_write
  end interface

  interface read_halos
    module procedure read_halos_mesh, read_halos_positions
  end interface read_halos

contains

  subroutine read_halos_mesh(filename, mesh, communicator)
    character(len = *), intent(in) :: filename
    type(mesh_type), intent(inout) :: mesh
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: error_count, i, lcommunicator, nowned_nodes, nprocs, procno
    integer, dimension(:), allocatable :: nreceives, nsends, receives, sends

    ewrite(1, *) "In read_halos_mesh"

    assert(continuity(mesh) == 0)
    assert(.not. associated(mesh%halos))
    assert(.not. associated(mesh%element_halos))

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    procno = getprocno(communicator = lcommunicator)
    nprocs = getnprocs(communicator = lcommunicator)

    error_count = chalo_reader_set_input(filename, len_trim(filename), procno - 1, nprocs)
    call allsum(error_count, communicator = lcommunicator)
    if(error_count > 0) then
      FLExit("Unable to read halos with name " // trim(filename))
    end if

    allocate(mesh%halos(2))
    allocate(nsends(nprocs))
    allocate(nreceives(nprocs))
    do i = 1, 2
      call chalo_reader_query_output(i, nprocs, nsends, nreceives)
      call allocate(mesh%halos(i), nsends, nreceives, name = trim(mesh%name) // "Level" // int2str(i) // "Halo", communicator = lcommunicator, &
        & data_type = HALO_TYPE_CG_NODE, ordering_scheme = HALO_ORDER_TRAILING_RECEIVES)

      allocate(sends(sum(nsends)))
      allocate(receives(sum(nreceives)))
      call chalo_reader_get_output(i, nprocs, nsends, nreceives, &
        & nowned_nodes, sends, receives)
      call set_halo_nowned_nodes(mesh%halos(i), nowned_nodes)
      call set_all_halo_sends(mesh%halos(i), sends)
      call set_all_halo_receives(mesh%halos(i), receives)
      deallocate(sends)
      deallocate(receives)
      assert(trailing_receives_consistent(mesh%halos(i)))

      if(.not. serial_storage_halo(mesh%halos(i))) then
        assert(halo_valid_for_communication(mesh%halos(i)))
        call create_global_to_universal_numbering(mesh%halos(i))
        call create_ownership(mesh%halos(i))
      end if
    end do
    deallocate(nsends)
    deallocate(nreceives)

    call chalo_reader_reset()

    if(all(serial_storage_halo(mesh%halos))) then
      allocate(mesh%element_halos(0))
    else
      allocate(mesh%element_halos(2))
      call derive_element_halo_from_node_halo(mesh, &
        & ordering_scheme = HALO_ORDER_GENERAL, create_caches = .true.)
    end if

    call refresh_topology(mesh)

    ewrite(1, *) "Exiting read_halos_mesh"
#else
    FLAbort("read_halos_mesh cannot be called without MPI support")
#endif

  end subroutine read_halos_mesh

  subroutine read_halos_positions(filename, positions, communicator)
    character(len = *), intent(in) :: filename
    type(vector_field), intent(inout) :: positions
    integer, optional, intent(in) :: communicator

    call read_halos(filename, positions%mesh, communicator = communicator)

  end subroutine read_halos_positions

  subroutine verify_halos(positions)
    type(vector_field), intent(inout) :: positions

    integer :: i, nhalos
    type(mesh_type) :: pwc_mesh
    type(vector_field) :: positions_ele

    ! Node halo verification
    nhalos = halo_count(positions)
    do i = 1, nhalos
      if(.not. serial_storage_halo(positions%mesh%halos(i))) then
         assert(halo_verifies(positions%mesh%halos(i), positions))
      end if
    end do

    ! Element halo verification
    nhalos = element_halo_count(positions)
    if(nhalos > 0) then
       if(.not. all(serial_storage_halo(positions%mesh%element_halos))) then
          pwc_mesh = piecewise_constant_mesh(positions%mesh, "PiecewiseConstantMesh")
          call allocate(positions_ele, positions%dim, pwc_mesh, positions%name)
          call deallocate(pwc_mesh)
          do i = 1, element_count(positions)
             ! Note that we are setting NODE values of ele_postions to
             !  ELEMENT values of positions. This is an abuse of notation
             !  caused by halo_verifies assuming all the world's a node halo.
             call set(positions_ele,i, sum(ele_val(positions,i),2)&
                  &/positions%dim)
          end do
          do i = 1, nhalos
             if(.not. serial_storage_halo(positions%mesh%element_halos(i))) then
                assert(halo_verifies(positions%mesh%element_halos(i), positions_ele))
             end if
          end do
          call deallocate(positions_ele)
       end if
    end if

  end subroutine verify_halos

  subroutine generate_substate_halos(external_mesh,subdomain_mesh,node_list,inverse_node_list)

    type(mesh_type), intent(in) :: external_mesh
    type(mesh_type), intent(inout) :: subdomain_mesh
    integer, dimension(:) :: node_list, inverse_node_list

    integer :: nhalos, communicator, nprocs, procno, ihalo, inode, iproc, nowned_nodes

    ewrite(1, *) "In generate_substate_halos"

    assert(continuity(subdomain_mesh) == 0)
    assert(.not. associated(subdomain_mesh%halos))
    assert(.not. associated(subdomain_mesh%element_halos))

    ! Initialise key MPI information:

    nhalos = halo_count(external_mesh)
    ewrite(2,*) "Number of subdomain_mesh halos = ",nhalos

    if(nhalos == 0) return

    communicator = halo_communicator(external_mesh%halos(nhalos))
    nprocs = getnprocs(communicator = communicator)
    ewrite(2,*) 'Number of processes = ', nprocs
    procno = getprocno(communicator = communicator)
    ewrite(2,*) 'Processor ID/number = ', procno

    ! Allocate subdomain mesh halos:
    allocate(subdomain_mesh%halos(nhalos))

    ! Derive subdomain_mesh halos:
    do ihalo = 1, nhalos

       subdomain_mesh%halos(ihalo) = derive_sub_halo(external_mesh%halos(ihalo),node_list)

       assert(trailing_receives_consistent(subdomain_mesh%halos(ihalo)))

       if(.not. serial_storage_halo(external_mesh%halos(ihalo))) then
          assert(halo_valid_for_communication(subdomain_mesh%halos(ihalo)))
          call create_global_to_universal_numbering(subdomain_mesh%halos(ihalo))
          call create_ownership(subdomain_mesh%halos(ihalo))
       end if

    end do ! ihalo

    if(all(serial_storage_halo(subdomain_mesh%halos))) then
      allocate(subdomain_mesh%element_halos(0))
    else
      allocate(subdomain_mesh%element_halos(nhalos))
      call derive_element_halo_from_node_halo(subdomain_mesh, &
        & ordering_scheme = HALO_ORDER_TRAILING_RECEIVES, create_caches = .true.)
    end if

    call refresh_topology(subdomain_mesh)

  end subroutine generate_substate_halos

  subroutine write_halos(filename, mesh)
    character(len = *), intent(in) :: filename
    type(mesh_type), intent(in) :: mesh

    integer :: communicator, error_count, i, nhalos, procno, nparts, nprocs
    integer, dimension(:), allocatable :: nreceives, nsends, receives, sends

    ewrite(1, *) "In write_halos"

    nhalos = halo_count(mesh)
    if(nhalos == 0) return

    communicator = halo_communicator(mesh%halos(nhalos))
    procno = getprocno(communicator = communicator)
    nparts = get_active_nparts(ele_count(mesh), communicator = communicator)

    if(procno <= nparts) then
      nprocs = getnprocs(communicator = communicator)

      call chalo_writer_initialise(procno - 1, nparts)

      allocate(nsends(nprocs))
      allocate(nreceives(nprocs))
      do i = 1, nhalos
        allocate(sends(halo_all_sends_count(mesh%halos(i))))
        allocate(receives(halo_all_receives_count(mesh%halos(i))))
        call extract_all_halo_sends(mesh%halos(i), sends, nsends = nsends)
        call extract_all_halo_receives(mesh%halos(i), receives, nreceives = nreceives)
        call chalo_writer_set_input(i, nparts, nsends(:nparts), nreceives(:nparts), &
          & halo_nowned_nodes(mesh%halos(i)), sends, receives)
        deallocate(sends)
        deallocate(receives)
      end do
      deallocate(nsends)
      deallocate(nreceives)

      error_count = chalo_writer_write(filename, len_trim(filename))
      call chalo_writer_reset()
    else
      error_count = 0
    end if

    call allsum(error_count, communicator = communicator)
    if(error_count > 0) then
      FLExit("Unable to write halos with name " // trim(filename))
    end if

    ewrite(1, *) "Exiting write_halos"

  end subroutine write_halos

  subroutine extract_raw_halo_data(halo, sends, send_starts, receives, receive_starts, nowned_nodes)
    !!< Extract raw halo data from the supplied halo

    type(halo_type), intent(in) :: halo

    !! Send nodes for all processes. Size halo_all_sends_count(halo).
    integer, dimension(:), intent(out) :: sends
    !! imem type indices into sends denoting the start points of process send
    !! nodes. Size halo_proc_count(halo) or halo_proc_count(halo) + 1.
    integer, dimension(:), intent(out) :: send_starts
    !! Receive nodes for all process. Size halo_all_receives_count(halo).
    integer, dimension(:), intent(out) :: receives
    !! imem type indices into receives denoting the start points of process
    !! receive nodes. Size halo_proc_count(halo) or halo_proc_count(halo) + 1.
    integer, dimension(:), intent(out) :: receive_starts
    !! Number of owned nodes
    integer, optional, intent(out) :: nowned_nodes

    integer :: nprocs, receives_size, sends_size

    nprocs = halo_proc_count(halo)
    sends_size = halo_all_sends_count(halo)
    receives_size = halo_all_receives_count(halo)

    assert(size(sends) == sends_size)
    assert(any(size(send_starts) == (/nprocs, nprocs + 1/)))
    assert(size(receives) == receives_size)
    assert(any(size(receive_starts) == (/nprocs, nprocs + 1/)))

    ! Form sends, receives, send_starts and receive_starts from the halo
    call extract_all_halo_sends(halo, sends, start_indices = send_starts(:nprocs))
    call extract_all_halo_receives(halo, receives, start_indices = receive_starts(:nprocs))
    if(size(send_starts) == nprocs + 1) send_starts(nprocs + 1) = sends_size + 1
    if(size(receive_starts) == nprocs + 1) receive_starts(nprocs + 1) = receives_size + 1

    if(present(nowned_nodes)) then
      ! Extract nowned_nodes from the halo
      nowned_nodes = halo_nowned_nodes(halo)
    end if

  end subroutine extract_raw_halo_data

  subroutine form_halo_from_raw_data(halo, nprocs, sends, send_starts, receives, receive_starts, nowned_nodes, ordering_scheme, create_caches)
    !!< Inverse of extract_legacy_halo_data. halo is allocated by this
    !!< routine.

    type(halo_type), intent(inout) :: halo
    integer, intent(in) :: nprocs
    integer, dimension(:), intent(in) :: sends
    integer, dimension(:), intent(in) :: send_starts
    integer, dimension(:), intent(in) :: receives
    integer, dimension(:), intent(in) :: receive_starts
    integer, optional, intent(in) :: nowned_nodes
    integer, optional, intent(in) :: ordering_scheme
    logical, optional, intent(in) :: create_caches

    integer :: i, lordering_scheme
    integer, dimension(:), allocatable :: nreceives, nsends
    logical :: lcreate_caches

    if(present(ordering_scheme)) then
      lordering_scheme = ordering_scheme
    else
      lordering_scheme = HALO_ORDER_TRAILING_RECEIVES
    end if

    lcreate_caches = .not. present_and_false(create_caches)

    ! Form nsends and nreceives from send_starts and receive_starts
    assert(nprocs > 0)
    assert(any(size(send_starts) == (/nprocs, nprocs + 1/)))
    assert(any(size(receive_starts) == (/nprocs, nprocs + 1/)))

    allocate(nsends(nprocs))
    allocate(nreceives(nprocs))

    do i = 1, size(send_starts) - 1
      nsends(i) = send_starts(i + 1) - send_starts(i)
      assert(nsends(i) >= 0)
    end do
    if(size(send_starts) == nprocs) nsends(nprocs) = size(sends) - send_starts(nprocs) + 1
    assert(sum(nsends) == size(sends))

    do i = 1, size(receive_starts) - 1
      nreceives(i) = receive_starts(i + 1) - receive_starts(i)
      assert(nreceives(i) >= 0)
    end do
    if(size(receive_starts) == nprocs) nreceives(nprocs) = size(receives) - receive_starts(nprocs) + 1
    assert(sum(nreceives) == size(receives))

    ! Allocate the halo
    call allocate(halo, nsends, nreceives, nprocs = nprocs, &
      & nowned_nodes = nowned_nodes, name = "HaloFormedFromRawData", &
      & ordering_scheme = lordering_scheme)
    deallocate(nsends)
    deallocate(nreceives)

    ! Copy sends and receives into the halo
    call set_all_halo_sends(halo, sends)
    call set_all_halo_receives(halo, receives)

    if(lcreate_caches .and. .not. serial_storage_halo(halo)) then
      call create_global_to_universal_numbering(halo)
      call create_ownership(halo)
    end if

  end subroutine form_halo_from_raw_data

end module halos_registration
