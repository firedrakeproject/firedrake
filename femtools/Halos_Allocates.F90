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

module halos_allocates

  use fldebug
  use global_parameters, only : empty_name, malloc, free
  use halo_data_types
  use halos_base
  use halos_debug
  use mpi_interfaces
  use parallel_tools
  use reference_counting
  use iso_c_binding

  implicit none

#ifdef __INTEL_COMPILER
  intrinsic sizeof
#define c_sizeof sizeof
#endif

  private

  public :: allocate, reallocate, deallocate, incref, has_references, &
    & deallocate_ownership_cache, deallocate_universal_numbering_cache, &
    & nullify

  interface allocate
    module procedure allocate_halo, allocate_halo_halo
  end interface allocate

  interface reallocate
    module procedure reallocate_halo
  end interface reallocate

  interface deallocate
    module procedure deallocate_halo, deallocate_halo_vector
  end interface deallocate

  interface nullify
    module procedure nullify_halo, nullify_halo_vector
  end interface nullify

#include "Reference_count_interface_halo_type.F90"

contains

#include "Reference_count_halo_type.F90"

  subroutine allocate_halo(halo, nsends, nreceives, name, communicator, nprocs, nowned_nodes, data_type, ordering_scheme)
    !!< Allocate a halo

    type(halo_type), intent(out) :: halo
    type(c_ptr), dimension(:), pointer :: tmp_ptr
    ! size(nprocs / communicator size)
    integer, dimension(:), intent(in) :: nsends
    ! size(nprocs / communicator size)
    integer, dimension(:), intent(in) :: nreceives
    character(len = *), optional, intent(in) :: name
    integer, optional, intent(in) :: communicator
    integer, optional, intent(in) :: nprocs
    integer, optional, intent(in) :: nowned_nodes
    integer, optional, intent(in) :: data_type
    integer, optional, intent(in) :: ordering_scheme

    integer :: i, lnprocs

#ifdef HAVE_MPI
    halo%communicator = MPI_COMM_FEMTOOLS
#else
    halo%communicator = -1
#endif

    ! Set nprocs
    if(present(communicator)) then
#ifdef HAVE_MPI
      lnprocs = getnprocs(communicator = communicator)
      halo%nprocs = lnprocs
      call set_halo_communicator(halo, communicator)

      if(present(nprocs)) then
        if(nprocs /= lnprocs) then
          FLAbort("Inconsistent communicator and nprocs supplied when allocating a halo")
        end if
      end if
#else
      FLAbort("Cannot assign a communicator to a halo without MPI support")
#endif
    else if(present(nprocs)) then
      halo%nprocs = nprocs
      lnprocs = nprocs
    else
      FLAbort("Either a communicator or nprocs must be supplied when allocating a halo")
    end if

    assert(lnprocs >= 0)
    assert(size(nsends) == lnprocs)
    assert(size(nreceives) == lnprocs)

    ! Allocate the sends

    halo%sends_c = malloc(lnprocs * c_sizeof(1_c_intptr_t))
    call c_f_pointer(halo%sends_c, tmp_ptr, [lnprocs])

    allocate(halo%sends(lnprocs))
    do i = 1, lnprocs
      assert(nsends(i) >= 0)
      tmp_ptr(i) = malloc(nsends(i) * c_sizeof(1_c_int))
      call c_f_pointer(tmp_ptr(i), halo%sends(i)%ptr, [nsends(i)])
    end do

    halo%receives_c = malloc(lnprocs * c_sizeof(1_c_intptr_t))
    call c_f_pointer(halo%receives_c, tmp_ptr, [lnprocs])

    allocate(halo%receives(lnprocs))
    do i = 1, lnprocs
      assert(nreceives(i) >= 0)
      tmp_ptr(i) = malloc(nreceives(i) * c_sizeof(1_c_int))
      call c_f_pointer(tmp_ptr(i), halo%receives(i)%ptr, [nreceives(i)])
    end do

    if(present(name)) then
      ! Set the name
      call set_halo_name(halo, name)
    else
      call set_halo_name(halo, empty_name)
    end if

    if(present(data_type)) then
      ! Set the data type
      call set_halo_data_type(halo, data_type)
    else
      call set_halo_data_type(halo, HALO_TYPE_CG_NODE)
    end if

    if(present(ordering_scheme)) then
      ! Set the ordering scheme
      call set_halo_ordering_scheme(halo, ordering_scheme)
    else
      call set_halo_ordering_scheme(halo, HALO_ORDER_TRAILING_RECEIVES)
    end if

    if(present(nowned_nodes)) then
      ! Set the number of owned nodes
      call set_halo_nowned_nodes(halo, nowned_nodes)
    end if

    call addref(halo)

  end subroutine allocate_halo

  subroutine allocate_halo_halo(output_halo, base_halo)
    !!< Allocate a halo based upon an existing halo

    type(halo_type), intent(out) :: output_halo
    type(halo_type), intent(in) :: base_halo

    integer :: nprocs
    integer, dimension(:), allocatable :: nreceives, nsends

    nprocs = halo_proc_count(base_halo)
    allocate(nsends(nprocs))
    allocate(nreceives(nprocs))
    call halo_send_counts(base_halo, nsends)
    call halo_receive_counts(base_halo, nreceives)

    call allocate(output_halo, &
      & nsends = nsends, &
      & nreceives = nreceives, &
      & name = halo_name(base_halo), &
      & communicator = halo_communicator(base_halo), &
      & nowned_nodes = halo_nowned_nodes(base_halo), &
      & data_type = halo_data_type(base_halo), &
      & ordering_scheme = halo_ordering_scheme(base_halo))

    deallocate(nsends)
    deallocate(nreceives)

  end subroutine allocate_halo_halo

  subroutine reallocate_halo(halo, nsends, nreceives)
    !!< Re-allocate a halo. This is useful if the send or receive allocation is
    !!< deferred.

    type(halo_type), intent(inout) :: halo
    type(c_ptr), dimension(:), pointer :: tmp_ptr
    integer, dimension(halo_proc_count(halo)), optional, intent(in) :: nsends
    integer, dimension(halo_proc_count(halo)), optional, intent(in) :: nreceives

    integer :: i, nprocs

    nprocs = halo_proc_count(halo)

    if(present(nsends)) then
      assert(associated(halo%sends))
      call c_f_pointer(halo%sends_c, tmp_ptr, [halo%nprocs])
      do i = 1, nprocs
        assert(associated(halo%sends(i)%ptr))
        call free(tmp_ptr(i))
        halo%sends(i)%ptr => null()
        tmp_ptr(i) = malloc(nsends(i) * c_sizeof(1_c_int))
        call c_f_pointer(tmp_ptr(i), halo%sends(i)%ptr, [nsends(i)])
      end do
    end if
    if(present(nreceives)) then
      assert(associated(halo%receives))
      call c_f_pointer(halo%receives_c, tmp_ptr, [halo%nprocs])
      do i = 1, nprocs
        assert(associated(halo%receives(i)%ptr))
        call free(tmp_ptr(i))
        halo%receives(i)%ptr => null()
        tmp_ptr(i) = malloc(nreceives(i) * c_sizeof(1_c_int))
        call c_f_pointer(tmp_ptr(i), halo%receives(i)%ptr, [nreceives(i)])
      end do
    end if

  end subroutine reallocate_halo

  subroutine deallocate_halo(halo)
    !!< Deallocate a halo type

    type(halo_type), intent(inout) :: halo
    type(c_ptr), dimension(:), pointer :: tmp_ptr
    integer :: i

    call decref(halo)
    if(has_references(halo)) return

    ! Deallocate the sends
    if(associated(halo%sends)) then
       call c_f_pointer(halo%sends_c, tmp_ptr, [halo%nprocs])
       do i = 1, size(halo%sends)
          call free(tmp_ptr(i))
          tmp_ptr(i) = C_NULL_PTR
          halo%sends(i)%ptr => null()
       end do
       call free(halo%sends_c)
       halo%sends_c = C_NULL_PTR
       halo%sends => null()
    end if

    ! Deallocate the receives
    if(associated(halo%receives)) then
       call c_f_pointer(halo%receives_c, tmp_ptr, [halo%nprocs])
       do i = 1, size(halo%receives)
          call free(tmp_ptr(i))
          tmp_ptr(i) = C_NULL_PTR
          halo%receives(i)%ptr => null()
       end do
       call free(halo%receives_c)
       halo%receives_c = C_NULL_PTR
       halo%receives => null()
    end if

    ! Deallocate caches
    call deallocate_ownership_cache(halo)
    call deallocate_universal_numbering_cache(halo)

    ! Reset variables
    call nullify(halo)

  end subroutine deallocate_halo

  subroutine deallocate_halo_vector(halos)
    !!< Deallocate each of a vector of halos.
    type(halo_type), dimension(:), intent(inout) :: halos

    integer :: i

    do i=1, size(halos)
       call deallocate(halos(i))
    end do

  end subroutine deallocate_halo_vector

  subroutine deallocate_ownership_cache(halo)
    !!< Deallocate the node ownership cache data

    type(halo_type), intent(inout) :: halo

    if(associated(halo%owners)) then
      deallocate(halo%owners)
      nullify(halo%owners)
    end if

  end subroutine deallocate_ownership_cache

  subroutine deallocate_universal_numbering_cache(halo)
    !!< Deallocate halo universal node numbering cache data

    type(halo_type), intent(inout) :: halo

    halo%unn_count = -1
    if (associated(halo%owned_nodes_unn_base)) then
      deallocate(halo%owned_nodes_unn_base)
    end if
    halo%my_owned_nodes_unn_base = -1

    if(associated(halo%receives_gnn_to_unn)) then
      call free(halo%receives_gnn_to_unn_c)
      halo%receives_gnn_to_unn_c = C_NULL_PTR
      nullify(halo%receives_gnn_to_unn)
    end if

    if(associated(halo%gnn_to_unn)) then
      deallocate(halo%gnn_to_unn)
      nullify(halo%gnn_to_unn)
    end if

  end subroutine deallocate_universal_numbering_cache

  subroutine nullify_halo(halo)
    !!< Return a halo type to its uninitialised state

    type(halo_type), intent(inout) :: halo

    type(halo_type) :: null_halo

    ! Initialise the null_halo name to prevent uninitialised variable access
    call set_halo_name(null_halo, empty_name)
    halo = null_halo

  end subroutine nullify_halo

  subroutine nullify_halo_vector(halo)
    !!< Return a halo type to its uninitialised state
    type(halo_type), dimension(:), intent(inout) :: halo

    integer :: i

    do i=1,size(halo)
       call nullify(halo(i))
    end do

  end subroutine nullify_halo_vector

end module halos_allocates
