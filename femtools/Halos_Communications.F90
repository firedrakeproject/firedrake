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

module halos_communications

  use fields_data_types
  use fields_base
  use fldebug
  use futils
  use halo_data_types
  use halos_allocates
  use halos_base
  use halos_debug
  use mpi_interfaces
  use parallel_tools
  use linked_lists
  use quicksort

  implicit none

  private

  public :: halo_update, halo_max, halo_verifies

  interface zero_halo_receives
    module procedure zero_halo_receives_array_integer, &
      & zero_halo_receives_array_real, zero_halo_receives_scalar_on_halo, &
      & zero_halo_receives_vector_on_halo_dim, zero_halo_receives_vector_on_halo
  end interface zero_halo_receives

  interface halo_update
    module procedure halo_update_array_integer, halo_update_array_integer_block, &
      & halo_update_array_integer_block2, halo_update_array_integer_star, &
      & halo_update_array_real, halo_update_array_real_block, halo_update_array_real_block2, &
      & halo_update_scalar_on_halo, halo_update_vector_on_halo, &
      & halo_update_tensor_on_halo, halo_update_scalar, halo_update_vector, &
      & halo_update_tensor
  end interface halo_update

  interface halo_max
    module procedure halo_max_array_real, halo_max_scalar_on_halo, &
      & halo_max_scalar
  end interface halo_max

  interface halo_verifies
    module procedure halo_verifies_array_integer, halo_verifies_array_real, &
      & halo_verifies_scalar, halo_verifies_vector_dim, halo_verifies_vector
  end interface halo_verifies

contains

  subroutine zero_halo_receives_array_integer(halo, integer_data)
    !!< Zero the receives for the supplied array of integer data

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(inout) :: integer_data

    integer :: i

    do i = 1, halo_proc_count(halo)
      integer_data(halo_receives(halo, i)) = 0
    end do

  end subroutine zero_halo_receives_array_integer

  subroutine zero_halo_receives_array_real(halo, real_data)
    !!< Zero the receives for the supplied array of real data

    type(halo_type), intent(in) :: halo
    real, dimension(:), intent(inout) :: real_data

    integer :: i

    do i = 1, halo_proc_count(halo)
      real_data(halo_receives(halo, i)) = 0.0
    end do

  end subroutine zero_halo_receives_array_real

  subroutine zero_halo_receives_scalar_on_halo(halo, sfield)
    !!< Zero the receives of the supplied halo for the supplied scalar field

    type(halo_type), intent(in) :: halo
    type(scalar_field), intent(inout) :: sfield

    call zero_halo_receives(halo, sfield%val)

  end subroutine zero_halo_receives_scalar_on_halo

  subroutine zero_halo_receives_vector_on_halo_dim(halo, vfield, dim)
    !!< Zero the receives of the supplied halo for the supplied vector field

    type(halo_type), intent(in) :: halo
    type(vector_field), intent(inout) :: vfield
    integer, intent(in) :: dim

    assert(dim >= 1)
    assert(dim <= vfield%dim)

    call zero_halo_receives(halo, vfield%val(dim,:))

  end subroutine zero_halo_receives_vector_on_halo_dim

  subroutine zero_halo_receives_vector_on_halo(halo, vfield)
    !!< Zero the receives of the supplied halo for the supplied vector field

    type(halo_type), intent(in) :: halo
    type(vector_field), intent(inout) :: vfield

    integer :: i

    do i = 1, vfield%dim
      call zero_halo_receives(halo, vfield, i)
    end do

  end subroutine zero_halo_receives_vector_on_halo

  subroutine halo_update_array_integer(halo, integer_data)
    !!< Update the supplied array of integer data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(inout) :: integer_data

    assert(size(integer_data, 1) >= max_halo_node(halo))

    call halo_update_array_integer_star(halo, integer_data, 1)

  end subroutine halo_update_array_integer

  subroutine halo_update_array_integer_block(halo, integer_data)
    !!< Update the supplied array of integer data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    integer, dimension(:,:), intent(inout) :: integer_data

    assert(size(integer_data, 2) >= max_halo_node(halo))

    call halo_update_array_integer_star(halo, integer_data, size(integer_data,1))

  end subroutine halo_update_array_integer_block

  subroutine halo_update_array_integer_block2(halo, integer_data)
    !!< Update the supplied array of integer data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    integer, dimension(:,:,:), intent(inout) :: integer_data

    assert(size(integer_data, 3) >= max_halo_node(halo))

    call halo_update_array_integer_star(halo, integer_data, size(integer_data,1)*size(integer_data,2))

  end subroutine halo_update_array_integer_block2

  subroutine halo_update_array_integer_star(halo, integer_data, block_size)
    !!< Update the supplied array of integer data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    integer, dimension(*), intent(inout) :: integer_data
    integer, intent(in) :: block_size

#ifdef HAVE_MPI
    integer :: communicator, i, ierr, nprocs, nreceives, nsends, rank
    integer, dimension(:), allocatable :: receive_types, requests, send_types, statuses
    integer tag
    assert(halo_valid_for_communication(halo))
    assert(.not. pending_communication(halo))

    nprocs = halo_proc_count(halo)
    communicator = halo_communicator(halo)

    ! Create indexed MPI types defining the indices into integer_data to be sent/received
    allocate(send_types(nprocs))
    allocate(receive_types(nprocs))
    send_types = MPI_DATATYPE_NULL
    receive_types = MPI_DATATYPE_NULL
    do i = 1, nprocs
      nsends = halo_send_count(halo, i)
      if(nsends > 0) then
        call mpi_type_create_indexed_block(nsends, block_size, &
          & (halo_sends(halo, i) - 1)*block_size, &
          & getpinteger(), send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      nreceives = halo_receive_count(halo, i)
      if(nreceives > 0) then
        call mpi_type_create_indexed_block(nreceives, block_size, &
          & (halo_receives(halo, i) - 1)*block_size, &
          & getpinteger(), receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Set up non-blocking communications
    allocate(requests(nprocs * 2))
    requests = MPI_REQUEST_NULL
    rank = getrank(communicator)
    tag = next_mpi_tag()

    do i = 1, nprocs
      ! Non-blocking sends
      if(halo_send_count(halo, i) > 0) then
         call mpi_isend(integer_data, 1, send_types(i), i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      ! Non-blocking receives
      if(halo_receive_count(halo, i) > 0) then
        call mpi_irecv(integer_data, 1, receive_types(i), i - 1, tag, communicator, requests(i + nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(statuses)
    deallocate(requests)

    ! Free the indexed MPI types
    do i = 1, nprocs
      if(send_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      if(receive_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do
    deallocate(send_types)
    deallocate(receive_types)
#else
    if(.not. valid_serial_halo(halo)) then
      FLAbort("Cannot update halos without MPI support")
    end if
#endif

  end subroutine halo_update_array_integer_star

  subroutine halo_update_array_real(halo, real_data)
    !!< Update the supplied array of real data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    real, dimension(:), intent(inout) :: real_data

    assert(size(real_data, 1) >= max_halo_node(halo))

    call halo_update_array_real_star(halo, real_data, 1)

  end subroutine halo_update_array_real

  subroutine halo_update_array_real_block(halo, real_data)
    !!< Update the supplied array of real data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    real, dimension(:,:), intent(inout) :: real_data

    assert(size(real_data, 2) >= max_halo_node(halo))

    call halo_update_array_real_star(halo, real_data, size(real_data,1))

  end subroutine halo_update_array_real_block

  subroutine halo_update_array_real_block2(halo, real_data)
    !!< Update the supplied array of real data. Fortran port of
    !!< FLComms::Update(...).

    type(halo_type), intent(in) :: halo
    real, dimension(:,:,:), intent(inout) :: real_data

    assert(size(real_data, 3) >= max_halo_node(halo))

    call halo_update_array_real_star(halo, real_data, size(real_data,1)*size(real_data,2))

  end subroutine halo_update_array_real_block2

  subroutine halo_update_array_real_star(halo, real_data, block_size)
    ! This is the actual workhorse for the previous versions of halo_update_real_...
    ! It simply takes in the begin address and the size of the blocks
    type(halo_type), intent(in) :: halo
    real, dimension(*), intent(inout) :: real_data
    integer, intent(in) :: block_size

#ifdef HAVE_MPI
    integer :: communicator, i, ierr, nprocs, nreceives, nsends, rank
    integer, dimension(:), allocatable :: receive_types, requests, send_types, statuses
    integer tag

    assert(halo_valid_for_communication(halo))
    assert(.not. pending_communication(halo))

    nprocs = halo_proc_count(halo)
    communicator = halo_communicator(halo)

    ! Create indexed MPI types defining the indices into real_data to be sent/received
    allocate(send_types(nprocs))
    allocate(receive_types(nprocs))
    send_types = MPI_DATATYPE_NULL
    receive_types = MPI_DATATYPE_NULL
    do i = 1, nprocs
      nsends = halo_send_count(halo, i)
      if(nsends > 0) then
        call mpi_type_create_indexed_block(nsends, block_size, &
          & (halo_sends(halo, i) - 1)*block_size, &
          & getpreal(), send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      nreceives = halo_receive_count(halo, i)
      if(nreceives > 0) then
        call mpi_type_create_indexed_block(nreceives, block_size, &
          & (halo_receives(halo, i) - 1)*block_size, &
          & getpreal(), receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Set up non-blocking communications
    allocate(requests(nprocs * 2))
    requests = MPI_REQUEST_NULL
    rank = getrank(communicator)
    tag = next_mpi_tag()

    do i = 1, nprocs
      ! Non-blocking sends
      if(halo_send_count(halo, i) > 0) then
        call mpi_isend(real_data, 1, send_types(i), i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      ! Non-blocking receives
      if(halo_receive_count(halo, i) > 0) then
        call mpi_irecv(real_data, 1, receive_types(i), i - 1, tag, communicator, requests(i + nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(statuses)
    deallocate(requests)

    ! Free the indexed MPI types
    do i = 1, nprocs
      if(send_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      if(receive_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do
    deallocate(send_types)
    deallocate(receive_types)
#else
    if(.not. valid_serial_halo(halo)) then
      FLAbort("Cannot update halos without MPI support")
    end if
#endif

  end subroutine halo_update_array_real_star

  subroutine halo_update_scalar_on_halo(halo, s_field)
    !!< Update the supplied scalar field on the suppied halo.

    type(halo_type), intent(in) :: halo
    type(scalar_field), intent(inout) :: s_field

    real, dimension(:), allocatable :: buffer

    ewrite(2, *) "Updating halo " // trim(halo%name) // " for field " // trim(s_field%name)

    select case(s_field%field_type)
      case(FIELD_TYPE_NORMAL)
        assert(associated(s_field%val))
        if(s_field%val_stride == 1) then
          call halo_update(halo, s_field%val)
        else
          ewrite(2,*) "Need to copy into temp. buffer because field has stride", s_field%val_stride
          ! A stride argument should be passed to halo_update_real_array. For
          ! now just use a buffer.
          allocate(buffer(node_count(s_field)))
          buffer = s_field%val
          call halo_update(halo, buffer)
          s_field%val = buffer
          deallocate(buffer)
        end if
      case(FIELD_TYPE_CONSTANT)
      case default
        ewrite(-1, "(a,i0)") "For field type ", s_field%field_type
        FLAbort("Unrecognised field type")
    end select

  end subroutine halo_update_scalar_on_halo

  subroutine halo_update_vector_on_halo(halo, v_field)
    !!< Update the supplied vector field on the suppied halo.

    type(halo_type), intent(in) :: halo
    type(vector_field), intent(inout) :: v_field

    integer :: i

    ewrite(2, *) "Updating halo " // trim(halo%name) // " for field " // trim(v_field%name)

    select case(v_field%field_type)
      case(FIELD_TYPE_NORMAL)
        call halo_update(halo, v_field%val)
      case(FIELD_TYPE_CONSTANT)
      case default
        ewrite(-1, "(a,i0)") "For field type ", v_field%field_type
        FLAbort("Unrecognised field type")
    end select

  end subroutine halo_update_vector_on_halo

  subroutine halo_update_tensor_on_halo(halo, t_field)
    !!< Update the supplied tensor field on the suppied halo.

    type(halo_type), intent(in) :: halo
    type(tensor_field), intent(inout) :: t_field

    integer :: i, j

    ewrite(2, *) "Updating halo " // trim(halo%name) // " for field " // trim(t_field%name)

    select case(t_field%field_type)
      case(FIELD_TYPE_NORMAL)
        assert(associated(t_field%val))
        call halo_update(halo, t_field%val)
      case(FIELD_TYPE_CONSTANT)
      case default
        ewrite(-1, "(a,i0)") "For field type ", t_field%field_type
        FLAbort("Unrecognised field type")
    end select

  end subroutine halo_update_tensor_on_halo

  subroutine halo_update_scalar(s_field, level)
    !!< Update the halos of the supplied field. If level is not supplied, the
    !!< field is updated on its largest halo.

    type(scalar_field), intent(inout) :: s_field
    integer, optional, intent(in) :: level

    integer :: llevel, nhalos

    nhalos = halo_count(s_field)
    if(present(level)) then
      assert(level > 0)
      llevel = min(level, nhalos)
    else
      llevel = nhalos
    end if

    if(nhalos > 0) then
      call halo_update(s_field%mesh%halos(llevel), s_field)
    end if

  end subroutine halo_update_scalar

  subroutine halo_update_vector(v_field, level)
    !!< Update the halos of the supplied field. If level is not supplied, the
    !!< field is updated on its largest halo.

    type(vector_field), intent(inout) :: v_field
    integer, optional, intent(in) :: level

    integer :: llevel, nhalos

    nhalos = halo_count(v_field)
    if(present(level)) then
      assert(level > 0)
      llevel = min(level, nhalos)
    else
      llevel = nhalos
    end if

    if(nhalos > 0) then
      call halo_update(v_field%mesh%halos(llevel), v_field)
    end if

  end subroutine halo_update_vector

  subroutine halo_update_tensor(t_field, level)
    !!< Update the halos of the supplied field. If level is not supplied, the
    !!< field is updated on its largest halo.

    type(tensor_field), intent(inout) :: t_field
    integer, optional, intent(in) :: level

    integer :: llevel, nhalos

    nhalos = halo_count(t_field)
    if(present(level)) then
      assert(level > 0)
      llevel = min(level, nhalos)
    else
      llevel = nhalos
    end if

    if(nhalos > 0) then
      call halo_update(t_field%mesh%halos(llevel), t_field)
    end if

  end subroutine halo_update_tensor

  subroutine halo_max_array_real(halo, real_data)
    type(halo_type), intent(in) :: halo
    real, dimension(:), intent(inout) :: real_data

#ifdef HAVE_MPI
    integer :: communicator, i, ierr, nprocs, nsends, nreceives, rank
    integer, dimension(:), allocatable :: requests, receive_types, send_types, statuses
    type(real_vector), dimension(:), allocatable :: receive_real_array
    integer tag

    assert(halo_valid_for_communication(halo))
    assert(.not. pending_communication(halo))

    assert(lbound(real_data, 1) <= min_halo_node(halo))
    assert(ubound(real_data, 1) >= max_halo_node(halo))

    nprocs = halo_proc_count(halo)
    communicator = halo_communicator(halo)

    ! Create indexed MPI types defining the indices into real_data to be sent
    allocate(send_types(nprocs))
    allocate(receive_types(nprocs))
    send_types = MPI_DATATYPE_NULL
    receive_types = MPI_DATATYPE_NULL
    do i = 1, nprocs
      nsends = halo_send_count(halo, i)
      if(nsends > 0) then
        call mpi_type_create_indexed_block(nsends, 1, &
          & halo_sends(halo, i) - lbound(real_data, 1), getpreal(), send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      nreceives = halo_receive_count(halo, i)
      if(nreceives > 0) then
        call mpi_type_create_indexed_block(nreceives, 1, &
          & halo_receives(halo, i) - lbound(real_data, 1), getpreal(), receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
        call mpi_type_commit(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Allocate receive arrays. Use these to collect values for the send/receive
    ! nodes for all sending processes.
    allocate(receive_real_array(nprocs * 2))

    ! Set up non-blocking communications
    allocate(requests(nprocs * 4))
    requests = MPI_REQUEST_NULL
    rank = getrank(communicator)
    tag = next_mpi_tag()

    do i = 1, nprocs
      ! Allocate receive arrays
      allocate(receive_real_array(i)%ptr(halo_send_count(halo, i)))
      allocate(receive_real_array(i + nprocs)%ptr(halo_receive_count(halo, i)))

      if(halo_send_count(halo, i) > 0) then
        ! Non-blocking sends on sends
        call mpi_isend(real_data, 1, send_types(i), i - 1, tag, communicator, requests(i), ierr)
        assert(ierr == MPI_SUCCESS)

        ! Non-blocking receives on sends
        call mpi_irecv(receive_real_array(i)%ptr, size(receive_real_array(i)%ptr), getpreal(), &
             i - 1, tag, communicator, requests(i + nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      if(halo_receive_count(halo, i) > 0) then
        ! Non-blocking sends on receives
        call mpi_isend(real_data, 1, receive_types(i), i - 1, tag, communicator, &
             requests(i + 2 * nprocs), ierr)
        assert(ierr == MPI_SUCCESS)

        ! Non-blocking receives on receives
        call mpi_irecv(receive_real_array(i + nprocs)%ptr, size(receive_real_array(i + nprocs)%ptr),&
             getpreal(), i - 1, tag, communicator, requests(i + 3 * nprocs), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do

    ! Wait for all non-blocking communications to complete
    allocate(statuses(MPI_STATUS_SIZE * size(requests)))
    call mpi_waitall(size(requests), requests, statuses, ierr)
    assert(ierr == MPI_SUCCESS)
    deallocate(statuses)
    deallocate(requests)

    ! Free the indexed MPI types
    do i = 1, nprocs
      if(send_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(send_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if

      if(receive_types(i) /= MPI_DATATYPE_NULL) then
        call mpi_type_free(receive_types(i), ierr)
        assert(ierr == MPI_SUCCESS)
      end if
    end do
    deallocate(send_types)
    deallocate(receive_types)

    ! Perform the allmax
    do i = 1, nprocs
      real_data(halo_sends(halo, i)) = max(real_data(halo_sends(halo, i)), receive_real_array(i)%ptr)
      real_data(halo_receives(halo, i)) = max(real_data(halo_receives(halo, i)), receive_real_array(i + nprocs)%ptr)
      deallocate(receive_real_array(i)%ptr)
      deallocate(receive_real_array(i + nprocs)%ptr)
    end do
    deallocate(receive_real_array)
#else
    if(.not. valid_serial_halo(halo)) then
      FLAbort("Cannot update halos without MPI support")
    end if
#endif

  end subroutine halo_max_array_real

  subroutine halo_max_scalar_on_halo(halo, s_field)
    type(halo_type), intent(in) :: halo
    type(scalar_field), intent(inout) :: s_field

    select case(s_field%field_type)
      case(FIELD_TYPE_NORMAL)
        call halo_max(halo, s_field%val)
      case(FIELD_TYPE_CONSTANT)
        call allmax(s_field%val(1), communicator = halo_communicator(halo))
      case default
        ewrite(-1, "(a,i0)") "For field type ", s_field%field_type
        FLAbort("Unrecognised field type")
    end select

  end subroutine halo_max_scalar_on_halo

  subroutine halo_max_scalar(s_field, level)
    type(scalar_field), intent(inout) :: s_field
    integer, optional, intent(in) :: level

    integer :: llevel, nhalos

    nhalos = halo_count(s_field)
    if(present(level)) then
      assert(level > 0)
      llevel = min(level, nhalos)
    else
      llevel = nhalos
    end if

    if(nhalos > 0) then
      call halo_max(s_field%mesh%halos(llevel), s_field)
    end if

  end subroutine halo_max_scalar

  function halo_verifies_array_integer(halo, integer_array) result(verifies)
    !!< Verify the halo against the supplied array of integer data

    type(halo_type), intent(in) :: halo
    integer, dimension(:), intent(in) :: integer_array

    logical :: verifies

#ifdef DDEBUG
    integer :: i, j, receive
#endif
    integer, dimension(size(integer_array)) :: linteger_array

    linteger_array = integer_array
    call zero_halo_receives(halo, linteger_array)

    call halo_update(halo, linteger_array)

    verifies = all(integer_array == linteger_array)
#ifdef DDEBUG
    if(.not. verifies) then
      do i = 1, halo_proc_count(halo)
        do j = 1, halo_receive_count(halo, i)
          receive = halo_receive(halo, i, j)
          if(integer_array(receive) /= linteger_array(receive)) then
            ewrite(0, *) "Warning: Halo receive ", receive, " for halo " // halo_name(halo) // " failed verification"
            ewrite(0, *) "Reference = ", integer_array(receive)
            ewrite(0, *) "Value in verification array = ", linteger_array(receive)
          end if
        end do
      end do

      do i = 1, size(integer_array)
        if(integer_array(i) /= linteger_array(i)) then
          ewrite(0, *) "Warning: Reference index ", i, " for halo " // halo_name(halo) // " failed verification"
          ewrite(0, *) "Reference = ", integer_array(i)
          ewrite(0, *) "Value in verification array = ", linteger_array(i)
        end if
      end do
    end if
#endif

    if(verifies) then
      ewrite(2, *) "halo_verifies_array_integer returning .true."
    else
      ewrite(2, *) "halo_verifies_array_integer returning .false."
    end if

  end function halo_verifies_array_integer

  function halo_verifies_array_real(halo, real_array) result(verifies)
    !!< Verify the halo against the supplied array of real data. Replaces
    !!< testhalo.

    type(halo_type), intent(in) :: halo
    real, dimension(:), intent(in) :: real_array

    real :: epsl

    logical :: verifies

#ifdef DDEBUG
    integer :: i, j, receive
#endif
    real, dimension(size(real_array)) :: lreal_array

    lreal_array = real_array
    call zero_halo_receives(halo, lreal_array)

    call halo_update(halo, lreal_array)

    epsl = spacing( maxval( abs( lreal_array ))) * 10000.
    call allmax(epsl)

    verifies = all(abs(real_array - lreal_array) < epsl )
#ifdef DDEBUG
    if(.not. verifies) then
      do i = 1, halo_proc_count(halo)
        do j = 1, halo_receive_count(halo, i)
          receive = halo_receive(halo, i, j)
          if(abs(real_array(receive) - lreal_array(receive)) >= epsl) then
            ewrite(0, *) "Warning: Halo receive ", receive, " for halo " // halo_name(halo) // " failed verification"
            ewrite(0, *) "Reference = ", real_array(receive)
            ewrite(0, *) "Value in verification array = ", lreal_array(receive)
          end if
        end do
      end do

      do i = 1, size(real_array)
         if(abs(real_array(i) - lreal_array(i)) >= epsl ) then
            ewrite(0, *) "Warning: Reference index ", i, " for halo " // halo_name(halo) // " failed verification"
            ewrite(0, *) "Reference = ", real_array(i)
            ewrite(0, *) "Value in verification array = ", lreal_array(i)
         end if
     end do
    end if
#endif

    if(verifies) then
      ewrite(2, *) "halo_verifies_array_real returning .true."
    else
      ewrite(2, *) "halo_verifies_array_real returning .false."
    end if

  end function halo_verifies_array_real

  function halo_verifies_scalar(halo, sfield) result(verifies)
    !!< Verify the halo against the supplied scalar field

    type(halo_type), intent(in) :: halo
    type(scalar_field), intent(in) :: sfield

    logical :: verifies

    verifies = halo_verifies(halo, sfield%val)

  end function halo_verifies_scalar

  function halo_verifies_vector_dim(halo, vfield, dim) result(verifies)
    !!< Verify the halo against one component of the supplied vector field

    type(halo_type), intent(in) :: halo
    type(vector_field), intent(in) :: vfield
    integer, intent(in) :: dim

    logical :: verifies

    type(scalar_field) :: sfield

    sfield = extract_scalar_field(vfield, dim)
    verifies = halo_verifies(halo, sfield)

  end function halo_verifies_vector_dim

  function halo_verifies_vector(halo, vfield) result(verifies)
    !!< Verify the halo against the supplied vector field

    type(halo_type), intent(in) :: halo
    type(vector_field), intent(in) :: vfield

    logical :: verifies

    integer :: i

    verifies = .true.
    do i = 1, vfield%dim
      verifies = halo_verifies(halo, vfield, i)
      if(.not. verifies) exit
    end do

  end function halo_verifies_vector

end module halos_communications
