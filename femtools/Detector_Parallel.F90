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

module detector_parallel
  use state_module
  use fields
  use spud
  use integer_hash_table_module
  use detector_data_types
  use detector_tools
  use halo_data_types
  use halos_numbering
  use mpi_interfaces
  use pickers

  implicit none

  private

  public :: distribute_detectors, exchange_detectors, register_detector_list, &
            get_num_detector_lists, get_registered_detector_lists, &
            deallocate_detector_list_array, sync_detector_coordinates

  type(detector_list_ptr), dimension(:), allocatable, target, save :: detector_list_array
  integer :: num_detector_lists = 0

contains

  subroutine register_detector_list(detector_list)
    ! Register a detector list, so that adaptivity and Zoltan will distribute detectors on adapts
    ! This adds a pointer to the given detector list to detector_list_array and assign detector list ID
    type(detector_linked_list), target, intent(inout) :: detector_list

    type(detector_list_ptr), dimension(:), allocatable :: tmp_list_array
    integer :: i, old_size

    ! Allocate a new detector list
    if (allocated(detector_list_array)) then
       old_size = size(detector_list_array)
       allocate(tmp_list_array(old_size+1))
       do i=1, old_size
          tmp_list_array(i)%ptr=>detector_list_array(i)%ptr
       end do
       deallocate(detector_list_array)
       allocate(detector_list_array(old_size+1))
       do i=1, old_size
          detector_list_array(i)%ptr=>tmp_list_array(i)%ptr
       end do
       detector_list_array(old_size+1)%ptr=>detector_list
    else
       ! Allocate and return first detector list
       allocate(detector_list_array(1))
       detector_list_array(1)%ptr=>detector_list
    end if

    ! Advance counter and assign list ID
    num_detector_lists=num_detector_lists+1
    detector_list%id=num_detector_lists

  end subroutine register_detector_list

  subroutine get_registered_detector_lists(all_registered_lists)
    ! Return a pointer to a lists of pointers to all registered detector lists
    type(detector_list_ptr), dimension(:), pointer, intent(out) :: all_registered_lists

    all_registered_lists=>detector_list_array
  end subroutine get_registered_detector_lists

  function get_num_detector_lists()
    ! Return the number of registered detector lists
    integer :: get_num_detector_lists

    get_num_detector_lists=num_detector_lists
  end function get_num_detector_lists

  subroutine deallocate_detector_list_array()
    type(detector_list_ptr), dimension(:), pointer :: detector_lists
    integer :: i

    if (allocated(detector_list_array)) then
       call get_registered_detector_lists(detector_lists)
       do i=1, get_num_detector_lists()
          call deallocate(detector_lists(i)%ptr)
       end do
       deallocate(detector_list_array)
       num_detector_lists = 0
    end if

  end subroutine deallocate_detector_list_array

  subroutine distribute_detectors(state, detector_list)
    ! Loop over all the detectors in the list and check that I own the element they are in.
    ! If not, they need to be sent to the processor owner before adaptivity happens
    type(state_type), intent(in) :: state
    type(detector_linked_list), intent(inout) :: detector_list

    type(detector_linked_list), dimension(:), allocatable :: send_list_array
    type(detector_linked_list) :: detector_bcast_list, lost_detectors_list
    type(detector_type), pointer :: detector, node_to_send, bcast_detector
    type(vector_field), pointer :: xfield
    integer :: i,j,k, nprocs, all_send_lists_empty, processor_owner, bcast_count, &
               ierr, ndata_per_det, bcast_rounds, round, accept_detector
    integer, dimension(:), allocatable :: ndets_being_bcast
    real, allocatable :: send_buff(:), recv_buff(:)
    type(element_type), pointer :: shape

    ewrite(2,*) "In distribute_detectors"

    xfield => extract_vector_field(state,"Coordinate")

    ! We allocate a point-to-point sendlist for every processor
    nprocs=getnprocs()
    allocate(send_list_array(nprocs))
    bcast_count=0

    detector => detector_list%first
    do while (associated(detector))

       if (detector%element>0) then
          ! If we know the element get the owner and send point-to-point
          processor_owner=element_owner(xfield%mesh,detector%element)

          if (processor_owner/= getprocno()) then
             node_to_send => detector
             detector => detector%next

             call move(node_to_send, detector_list, send_list_array(processor_owner))
          else
             detector => detector%next
          end if
       else
          ! If we dont know the element we will need to broadcast
          ewrite(2,*) "Found non-local detector, triggering broadcast..."
          bcast_count = bcast_count + 1

          ! Move detector to broadcast detector list
          bcast_detector => detector
          detector => detector%next
          call move(bcast_detector, detector_list, detector_bcast_list)
       end if
    end do

    ! Exchange detectors if there are any detectors to exchange
    ! via the point-to-point sendlists
    all_send_lists_empty=0
    do k=1, nprocs
       if (send_list_array(k)%length/=0) then
          all_send_lists_empty=1
       end if
    end do
    call allmax(all_send_lists_empty)
    if (all_send_lists_empty/=0) then
       call exchange_detectors(state,detector_list,send_list_array)
    end if

    ! Make sure send lists are empty and deallocate them
    do k=1, nprocs
       assert(send_list_array(k)%length==0)
    end do
    deallocate(send_list_array)

    ! Sanity check
    detector => detector_list%first
    do while (associated(detector))
       assert(element_owner(xfield%mesh,detector%element)==getprocno())
       detector=>detector%next
    end do

    ! Find out how many unknown detectors each process wants to broadcast
    allocate(ndets_being_bcast(getnprocs()))
    call mpi_allgather(bcast_count, 1, getPINTEGER(), ndets_being_bcast, 1 , getPINTEGER(), MPI_COMM_FEMTOOLS, ierr)
    assert(ierr == MPI_SUCCESS)

    ! If there are no unknow detectors exit
    if (all(ndets_being_bcast == 0)) then
       return
    else
       ewrite(2,*) "Broadcast required, initialising..."

       ! If there are unknow detectors we need to broadcast.
       ! Since we can not be sure whether any processor will accept a detector
       ! we broadcast one at a time so we can make sure somebody accepted it.
       ! If there are no takers we keep the detector with element -1,
       ! becasue it has gone out of the domain, and this will be caught at a later stage.
       bcast_rounds = maxval(ndets_being_bcast)
       call allmax(bcast_rounds)
       do round=1, bcast_rounds
          ndata_per_det = detector_buffer_size(xfield%dim,.false.)

          ! Broadcast detectors whose new owner we can't identify
          do i=1,getnprocs()
             if (ndets_being_bcast(i) >= round) then

                if (i == getprocno() .and. bcast_count>=round) then
                   ! Allocate memory for the detector we're going to send
                   allocate(send_buff(ndata_per_det))

                   ! Pack the first detector from the bcast_list
                   detector=>detector_bcast_list%first
                   call pack_detector(detector, send_buff(1:ndata_per_det), xfield%dim)
                   call delete(detector, detector_bcast_list)

                   ! Broadcast the detectors you want to send
                   ewrite(2,*) "Broadcasting detector"
                   call mpi_bcast(send_buff,ndata_per_det, getPREAL(), i-1, MPI_COMM_FEMTOOLS, ierr)
                   assert(ierr == MPI_SUCCESS)

                   ! This allmax matches the one below
                   accept_detector = 0
                   call allmax(accept_detector)

                   ! If we're the sender and nobody accepted the detector
                   ! we keep it, to deal with it later in the I/O routines
                   if (accept_detector == 0 .and. i == getprocno()) then
                      ewrite(2,*) "WARNING: Could not find processor for detector. Detector is probably outside the domain!"

                      ! Unpack detector again and put in a temporary lost_detectors_list
                      shape=>ele_shape(xfield,1)
                      detector=>null()
                      call allocate(detector, xfield%dim, local_coord_count(shape))
                      call unpack_detector(detector, send_buff(1:ndata_per_det), xfield%dim)
                      call insert(detector, lost_detectors_list)
                   end if

                   deallocate(send_buff)
                else
                   ! Allocate memory to receive into
                   allocate(recv_buff(ndata_per_det))

                   ! Receive broadcast
                   ewrite(2,*) "Receiving detector from process ", i
                   call mpi_bcast(recv_buff,ndata_per_det, getPREAL(), i-1, MPI_COMM_FEMTOOLS, ierr)
                   assert(ierr == MPI_SUCCESS)

                   ! Allocate and unpack the detector
                   shape=>ele_shape(xfield,1)
                   detector=>null()
                   call allocate(detector, xfield%dim, local_coord_count(shape))
                   call unpack_detector(detector, recv_buff(1:ndata_per_det), xfield%dim)

                   ! Try to find the detector position locally
                   call picker_inquire(xfield, detector%position, detector%element, detector%local_coords, global=.false.)
                   if (detector%element>0) then
                      ! We found a new home...
                      call insert(detector, detector_list)
                      accept_detector = 1
                      ewrite(2,*) "Accepted detector"
                   else
                      call delete(detector)
                      accept_detector = 0
                      ewrite(2,*) "Rejected detector"
                   end if

                   ! This allmax matches the one above
                   call allmax(accept_detector)
                   deallocate(recv_buff)
                end if
             end if
          end do
       end do

       ! Now move the lost detectors into our list again
       call move_all(lost_detectors_list, detector_list)
       assert(lost_detectors_list%length==0)
    end if

    ewrite(2,*) "Finished broadcast"

  end subroutine distribute_detectors

  subroutine exchange_detectors(state, detector_list, send_list_array)
    ! This subroutine serialises send_list_array, sends it,
    ! receives serialised detectors from all procs and unpacks them.
    type(state_type), intent(in) :: state
    type(detector_linked_list), intent(inout) :: detector_list
    type(detector_linked_list), dimension(:), intent(inout) :: send_list_array

    real, dimension(:,:), allocatable :: detector_buffer
    type(detector_type), pointer :: detector, detector_received
    type(vector_field), pointer :: xfield
    type(halo_type), pointer :: ele_halo
    type(integer_hash_table) :: ele_numbering_inverse
    integer :: j, dim, count, n_stages, target_proc, receive_proc, &
               det_size, ndet_to_send, ndet_received, &
               halo_level, nprocs, IERROR
    integer, parameter ::TAG=12
    integer, dimension(:), allocatable :: sendRequest, status
    logical :: have_update_vector

    ewrite(2,*) "In exchange_detectors"

    ! We want a sendlist for every processor
    nprocs=getnprocs()
    assert(size(send_list_array)==nprocs)

    xfield => extract_vector_field(state,"Coordinate")
    dim=xfield%dim
    allocate( sendRequest(nprocs) )

    ! Get the element halo
    halo_level = element_halo_count(xfield%mesh)
    if (halo_level /= 0) then
       ele_halo => xfield%mesh%element_halos(halo_level)
    else
       ewrite(-1,*) "In exchange_detectors: No element halo found to translate detector%element"
       FLAbort("Exchanging detectors requires halo_level > 0")
    end if

    ! Get buffer size, depending on whether RK-GS parameters are still allocated
    have_update_vector=associated(detector_list%move_parameters)
    if(have_update_vector) then
       n_stages=detector_list%move_parameters%n_stages
       det_size=detector_buffer_size(dim,have_update_vector,n_stages)
    else
       det_size=detector_buffer_size(dim,have_update_vector)
    end if

    ! Send to all procs
    do target_proc=1, nprocs
       ndet_to_send=send_list_array(target_proc)%length
       allocate(detector_buffer(ndet_to_send,det_size))

       if (ndet_to_send>0) then
          ewrite(2,*) " Sending", ndet_to_send, "detectors to process", target_proc
       end if

       detector => send_list_array(target_proc)%first
       if (ndet_to_send>0) then
          j=1
          do while (associated(detector))

             ! translate detector element to universal element
             assert(detector%element>0)
             detector%element = halo_universal_number(ele_halo, detector%element)

             if (have_update_vector) then
                call pack_detector(detector, detector_buffer(j,1:det_size), dim, nstages=n_stages)
             else
                call pack_detector(detector, detector_buffer(j,1:det_size), dim)
             end if

             ! delete also advances detector
             call delete(detector, send_list_array(target_proc))
             j = j+1
          end do
       end if

       ! getprocno() returns the rank of the processor + 1, hence target_proc-1
       call MPI_ISEND(detector_buffer,size(detector_buffer), &
            & getpreal(), target_proc-1, TAG, MPI_COMM_FEMTOOLS, sendRequest(target_proc), IERROR)
       assert(ierror == MPI_SUCCESS)

       ! deallocate buffer after sending
       deallocate(detector_buffer)
    end do

    allocate( status(MPI_STATUS_SIZE) )
    call get_universal_numbering_inverse(ele_halo, ele_numbering_inverse)

    ! Receive from all procs
    do receive_proc=1, nprocs
       call MPI_PROBE(receive_proc-1, TAG, MPI_COMM_FEMTOOLS, status(:), IERROR)
       assert(ierror == MPI_SUCCESS)

       call MPI_GET_COUNT(status(:), getpreal(), count, IERROR)
       assert(ierror == MPI_SUCCESS)

       ndet_received=count/det_size
       allocate(detector_buffer(ndet_received,det_size))

       if (ndet_received>0) then
          ewrite(2,*) " Receiving", ndet_received, "detectors from process", receive_proc
       end if

       call MPI_Recv(detector_buffer,count, getpreal(), status(MPI_SOURCE), TAG, MPI_COMM_FEMTOOLS, MPI_STATUS_IGNORE, IERROR)
       assert(ierror == MPI_SUCCESS)

       do j=1, ndet_received
          allocate(detector_received)

          ! Unpack routine uses ele_numbering_inverse to translate universal element
          ! back to local detector element
          if (have_update_vector) then
             call unpack_detector(detector_received,detector_buffer(j,1:det_size),dim,&
                    global_to_local=ele_numbering_inverse,coordinates=xfield,nstages=n_stages)
          else
             call unpack_detector(detector_received,detector_buffer(j,1:det_size),dim,&
                    global_to_local=ele_numbering_inverse,coordinates=xfield)
          end if

          ! If there is a list of detector names, use it, otherwise set ID as name
          if (allocated(detector_list%detector_names)) then
             detector_received%name=detector_list%detector_names(detector_received%id_number)
          else
             detector_received%name=int2str(detector_received%id_number)
          end if

          call insert(detector_received, detector_list)
       end do
       deallocate(detector_buffer)
    end do

    call MPI_WAITALL(nprocs, sendRequest, MPI_STATUSES_IGNORE, IERROR)
    assert(ierror == MPI_SUCCESS)

    call deallocate(ele_numbering_inverse)

    ewrite(2,*) "Exiting exchange_detectors"

  end subroutine exchange_detectors

  subroutine sync_detector_coordinates(state)
    ! Re-synchronise the physical and parametric coordinates
    ! of all detectors detectors in all lists after mesh movement.
    type(state_type), intent(in) :: state

    type(detector_list_ptr), dimension(:), pointer :: detector_list_array => null()
    type(vector_field), pointer :: coordinate_field => null()
    type(detector_type), pointer :: detector
    integer :: i

    ! Re-evaluate detector coordinates for every detector in all lists
    if (get_num_detector_lists()>0) then
       coordinate_field=>extract_vector_field(state,"Coordinate")
       call get_registered_detector_lists(detector_list_array)
       do i = 1, size(detector_list_array)

          ! In order to let detectors drift with the mesh
          ! we update det%position from the parametric coordinates
          if (detector_list_array(i)%ptr%move_with_mesh) then
             detector=>detector_list_array(i)%ptr%first
             do while (associated(detector))
                detector%position=detector_value(coordinate_field, detector)
                detector=>detector%next
             end do
          ! By default update detector element and local_coords from position
          else
             call search_for_detectors(detector_list_array(i)%ptr, coordinate_field)

             call distribute_detectors(state, detector_list_array(i)%ptr)
          end if
       end do
    end if

  end subroutine sync_detector_coordinates

end module detector_parallel
