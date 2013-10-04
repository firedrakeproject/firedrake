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

module parallel_tools

  use fldebug
  use mpi_interfaces
  use global_parameters, only: is_active_process, no_active_processes
  use iso_c_binding
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none

  private

  public :: halgetnb, halgetnb_simple, abort_if_in_parallel_region
  public :: allor, alland, allmax, allmin, allsum, allmean, allfequals,&
       get_active_nparts, getnprocs, getpinteger, getpreal, getprocno, getrank, &
       isparallel, parallel_filename, parallel_filename_len, &
       pending_communication, valid_communicator, next_mpi_tag, &
       MPI_COMM_FEMTOOLS, set_communicator

  integer(c_int), bind(c) :: MPI_COMM_FEMTOOLS = MPI_COMM_WORLD

  interface allmax
    module procedure allmax_integer, allmax_real
  end interface allmax

  interface allmin
    module procedure allmin_integer, allmin_real
  end interface allmin

  interface allsum
    module procedure allsum_integer, allsum_real, allsum_integer_vector, &
      & allsum_real_vector
  end interface allsum

  interface parallel_filename_len
    module procedure parallel_filename_no_extension_len, &
      &  parallel_filename_with_extension_len
  end interface

  interface parallel_filename
    module procedure parallel_filename_no_extension, &
      & parallel_filename_with_extension
  end interface

  interface pending_communication
    module procedure pending_communication_communicator
  end interface pending_communication

contains

  integer function next_mpi_tag()
#ifdef HAVE_MPI
    integer, save::last_tag=0, tag_ub=0
    integer flag, ierr
    if(tag_ub==0) then
       call MPI_Attr_get(MPI_COMM_FEMTOOLS, MPI_TAG_UB, tag_ub, flag, ierr)
    end if

    last_tag = mod(last_tag+1, tag_ub)
    if(last_tag==0) then
       last_tag = last_tag+1
    end if
    next_mpi_tag = last_tag
#else
    next_mpi_tag = 1
#endif
  end function next_mpi_tag

  function getprocno(communicator) result(procno)
    !!< This is a convenience routine which returns the MPI rank
    !!< number + 1 when MPI is being used and 1 otherwise.

    integer, optional, intent(in) :: communicator

    integer :: procno
#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    logical :: initialized

    call MPI_Initialized(initialized, ierr)
    if(initialized) then
       if(present(communicator)) then
          lcommunicator = communicator
       else
          lcommunicator = MPI_COMM_FEMTOOLS
       end if

       assert(valid_communicator(lcommunicator))
       call MPI_Comm_Rank(lcommunicator, procno, ierr)
       assert(ierr == MPI_SUCCESS)
       procno = procno + 1
    else
       procno = 1
    end if
#else
    procno = 1
#endif

  end function getprocno

  function getrank(communicator) result(rank)
    !!< This is a convience routine which returns the MPI rank
    !!< number of the process when MPI is being used and 0 otherwise.

    integer, optional, intent(in) :: communicator

    integer::rank
#ifdef HAVE_MPI
    integer :: ierr, lcommunicator

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    assert(valid_communicator(lcommunicator))
    call MPI_Comm_Rank(lcommunicator, rank, ierr)
    assert(ierr == MPI_SUCCESS)
#else
    rank = 0
#endif

  end function getrank

  function getnprocs(communicator) result(nprocs)
    !!< This is a convience routine which returns the number of processes
    !!< in a communicator (default MPI_COMM_FEMTOOLS) when MPI is being used and 1
    !!< otherwise.

    integer, optional, intent(in) :: communicator

    integer :: nprocs

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    logical :: initialized

    call MPI_Initialized(initialized, ierr)
    if(initialized) then
       if(present(communicator)) then
          assert(valid_communicator(communicator))
          lcommunicator = communicator
       else
          lcommunicator = MPI_COMM_FEMTOOLS
       end if

       assert(valid_communicator(lcommunicator))
       call MPI_Comm_Size(lcommunicator, nprocs, ierr)
       assert(ierr == MPI_SUCCESS)
    else
       nprocs = 1
    end if
#else
    nprocs = 1
#endif

  end function getnprocs

  function get_active_nparts(element_count, communicator) result(active_nparts)
    !!< Return the number of active partitions, based upon the supplied element
    !!< count. The inactive partitions must all be trailing processes.

    integer, intent(in) :: element_count
    integer, optional, intent(in) :: communicator

    integer :: active_nparts

#ifdef HAVE_MPI
#ifdef DDEBUG
    logical :: active_found
#endif
    integer, dimension(:), allocatable :: nelm
    integer :: lcommunicator, i, ierr

    assert(element_count >= 0)
    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    active_nparts = getnprocs()

    if(isparallel()) then
       allocate(nelm(getnprocs()))

       call MPI_allgather(element_count, 1, getpinteger(), &
         & nelm, 1, getpinteger(), lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)

      ! Interleaved #ifdef s assert that trailing process assumption is valid
      ! with debugging, and uses it to exit the loop early otherwise

#ifdef DDEBUG
       active_found = .false.
#endif
       do i = getnprocs(), 1, -1
         if(nelm(i) == 0) then
           active_nparts = active_nparts - 1
#ifdef DDEBUG
           if(active_found) then
             ewrite(-1, "(a,i0)") "For process ", i
             FLAbort("Inactive process is not a trailing process")
           end if
#endif
         else
#ifdef DDEBUG
           active_found = .true.
#else
           exit
#endif
         end if
       end do

       deallocate(nelm)
    end if
#else

    active_nparts = getnprocs()
#endif

  end function get_active_nparts

  logical function isparallel()
    !!< Return true if we are running in parallel, and false otherwise.

    isparallel = (getnprocs()>1)

  end function isparallel

  function getpinteger() result(pinteger)
    !!< This is a convience routine which returns the MPI integer type
    !!< being used. If MPI is not being used Pinteger is set to -1

    integer :: pinteger
#ifdef HAVE_MPI
    pinteger = MPI_INTEGER
#else
    pinteger = -1
#endif

  end function getpinteger

  function getpreal() result(preal)
    !!< This is a convience routine which returns the MPI real type
    !!< being used. If MPI is not being used PREAL is set to -1

    integer :: preal
#ifdef HAVE_MPI

#ifdef DOUBLEP
    preal = MPI_DOUBLE_PRECISION
#else
    preal = MPI_REAL
#endif

#else
    preal = -1
#endif

  end function getpreal

  logical function usingmpi()

    integer :: ierr
    usingmpi = .false.
#ifdef HAVE_MPI
    call MPI_Initialized(UsingMPI, IERR)
#endif

  end function usingmpi

  ! Abort run if we're in an OMP parallel region
  ! Call this routine at the start of functions that are known not to
  ! be thread safe (for example, populating caches) and should
  ! therefore never be called in a parallel region due to race
  ! conditions.
  subroutine abort_if_in_parallel_region()
#ifdef _OPENMP
    if (omp_in_parallel()) then
       FLAbort("Calling non-thread-safe code in OMP parallel region")
    endif
#else
    return
#endif
  end subroutine abort_if_in_parallel_region

  ! Array - points to the begining of the real array that stores the field values
  ! blockLen - the number of field values continiously stored per node
  ! stride - the distance distance between successive nodes data blocks
  ! fieldCnt - the total number of field variables to be communicated per node
  subroutine HalgetNB(Array, blockLen, stride, fieldCnt, ATOSEN, Gather, ATOREC, Scatter)
#ifdef HAVE_MPI
    integer PREAL
#endif

    integer, intent(in)::blockLen, stride, fieldCnt
    real Array(:)
    integer, intent(in)::ATOSEN(0:), ATOREC(0:)
    integer, intent(in)::Gather(:), Scatter(:)

    integer, ALLOCATABLE, DIMENSION(:)::recvRequest, sendRequest
    integer, ALLOCATABLE, DIMENSION(:,:)::Status
    SAVE recvRequest, sendRequest, Status

    integer, PARAMETER::TAG=12

    integer NProcs, Rank, I, J, Count, IERROR
    integer numBlocks, numBlocksPerNode, POS
    integer, ALLOCATABLE, DIMENSION(:)::haloType
    SAVE haloType
    integer typeRef
    integer MaxHaloLen
    integer, ALLOCATABLE, DIMENSION(:)::blens, disp
    SAVE blens, disp

    logical Initalized
    SAVE Initalized
    DATA Initalized /.false./

    integer NBits
    SAVE NBits
    DATA NBits /0/

#ifdef HAVE_MPI
    NProcs = GetNProcs()
    PREAL = GetPREAL()

    numBlocksPerNode = fieldCnt/blockLen

    ! Allocate space for the blens and disp arrays
    MaxHaloLen = 0
    do Rank=0, NProcs-1
       MaxHaloLen = MAX(MaxHaloLen, ATOREC(Rank+1)-ATOREC(Rank), ATOSEN(Rank+1)-ATOSEN(Rank))
    end do

    if(NBits.LT.(MaxHaloLen*numBlocksPerNode)) then
       NBits = MaxHaloLen*numBlocksPerNode

       if(ALLOCATED(blens)) DEallocate(blens)
       if(ALLOCATED(disp) ) DEallocate(disp)

       allocate( blens(NBits) )
       allocate(  disp(NBits) )
    end if

    ! The length of each block being sent. This is constant.
    do I=1, MaxHaloLen*numBlocksPerNode
       blens(I) = blockLen
    end do

    ! If this is the fist call to this routine then allocate some space
    ! for these arrays
    if(.NOT.Initalized) then
       allocate( recvRequest(0:NProcs-1) )
       allocate( sendRequest(0:NProcs-1) )
       allocate( Status(MPI_STATUS_SIZE, 0:NProcs-1) )
       allocate( haloType(NProcs*2) )
       do I=1, NProcs*2
          haloType(I) = MPI_DATATYPE_NULL
       end do
       Initalized = .true.
    end if

    typeRef = 1

    ! Set up all the receives first
    do Rank=0, NProcs-1
       Count = ATOREC(Rank+1)-ATOREC(Rank)
       IF (Count .EQ. 0) then
          ! Nothing to receive from Rank
          recvRequest(Rank) = MPI_REQUEST_NULL
       ELSE
          numBlocks = Count*numBlocksPerNode
          I = MaxHaloLen*numBlocksPerNode

          POS=1
          do I=0, numBlocksPerNode-1
             do J=1, Count
                disp(POS) = I*stride + (Scatter(ATOREC(Rank)+J-1) - 1)*blockLen
                POS = POS + 1
             end do
          end do

          call MPI_TYPE_INDEXED(numBlocks, blens, disp, PREAL, haloType(typeRef), IERROR)
          call MPI_TYPE_COMMIT(haloType(typeRef), IERROR)

          call MPI_IRECV(Array, 1, haloType(typeRef), Rank, TAG, MPI_COMM_FEMTOOLS, recvRequest(Rank), IERROR)

          typeRef = typeRef + 1
       end if
    end do

    ! Set up all the sends
    do Rank=0, NProcs-1
       Count = ATOSEN(Rank+1)-ATOSEN(Rank)
       IF (Count .EQ. 0) then
          ! Nothing to receive from Rank
          sendRequest(Rank) = MPI_REQUEST_NULL
       ELSE
          numBlocks = Count*numBlocksPerNode

          POS=1
          do I=0, numBlocksPerNode-1
             do J=1, Count
                disp(POS) = I*stride + (Gather(ATOSEN(Rank)+J-1) - 1)*blockLen
                POS = POS + 1
             end do
          end do

          call MPI_TYPE_INDEXED(numBlocks, blens, disp, PREAL, haloType(typeRef), IERROR)
          call MPI_TYPE_COMMIT(haloType(typeRef), IERROR)
          call MPI_ISEND(Array, 1, haloType(typeRef), Rank, TAG, MPI_COMM_FEMTOOLS, sendRequest(Rank), IERROR)

          typeRef = typeRef + 1
       end if
    end do

    ! Wait for everything to finish.
    call MPI_WAITALL(NProcs, sendRequest, Status, IERROR)
    call MPI_WAITALL(NProcs, recvRequest, Status, IERROR)

    ! Free all derived datatypes
    do I=1, typeRef-1
       call MPI_TYPE_FREE(haloType(I), IERROR)
       if(IERROR.NE.MPI_SUCCESS) then
          if(IERROR.EQ.MPI_ERR_TYPE) then
             ewrite(-1,*)  "Invalid datatype argument. May be an ", &
                  "uncommitted MPI_Datatype (see MPI_Type_commit)."
             call MPI_ABORT(MPI_COMM_FEMTOOLS, MPI_ERR_OTHER, IERROR)
          ELSE if(IERROR.EQ.MPI_ERR_ARG) then
             ewrite(-1,*)  "Invalid argument. Some argument is invalid and is not ", &
                  "identified by a specific error class (e.g., MPI_ERR_RANK)."
             call MPI_ABORT(MPI_COMM_FEMTOOLS, MPI_ERR_OTHER, IERROR)
          ELSE
             ewrite(-1,*)  "Unknown error from MPI_TYPE_FREE()"
             call MPI_ABORT(MPI_COMM_FEMTOOLS, MPI_ERR_OTHER, IERROR)
          end if
       end if
    end do

#endif

  end subroutine HalgetNB

  subroutine HalgetNB_simple(Array, FperN, ATOSEN, Gather, ATOREC, Scatter)
#ifdef HAVE_MPI
    integer PREAL
#endif

    real, intent(inout)::Array(:)
    integer, intent(in)::FperN

    integer, intent(in)::ATOSEN(0:), ATOREC(0:)
    integer, intent(in)::Gather(:), Scatter(:)

    integer, ALLOCATABLE, DIMENSION(:)::recvRequest, sendRequest
    integer, ALLOCATABLE, DIMENSION(:,:)::Status
    SAVE recvRequest, sendRequest, Status

    integer, PARAMETER::TAG=12

    integer NProcs, Rank, I, J, K, Count, toRecvCnt, IERROR
    SAVE NProcs

    real, ALLOCATABLE, DIMENSION(:)::bufferRecv, bufferSend
    SAVE bufferRecv, bufferSend

    integer BufferRecvLen
    SAVE BufferRecvLen
    DATA BufferRecvLen /0/

    integer BufferSendLen
    SAVE BufferSendLen
    DATA BufferSendLen /0/

    logical Initalized
    SAVE Initalized
    DATA Initalized /.false./

#ifdef HAVE_MPI
    PREAL = GetPREAL()

    ! If this is the fist call to this routine then allocate some space
    ! for these arrays
    if(.NOT.Initalized) then
       NProcs = GetNProcs()

       allocate( recvRequest(0:NProcs-1) )
       allocate( sendRequest(0:NProcs-1) )
       allocate( Status(MPI_STATUS_SIZE, 0:NProcs-1) )
       Initalized = .true.
    end if

    assert(size(atosen).eq.(NProcs+1))
    assert(size(atorec).eq.(NProcs+1))

    ! Make sure enough space has been allocated for the receive buffer
    if(BufferRecvLen.LT.(ATOREC(NProcs) - 1)) then
       if(ALLOCATED(bufferRecv)) DEallocate(bufferRecv)
       BufferRecvLen = ATOREC(NProcs) - 1

       allocate( bufferRecv(BufferRecvLen*FperN) )
    end if

    ! Make sure enough space has been allocated for the send buffer
    if(BufferSendLen.LT.(ATOSEN(NProcs) - 1)) then
       if(ALLOCATED(bufferSend)) DEallocate(bufferSend)
       BufferSendLen = ATOSEN(NProcs) - 1

       allocate( bufferSend(BufferSendLen*FperN) )
    end if

    ! Set up all the receives first
    toRecvCnt = 0
    do Rank=0, NProcs-1
       Count = ATOREC(Rank+1)-ATOREC(Rank)
       IF (Count .EQ. 0) then
          ! Nothing to receive from Rank
          recvRequest(Rank) = MPI_REQUEST_NULL
       ELSE
          call MPI_IRECV(bufferRecv(ATOREC(Rank)*FperN), Count*FperN, PREAL, &
               Rank, TAG, MPI_COMM_FEMTOOLS, recvRequest(Rank), IERROR)
          toRecvCnt = toRecvCnt + 1
       end if
    end do

    ! Set up all the sends
    do Rank=0, NProcs-1
       Count = ATOSEN(Rank+1)-ATOSEN(Rank)
       IF (Count .EQ. 0) then
          ! Nothing to receive from Rank
          sendRequest(Rank) = MPI_REQUEST_NULL
       ELSE
          do I=ATOSEN(Rank), ATOSEN(Rank+1)-1
             do J=0, FperN-1
                bufferSend(I*FperN+J) = Array(Gather(I)*FperN + J)
             end do
          end do

          call MPI_ISEND(bufferSend(ATOSEN(Rank)*FperN), Count*FperN, PREAL, &
               Rank, TAG, MPI_COMM_FEMTOOLS, sendRequest(Rank), IERROR)
       end if
    end do

    ! Wait for receives to finish.
    do K=1, toRecvCnt
       call MPI_WAITANY(NProcs, recvRequest, J, Status(:,0), IERROR)

       ! Unpack received data
       Rank = Status(MPI_SOURCE, 0)
       do I=ATOREC(Rank), ATOREC(Rank+1)-1
          do J=0, FperN-1
             Array(Scatter(I)*FperN + J) = bufferRecv(I*FperN+J)
          end do
       end do
    end do

    call MPI_WAITALL(NProcs, sendRequest, Status, IERROR)
#endif

  end subroutine HalgetNB_simple

  function pending_communication_communicator(communicator) result(pending)
    !!< Return whether there is a pending communication for the supplied communicator.

    integer, optional, intent(in) :: communicator

    logical :: pending

    integer :: lcommunicator

#ifdef HAVE_MPI
    integer :: ierr, ipending

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    call mpi_iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, lcommunicator, ipending, MPI_STATUS_IGNORE, ierr)
    assert(ierr == MPI_SUCCESS)

    pending = (ipending /= 0)

    ! Note - removing this mpi_barrier could result in a false
    ! positive on another process.
    call mpi_barrier(lcommunicator, ierr)
    assert(ierr == MPI_SUCCESS)
#else
    pending = .false.
#endif

  end function pending_communication_communicator

  function valid_communicator(communicator) result(valid)
    !!< Return whether the supplied MPI communicator is valid

    integer, intent(in) :: communicator

    logical :: valid

#ifdef HAVE_MPI
    integer :: ierr, size

    call mpi_comm_size(communicator, size, ierr)

    valid = (ierr == MPI_SUCCESS)
#else
    valid = .false.
#endif

  end function valid_communicator

  subroutine allor(value, communicator)
    !!< Or the logical value across all processes

    logical, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    logical :: or

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
      assert(valid_communicator(lcommunicator))
      call mpi_allreduce(value, or, 1, MPI_LOGICAL, MPI_LOR, lcommunicator, ierr)
      assert(ierr == MPI_SUCCESS)
      value = or
    end if
#endif

  end subroutine allor

  subroutine alland(value, communicator)
    !!< And the logical value across all processes

    logical, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    logical :: and

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
      assert(valid_communicator(lcommunicator))
      call mpi_allreduce(value, and, 1, MPI_LOGICAL, MPI_LAND, lcommunicator, ierr)
      assert(ierr == MPI_SUCCESS)
      value = and
    end if
#endif

  end subroutine alland

  subroutine allmax_integer(value, communicator)
    !!< Find the maxmimum value across all processes

    integer, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator, maximum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
       assert(valid_communicator(lcommunicator))
       call MPI_Allreduce(value, maximum, 1, getpinteger(), MPI_MAX, lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)
       value = maximum
    end if
#endif

  end subroutine allmax_integer

  subroutine allmax_real(value, communicator)
    !!< Find the maxmimum value across all processes

    real, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    real :: maximum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
       assert(valid_communicator(lcommunicator))
       call mpi_allreduce(value, maximum, 1, getpreal(), MPI_MAX, lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)
       value = maximum
    end if
#endif

  end subroutine allmax_real

  subroutine allmin_integer(value, communicator)
    !!< Find the minimum value across all processes

    integer, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: lcommunicator, mierr, minimum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(IsParallel()) then
       assert(valid_communicator(lcommunicator))
       call MPI_Allreduce(value, minimum, 1, getpinteger(), MPI_MIN, lcommunicator, mierr)
       assert(mierr == MPI_SUCCESS)
       value = minimum
    end if
#endif

  end subroutine allmin_integer

  subroutine allmin_real(value, communicator)
    !!< Find the minimum value across all processes

    real, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: lcommunicator, mierr
    real :: minimum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(IsParallel()) then
       assert(valid_communicator(lcommunicator))
       call MPI_Allreduce(value, minimum, 1, getpreal(), MPI_MIN, lcommunicator, mierr)
       assert(mierr == MPI_SUCCESS)
       value = minimum
    end if
#endif

  end subroutine allmin_real

  subroutine allsum_integer(value, communicator)
    !!< Sum the integer value across all processes

    integer, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    integer :: sum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
       assert(valid_communicator(lcommunicator))
       sum = 0.0
       call MPI_Allreduce(value, sum, 1, getpinteger(), MPI_SUM, lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)
       value = sum
    end if
#endif

  end subroutine allsum_integer

  subroutine allsum_real(value, communicator)
    !!< Sum the real value across all processes

    real, intent(inout) :: value
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: ierr, lcommunicator
    real :: sum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
       assert(valid_communicator(lcommunicator))
       sum = 0.0
       call MPI_Allreduce(value, sum, 1, getpreal(), MPI_SUM, lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)
       value = sum
    end if
#endif

  end subroutine allsum_real

  subroutine allmean(value, communicator)
    !!< Sum the real value across all processes

    real, intent(inout) :: value
    integer, optional, intent(in) :: communicator

    call allsum(value, communicator = communicator)
    value = value / getnprocs(communicator = communicator)

  end subroutine allmean

  subroutine allsum_integer_vector(value, communicator)
    !!< Sum the value across all processes

    integer, intent(inout) :: value(:)
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: lcommunicator, ierr
    integer, dimension(size(value)) :: sum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
       assert(valid_communicator(lcommunicator))
       sum = 0
       call MPI_Allreduce(value, sum, size(value), getpinteger(), MPI_SUM, lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)
       value = sum
    end if
#endif

  end subroutine allsum_integer_vector

  subroutine allsum_real_vector(value, communicator)
    !!< Sum the value across all processes

    real, intent(inout) :: value(:)
    integer, optional, intent(in) :: communicator

#ifdef HAVE_MPI
    integer :: lcommunicator, ierr
    real, dimension(size(value)) :: sum

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(isparallel()) then
       assert(valid_communicator(lcommunicator))
       sum = 0.0
       call MPI_Allreduce(value, sum, size(value), getpreal(), MPI_SUM, lcommunicator, ierr)
       assert(ierr == MPI_SUCCESS)
       value = sum
    end if
#endif

  end subroutine allsum_real_vector

  function allfequals(value, communicator, tol)
    !!< Return if all of value are almost equal across all processes

    real, intent(in) :: value
    integer, optional, intent(in) :: communicator
    real, optional, intent(in) :: tol

    logical :: allfequals

#ifdef HAVE_MPI
    integer :: lcommunicator, ierr
    real :: eps, zero_value

    if(present(communicator)) then
      lcommunicator = communicator
    else
      lcommunicator = MPI_COMM_FEMTOOLS
    end if

    if(present(tol)) then
      eps = tol
    else
      eps = 100.0 * epsilon(0.0)
    end if

    if(isparallel()) then
      assert(valid_communicator(lcommunicator))
      if(getrank(communicator = lcommunicator) == 0) zero_value = value
      call mpi_bcast(zero_value, 1, getpreal(), 0, lcommunicator, ierr)
      assert(ierr == MPI_SUCCESS)

      allfequals = abs(zero_value - value) < max(eps, abs(value) * eps)
      call alland(allfequals, communicator = lcommunicator)
    else
      allfequals = .true.
    end if
#else
    allfequals = .true.
#endif

  end function allfequals

  pure function parallel_filename_no_extension_len(filename) result(length)
    !!< Return the (maximum) length of a string containing:
    !!<   [filename]_[process number]

    character(len = *), intent(in) :: filename

    integer :: length

    length = len_trim(filename) + 1 + floor(log10(real(huge(0)))) + 1

  end function parallel_filename_no_extension_len

  function parallel_filename_no_extension(filename) result(pfilename)
    !!< Return a string containing:
    !!<   [filename]-[process-number]
    !!< Note that is it important to trim the returned string.

    character(len = *), intent(in) :: filename

    character(len = parallel_filename_len(filename)) :: pfilename

    if (is_active_process .and. no_active_processes == 1) then
      write(pfilename, "(a)") trim(filename)
    else
      write(pfilename, "(a, i0)") trim(filename) // "_", getrank()
    end if

  end function parallel_filename_no_extension

  pure function parallel_filename_with_extension_len(filename, extension) result(length)
    !!< Return the (maximum) length of a string containing:
    !!<   [filename]_[process number].[extension]

    character(len = *), intent(in) :: filename
    character(len = *), intent(in) :: extension

    integer :: length

    length = parallel_filename_len(filename) + len_trim(extension)

  end function parallel_filename_with_extension_len

  function parallel_filename_with_extension(filename, extension)  result(pfilename)
    !!< Return a string containing:
    !!<   [filename]-[process-number][extension]
    !!< Note that is it important to trim the returned string.

    character(len = *), intent(in) :: filename
    character(len = *), intent(in) :: extension

    character(len = parallel_filename_len(filename, extension)) :: pfilename

    pfilename = trim(parallel_filename(filename)) // trim(extension)

  end function parallel_filename_with_extension

  subroutine set_communicator(communicator) bind(c)
    !!< Set mpi_comm_femtools to the provided communicator
    !!< If this subroutine is not used, mpi_comm_femtools = mpi_comm_world

    integer(c_int), intent(in) :: communicator

    MPI_COMM_FEMTOOLS = communicator

  end subroutine set_communicator

end module parallel_tools
