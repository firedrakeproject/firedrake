!    Copyright (C) 2009 Imperial College London and others.
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

module memory_diagnostics
  use fldebug
  use global_parameters, only : integer_size, real_size
  use spud
  use parallel_tools
  implicit none

  private

  type memory_log
     !!< Current memory usage, and minimum and maximum since last reset.
     !!
     !! This is a sequence type to facilitate feeding it to mpi.
     sequence
     !! We use reals here to prevent overflows at 2G.
     real :: min=0, max=0, current=0
  end type memory_log

  ! This is the total number of bins we sort memory types into.
  integer, parameter :: MEMORY_TYPES=7
  ! This is the number of statistics we store for each memory type.
  integer, parameter :: MEMORY_STATS=3

  integer, parameter :: MESH_TYPE=1, &
       SCALAR_FIELD=2, &
       VECTOR_FIELD=3, &
       TENSOR_FIELD=4, &
       CSR_SPARSITY=5, &
       CSR_MATRIX=6, &
       TRANSFORM_CACHE=7

  character(len=20), dimension(0:MEMORY_TYPES) :: memory_type_names= (/ &
       "TotalMemory         ", &
       "MeshMemory          ", &
       "ScalarFieldMemory   ", &
       "VectorFieldMemory   ", &
       "TensorFieldMemory   ", &
       "MatrixSparsityMemory", &
       "MatrixMemory        ", &
       "TransformCacheMemory"  &
       /)

  character(len=7), dimension(MEMORY_STATS) :: memory_stat_names = (/ &
       "current", &
       "min    ", &
       "max    " &
       /)

  ! This vector should have as many entries as the types of data we track above.
  type(memory_log), dimension(0:MEMORY_TYPES), target, save :: memory_usage

  ! whether to write to the log for each (de)allocate:
  logical :: log_allocates

#ifdef HAVE_MEMORY_STATS
  public :: memory_log, memory_type_names, MEMORY_TYPES, memory_stat_names,&
       & memory_usage, register_allocation, register_deallocation,&
       & register_temporary_memory, reset_memory_logs, write_memory_stats, &
       & print_current_memory_stats, print_memory_stats
#endif


contains

  subroutine register_allocation(object_type, data_type, size, name)
    !!< Register the allocation of a new object.
    !!<
    !!< We sort the object data on the basis of the object_type while the
    !!< data_type and the size tell us how much memory has been allocated.
    character(len=*), intent(in) :: object_type, data_type
    integer, intent(in) :: size
    character(len=*), intent(in), optional :: name

    integer :: data_size
    type(memory_log), pointer :: this_log

    select case(data_type)
    case("real")
       data_size=real_size
    case("integer")
       data_size=integer_size
    case default
       FLAbort(trim(data_type)//" is not a supported data type.")
    end select

    select case(object_type)
    case("mesh_type")
       this_log=>memory_usage(MESH_TYPE)
    case("scalar_field")
       this_log=>memory_usage(SCALAR_FIELD)
    case("vector_field")
       this_log=>memory_usage(VECTOR_FIELD)
    case("tensor_field")
       this_log=>memory_usage(TENSOR_FIELD)
    case("csr_sparsity")
       this_log=>memory_usage(CSR_SPARSITY)
    case("csr_matrix")
       this_log=>memory_usage(CSR_MATRIX)
    case("transform_cache")
       this_log=>memory_usage(TRANSFORM_CACHE)
    case default
       FLAbort(trim(data_type)//" is not a supported object type.")
    end select

    if (log_allocates) then
      if (present(name)) then
        ewrite(2,*) "Allocating ",size*data_size," bytes of ",trim(object_type)&
           &, ", name ",trim(name)
      else
        ewrite(2,*) "Allocating ",size*data_size," bytes of ",trim(object_type)
      end if
    end if

    this_log%current=this_log%current + size*data_size

    this_log%max=max(this_log%max, this_log%current)
    this_log%min=min(this_log%min, this_log%current)

    ! Also account for total memory.
    memory_usage(0)%current=memory_usage(0)%current + size*data_size

    memory_usage(0)%max=max(memory_usage(0)%max, memory_usage(0)%current)
    memory_usage(0)%min=min(memory_usage(0)%min, memory_usage(0)%current)

  end subroutine register_allocation

  subroutine register_deallocation(object_type, data_type, size, name)
    !!< Register the deallocation of a new object.
    !!<
    !!< We sort the object data on the basis of the object_type while the
    !!< data_type and the size tell us how much memory has been allocated.
    character(len=*), intent(in) :: object_type, data_type
    integer, intent(in) :: size
    character(len=*), intent(in), optional :: name

    integer :: data_size
    type(memory_log), pointer :: this_log

    select case(data_type)
    case("real")
       data_size=real_size
    case("integer")
       data_size=integer_size
    case default
       FLAbort(trim(data_type)//" is not a supported data type.")
    end select

    select case(object_type)
    case("mesh_type")
       this_log=>memory_usage(MESH_TYPE)
    case("scalar_field")
       this_log=>memory_usage(SCALAR_FIELD)
    case("vector_field")
       this_log=>memory_usage(VECTOR_FIELD)
    case("tensor_field")
       this_log=>memory_usage(TENSOR_FIELD)
    case("csr_sparsity")
       this_log=>memory_usage(CSR_SPARSITY)
    case("csr_matrix")
       this_log=>memory_usage(CSR_MATRIX)
    case("transform_cache")
       this_log=>memory_usage(TRANSFORM_CACHE)
    case default
       FLAbort(trim(data_type)//" is not a supported object type.")
    end select

    if (log_allocates) then
      if (present(name)) then
        ewrite(2,*) "Deallocating ",size*data_size," bytes of ",trim(object_type)&
           &, ", name ",trim(name)
      else
        ewrite(2,*) "Deallocating ",size*data_size," bytes of ",trim(object_type)
      end if
    end if

    this_log%current=this_log%current - size*data_size

    this_log%max=max(this_log%max, this_log%current)
    this_log%min=min(this_log%min, this_log%current)

    ! Also account for total memory.
    memory_usage(0)%current=memory_usage(0)%current - size*data_size

    memory_usage(0)%max=max(memory_usage(0)%max, memory_usage(0)%current)
    memory_usage(0)%min=min(memory_usage(0)%min, memory_usage(0)%current)

  end subroutine register_deallocation

  subroutine register_temporary_memory(object_type, data_type, size)
    !!< Register some memory which has been used but already freed.
    character(len=*), intent(in) :: object_type, data_type
    integer, intent(in) :: size

    call register_allocation(object_type, data_type, size)
    call register_deallocation(object_type, data_type, size)

  end subroutine register_temporary_memory

  subroutine reset_memory_logs
    !!< Set the minimum and maximum values in the memory logs back to the
    !!< current value. This is primarily of use for calculating the peak
    !!< and minumum memory consumption during a timestep.
    integer :: i

    log_allocates=have_option('/io/log_output/memory_diagnostics')

    do i=0,MEMORY_TYPES
       memory_usage(i)%max=memory_usage(i)%current
       memory_usage(i)%min=memory_usage(i)%current
    end do

  end subroutine reset_memory_logs

  subroutine write_memory_stats(diag_unit, format)
    !!< Write the current memory stats out on the unit provided.
    integer, intent(in) :: diag_unit
    character(len=*), intent(in) :: format
    type(memory_log), dimension(0:MEMORY_TYPES) ::&
         & global_memory_usage
    real, dimension((MEMORY_TYPES+1)*MEMORY_STATS) :: buffer

    integer :: i

    if (isparallel()) then
       buffer=transfer(memory_usage, buffer)
       call allsum(buffer)
       global_memory_usage=transfer(buffer, memory_usage)
    else
       global_memory_usage=memory_usage
    end if

    ! Only output from process 0.
    if (getrank()==0) then
       do i=0,MEMORY_TYPES

          write(diag_unit, trim(format), advance="no") &
               memory_usage(i)%current
          write(diag_unit, trim(format), advance="no") &
               memory_usage(i)%min
          write(diag_unit, trim(format), advance="no") &
               memory_usage(i)%max

       end do
    end if

  end subroutine write_memory_stats

  subroutine print_memory_stats(priority)
    !!< Print out the current memory allocation statistics using ewrites
    !!< with the given priority.
    integer, intent(in) :: priority
    integer :: i

    ewrite(priority,*) "Memory usage in bytes:"
    ewrite(priority,'(a30,3a15)') "", "current", "min", "max"

    do i=0,MEMORY_TYPES

       ewrite(priority,'(a30,3f15.0)') memory_type_names(i), &
            memory_usage(i)%current, &
            memory_usage(i)%min, &
            memory_usage(i)%max
    end do

  end subroutine print_memory_stats

  subroutine print_current_memory_stats(priority)
    !!< Print out the current memory allocation statistics using ewrites
    !!< with the given priority.
    integer, intent(in) :: priority
    integer :: i

    if (all(memory_usage%current==0)) then
       ewrite(1,*) "No registered memory in use."
       return
    end if

    ewrite(priority,*) "Current memory usage in bytes:"

    do i=0,MEMORY_TYPES

       ewrite(priority,'(a30,f15.0)') memory_type_names(i), memory_usage(i)%current

    end do

  end subroutine print_current_memory_stats

end module memory_diagnostics
