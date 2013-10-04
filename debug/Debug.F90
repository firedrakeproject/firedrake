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

module fldebug
  !!< This module allows pure fortran programs to use the fdebug.h headers.

  use fldebug_parameters

  implicit none

  interface write_minmax
    module procedure write_minmax_real_array, write_minmax_integer_array
  end interface

contains

  function debug_unit(priority)
    !!< Decide where to send output based on the level of the error.

    integer :: debug_unit
    integer, intent(in) :: priority

    if (priority<1) then
       debug_unit=debug_error_unit
    else
       debug_unit=debug_log_unit
    end if

  end function debug_unit

  function debug_level()
    ! Simply return the current debug level. This makes the debug level
    ! effectively global.
    use fldebug_parameters
    implicit none
    integer :: debug_level

    debug_level=current_debug_level

  end function debug_level

  SUBROUTINE FLAbort_pinpoint(ErrorStr, FromFile, LineNumber)

    CHARACTER*(*) ErrorStr, FromFile
    INTEGER LineNumber
    LOGICAL UsingMPI
    INTEGER IERR
#ifdef HAVE_MPI
#include <mpif.h>
#endif

#ifdef HAVE_MPI
    CALL MPI_INITIALIZED(UsingMPI, IERR)
#endif
    ewrite(-1,FMT='(A)') "*** FLUIDITY ERROR ***"
    ewrite(-1,FMT='(3A,I5,A)') "Source location: (",FromFile,",",LineNumber,")"
    ewrite(-1,FMT='(2A)') "Error message: ",ErrorStr
    ewrite(-1,FMT='(A)') "Backtrace will follow if it is available:"
    call fprint_backtrace()
    ewrite(-1,FMT='(A)') "Use addr2line -e <binary> <address> to decipher."
    ewrite(-1,FMT='(A)') "Error is terminal."
#ifdef HAVE_MPI
    IF(UsingMPI) THEN
       !mpi_comm_femtools not required here.
       CALL MPI_ABORT(MPI_COMM_WORLD, MPI_ERR_OTHER, IERR)
    END IF
#endif

    STOP
  END SUBROUTINE FLAbort_pinpoint

  SUBROUTINE FLExit_pinpoint(ErrorStr, FromFile, LineNumber)

    CHARACTER*(*) ErrorStr, FromFile
    INTEGER LineNumber
    LOGICAL UsingMPI
    INTEGER IERR
#ifdef HAVE_MPI
#include <mpif.h>
#endif

#ifdef HAVE_MPI
    CALL MPI_INITIALIZED(UsingMPI, IERR)
#endif
    ewrite(-1,FMT='(A)') "*** ERROR ***"
#ifndef NDEBUG
    ewrite(-1,FMT='(3A,I5,A)') "Source location: (",FromFile,",",LineNumber,")"
#endif
    ewrite(-1,FMT='(2A)') "Error message: ",ErrorStr
#ifdef HAVE_MPI
    IF(UsingMPI) THEN
       !mpi_comm_femtools not required here.
       CALL MPI_ABORT(MPI_COMM_WORLD, MPI_ERR_OTHER, IERR)
    END IF
#endif

    STOP
  END SUBROUTINE FLExit_pinpoint

  subroutine write_minmax_real_array(array, array_expression)
    ! the array to print its min and max of
    real, dimension(:), intent(in):: array
    ! the actual array expression in the code
    character(len=*), intent(in):: array_expression

    ewrite(2,*) "Min, max of "//array_expression//" = ",minval(array), maxval(array)

  end subroutine write_minmax_real_array

  subroutine write_minmax_integer_array(array, array_expression)
    ! the array to print its min and max of
    integer, dimension(:), intent(in):: array
    ! the actual array expression in the code
    character(len=*), intent(in):: array_expression

    ewrite(2,*) "Min, max of "//array_expression//" = ",minval(array), maxval(array)

  end subroutine write_minmax_integer_array

end module fldebug
