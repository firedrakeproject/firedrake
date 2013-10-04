!! This module provides the functionality to call Python code
!! from Fortran:
!!  python_run_string(string::s)
!!  python_run_file(string::filename)

!! These should be called once (either from C or Fortran) before and after anything else of this module is used:
!! python_init() initializes the Python interpreter;
!! python_end() finalizes

!! Files belonging to this module:
!! python_utils.F90
!! python_statec.c
!! python_state_types.py

#include "fdebug.h"

module python_utils
  use fldebug
  use iso_c_binding

  implicit none

  private

  public :: python_init, python_reset
  public :: python_run_string, python_run_file
  public :: python_fetch_real, python_fetch_integer
  public :: python_remove_from_cache

  logical(c_bool), public, bind(c) :: python_cache = .True.

  interface
    !! Python init and end
    subroutine python_init()
    end subroutine python_init
    subroutine python_reset()
    end subroutine python_reset
    subroutine python_end()
    end subroutine python_end

    !! Run a python string and file
    subroutine python_run_stringc(s, slen, stat)
      implicit none
      integer, intent(in) :: slen
      character(len = slen), intent(in) :: s
      integer, intent(out) :: stat
    end subroutine python_run_stringc

    subroutine python_run_filec(s, slen, stat)
      implicit none
      integer, intent(in) :: slen
      character(len = slen), intent(in) :: s
      integer, intent(out) :: stat
    end subroutine python_run_filec

  end interface

contains

  subroutine python_run_string(s, stat)
    !!< Wrapper for function for python_run_stringc

    character(len = *), intent(in) :: s
    integer, optional, intent(out) :: stat

    integer :: lstat

    if(present(stat)) stat = 0

    call python_run_stringc(s, len_trim(s), lstat)
    if(lstat /= 0) then
      if(present(stat)) then
        stat = lstat
      else
        ewrite(-1, *) "Python error, Python string was:"
        ewrite(-1, *) trim(s)
        FLExit("Dying")
      end if
    end if

  end subroutine python_run_string

  subroutine python_run_file(s, stat)
    !!< Wrapper for function for python_run_filec

    character(len = *), intent(in) :: s
    integer, optional, intent(out) :: stat

    integer :: lstat

    if(present(stat)) stat = 0

    call python_run_filec(s, len_trim(s), lstat)
    if(lstat /= 0) then
      if(present(stat)) then
        stat = lstat
      else
        ewrite(-1, *) "Python error, Python file was:"
        ewrite(-1, *) trim(s)
        FLExit("Dying")
      end if
    end if

  end subroutine python_run_file

  function python_fetch_real(name) result(output)
    character(len=*), intent(in) :: name
    real :: output

    call python_fetch_real_c(name, len(name), output)
  end function python_fetch_real

  function python_fetch_integer(name) result(output)
    character(len=*), intent(in) :: name
    integer :: output

    call python_fetch_integer_c(name, len(name), output)
  end function python_fetch_integer

  subroutine python_remove_from_cache(type, uid)
    !!< Remove a field of type type with id uid from the python cache

    character(len=*), intent(in) :: type
    integer, intent(in) :: uid

    character(len=16) :: buf

    if (.not. python_cache) return

    write(unit=buf,fmt="(i0)")uid

    call python_run_string("if "//trim(adjustl(buf))//" in "//type//"_cache: "//type//"_cache.pop("//trim(adjustl(buf))//")")

  end subroutine python_remove_from_cache

end module python_utils
