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

module c_interfaces

  use fldebug

  implicit none

  interface
    subroutine get_environment_variable_c(name, name_len, val, val_len, stat)
      implicit none
      integer, intent(in) :: name_len
      integer, intent(inout) :: val_len
      character(len = name_len), intent(in) :: name
      character(len = val_len), intent(out) :: val
      integer, intent(out) :: stat
    end subroutine get_environment_variable_c
  end interface

  interface
    subroutine memcpy(dest, src, bytes)
      real, dimension(*), intent(out) :: dest
      integer, dimension(*), intent(in) :: src
      integer, intent(in) :: bytes
    end subroutine memcpy
  end interface

  interface
    function compare_pointers(ptr1, ptr2) result(cmp) bind(c, name='compare_pointers')
      use iso_c_binding
      type(c_ptr), intent(in), value :: ptr1
      type(c_ptr), intent(in), value :: ptr2
      logical(kind=c_bool) :: cmp
    end function compare_pointers
  end interface

  private

  public :: get_environment_variable, memcpy, compare_pointers

contains

  subroutine get_environment_variable(name, val, stat, default)
    character(len = *), intent(in) :: name
    character(len = *), intent(out) :: val
    integer, optional, intent(out) :: stat
    character(len = *), optional, intent(in) :: default

    integer :: lstat, val_len

    if(present(stat)) stat = 0

    val_len = len(val)
    call get_environment_variable_c(name, len_trim(name), val, val_len, lstat)
    if(val_len < len(val)) val(val_len + 1:) = ""

    if(lstat /= 0) then
      if(present(default)) then
        val = default
      else if(present(stat)) then
        stat = lstat
      else
        FLAbort("Failed to retrieve environment variable named " // trim(name))
      end if
    end if

  end subroutine get_environment_variable

end module c_interfaces
