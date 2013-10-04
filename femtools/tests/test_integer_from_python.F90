!    Copyright (C) 2006-2007 Imperial College London and others.
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

subroutine test_integer_from_python

  use embed_python
  use fldebug
  use unittest_tools

  implicit none

  character(len = *), parameter :: func = &
    & "def val(t):" // new_line("") // &
    & "  return int(t)"
  integer :: result, stat

  call integer_from_python(func, 0.1, result, stat = stat)
  call report_test("[integer_from_python]", stat /= 0, .false., "integer_from_python returned an error")
  call report_test("[Expected result]", result /= 0, .false., "integer_from_python returned incorrect integer")

  call integer_from_python(func, 1.1, result, stat = stat)
  call report_test("[integer_from_python]", stat /= 0, .false., "integer_from_python returned an error")
  call report_test("[Expected result]", result /= 1, .false., "integer_from_python returned incorrect integer")

end subroutine test_integer_from_python
