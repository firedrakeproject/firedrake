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

subroutine test_stream_io

  use fldebug
  use futils
  use unittest_tools

  implicit none

#ifdef STREAM_IO

  integer :: stat, unit
  real :: test_var

  unit = free_unit()

  open(unit = unit, file = "data/test_stream_io_out", status = "replace", access = "stream", form = "unformatted", action = "write", iostat = stat)
  call report_test("[stream open]", stat /= 0, .false., "open failure")

  test_var = 42.0
  write(unit, iostat = stat) test_var
  call report_test("[stream write]", stat /= 0, .false., "write failure")

  close(unit, iostat = stat)
  call report_test("[stream close]", stat /= 0, .false., "close failure")

  open(unit = unit, file = "data/test_stream_io_out", access = "stream", form = "unformatted", action = "read", iostat = stat)
  call report_test("[stream open]", stat /= 0, .false., "open failure")

  test_var = 0.0
  read(unit, iostat = stat) test_var
  call report_test("[stream read]", stat /= 0, .false., "read failure")

  close(unit, iostat = stat)
  call report_test("[stream close]", stat /= 0, .false., "close failure")

  call report_test("[stream read value]", test_var .fne. 42.0, .false., "Read incorrect value")

  open(unit = unit, file = "data/test_stream_io_out", status = "replace", access = "stream", form = "unformatted", action = "write", iostat = stat)
  call report_test("[stream open]", stat /= 0, .false., "open failure")

  test_var = 43.0
  write(unit, iostat = stat) test_var
  call report_test("[stream write]", stat /= 0, .false., "write failure")

  close(unit, iostat = stat)
  call report_test("[stream close]", stat /= 0, .false., "close failure")

  open(unit = unit, file = "data/test_stream_io_out", access = "stream", form = "unformatted", action = "read", iostat = stat)
  call report_test("[stream open]", stat /= 0, .false., "open failure")

  test_var = 0.0
  read(unit, iostat = stat) test_var
  call report_test("[stream read]", stat /= 0, .false., "read failure")

  close(unit, iostat = stat)
  call report_test("[stream close]", stat /= 0, .false., "close failure")

  call report_test("[stream read value]", test_var .fne. 43.0, .false., "Read incorrect value")

#else
  call report_test("[dummy]", .false., .false., "Dummy")
#endif

end subroutine test_stream_io
