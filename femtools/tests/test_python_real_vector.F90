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

subroutine test_python_real_vector
  !!< Test that we can set a field using python.
  use embed_python
  use unittest_tools
  use futils
  implicit none

#ifdef HAVE_PYTHON
  logical :: fail
  real, dimension(:), pointer :: result
  integer :: stat

  call real_vector_from_python(&
       "def val(t): return (1.0, 2.0, 3.0, 4.0)", 0.0,  result, stat)

  fail=any(result/=(/1.0, 2.0, 3.0, 4.0/))

  call report_test("[test_python_real_vector]", fail, .false., &
       "python and fortran should produce the same answer.")

  deallocate(result, stat=stat)

  call report_test("[test_python_real_vector deallocate]", fail, .false., &
       "failed to deallocate result vector")

#endif

end subroutine test_python_real_vector
