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

subroutine test_tokenize

  use fldebug
  use futils
  use unittest_tools

  implicit none

  character(len = 255), dimension(:), allocatable :: tokens

  call tokenize("One:Two:Three", tokens, ":")
  call report_test("[allocated(tokens)]", .not. allocated(tokens), .false., "tokens not allocated")
  call report_test("[size(tokens)]", size(tokens) /= 3, .false., "Incorrect tokens size")
  call report_test("[tokens(1)]", trim(tokens(1)) /= "One", .false., "Incorrect split")
  call report_test("[tokens(2)]", trim(tokens(2)) /= "Two", .false., "Incorrect split")
  call report_test("[tokens(3)]", trim(tokens(3)) /= "Three", .false., "Incorrect split")
  deallocate(tokens)

  call tokenize("One::Two::Three", tokens, "::")
  call report_test("[allocated(tokens)]", .not. allocated(tokens), .false., "tokens not allocated")
  call report_test("[size(tokens)]", size(tokens) /= 3, .false., "Incorrect tokens size")
  call report_test("[tokens(1)]", trim(tokens(1)) /= "One", .false., "Incorrect split")
  call report_test("[tokens(2)]", trim(tokens(2)) /= "Two", .false., "Incorrect split")
  call report_test("[tokens(3)]", trim(tokens(3)) /= "Three", .false., "Incorrect split")
  deallocate(tokens)

  call tokenize("One:::Two:::Three", tokens, ":::")
  call report_test("[allocated(tokens)]", .not. allocated(tokens), .false., "tokens not allocated")
  call report_test("[size(tokens)]", size(tokens) /= 3, .false., "Incorrect tokens size")
  call report_test("[tokens(1)]", trim(tokens(1)) /= "One", .false., "Incorrect split")
  call report_test("[tokens(2)]", trim(tokens(2)) /= "Two", .false., "Incorrect split")
  call report_test("[tokens(3)]", trim(tokens(3)) /= "Three", .false., "Incorrect split")
  deallocate(tokens)

end subroutine test_tokenize
