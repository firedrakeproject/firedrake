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

subroutine test_halo_allocation
  !!< Test allocation of the halo_type derived type

  use halos
  use reference_counting
  use unittest_tools

  implicit none

  integer :: i
  integer, dimension(:), allocatable :: nreceives, nsends
  integer, parameter :: nowned_nodes = 42, nprocs = 1
  logical :: fail
  type(halo_type) :: halo

  ! Set up halo node counts
  allocate(nsends(nprocs))
  allocate(nreceives(nprocs))
  do i = 1, nprocs
    nsends(i) = i * 5
    nreceives(i) = i * 10
  end do

  ! Allocate a halo
  call allocate(halo,  nsends, nreceives, nprocs = nprocs, name = "TestHalo", nowned_nodes = nowned_nodes)
  call report_test("[Has references]", .not. has_references(halo), .false., "Halo does not have references")
  call report_test("[References]", .not. associated(refcount_list%next), .false., "Have no references")

  call report_test("[Correct nprocs]", halo%nprocs /= nprocs, .false., "Incorrect nprocs")
  call report_test("[Correct nprocs]", halo_proc_count(halo) /= nprocs, .false., "Incorrect nprocs")

  call report_test("[Correct name]", trim(halo%name) /= "TestHalo", .false., "Incorrect name")

  call report_test("[Correct nowned_nodes]", halo%nowned_nodes /= nowned_nodes, .false., "Incorrect nowned_nodes")
  call report_test("[Correct nowned_nodes]", halo_nowned_nodes(halo) /= nowned_nodes, .false., "Incorrect nowned_nodes")

  call report_test("[sends allocated]", .not. associated(halo%sends), .false., "sends array not allocated")
  call report_test("[receives allocated]", .not. associated(halo%receives), .false., "receives array not allocated")
  call report_test("[sends has correct size]", size(halo%sends) /= nprocs, .false., "sends array has incorrect size")
  call report_test("[receives has correct size]", size(halo%receives) /= nprocs, .false., "receives array has incorrect size")

  fail = .false.
  do i = 1, halo_proc_count(halo)
    if(.not. associated(halo%sends(i)%ptr)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[sends elements allocated]", fail, .false., "At least one element of sends array not allocated")
  fail = .false.
  do i = 1, halo_proc_count(halo)
    if(.not. associated(halo%receives(i)%ptr)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[receives elements allocated]", fail, .false., "At least one element of receives array not allocated")
  fail = .false.
  do i = 1, halo_proc_count(halo)
    if(halo_send_count(halo, i) /= nsends(i)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[sends elements have correct sizes]", fail, .false., "At least one element of sends array has incorrect size")
  fail = .false.
  do i = 1, halo_proc_count(halo)
    if(halo_receive_count(halo, i) /= nreceives(i)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[receives elements have correct sizes]", fail, .false., "At least one element of receives array has incorrect size")

  ! Deallocate the halo
  call deallocate(halo)

  call report_test_no_references()

  call report_test("[Name reset]", len_trim(halo%name) > 0, .false., "Halo name not reset")
  call report_test("[sends not associated]", associated(halo%sends), .false., "sends still associated")
  call report_test("[receives not associated]", associated(halo%receives), .false., "receives still associated")

  deallocate(nsends)
  deallocate(nreceives)

end subroutine test_halo_allocation
