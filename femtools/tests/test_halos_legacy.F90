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

subroutine test_halos_legacy
  !!< Test halo_type derived type legacy interoperability

  use halos
  use unittest_tools

  implicit none

  integer :: i, index, j, npnodes
  integer, dimension(:), allocatable :: atorec, atosen, colgat, nreceives, nsends, scater
  integer, parameter :: nowned_nodes = 42, nprocs = 1
  logical :: fail
  type(halo_type) :: input_halo, output_halo

  ! Set up halo node counts
  allocate(nsends(nprocs))
  allocate(nreceives(nprocs))
  do i = 1, nprocs
    nsends(i) = i * 10
    nreceives(i) = i * 10
  end do

  ! Allocate a halo
  call allocate(input_halo, nsends, nreceives, nprocs = nprocs, name = "TestHalo", nowned_nodes = nowned_nodes)

  ! Set the halo nodes
  call zero(input_halo)
  index = 1
  do i = 1, halo_proc_count(input_halo)
    do j = 1, halo_send_count(input_halo, i)
      call set_halo_send(input_halo, i, j, index)
      index = index + 1
    end do
  end do
  index=nowned_nodes+1
  do i = 1, halo_proc_count(input_halo)
    do j = 1, halo_receive_count(input_halo, i)
      call set_halo_receive(input_halo, i, j, index)
      index = index + 1
    end do
  end do

  ! Allocate the legacy datatypes
  allocate(colgat(halo_all_sends_count(input_halo)))
  allocate(atosen(halo_proc_count(input_halo) + 1))
  allocate(scater(halo_all_receives_count(input_halo)))
  allocate(atorec(halo_proc_count(input_halo) + 1))

  ! Extract the legacy data
  call extract_raw_halo_data(input_halo, colgat, atosen, scater, atorec, nowned_nodes = npnodes)

  ! Form a new halo from the legacy data
  call form_halo_from_raw_data(output_halo, nprocs, colgat, atosen, scater, atorec, nowned_nodes = npnodes)

  ! Note: Test output halo against input halo and against raw data

  call report_test("[Correct nowned_nodes]", halo_nowned_nodes(output_halo) /= nowned_nodes, .false., "Incorrect nowned_nodes")
  call report_test("[Correct nowned_nodes]", halo_nowned_nodes(output_halo) /= halo_nowned_nodes(input_halo), .false., "Incorrect nowned_nodes")

  fail = .false.
  index = 1
  do i = 1, nprocs
    do j = 1, nsends(i)
      if(halo_send(output_halo, i, j) /= index) then
        fail = .true.
        exit
      end if
      index = index + 1
    end do
    if(fail) then
      exit
    end if
  end do
  call report_test("[Correct send nodes]", fail, .false., "Incorrect send nodes")
  fail = .false.
  do i = 1, halo_proc_count(output_halo)
    do j = 1, halo_send_count(output_halo, i)
      if(halo_send(output_halo, i, j) /= halo_send(input_halo, i, j)) then
        fail = .true.
        exit
      end if
    end do
    if(fail) then
      exit
    end if
  end do
  call report_test("[Correct send nodes]", fail, .false., "Incorrect send nodes")

  fail = .false.
  index = nowned_nodes + 1
  do i = 1, nprocs
    do j = 1, nreceives(i)
      if(halo_receive(output_halo, i, j) /= index) then
        fail = .true.
        exit
      end if
      index = index + 1
    end do
    if(fail) then
      exit
    end if
  end do
  call report_test("[Correct receive nodes]", fail, .false., "Incorrect receive nodes")
  fail = .false.
  do i = 1, halo_proc_count(output_halo)
    do j = 1, halo_receive_count(output_halo, i)
      if(halo_receive(output_halo, i, j) /= halo_receive(input_halo, i, j)) then
        fail = .true.
        exit
      end if
    end do
    if(fail) then
      exit
    end if
  end do
  call report_test("[Correct receive nodes]", fail, .false., "Incorrect receive nodes")

  deallocate(colgat)
  deallocate(atosen)
  deallocate(scater)
  deallocate(atorec)

  call deallocate(input_halo)
  call deallocate(output_halo)

  deallocate(nsends)
  deallocate(nreceives)

  call report_test_no_references()

end subroutine test_halos_legacy
