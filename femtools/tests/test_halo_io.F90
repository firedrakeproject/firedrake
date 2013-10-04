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
!    License as published by the Free Software Foundation; either
!    version 2.1 of the License, or (at your option) any later version.
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

subroutine test_halo_io

  use fields
  use halos
  use read_triangle
  use unittest_tools

  implicit none

  integer :: i
  type(mesh_type) :: mesh_2
  type(halo_type), pointer :: halo, halo_2
  type(vector_field) :: positions, positions_2

  positions = read_triangle_files("data/cube-parallel_0", quad_degree = 1)
  ! Use make_mesh to create a copy of positions%mesh with no halos, for use
  ! with positions_2 later
  mesh_2 = make_mesh(positions%mesh)

  call report_test("[No halos]", halo_count(positions) /= 0, .false., "Coordinate field has halos")
  call read_halos("data/cube-parallel", positions)
  call report_test("[Halos]", halo_count(positions) == 0, .false., "Coordinate field has no halos")
  call report_test("[2 halos]", halo_count(positions) /= 2, .false., "Coordinate field has incorrect number of halos")

  do i = 1, 2
    halo => positions%mesh%halos(i)
    call report_test("[nowned_nodes]", halo_nowned_nodes(halo) /= 665, .false., "Incorrect number of owned nodes")
    call report_test("[nprocs]", halo_proc_count(halo) /= 1, .false., "Incorrect number of processes")
    select case(i)
      case(1)
        call report_test("[nsends]", halo_send_count(halo, 1) /= 121, .false., "Incorrect number of sends")
        call report_test("[nreceives]", halo_receive_count(halo, 1) /= 121, .false., "Incorrect number of receives")
      case(2)
        call report_test("[nsends]", halo_send_count(halo, 1) /= 243, .false., "Incorrect number of sends")
        call report_test("[nreceives]", halo_receive_count(halo, 1) /= 242, .false., "Incorrect number of receives")
      case default
        FLAbort("Invalid loop index")
    end select
    call report_test("[trailing_receives_consistent]", .not. trailing_receives_consistent(halo), .false., "Not trailing receives consistent")
  end do

  ! Overwrite test output
  call allocate(positions_2, positions%dim, mesh_2, name = positions%name)
  call deallocate(mesh_2)
  call set(positions_2, positions)
  call report_test("[No halos]", halo_count(positions_2) /= 0, .false., "Coordinate field has halos")
  ! We need at least one halo for write_halos to do anything
  allocate(positions_2%mesh%halos(1))
  halo_2 => positions_2%mesh%halos(1)
  call allocate(halo_2, nsends = (/0/), nreceives = (/1/), nprocs = 1)
  call set_halo_nowned_nodes(halo_2, 0)
  call set_halo_receives(halo_2, 1, (/1/))
  call write_halos("data/test_halo_io_out", positions_2%mesh)
  call deallocate(halo_2)
  deallocate(positions_2%mesh%halos)
  nullify(positions_2%mesh%halos)

  ! Check that test output was overwritten
  call report_test("[No halos]", halo_count(positions_2) /= 0, .false., "Coordinate field has halos")
  call read_halos("data/test_halo_io_out", positions_2)
  call report_test("[Halos]", halo_count(positions_2) == 0, .false., "Coordinate field has no halos")
  call report_test("[2 halos]", halo_count(positions_2) /= 2, .false., "Coordinate field has incorrect number of halos")
  do i = 1, 2
    halo_2 => positions_2%mesh%halos(i)
    call report_test("[nowned_nodes]", halo_nowned_nodes(halo_2) /= 0, .false., "Incorrect number of owned nodes")
    call report_test("[nprocs]", halo_proc_count(halo_2) /= 1, .false., "Incorrect number of processes")
    call report_test("[nsends]", halo_send_count(halo_2, 1) /= 0, .false., "Incorrect number of sends")
    select case(i)
      case(1)
        call report_test("[nreceives]", halo_receive_count(halo_2, 1) /= 1, .false., "Incorrect number of receives")
        call report_test("[receives]", any(halo_receives(halo_2, 1) /= (/1/)), .false., "Incorrect sends")
      case(2)
        call report_test("[nreceives]", halo_receive_count(halo_2, 1) /= 0, .false., "Incorrect number of receives")
      case default
        FLAbort("Invalid loop index")
    end select
    call deallocate(halo_2)
  end do
  deallocate(positions_2%mesh%halos)
  nullify(positions_2%mesh%halos)
  deallocate(positions_2%mesh%element_halos)
  nullify(positions_2%mesh%element_halos)

  ! Now write test output
  call write_halos("data/test_halo_io_out", positions%mesh)

  ! Check the test output
  call report_test("[No halos]", halo_count(positions_2) /= 0, .false., "Coordinate field has halos")
  call read_halos("data/test_halo_io_out", positions_2)
  call report_test("[Halos]", halo_count(positions_2) == 0, .false., "Coordinate field has no halos")
  call report_test("[2 halos]", halo_count(positions_2) /= 2, .false., "Coordinate field has incorrect number of halos")
  do i = 1, 2
    halo => positions%mesh%halos(i)
    halo_2 => positions_2%mesh%halos(i)
    call report_test("[nowned_nodes]", halo_nowned_nodes(halo) /= halo_nowned_nodes(halo_2), .false., "Incorrect number of owned nodes")
    call report_test("[nprocs]", halo_proc_count(halo) /= halo_proc_count(halo_2), .false., "Incorrect number of processes")
    call report_test("[nsends]", halo_send_count(halo, 1) /= halo_send_count(halo_2, 1), .false., "Incorrect number of sends")
    call report_test("[nreceives]", halo_receive_count(halo, 1) /= halo_receive_count(halo_2, 1), .false., "Incorrect number of receives")
    call report_test("[sends]", any(halo_sends(halo, 1) /= halo_sends(halo_2, 1)), .false., "Incorrect sends")
    call report_test("[receives]", any(halo_receives(halo, 1) /= halo_receives(halo_2, 1)), .false., "Incorrect receives")
    call report_test("[trailing_receives_consistent]", .not. trailing_receives_consistent(halo_2), .false., "Not trailing receives consistent")
  end do

  call deallocate(positions)
  call deallocate(positions_2)
  call report_test_no_references()

end subroutine test_halo_io
