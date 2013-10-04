!    Copyright (C) 2011 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
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

subroutine test_cell_numbering
  use cell_numbering
  use unittest_tools
  use integer_hash_table_module

  implicit none
  logical :: fail

  type(cell_type), pointer :: cell
  character(len=200) :: errmsg

  call number_cells

  cell=>cells(CELL_INTERVAL)

  fail=any(vertices_entity(cell, [2])/=[0,2])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [2]), " should be 0,2"
  call report_test("[interval vertex entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2])/=[1,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2]), " should be 1,1"
  call report_test("[interval edge entity]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [0,1])/=[1])
  write(errmsg,'(i0,a)'),entity_vertices(cell, [0,1]), " should be 1"
  call report_test("[interval vertex vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [1,1])/=[1,2])
  write(errmsg,'(i0,",",i0,a)'),entity_vertices(cell, [1,1]), " should be 1,2"
  call report_test("[interval edge vertices]", fail, .false., trim(errmsg))

  cell=>cells(CELL_TRIANGLE)

  fail=any(vertices_entity(cell, [1])/=[0,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1]), " should be 0,1"
  call report_test("[triangle vertex entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2])/=[1,3])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2]), " should be 1,3"
  call report_test("[triangle edge entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2,3])/=[2,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2,3]), " should be 2,1"
  call report_test("[triangle face_entity]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [0,1])/=[1])
  write(errmsg,'(i0,a)'),entity_vertices(cell, [0,1]), " should be 1"
  call report_test("[triangle vertex vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [1,3])/=[1,2])
  write(errmsg,'(i0,",",i0,a)'),entity_vertices(cell, [1,3]), " should be 1,2"
  call report_test("[triangle edge vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [2,1])/=[1,2,3])
  write(errmsg,'(i0,",",i0",",i0,a)'),entity_vertices(cell, [2,1]), &
       " should be 1,2,3"
  call report_test("[triangle face_vertices]", fail, .false., trim(errmsg))

  cell=>cells(CELL_QUAD)

  fail=any(vertices_entity(cell, [1])/=[0,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1]), " should be 0,1"
  call report_test("[quad vertex entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2])/=[1,4])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2]), " should be 1,3"
  call report_test("[quad edge entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2,3,4])/=[2,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2,3,4]), " should be 2,1"
  call report_test("[quad face_entity]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [0,1])/=[1])
  write(errmsg,'(i0,a)'),entity_vertices(cell, [0,1]), " should be 1"
  call report_test("[quad vertex vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [1,3])/=[1,4])
  write(errmsg,'(i0,",",i0,a)'),entity_vertices(cell, [1,3]), " should be 1,2"
  call report_test("[quad edge vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [2,1])/=[1,2,3,4])
  write(errmsg,'(i0,",",i0,",",i0",",i0,a)'),entity_vertices(cell, [2,1]), &
       " should be 1,2,3,4"
  call report_test("[quad face_vertices]", fail, .false., trim(errmsg))

  cell=>cells(CELL_TET)

  fail=any(vertices_entity(cell, [1])/=[0,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1]), " should be 0,1"
  call report_test("[tet vertex entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2])/=[1,6])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2]), " should be 1,6"
  call report_test("[tet edge entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2,3])/=[2,4])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2,3]), " should be 2,4"
  call report_test("[tet face_entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2,3,4])/=[3,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2,3,4]), " should be 3,1"
  call report_test("[tet_cell_entity]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [0,1])/=[1])
  write(errmsg,'(i0,a)'),entity_vertices(cell, [0,1]), " should be 1"
  call report_test("[tet vertex vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [1,3])/=[2,3])
  write(errmsg,'(i0,",",i0,a)'),entity_vertices(cell, [1,3]), " should be 2,3"
  call report_test("[tet edge vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [2,1])/=[2,3,4])
  write(errmsg,'(i0,",",i0",",i0,a)'),entity_vertices(cell, [2,1]), &
       " should be 2,3,4"
  call report_test("[tet face_vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [3,1])/=[1,2,3,4])
  write(errmsg,'(i0,",",i0,",",i0",",i0,a)'),entity_vertices(cell, [3,1]), &
       " should be 1,2,3,4"
  call report_test("[tet_cell_vertices]", fail, .false., trim(errmsg))

  cell=>cells(CELL_HEX)

  fail=any(vertices_entity(cell, [1])/=[0,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1]), " should be 0,1"
  call report_test("[hex vertex entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2])/=[1,12])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2]), " should be 1,12"
  call report_test("[hex edge entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2,3,4])/=[2,6])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2,3,4]), " should be 2,6"
  call report_test("[hex_face_entity]", fail, .false., trim(errmsg))

  fail=any(vertices_entity(cell, [1,2,3,4,5,6,7,8])/=[3,1])
  write(errmsg,'(i0,",",i0,a)'),vertices_entity(cell, [1,2,3,4,5,6,7,8]),&
       " should be 2,4"
  call report_test("[hex cell_entity]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [0,1])/=[1])
  write(errmsg,'(i0,a)'),entity_vertices(cell, [0,1]), " should be 1"
  call report_test("[hex vertex vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [1,3])/=[5,8])
  write(errmsg,'(i0,",",i0,a)'),entity_vertices(cell, [1,3]), " should be 5,8"
  call report_test("[hex edge vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [2,1])/=[5,6,7,8])
  write(errmsg,'(i0,",",i0,",",i0",",i0,a)'),entity_vertices(cell, [2,1]), &
       " should be 5,6,7,8"
  call report_test("[hex_face_vertices]", fail, .false., trim(errmsg))

  fail=any(entity_vertices(cell, [3,1])/=[1,2,3,4,5,6,7,8])
  write(errmsg,'(i0,7(",",i0),a)'),entity_vertices(cell, [3,1]), &
       " should be 1,2,3,4,5,6,7,8"
  call report_test("[hex_cell_vertices]", fail, .false., trim(errmsg))


end subroutine test_cell_numbering

