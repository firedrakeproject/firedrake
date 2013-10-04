!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineeringp
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

subroutine test_adaptive_timestepping
  !!< Test adaptive timestepping

  use adaptive_timestepping
  use fields
  use fields_data_types
  use read_triangle
  use unittest_tools

  implicit none

  real :: dt
  type(scalar_field) :: cflnumber_field
  type(vector_field) :: mesh_field

  mesh_field = read_triangle_files("data/tet", quad_degree = 1)

  call allocate(cflnumber_field, mesh_field%mesh, "CFLNumber")

  call set(cflnumber_field, 0.1)

  dt = cflnumber_field_based_dt(cflnumber_field, current_dt = 1.0, max_cfl_requested = 1.0, min_dt = tiny(0.0), max_dt = huge(0.0), increase_tolerance = huge(0.0) * epsilon(0.0))
  call report_test("[Correct timestep size (increasing dt)]", dt .fne. 10.0, .false., "Incorrect timestep size")

  dt = cflnumber_field_based_dt(cflnumber_field, current_dt = 1.0, max_cfl_requested = 1.0, min_dt = tiny(0.0), max_dt = 5.0, increase_tolerance = huge(0.0) * epsilon(0.0))
  call report_test("[Max timestep size (set via max_dt)]", dt .fne. 5.0, .false., "Incorrect timestep size")

  dt = cflnumber_field_based_dt(cflnumber_field, current_dt = 1.0, max_cfl_requested = 1.0, min_dt = tiny(0.0), max_dt = huge(0.0), increase_tolerance = 5.0)
  call report_test("[Max timestep size (set via increase_tolerance)]", dt .fne. 5.0, .false., "Incorrect timestep size")

  call set(cflnumber_field, 10.0)

  dt = cflnumber_field_based_dt(cflnumber_field, current_dt = 1.0, max_cfl_requested = 1.0, min_dt = tiny(0.0), max_dt = huge(0.0), increase_tolerance = huge(0.0) * epsilon(0.0))
  call report_test("[Correct timestep size (decreasing dt)]", dt .fne. 0.1, .false., "Incorrect timestep size")

  dt = cflnumber_field_based_dt(cflnumber_field, current_dt = 1.0, max_cfl_requested = 1.0, min_dt = 0.2, max_dt = huge(0.0), increase_tolerance = huge(0.0) * epsilon(0.0))
  call report_test("[Min timestep size]", dt .fne. 0.2, .false., "Incorrect timestep size")

  call deallocate(cflnumber_field)
  call deallocate(mesh_field)

end subroutine test_adaptive_timestepping
