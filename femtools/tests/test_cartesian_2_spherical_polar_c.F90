!    Copyright (C) 2012 Imperial College London and others.
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
subroutine test_cartesian_2_spherical_polar_c
  !Subroutine/unit-test of correct transformation of point coordinates from a
  ! Cartesian system to a spherical-polar system. This subroutine will test
  ! the C-inoperable version of the conversion.
  use fields
  use vtk_interfaces
  use state_module
  use Coordinates
  use unittest_tools
  use iso_c_binding
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: CartesianCoordinate
  type(vector_field), pointer :: PolarCoordinate
  type(vector_field) :: difference
  integer :: node
  real(kind=c_double), dimension(3) :: XYZ, RTP !Arrays containing a single node's
                                 ! position vector components in Cartesian &
                                 ! spherical-polar bases.
  logical :: fail

  !Extract the vector fields of position in vtu file in polar coordinates and
  ! cartesian coordiantes
  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  PolarCoordinate => extract_vector_field(state, "PolarCoordinate")

  !Apply transformation to Cartesian components and compare with components of
  ! spherical-polar position-vector.
  call allocate(difference, 3 , mesh, 'difference')
  do node=1,node_count(PolarCoordinate)
    XYZ = node_val(CartesianCoordinate, node)
    call cartesian_2_spherical_polar_c(XYZ(1), XYZ(2), XYZ(3), RTP(1), RTP(2), RTP(3))
    call set(difference, node, RTP)
  enddo
  call addto(difference, PolarCoordinate, -1.0)

  fail = any(difference%val > 1e-8)
  call report_test("[Coordinate change: Cartesian to Spherical-polar(C-types).]", &
                   fail, .false., "Position vector components not transformed correctly.")

  call deallocate(difference)

end subroutine
