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
subroutine test_cartesian_2_spherical_polar
  !Subroutine for testing conversion of the position vector components from a Cartesian
  ! basis to a spherical-polar basis.
  use fields
  use vtk_interfaces
  use state_module
  use Coordinates
  use unittest_tools
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: CartesianCoordinate
  type(vector_field), pointer :: PolarCoordinate
  type(vector_field) :: difference
  integer :: node
  real, dimension(3) :: X, RTP !Arrays containing a single node's position vector
                               ! components in Cartesian & spherical-polar bases.
  logical :: fail

  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  PolarCoordinate => extract_vector_field(state, "PolarCoordinate")

  !Extract the components of points in vtu file in Cartesian coordinates,
  ! apply transformation and compare with position-vector is spherical-polar
  ! coordinates.
  call allocate(difference, 3 , mesh, 'difference')
  do node=1,node_count(CartesianCoordinate)
    X = node_val(CartesianCoordinate, node)
    call cartesian_2_spherical_polar(X(1), X(2), X(3), RTP(1), RTP(2), RTP(3))
    call set(difference, node, RTP)
  enddo
  call addto(difference, PolarCoordinate, -1.0)

  fail = any(difference%val > 1e-8)
  call report_test("[Coordinate change: Cartesian to Spherical-polar.]", &
                   fail, .false., "Position vector components not transformed correctly.")

  call deallocate(difference)

end subroutine
