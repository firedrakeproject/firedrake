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
subroutine test_tensor_spherical_polar_2_cartesian
  !Subroutine/unit-test of correct transformation of tensor components from a
  ! spherical-polar system to a Cartesian system.
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
  type(tensor_field), pointer :: tensor_cartesian
  type(tensor_field), pointer :: tensor_sphericalPolar
  type(tensor_field) :: difference
  integer :: node
  real, dimension(3) :: XYZ, RTP !Arrays containing a signle node's position vector
                                 ! components in Cartesian & spherical-polar bases.
  real, dimension(3,3) :: sphericalPolarComponents !Array containing the tensor
                                 ! components in a spherical-polar basis, at
                                 ! a sinlge point.
  real, dimension(3,3) :: cartesianComponents      !Array containing the tensor
                                 ! components in a spherical-polar basis, at
                                 ! a sinlge point.
  logical :: fail

  !Extract the vector fields of position in vtu file in polar coordinates and
  ! cartesian coordiantes
  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  PolarCoordinate => extract_vector_field(state, "PolarCoordinate")
  tensor_cartesian => extract_tensor_field(state, "Tensor_inCartesian")
  tensor_sphericalPolar => extract_tensor_field(state, "Tensor_inPolar")

  !Apply transformation to spherical-polar components and compare with components
  ! in Cartesian basis.
  call allocate(difference, mesh, 'difference')
  do node=1,node_count(PolarCoordinate)
    RTP = node_val(PolarCoordinate, node)
    sphericalPolarComponents = node_val(tensor_sphericalPolar, node)
    call tensor_spherical_polar_2_cartesian(sphericalPolarComponents, &
                                            RTP(1), RTP(2), RTP(3), &
                                            cartesianComponents, &
                                            XYZ(1), XYZ(2), XYZ(3))
    call set(difference, node, cartesianComponents)
  enddo
  call addto(difference, tensor_cartesian, -1.0)

  fail = any(difference%val > 1e-8)
  call report_test("[Tensor change of basis: Spherical-polar to Cartesian.]", &
                   fail, .false., "Tensor components not transformed correctly.")

  call deallocate(difference)

end subroutine
