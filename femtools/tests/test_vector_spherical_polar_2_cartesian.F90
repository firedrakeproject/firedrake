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
subroutine test_vector_spherical_polar_2_cartesian
  !Subroutine/unit-test of correct vector basis change from a
  ! spherical-polar system to a Cartesian system.

  use fields
  use vtk_interfaces
  use state_module
  use Coordinates
  use unittest_tools
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: CartesianCoordinate, PolarCoordinate
  type(vector_field), pointer :: UnitRadialVector_inCartesian
  type(vector_field), pointer :: UnitPolarVector_inCartesian
  type(vector_field), pointer :: UnitAzimuthalVector_inCartesian
  type(vector_field), pointer :: UnitRadialVector_inPolar
  type(vector_field), pointer :: UnitPolarVector_inPolar
  type(vector_field), pointer :: UnitAzimuthalVector_inPolar
  type(vector_field) :: radialVectorDifference, &
                        polarVectorDifference, &
                        azimuthalVectorDifference
  real, dimension(3) :: sphericalPolarVectorComponents, &
                        cartesianVectorComponents
  real, dimension(3) :: XYZ, RTP !Arrays containing a single node's position vector
                                 ! components in Cartesian & spherical-polar bases.
  integer :: node
  logical :: fail

  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  PolarCoordinate => extract_vector_field(state, "PolarCoordinate")
  UnitRadialVector_inCartesian => extract_vector_field(state, &
                                                 "UnitRadialVector_inCartesian")
  UnitPolarVector_inCartesian => extract_vector_field(state, &
                                                 "UnitPolarVector_inCartesian")
  UnitAzimuthalVector_inCartesian => extract_vector_field(state, &
                                                 "UnitAzimuthalVector_inCartesian")
  UnitRadialVector_inPolar => extract_vector_field(state, "UnitRadialVector_inPolar")
  UnitPolarVector_inPolar => extract_vector_field(state, "UnitPolarVector_inPolar")
  UnitAzimuthalVector_inPolar => extract_vector_field(state, &
                                                 "UnitAzimuthalVector_inPolar")

  !Test the change of basis from spherical-polar to Cartesian.

  call allocate(radialVectorDifference, 3 , mesh, 'radialVectorDifference')
  call zero(radialVectorDifference)
  call allocate(polarVectorDifference, 3 , mesh, 'polarVectorDifference')
  call zero(polarVectorDifference)
  call allocate(azimuthalVectorDifference, 3 , mesh, 'azimuthalVectorDifference')
  call zero(azimuthalVectorDifference)

  !Convert unit-radial vector components into to Cartesian basis. Then compare
  ! with vector already in Cartesian basis, obtained from vtu.
  do node=1,node_count(PolarCoordinate)
    RTP = node_val(PolarCoordinate, node)
    sphericalPolarVectorComponents = node_val(UnitRadialVector_inPolar, node)
    call vector_spherical_polar_2_cartesian(sphericalPolarVectorComponents(1), &
                                            sphericalPolarVectorComponents(2), &
                                            sphericalPolarVectorComponents(3), &
                                            RTP(1), &
                                            RTP(2), &
                                            RTP(3), &
                                            cartesianVectorComponents(1), &
                                            cartesianVectorComponents(2), &
                                            cartesianVectorComponents(3), &
                                            XYZ(1), &
                                            XYZ(2), &
                                            XYZ(3))
    call set(radialVectorDifference, node, cartesianVectorComponents)
  enddo
  call addto(radialVectorDifference, UnitRadialVector_inCartesian, -1.0)
  fail = any(radialVectorDifference%val > 1e-12)
  call report_test( &
      "[Vector basis change: Spherical-polar to Cartesian of unit-radial vector.]", &
      fail, .false., "Radial unit vector components not transformed correctly.")

  !Convert unit-polar vector components into Cartesian basis. Then compare
  ! with vector already in Cartesian basis, obtained from vtu.
  do node=1,node_count(PolarCoordinate)
    RTP = node_val(PolarCoordinate, node)
    sphericalPolarVectorComponents = node_val(UnitPolarVector_inPolar, node)
    call vector_spherical_polar_2_cartesian(sphericalPolarVectorComponents(1), &
                                            sphericalPolarVectorComponents(2), &
                                            sphericalPolarVectorComponents(3), &
                                            RTP(1), &
                                            RTP(2), &
                                            RTP(3), &
                                            cartesianVectorComponents(1), &
                                            cartesianVectorComponents(2), &
                                            cartesianVectorComponents(3), &
                                            XYZ(1), &
                                            XYZ(2), &
                                            XYZ(3))
    call set(polarVectorDifference, node, cartesianVectorComponents)
  enddo
  call addto(polarVectorDifference, UnitPolarVector_inCartesian, -1.0)
  fail = any(polarVectorDifference%val > 1e-12)
  call report_test( &
       "[Vector basis change: Spherical-polar to Cartesian of unit-polar vector.]", &
       fail, .false., "Polar unit vector components not transformed correctly.")

  !Convert unit-azimuthal vector components into Cartesian basis. Then compare
  ! with vector already in Cartesian basis, obtained from vtu.
  do node=1,node_count(PolarCoordinate)
    RTP = node_val(PolarCoordinate, node)
    sphericalPolarVectorComponents = node_val(UnitAzimuthalVector_inPolar, node)
    call vector_spherical_polar_2_cartesian(sphericalPolarVectorComponents(1), &
                                            sphericalPolarVectorComponents(2), &
                                            sphericalPolarVectorComponents(3), &
                                            RTP(1), &
                                            RTP(2), &
                                            RTP(3), &
                                            cartesianVectorComponents(1), &
                                            cartesianVectorComponents(2), &
                                            cartesianVectorComponents(3), &
                                            XYZ(1), &
                                            XYZ(2), &
                                            XYZ(3))
    call set(azimuthalVectorDifference, node, cartesianVectorComponents)
  enddo
  call addto(azimuthalVectorDifference, UnitAzimuthalVector_inCartesian, -1.0)
  fail = any(azimuthalVectorDifference%val > 1e-12)
  call report_test( &
       "[Vector basis change: Spherical-polar to Cartesian of unit-azimuthal vector.]", &
       fail, .false., "Azimuthal unit vector components not transformed correctly.")

  call deallocate(radialVectorDifference)
  call deallocate(polarVectorDifference)
  call deallocate(azimuthalVectorDifference)

end subroutine
