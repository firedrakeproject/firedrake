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
subroutine test_vector_spherical_polar_2_cartesian_field
  !Subroutine/unit-test of correct vector basis change from a
  ! spherical-polar system to a Cartesian system, for a femtools
  ! vector field.
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
  type(vector_field), pointer :: UnitRadialVector_inCartesian
  type(vector_field), pointer :: UnitPolarVector_inCartesian
  type(vector_field), pointer :: UnitAzimuthalVector_inCartesian
  type(vector_field), pointer :: UnitRadialVector_inPolar
  type(vector_field), pointer :: UnitPolarVector_inPolar
  type(vector_field), pointer :: UnitAzimuthalVector_inPolar
  type(vector_field), target :: &                 !Vector fields used for temporary
                       radialVectorDifference, &  ! storage of the vector fields in
                       polarVectorDifference, &   ! Cartesian basis, as well as the
                       azimuthalVectorDifference  ! difference between calculated
                                                  ! and expected values.
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

  call allocate(radialVectorDifference, 3, mesh, 'radialVectorDifference')
  call zero(radialVectorDifference)
  call allocate(polarVectorDifference, 3, mesh, 'polarVectorDifference')
  call zero(polarVectorDifference)
  call allocate(azimuthalVectorDifference, 3, mesh, 'azimuthalVectorDifference')
  call zero(azimuthalVectorDifference)

  !Set the components difference-vector equal to the unit radial vector, and apply
  ! transformation to Cartesian basis. Then compare with vector already in Cartesian
  ! basis, obtained from vtu.
  call vector_spherical_polar_2_cartesian(UnitRadialVector_inPolar, &
                                          PolarCoordinate, &
                                          radialVectorDifference, &
                                          CartesianCoordinate)
  call addto(radialVectorDifference, UnitRadialVector_inCartesian, -1.0)
  fail = any(radialVectorDifference%val > 1e-12)
  call report_test( &
    "[Vector basis change: Spherical-polar to Cartesian of unit-radial vector field.]", &
    fail, .false., "Radial unit vector components not transformed correctly.")

  !Set the components difference-vector equal to the unit-polar vector, and apply
  ! transformation to Cartesian basis. Then compare with vector already in Cartesian
  ! basis, obtained from vtu.
  call vector_spherical_polar_2_cartesian(UnitPolarVector_inPolar, &
                                          PolarCoordinate, &
                                          polarVectorDifference, &
                                          CartesianCoordinate)
  call addto(polarVectorDifference, UnitPolarVector_inCartesian, -1.0)
  fail = any(polarVectorDifference%val > 1e-12)
  call report_test( &
    "[Vector basis change: Spherical-polar to Cartesian of unit-polar vector field.]", &
    fail, .false., "Polar unit vector components not transformed correctly.")

  !Set the components difference-vector equal to the unit-azimuthal vector, and apply
  ! transformation to Cartesian basis. Then compare with vector already in Cartesian
  ! basis, obtained from vtu.
  call vector_spherical_polar_2_cartesian(UnitAzimuthalVector_inPolar, &
                                          PolarCoordinate, &
                                          azimuthalVectorDifference, &
                                          CartesianCoordinate)
  call addto(azimuthalVectorDifference, UnitAzimuthalVector_inCartesian, -1.0)
  fail = any(azimuthalVectorDifference%val > 1e-12)
  call report_test( &
    "[Vector basis change: Spherical-polar to Cartesian of unit-azimuthal vector field.]", &
    fail, .false., "Azimuthal unit vector components not transformed correctly.")

  call deallocate(radialVectorDifference)
  call deallocate(polarVectorDifference)
  call deallocate(azimuthalVectorDifference)

end subroutine
