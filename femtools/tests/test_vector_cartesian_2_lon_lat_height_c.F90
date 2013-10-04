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
subroutine test_vector_cartesian_2_lon_lat_height_c
  !Subroutine/unit-test of correct vector basis change from a
  ! Cartesian system to a meridional-zonal-vertical system.
  ! This subroutine will test the C-inter-operable version of the conversion.

  use fields
  use vtk_interfaces
  use state_module
  use Coordinates
  use unittest_tools
  use iso_c_binding
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: CartesianCoordinate, LonLatHeightCoordinate
  type(vector_field), pointer :: UnitRadialVector_inCartesian
  type(vector_field), pointer :: UnitPolarVector_inCartesian
  type(vector_field), pointer :: UnitAzimuthalVector_inCartesian
  type(vector_field), pointer :: UnitRadialVector_inZonalMeridionalRadial
  type(vector_field), pointer :: UnitPolarVector_inZonalMeridionalRadial
  type(vector_field), pointer :: UnitAzimuthalVector_inZonalMeridionalRadial
  type(vector_field) :: radialVectorDifference, &
                        polarVectorDifference, &
                        azimuthalVectorDifference
  real(kind=c_double), dimension(3) ::  &
                        meridionalZonalVerticalVectorComponents_c, &
                        cartesianVectorComponents_c !C-inter-operable arrays containing
                                                    ! the vector components in Cartesian
                                                    ! & lon-lat-height bases.
  real(kind=c_double), dimension(3) :: &
                        XYZ_c, LLH_c !C-inter-operable arrays containing a signle
                                     ! node's position vector components in Cartesian
                                     ! & lon-lat-height bases.
  real, dimension(3) :: meridionalZonalVerticalVectorComponents, &
                        cartesianVectorComponents
  real, dimension(3) :: XYZ, LLH !Arrays containing a signle node's position vector
                                 ! components in Cartesian & lon-lat-height bases.
  integer :: node
  logical :: fail

  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  LonLatHeightCoordinate => extract_vector_field(state, "lonlatradius")
  UnitRadialVector_inCartesian => extract_vector_field(state, &
                                      "UnitRadialVector_inCartesian")
  UnitPolarVector_inCartesian => extract_vector_field(state, &
                                      "UnitPolarVector_inCartesian")
  UnitAzimuthalVector_inCartesian => extract_vector_field(state, &
                                      "UnitAzimuthalVector_inCartesian")
  UnitRadialVector_inZonalMeridionalRadial => extract_vector_field(state, &
                                      "UnitRadialVector_inZonalMeridionalRadial")
  UnitPolarVector_inZonalMeridionalRadial => extract_vector_field(state, &
                                      "UnitPolarVector_inZonalMeridionalRadial")
  UnitAzimuthalVector_inZonalMeridionalRadial => extract_vector_field(state, &
                                      "UnitAzimuthalVector_inZonalMeridionalRadial")

  call allocate(radialVectorDifference, 3 , mesh, 'radialVectorDifference')
  call allocate(polarVectorDifference, 3 , mesh, 'polarVectorDifference')
  call allocate(azimuthalVectorDifference, 3 , mesh, 'azimuthalVectorDifference')

  !Convert unit-radial vector components into to zonal-meridional-vertical basis.
  ! Then compare with vector already in that basis, obtained from vtu.
  do node=1,node_count(CartesianCoordinate)
    XYZ = node_val(CartesianCoordinate, node)
    cartesianVectorComponents = &
                   node_val(UnitRadialVector_inCartesian, node)
    XYZ_c = real(XYZ, kind=c_double)
    cartesianVectorComponents_c = real(cartesianVectorComponents, kind=c_double)
    call vector_cartesian_2_lon_lat_height_c(cartesianVectorComponents_c(1), &
                                             cartesianVectorComponents_c(2), &
                                             cartesianVectorComponents_c(3), &
                                             XYZ_c(1), &
                                             XYZ_c(2), &
                                             XYZ_c(3), &
                                             meridionalZonalVerticalVectorComponents_c(1), &
                                             meridionalZonalVerticalVectorComponents_c(2), &
                                             meridionalZonalVerticalVectorComponents_c(3), &
                                             LLH_c(1), &
                                             LLH_c(2), &
                                             LLH_c(3), 0.0)
    meridionalZonalVerticalVectorComponents = real(meridionalZonalVerticalVectorComponents_c)
    LLH = real(LLH_c)
    call set(radialVectorDifference, node, meridionalZonalVerticalVectorComponents)
  enddo
  call addto(radialVectorDifference, UnitRadialVector_inZonalMeridionalRadial, -1.0)
  fail = any(radialVectorDifference%val > 1e-12)
  call report_test( &
   "[Vector basis change: Cartesian to Zonal-Meridional-Vertical of unit-radial vector (C-types).]", &
   fail, .false., "Radial unit vector components not transformed correctly.")

  !Convert unit-polar vector components into zonal-meridional-vertical basis.
  ! Then compare with vector already in that basis, obtained from vtu.
  do node=1,node_count(CartesianCoordinate)
    XYZ = node_val(CartesianCoordinate, node)
    cartesianVectorComponents = &
                   node_val(UnitPolarVector_inCartesian, node)
    XYZ_c = real(XYZ, kind=c_double)
    cartesianVectorComponents_c = real(cartesianVectorComponents, kind=c_double)
    call vector_cartesian_2_lon_lat_height_c(cartesianVectorComponents_c(1), &
                                             cartesianVectorComponents_c(2), &
                                             cartesianVectorComponents_c(3), &
                                             XYZ_c(1), &
                                             XYZ_c(2), &
                                             XYZ_c(3), &
                                             meridionalZonalVerticalVectorComponents_c(1), &
                                             meridionalZonalVerticalVectorComponents_c(2), &
                                             meridionalZonalVerticalVectorComponents_c(3), &
                                             LLH_c(1), &
                                             LLH_c(2), &
                                             LLH_c(3), 0.0)
    meridionalZonalVerticalVectorComponents = real(meridionalZonalVerticalVectorComponents_c)
    LLH = real(LLH_c)
    call set(polarVectorDifference, node, meridionalZonalVerticalVectorComponents)
  enddo
  call addto(polarVectorDifference, UnitPolarVector_inZonalMeridionalRadial, -1.0)
  fail = any(polarVectorDifference%val > 1e-12)
  call report_test( &
   "[Vector basis change: Cartesian to Zonal-Meridional-Vertical of unit-polar vector (C-types).]", &
   fail, .false., "Polar unit vector components not transformed correctly.")

  !Convert unit-azimuthal vector components into Cartesian basis. Then compare
  ! with vector already in Cartesian basis, obtained from vtu.
  do node=1,node_count(CartesianCoordinate)
    XYZ = node_val(CartesianCoordinate, node)
    cartesianVectorComponents = &
                   node_val(UnitAzimuthalVector_inCartesian, node)
    XYZ_c = real(XYZ, kind=c_double)
    cartesianVectorComponents_c = real(cartesianVectorComponents, kind=c_double)
    call vector_cartesian_2_lon_lat_height_c(cartesianVectorComponents_c(1), &
                                             cartesianVectorComponents_c(2), &
                                             cartesianVectorComponents_c(3), &
                                             XYZ_c(1), &
                                             XYZ_c(2), &
                                             XYZ_c(3), &
                                             meridionalZonalVerticalVectorComponents_c(1), &
                                             meridionalZonalVerticalVectorComponents_c(2), &
                                             meridionalZonalVerticalVectorComponents_c(3), &
                                             LLH_c(1), &
                                             LLH_c(2), &
                                             LLH_c(3), 0.0)
    meridionalZonalVerticalVectorComponents = real(meridionalZonalVerticalVectorComponents_c)
    LLH = real(LLH_c)
    call set(azimuthalVectorDifference, node, meridionalZonalVerticalVectorComponents)
  enddo
  call addto(azimuthalVectorDifference, UnitAzimuthalVector_inZonalMeridionalRadial, -1.0)
  fail = any(azimuthalVectorDifference%val > 1e-12)
  call report_test( &
   "[Vector basis change: Cartesian to Zonal-Meridional-Vertical of unit-azimuthal vector (C-types).]", &
   fail, .false., "Azimuthal unit vector components not transformed correctly.")

  call deallocate(radialVectorDifference)
  call deallocate(polarVectorDifference)
  call deallocate(azimuthalVectorDifference)

end subroutine
