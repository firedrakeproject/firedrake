subroutine test_rotate_velocity_back_sphere

  use fields
  use vtk_interfaces
  use state_module
  use Coordinates
  use unittest_tools
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: CartesianCoordinate
  type(vector_field), pointer :: UnitRadialVector_inCartesian
  type(vector_field), pointer :: UnitPolarVector_inCartesian
  type(vector_field), pointer :: UnitAzimuthalVector_inCartesian
  type(vector_field), pointer :: UnitRadialVector_inPolar
  type(vector_field), pointer :: UnitPolarVector_inPolar
  type(vector_field), pointer :: UnitAzimuthalVector_inPolar
  type(vector_field) :: difference
  logical :: fail

  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  UnitRadialVector_inCartesian => extract_vector_field(state, "UnitRadialVector_inCartesian")
  UnitPolarVector_inCartesian => extract_vector_field(state, "UnitPolarVector_inCartesian")
  UnitAzimuthalVector_inCartesian => extract_vector_field(state, "UnitAzimuthalVector_inCartesian")
  UnitRadialVector_inPolar => extract_vector_field(state, "UnitRadialVector_inPolar")
  UnitPolarVector_inPolar => extract_vector_field(state, "UnitPolarVector_inPolar")
  UnitAzimuthalVector_inPolar => extract_vector_field(state, "UnitAzimuthalVector_inPolar")

  !Test the change of basis from spherical-polar to Cartesian.

  call allocate(difference, 3 , mesh, 'difference')

  !Set the components difference-vector equal to the unit radial vector, and then apply
  ! transformation to Cartesian basis. Then compare with vector already in Cartesian
  ! basis, obtained from vtu.
  call set(difference, UnitRadialVector_inPolar)
  call rotate_velocity_back_sphere(difference, state)
  call addto(difference, UnitRadialVector_inCartesian, -1.0)
  fail = any(difference%val > 1e-12)
  call report_test("[vector basis change: Spherical-polar to Cartesian of unit-radial vector.]", &
                   fail, .false., "Radial unit vector components not transformed correctly.")

  !Set the components difference-vector equal to the unit-polar vector, and then apply
  ! transformation to Cartesian basis. Then compare with vector already in Cartesian
  ! basis, obtained from vtu.
  call set(difference, UnitPolarVector_inPolar)
  call rotate_velocity_back_sphere(difference, state)
  call addto(difference, UnitPolarVector_inCartesian, -1.0)
  fail = any(difference%val > 1e-12)
  call report_test("[vector basis change: Spherical-polar to Cartesian of unit-polar vector.]", &
                   fail, .false., "Polar unit vector components not transformed correctly.")

  !Set the components difference-vector equal to the unit-azimuthal vector, and then apply
  ! transformation to Cartesian basis. Then compare with vector already in Cartesian
  ! basis, obtained from vtu.
  call set(difference, UnitAzimuthalVector_inPolar)
  call rotate_velocity_back_sphere(difference, state)
  call addto(difference, UnitAzimuthalVector_inCartesian, -1.0)
  fail = any(difference%val > 1e-12)
  call report_test("[vector basis change: Spherical-polar to Cartesian of unit-azimuthal vector.]", &
                   fail, .false., "Azimuthal unit vector components not transformed correctly.")

  call deallocate(difference)

end subroutine
