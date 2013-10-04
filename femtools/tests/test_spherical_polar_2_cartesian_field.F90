subroutine test_spherical_polar_2_cartesian_field
  !Test for routine test_spherical_polar_2_cartesian_field, ensuring conversion of a vector field
  ! containing the position vector in sperical-polar coordiantes into a vector field containing
  ! the position vector in Cartesian coordinates is correct.

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
  logical :: fail

  call vtk_read_state("data/on_sphere_rotations/spherical_shell_withFields.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  CartesianCoordinate => extract_vector_field(state, "CartesianCoordinate")
  PolarCoordinate => extract_vector_field(state, "PolarCoordinate")

  !Apply transformation to polar-coordinate field obtained from vtu file and
  ! compare with cartesian position-vector.
  call allocate(difference, 3 , mesh, 'difference')
  call spherical_polar_2_cartesian(PolarCoordinate, difference)
  call addto(difference, CartesianCoordinate, -1.0)

  fail = any(difference%val > 1e-8)
  call report_test("[Coordinate change of whole field: Spherical-polar to Cartesian.]", &
                   fail, .false., "Position vector components not transformed correctly.")

  call deallocate(difference)

end subroutine
