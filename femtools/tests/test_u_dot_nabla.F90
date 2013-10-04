! Tests the u_dot_nabla routine

#include "fdebug.h"

subroutine test_u_dot_nabla

  use field_derivatives
  use fields
  use fields_data_types
  use state_module
  use unittest_tools
  use vtk_interfaces

  implicit none

  character(len = 32) :: buffer
  integer :: i
  real :: max_norm, max_val
  real, dimension(3) :: pos
  type(mesh_type), pointer :: mesh
  type(state_type) :: state
  type(vector_field) ::  u_dot_nabla_field, vel_field
  type(vector_field), pointer :: positions

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  assert(mesh_dim(mesh) == 3)

  call allocate(u_dot_nabla_field, 3, mesh, "UDotNabla")
  call allocate(vel_field, 3, mesh, "Velocity")

  positions => extract_vector_field(state, "Coordinate")
  do i = 1, node_count(vel_field)
    pos = node_val(positions, i)
    call set(vel_field, i, (/pos(2), pos(1), 0.0/))
  end do

  call u_dot_nabla(vel_field, vel_field, positions, u_dot_nabla_field)

  call vtk_write_fields("data/test_u_dot_nabla_out", 0, positions, mesh, &
    & vfields = (/positions, u_dot_nabla_field, vel_field/))

  max_val = 0.0
  max_norm = 0.0
  do i = 1, node_count(u_dot_nabla_field)
    pos = node_val(positions, i)
    max_norm = max(max_norm, norm2(node_val(u_dot_nabla_field, i)))
    max_val = max(max_val, &
      & norm2((/pos(1), pos(2), 0.0/) - node_val(u_dot_nabla_field, i)))
  end do

  write(buffer, *) max_val
  call report_test("[(u dot nabla) test: Solid body rotation]", &
    & fnequals(max_val, 0.0, tol = spacing(max_norm) * 100.0), .false., &
    & "(u dot nabla) u /= r - Max. difference norm2: " // buffer)

  call deallocate(u_dot_nabla_field)
  call deallocate(vel_field)

  call deallocate(state)

end subroutine test_u_dot_nabla
