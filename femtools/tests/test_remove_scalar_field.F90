subroutine test_remove_scalar_field

  use vtk_interfaces
  use state_module
  use unittest_tools
  use fields
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer  :: mesh
  type(scalar_field) :: t_field
  logical :: fail

  call vtk_read_state("data/mesh_0.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  call allocate(t_field, mesh, "ScalarField")

  call insert(state, t_field, "ScalarField")

  fail = .not. has_scalar_field(state, "ScalarField")
  call report_test("[remove_scalar_field]", fail, .false., "")

  call remove_scalar_field(state, "ScalarField")
  fail = has_scalar_field(state, "ScalarField")
  call report_test("[remove_scalar_field]", fail, .false., "")

  call deallocate(state)
end subroutine test_remove_scalar_field
