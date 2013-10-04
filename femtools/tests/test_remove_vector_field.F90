subroutine test_remove_vector_field

  use vtk_interfaces
  use state_module
  use unittest_tools
  use fields
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer  :: mesh
  type(vector_field) :: t_field
  logical :: fail

  call vtk_read_state("data/mesh_0.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  call allocate(t_field, 3, mesh, "VectorField")

  call insert(state, t_field, "VectorField")

  fail = .not. has_vector_field(state, "VectorField")
  call report_test("[remove_vector_field]", fail, .false., "")

  call remove_vector_field(state, "VectorField")
  fail = has_vector_field(state, "VectorField")
  call report_test("[remove_vector_field]", fail, .false., "")

  call deallocate(state)
end subroutine test_remove_vector_field
