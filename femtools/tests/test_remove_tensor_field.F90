subroutine test_remove_tensor_field

  use vtk_interfaces
  use state_module
  use unittest_tools
  use fields
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer  :: mesh
  type(tensor_field) :: t_field
  logical :: fail

  call vtk_read_state("data/mesh_0.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  call allocate(t_field, mesh, "TensorField")

  call insert(state, t_field, "TensorField")

  fail = .not. has_tensor_field(state, "TensorField")
  call report_test("[remove_tensor_field]", fail, .false., "")

  call remove_tensor_field(state, "TensorField")
  fail = has_tensor_field(state, "TensorField")
  call report_test("[remove_tensor_field]", fail, .false., "")

  call deallocate(state)
end subroutine test_remove_tensor_field
