subroutine test_seamount_hessian

  use vtk_interfaces
  use field_derivatives
  use unittest_tools
  use state_module
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: position_field
  type(scalar_field), pointer :: temp
  type(tensor_field) :: hessian
  logical :: fail

  call vtk_read_state("data/seamount.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  position_field => extract_vector_field(state, "Coordinate")
  temp => extract_scalar_field(state, "Temperature")
  call allocate(hessian, mesh, "Hessian")

  call compute_hessian(temp, position_field, hessian)
  call vtk_write_fields("data/seamount_hessian", 0, position_field, mesh, sfields=(/temp/), tfields=(/hessian/))

  fail = .false.
  call report_test("[seamount hessian]", fail, .false., "The hessian of x^2 is not what it should be!")

  call deallocate(hessian)
  call deallocate(state)

end subroutine test_seamount_hessian
