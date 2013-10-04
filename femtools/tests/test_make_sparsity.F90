subroutine test_make_sparsity

  use vtk_interfaces
  use fields
  use sparsity_patterns
  use state_module
  use unittest_tools
  implicit none

  type(mesh_type), pointer :: rowmesh, colmesh
  type(state_type) :: state1, state2
  type(csr_sparsity) :: sparsity

  call vtk_read_state("data/sparsity_0.vtu", state1)
  rowmesh => extract_mesh(state1, "Mesh")

  call vtk_read_state("data/sparsity_1.vtu", state2)
  colmesh => extract_mesh(state2, "Mesh")

  sparsity = make_sparsity(rowmesh, colmesh, name='Sparsity')

  call report_test("[make sparsity]", .false., .false., "Make sparsity should run")

  call deallocate(state1)
  call deallocate(state2)
end subroutine test_make_sparsity
