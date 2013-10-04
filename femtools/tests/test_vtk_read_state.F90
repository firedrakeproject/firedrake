subroutine test_vtk_read_state
  !!< Does vtk_read_state read in a mesh properly?

  use vtk_interfaces
  use elements
  use state_module
  use unittest_tools
  use fields
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer  :: mesh

  ! This is what it SHOULD be.
  integer :: nodes = 8
  integer :: elementcnt = 6
!  integer, dimension(24) :: ndglno = (/2, 4, 3, 7, 6, 7, 8, 4, 2, 7, 6, 4, 2, 1, 4, 5, 6, 8, 5, 4, 2, 6, 5, 4/)
  integer :: loccount = 4
  integer :: dim = 3
  integer :: degree = 1
  logical :: fail = .false., warn = .false.

  call vtk_read_state("data/mesh_0.vtu", state)
  mesh => extract_mesh(state, "Mesh")

  if (mesh%nodes /= nodes) fail = .true.
  call report_test("[vtk_read_state nodecount]", fail, warn, "Nodecount should be the known value.")

  fail = .false.
  if (mesh%elements /= elementcnt) fail = .true.
  call report_test("[vtk_read_state element count]", fail, warn, "Element count should be the known value.")


  fail = .false.
  if (ele_loc(mesh, 1) /= loccount) fail = .true.
  call report_test("[vtk_read_state loccount]", fail, warn, "Element type should be the known value.")

  fail = .false.
  if (mesh%shape%dim /= dim) fail = .true.
  call report_test("[vtk_read_state dim]", fail, warn, "Dimension should be the known value.")

  fail = .false.
  if (mesh%shape%degree /= degree) fail = .true.
  call report_test("[vtk_read_state degree]", fail, warn, "Polynomial degree should be the known value")

  fail = .false.
  if (associated(state%scalar_fields)) fail = .true.
  call report_test("[vtk_read_state fields]", fail, warn, "This VTU has no scalar fields.")

  call deallocate(state)
end subroutine test_vtk_read_state
