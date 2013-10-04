subroutine test_scalar_field_view

  use fields
  use state_module
  use vtk_interfaces
  use unittest_tools
  implicit none

  type(state_type) :: state
  type(scalar_field), pointer :: x
  integer :: stat
  logical :: allocated
  logical :: fail

  call vtk_read_state("data/pseudo2d.vtu", state)
  x => extract_scalar_field(state, "Coordinatz%1", stat=stat, allocated=allocated) ! should fail
  fail = (stat == 0 .or. allocated)
  call report_test("[scalar field view]", fail, .false., "Searching for a component of a nonexistant field is a problem.")
  x => extract_scalar_field(state, "Coordinate%10", stat=stat, allocated=allocated) ! should fail
  fail = (stat == 0 .or. allocated)
  call report_test("[scalar field view]", fail, .false., "Searching for a nonexistant component of a field is a problem.")
  x => extract_scalar_field(state, "Coordinate%1", stat=stat, allocated=allocated) ! should work
  fail = (stat /= 0 .or. (allocated .eqv. .false.) .or. (.not. associated(x)))
  call report_test("[scalar field view]", fail, .false., "Searching for a component of a field should work.")
end subroutine test_scalar_field_view
