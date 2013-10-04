subroutine test_elementwise_fields

  use vtk_interfaces
  use fields
  use sparsity_patterns
  use state_module
  use unittest_tools
  implicit none

  type(state_type) :: state
  type(scalar_field) :: elementwise
  integer :: i

  call vtk_read_state("data/sparsity_0.vtu", state)
  elementwise = piecewise_constant_field(state%meshes(1), "Element numbering")

  do i=1,element_count(elementwise)
    call addto(elementwise, i, float(i))
  end do

  call insert(state, elementwise, "Element numbering")
  call vtk_write_state("data/elementwise", 0, state=state)

  call report_test("[elementwise fields]", .false., .false., "If it doesn't crash you're on to a winner")

  call deallocate(state)
end subroutine test_elementwise_fields
