#include "fdebug.h"
subroutine test_bound_field

  use fields
  use vtk_interfaces
  use state_module
  use bound_field_module
  use unittest_tools
  use reference_counting
  implicit none

#ifdef HAVE_ALGENCAN

  type(state_type) :: state
  type(scalar_field), pointer :: galerkin_proj, ub, lb, lumped_mass, bounded_soln
  type(vector_field), pointer :: positions
  real :: old_integral, new_integral, ctol
  integer :: node
  logical :: fail

  call vtk_read_state("data/bound_field.vtu", state)

  galerkin_proj => extract_scalar_field(state, "MaterialVolumeFraction")
  ub => extract_scalar_field(state, "MaxBound")
  lb => extract_scalar_field(state, "MinBound")
  bounded_soln => extract_scalar_field(state, "BoundedSolution")
  lumped_mass => extract_scalar_field(state, "PressureMeshLumpedMassMatrixB")
  positions => extract_vector_field(state, "Coordinate")

  old_integral = dot_product(lumped_mass%val, galerkin_proj%val)

  call bound_field_algencan(galerkin_proj, positions, ub, lb, lumped_mass, bounded_soln)

  ctol = 1.0e4 * epsilon(0.0)
  fail = .false.
  do node=1,node_count(galerkin_proj)
    if (node_val(galerkin_proj, node) > node_val(ub, node) + ctol) then
      fail = .true.
    end if
    if (node_val(galerkin_proj, node) < node_val(lb, node) - ctol) then
      fail = .true.
    end if
  end do

  call report_test("[bound_field boundedness]", fail, .false., "Should be bounded")

  new_integral = dot_product(lumped_mass%val, galerkin_proj%val)
  fail = (abs(old_integral - new_integral) > ctol)
  call report_test("[bound_field conservation]", fail, .false., "Should be conservative")

  call vtk_write_fields("data/bound_field_out", position=positions, model=galerkin_proj%mesh, &
   sfields=(/galerkin_proj, ub, lb, bounded_soln, lumped_mass/))

  call deallocate(state)

  call print_references(-1)

#else
  call report_test("[bound_field dummy]", .false., .false., "You can't fail this one .. ")
#endif

end subroutine test_bound_field
