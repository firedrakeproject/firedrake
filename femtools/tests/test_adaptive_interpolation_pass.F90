subroutine test_adaptive_interpolation_pass

  use quadrature
  use fields
  use adaptive_interpolation_module
  use read_triangle
  use unittest_tools
  implicit none

  type(vector_field) :: positionsA, positionsB
  type(mesh_type) :: dg_mesh
  type(scalar_field) :: in_field, out_field
  real :: achieved_error
  integer :: no_refinements
  logical :: fail
  real :: error_tolerance

  interface
    function field_func(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  positionsA = read_triangle_files("data/pslgA", quad_degree=2*max_ai_degree, no_faces=.true., quad_family=FAMILY_GM)
  positionsB = read_triangle_files("data/pslgA", quad_degree=2*max_ai_degree, no_faces=.true., quad_family=FAMILY_GM)

  call allocate(in_field, positionsA%mesh, "InField")
  call set_from_function(in_field, field_func, positionsA)

  dg_mesh = make_mesh(positionsB%mesh, positionsB%mesh%shape, -1, "DgMesh")
  call allocate(out_field, dg_mesh, "OutField")
  call set(out_field, -10000.0)

  error_tolerance = 1.0e-14

  call adaptive_interpolation(in_field, positionsA, out_field, positionsB, error_tolerance, achieved_error, no_refinements)

  fail = (achieved_error > error_tolerance)
  call report_test("[adaptive interpolation pass]", fail, .false., "Achieved error must be less than tolerance")

  fail = (no_refinements /= 0)
  call report_test("[adaptive interpolation pass]", fail, .false., "And you shouldn't need to refine to get it!")

end subroutine test_adaptive_interpolation_pass

function field_func(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = pos(1) + pos(2)
end function field_func
