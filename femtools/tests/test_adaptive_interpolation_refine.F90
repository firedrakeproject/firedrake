subroutine test_adaptive_interpolation_refine

  use quadrature
  use fields
  use adaptive_interpolation_module
  use read_triangle
  use unittest_tools
  use vtk_interfaces
  implicit none

  type(vector_field) :: positionsA, positionsB
  type(mesh_type) :: dg_mesh, quadratic_mesh
  type(scalar_field) :: in_field, out_field
  type(element_type) :: quadratic_shape
  real :: achieved_error
  integer :: no_refinements
  logical :: fail
  real :: error_tolerance

  interface
    function field_func(pos)
      real, dimension(:) :: pos
      real :: solution
    end function
  end interface

  call set_global_debug_level(3)

  positionsA = read_triangle_files("data/laplacian_grid.2", quad_degree=2*max_ai_degree, no_faces=.true., quad_family=FAMILY_GM)
  positionsB = read_triangle_files("data/laplacian_grid.3", quad_degree=2*max_ai_degree, no_faces=.true., quad_family=FAMILY_GM)

  quadratic_shape = make_element_shape(vertices = ele_loc(positionsA, 1), dim =positionsA%dim, degree=2, quad=positionsA%mesh%shape%quadrature)
  quadratic_mesh = make_mesh(positionsA%mesh, quadratic_shape, -1, "QuadraticDgMesh")
  call allocate(in_field, quadratic_mesh, "InField")
  call set_from_function(in_field, field_func, positionsA)

  dg_mesh = make_mesh(positionsB%mesh, positionsB%mesh%shape, -1, "DgMesh")
  call allocate(out_field, dg_mesh, "OutField")
  call set(out_field, -10000.0)

  error_tolerance = 5.0e-12

  call adaptive_interpolation(in_field, positionsA, out_field, positionsB, error_tolerance, achieved_error, no_refinements)

  call vtk_write_fields("data/adaptive_interpolation_p", position=positionsB, model=out_field%mesh, write_region_ids=.true.)

  write(0,*) "achieved_error: ", achieved_error
  write(0,*) "error_tolerance: ", error_tolerance

  fail = (achieved_error > error_tolerance)
  call report_test("[adaptive interpolation refine]", fail, .false., "Achieved error must be less than tolerance")

  fail = (no_refinements /= 3)
  call report_test("[adaptive interpolation refine]", fail, .false., "And you shouldn't need to refine to get it!")

end subroutine test_adaptive_interpolation_refine

function field_func(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = pos(1)**3 + 3*pos(2)**2
end function field_func
