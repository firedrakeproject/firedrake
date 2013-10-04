subroutine test_interpolation_quadratic

  use elements
  use read_triangle
  use fields
  use state_module
  use vector_tools
  use unittest_tools
  use interpolation_module
  use vtk_interfaces
  implicit none

  type(vector_field) :: old_positions, new_positions
  type(mesh_type) :: p2_old_mesh, p2_new_mesh
  type(scalar_field) :: p2_old_field, p2_new_field
  type(element_type) :: p2_shape
  type(state_type) :: old_state, new_state
  real :: old_integral, new_integral

  logical :: fail

  interface
    function solution(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  old_positions = read_triangle_files("data/pslgA", 4)
  new_positions = read_triangle_files("data/pslgB", 4)

  p2_shape = make_element_shape(vertices = 3, dim =2, degree=2, quad=old_positions%mesh%shape%quadrature)
  p2_old_mesh = make_mesh(old_positions%mesh, p2_shape, name="QuadraticMesh")
  call allocate(p2_old_field, p2_old_mesh, "P2Field")
  call set_from_function(p2_old_field, solution, old_positions)
  old_integral = field_integral(p2_old_field, old_positions)

  p2_new_mesh = make_mesh(new_positions%mesh, p2_shape, name="QuadraticMesh")
  call allocate(p2_new_field, p2_new_mesh, "P2Field")
  call zero(p2_new_field)

  call insert(old_state, old_positions, "Coordinate")
  call insert(old_state, p2_old_mesh, "Mesh")
  call insert(old_state, p2_old_field, "P2Field")

  call insert(new_state, new_positions, "Coordinate")
  call insert(new_state, p2_new_mesh, "Mesh")
  call insert(new_state, p2_new_field, "P2Field")

  call linear_interpolation(old_state, new_state)

  call vtk_write_state("data/quadratic_interpolation", 0, state=(/old_state/))
  call vtk_write_state("data/quadratic_interpolation", 1, state=(/new_state/))

  new_integral = field_integral(p2_new_field, new_positions)

  fail = (abs(old_integral - new_integral) > epsilon(0.0))
  call report_test("[test_interpolation_quadratic]", fail, .false., "Should be exact")

end subroutine test_interpolation_quadratic

function solution(pos)
  real, dimension(:) :: pos
  real :: solution
  real :: x,y
  x = pos(1); y = pos(2)

  solution = x**2
end function solution
