subroutine test_div

  use fields
  use field_derivatives
  use vtk_interfaces
  use state_module
  use unittest_tools
  implicit none

  type(vector_field) :: field
  type(scalar_field) :: divergence
  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: positions
  logical :: fail

  interface
    function solution(pos)
      real, dimension(:), intent(in) :: pos
      real, dimension(size(pos)) :: solution
    end function
  end interface

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  positions => extract_vector_field(state, "Coordinate")
  call allocate(field, 3, mesh, "Field")
  call allocate(divergence, mesh, "Div")


  call set_from_function(field, solution, positions)
  call div(field, positions, divergence)

  call vtk_write_fields("data/div_out", 0, positions, mesh, sfields=(/divergence/), &
                                                           & vfields=(/field/))


  fail = any(divergence%val > 1e-12)
  call report_test("[div]", fail, .false., "div(constant) == 0 everywhere, remember?")

end subroutine test_div

function solution(pos)
  real, dimension(:) :: pos
  real, dimension(size(pos)) :: solution
  real :: x,y,z
  x = pos(1); y = pos(2); z = pos(3)

  solution = (/1.0, 2.0, 3.0/)
end function solution

