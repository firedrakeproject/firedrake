subroutine test_curl

  use fields
  use field_derivatives
  use vtk_interfaces
  use state_module
  use unittest_tools
  implicit none

  type(scalar_field) :: field
  type(vector_field) :: grad_field, curl_field
  type(scalar_field) :: curl_norm
  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: positions
  logical :: fail

  interface
    function solution(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  positions => extract_vector_field(state, "Coordinate")
  call allocate(field, mesh, "Field")
  call allocate(grad_field, 3, mesh, "Vfield")
  call allocate(curl_norm, mesh, "Norm of curl")
  call allocate(curl_field, 3, mesh, "Curl")


  call set_from_function(field, solution, positions)
  call grad(field, positions, grad_field)
  call curl(grad_field, positions, curl_norm=curl_norm, curl_field=curl_field)

  call vtk_write_fields("data/curl_out", 0, positions, mesh, sfields=(/field, curl_norm/), &
                                                           & vfields=(/curl_field/))


  fail = curl_norm%val .fne. 0.0
  call report_test("[curl]", fail, .false., "curl(grad(phi)) == 0 everywhere, remember?")

end subroutine test_curl

function solution(pos)
  real :: solution
  real, dimension(:) :: pos
  real :: x,y,z
  x = pos(1); y = pos(2); z = pos(3)

  solution = x + y
end function solution

