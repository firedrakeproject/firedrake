subroutine test_strain_rate

  use fields
  use field_derivatives
  use vtk_interfaces
  use read_triangle
  use state_module
  use unittest_tools
  implicit none

  type(vector_field) :: field
  type(tensor_field) :: strain_rate_field, solution_field, diff_field
  type(mesh_type), pointer :: mesh
  type(vector_field), target :: positions
  logical :: fail

  interface
    function velocity(pos)
      real, dimension(:), intent(in) :: pos
      real, dimension(size(pos)) :: velocity
    end function
    function solution(pos)
      real, dimension(:), intent(in) :: pos
      real, dimension(size(pos),size(pos)) :: solution
    end function
  end interface

  positions=read_triangle_files("data/cube.3", quad_degree=4)
  mesh => positions%mesh

  call allocate(field, 3, mesh, "Field")
  call allocate(strain_rate_field, mesh, "StrainRate")
  call allocate(solution_field, mesh, "Solution")
  call allocate(diff_field, mesh, "Difference")

  ! set our input velocity
  call set_from_function(field, velocity, positions)
  ! compute the strain rate
  call strain_rate(field, positions, strain_rate_field)
  ! now compute the expected solution
  call set_from_function(solution_field, solution, positions)

  call set(diff_field, strain_rate_field)
  call addto(diff_field, solution_field, scale=-1.0)

  call vtk_write_fields("data/strain_rate_out", 0, positions, mesh, &
     vfields=(/ field/), tfields=(/ strain_rate_field, solution_field, diff_field /))

  fail = maxval( abs( diff_field%val ))> 1e-10
  call report_test("[strain_rate]", fail, .false., "strain_rate different than expected")

end subroutine test_strain_rate

function velocity(pos)
  real, dimension(3) :: velocity
  real, dimension(:) :: pos
  real :: x,y,z
  x = pos(1); y = pos(2); z = pos(3)

  velocity(1) = x + 2*y + 3*z
  velocity(2) = 4*x + 5*y + 6*z
  velocity(3) = 7*x + 8*y + 9*z

end function velocity

function solution(pos)
  real, dimension(3,3) :: solution
  real, dimension(:) :: pos
  real :: x,y,z
  x = pos(1); y = pos(2); z = pos(3)

  solution(1,1)=1
  solution(1,2)=3
  solution(1,3)=5
  solution(2,1)=3
  solution(2,2)=5
  solution(2,3)=7
  solution(3,1)=5
  solution(3,2)=7
  solution(3,3)=9

end function solution
