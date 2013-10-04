subroutine test_tensor_second_invariant

  use fields
  use field_derivatives
  use vtk_interfaces
  use read_triangle
  use state_module
  use unittest_tools
  implicit none

  type(scalar_field) :: solution_field, tensor_second_invariant_field, diff_field
  type(tensor_field) :: field
  type(mesh_type), pointer :: mesh
  type(vector_field), target :: positions
  logical :: fail

  interface
    function strainrate(pos)
      real, dimension(:), intent(in) :: pos
      real, dimension(size(pos), size(pos)) :: strainrate
    end function
    function solution(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  positions=read_triangle_files("data/cube.3", quad_degree=4)
  mesh => positions%mesh

  call allocate(field, mesh, "Field")
  call allocate(tensor_second_invariant_field, mesh, "TensorSecondInvariant")
  call allocate(solution_field, mesh, "Solution")
  call allocate(diff_field, mesh, "Difference")

  ! set our input strain rate
  call set_from_function(field, strainrate, positions)
  ! compute the second invariant of the strain rate tensor
  call tensor_second_invariant(field, tensor_second_invariant_field)
  ! now compute the expected solution
  call set_from_function(solution_field, solution, positions)

  call set(diff_field, tensor_second_invariant_field)
  call addto(diff_field, solution_field, scale=-1.0)

  call vtk_write_fields("data/tensor_second_invariant_out", 0, positions, mesh, &
     sfields=(/ tensor_second_invariant_field, solution_field, diff_field /), &
     tfields=(/ field  /))

  fail = maxval( abs( diff_field%val ))> 1e-10
  call report_test("[tensor_second_invariant]", fail, .false., "second invariant different than expected")

end subroutine test_tensor_second_invariant

function strainrate(pos)
  real, dimension(3,3) :: strainrate
  real, dimension(:) :: pos
  real :: x,y,z
  x = pos(1); y = pos(2); z = pos(3)

  strainrate(1,1)=1*x*z
  strainrate(1,2)=3*x
  strainrate(1,3)=5*z
  strainrate(2,1)=3*x
  strainrate(2,2)=-2*x*z
  strainrate(2,3)=7
  strainrate(3,1)=5*z
  strainrate(3,2)=7
  strainrate(3,3)=1*x*z

end function strainrate

function solution(pos)
  real, dimension(:) :: pos
  real :: solution
  real :: x,y,z, sum

  x = pos(1); y = pos(2) ; z = pos(3)

  sum = (1*x*z)**2 + (3*x)**2    + (5*z)**2   &
      + (3*x)**2   + (-2*x*z)**2 + (7)**2     &
      + (5*z)**2   + (7)**2      + (1*x*z)**2

  solution = sqrt(sum / 2.)

end function solution

