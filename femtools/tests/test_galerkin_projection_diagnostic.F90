subroutine test_galerkin_projection_diagnostic

  use diagnostic_fields
  use state_module
  use fields
  use read_triangle
  use unittest_tools
  use solvers
  use spud
  implicit none

  interface
    function field_func_tensor(pos) result(solution)
      real, dimension(:), intent(in) :: pos
      real, dimension(size(pos), size(pos)) :: solution
    end function
  end interface

  type(vector_field) :: positions
  type(tensor_field) :: tensA, tensB
  type(mesh_type) :: pwc_mesh
  character(len=255) :: path = "/solvers"
  type(state_type) :: state
  integer :: stat
  logical :: fail

  positions = read_triangle_files("data/pslgA", quad_degree=4)
  pwc_mesh = piecewise_constant_mesh(positions%mesh, "PiecewiseConstantMesh")
  call allocate(tensA, pwc_mesh, "TensorA")
  call set_from_function(tensA, field_func_tensor, positions)

  call allocate(tensB, positions%mesh, "TensorB")
  call zero(tensB)
  tensB%option_path = "/hello"
  call set_option(trim(tensB%option_path) // '/diagnostic/source_field_name', 'TensorA', stat=stat)

  call set_solver_options(path, ksptype='cg', pctype='sor', rtol=1.0e-10, max_its=10000)
  call insert(state, tensA, "TensorA")
  call insert(state, positions, "Coordinate")

  call calculate_galerkin_projection(state, tensB, path)

  fail = node_val(tensA, 1) .fne. node_val(tensB, 1)
  call report_test("[galerkin_projection_diagnostic]", fail, .false., "")

end subroutine test_galerkin_projection_diagnostic

function field_func_tensor(pos) result(solution)
  use unittest_tools
  real, dimension(:) :: pos
  real, dimension(size(pos), size(pos)) :: solution

  solution = get_matrix_identity(size(pos))
end function
