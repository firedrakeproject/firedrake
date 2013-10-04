subroutine test_python_fields

  use fields
  use vtk_interfaces
  use state_module
  use unittest_tools
  use global_parameters, only: PYTHON_FUNC_LEN, ACCTIM
  implicit none

  type(scalar_field) :: sfield
  type(vector_field) :: vfield
  type(tensor_field) :: tfield
  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: positions
  logical :: fail
  real, dimension(4) :: ele_s
  real, dimension(4) :: posx_s
  real, dimension(3, 4) :: ele_v
  real, dimension(3, 3, 4) :: ele_t
  real :: node_s
  real, dimension(3) :: node_v
  real, dimension(3, 3) :: node_t
  integer :: i, j, k, l
  character, parameter:: NEWLINE_CHAR=achar(10)
  character(len=*), parameter:: func = &
     "def val(X, t):"//NEWLINE_CHAR// &
     "  import math"//NEWLINE_CHAR// &
     "  return X[0]"
  external :: compute_nodes_python, compute_nodes_stored

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  positions => extract_vector_field(state, "Coordinate")

  ACCTIM = 0.0


  call allocate(sfield, mesh, "ScalarField", FIELD_TYPE_PYTHON, py_func=func, py_positions=positions)

  do i=1,5
    call compute_nodes_python(sfield)
    call compute_nodes_stored(positions)
  end do

  call deallocate(sfield)
end subroutine test_python_fields

subroutine compute_nodes_python(sfield)
  use fields
  type(scalar_field), intent(in) :: sfield

  integer :: node, ele
  real :: whatever
  real, dimension(4) :: whatever_n

!  do node=1,node_count(sfield)
!    whatever = node_val(sfield, node)
!  end do
  do ele=1,ele_count(sfield)
    whatever_n = ele_val(sfield, ele)
  end do
end subroutine compute_nodes_python

subroutine compute_nodes_stored(vfield)
  use fields
  type(vector_field), intent(in) :: vfield

  integer :: node, ele
  real :: whatever
  real, dimension(4) :: whatever_n

!  do node=1,node_count(vfield)
!    whatever = node_val(vfield, node, 1)
!  end do
  do ele=1,ele_count(vfield)
    whatever_n = ele_val(vfield, ele, 1)
  end do
end subroutine compute_nodes_stored

