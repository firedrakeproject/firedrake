subroutine test_invert_dg_mass_matrix

  use fldebug
  use fields
  use read_triangle
  use DGtools
  use sparse_tools
  use transform_elements
  use Unittest_tools
  use FETOols

  type(csr_matrix) :: inverse_mass
  type(scalar_field), target :: u, rhs, u_check
  integer :: quad_degree
  type(quadrature_type), target :: quad
  type(element_type), target :: X_shape, u_shape
  type(mesh_type) :: u_mesh
  type(vector_field) :: positions

  logical :: fail, warn

  quad_degree = 4
  quad=make_quadrature(vertices = 3, dim =2, degree=quad_degree)
  X_shape=make_element_shape(vertices = 3, dim =2, degree=1, quad=quad)
  positions=read_triangle_files('data/square.1', X_shape)
  u_shape=make_element_shape(vertices = 3, dim =2, degree=1, quad=quad)
  u_mesh = make_mesh(positions%mesh,u_shape,-1,'u_mesh')

  call get_dg_inverse_mass_matrix(inverse_mass,u_mesh,positions)

  call allocate(rhs,u_mesh,'RHS')
  call allocate(u,u_mesh,'u')
  call allocate(u_check,u_mesh,'u_check')

  u%val = 1.0

  call get_rhs(rhs,u,positions)

  u_check%val = 0.0
  call mult(u_check%val,inverse_mass,rhs%val)

  warn = maxval(abs(u%val-u_check%val))>1.0e-13
  fail = maxval(abs(u%val-u_check%val))>1.0e-10

  call report_test("[inverse dg mass matrix formed correctly using dynamic csr matrices]", warn, fail, &
  "Inverse dg mass matrix not formed correctly")

  contains

    subroutine get_rhs(rhs,u,positions)
      type(scalar_field), intent(in) :: u
      type(scalar_field), intent(inout) :: rhs
      type(vector_field), intent(in) :: positions

      !
      integer :: ele

      call zero(rhs)

      do ele = 1, element_count(u)

         call assemble_rhs(ele,rhs,u,positions)

      end do

    end subroutine get_rhs

    subroutine assemble_rhs(ele,rhs,u,positions)
      integer, intent(in) :: ele
      type(scalar_field), intent(in) :: u
      type(scalar_field), intent(inout) :: rhs
      type(vector_field), intent(in) :: positions

      ! Coordinate transform * quadrature weights.
      real, dimension(ele_ngi(positions,ele)) :: detwei
      ! Node numbers of field element.
      integer, dimension(:), pointer :: ele_u
      ! Shape functions.
      type(element_type), pointer :: shape_u
      ! local mass matrix
      real, dimension(ele_loc(u,ele),ele_loc(u,ele)) :: mass_loc

      ele_u=>ele_nodes(u, ele)
      shape_u=>ele_shape(u, ele)
      ! Transform derivatives and weights into physical space.
      call transform_to_physical(positions, ele, detwei=detwei)

      mass_loc = shape_shape(shape_u, shape_u, detwei)

      call addto(rhs,ele_u,matmul(mass_loc,u%val(ele_u)))

    end subroutine assemble_rhs

end subroutine test_invert_dg_mass_matrix
