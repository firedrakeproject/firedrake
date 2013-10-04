subroutine test_compute_inner_product_sa

  use conservative_interpolation_module
  use fields
  use interpolation_module
  use read_triangle
  use state_module
  use supermesh_assembly
  use unittest_tools

  implicit none

  type(vector_field) :: positions_a, positions_b, positions_remap
  type(mesh_type) :: mesh_a, mesh_b, mesh_b_proj
  type(scalar_field) :: field_a, field_b, field_b_proj, field_b_proj2
  type(element_type) :: shape, positions_shape
  type(state_type), dimension(1) :: state_a, state_b
  integer :: i
  real :: prod
  logical :: fail

  positions_a = read_triangle_files("data/laplacian_grid.1", quad_degree=4)
  positions_b = read_triangle_files("data/laplacian_grid.2", quad_degree=4)
  !positions_a = read_triangle_files("data/cube.1", quad_degree=4)
  !positions_b = read_triangle_files("data/cube.2", quad_degree=4)
  !positions_a%val(1,:) = positions_a%val(1,:) + 3.0
  !call scale(positions_a, 1.0 / 6.0)
  !positions_b%val(1,:) = positions_b%val(1,:) + 3.0
  !call scale(positions_b, 1.0 / 6.0)
  !positions_b = positions_a
  !call incref(positions_b)

  positions_shape = ele_shape(positions_a, 1)
  shape = make_element_shape(vertices = ele_loc(positions_a, 1), dim  = positions_a%dim, degree = 1, quad = positions_shape%quadrature)
  mesh_a = make_mesh(positions_a%mesh, shape = shape, continuity = +1, name = "MeshA")
  mesh_b = make_mesh(positions_b%mesh, shape = shape, continuity = +1, name = "MeshB")
  call deallocate(shape)

  call allocate(field_a, mesh_a, "FieldA", field_type = FIELD_TYPE_CONSTANT)
  call allocate(field_b, mesh_b, "FieldB", field_type = FIELD_TYPE_CONSTANT)
  call deallocate(mesh_a)
  call deallocate(mesh_b)

  call set(field_a, 2.5)
  call set(field_b, 1.2)

  prod = compute_inner_product_sa(positions_a, positions_b, field_a, field_b)
  fail = (prod .fne. 3.0)
  call report_test("[test_compute_inner_product_sa]", fail, .false., "Should be 3")
  if(fail) then
    print *, "But got ", prod
  end if

  call deallocate(field_a)
  call deallocate(field_b)

  shape = make_element_shape(vertices = ele_loc(positions_a, 1), dim  = positions_a%dim, degree = 1, quad = positions_shape%quadrature)
  mesh_a = make_mesh(positions_a%mesh, shape = shape, continuity = +1, name = "MeshA")
  call deallocate(shape)
  shape = make_element_shape(vertices = ele_loc(positions_b, 1), dim  = positions_b%dim, degree = 1, quad = positions_shape%quadrature)
  mesh_b = make_mesh(positions_b%mesh, shape = shape, continuity = +1, name = "MeshB")
  call deallocate(shape)

  call allocate(field_a, mesh_a, "FieldA")
  call allocate(field_b, mesh_b, "FieldB")

  call allocate(positions_remap, positions_a%dim, mesh_a, "CoordinateRemap")
  call remap_field(positions_a, positions_remap)
  do i = 1, node_count(field_a)
    call set(field_a, i, node_val(positions_remap, 1, i))
  end do
  call deallocate(positions_remap)

  call allocate(positions_remap, positions_a%dim, mesh_b, "CoordinateRemap")
  call remap_field(positions_b, positions_remap)
  do i = 1, node_count(field_b)
    call set(field_b, i, node_val(positions_remap, 1, i))
  end do
  call deallocate(positions_remap)

  prod = compute_inner_product_sa(positions_a, positions_b, field_a, field_b)
  fail = (prod .fne. 1.0/3.0)
  call report_test("[test_compute_inner_product_sa]", fail, .false., "Should be 1.0 / 3.0")
  if(fail) then
    print *, "But got ", prod
  end if

  call deallocate(field_a)
  call deallocate(field_b)
  call deallocate(mesh_a)
  call deallocate(mesh_b)

  shape = make_element_shape(vertices = ele_loc(positions_a, 1), dim  = positions_a%dim, degree = 1, quad = positions_shape%quadrature)
  mesh_a = make_mesh(positions_a%mesh, shape = shape, continuity = -1, name = "MeshA")
  call deallocate(shape)
  shape = make_element_shape(vertices = ele_loc(positions_b, 1), dim  = positions_b%dim, degree = 2, quad = positions_shape%quadrature)
  mesh_b = make_mesh(positions_b%mesh, shape = shape, continuity = -1, name = "MeshB")
  call deallocate(shape)

  call allocate(field_a, mesh_a, "FieldA")
  call allocate(field_b, mesh_b, "FieldB")

  call allocate(positions_remap, positions_a%dim, mesh_a, "CoordinateRemap")
  call remap_field(positions_a, positions_remap)
  do i = 1, node_count(field_a)
    call set(field_a, i,node_val(positions_remap, 1, i))
  end do
  call deallocate(positions_remap)

  call allocate(positions_remap, positions_a%dim, mesh_b, "CoordinateRemap")
  call remap_field(positions_b, positions_remap)
  do i = 1, node_count(field_b)
    call set(field_b, i, node_val(positions_remap, 1, i) ** 2)
  end do
  call deallocate(positions_remap)

  prod = compute_inner_product_sa(positions_a, positions_b, field_a, field_b)
  fail = (prod .fne. 0.25)
  call report_test("[test_compute_inner_product_sa]", fail, .false., "Should be 0.25")

  print *, prod

  !shape = make_element_shape(vertices = ele_loc(positions_b, 1), dim  = positions_b%dim, degree = 2, quad = positions_shape%quadrature)
  shape = make_element_shape(vertices = ele_loc(positions_b, 1), dim  = positions_b%dim, degree = 1, quad = positions_shape%quadrature)

  mesh_b_proj = make_mesh(positions_a%mesh, shape = shape, continuity = -1, name = "MeshBProj")
  call deallocate(shape)
  call allocate(field_b_proj, mesh_b_proj, "FieldBProjected")
  call deallocate(mesh_b_proj)
  call insert(state_a(1), field_b_proj, name = field_a%name)
  call insert(state_b(1), field_b, name = field_a%name)

  call insert(state_a(1), positions_a, name = positions_a%name)
  call insert(state_b(1), positions_b, name = positions_b%name)
  call insert(state_a(1), mesh_a, name = mesh_a%name)
  call insert(state_b(1), mesh_b, name = mesh_b%name)
  call linear_interpolation(state_b(1), state_a(1))
  !call interpolation_galerkin(state_b, positions_b, state_a, positions_a)

  call deallocate(state_a(1))
  call deallocate(state_b(1))

  !call allocate(field_b_proj2, mesh_a, "FieldBProjected")
  !do i = 1, ele_count(field_b_proj)
  !  call gp_ele(i, positions_a, field_b_proj, field_b_proj2)
  !end do
  !print *, compute_inner_product(positions_a, field_a, field_b_proj2)
  !print *, compute_inner_product(positions_a, field_a, field_b_proj)

  call deallocate(field_b_proj)
  !call deallocate(field_b_proj2)

  call deallocate(field_a)
  call deallocate(field_b)
  call deallocate(mesh_a)
  call deallocate(mesh_b)

  call deallocate(positions_a)
  call deallocate(positions_b)

  call report_test_no_references()

contains

  subroutine gp_ele(ele, positions, field_a, field_b)
    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(scalar_field), intent(in) :: field_a
    type(scalar_field), intent(inout) :: field_b

    real, dimension(ele_ngi(positions, ele)) :: detwei
    real, dimension(ele_loc(field_b, ele), ele_loc(field_b, ele)) :: mass
    real, dimension(ele_loc(field_b, ele)) :: rhs

    call transform_to_physical(positions, ele, detwei = detwei)

    mass = shape_shape(ele_shape(field_b, ele), ele_shape(field_b, ele), detwei)
    rhs = shape_rhs(ele_shape(field_b, ele), detwei * ele_val_at_quad(field_a, ele))

    call solve(mass, rhs)
    call set(field_b, ele_nodes(field_b, ele), rhs)

  end subroutine gp_ele

  function compute_inner_product(positions, field_a, field_b) result(val)
    type(vector_field), intent(in) :: positions
    type(scalar_field), intent(in) :: field_a
    type(scalar_field), intent(in) :: field_b

    real :: val

    integer :: i

    val = 0.0
    do i = 1, ele_count(positions)
      call add_inner_product_ele(i, positions, field_a, field_b, val)
    end do

  end function compute_inner_product

  subroutine add_inner_product_ele(ele, positions, field_a, field_b, val)
    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(scalar_field), intent(in) :: field_a
    type(scalar_field), intent(in) :: field_b
    real, intent(inout) :: val

    real, dimension(ele_ngi(positions, ele)) :: detwei

    call transform_to_physical(positions, ele, detwei = detwei)

    val = val + dot_product(ele_val(field_a, ele), matmul(&
      &  shape_shape(ele_shape(field_a, ele), ele_shape(field_b, ele), detwei), ele_val(field_b, ele)))

  end subroutine add_inner_product_ele

end subroutine test_compute_inner_product_sa
