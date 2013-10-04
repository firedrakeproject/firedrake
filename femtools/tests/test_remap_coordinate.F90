subroutine test_remap_coordinate

  use elements
  use fields
  use fields_data_types
  use reference_counting
  use state_module
  use unittest_tools
  use fefields

  implicit none

  type(quadrature_type) :: quad
  type(element_type) :: baseshape, toshape
  type(mesh_type) :: basemesh, tomesh
  type(vector_field) :: basex, tox

  integer :: i, j, dim, vertices

  dim = 2
  vertices = 3

  ! Make a P1 single triangle mesh
  quad = make_quadrature(vertices = vertices, dim  = dim, degree = 1)
  baseshape = make_element_shape(vertices = vertices, dim  = dim, degree = 1, quad = quad)
  toshape = make_element_shape(vertices = vertices, dim  = dim, degree = 0, quad = quad)
  call allocate(basemesh, nodes = baseshape%ndof, elements = 1, shape = baseshape, name = "BaseMesh")
  call allocate(tomesh, nodes = toshape%ndof, elements = 1, shape = toshape, name = "ToMesh")
  call allocate(basex, mesh_dim(basemesh), basemesh, "BaseCoordinate")
  call allocate(tox, mesh_dim(tomesh), tomesh, "ToCoordinate")

  do i = 1, size(basemesh%ndglno)
    basemesh%ndglno(i) = i
  end do

  do i = 1, size(tomesh%ndglno)
    tomesh%ndglno(i) = i
  end do

  call set(basex, 1, (/0.0, 0.0/))
  call set(basex, 2, (/1.0, 0.0/))
  call set(basex, 3, (/0.0, 1.0/))
!  call set(basex, 3, (/0.5, sqrt(3.0)/2.0/))

  do i = 1, node_count(basex)
    write(0,*) 'i = ', i
    do j = 1, mesh_dim(basex)
      write(0,*) 'dim = ', j
      write(0,*) node_val(basex, i, j)
    end do
  end do

  call remap_field(from_field=basex, to_field=tox)

  do i = 1, node_count(tox)
    write(0,*) 'i = ', i
    do j = 1, mesh_dim(tox)
      write(0,*) 'dim = ', j
      write(0,*) node_val(tox, i, j)
    end do
  end do

end subroutine test_remap_coordinate
