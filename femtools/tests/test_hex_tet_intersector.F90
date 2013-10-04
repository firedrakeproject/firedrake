subroutine test_hex_tet_intersector

  use read_triangle
  use tetrahedron_intersection_module
  use fields
  use unittest_tools
  use vtk_interfaces

  type(vector_field) :: cube, tet, intersection
  integer :: stat
  type(plane_type), dimension(6) :: planes
  type(tet_type) :: tet_t
  logical :: fail
  real :: tet_vol, int_vol

  cube = read_triangle_files("data/unit_cube", quad_degree=1)
  tet  = read_triangle_files("data/unit_tet",  quad_degree=1)

  planes = get_planes(cube, 1)
  tet_t%v = ele_val(tet, 1)

  call intersect_tets(tet_t, planes, ele_shape(tet, 1), stat=stat, output=intersection)
  fail = (stat /= 0)

  call report_test("[hex tet intersector existence]", fail, .false., "")

  tet_vol = simplex_volume(tet, 1)
  int_vol = abs(simplex_volume(intersection, 1))

  fail = (tet_vol .fne. int_vol)

  call report_test("[hex tet intersector volume]", fail, .false., "")

  call deallocate(cube)
  call deallocate(tet)
  call deallocate(intersection)

end subroutine test_hex_tet_intersector
