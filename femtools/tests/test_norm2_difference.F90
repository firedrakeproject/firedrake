subroutine test_norm2_difference

  use read_triangle
  use fields
  use unittest_tools
  implicit none

  type(vector_field) :: positionsA, positionsB
  type(mesh_type) :: meshA, meshB
  type(element_type) :: shape
  type(scalar_field) :: fieldA, fieldB
  integer :: degree
  real :: norm
  logical :: fail
  integer :: i
  real :: domain_volume
  integer :: ele

  positionsA = read_triangle_files("data/pslgA", quad_degree=4)
  positionsB = read_triangle_files("data/pslgB", quad_degree=4)

  domain_volume = 0.0
  do ele=1,ele_count(positionsA)
    domain_volume = domain_volume + simplex_volume(positionsA, ele)
  end do

  do degree=1,3
    shape = make_element_shape(vertices = ele_loc(positionsA, 1), dim  = positionsA%dim, degree = degree, quad = positionsA%mesh%shape%quadrature)
    meshA = make_mesh(positionsA%mesh, shape, name="MeshA")
    meshB = make_mesh(positionsB%mesh, shape, name="MeshB")
    call allocate(fieldA, meshA, "FieldA")
    call allocate(fieldB, meshB, "FieldB")

    fieldA%val = 1.0
    fieldB%val = 0.0

    norm = norm2_difference(fieldA, positionsA, fieldB, positionsB)
    fail = (norm .fne. norm2(fieldA, positionsA))
    call report_test("[norm2 difference]", fail, .false., "|A - 0| = |A|")

    do i=1,size(fieldA%val)
      call random_number(fieldA%val(i))
    end do

    norm = norm2_difference(fieldA, positionsA, fieldA, positionsA)
    fail = (norm .fne. 0.0)
    call report_test("[norm2 difference]", fail, .false., "|A - A| = |0| = 0")

    fieldA%val = 1.0
    fieldB%val = 2.0

    norm = norm2_difference(fieldA, positionsA, fieldB, positionsB)
    fail = (norm .fne. sqrt(domain_volume))
    call report_test("[norm2 difference]", fail, .false., "|1| = |\Omega|**0.5")

    call deallocate(fieldA)
    call deallocate(fieldB)
    call deallocate(meshA)
    call deallocate(meshB)
    call deallocate(shape)
  end do

end subroutine test_norm2_difference
