subroutine test_node_lies_on_boundary

  use node_boundary
  use state_module
  use vtk_interfaces
  use unittest_tools
  use node_boundary, only: pseudo2d_coord
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: positions
  integer :: i
  real :: x, y, z
  logical :: fail, expected, output
  integer, dimension(:), pointer:: surf_ids

  pseudo2d_coord = 3

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  positions => extract_vector_field(state, "Coordinate")

  call add_faces(mesh)
  ! Update the mesh descriptor on positions to have faces.
  positions%mesh=mesh

  allocate(surf_ids(surface_element_count(mesh)))
  call get_coplanar_ids(mesh, positions, surf_ids)
  call initialise_boundcount(mesh, positions)

  fail = .false.
  do i=1,mesh%nodes
    x = positions%val(1,i)
    y = positions%val(2,i)
    z = positions%val(3,i)

    output = node_lies_on_boundary(mesh, positions, i, expected=1)
    if (x == 0.0 .or. x == 30.0 .or. y == 0.0 .or. y == 15.0) then
      expected = .true.
    else
      expected = .false.
    end if

    if (output .neqv. expected) then
      fail = .true.
      write(0,*) "node: ", i
      write(0,*) "position: (", x, ", ", y, ", ", z, ")"
      write(0,*) "expected: ", expected
      write(0,*) "output: ", output
    end if
  end do

  positions%mesh=mesh
  call vtk_write_surface_mesh("coplanar_ids", index = 0, position = positions)

  call report_test("[node_lies_on_boundary 2d]", fail, .false., "Output &
  & should match expected output.")

  fail = .false.
  if (maxval(surf_ids) /= 6) fail = .true.
  call report_test("[surface ids]", fail, .false., "The maximal surface id should be 6!")

  call deallocate(state)

end subroutine test_node_lies_on_boundary
