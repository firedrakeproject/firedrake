subroutine test_differentiate_field_discontinuous
  ! unit test to test differentiate_field_discontinuous in field_derivatives
  ! computes linear discontuous gradient of pressure field from
  ! quadratic polynomial, so should give the exact answer
  use quadrature
  use fields
  use field_derivatives
  use state_module
  use vtk_interfaces
  use unittest_tools
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(mesh_type) qmesh, dgmesh
  type(vector_field), pointer :: position_field
  type(vector_field) qposition_field
  type(scalar_field) :: pressure_field
  type(scalar_field), dimension(3) :: outfields
  type(element_type) qshape
  logical, dimension(3) :: derivatives
  logical :: failx, faily, failz, warn
  integer :: i, ele
  integer, dimension(:), pointer:: cgnodes, dgnodes
  character(len=20) :: buf
  real :: x, y, z, xyz(3), derx, dery, derz

  call vtk_read_state("data/test_spr.vtu", state)

  mesh => extract_mesh(state, "Mesh")
  position_field => extract_vector_field(state, "Coordinate")
  ! quadratic shape and quadratic continuous mesh
  qshape=make_element_shape(4, 3, 2, mesh%shape%quadrature)
  qmesh=make_mesh(mesh, qshape, name="QuadraticMesh")
  ! linear discontinuous mesh
  dgmesh=make_mesh(mesh, continuity=-1, name="LinearDGMesh")

  call allocate(pressure_field, qmesh, "Pressure")
  call allocate(qposition_field, 3, qmesh, "QuadraticCoordinate")
  call remap_field(position_field, qposition_field)

  do i=1,3
    write(buf,'(i0)') i
    call allocate(outfields(i), dgmesh, "Derivative " // trim(buf))
  end do

  do i=1, node_count(qmesh)
    xyz=node_val(qposition_field, i)
    x = xyz(1)
    y = xyz(2)
    z = xyz(3)
    call set(pressure_field, i, (x+2*y+3*z+7)*(4*x+5*y+6*z+8))
  end do

  derivatives=.true. ! ask for all derivatives
  call differentiate_field(pressure_field, position_field, derivatives, outfields)
  call vtk_write_fields("data/differentiate_field", 0, position_field, dgmesh, sfields=outfields)
  failx=.false.
  faily=.false.
  failz=.false.

  do ele=1, ele_count(dgmesh)
    cgnodes => ele_nodes(mesh, ele)
    dgnodes => ele_nodes(dgmesh, ele)
    do i=1, ele_loc(dgmesh, ele)
      xyz=node_val(position_field, cgnodes(i))
      x = xyz(1)
      y = xyz(2)
      z = xyz(3)
      derx=node_val(outfields(1), dgnodes(i))
      if (.not. fequals(derx, 8*x+13*y+18*z+36, tol = 1.0e-10)) failx=.true.
      dery=node_val(outfields(2), dgnodes(i))
      if (.not. fequals(dery, 13*x+20*y+27*z+51, tol = 1.0e-10)) faily=.true.
      derz=node_val(outfields(3), dgnodes(i))
      if (.not. fequals(derz, 18*x+27*y+36*z+66, tol = 1.0e-10)) failz=.true.
    end do
  end do

  warn=.false.

  call report_test("[linear discontinous exact x derivative]", failx, warn, "X derivative is wrong")

  call report_test("[linear discontinous exact y derivative]", faily, warn, "Y derivative is wrong")

  call report_test("[linear discontinous exact z derivative]", failz, warn, "Z derivative is wrong")

end subroutine test_differentiate_field_discontinuous
