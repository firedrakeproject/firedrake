subroutine test_compute_hessian

  use field_derivatives
  use state_module
  use vtk_interfaces
  use unittest_tools

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: position_field
  type(scalar_field) :: pressure_field
  type(tensor_field) :: hessian
  logical :: fail = .false., warn = .false.
  integer :: i
  real :: x, y, z

  call vtk_read_state("data/test_spr.vtu", state)

  mesh => extract_mesh(state, "Mesh")
  position_field => extract_vector_field(state, "Coordinate")
  call allocate(pressure_field, mesh, "Pressure")

  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    pressure_field%val(i) = 0.5 * x * x + 0.5 * y * y
    !pressure_field%val(i) = x
  end do

  call allocate(hessian, mesh, "Hessian")

  call compute_hessian(pressure_field, position_field, hessian)

  fail = .false.

  ! X,X derivative should be 1, +/- 0.3
  do i=1,mesh%nodes
    if (.not. fequals(hessian%val(1, 1, i), 1.0, 0.3)) fail = .true.
  end do

  call report_test("[cube x, x component]", fail, warn, "X, X component should be 1.0")

  fail = .false.

  ! Y,Y derivative should be 1, +/- 0.3
  do i=1,mesh%nodes
    if (.not. fequals(hessian%val(2, 2, i), 1.0, 0.3)) fail = .true.
  end do

  call report_test("[cube y, y component]", fail, warn, "Y, Y component should be 1.0")

  fail = .false.

  ! Z,Z derivative should be 0.0
  do i=1,mesh%nodes
    if (.not. fequals(hessian%val(3, 3, i), 0.0)) fail = .true.
  end do

  call report_test("[cube z, z component]", fail, warn, "Z, Z component should be 0.0")

  fail = .false.

  ! X,Y derivative should be 0.0, +/- 0.20
  do i=1,mesh%nodes
    if (.not. fequals(hessian%val(1, 2, i), 0.0, 0.20)) then
      print *, "i == ", i, "; hessian(1, 2) == ", hessian%val(1, 2, i)
      fail = .true.
    end if
  end do

  call report_test("[cube x, y component]", fail, warn, "X, Y component should be 0.0")

  call vtk_write_fields("data/compute_hessian", 0, position_field, mesh, sfields=(/pressure_field/), &
                        tfields=(/hessian/))

  call deallocate(pressure_field)
  call deallocate(state)
  call deallocate(hessian)
end subroutine test_compute_hessian
