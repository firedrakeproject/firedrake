subroutine test_differentiate_field

  use fields
  use field_derivatives
  use state_module
  use vtk_interfaces
  use unittest_tools

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: position_field
  type(scalar_field) :: pressure_field
  type(scalar_field), dimension(3) :: outfields
  logical, dimension(3) :: derivatives = .true.
  logical :: fail = .false., warn = .false.
  integer :: i
  character(len=20) :: buf
  real :: x, y, z

  call vtk_read_state("data/test_spr.vtu", state)

  mesh => extract_mesh(state, "Mesh")
  position_field => extract_vector_field(state, "Coordinate")
  call allocate(pressure_field, mesh, "Pressure")

  do i=1,3
    write(buf,'(i0)') i
    call allocate(outfields(i), mesh, "Derivative " // trim(buf))
  end do

  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    pressure_field%val(i) = 2.0 * x + 3.0 * y
  end do

  call differentiate_field(pressure_field, position_field, derivatives, outfields)
  call vtk_write_fields("data/differentiate_field", 0, position_field, mesh, sfields=(/pressure_field, &
                        outfields/))

  ! X derivative
  do i=1,mesh%nodes
    if (.not. fequals(outfields(1)%val(i), 2.0)) fail = .true.
  end do

  call report_test("[linear exact x derivative]", fail, warn, "X derivative should be constant 2.0")

  fail = .false.

  ! Y derivative
  do i=1,mesh%nodes
    if (.not. fequals(outfields(2)%val(i), 3.0)) fail = .true.
  end do

  call report_test("[linear exact y derivative]", fail, warn, "Y derivative should be constant 3.0")

  fail = .false.

  ! Z derivative
  do i=1,mesh%nodes
    if (.not. fequals(outfields(3)%val(i), 0.0)) fail = .true.
  end do

  call report_test("[linear exact z derivative]", fail, warn, "Z derivative should be constant 0.0")

  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    pressure_field%val(i) = 0.5 * x * x + 0.5 * y * y
  end do

  call differentiate_field(pressure_field, position_field, derivatives, outfields)
  call vtk_write_fields("data/differentiate_field", 1, position_field, mesh, sfields=(/pressure_field, &
                        outfields/))

  fail = .false.

  ! X derivative
  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    if (.not. fequals(outfields(1)%val(i), x, 0.15)) then
      print *," i == ", i, "; x == ", x, "; diffx == ", outfields(1)%val(i)
      fail = .true.
    end if
  end do

  call report_test("[quadratic exact x derivative]", fail, warn, "X derivative should be x")

  fail = .false.

  ! Y derivative
  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    if (.not. fequals(outfields(2)%val(i), y, 0.15)) then
      print *," i == ", i, "; y == ", y, "; diffy == ", outfields(2)%val(i)
      fail = .true.
    end if
  end do

  call report_test("[quadratic exact y derivative]", fail, warn, "Y derivative should be y")

  fail = .false.

  ! Z derivative
  do i=1,mesh%nodes
    if (abs(outfields(3)%val(i)) .gt. epsilon(0.0_4)) then
      print *," i == ", i, "; diffz == ", outfields(3)%val(i)
      fail = .true.
    end if
  end do

  call report_test("[quadratic exact z derivative]", fail, warn, "Z derivative should be 0.0")
end subroutine test_differentiate_field
