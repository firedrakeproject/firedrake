subroutine test_pseudo2d_hessian

  use vtk_interfaces
  use field_derivatives
  use unittest_tools
  use state_module
  use node_boundary, only: pseudo2d_coord
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: position_field
  type(scalar_field) :: pressure_field
  type(tensor_field) :: hessian
  real, dimension(3, 3) :: answer
  integer :: i
  logical :: fail
  real :: x, y, z

  pseudo2d_coord = 3

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  position_field => extract_vector_field(state, "Coordinate")
  call allocate(pressure_field, mesh, "Pressure")
  call allocate(hessian, mesh, "Hessian")

  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    pressure_field%val(i) = x * x
  end do

  call compute_hessian(pressure_field, position_field, hessian)
  call vtk_write_fields("data/pseudo2d_hessian", 0, position_field, mesh, sfields=(/pressure_field/), tfields=(/hessian/))

  answer = 0.0; answer(1, 1) = 2.0

  fail = .false.
  do i=1,mesh%nodes
    x = position_field%val(1,i)
    if (x <= 1.0 .or. x >= 29.0) cycle

    y = position_field%val(2,i)
    if (y <= 1.0 .or. y >= 14.0) cycle

    if (.not. fequals(hessian%val(1, 1, i), answer(1, 1), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(1, 1, i) == ", hessian%val(1, 1, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
  end do

  call report_test("[pseudo2d hessian x,x]", fail, .false., "The hessian of x^2 is not what it should be!")

  fail = .false.
  do i=1,mesh%nodes
    x = position_field%val(1,i)
    if (x <= 1.0 .or. x >= 29.0) cycle

    y = position_field%val(2,i)
    if (y <= 1.0 .or. y >= 14.0) cycle

    if (.not. fequals(hessian%val(1, 2, i), answer(1, 2), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(1, 2, i) == ", hessian%val(1, 2, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(1, 3, i), answer(1, 3), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(1, 3, i) == ", hessian%val(1, 3, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(2, 1, i), answer(2, 1), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(2, 1, i) == ", hessian%val(2, 1, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(2, 2, i), answer(2, 2), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(2, 2, i) == ", hessian%val(2, 2, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(2, 3, i), answer(2, 3), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(2, 3, i) == ", hessian%val(2, 3, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(3, 1, i), answer(3, 1), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(3, 1, i) == ", hessian%val(3, 1, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(3, 2, i), answer(3, 2), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(3, 2, i) == ", hessian%val(3, 2, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
    if (.not. fequals(hessian%val(3, 3, i), answer(3, 3), 0.15)) then
      write(0,*) "i == ", i, "; hessian%val(3, 3, i) == ", hessian%val(3, 3, i)
      write(0,*) "x == ", x, "; y == ", y
      fail = .true.
    end if
  end do

  call report_test("[pseudo2d hessian others]", fail, .false., "The hessian of x^2 is not what it should be!")

  call deallocate(hessian)
  call deallocate(pressure_field)
  call deallocate(state)

end subroutine test_pseudo2d_hessian
