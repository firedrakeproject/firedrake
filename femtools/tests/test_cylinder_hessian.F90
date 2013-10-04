subroutine test_cylinder_hessian

  use fields
  use field_derivatives
  use state_module
  use vtk_interfaces
  use unittest_tools
  use node_boundary, only: pseudo2d_coord
  implicit none

  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: position_field
  type(scalar_field) :: pressure_field
  type(tensor_field) :: hessian
  logical :: fail = .false., warn = .false.
  integer :: i, j, k
  real :: x, y, z

  !pseudo2d_coord = 0

  call vtk_read_state("data/test_cyl.vtu", state)

  mesh => extract_mesh(state, "Mesh")
  call add_faces(mesh)
  position_field => extract_vector_field(state, "Coordinate")
  ! Update mesh descriptor on positons
  position_field%mesh=mesh

  call allocate(pressure_field, mesh, "Pressure")

  do i=1,mesh%nodes
    x = position_field%val(1,i)
    y = position_field%val(2,i)
    z = position_field%val(3,i)
    pressure_field%val(i) = x
  end do

  call allocate(hessian, mesh, "Hessian")
  hessian%val = -99999.9

  call compute_hessian(pressure_field, position_field, hessian)

  fail = .false.

  ! No element of val should be -99999.9
  if (any(hessian%val == -99999.9)) fail = .true.

  call report_test("[every value set]", fail, warn, "No element of val should be unwritten.")

  fail = .false.

  ! Every element of val should be 0.0
  do i=1,mesh%nodes
    do j=1,3
      do k=1,3
        if (abs(hessian%val(j, k, i)) > epsilon(0.0_4)) then
          print *, "i == ", i, "; j == ", j, "; k == ", k, "; val == ", hessian%val(j, k, i)
          fail = .true.
        end if
      end do
    end do
  end do

  call report_test("[every value correct]", fail, warn, "The Hessian of a linear field is identically zero.")
  call vtk_write_fields("data/cylinder_hessian", 0, position_field, mesh, sfields=(/pressure_field/), tfields=(/hessian/))

end subroutine test_cylinder_hessian
