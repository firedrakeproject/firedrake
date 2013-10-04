subroutine test_constant_fields

  use fields
  use vtk_interfaces
  use state_module
  use unittest_tools
  implicit none

  type(scalar_field) :: sfield
  type(vector_field) :: vfield
  type(tensor_field) :: tfield
  type(state_type) :: state
  type(mesh_type), pointer :: mesh
  type(vector_field), pointer :: positions
  logical :: fail
  real, dimension(4) :: ele_s
  real, dimension(3, 4) :: ele_v
  real, dimension(3, 3, 4) :: ele_t
  real :: node_s
  real, dimension(3) :: node_v
  real, dimension(3, 3) :: node_t
  integer :: i, j, k, l

  call vtk_read_state("data/pseudo2d.vtu", state)
  mesh => extract_mesh(state, "Mesh")
  positions => extract_vector_field(state, "Coordinate")

  call allocate(sfield, mesh, "ScalarField", FIELD_TYPE_CONSTANT)
  call set(sfield, 1.0)

  fail = .false.
  if (node_count(sfield) /= node_count(mesh)) then
    fail = .true.
  end if
  call report_test("[constant scalar fields]", fail, .false., "Constant fields have the same number of nodes.")

  fail = .false.
  do i=1,node_count(sfield)
    node_s = node_val(sfield, i)
    if (node_s /= 1.0) then
      fail = .true.
    end if
  end do
  call report_test("[constant scalar fields]", fail, .false., "Constant fields should return the value you give them.")

  fail = .false.
  do i=1,ele_count(sfield)
    ele_s = ele_val(sfield, i)
    if (any(ele_s /= 1.0)) then
      fail = .true.
    end if
  end do
  call report_test("[constant scalar fields]", fail, .false., "Constant fields should return the value you give them.")

  fail = .false.
  if (size(sfield%val) /= 1) then
    fail = .true.
    write(0,*) "size(sfield%val) == ", size(sfield%val)
  end if
  call report_test("[constant scalar fields]", fail, .false., "Constant fields shouldn't allocate more than they need to.")

  call deallocate(sfield)
  call allocate(vfield, 3, mesh, "VectorField", FIELD_TYPE_CONSTANT)
  call set(vfield, (/1.0, 2.0, 3.0/))

  fail = .false.
  if (node_count(vfield) /= node_count(mesh)) then
    fail = .true.
  end if
  call report_test("[constant vector fields]", fail, .false., "Constant fields have the same number of nodes.")

  fail = .false.
  do i=1,node_count(vfield)
    node_v = node_val(vfield, i)
    if (any(node_v /= (/1.0, 2.0, 3.0/))) then
      fail = .true.
    end if
  end do
  call report_test("[constant vector fields]", fail, .false., "Constant fields should return the value you give them.")

  fail = .false.
  do i=1,ele_count(vfield)
    ele_v = ele_val(vfield, i)
    do j=1,ele_loc(mesh, 1)
      do k=1,3
        if (ele_v(k, j) /= float(k)) then
          fail = .true.
        end if
      end do
    end do
  end do
  call report_test("[constant vector fields]", fail, .false., "Constant fields should return the value you give them.")

  fail = .false.
  if (size(vfield%val(1,:)) /= 1) then
    fail = .true.
  end if
  call report_test("[constant vector fields]", fail, .false., "Constant fields shouldn't allocate more than they need to.")

  call deallocate(vfield)
  call allocate(tfield, mesh, "TensorField", FIELD_TYPE_CONSTANT)
  node_t(1, :) = (/1.0, 2.0, 3.0/)
  node_t(2, :) = (/4.0, 5.0, 6.0/)
  node_t(3, :) = (/7.0, 8.0, 9.0/)
  call set(tfield, node_t)

  fail = .false.
  if (node_count(tfield) /= node_count(mesh)) then
    fail = .true.
  end if
  call report_test("[constant tensor fields]", fail, .false., "Constant fields have the same number of nodes.")

  fail = .false.
  do i=1,node_count(tfield)
    node_t = node_val(tfield, i)
    do j=1,3
      do k=1,3
        if (node_t(j, k) /= 3.0 * (j-1) + k) then
          fail = .true.
        end if
      end do
    end do
  end do
  call report_test("[constant tensor fields]", fail, .false., "Constant fields should return the value you give them.")

  fail = .false.
  do i=1,ele_count(tfield)
    ele_t = ele_val(tfield, i)
    do j=1,ele_loc(mesh, 1)
      do k=1,3
        do l=1,3
          if (ele_t(k, l, j) /= 3.0 * (k-1) + l) then
            fail = .true.
          end if
        end do
      end do
    end do
  end do
  call report_test("[constant tensor fields]", fail, .false., "Constant fields should return the value you give them.")

  fail = .false.
  if (size(tfield%val, 3) /= 1) then
    fail = .true.
  end if
  call report_test("[constant tensor fields]", fail, .false., "Constant fields shouldn't allocate more than they need to.")

  !call vtk_write_fields("data/const_field", 0, positions, mesh, sfields=(/sfield/))
end subroutine test_constant_fields
