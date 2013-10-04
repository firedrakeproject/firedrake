subroutine test_tet_intersector

  use read_triangle
  use tetrahedron_intersection_module
  use supermesh_construction
  use fields
  use unittest_tools
  implicit none

  type(vector_field) :: positionsA, positionsB
  type(vector_field) :: libwm, fort
  integer :: ele_A, ele_B, ele_C
  real :: vol_libwm, vol_fort
  logical :: fail
  integer :: stat
  type(tet_type) :: tet_A, tet_B
  type(plane_type), dimension(4) :: planes_B

  positionsA = read_triangle_files("data/plcA", quad_degree=4)
  positionsB = read_triangle_files("data/plcB", quad_degree=4)

  call intersector_set_dimension(3)
  call intersector_set_exactness(.false.)

  do ele_A=1,ele_count(positionsA)
    do ele_B=1,ele_count(positionsB)
      libwm = intersect_elements(positionsB, ele_B, ele_val(positionsA, ele_A), ele_shape(positionsB, 1))
      tet_A%v = ele_val(positionsA, ele_A)
      tet_B%v = ele_val(positionsB, ele_B)
      planes_B = get_planes(tet_B)
      call intersect_tets(tet_A, planes_B, shape=ele_shape(positionsB, 1), stat=stat, output=fort)

      fail = (ele_count(libwm) /= ele_count(fort))
!      call report_test("[tet_intersector counts]", fail, .false., "Should give the same number of elements")

      vol_libwm = 0.0
      do ele_C=1,ele_count(libwm)
        vol_libwm = vol_libwm + abs(simplex_volume(libwm, ele_C))
      end do
      vol_fort = 0.0
      if (stat == 0) then
        do ele_C=1,ele_count(fort)
          vol_fort = vol_fort + abs(simplex_volume(fort, ele_C))
        end do
      end if

      fail = (vol_libwm .fne. vol_fort)
      call report_test("[tet_intersector volumes]", fail, .false., "Should give the same volumes of intersection")

      call deallocate(libwm)
      if (stat == 0) then
        call deallocate(fort)
      end if
    end do
  end do
  call deallocate(positionsA)
  call deallocate(positionsB)

end subroutine test_tet_intersector
