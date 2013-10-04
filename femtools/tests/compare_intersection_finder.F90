#include "fdebug.h"

subroutine compare_intersection_finder
  !!< Compares intersection finder algorithms. For intersection finding paper
  !!< results.

  use fields
  use global_parameters, only : debug_error_unit
  use intersection_finder_module
  use linked_lists
  use read_triangle

  implicit none

  character(len = 255) :: base1, base2
  integer :: i, intersections, j, k
  real :: end_cpu_time, start_cpu_time
  type(ilist), dimension(:), allocatable :: map_ab, map_ab_af
  type(vector_field) :: mesh_field_a, mesh_field_b

  do i = 8, 1, -1
    base1 = int2str(i) // ".1"
    base2 = int2str(i) // ".2"

    ewrite(0, *), "################"
    ewrite(0, *), "### New base ###"
    ewrite(0, *), "################"
    ewrite(0, *), "Base1 = " // trim(base1)
    ewrite(0, *), "Base2 = " // trim(base2)

    ! Load in the mesh fields
    mesh_field_a = read_triangle_files("data/" // trim(base1), quad_degree = 1)
    mesh_field_b = read_triangle_files("data/" // trim(base2), quad_degree = 1)

    allocate(map_ab_af(ele_count(mesh_field_a)))
    allocate(map_ab(ele_count(mesh_field_a)))

!    do j = 1, 3
    do j = 1, 1

      ! Advancing front
      call reset_intersection_tests_counter()
!      call cpu_time(start_cpu_time)
      map_ab_af = advancing_front_intersection_finder(mesh_field_a, mesh_field_b)
!      call cpu_time(end_cpu_time)
!      ewrite(0, *), "Advancing front, loop " // int2str(j) // ", CPU time: ", end_cpu_time - start_cpu_time
      intersections = 0
      do k = 1, size(map_ab_af)
        intersections = intersections + map_ab_af(k)%length
      end do
      ewrite(0, *) "Advancing front, loop " // int2str(j) // ", intersections: " // int2str(intersections)
      ewrite(0, *) "Advancing front, loop " // int2str(j) // ", intersection tests: " // int2str(intersection_tests())

      call flush(debug_error_unit)

!      ! Rtree
!     call cpu_time(start_cpu_time)
     map_ab = rtree_intersection_finder(mesh_field_a, mesh_field_b)
!     call cpu_time(end_cpu_time)
!     ewrite(0, *), "Rtree, loop " // int2str(j) // ", CPU time: ", end_cpu_time - start_cpu_time
     intersections = 0
     do k = 1, size(map_ab)
        intersections = intersections + map_ab(k)%length
      end do
      ewrite(0, *) "Rtree, loop " // int2str(j) // ", intersections: " // int2str(intersections)
      if(j == 1) then
        call verify_map(mesh_field_a, mesh_field_b, map_ab_af, map_ab)
        ewrite(0, *), "Advancing front map verified against rtree"
      end if
      call flush_lists(map_ab)

      call flush(debug_error_unit)

!      ! Brute force
      call reset_intersection_tests_counter()
!      call cpu_time(start_cpu_time)
      map_ab = brute_force_intersection_finder(mesh_field_a, mesh_field_b)
!      call cpu_time(end_cpu_time)
      ewrite(0, *), "Brute force, loop " // int2str(j) // ", CPU time: ", end_cpu_time - start_cpu_time
      intersections = 0
      do k = 1, size(map_ab)
        intersections = intersections + map_ab(k)%length
      end do
      if(j == 1) then
        call verify_map(mesh_field_a, mesh_field_b, map_ab_af, map_ab)
        ewrite(0, *), "Advancing front map verified against brute force"
      end if
      ewrite(0, *) "Brute force, loop " // int2str(j) // ", intersections: " // int2str(intersections)
      ewrite(0, *) "Brute force, loop " // int2str(j) // ", intersection tests: " // int2str(intersection_tests())
      call flush_lists(map_ab)

      call flush(debug_error_unit)

      call flush_lists(map_ab_af)
    end do

    ! Deallocate
    call deallocate(mesh_field_a)
    call deallocate(mesh_field_b)

    deallocate(map_ab_af)
    deallocate(map_ab)

  end do

end subroutine compare_intersection_finder
