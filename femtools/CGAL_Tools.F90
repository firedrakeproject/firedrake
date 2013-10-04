#include "fdebug.h"
module cgal_tools

  use fields
  implicit none

  interface
    subroutine convex_hull_area_3d(nodes, sz, area)
      integer, intent(in) :: sz
      real, dimension(sz*3), intent(in) :: nodes
      real, intent(out) :: area
    end subroutine convex_hull_area_3d
  end interface

  public :: convex_hull_area

  contains

  function convex_hull_area(positions) result(area)
    type(vector_field), intent(in) :: positions
    real :: area
#ifndef HAVE_LIBCGAL
    FLAbort("Called a routine which depends on CGAL without CGAL")
    area = 0.0 * positions%dim
#else
    real, dimension(positions%dim * node_count(positions)) :: nodes
    integer :: head, i

    if (positions%dim /= 3) then
      FLExit("Sorry, have only written the CGAL for 3D. But the 2D generalisation is easy!")
    end if

    if (positions%mesh%shape%degree /= 1 .or. cell_family(positions%mesh%shape) /= FAMILY_SIMPLEX) then
      FLExit("Sorry, only have CGAL support for linear tets")
    end if

    head = 1
    do i=1,node_count(positions)
      nodes(head:head+positions%dim-1) = node_val(positions, i)
      head = head + positions%dim
    end do

    call convex_hull_area_3d(nodes, node_count(positions), area)
#endif
  end function convex_hull_area
end module cgal_tools
