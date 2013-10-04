#include "CGAL_Tools_C.h"

extern "C"
{
  void F77_FUNC_(convex_hull_area_3d,CONVEX_HULL_AREA_3D)(real* nodes, int* nonods);
}

void F77_FUNC_(convex_hull_area_3d,CONVEX_HULL_AREA_3D)(real* nodes, int* nonods, real* area)
{
#ifdef HAVE_LIBCGAL
  /* We want to compute the surface area of the convex hull of a set of points.
     This is done by creating the Delaunay triangulation of the points,
     looping over the infinite cells, and computing the surface area of their
     finite facets. */

  /* See
     http://www.cgal.org/Manual/last/doc_html/cgal_manual/Triangulation_3_ref/Class_Triangulation_3.html
     and
     http://www.cgal.org/Manual/last/doc_html/cgal_manual/Kernel_23_ref/Class_Triangle_3.html */

  Delaunay_3 T;

  /* Insert everything into the Delaunay triangulation */
  int j = 0;
  for (int i = 0; i < *nonods; i++)
  {
    T.insert(Point_3(nodes[j], nodes[j+1], nodes[j+2]));
    j = j + 3;
  }

  /* Loop over all cells */
  *area = 0.0;
  for (Delaunay_3::Cell_iterator cell = T.all_cells_begin(); cell != T.all_cells_end(); cell++)
  {
    /* Skip the finite cells */
    if (!T.is_infinite(cell)) continue;
    for (int face = 0; face < 4; face++)
    {
      /* If we have a finite face of an infinite cell, we want to add its area. So ... */
      if (!T.is_infinite(cell, face))
      {
        Triangle_3 tri = T.triangle(cell, face);
        *area = *area + sqrt(tri.squared_area());
      }
    }
  }
#else
  *area = 0.0;
#endif
}
