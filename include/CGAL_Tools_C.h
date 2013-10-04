#ifndef CGAL_TOOLS_C_H
#define CGAL_TOOLS_C_H

#include "confdefs.h"

#ifdef HAVE_LIBCGAL
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K>          Delaunay_3;
typedef Delaunay_3::Point                             Point_3;
typedef CGAL::Triangle_3<K>                        Triangle_3;
typedef Delaunay_3::Cell_handle                 Cell_handle_3;

#endif

#ifndef DDEBUG
#ifdef assert
#undef assert
#endif
#define assert
#endif

#ifdef DOUBLEP
#define real double
#else
#define real float
#endif

#endif
