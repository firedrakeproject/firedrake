/*  Copyright (C) 2006 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Prof. C Pain
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    amcgsoftware@imperial.ac.uk

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
*/

#ifndef ELEMENT_INTERSECTION_H
#define ELEMENT_INTERSECTION_H

#include "confdefs.h"

#include "Wm4Intersector.h"
#include "Wm4IntrTriangle2Triangle2.h"
#include "Wm4Triangle2.h"
#include "Wm4IntrQuad2Quad2.h"
#include "Wm4Quad2.h"
#include "Wm4IntrTetrahedron3Tetrahedron3.h"
#include "Wm4Tetrahedron3.h"
#include "Wm4Vector3.h"

#include <strings.h>
#include <spatialindex/SpatialIndex.h>

#ifdef HAVE_LIBCGAL
#include <CGAL/Cartesian.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Gmpzf.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Quotient.h>
#include <CGAL/Triangulation_2.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#endif

#include <cassert>
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include <set>

#ifndef DDEBUG
#ifdef assert
#undef assert
#endif
#define assert
#endif

#include "MeshDataStream.h"
#include "Precision.h"

#define GEOM_REAL double

namespace Fluidity
{
  // StorageManager parameters
  const int capacity = 10;
  const int writeThrough = false;

  // R-Tree parameters
  const SpatialIndex::RTree::RTreeVariant variant = SpatialIndex::RTree::RV_RSTAR;
  // Minimum fraction (of maximum) of entries in any node (index or leaf)
  const double fillFactor = 0.7;
  // Node index capacity in the rtree
  const unsigned long indexCapacity = 10;
  // Node leaf capacity in the rtree
  const unsigned long leafCapacity = 10;

  // Customised version of PyListVisitor class in
  // wrapper.cc in Rtree 0.4.1
  class ElementListVisitor : public SpatialIndex::IVisitor, public std::vector< int >
  {
    public:
      inline ElementListVisitor()
      {
        return;
      }

      inline virtual ~ElementListVisitor()
      {
        return;
      }

      inline virtual void visitNode(const SpatialIndex::INode& node)
      {
        return;
      }

      inline virtual void visitData(const SpatialIndex::IData& data)
      {
        push_back(data.getIdentifier());

        return;
      }

      inline virtual void visitData(std::vector< const SpatialIndex::IData* >& vector)
      {
        return;
      }
  };

  class InstrumentedRegion : public SpatialIndex::Region
  {
    public:
      InstrumentedRegion();
      InstrumentedRegion(const double* pLow, const double* pHigh, size_t dimension);
      InstrumentedRegion(const SpatialIndex::Point& low, const SpatialIndex::Point& high);
      InstrumentedRegion(const SpatialIndex::Region& in);

      virtual bool intersectsRegion(const Region& in);
      virtual bool containsRegion(const Region& in);
      virtual bool touchesRegion(const Region& in);
      virtual int getPredicateCount(void) const;
    private:
      int predicateCount;
  };

  // Interface to spatialindex to calculate element intersection lists between
  // meshes using bulk storage
  // Uses code from gispatialindex.{cc,h} in Rtree 0.4.1
  class ElementIntersectionFinder
  {
    public:
      ElementIntersectionFinder();
      ~ElementIntersectionFinder();

      int Reset();
      void SetInput(const double*& positions, const int& nnodes, const int& dim,
                    const int*& enlist, const int& nelements, const int& loc);
      void SetTestElement(const double*& positions, const int& dim, const int& loc);
      void QueryOutput(int& nelms) const;
      void GetOutput(int& id, const int& index) const;
    protected:
      void Initialise();
      void Free();

      int dim, loc;
      SpatialIndex::IStorageManager* storageManager;
      SpatialIndex::StorageManager::IBuffer* storage;
      SpatialIndex::ISpatialIndex* rTree;
      ElementListVisitor visitor;

      int predicateCount;
  };

  class ElementIntersector
  {
    public:
      virtual ~ElementIntersector();

      virtual unsigned int GetDim() const = 0;
      virtual unsigned int GetExactness() const;

      virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc);
      virtual void Intersect() = 0;
      virtual void QueryOutput(int& nnodes, int& nelms) const = 0;
      virtual void GetOutput(double*& positions, int*& enlist) const = 0;
    protected:
      ElementIntersector();

      double* positionsA;
      double* positionsB;
      int loc;
      int dim;
      int exactness;
  };

  class ElementIntersector1D : public ElementIntersector
  {
    public:
      ElementIntersector1D();
      virtual ~ElementIntersector1D();

      inline virtual unsigned int GetDim() const
      {
        return 1;
      }

      inline virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
      {
        assert(dim == 1);
        assert(loc == 2);
        ElementIntersector::SetInput(positionsA, positionsB, dim, loc);

        return;
      }

      virtual void Intersect();
      virtual void QueryOutput(int& nnodes, int& nelms) const;
      virtual void GetOutput(double*& positions, int*& enlist) const;
    protected:
      double positionsC[2];
      bool intersection;
  };

  class ElementIntersector2D : public ElementIntersector
  {
    public:
      ElementIntersector2D();
      virtual ~ElementIntersector2D();

      inline virtual unsigned int GetDim() const
      {
        return 2;
      }

      inline virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
      {
        assert(dim == 2);
        assert(loc == 3 || loc == 4);
        ElementIntersector::SetInput(positionsA, positionsB, dim, loc);

        return;
      }
      virtual void Intersect();
      virtual void QueryOutput(int& nnodes, int& nelms) const;
      virtual void GetOutput(double*& positions, int*& enlist) const;

      typedef Wm4::IntrTriangle2Triangle2<GEOM_REAL> IntrTriangle2Triangle2;
      typedef Wm4::Triangle2<GEOM_REAL> Triangle2;
      typedef Wm4::Vector2<GEOM_REAL> Vector2;
      typedef Wm4::IntrQuad2Quad2<GEOM_REAL> IntrQuad2Quad2;
      typedef Wm4::Quad2<GEOM_REAL> Quad2;
      typedef Wm4::Intersector<GEOM_REAL, Vector2> Intersector2d;

    protected:
       Intersector2d* intersection;
  };

  class ElementIntersectorCGAL2D : public ElementIntersector
  {
    public:
      ElementIntersectorCGAL2D();
      virtual ~ElementIntersectorCGAL2D();

      inline virtual unsigned int GetDim() const
      {
        return 2;
      }

      inline virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
      {
        assert(dim == 2);
        assert(loc == 3 || loc == 4);
        ElementIntersector::SetInput(positionsA, positionsB, dim, loc);

        return;
      }
      virtual void Intersect();
      virtual void QueryOutput(int& nnodes, int& nelms) const;
      virtual void GetOutput(double*& positions, int*& enlist) const;

#ifdef HAVE_LIBCGAL
      typedef CGAL::Lazy_exact_nt< CGAL::Quotient<CGAL::MP_Float> > NT;
      typedef CGAL::Cartesian< NT > Kernel;
      typedef Kernel::Point_2 Point_2;
      typedef Kernel::Triangle_2 Triangle_2;
      typedef Kernel::Segment_2 Segment_2;
      typedef CGAL::Polygon_2<Kernel> Polygon_2;
      typedef CGAL::Triangulation_2< Kernel > Triangulation;
      typedef Triangulation::Point_iterator Point_iterator;
      typedef Triangulation::Finite_faces_iterator Finite_faces_iterator;
#endif

    protected:
#ifdef HAVE_LIBCGAL
       Triangulation* triangulation;
#endif
  };

  class ElementIntersector3D : public ElementIntersector
  {
    public:
      inline ElementIntersector3D() {return;}
      inline virtual ~ElementIntersector3D() {return;}

      inline virtual unsigned int GetDim() const
      {
        return 3;
      }

      inline virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
      {
        assert(dim == 3);

        ElementIntersector::SetInput(positionsA, positionsB, dim, loc);

        return;
      }

      virtual void Intersect() = 0;
      virtual void QueryOutput(int& nnodes, int& nelms) const = 0;
      virtual void GetOutput(double*& positions, int*& enlist) const = 0;
  };

  class WmElementIntersector3D : public ElementIntersector3D
  {
    public:
      WmElementIntersector3D();
      virtual ~WmElementIntersector3D();

      inline virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
      {
        assert(loc == 4);
        ElementIntersector3D::SetInput(positionsA, positionsB, dim, loc);

        return;
      }

      virtual void Intersect();
      virtual void QueryOutput(int& nnodes, int& nelms) const;
      virtual void GetOutput(double*& positions, int*& enlist) const;

      typedef Wm4::IntrTetrahedron3Tetrahedron3<GEOM_REAL> IntrTetrahedron3Tetrahedron3;
      typedef Wm4::Tetrahedron3<GEOM_REAL> Tetrahedron3;
      typedef Wm4::Vector3<GEOM_REAL> Vector3;
    protected:
      IntrTetrahedron3Tetrahedron3* intersection;
      std::vector<GEOM_REAL>* volumes;
      int nodes, elements;
  };

  class ElementIntersectorCGAL3D : public ElementIntersector3D
  {
    public:
      ElementIntersectorCGAL3D();
      virtual ~ElementIntersectorCGAL3D();

      inline virtual unsigned int GetDim() const
      {
        return 3;
      }

      inline virtual void SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
      {
        assert(loc == 4);
        ElementIntersector3D::SetInput(positionsA, positionsB, dim, loc);

        return;
      }

      virtual void Intersect();
      virtual void QueryOutput(int& nnodes, int& nelms) const;
      virtual void GetOutput(double*& positions, int*& enlist) const;

#ifdef HAVE_LIBCGAL
      typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
      typedef CGAL::Triangulation_3< Kernel > Triangulation;
      typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
      typedef Kernel::Point_3 Point_3;
      typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron;
      typedef Triangulation::Point_iterator Point_iterator;
      typedef Triangulation::Finite_cells_iterator Finite_cells_iterator;
#endif

    protected:
#ifdef HAVE_LIBCGAL
       Triangulation* triangulation;
#endif
  };
}

extern Fluidity::ElementIntersector* elementIntersector;

extern Fluidity::ElementIntersectionFinder elementIntersectionFinder;

extern "C"
{
#define cIntersectorGetDimension F77_FUNC(cintersector_get_dimension, CINTERSECTOR_GET_DIMENSION)
  int cIntersectorGetDimension();

#define cIntersectorSetDimension F77_FUNC(cintersector_set_dimension, CINTERSECTOR_SET_DIMENSION)
  void cIntersectorSetDimension(const int* dim);

#define cIntersectorSetExactness F77_FUNC(cintersector_set_exactness, CINTERSECTOR_SET_EXACTNESS)
  void cIntersectorSetExactness(const int* exact);

#define cIntersectorSetInput F77_FUNC(cintersector_set_input, CINTERSECTOR_SET_INPUT)
  void cIntersectorSetInput(double* positionsA, double* positionsB, const int* dim, const int* loc);

#define cIntersectorDrive F77_FUNC(cintersector_drive, CINTERSECTOR_DRIVE)
  void cIntersectorDrive();

#define cIntersectorQuery F77_FUNC(cintersector_query, CINTERSECTOR_QUERY)
  void cIntersectorQuery(int* nnodes, int* nelms);

#define cIntersectorGetOutput F77_FUNC(cintersector_get_output, CINTERSECTOR_GET_OUTPUT)
  void cIntersectorGetOutput(const int* nnodes, const int* nelms, const int* dim, const int* loc, double* positions, int* enlist);

#define cIntersectionFinderReset F77_FUNC(cintersection_finder_reset, CINTERSECTION_FINDER_RESET)
  void cIntersectionFinderReset(int* ntests);

#define cIntersectionFinderSetInput F77_FUNC(cintersection_finder_set_input, CINTSERSECTION_FINDER_SET_INPUT)
  void cIntersectionFinderSetInput(const double* positions, const int* enlist, const int* dim, const int* loc, const int* nnodes, const int* nelements);

#define cIntersectionFinderFind F77_FUNC(cintersection_finder_find, CINTSERSECTION_FINDER_FIND)
  void cIntersectionFinderFind(const double* positions, const int* dim, const int* loc);

#define cIntersectionFinderQueryOutput F77_FUNC(cintersection_finder_query_output, CINTSERSECTION_FINDER_QUERY_OUTPUT)
  void cIntersectionFinderQueryOutput(int* nelms);

#define cIntersectionFinderGetOutput F77_FUNC(cintersection_finder_get_output, CINTSERSECTION_FINDER_GET_OUTPUT)
  void cIntersectionFinderGetOutput(int* id, const int* index);
}

#endif
