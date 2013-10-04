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

#include "Element_Intersection.h"

using namespace SpatialIndex;

using namespace std;

using namespace Fluidity;

InstrumentedRegion::InstrumentedRegion()
{
  predicateCount = 0;
}

InstrumentedRegion::InstrumentedRegion(const double* pLow, const double* pHigh, size_t dimension)
{
  predicateCount = 0;
}

InstrumentedRegion::InstrumentedRegion(const Point& low, const Point& high)
{
  predicateCount = 0;
}

InstrumentedRegion::InstrumentedRegion(const Region& in)
{
  predicateCount = 0;
}

bool InstrumentedRegion::intersectsRegion(const Region& in)
{
  predicateCount++;
  return ((Region*)this)->intersectsRegion(in);
}

bool InstrumentedRegion::containsRegion(const Region& in)
{
  predicateCount++;
  return ((Region*)this)->containsRegion(in);
}

bool InstrumentedRegion::touchesRegion(const Region& in)
{
  predicateCount++;
  return ((Region*)this)->touchesRegion(in);
}

int InstrumentedRegion::getPredicateCount(void) const
{
  return predicateCount;
}

MeshDataStream::MeshDataStream(const double*& positions, const int& nnodes, const int& dim,
                               const int*& enlist, const int& nelements, const int& loc)
{
  assert(positions);
  assert(nnodes >= 0);
  assert(dim >= 0);
  assert(enlist);
  assert(nelements >= 0);
#ifdef DDEBUG
  if(nelements > 0)
  {
    assert(loc >= 1);
  }
  for(int i = 0;i < nelements*loc;i++)
  {
    assert(enlist[i] <= nnodes);
  }
#endif

  this->positions = positions;
  this->nnodes = nnodes;
  this->dim = dim;
  this->enlist = enlist;
  this->nelements = nelements;
  this->loc = loc;

  index = 0;
  predicateCount = 0;
}

MeshDataStream::~MeshDataStream()
{
}

IData* MeshDataStream::getNext()
{
  if(index >= nelements)
  {
    return NULL;
  }

  int node;

  double high[dim], low[dim];

  node = enlist[loc * index] - 1;

  for(int i = 0;i < dim;i++)
  {
    high[i] = positions[dim * node + i];
    low[i] = positions[dim * node + i];
  }
  for(int i = 1; i < loc; i++)
  {
    node = enlist[loc * index + i] - 1;
    for(int j = 0;j < dim;j++)
    {
      if(positions[dim * node + j] > high[j])
      {
        high[j] = positions[dim * node + j];
      }
      else if(positions[dim * node + j] < low[j])
      {
        low[j] = positions[dim * node + j];
      }
    }
  }

  SpatialIndex::Region region = SpatialIndex::Region(low, high, dim);
  IData* data = new RTree::Data(0, 0, region, ++index);

  return data;
}

bool MeshDataStream::hasNext()
{
  return index < nelements;
}

uint32_t MeshDataStream::size()
{
  return nelements;
}

void MeshDataStream::rewind()
{
  index = 0;

  return;
}

int MeshDataStream::getPredicateCount()
{
  return predicateCount;
}

bool MeshDataStream::MDSInstrumentedRegion::intersectsRegion(const Region& in)
{
  mds->predicateCount++;
  return ((Region*)this)->intersectsRegion(in);
}

bool MeshDataStream::MDSInstrumentedRegion::containsRegion(const Region& in)
{
  mds->predicateCount++;
  return ((Region*)this)->containsRegion(in);
}

bool MeshDataStream::MDSInstrumentedRegion::touchesRegion(const Region& in)
{
  mds->predicateCount++;
  return ((Region*)this)->touchesRegion(in);
}

IData* ExpandedMeshDataStream::getNext()
{
  if(index >= nelements)
  {
    return NULL;
  }

  int node;

  double high[dim], low[dim];

  node = enlist[loc * index] - 1;

  for(int i = 0;i < dim;i++)
  {
    high[i] = positions[dim * node + i];
    low[i] = positions[dim * node + i];
  }
  for(int i = 1; i < loc; i++)
  {
    node = enlist[loc * index + i] - 1;
    for(int j = 0;j < dim;j++)
    {
      if(positions[dim * node + j] > high[j])
      {
        high[j] = positions[dim * node + j];
      }
      else if(positions[dim * node + j] < low[j])
      {
        low[j] = positions[dim * node + j];
      }
    }
  }

  for(int i = 0;i < dim;i++)
  {
    assert(high[i] > low[i]);
    double expansion = max((high[i] - low[i]) * expansionFactor, 100.0 * numeric_limits<flfloat_t>::epsilon());
    high[i] += expansion;
    low[i] -= expansion;
  }

  SpatialIndex::Region region = SpatialIndex::Region(low, high, dim);
  IData* data = new RTree::Data(0, 0, region, ++index);

  return data;
}

ElementIntersectionFinder::ElementIntersectionFinder()
{
  Initialise();

  return;
}

ElementIntersectionFinder::~ElementIntersectionFinder()
{
  Free();

  return;
}

int ElementIntersectionFinder::Reset()
{
  int ntests = predicateCount;
  Free();
  Initialise();

  return ntests;
}

void ElementIntersectionFinder::SetInput(const double*& positions, const int& nnodes, const int& dim,
                                         const int*& enlist, const int& nelements, const int& loc)
{
  assert(positions);
  assert(enlist);
  assert(nnodes >= 0);
  assert(dim >= 0);
  assert(nelements >= 0);
  assert(loc >= 0);

  Reset();

  this->dim = dim;
  this->loc = loc;

  MeshDataStream stream(positions, nnodes, dim,
                        enlist, nelements, loc);
  // As in regressiontest/rtree/RTreeBulkLoad.cc in spatialindex 1.2.0
  id_type id = 1;
  rTree = RTree::createAndBulkLoadNewRTree(RTree::BLM_STR, stream, *storageManager, fillFactor, indexCapacity, leafCapacity, dim, SpatialIndex::RTree::RV_RSTAR, id);

  predicateCount += stream.getPredicateCount();

  return;
}

void ElementIntersectionFinder::SetTestElement(const double*& positions, const int& dim, const int& loc)
{
  assert(positions);
  assert(dim == this->dim);

  visitor.clear();

  double high[dim], low[dim];
  for(int i = 0;i < dim;i++)
  {
    high[i] = positions[i];
    low[i] = positions[i];
  }
  for(int i = 0;i < loc;i++)
  {
    for(int j = 0;j < dim;j++)
    {
      if(positions[dim * i + j] > high[j])
      {
        high[j] = positions[dim * i + j];
      }
      else if(positions[dim * i + j] < low[j])
      {
        low[j] = positions[dim * i + j];
      }
    }
  }

  SpatialIndex::Region* region = new SpatialIndex::Region(low, high, dim);
  rTree->intersectsWithQuery(*region, visitor);

  delete region;

  return;
}

void ElementIntersectionFinder::QueryOutput(int& nelms) const
{
  nelms = visitor.size();

  return;
}

void ElementIntersectionFinder::GetOutput(int& id, const int& index) const
{
  assert(index > 0);
  assert(index <= (int) visitor.size());

  id = visitor[index-1];

  return;
}

void ElementIntersectionFinder::Initialise()
{
  storageManager = StorageManager::createNewMemoryStorageManager();
  storage = StorageManager::createNewRandomEvictionsBuffer(*storageManager, capacity, writeThrough);
  rTree = NULL;

  dim = 0;
  loc = 0;

  predicateCount = 0;
}

void ElementIntersectionFinder::Free()
{
  if(rTree)
  {
    delete rTree;
    rTree = NULL;
  }
  delete storage;
  delete storageManager;

  visitor.clear();

  return;
}

ElementIntersector::ElementIntersector()
{
  positionsA = NULL;
  positionsB = NULL;

  loc = 0;

  return;
}

ElementIntersector::~ElementIntersector()
{
  if(this->positionsA)
  {
    free(this->positionsA);
    this->positionsA = NULL;
  }
  if(this->positionsB)
  {
    free(this->positionsB);
    this->positionsB = NULL;
  }

  return;
}

unsigned int ElementIntersector::GetExactness() const
{
  return exactness;
}

void ElementIntersector::SetInput(double*& positionsA, double*& positionsB, const int& dim, const int& loc)
{
  assert(positionsA);
  assert(positionsB);

  this->dim = dim;
  this->loc = loc;

  assert(dim >= 0);
  assert(loc >= 0);

  if(this->positionsA)
  {
    free(this->positionsA);
    this->positionsA = NULL;
  }
  this->positionsA = (double*)malloc(loc * dim * sizeof(double));
  assert(this->positionsA);
  memcpy(this->positionsA, positionsA, loc * dim * sizeof(double));

  if(this->positionsB)
  {
    free(this->positionsB);
    this->positionsB = NULL;
  }
  this->positionsB = (double*)malloc(loc * dim * sizeof(double));
  assert(positionsB);
  memcpy(this->positionsB, positionsB, loc * dim * sizeof(double));

  this->loc = loc;

  return;
}

ElementIntersectorCGAL2D::ElementIntersectorCGAL2D()
{
#ifdef HAVE_LIBCGAL
  triangulation = NULL;
#endif
  exactness = 1;
}

ElementIntersectorCGAL2D::~ElementIntersectorCGAL2D()
{
#ifdef HAVE_LIBCGAL
  if(triangulation)
  {
    delete triangulation;
    triangulation = NULL;
  }
#endif
}

void ElementIntersectorCGAL2D::Intersect()
{
#ifdef HAVE_LIBCGAL
  assert(positionsA);
  assert(positionsB);

  Point_2 pointA1(positionsA[0], positionsA[1]);
  Point_2 pointA2(positionsA[2], positionsA[3]);
  Point_2 pointA3(positionsA[4], positionsA[5]);
  Triangle_2 triA = Triangle_2(pointA1, pointA2, pointA3);

  Point_2 pointB1(positionsB[0], positionsB[1]);
  Point_2 pointB2(positionsB[2], positionsB[3]);
  Point_2 pointB3(positionsB[4], positionsB[5]);
  Triangle_2 triB = Triangle_2(pointB1, pointB2, pointB3);

  assert(!triA.is_degenerate());
  assert(!triB.is_degenerate());

  CGAL::Object result;
  Point_2 point;
  Segment_2 segment;
  Triangle_2 tri;
  vector<Point_2> ptlist;

  if(triangulation)
  {
    delete triangulation;
    triangulation = NULL;
  }
  triangulation = new Triangulation();

  result = CGAL::intersection(triA, triB);
  if(CGAL::assign(tri, result))
  {
    for (int i = 0; i < 3; i++)
    {
      triangulation->push_back(tri[i]);
    }
  }
  else if ((CGAL::assign(ptlist, result)))
  {
    for(vector< Point_2 >::iterator iter = ptlist.begin();iter != ptlist.end();iter++)
    {
      triangulation->push_back(*iter);
    }
  }

#else
  cerr << "Cannot compute 2D element intersections without CGAL" << endl;
  exit(-1);
#endif

}

void ElementIntersectorCGAL2D::QueryOutput(int& nnodes, int& nelms) const
{
#ifdef HAVE_LIBCGAL
  assert(triangulation);
  nnodes = triangulation->number_of_vertices();
  nelms = triangulation->number_of_faces();
#else
  cerr << "Cannot compute 2D element intersections without CGAL" << endl;
  exit(-1);
#endif

  return;
}

void ElementIntersectorCGAL2D::GetOutput(double*& positions, int*& enlist) const
{
#ifdef HAVE_LIBCGAL
  assert(triangulation);

  map<Point_2, size_t> coordToId;
  size_t index = 0;
  int nnodes = triangulation->number_of_vertices();
  for(Point_iterator iter = triangulation->points_begin();iter != triangulation->points_end();iter++, index++)
  {
    positions[index] = CGAL::to_double(iter->x());
    positions[index + nnodes] = CGAL::to_double(iter->y());
    coordToId[*iter] = index + 1;
  }

  index = 0;
  for(Finite_faces_iterator iter = triangulation->faces_begin();iter != triangulation->faces_end();iter++)
  {
    enlist[index++] = coordToId[iter->vertex(0)->point()];
    enlist[index++] = coordToId[iter->vertex(1)->point()];
    enlist[index++] = coordToId[iter->vertex(2)->point()];
  }
#else
  cerr << "Cannot compute 2D element intersections without CGAL" << endl;
  exit(-1);
#endif

  return;
}

ElementIntersectorCGAL3D::ElementIntersectorCGAL3D()
{
#ifdef HAVE_LIBCGAL
  triangulation = NULL;
#endif
  exactness = 1;
}

ElementIntersectorCGAL3D::~ElementIntersectorCGAL3D()
{
#ifdef HAVE_LIBCGAL
  if(triangulation)
  {
    delete triangulation;
    triangulation = NULL;
  }
#endif
}

void ElementIntersectorCGAL3D::Intersect()
{
#ifdef HAVE_LIBCGAL
  assert(positionsA);
  assert(positionsB);

  Polyhedron tetA, tetB;
  tetA.make_tetrahedron(
    Point_3(positionsA[0], positionsA[1], positionsA[2]),
    Point_3(positionsA[3], positionsA[4], positionsA[5]),
    Point_3(positionsA[6], positionsA[7], positionsA[8]),
    Point_3(positionsA[9], positionsA[10], positionsA[11]));
  tetB.make_tetrahedron(
    Point_3(positionsB[0], positionsB[1], positionsB[2]),
    Point_3(positionsB[3], positionsB[4], positionsB[5]),
    Point_3(positionsB[6], positionsB[7], positionsB[8]),
    Point_3(positionsB[9], positionsB[10], positionsB[11]));

  Nef_polyhedron NA(tetA);
  Nef_polyhedron NB(tetB);

  Nef_polyhedron intersection(NA * NB);

  if(triangulation)
  {
    delete triangulation;
    triangulation = NULL;
  }

  if (intersection.is_empty() || !intersection.is_simple())
  {
    triangulation = new Triangulation();
  }
  else
  {
    Polyhedron pIntersection;
    intersection.convert_to_Polyhedron(pIntersection);

    triangulation = new Triangulation(pIntersection.points_begin(), pIntersection.points_end());
  }
#else
  cerr << "Cannot compute element intersections without CGAL" << endl;
  exit(-1);
#endif
}

void ElementIntersectorCGAL3D::QueryOutput(int& nnodes, int& nelms) const
{
#ifdef HAVE_LIBCGAL
  assert(triangulation);

  nnodes = triangulation->number_of_vertices();
  nelms = triangulation->number_of_finite_cells();

#else
  cerr << "Cannot compute 3D element intersections without CGAL" << endl;
  exit(-1);
#endif

  return;
}

void ElementIntersectorCGAL3D::GetOutput(double*& positions, int*& enlist) const
{
#ifdef HAVE_LIBCGAL
  assert(triangulation);

  map<Point_3, size_t> coordToId;
  size_t index = 0;
  int nnodes = triangulation->number_of_vertices();
  for(Point_iterator iter = triangulation->points_begin();iter != triangulation->points_end();iter++, index++)
  {
    positions[index] = CGAL::to_double(iter->x());
    positions[index + nnodes] = CGAL::to_double(iter->y());
    positions[index + 2 * nnodes] = CGAL::to_double(iter->z());
    coordToId[*iter] = index + 1;
  }

  index = 0;
  for(Finite_cells_iterator iter = triangulation->finite_cells_begin();iter != triangulation->finite_cells_end();iter++)
  {
    enlist[index++] = coordToId[iter->vertex(0)->point()];
    enlist[index++] = coordToId[iter->vertex(1)->point()];
    enlist[index++] = coordToId[iter->vertex(2)->point()];
    enlist[index++] = coordToId[iter->vertex(3)->point()];
  }
#else
  cerr << "Cannot compute element intersections without CGAL" << endl;
  exit(-1);
#endif

  return;
}

ElementIntersector1D::ElementIntersector1D()
{
  exactness = 0;

  intersection = false;

  return;
}

ElementIntersector1D::~ElementIntersector1D()
{
  return;
}

void ElementIntersector1D::Intersect()
{
  assert(positionsA);
  assert(positionsB);

  double rightA = max(positionsA[0], positionsA[1]);
  double leftB = min(positionsB[0], positionsB[1]);

  if(leftB > rightA)
  {
    intersection = false;
    return;
  }

  intersection = true;
  double leftA = min(positionsA[0], positionsA[1]);
  double rightB = max(positionsB[0], positionsB[1]);

  positionsC[0] = max(leftA, leftB);
  positionsC[1] = min(rightA, rightB);

  return;
}

void ElementIntersector1D::QueryOutput(int& nnodes, int& nelms) const
{
  if(intersection)
  {
    nnodes = 2;
    nelms = 1;
  }
  else
  {
    nnodes = 0;
    nelms = 0;
  }

  return;
}

void ElementIntersector1D::GetOutput(double*& positions, int*& enlist) const{
  if(intersection)
  {
    positions[0] = positionsC[0];
    positions[1] = positionsC[1];
    enlist[0] = 1;
    enlist[1] = 2;
  }

  return;
}

ElementIntersector2D::ElementIntersector2D()
{
  intersection = NULL;
  exactness = 0;

  return;
}
ElementIntersector2D::~ElementIntersector2D()
{
  if(intersection)
  {
    delete intersection;
    intersection = NULL;
  }

  return;
}

void ElementIntersector2D::Intersect()
{
  assert(positionsA);
  assert(positionsB);

  if(intersection)
  {
    delete intersection;
    intersection = NULL;
  }

  if (loc == 3)
  {
    GEOM_REAL vec[2];
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[0 + i];
    Vector2 pointA1(vec);

    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[2 + i];
    Vector2 pointA2(vec);

    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[4 + i];
    Vector2 pointA3(vec);

    Wm4::Triangle2<GEOM_REAL> triA = Triangle2(pointA1, pointA2, pointA3);
    triA.Orient();

    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[0 + i];
    Vector2 pointB1(vec);

    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[2 + i];
    Vector2 pointB2(vec);

    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[4 + i];
    Vector2 pointB3(vec);

    Wm4::Triangle2<GEOM_REAL> triB = Triangle2(pointB1, pointB2, pointB3);
    triB.Orient();

    intersection = new Wm4::IntrTriangle2Triangle2<GEOM_REAL>(triA, triB);
    intersection->Find();
  }
  else if (loc == 4)
  {
    GEOM_REAL vec[2];
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[0 + i];
    Vector2 pointA1(vec);
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[2 + i];
    Vector2 pointA2(vec);
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[6 + i];
    Vector2 pointA3(vec);
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsA[4 + i];
    Vector2 pointA4(vec);
    Quad2 quadA = Quad2(pointA1, pointA2, pointA3, pointA4);

    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[0 + i];
    Vector2 pointB1(vec);
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[2 + i];
    Vector2 pointB2(vec);
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[6 + i];
    Vector2 pointB3(vec);
    for (int i = 0; i < 2; i++)
      vec[i] = (GEOM_REAL) positionsB[4 + i];
    Vector2 pointB4(vec);
    Quad2 quadB = Quad2(pointB1, pointB2, pointB3, pointB4);

    intersection = new IntrQuad2Quad2(quadA, quadB);
    intersection->Find();
  }
  else
  {
    cerr << "Invalid number of element nodes" << endl;
    exit(-1);
  }
}

void ElementIntersector2D::QueryOutput(int& nnodes, int& nelms) const
{
  assert(intersection);
  nnodes = intersection->GetQuantity();
  if (nnodes > 2)
    nelms = nnodes - 2;
  else
    nelms = 0;

  return;
}

void ElementIntersector2D::GetOutput(double*& positions, int*& enlist) const
{
  assert(intersection);

  int nnodes = intersection->GetQuantity();
  for (int node = 0; node < nnodes; node++)
  {
    Vector2 V = intersection->GetPoint(node);
    positions[node] = V[0];
    positions[node + nnodes] = V[1];
  }

  for (int ele = 0; ele < intersection->GetQuantity() - 2; ele++)
  {
    enlist[ele * 3] = 1;
    enlist[ele * 3 + 1] = ele + 2;
    enlist[ele * 3 + 2] = ele + 3;
  }

  return;
}

WmElementIntersector3D::WmElementIntersector3D()
{
  intersection = NULL;
  volumes = NULL;
  elements = -1;
  nodes = -1;
  exactness = 0;

  return;
}
WmElementIntersector3D::~WmElementIntersector3D()
{
  if(intersection)
  {
    delete intersection;
    intersection = NULL;
  }
  if(volumes)
  {
    delete volumes;
    volumes = NULL;
  }

  return;
}

void WmElementIntersector3D::Intersect()
{
  assert(positionsA);
  assert(positionsB);

  if(intersection)
  {
    delete intersection;
    intersection = NULL;
  }
  if(volumes)
  {
    delete volumes;
    volumes = NULL;
  }

  Vector3 cur_vector;

  // The reason we don't pass in positionsA[0], positionsA[3], etc.
  // is because we need to cast it to GEOM_REAL
  GEOM_REAL tmp_vec[3];
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsA[0 + i];
  Vector3 ptA1(tmp_vec);
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsA[3 + i];
  Vector3 ptA2(tmp_vec);
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsA[6 + i];
  Vector3 ptA3(tmp_vec);
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsA[9 + i];
  Vector3 ptA4(tmp_vec);
  Tetrahedron3 input_tetA(ptA1, ptA2, ptA3, ptA4);

  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsB[0 + i];
  Vector3 ptB1(tmp_vec);
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsB[3 + i];
  Vector3 ptB2(tmp_vec);
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsB[6 + i];
  Vector3 ptB3(tmp_vec);
  for (int i = 0; i < 3; i++)
    tmp_vec[i] = (GEOM_REAL) positionsB[9 + i];
  Vector3 ptB4(tmp_vec);
  Tetrahedron3 input_tetB(ptB1, ptB2, ptB3, ptB4);

  intersection = new IntrTetrahedron3Tetrahedron3(input_tetA, input_tetB);

#ifdef DDEBUG
  Tetrahedron3 tetA = intersection->GetTetrahedron0();
  Tetrahedron3 tetB = intersection->GetTetrahedron1();
  size_t pos_index = 0;
  for (size_t vector_idx = 0; vector_idx < 4; vector_idx++)
  {
    cur_vector = tetA.V[vector_idx];
    for (size_t vec_idx = 0; vec_idx < 3; vec_idx++)
    {
      assert(positionsA[pos_index++] == cur_vector[vec_idx]);
    }
  }
  pos_index = 0;
  for (size_t vector_idx = 0; vector_idx < 4; vector_idx++)
  {
    cur_vector = tetB.V[vector_idx];
    for (size_t vec_idx = 0; vec_idx < 3; vec_idx++)
    {
      assert(positionsB[pos_index++] == cur_vector[vec_idx]);
    }
  }
#endif

  intersection->Find();

  vector<Tetrahedron3> vec;
  vec = intersection->GetIntersection();
  elements = vec.size();
  volumes = new vector<GEOM_REAL>(elements);
  size_t tet_count = 0;
  GEOM_REAL vol;
  for (vector<Tetrahedron3>::iterator tet = vec.begin(); tet != vec.end(); tet++)
  {
    // Remove degenerate elements
    vol = (*tet).GetVolume();
    (*volumes)[tet_count++] = vol;
    if (vol <= 0.0)
      elements--;
  }

  nodes = elements * loc;

  return;
}

void WmElementIntersector3D::QueryOutput(int& nnodes, int& nelms) const
{
  assert(elements >= 0);
  assert(nodes >= 0);

  nelms = elements;
  nnodes = nodes;

  return;
}

void WmElementIntersector3D::GetOutput(double*& positions, int*& enlist) const
{
  assert(intersection);
  assert(volumes);
  assert(elements >= 0);
  assert(nodes >= 0);

  vector<Tetrahedron3> vec;
  vec = intersection->GetIntersection();

  for (size_t enlist_index = 0; enlist_index < (size_t) loc * elements; enlist_index++)
  {
    enlist[enlist_index] = enlist_index + 1;
  }

  size_t pos_index = 0;
  size_t vector_idx;
  Vector3 cur_vector;
  size_t tet_count = 0;
  for (vector<Tetrahedron3>::iterator tet = vec.begin(); tet != vec.end(); tet++, tet_count++)
  {
    if ((*volumes)[tet_count] > 0.0)
    {
      for (vector_idx = 0; vector_idx < 4; vector_idx++, pos_index++)
      {
        cur_vector = (*tet).V[vector_idx];
        positions[pos_index] = cur_vector[0];
        positions[pos_index + nodes] = cur_vector[1];
        positions[pos_index + 2 * nodes] = cur_vector[2];
      }
    }
  }

  return;
}

ElementIntersector* elementIntersector = NULL;

ElementIntersectionFinder elementIntersectionFinder;

extern "C"
{
  int cIntersectorGetDimension()
  {
    assert(elementIntersector);

    return elementIntersector->GetDim();
  }

  void cIntersectorSetDimension(const int* dim)
  {
    if(elementIntersector)
    {
      if((int) elementIntersector->GetDim() == *dim)
      {
        return;
      }

      delete elementIntersector;
      elementIntersector = NULL;
    }

    ElementIntersector1D* intersector_1;
    ElementIntersector2D* intersector_2;
    WmElementIntersector3D* intersector_3;
    switch(*dim)
    {
      case 1:
        intersector_1 = new ElementIntersector1D();
        elementIntersector = (ElementIntersector*)intersector_1;
        break;
      case 2:
        intersector_2 = new ElementIntersector2D();
        elementIntersector = (ElementIntersector*)intersector_2;
        break;
      case 3:
        intersector_3 = new WmElementIntersector3D();
        elementIntersector = (ElementIntersector*)intersector_3;
        break;
      default:
        cerr << "Invalid element intersector dimension" << endl;
        exit(-1);
    }

    return;
  }

  void cIntersectorSetExactness(const int* exact)
  {
    /*Sets the exactness (whether we're using cgal or not)
    exact = 0 means inexact (not using cgal)
    exact = 1 means exact (using cgal)*/

    int dim;

    assert(elementIntersector);
    dim = elementIntersector->GetDim();
    if((int) elementIntersector->GetExactness() == *exact)
    {
      return;
    }
    else
    {
      delete elementIntersector;
    }

    ElementIntersector1D* intersector_1_inexact;
    ElementIntersector2D* intersector_2_inexact;
    ElementIntersectorCGAL2D* intersector_2_exact;
    ElementIntersector3D* intersector_3_inexact;
    ElementIntersectorCGAL3D* intersector_3_exact;

    switch(dim)
    {
      case 1:
        if (*exact == 0)
        {
          intersector_1_inexact = new ElementIntersector1D();
          elementIntersector = (ElementIntersector*)intersector_1_inexact;
        }
        else{
          cerr << "Exact intersector not available in 1D" << endl;
          exit(-1);
        }
        break;
      case 2:
        if (*exact == 0)
        {
          intersector_2_inexact = new ElementIntersector2D();
          elementIntersector = (ElementIntersector*)intersector_2_inexact;
        }
        else
        {
          intersector_2_exact = new ElementIntersectorCGAL2D();
          elementIntersector = (ElementIntersector*)intersector_2_exact;
        }
        break;
      case 3:
        if (*exact == 0)
        {
          intersector_3_inexact = new WmElementIntersector3D();
          elementIntersector = (ElementIntersector*)intersector_3_inexact;
        }
        else
        {
          intersector_3_exact = new ElementIntersectorCGAL3D();
          elementIntersector = (ElementIntersector*)intersector_3_exact;
        }
        break;
      default:
        cerr << "Invalid element intersector dimension" << endl;
        exit(-1);
    }

    return;
  }

  void cIntersectorSetInput(double* positionsA, double* positionsB, const int* dim, const int* loc)
  {
    assert(elementIntersector);
    assert(*dim >= 0);
    assert(*loc >= 0);

    elementIntersector->SetInput(positionsA, positionsB, *dim, *loc);

    return;
  }

  void cIntersectorDrive()
  {
    assert(elementIntersector);

    elementIntersector->Intersect();

    return;
  }

  void cIntersectorQuery(int* nnodes, int* nelms)
  {
    assert(elementIntersector);

    elementIntersector->QueryOutput(*nnodes, *nelms);

    return;
  }

  void cIntersectorGetOutput(const int* nnodes, const int* nelms, const int* dim, const int* loc, double* positions, int* enlist)
  {
    assert(elementIntersector);

#ifdef DDEBUG
    int nnodesQuery, nelmsQuery;
    elementIntersector->QueryOutput(nnodesQuery, nelmsQuery);
    assert(*nnodes == nnodesQuery);
    assert(*nelms == nelmsQuery);
    switch(elementIntersector->GetDim())
    {
      case 1:
        assert(*dim == 1);
        assert(*loc == 2);
        break;
      case 2:
        assert(*dim == 2);
        assert(*loc == 3 || *loc == 4);
        break;
      case 3:
        assert(*dim == 3);
        assert(*loc == 4);
        break;
      default:
        cerr << "Invalid intersector dimension" << endl;
        exit(-1);
    }
#endif

    elementIntersector->GetOutput(positions, enlist);

    return;
  }

  void cIntersectionFinderReset(int* ntests)
  {
    *ntests = elementIntersectionFinder.Reset();

    return;
  }

  void cIntersectionFinderSetInput(const double* positions, const int* enlist, const int* dim, const int* loc, const int* nnodes, const int* nelements)
  {
    assert(*dim >= 0);
    assert(*loc >= 0);
    assert(*nnodes >= 0);
    assert(*nelements >= 0);

    elementIntersectionFinder.SetInput(positions, *nnodes, *dim, enlist, *nelements, *loc);

    return;
  }

  void cIntersectionFinderFind(const double* positions, const int* dim, const int* loc)
  {
    assert(*dim >= 0);
    assert(*loc >= 0);

    elementIntersectionFinder.SetTestElement(positions, *dim, *loc);

    return;
  }

  void cIntersectionFinderQueryOutput(int* nelms)
  {
    elementIntersectionFinder.QueryOutput(*nelms);

    return;
  }

  void cIntersectionFinderGetOutput(int* id, const int* index)
  {
    elementIntersectionFinder.GetOutput(*id, *index);

    return;
  }
}
