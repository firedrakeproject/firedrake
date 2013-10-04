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

#include "Node_Owner_Finder.h"

#include <algorithm>
#include <vector>

using namespace SpatialIndex;

using namespace std;

using namespace Fluidity;

NodeOwnerFinder::NodeOwnerFinder()
{
  Initialise();

  return;
}

NodeOwnerFinder::~NodeOwnerFinder()
{
  Free();

  return;
}

void NodeOwnerFinder::Reset()
{
  Free();
  Initialise();
}

void NodeOwnerFinder::SetInput(const double*& positions, const int& nnodes, const int& dim,
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

  if (dim==1)
  {
    // preallocate enough elements
    mesh1d.reserve(nelements);
    for( int i=0; i<nelements; i++)
      // add element with fortran id=i+1
      mesh1d.push_back(
        Element1D(i+1, positions[enlist[i*loc]-1], positions[enlist[(i+1)*loc-1]-1]) );
    // now sort on location (see Element1D::operator< below)
    sort( mesh1d.begin(), mesh1d.end());
  }
  else
  {

    ExpandedMeshDataStream stream(positions, nnodes, dim,
                                enlist, nelements, loc,
                                // Expand the bounding boxes by 10%
                                0.1);

    // As in regressiontest/rtree/RTreeBulkLoad.cc in spatialindex 1.2.0
    id_type id = 1;
    rTree = RTree::createAndBulkLoadNewRTree(RTree::BLM_STR, stream, *storageManager, fillFactor, indexCapacity, leafCapacity, dim, SpatialIndex::RTree::RV_RSTAR, id);
  }

  return;
}

void NodeOwnerFinder::SetTestPoint(const double*& position, const int& dim)
{
  assert(position);
  assert(dim == this->dim);

  visitor.clear();

  if (dim==1){
    vector<Element1D>::iterator candidate;
    // Finds the first 1d element that is not to the left of
    // the specified point.
    candidate=lower_bound( mesh1d.begin(), mesh1d.end(), *position );

    // Because the point may be coincident with
    // the upper bound of the element we're looking for, this element
    // may still be considered to the left of the point due to round off
    // We therefore go back one to make sure it is included.
    if (candidate!=mesh1d.begin()) candidate--;

    // add candidates to visitor list as long as the candidate element
    // is not to the right of the specified point
    for (;
      !(candidate==mesh1d.end() || *candidate > *position) ;
      candidate++)
    {
      visitor.push_back( (*candidate).id );
    };

    // Similarly to above, because the point may be coincident with
    // the lower bound of the element we're looking for, this element
    // may be considered to the right of the point due to round off.
    // Add one extra candidate to make sure it is included.
    if (candidate!=mesh1d.end())
      visitor.push_back( (*candidate).id );

  }
  else
  {
    SpatialIndex::Point* point = new SpatialIndex::Point(position, dim);
    rTree->intersectsWithQuery(*point, visitor);
    delete point;
  }

  return;
}

void NodeOwnerFinder::QueryOutput(int& nelms) const
{
  nelms = visitor.size();

  return;
}

void NodeOwnerFinder::GetOutput(int& id, const int& index) const
{
  assert(index > 0);
  assert(index <= (int) visitor.size());

  id = visitor[index-1];

  return;
}

void NodeOwnerFinder::Initialise()
{
  storageManager = StorageManager::createNewMemoryStorageManager();
  storage = StorageManager::createNewRandomEvictionsBuffer(*storageManager, capacity, writeThrough);
  rTree = NULL;

  dim = 0;
  loc = 0;

  predicateCount = 0;

  return;
}

void NodeOwnerFinder::Free()
{
  if(rTree)
  {
    delete rTree;
    rTree = NULL;
  }
  delete storage;
  delete storageManager;

  mesh1d.clear();

  visitor.clear();

  return;
}

map<int, NodeOwnerFinder*> nodeOwnerFinder;

Element1D::Element1D(const int id, const double StartPoint, const double EndPoint)
{
  this->id=id;
  if (StartPoint<EndPoint)
  {
    this->StartPoint=StartPoint;
    this->EndPoint=EndPoint;
  }
  else
  {
    this->StartPoint=EndPoint;
    this->EndPoint=StartPoint;
  }

  return;
}

// comparison operator, only valid for comparing non-overlapping elements
int Element1D::operator<(const Element1D &rhs) const
{
  // as elements are non-overlapping, this should be safe
  // (comparing this->EndPoint with rhs.StartPoint we'd have to worry about round off)
  return this->StartPoint < rhs.StartPoint;
}

// compare 1d element with point, returns true
// if the element is completely to the left of the point
// (if the point is on top the EndPoint round off may be an issue)
int Element1D::operator<(const double &rhs) const
{
  return this->EndPoint < rhs;
}

// compare 1d element with point, returns true
// if the element is completely to the right of the point
// (if the point is on top the StartPoint round off may be an issue)
int Element1D::operator>(const double &rhs) const
{
  return this->StartPoint > rhs;
}

extern "C" {
  void cNodeOwnerFinderReset(const int* id)
  {
    if(nodeOwnerFinder.count(*id) > 0)
    {
      nodeOwnerFinder[*id]->Reset();
      delete nodeOwnerFinder[*id];
      nodeOwnerFinder.erase(*id);
    }

    return;
  }

  void cNodeOwnerFinderSetInput(int* id, const double* positions, const int* enlist, const int* dim, const int* loc, const int* nnodes, const int* nelements)
  {
    assert(*dim >= 0);
    assert(*loc >= 0);
    assert(*nnodes >= 0);
    assert(*nelements >= 0);

    *id = 1;
    while(nodeOwnerFinder.count(*id) > 0)
    {
      (*id)++;
    }

    nodeOwnerFinder[*id] = new NodeOwnerFinder();

    nodeOwnerFinder[*id]->SetInput(positions, *nnodes, *dim, enlist, *nelements, *loc);

    return;
  }

  void cNodeOwnerFinderFind(const int* id, const double* position, const int* dim)
  {
    assert(nodeOwnerFinder.count(*id) > 0);
    assert(nodeOwnerFinder[*id]);
    assert(*dim >= 0);

    nodeOwnerFinder[*id]->SetTestPoint(position, *dim);

    return;
  }

  void cNodeOwnerFinderQueryOutput(const int* id, int* nelms)
  {
    assert(nodeOwnerFinder.count(*id) > 0);
    assert(nodeOwnerFinder[*id]);

    nodeOwnerFinder[*id]->QueryOutput(*nelms);

    return;
  }

  void cNodeOwnerFinderGetOutput(const int* id, int* ele_id, const int* index)
  {
    assert(nodeOwnerFinder.count(*id) > 0);
    assert(nodeOwnerFinder[*id]);

    nodeOwnerFinder[*id]->GetOutput(*ele_id, *index);

    return;
  }
}
