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

#ifndef NODE_OWNERSHIP_H
#define NODE_OWNERSHIP_H

#include <map>
#include <vector>

#include "Element_Intersection.h"
#include "MeshDataStream.h"

namespace Fluidity {

  // Class to store 1d elements, used in NodeOwnerFinder below
  class Element1D
  {
    public:
      Element1D(const int id, const double StartPoint, const double EndPoint);

      int operator<(const Element1D &rhs) const;
      // element to the left of point
      int operator<(const double &rhs) const;
      // element to the right of point
      int operator>(const double &rhs) const;

      double StartPoint, EndPoint;
      int id;

  };

  // Interface to spatialindex to calculate node ownership lists using bulk
  // storage
  // Uses code from gispatialindex.{cc,h} in Rtree 0.4.1
  class NodeOwnerFinder
  {
    public:
      NodeOwnerFinder();
      ~NodeOwnerFinder();

      void Reset();
      void SetInput(const double*& positions, const int& nnodes, const int& dim,
                    const int*& enlist, const int& nelements, const int& loc);
      void SetTestPoint(const double*& position, const int& dim);
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

      std::vector<Element1D> mesh1d;
  };

}

extern std::map<int, Fluidity::NodeOwnerFinder*> nodeOwnerFinder;

extern "C" {
#define cNodeOwnerFinderReset F77_FUNC(cnode_owner_finder_reset, CNODE_OWNER_FINDER_RESET)
  void cNodeOwnerFinderReset(const int* id);

#define cNodeOwnerFinderSetInput F77_FUNC(cnode_owner_finder_set_input, CNODE_OWNER_FINDER_SET_INPUT)
  void cNodeOwnerFinderSetInput(int* id, const double* positions, const int* enlist, const int* dim, const int* loc, const int* nnodes, const int* nelements);

#define cNodeOwnerFinderFind F77_FUNC(cnode_owner_finder_find, CNODE_OWNER_FINDER_FIND)
  void cNodeOwnerFinderFind(const int* id, const double* position, const int* dim);

#define cNodeOwnerFinderQueryOutput F77_FUNC(cnode_owner_finder_query_output, CNODE_OWNER_FINDER_QUERY_OUTPUT)
  void cNodeOwnerFinderQueryOutput(const int* id, int* nelms);

#define cNodeOwnerFinderGetOutput F77_FUNC(cnode_owner_finder_get_output, CNODE_OWNER_FINDER_GET_OUTPUT)
  void cNodeOwnerFinderGetOutput(const int* id, int* ele_id, const int* index);
}

#endif
