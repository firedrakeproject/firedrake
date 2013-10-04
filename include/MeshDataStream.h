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

#ifndef MESHDATASTREAM_H
#define MESHDATASTREAM_H

#include "confdefs.h"

#include <strings.h>
#include <spatialindex/SpatialIndex.h>

#include <cassert>
#include <iostream>
#include <map>
#include <vector>
#include <set>

namespace Fluidity{

  // Customised version of MyDataStream class in
  // regressiontest/rtree/RTreeBulkLoad.cc in spatialindex 1.2.0
  class MeshDataStream : public SpatialIndex::IDataStream{
  public:
    MeshDataStream(const double*& positions, const int& nnodes, const int& dim,
                   const int*& enlist, const int& nelements, const int& loc);
    virtual ~MeshDataStream();

    virtual SpatialIndex::IData* getNext();
    virtual bool hasNext();
    virtual uint32_t size();
    virtual void rewind();

    virtual int getPredicateCount();

    class MDSInstrumentedRegion : public SpatialIndex::Region
    {
      public:
        MDSInstrumentedRegion(MeshDataStream* mdss){mds = mdss;}
        MDSInstrumentedRegion(MeshDataStream* mdss, const double* pLow, const double* pHigh, size_t dimension){mds = mdss;}
        MDSInstrumentedRegion(MeshDataStream* mdss, const SpatialIndex::Point& low, const SpatialIndex::Point& high){mds = mdss;}
        MDSInstrumentedRegion(MeshDataStream* mdss, const SpatialIndex::Region& in){mds = mdss;}

        virtual bool intersectsRegion(const Region& in);
        virtual bool containsRegion(const Region& in);
        virtual bool touchesRegion(const Region& in);

      private:
        MeshDataStream* mds;
    };

    int predicateCount;

  protected:
    const double* positions;
    const int* enlist;
    int dim, index, nelements, loc, nnodes;
  };

  class ExpandedMeshDataStream : public MeshDataStream
  {
    public:
      inline ExpandedMeshDataStream(const double*& positions, const int& nnodes, const int& dim,
                   const int*& enlist, const int& nelements, const int& loc, const double& expansionFactor)
        : MeshDataStream(positions, nnodes, dim, enlist, nelements, loc),
          expansionFactor(fabs(expansionFactor))
      {
        return;
      }

      inline virtual ~ExpandedMeshDataStream()
      {
        return;
      }

      virtual SpatialIndex::IData* getNext();
    private:
      double expansionFactor;
  };
}

#endif
