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

#ifndef PARTITION_H
#define PARTITION_H

#include <set>
#include <vector>

namespace Fluidity{

  int FormGraph(const std::vector<int>& ENList, const int& dim, const int& nloc, const int& nnodes,
                std::vector<std::set<int> >& graph);

  int partition(const std::vector<int> &ENList, const int& dim, int nloc, int nnodes,
                std::vector<int>& npartitions, int partition_method, std::vector<int> &decomp);

  int partition(const std::vector<int> &ENList, int nloc, int nnodes,
                std::vector<int>& npartitions, int partition_method, std::vector<int> &decomp);

  int partition(const std::vector<int> &ENList, const std::vector<int> &surface_nids, const int& dim, int nloc,
                int nnodes, std::vector<int>& npartitions, int partition_method, std::vector<int> &decomp);

  int partition(const std::vector<int> &ENList, const std::vector<int> &surface_nids, int nloc, int nnodes,
                std::vector<int>& npartitions, int partition_method, std::vector<int> &decomp);
}

#endif
