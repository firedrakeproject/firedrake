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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <set>

#include "fmangle.h"

using namespace std;

extern "C" { // This is the glue between METIS and fluidity

  // Declarations needed from METIS
  typedef int idxtype;
  void METIS_PartGraphKway(int *,idxtype *,idxtype *,idxtype *,idxtype *,int *,int *,int *,
                           int *,int *,idxtype *);
  void METIS_PartGraphRecursive(int *,idxtype *,idxtype *,idxtype *,idxtype *,int *,int *,
                                int *,int *,int *,idxtype *);

#ifndef HAVE_MPI
        void METIS_PartGraphKway(int *a, idxtype *b, idxtype *c, idxtype *d, idxtype *e,
                                 int *f, int *g, int *h, int *i, int *j, idxtype *k){}
        void METIS_PartGraphRecursive(int *a, idxtype *b, idxtype *c, idxtype *d,
                                      idxtype *e, int *f, int *g, int *h, int *i, int *j, idxtype *k){}
#endif
}

namespace Fluidity{
  int FormGraph(const vector<int>& ENList, const int& dim, const int& nloc, const int& nnodes,
                vector<set<int> >& graph){
    int num_elems = ENList.size()/nloc;

    graph.clear();  graph.resize(nnodes);

    switch (dim){
    case 1:
      switch(nloc){
      case 2:
        // Lines
        for(int i=0;i<num_elems;i++){
          graph[ENList[i*nloc]-1].insert(ENList[i*nloc+1]);
          graph[ENList[i*nloc+1]-1].insert(ENList[i*nloc]);
        }
        break;
      default:
        cerr<<"ERROR: element type not recognised - dim = "<<dim<<", nloc = "<<nloc<<endl;
        return -1;
      }
      break;
    case 2:
      switch(nloc){
      case 3:
        // Linear triangles
        for(int i=0;i<num_elems;i++){
          for(int j=0;j<nloc;j++){
            for(int k=0;k<nloc;k++){
              if(j!=k)
                graph[ENList[i*nloc+j]-1].insert(ENList[i*nloc+k]);
            }
          }
        }
        break;
      default:
        cerr<<"ERROR: element type not recognised - dim = "<<dim<<", nloc = "<<nloc<<endl;
        return -1;
      }
      break;
    case 3:
      switch (nloc){
      case 3:
        // Linear triangles
        for(int i=0;i<num_elems;i++){
          for(int j=0;j<nloc;j++){
            for(int k=0;k<nloc;k++){
              if(j!=k)
                graph[ENList[i*nloc+j]-1].insert(ENList[i*nloc+k]);
            }
          }
        }
        break;
      case 4:
        // Linear tets
        for(int i=0;i<num_elems;i++){
          for(int j=0;j<nloc;j++){
            for(int k=0;k<nloc;k++){
              if(j!=k)
                graph[ENList[i*nloc+j]-1].insert(ENList[i*nloc+k]);
            }
          }
        }
        break;
      case 8:
        // Linear hexes
        for(int i=0;i<num_elems;i++){
          graph[ENList[i*nloc  ]-1].insert(ENList[i*nloc+1]);
          graph[ENList[i*nloc  ]-1].insert(ENList[i*nloc+2]);
          graph[ENList[i*nloc  ]-1].insert(ENList[i*nloc+4]);

          graph[ENList[i*nloc+1]-1].insert(ENList[i*nloc  ]);
          graph[ENList[i*nloc+1]-1].insert(ENList[i*nloc+3]);
          graph[ENList[i*nloc+1]-1].insert(ENList[i*nloc+5]);

          graph[ENList[i*nloc+2]-1].insert(ENList[i*nloc  ]);
          graph[ENList[i*nloc+2]-1].insert(ENList[i*nloc+3]);
          graph[ENList[i*nloc+2]-1].insert(ENList[i*nloc+6]);

          graph[ENList[i*nloc+3]-1].insert(ENList[i*nloc+1]);
          graph[ENList[i*nloc+3]-1].insert(ENList[i*nloc+2]);
          graph[ENList[i*nloc+3]-1].insert(ENList[i*nloc+7]);

          graph[ENList[i*nloc+4]-1].insert(ENList[i*nloc  ]);
          graph[ENList[i*nloc+4]-1].insert(ENList[i*nloc+5]);
          graph[ENList[i*nloc+4]-1].insert(ENList[i*nloc+6]);

          graph[ENList[i*nloc+5]-1].insert(ENList[i*nloc+1]);
          graph[ENList[i*nloc+5]-1].insert(ENList[i*nloc+4]);
          graph[ENList[i*nloc+5]-1].insert(ENList[i*nloc+7]);

          graph[ENList[i*nloc+6]-1].insert(ENList[i*nloc+2]);
          graph[ENList[i*nloc+6]-1].insert(ENList[i*nloc+4]);
          graph[ENList[i*nloc+6]-1].insert(ENList[i*nloc+7]);

          graph[ENList[i*nloc+7]-1].insert(ENList[i*nloc+3]);
          graph[ENList[i*nloc+7]-1].insert(ENList[i*nloc+5]);
          graph[ENList[i*nloc+7]-1].insert(ENList[i*nloc+6]);
        }
        break;
      default:
        cerr<<"ERROR: element type not recognised - dim = "<<dim<<", nloc = "<<nloc<<endl;
      return -1;
      }
      break;
    default:
      cerr<<"ERROR: element type not recognised - dim = "<<dim<<", nloc = "<<nloc<<endl;
      return -1;
    }

    return 0;
  }

  int gpartition(const vector< set<int> > &graph, int npartitions, int partition_method, vector<int> &decomp){
    // If no partitioning method is set, choose a default.
    if(partition_method<0){
      if(npartitions<=8)
        partition_method = 0; // METIS PartGraphRecursive
      else
        partition_method = 1; // METIS PartGraphKway
    }

    int nnodes = graph.size();

    // Compress graph
    vector<idxtype> xadj(nnodes+1), adjncy;
    int pos=0;
    xadj[0]=1;
    for(int i=0;i<nnodes;i++){
      for(set<int>::iterator jt=graph[i].begin();jt!=graph[i].end();jt++){
        adjncy.push_back(*jt);
        pos++;
      }
      xadj[i+1] = pos+1;
    }

    // Partition graph
    decomp.resize(nnodes);
    int wgtflag=0, numflag=1, options[] = {0}, edgecut=0;

    if(partition_method){
      METIS_PartGraphKway(&nnodes, &(xadj[0]), &(adjncy[0]), NULL, NULL, &wgtflag,
                          &numflag, &npartitions, options, &edgecut, &(decomp[0]));
    }else{
      METIS_PartGraphRecursive(&nnodes, &(xadj[0]), &(adjncy[0]), NULL, NULL, &wgtflag,
                               &numflag, &npartitions, options, &edgecut, &(decomp[0]));
    }

    // number from zero
    for(int i=0;i<nnodes;i++)
      decomp[i]--;

    return edgecut;
  }

  int hpartition(const vector< set<int> > &graph, vector<int>& npartitions, int partition_method, vector<int> &decomp){

    vector<int> hdecomp;
    int edgecut = gpartition(graph, npartitions[0], partition_method, hdecomp);

    if(npartitions.size()==2){
      decomp.resize(graph.size());
      for(int i=0;i<npartitions[0];i++){
        map<int, int> renumber;
        int cnt=0;
        for(size_t j=0;j<hdecomp.size();j++){
          if(i==hdecomp[j]){
            renumber[j]=cnt++;
          }
        }

        vector< set<int> > pgraph(renumber.size());
        for(map<int, int>::const_iterator it=renumber.begin();it!=renumber.end();++it){
          for(set<int>::const_iterator jt=graph[it->first].begin();jt!=graph[it->first].end();++jt){
            if(renumber.find((*jt)-1)!=renumber.end()){
              pgraph[it->second].insert(renumber[(*jt)-1]+1);
            }
          }
        }

        vector<int> pdecomp, ncores(npartitions[1], 0);
        set<int> parts;
        int sedgecut = gpartition(pgraph, npartitions[1], partition_method, pdecomp);
        for(map<int, int>::const_iterator it=renumber.begin();it!=renumber.end();++it){
          ncores[pdecomp[it->second]]++;
          decomp[it->first] = hdecomp[it->first]*npartitions[1] + pdecomp[it->second];
          parts.insert(decomp[it->first]);
        }
      }
    }else{
      decomp.swap(hdecomp);
    }

    return edgecut;
  }

  int partition(const vector<int> &ENList, const int& dim, int nloc, int nnodes, vector<int>& npartitions, int partition_method, vector<int> &decomp){
    // Build graph
    vector< set<int> > graph;
    int ret = FormGraph(ENList, dim, nloc, nnodes, graph);
    if(ret != 0){
      return ret;
    }

    int edgecut = hpartition(graph, npartitions, partition_method, decomp);

    return edgecut;
  }

  int partition(const vector<int> &ENList, int nloc, int nnodes, vector<int>& npartitions, int partition_method, vector<int> &decomp){
    return partition(ENList, 3, nloc, nnodes, npartitions, partition_method, decomp);
  }

  int partition(const vector<int> &ENList, const vector<int> &surface_nids, const int& dim, int nloc, int nnodes, vector<int>& npartitions, int partition_method, vector<int> &decomp){
    int num_elems = ENList.size()/nloc;

    set<int> surface_nodes;
    for(vector<int>::const_iterator it=surface_nids.begin(); it!=surface_nids.end(); ++it){
      surface_nodes.insert(*it);
    }

    // Build graph
    map<int, set<int> > graph;
    switch (dim){
    case 3:
      switch (nloc){
      case 4:
        for(int i=0;i<num_elems;i++){
          for(int j=0;j<nloc;j++){
            if(surface_nodes.find(ENList[i*nloc+j])!=surface_nodes.end())
              for(int k=0;k<nloc;k++){
                if((j!=k)&&(surface_nodes.find(ENList[i*nloc+k])!=surface_nodes.end()))
                  graph[ENList[i*nloc+j]].insert(ENList[i*nloc+k]);
              }
          }
        }
        break;
      case 8:
        cerr<<"ERROR: Extrude support not implemented for hex's\n";
        exit(-1);
      default:
        cerr<<"ERROR: element type not recognised - dim = "<<dim<<", nloc = "<<nloc<<endl;
        return -1;
      }
      break;
    default:
      cerr<<"ERROR: element type not recognised - dim = "<<dim<<", nloc = "<<nloc<<endl;
      return -1;
    }

    int snnodes = graph.size();

    vector< set<int> > cgraph(snnodes);
    for(map<int, set<int> >::const_iterator it=graph.begin(); it!=graph.end(); ++it)
      for(set<int>::const_iterator jt=it->second.begin(); jt!=it->second.end(); ++jt){
        cgraph[surface_nids[it->first-1]-1].insert(surface_nids[*jt-1]);
      }
    vector<int> sdecomp(snnodes);
    int edgecut = hpartition(cgraph, npartitions, partition_method, sdecomp);

    // Map 2D decomposition onto 3D mesh.
    decomp.resize(nnodes);
    for(int i=0;i<nnodes;i++)
      decomp[i] = sdecomp[surface_nids[i]-1];

    return edgecut;
  }

  int partition(const vector<int> &ENList, const vector<int> &surface_nids, int nloc, int nnodes, vector<int>& npartitions, int partition_method, vector<int> &decomp){

    return partition(ENList, surface_nids, 3, nloc, nnodes, npartitions, partition_method, decomp);
  }
}
