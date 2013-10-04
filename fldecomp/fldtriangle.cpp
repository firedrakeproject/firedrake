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


#include "fldecomp.h"



// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------




void write_partitions_triangle(bool verbose,
                  string filename, string file_format,
                  int nparts, int nnodes, int dim, int no_coords,
                  const vector<double>&x, const vector<int>& decomp,
                  int nloc, const vector<int>& ENList,
                  const vector<int>& regionIds,
                  int snloc, const deque< vector<int> >& SENList,
                  const vector<int>& boundaryIds)
{
  // Construct:
  //   nodes    - nodes in each partition (private and halo), numbered from
  //              one
  //   npnodes  - number of nodes private to each partition
  //   elements - elements ids in each partition (private and halo), numbered
  //              from zero
  //   halo1    - receive nodes (numbered from one) in the first halo for each
  //              partition
  //   halo2    - receive nodes (numbered from one) in the second halo for
  //              each partition
  // where in each case partitions are numbered from zero.


  if(verbose)
    cout<<"void write_partitions_triangle( ... )";

  int nelms = ENList.size()/nloc;

  vector< vector< set<int> > > halo1(nparts);
  vector< vector< set<int> > > halo2(nparts);
  vector< map<int, int> > renumber(nparts);
  for(int part=0; part<nparts; part++){
    halo1[part].resize(nparts);
    halo2[part].resize(nparts);
  }
  vector<int> npnodes(nparts, 0);

#pragma omp parallel
  {
#pragma omp for
  for(int part=0; part<nparts; part++){
    if(verbose)
      cout<<"Making partition "<<part<<endl;

    deque<int> *nodes = new deque<int>;
    deque<int> *elements = new deque<int>;

    vector<bool> *halo_nodes = new vector<bool>(nnodes, false);
    vector<bool> *more_halo_nodes = new vector<bool>(nnodes, false);

    // Find owned nodes
    for(int nid=0; nid<nnodes; nid++){
      if(decomp[nid]==part)
        nodes->push_back(nid+1);
    }
    npnodes[part] = nodes->size();
    if(verbose)
      cout<<"Found "<<npnodes[part]<<" owned nodes\n";

    // Find elements with owned nodes and halo1
    deque< pair<int, int> > *sorted_elements = new deque< pair<int, int> >;
    for(int eid=0;eid<nelms;eid++){
      int halo_count=0;
      pair<int, int> *owned_elm = new pair<int, int>(decomp[ENList[eid*nloc] - 1], eid);
      if(decomp[ENList[eid*nloc] - 1]!=part){
        halo_count++;
      }

      for(int j=1;j<nloc;j++){
        owned_elm->first = min(owned_elm->first, decomp[ENList[eid*nloc+j] - 1]);
        if(decomp[ENList[eid*nloc+j] - 1]!=part){
          halo_count++;
        }
      }

      if(halo_count<nloc){
        sorted_elements->push_back(*owned_elm);
        if(halo_count>0){
          for(int j=0;j<nloc;j++){
            int nid = ENList[eid*nloc+j] - 1;
            if(decomp[nid]!=part){
              halo1[part][decomp[nid]].insert(nid+1);
              (*halo_nodes)[nid] = true;
            }
          }
        }
      }
      delete owned_elm;
    }

    if(verbose)
      cout<<"Found halo1 nodes\n";

    for(deque< pair<int, int> >::const_iterator it=sorted_elements->begin();it!=sorted_elements->end();++it)
      if(it->first==part)
        elements->push_back(it->second);
    for(deque< pair<int, int> >::const_iterator it=sorted_elements->begin();it!=sorted_elements->end();++it)
      if(it->first!=part)
        elements->push_back(it->second);
    delete sorted_elements;

    if(verbose)
      cout<<"Sorted elements\n";

    // Find halo2 elements and nodes
    set<int> *halo2_elements = new set<int>;
    for(int eid=0; eid<nelms; eid++){
      int owned_node_count=0;
      bool touches_halo1=false;
      for(int j=0;j<nloc;j++){
        int fnid = ENList[eid*nloc+j];

        touches_halo1 = touches_halo1 || (*halo_nodes)[fnid-1];

        if(decomp[fnid-1]==part)
          owned_node_count++;
      }

      if(touches_halo1&&(owned_node_count==0)){
        halo2_elements->insert(halo2_elements->end(), eid);
        for(int j=0;j<nloc;j++){
          int fnid = ENList[eid*nloc+j];
          if(!(*halo_nodes)[fnid-1]){
            halo2[part][decomp[fnid-1]].insert(fnid);
            (*more_halo_nodes)[fnid-1]=true;
          }
        }
      }
    }

    if(verbose)
      cout<<"Found "<<halo2_elements->size()<<" halo2 elements\n";

    for(int i=0;i<nparts;i++)
      halo2[part][i].insert(halo1[part][i].begin(), halo1[part][i].end());

    for(int i=0;i<nnodes;i++)
      if((*halo_nodes)[i])
        nodes->push_back(i+1);
    delete halo_nodes;

    for(int i=0;i<nnodes;i++)
      if((*more_halo_nodes)[i])
        nodes->push_back(i+1);
    delete more_halo_nodes;

    for(set<int>::const_iterator it=halo2_elements->begin(); it!=halo2_elements->end(); ++it)
      elements->push_back(*it);
    delete halo2_elements;

    if(verbose)
      cout<<"Partition: "<<part<<", Private nodes: "<<npnodes[part]<<", Total nodes: "<<nodes->size()<<"\n";

    // Write out data for each partition
    if(verbose)
      cout<<"Write mesh data for partition "<<part<<"\n";

    // Map from global node numbering (numbered from one) to partition node
    // numbering (numbered from one)
    for(size_t j=0;j<nodes->size();j++){
      renumber[part].insert(renumber[part].end(), pair<int, int>((*nodes)[j], j+1));
    }

    // Coordinate data
    vector<double> *partX = new vector<double>(nodes->size()*no_coords);
    for(size_t j=0;j<nodes->size();j++){
      for(int k=0;k<no_coords;k++){
        (*partX)[j * no_coords + k] = x[((*nodes)[j] - 1) * no_coords + k];
      }
    }

    // Volume element data
    vector<int> *partENList = new vector<int>;
    partENList->reserve(elements->size()*nloc);

    vector<int> *partRegionIds = new vector<int>;
    partRegionIds->reserve(elements->size());

    // Map from partition node numbers (numbered from one) to partition
    // element numbers (numbered from zero)
    vector< set<int> > *partNodesToEid = new vector< set<int> >(nodes->size()+1);
    int ecnt=0;
    for(deque<int>::const_iterator iter=elements->begin();iter!=elements->end();iter++){
      for(int j=0;j<nloc;j++){

        int nid = ENList[*iter*nloc+j];
        int gnid = renumber[part].find(nid)->second;
        partENList->push_back(gnid);
        (*partNodesToEid)[gnid].insert(ecnt);
      }
      partRegionIds->push_back(regionIds[*iter]);
      ecnt++;
    }

    // Surface element data
    vector<int> *partSENList = new vector<int>;
    vector<int> *partBoundaryIds = new vector<int>;
    for(size_t j=0;j<SENList.size();j++){
      // In order for a global surface element to be a partition surface
      // element, all of its nodes must be attached to at least one partition
      // volume element
      if(SENList[j].size()==0 or
         renumber[part].find(SENList[j][0])==renumber[part].end() or
         (*partNodesToEid)[renumber[part].find(SENList[j][0])->second].empty()){
        continue;
      }

      bool SEOwned=false;
      set<int> &lpartNodesToEid = (*partNodesToEid)[renumber[part].find(SENList[j][0])->second];
      for(set<int>::const_iterator iter=lpartNodesToEid.begin();iter!=lpartNodesToEid.end();iter++){
        SEOwned=true;
        set<int> *VENodes = new set<int>;
        for(int k=(*iter)*nloc;k<(*iter)*nloc+nloc;k++){
          VENodes->insert((*partENList)[k]);
        }
        for(size_t k=1;k<SENList[j].size();k++){
          if(renumber[part].find(SENList[j][k])==renumber[part].end() or
             VENodes->count(renumber[part].find(SENList[j][k])->second)==0){
            SEOwned=false;
            break;
          }
        }
        if(SEOwned){
          break;
        }
        delete VENodes;
      }

      if(SEOwned){
        for(size_t k=0;k<SENList[j].size();k++){
          partSENList->push_back(renumber[part].find(SENList[j][k])->second);
        }
        partBoundaryIds->push_back(boundaryIds[j]);
      }
    }
    delete partNodesToEid;

    // Write out the partition mesh
    ostringstream basename;
    basename<<filename<<"_"<<part;
    if(verbose)
      cout<<"Writing out triangle mesh for partition "<<part<<" to files with base name "<<basename.str()<<"\n";

    ofstream nodefile;
    nodefile.open(string(basename.str()+".node").c_str());
    nodefile.precision(16);
    nodefile<<nodes->size()<<" "<<dim<<" 0 0\n";
    for(size_t j=0;j<nodes->size();j++){
      nodefile<<j+1<<" ";
      for(int k=0;k<no_coords;k++){
        nodefile<<(*partX)[j * no_coords + k]<<" ";
      }
      nodefile<<endl;
    }
    nodefile<<"# Produced by: fldecomp\n";
    nodefile.close();
    delete nodes;
    delete partX;

    ofstream elefile;
    elefile.open(string(basename.str()+".ele").c_str());
    if(partRegionIds->size())
      elefile<<elements->size()<<" "<<nloc<<" 1\n";
    else
      elefile<<elements->size()<<" "<<nloc<<" 0\n";
    for(int i=0;i<elements->size();i++){
      elefile<<i+1<<" ";
      for(int j=0;j<nloc;j++)
        elefile<<(*partENList)[i*nloc+j]<<" ";
      if(partRegionIds->size())
        elefile<<(*partRegionIds)[i];
      elefile<<endl;
    }
    elefile<<"# Produced by: fldecomp\n";
    elefile.close();
    delete elements;
    delete partENList;

    ofstream facefile;
    if(snloc==1)
      facefile.open(string(basename.str()+".bound").c_str());
    else if(snloc==2)
      facefile.open(string(basename.str()+".edge").c_str());
    else
      facefile.open(string(basename.str()+".face").c_str());
    int nfacets = partSENList->size()/snloc;
    facefile<<nfacets<<" 1\n";
    for(int i=0;i<nfacets;i++){
      facefile<<i+1<<" ";
      for(int j=0;j<snloc;j++)
        facefile<<(*partSENList)[i*snloc+j]<<" ";
      facefile<<" "<<(*partBoundaryIds)[i]<<endl;
    }
    facefile.close();
    delete partSENList;
    delete partBoundaryIds;

    basename.str("");
  }
  }

  const int halo1_level = 1, halo2_level = 2;
  for(int i=0;i<nparts;i++){
    // Extract halo data
    if(verbose)
      cout<<"Extracting halo data for partition "<<i<<"\n";
    map<int, vector< vector<int> > > send, recv;
    map<int, int> npnodes_handle;

    recv[halo1_level].resize(nparts);
    send[halo1_level].resize(nparts);

    recv[halo2_level].resize(nparts);
    send[halo2_level].resize(nparts);

    for(int j=0;j<nparts;j++){
      for(set<int>::const_iterator it=halo1[i][j].begin();it!=halo1[i][j].end();++it){
        recv[halo1_level][j].push_back(renumber[i][*it]);
      }
      for(set<int>::const_iterator it=halo1[j][i].begin();it!=halo1[j][i].end();++it){
        send[halo1_level][j].push_back(renumber[i][*it]);
      }

      for(set<int>::const_iterator it=halo2[i][j].begin();it!=halo2[i][j].end();++it){
        recv[halo2_level][j].push_back(renumber[i][*it]);
      }
      for(set<int>::const_iterator it=halo2[j][i].begin();it!=halo2[j][i].end();++it){
        send[halo2_level][j].push_back(renumber[i][*it]);
      }
    }

    npnodes_handle[halo1_level]=npnodes[i];
    npnodes_handle[halo2_level]=npnodes[i];

    ostringstream buffer;
    buffer<<filename<<"_"<<i<<".halo";
    if(verbose)
      cout<<"Writing out halos for partition "<<i<<" to file "<<buffer.str()<<"\n";

    if(WriteHalos(buffer.str(), i, nparts, npnodes_handle, send, recv)){
      cerr<<"ERROR: failed to write halos to file "<<buffer.str()<<endl;
      exit(-1);
    }
    buffer.str("");
  }

  return;
}




// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------




int decomp_triangle( map<char, string> flArgs, bool verbose,
                              string filename, string file_format,
                              int nparts, int ncores )
{

  bool extruded = (flArgs.find('t')!=flArgs.end());
  if(extruded&&verbose){
    // This triangle file came from Terreno and should have
    // attribure data indicating the surface node lieing above the
    // node in the extruded mesh.
    cout<<"Reading in extrusion information\n";
  }

  bool shell = (flArgs.find('s')!=flArgs.end());

  string filename_node = filename+".node";
  if(verbose)
    cout<<"Reading "<<filename_node<<endl;

  fstream node_file;
  node_file.open(filename_node.c_str(), ios::in);
  if(!node_file.is_open()){
    cerr<<"ERROR: Triangle file, "<<filename_node
        <<", cannot be opened. Does it exist? Have you read permission?\n";
    exit(-1);
  }

  int nnodes, dim, natt, nboundary;
  node_file>>nnodes>>dim>>natt>>nboundary;
  if(extruded){
    if(natt!=1){
      cerr<<"ERROR: The -t option is specified but there is not the right number "
          <<"of attributes in the .node file.\n";
    }
  }

  int no_coords; // The number of coordinates
  if(shell){
    no_coords=dim+1; // Shell meshes have and x, y and z coordinate.
  }else{
    no_coords=dim;
  }

  vector<double> x(nnodes*no_coords);
  vector<int> surface_nids;
  if(extruded||(natt==1))
    surface_nids.resize(nnodes);

  {
    int id, pos=0;
    for(int i=0;i<nnodes;i++){
      node_file>>id;
      for(int j=0;j<no_coords;j++)
        node_file>>x[pos++];
      if(natt)
        node_file>>surface_nids[i];
    }
  }
  node_file.close();

  string filename_ele;
  filename_ele = filename+".ele";
  if(verbose)
    cout<<"Reading "<<filename_ele<<endl;

  fstream ele_file;
  ele_file.open(filename_ele.c_str(), ios::in);
  if(!ele_file.is_open()){
    cerr<<"ERROR: Triangle file, "<<filename_ele
        <<", cannot be opened. Does it exist? Have you read permission?\n";
    exit(-1);
  }

  vector<int> ENList, regionIds;
  int nele, nloc;
  {
    int natt, id, pos=0;
    ele_file>>nele>>nloc>>natt;
    ENList.resize(nele*nloc);
    regionIds.resize(nele);
    if(natt>1){
      cerr<<"ERROR: Don't know what to do with more than 1 attribute.\n";
      exit(-1);
    }

    for(int i=0;i<nele;i++){
      ele_file>>id;
      for(int j=0;j<nloc;j++)
        ele_file>>ENList[pos++];
      if(natt)
        ele_file>>regionIds[i];
      else
        regionIds[i]=0;
    }
  }
  ele_file.close();

  string filename_face;
  if((dim==3)&&(nloc==3)){
    filename_face = filename+".edge";
  }else if(dim==3){
    filename_face = filename+".face";
  }else if(dim==2){
    filename_face = filename+".edge";
  }else if(dim==1){
    filename_face = filename+".bound";
  }else{
    cerr<<"ERROR: dim=="<<dim<<" not supported.\n";
    exit(-1);
  }
  if(verbose)
    cout<<"Reading "<<filename_face<<endl;

  fstream face_file;

  face_file.open(filename_face.c_str(), ios::in);

  if(!face_file.is_open()){
    cerr<<"ERROR: Triangle file, "<<filename_face
        <<", cannot be opened. Does it exist? Have you read permission?\n";
    exit(-1);
  }

  deque< vector<int> > SENList;
  deque< vector<int> > columnSENList;
  vector<int> topSENList;
  // Set the boundary id of the extruded surface -- default 1.
  int bid = 1;
  if(flArgs.count('t'))
    bid = atoi(flArgs['t'].c_str());
  vector<int> boundaryIds;
  int nsele, snloc, snnodes;
  int max_snnodes=0, min_snnodes=0;
  {
    int natt, id;
    face_file>>nsele>>natt;
    if((nloc==4)&&(dim==3))
      snloc=3;
    else if((nloc==4)&&(dim==2))
      snloc=2;
    else if((nloc==2)&&(dim==1))
      snloc=1;
    else if(nloc==3)
      snloc=2;
    else if(nloc==8)
      snloc=4;
    else{
      cerr<<"ERROR: no idea what snloc is.\n";
      exit(-1);
    }
    SENList.resize(nsele);
    columnSENList.resize(nsele);
    if(natt>1){
      cerr<<"ERROR: Don't know what to do with more than 1 attribute.\n";
      exit(-1);
    }
    if(natt)
      boundaryIds.resize(nsele);

    for(int i=0;i<nsele;i++){
      vector<int> facet(snloc);
      vector<int> facet_columns(snloc);
      face_file>>id;
      for(int j=0;j<snloc;j++){
        face_file>>facet[j];
        if(surface_nids.size())
          facet_columns[j]=surface_nids[facet[j]-1];
      }
      SENList[i] = facet;
      if(surface_nids.size())
        columnSENList[i] = facet_columns;
      if(natt){
        face_file>>boundaryIds[i];
        if(boundaryIds[i]==bid){
          for(int j=0;j<snloc;j++){
            if(surface_nids.size()){
              topSENList.push_back(columnSENList[i][j]);
              max_snnodes=max(max_snnodes, columnSENList[i][j]);
              min_snnodes=min(min_snnodes, columnSENList[i][j]);
              snnodes=max_snnodes-min_snnodes+1;
            }else{
              topSENList.push_back(SENList[i][j]);
              max_snnodes=max(max_snnodes, SENList[i][j]);
              min_snnodes=min(min_snnodes, SENList[i][j]);
              snnodes=max_snnodes-min_snnodes+1;
            }
          }
        }
      }
    }
  }
  columnSENList.clear();
  face_file.close();

  vector<int> decomp;
  int partition_method = -1;

  if(flArgs.count('r')){
    partition_method = 0; // METIS PartGraphRecursive

    if(flArgs.count('k')){
      cerr<<"WARNING: should not specify both -k and -r. Choosing -r.\n";
    }
  }
  if(flArgs.count('k')){
    partition_method = 1; // METIS PartGraphKway
  }

  vector<int> npartitions;
  if(ncores>1){
    npartitions.push_back(nparts/ncores);
    npartitions.push_back(ncores);
  }else{
    npartitions.push_back(nparts);
  }

  int edgecut=0;
  if(surface_nids.size()){
    // Partition the mesh
    if(verbose)
      cout<<"Partitioning the extruded Terreno mesh\n";

    // Partition the mesh. Generates a map "decomp" from node number
    // (numbered from zero) to partition number (numbered from
    // zero).
    edgecut = partition(topSENList, 2, snloc, snnodes, npartitions, partition_method, decomp);
    topSENList.clear();
    decomp.resize(nnodes);
    vector<int> decomp_temp;
    decomp_temp.resize(nnodes);
    for(int i=0;i<nnodes;i++){
      decomp_temp[i] = decomp[surface_nids[i]-1]; // surface_nids=column number
    }
    decomp=decomp_temp;
    decomp_temp.clear();
  }else{
    // Partition the mesh
    if(verbose)
      cout<<"Partitioning the mesh\n";

    // Partition the mesh. Generates a map "decomp" from node number
    // (numbered from zero) to partition number (numbered from
    // zero).
    edgecut = partition( ENList, dim, nloc, nnodes, npartitions,
                         partition_method, decomp );
  }

  if(flArgs.count('d')){
    cout<<"Edge-cut: "<<edgecut<<endl;
  }

  // Process the partitioning
  if(verbose)
    cout<<"Processing the mesh partitions\n";

  write_partitions_triangle(verbose, filename, file_format,
                            nparts, nnodes, dim, no_coords,
                            x, decomp,
                            nloc, ENList, regionIds,
                            snloc, SENList, boundaryIds);

  return(0);
}

