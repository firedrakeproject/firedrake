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


// Matches element type with number of nodes.
int elemNumNodes[] =
  {2, 2, 3, 4, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };



// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------



// Element class to make things easier.

class Element
{
public:
  int ID, type, numTags, numNodes;
  vector <int> nodeIDs, tags;
  int physicalID, elementary;

  // Read method
  int read(fstream *efile, int _elemType, int _numNodes, int _numTags)
  {
    int n, t;

    type = _elemType;
    numNodes = _numNodes;
    numTags = _numTags;

    nodeIDs.resize(numNodes);
    tags.resize(numTags);

    efile->read( (char *)&ID, sizeof(int) );

    for(t=0; t<numTags; t++)
      efile->read( (char *)&tags[t], sizeof(int) );

    for(n=0; n<numNodes; n++)
      efile->read( (char *)&nodeIDs[n], sizeof(int) );

    // Standard tags
    if(numTags >= 2)
      {
        physicalID = tags[0];
        elementary = tags[1];
      }

    return 0;
  };

};





// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/*
  After the partitioning has been done, this bit writes out the main
  GMSH mesh file.
*/


void write_part_main_mesh( bool verbose, string filename, int part,
                           const vector<double> *partX,
                           deque <int> *nodes,
                           const int no_coords, const int nloc,
                           deque <int> *elements,
                           const vector<int> *partENList,
                           const vector<int> *partRegionIds,
                           const int snloc, vector<int> *partSENList,
                           const vector <int> *partBoundaryIds,
                           const int normElemType, const int faceType )
{

  // Can toggle GMSH binary/ascii format for debugging.
  int binaryFile=1;
  ofstream gmshfile;

  // Construct mesh file name from base name.
  ostringstream basename;
  basename << filename << "_" << part;
  string lfilename = basename.str() + ".msh";

  if(verbose)
    cout<<"Writing out GMSH mesh for partition "<<part
        <<" to file "<<lfilename<<"\n";


  // Open stream according to binary or ascii
  if(binaryFile)
    gmshfile.open( lfilename.c_str(), ios::binary | ios::out );
  else {
    gmshfile.open( lfilename.c_str(), ios::out );
    gmshfile.precision(16);
  }

  char newlineChar[2]="\n";

  // GMSH format header
  if(binaryFile)
    {
      gmshfile << "$MeshFormat\n";
      gmshfile << "2.1 1 " << sizeof(double) << "\n";
      int oneInt=1;
      gmshfile.write( (char *)&oneInt, sizeof(int) );
      gmshfile.write( (char *)&newlineChar, 1 );
      gmshfile << "$EndMeshFormat\n";
    } else {
    gmshfile << "$MeshFormat\n";
    gmshfile << "2.1 0 " << sizeof(double) << "\n";
    gmshfile << "$EndMeshFormat\n";
  }

  // Nodes section

  gmshfile << "$Nodes\n";
  gmshfile << nodes->size() << "\n";

  for(size_t j=0;j<nodes->size();j++)
    {
      int nodeID = j+1;

      if(binaryFile) {
      // Binary format

        gmshfile.write( (char *)&nodeID, sizeof(int) );

        for(int k=0;k<no_coords;k++)
          gmshfile.write( (char *)&(*partX)[j*no_coords+k], sizeof(double) );

        for(int k=no_coords; k<3; k++)
          {
            double zeroFloat=0;
            gmshfile.write( (char *)&zeroFloat, sizeof(double) );
          }
      } else {
        // ASCII format

        gmshfile << nodeID;
        for(int k=0;k<no_coords;k++)
          gmshfile << " " << (*partX)[j*no_coords+k];

        for(int k=no_coords; k<3; k++)
          gmshfile << " 0";

        gmshfile << "\n";
      }

    }


  if (binaryFile) gmshfile.write( (char *)&newlineChar, 1 );
  gmshfile << "$EndNodes\n";


  // GMSH element section,
  // which includes both faces and regular elements

  int totalElements = elements->size() + partSENList->size()/snloc;

  gmshfile << "$Elements\n";
  gmshfile << totalElements << "\n";

  // Write out faces

  int nfacets = partSENList->size()/snloc;
  int numFaceTags;

  // boundary IDs ?
  // (GMSH format supports more element tags, but not needed yet)

  if (partBoundaryIds->size()>0 )
    numFaceTags=1;
  else
    numFaceTags=0;

  // Print out faces

  if(binaryFile)
    {
      gmshfile.write( (char *)&faceType, sizeof(int) );
      gmshfile.write( (char *)&nfacets, sizeof(int) );
      gmshfile.write( (char *)&numFaceTags, sizeof(int) );
    }

  // For each face: ID, boundaryID (if present), vertex node IDs
  for(int i=0;i<nfacets;i++)
    {
      int faceID=i+1;

      if(binaryFile) {
        // Binary format

        gmshfile.write( (char *)&faceID, sizeof(int) );

        if(numFaceTags>0)
          gmshfile.write( (char *)&(*partBoundaryIds)[i], sizeof(int) );

        for(int j=0;j<snloc;j++)
          gmshfile.write( (char *)&(*partSENList)[i*snloc+j], sizeof(int) );

      } else {
        // ASCII format

        gmshfile << faceID << " " << faceType << " "
                 << numFaceTags;
        if(numFaceTags>0)
          gmshfile << " " << (*partBoundaryIds)[i];

        for(int j=0;j<snloc;j++)
          gmshfile << " " << (*partSENList)[i*snloc+j];

        gmshfile << "\n";
      }
    }



  // Regular elements now

  int nElems = elements->size();
  int numElemTags=0;
  if(partRegionIds->size()) numElemTags=1;

  if(binaryFile)
    {
      gmshfile.write( (char *)&normElemType, sizeof(int) );
      gmshfile.write( (char *)&nElems, sizeof(int) );
      gmshfile.write( (char *)&numElemTags, sizeof(int) );
    }


  for(int i=0;i<nElems;i++)
    {
      // After the faces, so have to offset the ID counter
      int elementID=i+nfacets+1;

      if(binaryFile) {

        gmshfile.write( (char *)&elementID, sizeof(int) );
        if( numElemTags>0 )
          gmshfile.write( (char *)&(*partRegionIds)[i], sizeof(int) );
        for(int j=0;j<nloc;j++)
          gmshfile.write( (char *)&(*partENList)[i*nloc+j], sizeof(int) );

      } else {
        gmshfile << elementID << " " << normElemType << " "
                 << numElemTags;

        if( numElemTags>0 )
          gmshfile << " " << (*partRegionIds)[i];

        for(int j=0;j<nloc;j++)
          gmshfile << " " << (*partENList)[i*nloc+j];

        gmshfile << "\n";
      }
    }


  gmshfile << "\n$EndElements\n";
  gmshfile.close();
}




// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/* Write out partitions and halos for all partitions */


void write_partitions_gmsh(bool verbose,
                           string filename, string file_format,
                           const int nparts, const int nnodes,
                           const int dim, const int no_coords,
                           const vector<double>&x, const vector<int>& decomp,
                           const int nloc, const vector<int>& ENList,
                           const vector<int>& regionIds,
                           const int snloc, const deque <vector<int> >& SENList,
                           const vector<int>& boundaryIds,
                           int normElemType, int faceType )
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
    cout<<"void write_partitions_gmsh( ... )";

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
    for(int part=0; part<nparts; part++)
      {
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
        for(int eid=0;eid<nelms;eid++)
          {
            int halo_count=0;
            pair<int, int> *owned_elm = new pair<int, int>(decomp[ENList[eid*nloc] - 1], eid);
            if(decomp[ENList[eid*nloc] - 1]!=part)
              halo_count++;

            for(int j=1;j<nloc;j++)
              {
                owned_elm->first = min(owned_elm->first, decomp[ENList[eid*nloc+j] - 1]);
                if(decomp[ENList[eid*nloc+j] - 1]!=part){
                  halo_count++;
                }
              }

            if(halo_count<nloc){
              sorted_elements->push_back(*owned_elm);
              if(halo_count>0){

                for(int j=0;j<nloc;j++)
                  {
                    int nid = ENList[eid*nloc+j] - 1;
                    if(decomp[nid]!=part)
                      {
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

        // Write out GMSH mesh file for this partition
        write_part_main_mesh( verbose, filename, part,
                              partX,
                              nodes,
                              no_coords,
                              nloc,
                              elements,
                              partENList,
                              partRegionIds,
                              snloc, partSENList,
                              partBoundaryIds,
                              normElemType, faceType );

        delete partNodesToEid;

        delete nodes;
        delete partX;

        delete partSENList;
        delete partBoundaryIds;

        delete elements;
        delete partENList;

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


/* Read in GMSH header from open file.
   This is more format error checking than anything.
*/



void read_gmsh_header(fstream &gmshfile)
{
  int errorCode=1;

  // Prelimary format checking

  string beginTag;
  float gmshVersion;
  int gmshFormat, doubleFloatLen;

  gmshfile >> beginTag >> gmshVersion >> gmshFormat >> doubleFloatLen;


  if( beginTag != "$MeshFormat" ) {
    cerr << "Can't find $MeshFormat tag\n";
    exit(errorCode);
  }

  if( gmshVersion < 2 || gmshVersion >= 3 ) {
    cerr << "Currently only GMSH format v2.x is supported\n";
    exit(errorCode);
  }

  if( doubleFloatLen != sizeof(double) ) {
    cerr << "Double float size in GMSH file differs from system double\n";
    exit(errorCode);
  }

  // For performance reasons only binary GMSH files are permitted with fldecomp
  if( gmshFormat == 0 ) {
    cerr << "** GMSH ASCII files are verboten:\n"
         << "** please use 'gmsh -bin' to generate binary format.\n";
    exit(errorCode);
  }

  // Skip newline character
  char newLineChar[2];
  gmshfile.read( newLineChar, 1 );

  // Read in integer '1' written in binary format: check for endianness
  int oneInt;
  gmshfile.read( (char *)&oneInt, sizeof(int) );
  gmshfile.read( newLineChar, 1 );


  if( oneInt != 1 ){
    cerr << "** Oh dear: internal and file binary formats for integers differ (endianness?)\n";
    exit(errorCode);
  }


  char sectionTag[80];
  gmshfile.getline( sectionTag, sizeof("$EndMeshFormat") );

  // Check for end of formatting section
  if( string(sectionTag) != "$EndMeshFormat" ) {
    cerr << "Can't find $EndMeshFormat tag\n";
    exit(errorCode);
  }
}




// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/* Read in mesh section from an already open GMSH file */


int read_gmsh_nodes(fstream &gmshfile, vector<int> &nodeIDs, vector<double> &x)
{
  int errorCode=1;
  char sectionTag[80];
  char newLineChar[2]="\n";

  // Start of nodes section
  gmshfile.getline( sectionTag, sizeof("$Nodes") );

  if( string(sectionTag) != "$Nodes" ) {
    cerr << "Can't find $Nodes tag\n";
    exit(errorCode);
  }

  int numNodes;
  gmshfile >> numNodes;
  gmshfile.read( newLineChar, 1 );

  // Read in binary node data
  nodeIDs.resize(numNodes);
  x.resize(numNodes*3);
  //vector<int> surface_nids;

  int n;
  for(n=0; n<numNodes; n++)
    {
      gmshfile.read( (char *)&nodeIDs[n], sizeof(int) );
      gmshfile.read( (char *)&x[n*3], sizeof(double) );
      gmshfile.read( (char *)&x[n*3+1], sizeof(double) );
      gmshfile.read( (char *)&x[n*3+2], sizeof(double) );
    }
  gmshfile.read( newLineChar, 1 );


  // End of nodes section
  gmshfile.getline( sectionTag, sizeof("$EndNodes") );

  if( string(sectionTag) != "$EndNodes" )
    {
      cerr << "Can't find $EndNodes tag\n";
      exit(errorCode);
    }

  return numNodes;
}





// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/* Read in elements section of element file,
   returning a pointer to an array of Element objects */




Element *read_gmsh_elements(fstream &gmshfile,
                            int &numDimen,
                            int &numElements, int &elemType, int &nloc,
                            int &numFaces, int &faceType, int &snloc)
{
  int errorCode=1;
  char sectionTag[80];

  char newLineChar[2]="\n";

  // Elements section
  gmshfile.getline( sectionTag, sizeof("$Elements") );
  int totalElements;

  if( string(sectionTag) != "$Elements" ){
    cerr << "Can't find $Elements tag\n";
    exit(errorCode);
  }

  gmshfile >> totalElements;

  gmshfile.read( newLineChar, 1 );

  int eIx=0;

  // Element array
  Element *gmshElements = new Element[totalElements];

  // Read in all elements, including faces
  int numEdges=0, numTriangles=0, numQuads=0, numTets=0, numHexes=0;

  while( eIx < totalElements )
    {
      int numBlockElems, numTags;

      gmshfile.read( (char *)&elemType, sizeof(int) );
      gmshfile.read( (char *)&numBlockElems, sizeof(int) );
      gmshfile.read( (char *)&numTags, sizeof(int) );

      // Do we know about this type of element?
      if( elemNumNodes[elemType] == -1 )
        {
          cerr << "Element type not supported by fldecomp\n";
          exit(errorCode);
        }

      // Read in all the elements of this type
      for(int e=0; e<numBlockElems; e++)
        {
          gmshElements[eIx].read(&gmshfile, elemType,
                                 elemNumNodes[elemType], numTags);

          // Here, we count up the number of different types of elements.
          // This allows us to decide what are faces, and what are internal
          // elements.
          switch( gmshElements[eIx].type )
            {
            case 1:
              numEdges++;
              break;
            case 2:
              numTriangles++;
              break;
            case 3:
              numQuads++;
              break;
            case 4:
              numTets++;
              break;
            case 5:
              numHexes++;
              break;
            case 15:
              break;
            default:
              cerr << "Unsupported element type in GMSH mesh\n";
              cerr << "type: "<< gmshElements[eIx].type << "\n";
              exit(errorCode);
              break;
            }

          eIx++;
        }
    }

  gmshfile.read( newLineChar, 1 );

  // End of Elements section
  gmshfile.getline( sectionTag, sizeof("$EndElements") );


  if( string(sectionTag) != "$EndElements" ) {
      cerr << "Can't find $EndElements tag\n";
      exit(errorCode);
    }

  // Make some calculations based on the different types of elements
  // collected.

  if(numTets>0) {
    numFaces = numTriangles;
    faceType = 2;
    elemType = 4;
    numDimen = 3;

  } else if(numTriangles>0) {
    numFaces = numEdges;
    faceType = 1;
    elemType = 2;
    numDimen = 2;

  } else if(numHexes>0) {
    numFaces = numQuads;
    faceType = 3;
    elemType= 5;
    numDimen = 3;

  } else if(numQuads>0) {
    numFaces = numEdges;
    faceType = 1;
    elemType = 3;
    numDimen = 2;

  } else {
    cerr << "Unsupported mixture of face/element types\n";
    exit(errorCode);

  }

  // Set some handy variables to be used elsewhere
  numElements = totalElements-numFaces;

  nloc = elemNumNodes[elemType];
  snloc = elemNumNodes[faceType];

  // Return list of GMSH elements (eg. normal elements and faces)
  return gmshElements;
}





// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/*
  Main routine for GMSH mesh decomposition, handling calls to IO routines, etc.

  Called from main()
*/



int decomp_gmsh( map<char, string> flArgs, bool verbose,
                 string filename, string file_format,
                 int nparts, int ncores )
{
  int errorCode=1;

  // base name + file extension
  string lfilename = filename+".msh";

  fstream gmshfile;

  // Open the GMSH file.
  gmshfile.open(lfilename.c_str(), ios::in | ios::binary);
  if(!gmshfile.is_open()){
    cerr<<"ERROR: GMSH file, "<< lfilename
        <<", cannot be opened. Does it exist? Have you read permission?\n";
    exit(-1);
  }


  // Read in data from GMSH file, eg. node and element data.
  vector<int> nodeIDs;
  vector<double> x;
  int numDimen, numElements, elemType, nloc, numFaces, faceType, snloc;

  read_gmsh_header(gmshfile);
  int numNodes = read_gmsh_nodes(gmshfile, nodeIDs, x);
  Element *gmshElements = read_gmsh_elements(gmshfile, numDimen, numElements, elemType, nloc, numFaces, faceType, snloc);
  // Every GMSH mesh is 3D, with a zeroed z-coordinate. We just keep this
  // to 3D (for now) to keep the parition routines etc. happy.
  numDimen=3;


  // Close the GMSH file.
  gmshfile.close();





  vector<int> ENList, regionIds;
  ENList.resize(numElements*nloc);
  regionIds.resize(numElements);

  deque< vector<int> > SENList;
  vector<int> topSENList;

  vector<int> boundaryIds;

  vector<int> facet(snloc);
  SENList.resize(numFaces);
  boundaryIds.resize(numFaces);


  // Loop around gmshElement[], loading into data structures that
  // external functions such as partition() understand.
  int elepos=0, enlistpos=0, facepos=0;
  for(int g=0; g<numFaces+numElements; g++)
    {
      // If we have a regular element
      if(gmshElements[g].type==elemType)
        {
          for(int j=0; j<nloc; j++)
            ENList[enlistpos++] = gmshElements[g].nodeIDs[j];

          regionIds[elepos] = gmshElements[g].physicalID;
          elepos++;

        } else if(gmshElements[g].type==faceType)
        {
          // This is a face

          for(int j=0; j<snloc; j++)
            facet[j] = gmshElements[g].nodeIDs[j];

          SENList[facepos] = facet;
          boundaryIds[facepos] = gmshElements[g].physicalID;

          facepos++;
        } else {
        // default case
        cerr << "fldecomp GMSH support only works one type of mesh element.\n";
        exit(errorCode);
      }
    }



  // Preparing to partition the mesh.

  vector<int> decomp;
  int partition_method = -1;

  if(flArgs.count('r')){
    partition_method = 0; // METIS PartGraphRecursive
    if(flArgs.count('k'))cerr<<"WARNING: should not specify both -k and -r. Choosing -r.\n";
  }
  if(flArgs.count('k')) partition_method = 1; // METIS PartGraphKway


  vector<int> npartitions;
  if(ncores>1){
    npartitions.push_back(nparts/ncores);
    npartitions.push_back(ncores);
  }else{
    npartitions.push_back(nparts);
  }

  int edgecut=0;



  // Partition the mesh. Generates a map "decomp" from node number
  // (numbered from zero) to partition number (numbered from
  // zero).

  edgecut = partition( ENList, numDimen, nloc, numNodes, npartitions,
                       partition_method, decomp );

  if(flArgs.count('d')){
    cout<<"Edge-cut: "<<edgecut<<endl;
  }

  // Process the partitioning
  if(verbose)
    cout<<"Processing the mesh partitions\n";

  int no_coords=numDimen;


  // Write partition mesh files and halo out

  write_partitions_gmsh( verbose, filename, file_format,
                         nparts, numNodes, numDimen, no_coords,
                         x, decomp,
                         nloc, ENList, regionIds,
                         snloc, SENList, boundaryIds,
                         elemType, faceType );

  return(0);

}

