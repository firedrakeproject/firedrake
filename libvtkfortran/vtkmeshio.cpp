/* Copyright (C) 2004- Imperial College London and others.

   Please see the AUTHORS file in the main source directory for a full
   list of copyright holders.

   Adrian Umpleby
   Applied Modelling and Computation Group
   Department of Earth Science and Engineering
   Imperial College London

   adrian@imperial.ac.uk

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
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

#include "confdefs.h"

#ifdef DOUBLEP
#define REAL double
#else
#define REAL float
#endif

#include <iostream>

#ifdef HAVE_VTK
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <vtk.h>

#include "vtkmeshio.h"


using namespace std;

void AddToFront(vtkIdType *front, vtkIdType i)
{
  static int frend;
  if(front==NULL) {
    frend = 0;
    return;
  }
  frend++;
  front[frend] = i;
  //cerr<<"added element "<<i<<" to front\n";
}

vtkIdType NextFrontCell(vtkIdType *front)
{
  static int frpos;
  if(front==NULL) {
    frpos = 0;
    return 0;
  }
  return front[frpos++];
}

vtkIdType *InitFront(vtkIdType n)
{
  vtkIdType *front=(vtkIdType *)malloc(n*sizeof(vtkIdType));
  if(front) {
    front[0] = 0;
    NextFrontCell(NULL);
    AddToFront(NULL,0);
  }
  return front;
}

char FaceCutType(vtkUnstructuredGrid *dataSet, int *hexcut,
                        vtkIdType i,vtkIdList *fcpts)
{
    vtkIdType npts;
    vtkIdType *pts;
    int locpos[4];
    int fcid=0,maxpos=0,d1=0,d2=0;
    dataSet->GetCellPoints(i,npts,pts);
    assert(npts==8);
    for(int j=0;j<4;j++) {
      locpos[j]=999;
      for(int k=0;k<8;k++) {
        if(fcpts->GetId(j)==pts[k]) {
          locpos[j]=k;
          if(k>maxpos) maxpos=k;
          break;
        }
      }
    }
    fcid = locpos[0]+locpos[1]+locpos[2]+locpos[3];
    if( fcid==6 ) {
      fcid = 0;
      if(hexcut[i*6]==1) {
        d1 = 0;
        d2 = 2;
      } else {
        assert(hexcut[i*6]==2);
        d1 = 1;
        d2 = 3;
      }
    } else if( fcid==22 ) {
      fcid = 1;
      if(hexcut[i*6+1]==1) {
        d1 = 4;
        d2 = 6;
      } else {
        assert(hexcut[i*6+1]==2);
        d1 = 5;
        d2 = 7;
      }
    } else if( fcid==10 ) {
      fcid = 4;
      if(hexcut[i*6+4]==1) {
        d1 = 1;
        d2 = 4;
      } else {
        assert(hexcut[i*6+4]==2);
        d1 = 5;
        d2 = 0;
      }
    } else if( fcid==18 ) {
      fcid = 5;
      if(hexcut[i*6+5]==1) {
        d1 = 2;
        d2 = 7;
      } else {
        assert(hexcut[i*6+5]==2);
        d1 = 6;
        d2 = 3;
      }
    } else if( fcid==14 ) {
      if( maxpos==7 ) {
        fcid = 3;
        if(hexcut[i*6+3]==1) {
          d1 = 0;
          d2 = 7;
        } else {
          assert(hexcut[i*6+3]==2);
          d1 = 3;
          d2 = 4;
        }
      } else if(maxpos==6) {
        fcid = 2;
        if(hexcut[i*6+2]==1) {
          d1 = 1;
          d2 = 6;
        } else {
          assert(hexcut[i*6+2]==2);
          d1 = 2;
          d2 = 5;
        }
      }
    } else
      fcid = fcid+100;
    if( fcid>5 || hexcut[i*6+fcid]>2 ) {
      //if(fcid>5)
        //printf("Can't find face for element ",i," ",fcid,"\n";)
      //else
        //printf("EH?? ",i," ",fcid," ",(int)hexcut[i*6+fcid},"\n";
//      cerr<<pts[0]<<" "<<pts[1]<<" "<<pts[2]<<" "<<pts[3]<<" "<<pts[4]<<" "<<pts[5]<<" "<<pts[6]<<" "<<pts[7]<<endl;
//      cerr<<fcpts->GetId(0)<<" "<<fcpts->GetId(1)<<" "<<fcpts->GetId(2)<<" "<<fcpts->GetId(3)<<endl;
      return 0;
    }
    assert(hexcut[i*6+fcid]>0);
    assert(hexcut[i*6+fcid]<3);
    if(locpos[0]==d1 && locpos[2]==d2) return 1;
    if(locpos[0]==d2 && locpos[2]==d1) return 1;
    if(locpos[1]==d1 && locpos[3]==d2) return 2;
    if(locpos[1]==d2 && locpos[3]==d1) return 2;
//    cerr<<"Failed FaceCutType for element "<<i<<" "<<fcid<<" "<<d1<<" "<<d2<<endl;
//    cerr<<pts[0]<<" "<<pts[1]<<" "<<pts[2]<<" "<<pts[3]<<" "<<pts[4]<<" "<<pts[5]<<" "<<pts[6]<<" "<<pts[7]<<endl;
//    cerr<<fcpts->GetId(0)<<" "<<fcpts->GetId(1)<<" "<<fcpts->GetId(2)<<" "<<fcpts->GetId(3)<<endl;
//    cerr<<locpos[0]<<" "<<locpos[1]<<" "<<locpos[2]<<" "<<locpos[3]<<endl;
    return 0;
}

void CheckConnected(vtkUnstructuredGrid *dataSet, int *hexcut, const int kk,
                        vtkIdType *front, vtkIdType i, vtkIdList *fc1pts, vtkIdList *fc2pts)
{
      const int k = kk%10;
      const int f = kk/10;
      vtkIdType n1=0,n2=0;
      vtkIdList *nghbrs1=vtkIdList::New();
      vtkIdList *nghbrs2=vtkIdList::New();
      nghbrs1->SetNumberOfIds(0);
      nghbrs2->SetNumberOfIds(0);

      hexcut[i*6+k] = 0;

      if( f & 1 ) {
      dataSet->GetCellNeighbors(i,fc1pts,nghbrs1);
      n1 = nghbrs1->GetNumberOfIds();
      if(n1>0) {
        vtkIdType j = nghbrs1->GetId(0);
        if(hexcut[j*6]==0) {
          AddToFront(front,j);
          hexcut[j*6] = 100;
        } else if(hexcut[j*6]<100) {
          hexcut[i*6+k] = FaceCutType(dataSet,hexcut,j,fc1pts);
          assert(hexcut[i*6+k]!=0);
        }
      }
      assert(n1<2);
      }

      hexcut[i*6+k+1] = 0;
      if( f & 2 ) {
      dataSet->GetCellNeighbors(i,fc2pts,nghbrs2);
      n2 = nghbrs2->GetNumberOfIds();
      //if(n1+n2<=0) {
      //  cerr<<"No neighbours for element "<<i<<" "<<k<<endl;
      //  cerr<<fc1pts->GetId(0)<<" "<<fc1pts->GetId(1)<<" "<<fc1pts->GetId(2)<<" "<<fc1pts->GetId(3)<<endl;
      //  cerr<<fc2pts->GetId(0)<<" "<<fc2pts->GetId(1)<<" "<<fc2pts->GetId(2)<<" "<<fc2pts->GetId(3)<<endl;
      //  cerr<<fc1pts->GetNumberOfIds()<<" "<<fc2pts->GetNumberOfIds()<<endl;
      //}
      //assert(n1+n2>0);
      assert(n2<2);
      }

      if(n2>0) {
        vtkIdType j = nghbrs2->GetId(0);
        if(hexcut[j*6]==0) {
          AddToFront(front,j);
          hexcut[j*6] = 100;
        } else if(hexcut[j*6]<100) {
          hexcut[i*6+k+1] = FaceCutType(dataSet,hexcut,j,fc2pts);
          assert(hexcut[i*6+k+1]!=0);
        }
      }

      if( hexcut[i*6+k]==0 ) {
        if( hexcut[i*6+k+1]==0 ) {
          hexcut[i*6+k  ] = 1;
          hexcut[i*6+k+1] = 1;
        } else
          hexcut[i*6+k] = hexcut[i*6+k+1];
      } else if( hexcut[i*6+k+1]==0 )
        hexcut[i*6+k+1] = hexcut[i*6+k];
      else if( hexcut[i*6+k]!=hexcut[i*6+k+1] ) {
        //printf("OOPS! ",i," ",k," ",(int)hexcut[i*6+k]," ",(int)hexcut[i*6+k+1],"\n";
        //cerr<<fc1pts->GetId(0)<<" "<<fc1pts->GetId(1)<<" "<<fc1pts->GetId(2)<<" "<<fc1pts->GetId(3)<<endl;
        //cerr<<fc2pts->GetId(0)<<" "<<fc2pts->GetId(1)<<" "<<fc2pts->GetId(2)<<" "<<fc2pts->GetId(3)<<endl;
      }
}

REAL CheckTetVol(vtkIdType p1, vtkIdType p2, vtkIdType p3, vtkIdType p4,
                 REAL *X, REAL *Y, REAL *Z)
{
  double vol=0.0;

  if( p1==p2 ) return 0.0;
  if( p1==p3 ) return 0.0;
  if( p1==p4 ) return 0.0;
  if( p2==p3 ) return 0.0;
  if( p2==p4 ) return 0.0;
  if( p3==p4 ) return 0.0;

  {
    double r0[3], r1[3], r2[3], r3[3];

    r0[0] = X[p1];
    r0[1] = Y[p1];
    r0[2] = Z[p1];
    r1[0] = X[p2];
    r1[1] = Y[p2];
    r1[2] = Z[p2];
    r2[0] = X[p3];
    r2[1] = Y[p3];
    r2[2] = Z[p3];
    r3[0] = X[p4];
    r3[1] = Y[p4];
    r3[2] = Z[p4];

    double a1 = r1[0]-r0[0];
    double a2 = r1[1]-r0[1];
    double a3 = r1[2]-r0[2];
    double d10 = a1*a1 + a2*a2 + a3*a3;
    double dmin = d10;

    double b1 = r2[0]-r0[0];
    double b2 = r2[1]-r0[1];
    double b3 = r2[2]-r0[2];
    double d20 = b1*b1 + b2*b2 + b3*b3;
    if( d20 < dmin ) dmin = d20;

    double c1 = r3[0]-r0[0];
    double c2 = r3[1]-r0[1];
    double c3 = r3[2]-r0[2];
    double d30 = c1*c1 + c2*c2 + c3*c3;
    if( d30 < dmin ) dmin = d30;

    double d1 = r2[0]-r1[0];
    double d2 = r2[1]-r1[1];
    double d3 = r2[2]-r1[2];
    double d21 = d1*d1 + d2*d2 + d3*d3;
    if( d21 < dmin ) dmin = d21;

    double e1 = r3[0]-r1[0];
    double e2 = r3[1]-r1[1];
    double e3 = r3[2]-r1[2];
    double d31 = e1*e1 + e2*e2 + e3*e3;
    if( d31 < dmin ) dmin = d31;

    double f1 = r3[0]-r2[0];
    double f2 = r3[1]-r2[1];
    double f3 = r3[2]-r2[2];
    double d32 = f1*f1 + f2*f2 + f3*f3;
    if( d32 < dmin ) dmin = d32;

    dmin = dmin*sqrt(dmin)/50000;

    // volume = | r_a r_b r_c | / 6
    vol = (a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2))/6.0;

    if(fabs(vol)<=fabs(dmin)) {
      cerr<<"Found very small volume "<<vol<<" "<<d10<<" "<<d20<<" "<<d30<<" "<<d21<<" "<<d31<<" "<<d32<<" "<<dmin<<"\n";
      //printf("Found very small elem ",vol," ",d10," ",d20," ",d30," ",d21," ",d31," ",d32," ",dmin,"\n";
      //return 0.0;
    }
    //if(vol<-dmin) {
      //cerr<<"Found inside-out elem "<<vol<<" "<<d10<<" "<<d20<<" "<<d30<<" "<<d21<<" "<<d31<<" "<<d32<<" "<<dmin<<"\n";
      //return vol;
    //}
    //cerr<<"Found fine elem "<<vol<<" "<<d10<<" "<<d20<<" "<<d30<<" "<<d21<<" "<<d31<<" "<<d32<<" "<<dmin<<"\n";
  }

  return (REAL) vol;
}

int AddOneLine(vtkIdType *pts, int cellcnt, int *ENLBas, int *ENList,
               REAL *X, REAL *Y, REAL *Z )
{
  if( ENLBas != NULL && ENList != NULL ) {
    int ibas = ENLBas[cellcnt];
    ENList[ibas  ] = pts[0]+1;
    ENList[ibas+1] = pts[1]+1;
    ENLBas[cellcnt+1] = ibas + 2;
  }
  return cellcnt+1;
}

int AddOneAnything(vtkIdType *pts, int cellcnt, int *ENLBas, int *ENList,
               REAL *X, REAL *Y, REAL *Z, int nloc)
// general version only works for elements that don't need reordering
{
  if( ENLBas != NULL && ENList != NULL ) {
    int ibas = ENLBas[cellcnt];
    for(int i=0; i<nloc; i++)
      ENList[ibas+i] = pts[i]+1;
    ENLBas[cellcnt+1] = ibas + nloc;
  }
  return cellcnt+1;
}

int AddOneTri(vtkIdType *pts, int cellcnt, int *ENLBas, int *ENList,
              REAL *X, REAL *Y, REAL *Z )
{
  if( ENLBas != NULL && ENList != NULL ) {
    int ibas = ENLBas[cellcnt];
    for(int i=0; i<3; i++)
      ENList[ibas+i] = pts[i]+1;
    ENLBas[cellcnt+1] = ibas + 3;
  }
  return cellcnt+1;
}

int AddOneQuad(vtkIdType *pts, int cellcnt, int *ENLBas, int *ENList,
               REAL *X, REAL *Y, REAL *Z )
{
  if( ENLBas != NULL && ENList != NULL ) {
    int ibas = ENLBas[cellcnt];
    // FIXME!! -is this node ordering correct?
    ENList[ibas  ] = pts[0]+1;
    ENList[ibas+1] = pts[1]+1;
    ENList[ibas+2] = pts[3]+1;
    ENList[ibas+3] = pts[2]+1;
    ENLBas[cellcnt+1] = ibas + 4;
  }
  return cellcnt+1;
}

int AddOneHexa(vtkIdType *pts, int cellcnt, int *ENLBas, int *ENList,
               REAL *X, REAL *Y, REAL *Z )
{
  if( ENLBas != NULL && ENList != NULL ) {
    int ibas = ENLBas[cellcnt];
    // FIXME!! -is this node ordering correct?
    for(int i=0; i<8; i++)
      ENList[ibas+i] = pts[i]+1;
    ENList[ibas+2] = pts[3]+1;
    ENList[ibas+3] = pts[2]+1;
    ENList[ibas+6] = pts[7]+1;
    ENList[ibas+7] = pts[6]+1;
    ENLBas[cellcnt+1] = ibas + 8;
  }
  return cellcnt+1;
}

int AddOneWedge(vtkIdType *pts, int cellcnt, int *ENLBas, int *ENList,
                REAL *X, REAL *Y, REAL *Z )
{
  if( ENLBas != NULL && ENList != NULL ) {
    int ibas = ENLBas[cellcnt];
    // FIXME!! -the node ordering is probably not correct
    for(int i=0; i<6; i++)
      ENList[ibas+i] = pts[i]+1;
    ENLBas[cellcnt+1] = ibas + 6;
  }
  return cellcnt+1;
}

int AddOneTetra(vtkIdType p1, vtkIdType p2, vtkIdType p3, vtkIdType p4,
             int tetcnt, int *ENLBas, int *ENList, REAL *X, REAL *Y, REAL *Z )
{
  static int iocnt=0;
  static REAL minvol=0.0, maxvol=0.0;
  REAL ort=0.0;

  // special case to return inside-out count and set counter to zero
  if( p1==0 && p2==0 && p3==0 && p4==0 && tetcnt==0 ) {
    ort = (REAL) iocnt;
    iocnt = 0;
    minvol = 0.0;
    maxvol = 0.0;
    return (int) ort;
  }

  if( p1==p2 || p1==p3 || p1==p4 || p2==p3 || p2==p4 || p3==p4 ) {
    cerr<<"Found collapsed element: "<<p1<<" "<<p2<<" "<<p3<<" "<<p4<<"\n";
    return tetcnt;
  }
  ort = CheckTetVol(p1,p2,p3,p4,X,Y,Z);
  if( ort==0.0 ) return tetcnt;
  if( ort>0.0 ) {
    if( ENList != NULL && ENLBas != NULL ) {
      int ibas = ENLBas[tetcnt];
      ENList[ibas  ] = p1+1;
      ENList[ibas+1] = p2+1;
      ENList[ibas+2] = p3+1;
      ENList[ibas+3] = p4+1;
      ENLBas[tetcnt+1] = ibas+4;
    }
    if( ort>maxvol ) maxvol = ort;
    if( ort<minvol ) minvol = ort;
  } else {
    if( ENList != NULL && ENLBas != NULL ) {
      int ibas = ENLBas[tetcnt];
      ENList[ibas  ] = p2+1;
      ENList[ibas+1] = p1+1;
      ENList[ibas+2] = p3+1;
      ENList[ibas+3] = p4+1;
      ENLBas[tetcnt+1] = ibas+4;
    }
    iocnt++;
    if( -ort>maxvol ) maxvol = -ort;
    if( -ort<minvol ) minvol = -ort;
  }
  return tetcnt+1;
}

int readVTKFile(const char * const filename,
                int *NumNodes, int *NumElms,
                int *NumFields, int *NumProps,
                int *szENLs, int *ndim,
                Field_Info *fieldlst,
                REAL **X, REAL **Y, REAL **Z,
                int **ENLBas, int **ENList,
                REAL **Fields, REAL **Properties,
                int onlyinfo, int onlytets )
{
  vtkDataSetReader *read1=NULL;
  vtkXMLUnstructuredGridReader *read2=NULL;
  vtkXMLPUnstructuredGridReader *read3=NULL;
  vtkUnstructuredGrid *dataSet=NULL;
  //vtkDataSet *dataSet=NULL;
  Field_Info *lastfld=NULL;
  int addall = 0, filetype = -1;
  char typnam[30];

  if( sizeof(float) != 4 ) {
    cerr<<"ERROR: Float has wrong size: "<<sizeof(float)<<endl;
    return -1;
  }

  if( sizeof(double) != 8 ) {
    cerr<<"ERROR: Double has wrong size: "<<sizeof(double)<<endl;
    return -1;
  }

  if(strlen(filename)<5) {
    cerr<<"ERROR: Got bad filename: "<<filename<<endl;
    return -1;
  }
  const char * const ext = filename + strlen(filename) - 4;

  if( strncmp(ext,".vtk",4)==0 ) {
    //printf("Reading from VTK file: %s\n",filename);
    read1 = vtkDataSetReader::New();
    if( read1==NULL ) {
      cerr<<"ERROR: Failed to read!\n";
      return -1;
    }
    read1->SetFileName(filename);
    filetype = read1->ReadOutputType();
    if( filetype == VTK_POLY_DATA )
      sprintf(typnam,"vtkPolyData");
    else if( filetype == VTK_STRUCTURED_POINTS )
      sprintf(typnam,"vtkStructuredPoints");
    else if( filetype == VTK_STRUCTURED_GRID )
      sprintf(typnam,"vtkStructuredGrid");
    else if( filetype == VTK_RECTILINEAR_GRID )
      sprintf(typnam,"vtkRectilinearGrid");
    else if( filetype == VTK_UNSTRUCTURED_GRID )
      sprintf(typnam,"YES"); // length is 3 - see later comment
    else if( filetype == VTK_PIECEWISE_FUNCTION )
      sprintf(typnam,"vtkPiecewiseFunction");
    else if( filetype == VTK_IMAGE_DATA )
      sprintf(typnam,"vtkImageData");
    else if( filetype == VTK_DATA_OBJECT )
      sprintf(typnam,"vtkDataObject");
    else if( filetype == VTK_DATA_SET )
      sprintf(typnam,"vtkDataSet");
    else if( filetype == VTK_POINT_SET )
      sprintf(typnam,"vtkPointSet");
    else if( filetype == -1 ) {
      cerr<<"ERROR: Cannot read file - does it exist?\n";
      return -1;
    } else {
      cerr<<"ERROR: File contains unknown data type number "
          <<filetype<<" must be vktUnstructuredGrid\n";
      return -1;
    }
    if( strlen(typnam) != 3 ) { // length 3 was used above for VTK_UNSTRUCTURED_GRID
      cerr<<"ERROR: Cannot read file containing "<<typnam<<" must be vktUnstructuredGrid\n";
      return -1;
    }else
      dataSet = read1->GetUnstructuredGridOutput();
  } else if( strncmp(ext,".vtu",4)==0 ) {
    //printf("Reading from VTK XML file: %s\n",filename);
    read2 = vtkXMLUnstructuredGridReader::New();
    if( read2==NULL ) {
      cerr<<"ERROR: Failed to read!\n";
      return -1;
    }
    read2->SetFileName(filename);
    //reader->SetScalarsName("temp");
    dataSet = read2->GetOutput();
  }else{
    const char * const pext = filename + strlen(filename) - 5;
    if( strncmp(pext,".pvtu",5)==0 ) {
      //printf("Reading from VTK XML parallel file: %s\n",filename);
      read3 = vtkXMLPUnstructuredGridReader::New();
      if( read3==NULL ) {
        cerr<<"ERROR: Failed to read!\n";
        return -1;
      }
      read3->SetFileName(filename);
      //reader->SetScalarsName("temp");
      dataSet = read3->GetOutput();
    }else{
      cerr<<"ERROR: Filename: "<<filename<<endl
          <<"Don't know what this file is (should end in .vtk, .vtu or .pvtu)\n";
      return -1;
    }
  }

  if(dataSet==NULL) {
    cerr<<"ERROR: Unstructured Grid data not found in file!\n";
    return -1;
  }

  if( fieldlst==NULL ) {
    cerr<<"ERROR: Empty field list sent into readVTKFile!\n";
    return -1;
  }

  dataSet->Update();
  vtkIdType nnodes = dataSet->GetNumberOfPoints();
  if( nnodes==0 ) {
    cerr<<"ERROR: Something went wrong (got no nodes) - aborting\n";
    return -1;
  }

  vtkIdType ncells = dataSet->GetNumberOfCells();
  if( ncells==0 ) {
    cerr<<"ERROR: Something went wrong (got no cells) - aborting\n";
    return -1;
  }

  if( fieldlst->ncomponents < 0 ) {
    lastfld = fieldlst;
    addall = 1;
  }

  vtkIdType ntets = ncells;
  int numfld=0, numprops=0;

  {
    vtkIdType npts;
    vtkIdType *pts;
    dataSet->GetCellPoints(0,npts,pts);
    if(npts==8 && onlytets!=0) ntets = 6*ncells;
  }


  vtkPointData *flds = dataSet->GetPointData();
//  cerr<<*flds<<endl;
//  cerr<<";) #arrays = "<<flds->GetNumberOfTuples()<<endl;
  vtkDataArray* T = flds->GetArray(0);

  if( T == NULL ) {
//    cerr<<"ERROR: No field arrays in VTK file\n";
    numfld=0;
  } else {
    int i=0;
    while( T != NULL ) {
      T = flds->GetArray(i);
      if( T == NULL )
        0;
      else {
        int k=T->GetNumberOfComponents();
        if(T->GetName()==NULL) {
          0;
        } else {
          unsigned int l = strlen(T->GetName());
          if( addall==0 ) {
            Field_Info *newfld=fieldlst, *gotfld=NULL;
            while( newfld!=NULL ) {
              if( strlen(&(newfld->name))==l ) {
                if( strncmp(&(newfld->name),T->GetName(),l)==0 ) {
                  gotfld = newfld;
                  gotfld->ncomponents = k;
                  gotfld->identifier = i;
                  newfld = NULL;
                  numfld += k;
                }
              }
              if(newfld!=NULL) newfld = newfld->next;
            }
            if(gotfld==NULL)
              //printf(" (not in user's field list)\n");
              0;
            else if( gotfld->interperr>0.0 )
              //printf("\n");
              0;
              //cerr<<" (adapt err = "<<gotfld->interperr<<")\n";
            else if( gotfld->interperr==0.0 )
              //printf("\n");
              0;
              //cerr<<" (only for output)\n";
          } else {
            Field_Info *newfld=(Field_Info *)malloc(l+sizeof(Field_Info));
            assert(newfld!=NULL);
            strcpy(&(newfld->name),T->GetName());
            newfld->interperr = 1.0;
            newfld->ncomponents = k;
            newfld->next = NULL;
            newfld->identifier = i;
            //printf("\n");
            lastfld->next = newfld;
            lastfld = newfld;
            numfld += k;
          }
        }
      }
      i++;
    }
  }
  //printf("Total no. of field components: %d\n",numfld);

  vtkCellData *props = dataSet->GetCellData();
//  cerr<<*flds<<endl;
//  cerr<<";) #arrays = "<<flds->GetNumberOfTuples()<<endl;
  vtkDataArray* P = props->GetArray(0);

  if( P == NULL ) {
    //printf("--- No cell property arrays in VTK file\n");
    numprops=0;
  } else {
    int i=0;
    while( P != NULL ) {
      //printf("Trying to get cell property array %d...",i);
      P = props->GetArray(i);
      if( P == NULL )
        //printf("does not exist\n");
        0;
      else {
        int k=P->GetNumberOfComponents();
        //cerr<<"  type: "<<T->GetDataType();
        //printf("  name: '%s'",P->GetName());
        //printf("  components: %d",k);
        if(P->GetName()==NULL) {
          //printf("\n");
          0;
        } else {
          unsigned int l = strlen(P->GetName());
          if( addall==0 ) {
            Field_Info *newfld=fieldlst, *gotfld=NULL;
            while( newfld!=NULL ) {
              if( strlen(&(newfld->name))==l ) {
                if( strncmp(&(newfld->name),P->GetName(),l)==0 ) {
                  gotfld = newfld;
                  gotfld->ncomponents = k;
                  gotfld->identifier = i;
                  newfld = NULL;
                  numprops += k;
                }
              }
              if(newfld!=NULL) newfld = newfld->next;
            }
            if(gotfld==NULL)
              //printf(" (not in user's field list)\n");
              0;
            else if( gotfld->interperr>0.0 )
              //printf("\n");
              0;
              //cerr<<" (adapt err = "<<gotfld->interperr<<")\n";
            else if( gotfld->interperr==0.0 )
              //printf("\n");
              0;
              //cerr<<" (only for output)\n";
          } else {
            Field_Info *newfld=(Field_Info *)malloc(l+sizeof(Field_Info));
            assert(newfld!=NULL);
            strcpy(&(newfld->name),P->GetName());
            newfld->interperr = 1.0;
            newfld->ncomponents = k;
            newfld->next = NULL;
            newfld->identifier = i;
            //printf("\n");
            lastfld->next = newfld;
            lastfld = newfld;
            numprops += k;
          }
        }
      }
      i++;
    }
  }
//  printf("Total no. of cell property components: %d\n",numprops);
  *NumProps = numprops;

  // Get rid of unused nodes and find dimension of problem
  {
    REAL *NODX=NULL, *NODY=NULL, *NODZ=NULL, *NODF=NULL, *ELMP=NULL;
    int nodcnt, curdim=0, cellcnt=0;
    int *used=(int *) malloc(nnodes*sizeof(int));
    assert(used!=NULL);
    if( *ndim == 0 ) {
//      printf("1D element types: LINE=%d\n",VTK_LINE);
//      printf("2D element types: TRI=%d  QUAD=%d\n",VTK_TRIANGLE,VTK_QUAD);
//      printf("3D element types: TET=%d  WEDGE=%d  HEX=%d\n",VTK_TETRA,VTK_WEDGE,VTK_HEXAHEDRON);
      for(vtkIdType i=0; i<ncells; i++) {
        vtkIdType npts=0, ct;
        vtkIdType *pts;
        ct = dataSet->GetCellType(i);
        dataSet->GetCellPoints(i,npts,pts);
        if( npts == 2 ) {
          if( curdim < 1 ) curdim = 1;
          if( ct != VTK_LINE ) {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
            //dataSet->SetCellType(i,VTK_LINE);
          }
        } else if( npts == 3 ) {
          if( curdim < 2 ) curdim = 2;
          if( ct != VTK_TRIANGLE ) {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
            //dataSet->SetCellType(i,VTK_TRIANGLE);
          }
        } else if( npts == 4 ) {
          if( curdim < 2 ) {
            if( ct == VTK_QUAD )
              curdim = 2;
            else if( ct == VTK_TETRA )
              curdim = 3;
            else
              cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
          } else if( curdim == 2 ) {
            if( ct != VTK_QUAD ) {
              cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
             // dataSet->SetCellType(i,VTK_QUAD);
            }
          } else if( curdim == 3 ) {
            if( ct != VTK_TETRA ) {
              cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
              //dataSet->SetCellType(i,VTK_TETRA);
            }
          }
        } else if( npts == 6 ) {
          if( ct == VTK_WEDGE ) {
            curdim = 3;
          } else if( ct == VTK_QUADRATIC_TRIANGLE ){
            curdim = 2;
          } else {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
            //dataSet->SetCellType(i,VTK_WEDGE);
          }
        } else if( npts == 8 ) {
          curdim = 3;
          if( ct != VTK_HEXAHEDRON ) {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
          }
        } else if( npts == 9 ) {
          curdim = 2;
          if(ct != VTK_QUADRATIC_QUAD) {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
          }
        } else if( npts == 10 ) {
          curdim = 3;
          if(ct != VTK_QUADRATIC_TETRA) {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
          }
        } else if( npts == 27 ) {
          curdim = 3;
          if(ct != VTK_QUADRATIC_TETRA) {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
          }
        } else {
            cerr<<"INCONSISTENT: Element "<<i<<" type: "<<ct<<" nodes: "<<npts<<endl;
          free(used);
          return -1;
        }
      }
      *ndim = curdim;
//      printf("Found dimension of mesh: %dd\n",curdim);
    } else {
      curdim = *ndim;
//      printf("Only registering %dd cells\n",curdim);
    }
    for(int i=0; i<nnodes; i++)
      used[i] = 0;
    for(vtkIdType i=0; i<ncells; i++) {
      vtkIdType npts=0, ct;
      vtkIdType *pts;
      ct = dataSet->GetCellType(i);
      dataSet->GetCellPoints(i,npts,pts);
      if( curdim == 1 ) {
        if( npts != 2 ) npts = 0;
      } else if( curdim == 2 ) {
        if( npts < 3 || npts > 6 ) npts = 0;
      } else if( curdim == 3 ) {
        if( npts < 4 ) npts = 0;
      }
      if( npts > 0 ) {
        cellcnt++;
        for(vtkIdType j=0; j<npts; j++) {
          if(pts[j]<0 || pts[j]>=nnodes) {
            //cerr<<"Cell "<<i<<" has out-of-range node ("<<pts[j]<<")!\n";
            assert(pts[j]>=0 && pts[j]<nnodes);
          }
          used[pts[j]]++;
        }
      }
    }
//    printf("Found %d %dd cells\n",cellcnt,curdim);
    // create counter for renumbering the nodes
    nodcnt=0;
    for(int i=0; i<nnodes; i++) {
      if(used[i]>0) {
        nodcnt++;
        used[i] = nodcnt;
      }
    }

    if(nodcnt<nnodes) {
//      printf("Found %d unused nodes - removing...\n",nnodes-nodcnt);
      // renumber the cell data
      for(vtkIdType i=0; i<ncells; i++) {
        vtkIdType npts;
        vtkIdType *pts;
        dataSet->GetCellPoints(i,npts,pts);
        for(vtkIdType j=0; j<npts; j++)
          pts[j] = used[pts[j]]-1;
        dataSet->ReplaceCell(i,npts,pts);
      }
    }
    {
      // allocate space for nodes, if not already provided
      if( *X == NULL )
        NODX = (REAL *) malloc(nodcnt*sizeof(REAL));
      else
        NODX = *X;
      assert(NODX!=NULL);
      if( *Y == NULL )
        NODY = (REAL *) malloc(nodcnt*sizeof(REAL));
      else
        NODY = *Y;
      assert(NODY!=NULL);
      if( *Z == NULL )
        NODZ = (REAL *) malloc(nodcnt*sizeof(REAL));
      else
        NODZ = *Z;
      assert(NODZ!=NULL);
      // extract the nodes, skipping unused ones
      for(vtkIdType i=0; i<nnodes; i++) {
        if(used[i]>0) {
// what VTK uses is independent of our choice of single or double for REAL when compiling
// older VTK used to use float, but it's quite a while back, so we'll assume double now
          double x[3];
//          float v=T->GetTuple1(i);
          dataSet->GetPoint(i,x);
          NODX[used[i]-1] =(REAL) x[0];
          NODY[used[i]-1] =(REAL) x[1];
          NODZ[used[i]-1] =(REAL) x[2];
        }
      }
      // special switch to prevent uneccessary waste of space for fields
      // when only mesh info is required
      if( onlyinfo < 2 && numfld+numprops > 0 ) {
        int fldcntr=0;
        // allocate space for fields, if not already provided
        if( numfld > 0 ) {
          if( *Fields == NULL )
            NODF = (REAL *) malloc(nodcnt*numfld*sizeof(REAL));
          else
            NODF = *Fields;
          assert(NODF!=NULL);
        }
        if( numprops > 0 ) {
          if( *Properties == NULL )
            ELMP = (REAL *) malloc(ncells*numprops*sizeof(REAL));
          else
            ELMP = *Properties;
          assert(ELMP!=NULL);
        }
        Field_Info *newfld = fieldlst;
        while(newfld!=NULL) {
          if( newfld->ncomponents>0 && newfld->identifier>=0 ) {
            if( fldcntr < numfld ) {
              T = flds->GetArray(newfld->identifier);
              assert(T!=NULL);
              assert(T->GetNumberOfComponents()==newfld->ncomponents);
              for(int j=0; j<T->GetNumberOfComponents(); j++) {
                int offset=nodcnt*fldcntr;
                fldcntr++;
                for(vtkIdType i=0; i<nnodes; i++) {
                  if(used[i]>0)
                    NODF[offset+used[i]-1] =(REAL) T->GetComponent(i,j);
                }
              }
            } else {
              P = props->GetArray(newfld->identifier);
              assert(P!=NULL);
              assert(P->GetNumberOfComponents()==newfld->ncomponents);
              for(int j=0; j<P->GetNumberOfComponents(); j++) {
                int offset=ncells*(fldcntr-numfld);
                fldcntr++;
                for(vtkIdType i=0; i<ncells; i++) {
                  ELMP[offset+i] =(REAL) P->GetComponent(i,j);
                }
              }
            }
          }
          newfld = newfld->next;
        }
        if( numfld > 0 )
          *Fields = &(NODF[0]);
        if( numprops > 0 )
          *Properties = &(ELMP[0]);
      }
      nnodes = nodcnt;

    }
    free(used);

    *X      = &(NODX[0]);
    *Y      = &(NODY[0]);
    *Z      = &(NODZ[0]);
    *NumFields = numfld;
  }

  *NumNodes = nnodes;

//  printf("Nodes in final mesh: %d\n",nnodes);

  if( ntets==ncells || *ndim!=3 ) {
    // Extract tets
    int *ENLST = NULL, *ENLBS = NULL;
    int tetcnt = 0, szenls = 0, curdim = *ndim;
    // this zeros inside-out counter - what's returned may be rubbish
    int iocnt=AddOneTetra(0,0,0,0,0,NULL,NULL,*X,*Y,*Z);
//    printf("Counting allowable %dd cells...\n",*ndim);
    for(vtkIdType i=0; i<ncells; i++){
      vtkIdType npts;
      vtkIdType *pts;
      dataSet->GetCellPoints(i,npts,pts);
      if( curdim == 3 ) {
        if( npts == 4 ) {
          tetcnt=AddOneTetra(pts[0], pts[1], pts[2], pts[3],
                             tetcnt, NULL, NULL, *X, *Y, *Z );
        } else if( npts == 6 ) {
          tetcnt=AddOneWedge(pts, tetcnt, NULL, NULL, *X, *Y, *Z );
        } else if( npts == 8 ) {
          tetcnt=AddOneHexa(pts, tetcnt, NULL, NULL, *X, *Y, *Z );
        } else {
          tetcnt=AddOneAnything(pts, tetcnt, NULL, NULL, *X, *Y, *Z, npts );
        }
      } else if( curdim == 2 ) {
        if( npts == 4 ) {
          tetcnt=AddOneQuad(pts, tetcnt, NULL, NULL, *X, *Y, *Z );
        } else if( npts == 3 ) {
          tetcnt=AddOneTri(pts, tetcnt, NULL, NULL, *X, *Y, *Z );
        } else {
          tetcnt=AddOneAnything(pts, tetcnt, NULL, NULL, *X, *Y, *Z, npts );
        }
      } else if( curdim == 1 ) {
        if( npts == 2 ) {
          tetcnt=AddOneLine(pts, tetcnt, NULL, NULL, *X, *Y, *Z );
        }
      }
      szenls += npts;
    }
//    printf("Ended up with %d cells (was %d)\n",tetcnt,ntets);
    ntets = tetcnt;
    *NumElms = ntets;
    if( onlyinfo == 0 ) {
      // Now we know how many, we can allocate the element-node list
      if( *ENLBas == NULL )
        ENLBS = (int *) malloc((ntets+1)*sizeof(int));
      else
        ENLBS = *ENLBas;
      assert(ENLBS!=NULL);
      if( *ENList == NULL )
        ENLST = (int *) malloc(szenls*sizeof(int));
      else
        ENLST = *ENList;
      assert(ENLST!=NULL);
      tetcnt = 0;
      // this zeros inside-out counter, but also returns real inside-out count
      iocnt=AddOneTetra(0,0,0,0,0,NULL,NULL,*X,*Y,*Z);
//      printf("Found %d inside-out elements\n",iocnt);
      ENLBS[0] = 0;
      for(vtkIdType i=0; i<ncells; i++){
        vtkIdType npts;
        vtkIdType *pts;
        dataSet->GetCellPoints(i,npts,pts);
        if( curdim == 3 ) {
          if( npts == 4 ) {
            tetcnt=AddOneTetra(pts[0], pts[1], pts[2], pts[3],
                               tetcnt, ENLBS, ENLST, *X, *Y, *Z );
          } else if( npts == 6 ) {
            tetcnt=AddOneWedge(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z );
          } else if( npts == 8 ) {
            tetcnt=AddOneHexa(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z );
          } else {
            tetcnt=AddOneAnything(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z, npts );
          }
        } else if( curdim == 2 ) {
          if( npts == 4 ) {
            tetcnt=AddOneQuad(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z );
          } else if( npts == 3 ) {
            tetcnt=AddOneTri(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z );
          } else {
            tetcnt=AddOneAnything(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z, npts );
          }
        } else if( curdim == 1 ) {
          if( npts == 2 ) {
            tetcnt=AddOneLine(pts, tetcnt, ENLBS, ENLST, *X, *Y, *Z );
          }
        }
        //if(i>ncells-10) cerr<<"  tried to add "<<i<<" "<<tetcnt<<"\n";
      }
      // this zeros inside-out counter, but also returns real inside-out count
      iocnt=AddOneTetra(0,0,0,0,0,NULL,NULL,*X,*Y,*Z);
      //cerr<<"Found "<<iocnt<<" inside-out elements\n";
      *ENList = &(ENLST[0]);
      *ENLBas = &(ENLBS[0]);
/*      printf("First 10 elements:\n");
      for(int i=0; i<10; i++) {
        int ibas = ENLBS[i];
        int l = ENLBS[i+1];
        printf("Nodes (%d to %d): ",ibas,l-1);
        for( int j=ibas; j<l; j++ )
          printf("%d ",ENLST[j]);
        printf("\n");
      }
      printf("Last 10 elements:\n");
      for(int i=tetcnt-10; i<tetcnt; i++) {
        int ibas = ENLBS[i];
        int l = ENLBS[i+1];
        printf("Nodes (%d to %d): ",ibas,l-1);
        for( int j=ibas; j<l; j++ )
          printf("%d ",ENLST[j]);
        printf("\n");
      }*/
    }
    *szENLs = szenls;
  } else {
    // Extract hexes, converting to tets
//    cerr<<"Hex mesh extraction not yet available\n";
//    assert(ntets=ncells);
    int *ENLST = NULL, *ENLBS = NULL;
    //int hexcut[ncells*6];
    int count[8];
    int val, tetcnt;
    int *hexcut = (int *) malloc(ncells*6*sizeof(int));
    assert(hexcut!=NULL);
    vtkIdType *front;
    vtkIdList *fc1pts=vtkIdList::New();
    vtkIdList *fc2pts=vtkIdList::New();
    fc1pts->SetNumberOfIds(4);
    fc2pts->SetNumberOfIds(4);
//    printf("Extracting hex mesh and converting to %d tets...\n",ntets);
    int npass = 2;
    if( onlyinfo != 0 ) npass = 1;
    for(int pass=0; pass<npass; pass++) {
      // this zeros inside-out counter - what's returned may be rubbish
      int iocnt=AddOneTetra(0,0,0,0,0,NULL,NULL,*X,*Y,*Z);
      tetcnt = 0;
      val = 0;
      front=InitFront(ncells);
      assert(front!=NULL);
      for(int i=0; i<8; i++)
        count[i]=0;
      for(vtkIdType k=0; k<ncells; k++)
        hexcut[k*6] = 0;
      for(vtkIdType k=0; k<ncells; k++){
        vtkIdType npts;
        vtkIdType *pts;
        vtkIdType i = NextFrontCell(front);
        dataSet->GetCellPoints(i,npts,pts);
        assert(npts==8);
        //cerr<<"Checking element "<<i<<" ";
        //cerr<<pts[0]<<" "<<pts[1]<<" "<<pts[2]<<" "<<pts[3]<<" "<<pts[4]<<" "<<pts[5]<<" "<<pts[6]<<" "<<pts[7]<<endl;
        // check faces 0123 and 4567
        fc1pts->SetId(0,pts[0]);
        fc1pts->SetId(1,pts[1]);
        fc1pts->SetId(2,pts[2]);
        fc1pts->SetId(3,pts[3]);
        fc2pts->SetId(0,pts[4]);
        fc2pts->SetId(1,pts[5]);
        fc2pts->SetId(2,pts[6]);
        fc2pts->SetId(3,pts[7]);
        CheckConnected(dataSet,hexcut,30,front,i,fc1pts,fc2pts);
        assert(hexcut[i*6]==hexcut[i*6+1]);
        // check faces 1265 and 0374
        fc1pts->SetId(0,pts[1]);
        fc1pts->SetId(1,pts[2]);
        fc1pts->SetId(2,pts[6]);
        fc1pts->SetId(3,pts[5]);
        fc2pts->SetId(0,pts[0]);
        fc2pts->SetId(1,pts[3]);
        fc2pts->SetId(2,pts[7]);
        fc2pts->SetId(3,pts[4]);
        {
          int qq=30;
          if( pts[1]==pts[5] && pts[2]==pts[6] ) qq-=10;
          if( pts[0]==pts[4] && pts[3]==pts[7] ) qq-=20;
          if( qq>0 ) CheckConnected(dataSet,hexcut,qq+2,front,i,fc1pts,fc2pts);
        }
        assert(hexcut[i*6+2]==hexcut[i*6+3]);
        // check faces 1540 and 2673
        fc1pts->SetId(0,pts[1]);
        fc1pts->SetId(1,pts[5]);
        fc1pts->SetId(2,pts[4]);
        fc1pts->SetId(3,pts[0]);
        fc2pts->SetId(0,pts[2]);
        fc2pts->SetId(1,pts[6]);
        fc2pts->SetId(2,pts[7]);
        fc2pts->SetId(3,pts[3]);
        {
          int qq=30;
          if( pts[1]==pts[5] && pts[4]==pts[0] ) qq-=10;
          if( pts[2]==pts[6] && pts[3]==pts[7] ) qq-=20;
          if( qq>0 ) CheckConnected(dataSet,hexcut,qq+4,front,i,fc1pts,fc2pts);
        }
        assert(hexcut[i*6+4]==hexcut[i*6+5]);
        val = 4*hexcut[i*6] + 2*hexcut[i*6+2] + hexcut[i*6+4] - 7;
        if(val==0) {
            // Everything is already the right way around
            tetcnt=AddOneTetra(pts[0], pts[1], pts[2], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[4], pts[1], pts[6], pts[5],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[6], pts[1], pts[4], pts[2],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[6], pts[4], pts[2],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[4], pts[0], pts[2],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[0], pts[3], pts[7], pts[2],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==3) {
            // swap face 0123 with 4567, and invert tets
            tetcnt=AddOneTetra(pts[5], pts[4], pts[6], pts[0],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[5], pts[0], pts[2], pts[1],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[5], pts[2], pts[0], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[2], pts[3], pts[0], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[0], pts[3], pts[4], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[4], pts[3], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==6) {
            // swap face 1540 with 2673, and invert tets
            tetcnt=AddOneTetra(pts[2], pts[3], pts[1], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[2], pts[7], pts[5], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[2], pts[5], pts[7], pts[1],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[5], pts[4], pts[7], pts[1],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[4], pts[3], pts[1],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[0], pts[3], pts[4], pts[1],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==5) {
            // swap face 1265 with 0374, and invert tets
            tetcnt=AddOneTetra(pts[0], pts[1], pts[3], pts[5],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[0], pts[5], pts[7], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[0], pts[7], pts[5], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[6], pts[5], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[5], pts[6], pts[1], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[2], pts[1], pts[6], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==4) {
            // Everything is already the right way around
            tetcnt=AddOneTetra(pts[0], pts[1], pts[3], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[2], pts[3], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[0], pts[4], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[4], pts[5], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[5], pts[6], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[6], pts[2], pts[7],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==2) {
            // swap face 1540 with 2673, and invert tets
            tetcnt=AddOneTetra(pts[2], pts[3], pts[0], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[2], pts[0], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[3], pts[2], pts[7], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[2], pts[6], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[6], pts[2], pts[5], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[5], pts[2], pts[1], pts[4],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==1) {
            // swap face 1265 with 0374, and invert tets
            tetcnt=AddOneTetra(pts[0], pts[1], pts[2], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[3], pts[0], pts[2], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[0], pts[5], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[5], pts[0], pts[4], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[4], pts[0], pts[7], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[7], pts[0], pts[3], pts[6],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else if(val==7) {
            // swap face 0123 with 4567, and invert tets
            tetcnt=AddOneTetra(pts[5], pts[4], pts[7], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[6], pts[5], pts[7], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[4], pts[5], pts[0], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[0], pts[5], pts[1], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[1], pts[5], pts[2], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
            tetcnt=AddOneTetra(pts[2], pts[5], pts[6], pts[3],tetcnt,ENLBS,ENLST,*X,*Y,*Z);
        } else {
          //cerr<<"val failed! "<<i<<" "<<val<<endl;
          assert(val==-999);
        }
        count[val]++;
      }
      tetcnt = 0;
      // zeros inside-out counter, but also returns real inside-out count
      iocnt=AddOneTetra(0,0,0,0,0,NULL,NULL,*X,*Y,*Z);
      if( ENLST==NULL && onlyinfo == 0 ) {
//        printf("Ended up with %d tets (was %d)\n",tetcnt,ntets);
        //cerr<<"Stats: "<<count[0]<<" "<<count[2]<<" "<<count[1]<<" "<<count[3]<<" "<<count[4]<<" "<<count[5]<<" "<<count[6]<<" "<<count[7]<<endl;
        //cerr<<"Found "<<iocnt<<" inside-out elements\n";
        if( *ENLBas == NULL )
          ENLBS = (int *) malloc((ntets+1)*sizeof(int));
        else
          ENLBS = *ENLBas;
        assert(ENLBS!=NULL);
        ENLBS[0] = 0;
        if( *ENList == NULL )
          ENLST = (int *) malloc(ntets*4*sizeof(int));
        else
          ENLST = *ENList;
        assert(ENLST!=NULL);
        free(front);
        front = NULL;
//        printf("Done first pass to find size - starting second pass...\n");
      }
    }
    free(hexcut);
    *NumElms = ntets;
    *szENLs = 4*ntets;
    if( onlyinfo == 0 ) {
      assert( ENLST != NULL );
      assert( ENLBS != NULL );
      *ENList = &(ENLST[0]);
      *ENLBas = &(ENLBS[0]);
    }
  }

  if(onlyinfo==0) {
    if(read1) read1->ReleaseDataFlagOn();
    if(read2) read2->ReleaseDataFlagOn();
    if(read3) read3->ReleaseDataFlagOn();
    dataSet->ReleaseDataFlagOn();
    dataSet->Update();
  }

  //dataSet->Delete();
  if(read1) read1->Delete();
  if(read2) read2->Delete();
  if(read3) read3->Delete();
  //if( read ) read->Delete();

  return 0;
}

int fgetvtksizes(char *fortname, int *namelen,
                  int *NNOD, int *NELM, int *SZENLS,
                  int *NFIELD, int *NPROP,
                  int *NDIM, int *maxlen )
{
  int status=0;
  int *ENLBAS=NULL, *ENLIST=NULL, *SNLIST=NULL;
  REAL *X=NULL, *Y=NULL, *Z=NULL, *F=NULL, *P=NULL;

  // the filename string passed down from Fortan needs terminating,
  // so make a copy and fiddle with it (remember to free it)
  char *filename = (char *)malloc(*namelen+3);
  memcpy( filename, fortname, *namelen );
  filename[*namelen] = 0;
//  printf("Terminated at %d: %s\n",*namelen,filename);

  // we must create an empty Field_Info record for readVTKFile,
  // so that it can append all the fields after it.
  Field_Info *fieldlst = (Field_Info *)malloc(sizeof(Field_Info));
  assert( fieldlst!=NULL );
  fieldlst->ncomponents = -1;
  fieldlst->next = NULL;

  // read VTK file, placing required info into appropriate arrays
  status = readVTKFile( filename, NNOD, NELM, NFIELD, NPROP,
                        SZENLS, NDIM, fieldlst,
                        &X, &Y, &Z, &ENLBAS, &ENLIST, &F, &P, 2, 0 );

  free( filename );

  // we remove the leading record (created before readVTKFile),
  if( fieldlst->ncomponents==-1 ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    free(newfld);
    newfld = fieldlst;
  }

  if( status ) {
    //fprintf(stderr,"*** fgetvtksizes: Got error %d from readVTKFile\n",status);
    return status;
  }

  *maxlen = 0;
  // now check lengths of field names, and free the field info list
  while( fieldlst != NULL ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    char *thisname=&(newfld->name);
    int l = strlen(thisname);
    if( newfld->ncomponents > 9 )
      l+=2;
    else if( newfld->ncomponents > 1 )
      l++;
    if( l > *maxlen )  *maxlen = l;
    free(newfld);
  }

  return status;
}

int freadvtkfile(char *fortname, int *namelen,
                  int *NNOD, int *NELM, int *SZENLS,
                  int *NFIELD, int *NPROP, int *NDIM,
                  REAL *X, REAL *Y, REAL *Z,
                  REAL *FIELDS, REAL *PROPS,
                  int *ENLBAS, int *ENLIST,
                  char *NAMES, int *maxlen )
{
  int status=0;

  // the filename string passed down from Fortan needs terminating
  // so make a copy and fiddle with it (remember to free it)
  char *filename = (char *)malloc(*namelen+3);
  memcpy( filename, fortname, *namelen );
  filename[*namelen] = 0;

  // we must create an empty Field_Info record for readVTKFile,
  // so that it can append all the fields after it.
  Field_Info *fieldlst = (Field_Info *)malloc(sizeof(Field_Info));
  assert( fieldlst!=NULL );
  fieldlst->ncomponents = -1;
  fieldlst->next = NULL;

  // read VTK file, placing required info into appropriate arrays
  status = readVTKFile( filename, NNOD, NELM, NFIELD, NPROP,
                        SZENLS, NDIM, fieldlst,
                        &X, &Y, &Z, &ENLBAS, &ENLIST, &FIELDS, &PROPS,
                        0, 0 );

  free( filename );

  // we remove the leading record (created before readVTKFile),
  if( fieldlst->ncomponents==-1 ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    free(newfld);
    newfld = fieldlst;
  }

  if( status ) {
    //fprintf(stderr,"*** freadvtkfile: Got error %d from readVTKFile\n",status);
    return status;
  }

  // now put field names into NAMES (truncating at maxlen if needed),
  // and free up the field info list
  int ipos = 0;
  while( fieldlst != NULL ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    char *thisname=&(newfld->name);
    int l = strlen(thisname), j=0;
    if( l>*maxlen )
      l = *maxlen;
    for( int k=1; k<=newfld->ncomponents; k++) {
      // copy characters into NAMES
      for( int i=0; i<l; i++ )
        NAMES[ipos+i] = thisname[i];
      j = l;
      if( newfld->ncomponents > 1 ) {
        if( j == *maxlen ) j--;
        if( newfld->ncomponents > 9 ) {
          if( j == *maxlen-1 ) j--;
          if( k > 9 ) {
            NAMES[ipos+j] = 48 + k/10;
            j++;
          }
        }
        NAMES[ipos+j] = 48 + k%10;
        j++;
      }
      // pad with spaces up to maxlen
      for( int i=j; i<*maxlen; i++ )
        NAMES[ipos+i] = 32;
      ipos += *maxlen;
    }
    free(newfld);
  }

  return status;
}
#else
#include "vtkmeshio-dummy.cpp"
#endif
