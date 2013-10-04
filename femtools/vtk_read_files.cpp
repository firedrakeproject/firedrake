/*  Copyright (C) 2006 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Prof. C Pain
    Applied Modelling and Computation Group
    Department of Earth Science and Engineeringp
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

#include "confdefs.h"
#include "fmangle.h"

#include <iostream>

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <vtk.h>

#include "Precision.h"
#define REAL flfloat_t
#include "vtkmeshio.h"

using namespace std;

extern "C" {

int vtk_get_sizes_fc(char *fortname, int *namelen,
  // number of nodes, elements and entries in the returned enls (ndglno):
                  int *NNOD, int *NELM, int *SZENLS,
  // total number of components over pointwise and cell-wise fields
  // i.e. no_scalar_field+ndim*no_vector_fields+ndim**2*no_tensor_fields
  // NOTE that the vtus usually contain 3 and 9 dimensional vector and tensor fields
  // even for 2 and 1 dimensional grids (so ndim here is typically 3,
  // whereas *NDIM below might be different)
                  int *nfield_components, int *nprop_components,
  // number of point-wise (nfields) and cell-wise fields (nprops):
                  int *nfields, int *nprops,
  // dimension of the vtu mesh and maximum length of
                  int *NDIM, int *maxlen );

  int vtk_read_file_fc(char *fortname, int *namelen,
  // number of nodes, elements and entries in the returned enlist:
                  int *NNOD, int *NELM, int *SZENLS,
  // total number of components over pointwise and cell-wise fields
  // i.e. no_scalar_field+ndim*no_vector_fields+ndim**2*no_tensor_fields
  // NOTE that the vtus usually contain 3 and 9 dimensional vector and tensor fields
  // even for 2 and 1 dimensional grids (so ndim here is typically 3,
  // whereas *NDIM below might be different)
                  int *nfield_components, int *nprop_components,
  // number of point-wise (nfields) and cell-wise fields (nprops):
                  int *nfields, int *nprops,
  // dimension of the vtu mesh and maximum length of field names
                  int *NDIM, int *maxlen,
  // positions field (always provide 3 dimensions!)
                  flfloat_t *X, flfloat_t *Y, flfloat_t *Z,
  // number of components for each field field_components[0:*nfields-1]
                  int *field_components,
  // number of components for each property prop_components[0:*nprops-1]
                  int *prop_components,
  // fields[0:*nfield_components,0:*nnod] point-wise field values
  // props[0:*nprop_components,0:*nelm] cell-wise field values ("properties")
                  flfloat_t *FIELDS, flfloat_t *PROPS,
  // start of each element in ENLIST: ENLBAS[0:*NELM]
  // element-node list: ENLIST[0:SZENLS-1]
                  int *ENLBAS, int *ENLIST,
  // names of point-wise fields and cell-wise fields
                  char *field_names, char *prop_names);

}


int vtk_get_sizes_fc(char *fortname, int *namelen,
  // number of nodes, elements and entries in the returned enls (ndglno):
                  int *NNOD, int *NELM, int *SZENLS,
  // total number of components over pointwise and cell-wise fields
  // i.e. no_scalar_field+ndim*no_vector_fields+ndim**2*no_tensor_fields
  // NOTE that the vtus usually contain 3 and 9 dimensional vector and tensor fields
  // even for 2 and 1 dimensional grids (so ndim here is typically 3,
  // whereas *NDIM below might be different)
                  int *nfield_components, int *nprop_components,
  // number of point-wise (nfields) and cell-wise fields (nprops):
                  int *nfields, int *nprops,
  // dimension of the vtu mesh and maximum length of
                  int *NDIM, int *maxlen )
{
#ifdef HAVE_VTK
  int status=0;
  int *ENLBAS=NULL, *ENLIST=NULL;
  flfloat_t *X=NULL, *Y=NULL, *Z=NULL, *F=NULL, *P=NULL;

  // the filename string passed down from Fortran needs terminating,
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
  status = readVTKFile( filename, NNOD, NELM, nfield_components, nprop_components,
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
    return status;
  }

  *maxlen = 0;
  int ncomponents=0;
  *nfields=0;
  *nprops=0;
  // now check lengths of field names, and free the field info list
  while( fieldlst != NULL ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    char *thisname=&(newfld->name);
    int l = strlen(thisname);
    if( l > *maxlen )  *maxlen = l;
    if (ncomponents<*nfield_components) {
      (*nfields)++;
    } else {
      (*nprops)++;
    }
    ncomponents += newfld->ncomponents;
    free(newfld);
  }

  return status;
#else
  cerr<<"ERROR: No VTK support compiled\n";
  return -1;
#endif
}

int vtk_read_file_fc(char *fortname, int *namelen,
  // number of nodes, elements and entries in the returned enlist:
                  int *NNOD, int *NELM, int *SZENLS,
  // total number of components over pointwise and cell-wise fields
  // i.e. no_scalar_field+ndim*no_vector_fields+ndim**2*no_tensor_fields
  // NOTE that the vtus usually contain 3 and 9 dimensional vector and tensor fields
  // even for 2 and 1 dimensional grids (so ndim here is typically 3,
  // whereas *NDIM below might be different)
                  int *nfield_components, int *nprop_components,
  // number of point-wise (nfields) and cell-wise fields (nprops):
                  int *nfields, int *nprops,
  // dimension of the vtu mesh and maximum length of field names
                  int *NDIM, int *maxlen,
  // positions field (always provide 3 dimensions!)
                  flfloat_t *X, flfloat_t *Y, flfloat_t *Z,
  // number of components for each field field_components[0:*nfields-1]
                  int *field_components,
  // number of components for each property prop_components[0:*nprops-1]
                  int *prop_components,
  // fields[0:*nfield_components,0:*nnod] point-wise field values
  // props[0:*nprop_components,0:*nelm] cell-wise field values ("properties")
                  flfloat_t *FIELDS, flfloat_t *PROPS,
  // start of each element in ENLIST: ENLBAS[0:*NELM]
  // element-node list: ENLIST[0:*SZENLS-1]
                  int *ENLBAS, int *ENLIST,
  // names of point-wise fields and cell-wise fields
                  char *field_names, char *prop_names)
{
#ifdef HAVE_VTK
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
  status = readVTKFile( filename, NNOD, NELM, nfield_components, nprop_components,
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
    return status;
  }

  int ncomponents=0;
  // now put field names into field_names (truncating at maxlen if needed),
  // and free up the field info list
  *nfields = 0;
  int ipos = 0;
  while( ncomponents<*nfield_components ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    char *thisname=&(newfld->name);
    int l = strlen(thisname);
    if( l>*maxlen )
      l = *maxlen;
    for( int i=0; i<l; i++ )
      field_names[ipos+i] = thisname[i];
    // pad with spaces up to maxlen
    for( int i=l; i<*maxlen; i++ )
      field_names[ipos+i] = 32;
    ipos += *maxlen;
    ncomponents += newfld->ncomponents;
    field_components[*nfields] = newfld->ncomponents;
    free(newfld);
    (*nfields)++;
  }

  // now put field names into field_names (truncating at maxlen if needed),
  // and free up the field info list
  *nprops=0;
  ipos = 0;
  while( fieldlst != NULL ) {
    Field_Info *newfld = fieldlst;
    fieldlst = newfld->next;
    char *thisname=&(newfld->name);
    int l = strlen(thisname);
    if( l>*maxlen )
      l = *maxlen;
    for( int i=0; i<l; i++ )
      prop_names[ipos+i] = thisname[i];
    // pad with spaces up to maxlen
    for( int i=l; i<*maxlen; i++ )
      prop_names[ipos+i] = 32;
    ipos += *maxlen;
    prop_components[*nprops] = newfld->ncomponents;
    (*nprops)++;
    free(newfld);
  }

  return status;
#else
  cerr<<"ERROR: No VTK support compiled\n";
  return -1;
#endif
}
