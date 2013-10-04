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

#ifndef VTKIO_H
#define VTKIO_H

#include "confdefs.h"

typedef struct _Field_Info Field_Info;

// allocate a Field_Info record using length of name string + size:
// afield = (Field_Info *)malloc(strlen(name)+sizeof(Field_Info))
struct _Field_Info
{
    int          ncomponents;
    int          identifier;
    float        interperr; // >0 means adapt+interpolate this field
                            // =0 means interpolate only
                            // <0 means ignore this field
    float        cutoff; // >0 means use rel. Hessian, with abs. cut-off
                         // =0 means use absolute Hessian
                         // <0 means use rel. Hessian, with rel. cut-off
    Field_Info   *next;
    char         name; // rest of record can be arbitrary length
};

int readVTKFile(const char * const filename,
                int *NumNodes, int *NumElms,
                int *NumFields, int *NumProps,
                int *szENLs, int *ndim,
                Field_Info *fieldlst,
                REAL **X, REAL **Y, REAL **Z,
                int **ENLBas, int **ENList,
                REAL **Fields, REAL **Properties,
                int onlyinfo, int onlytets );

int writeVTKFile(const char * const filename,
                 int NumNodes, int NumElms,
                 int NumFields, int NumProps, int ndim,
                 Field_Info *fieldlst,
                 REAL *X, REAL *Y, REAL *Z,
                 int *ENLBas, int *ENList,
                 REAL *Fields, REAL *Properties );

extern "C" {

int fgetvtksizes_fc(char *filename, int *namelen,
                  int *NNOD, int *NELM, int *SZENLS,
                  int *NFIELD, int *NPROP,
                  int *NDIM, int *maxlen );

int freadvtkfile_fc(char *filename, int *namelen,
                  int *NNOD, int *NELM, int *SZENLS,
                  int *NFIELD, int *NPROP, int *NDIM,
                  REAL *X, REAL *Y, REAL *Z,
                  REAL *FIELDS, REAL *PROPS,
                  int *ENLBAS, int *ENLIST,
                  char *NAMES, int *maxlen );

int fwritevtkfile_fc(char *filename, int *namelen,
                   int *NNOD, int *NELM, int *SZENLS,
                   int *NFIELD, int *NPROP, int *NDIM,
                   REAL *X, REAL *Y, REAL *Z,
                   REAL *FIELDS, REAL *PROPS,
                   int *ENLBAS, int *ENLIST,
                   char *NAMES, int *maxlen );
}

#endif
