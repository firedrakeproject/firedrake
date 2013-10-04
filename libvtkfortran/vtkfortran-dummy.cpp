/* Copyright (C) 2004-2006 by Gerard Gorman
   Copyright (C) 2006- Imperial College London and others.

   Please see the AUTHORS file in the main source directory for a full list
   of copyright holders.

   Dr Gerard J Gorman
   Applied Modelling and Computation Group
   Department of Earth Science and Engineering
   Imperial College London

   g.gorman@imperial.ac.uk

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

extern "C"{
  void vtkopen(char *outName, int *len1, char *vtkTitle, int *len2){}
  void vtkwritemesh(int *NNodes, int *NElems,
		       float *x, float *y, float *z,
		       int *enlist, int *elementTypes, int *elementSizes){}
  void vtkwritemeshd(int *NNodes, int *NElems,
			double *x, double *y, double *z,
			int *enlist, int *elementTypes, int *elementSizes){}
  void vtkstartn(){}
  void vtkwriteghostlevels(int *ghost_levels){}
  void vtkwriteisn(int *vect, char *name, int *len){}
  void vtkwritefsn(float *vect, char *name, int *len){}
  void vtkwritedsn(double *vect, char *name, int *len){}
  void vtkwritefvn(float *vx, float *vy, float *vz,
		      char *name, int *len){}
  void vtkwritedvn(double *vx, double *vy, double *vz,
		      char *name, int *len){}
  void vtkwriteftn(float *v1, float *v2, float *v3,
          float *v4, float *v5, float *v6,
          float *v7, float *v8, float *v9,
          char *name, int *len){}
  void vtkwriteftc(float *v1, float *v2, float *v3,
          float *v4, float *v5, float *v6,
          float *v7, float *v8, float *v9,
          char *name, int *len){}

  void vtkwritedtn(double *v1, double *v2, double *v3,
          double *v4, double *v5, double *v6,
          double *v7, double *v8, double *v9,
          char *name, int *len){}
  void vtkwritedtc(double *v1, double *v2, double *v3,
          double *v4, double *v5, double *v6,
          double *v7, double *v8, double *v9,
          char *name, int *len){}

  void vtkwriteftn2(float *v1, float *v2, float *v3,
		       float *v4, float *v5, float *v6,
		       char *name, int *len){}

  void vtkstartc(){}
  void vtkwriteisc(int *vect, char *name, int *len){}
  void vtkwritefsc(float *vect, char *name, int *len){}
  void vtkwritedsc(double *vect, char *name, int *len){}
  void vtkwritefvc(float *vx, float *vy, float *vz,
          char *name, int *len){}
  void vtkwritedvc(double *vx, double *vy, double *vz,
          char *name, int *len){}
  void vtkclose(){}
  void vtkpclose(int *rank, int *npartitions){}
  void vtksetactivescalars(char* name){}
  void vtksetactivevectors(char* name){}
  void vtksetactivetensors(char* name){}

}
