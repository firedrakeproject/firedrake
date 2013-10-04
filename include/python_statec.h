#include "confdefs.h"
#include "spud.h"

#ifdef HAVE_PYTHON
#include "Python.h"
#endif
#ifdef HAVE_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL fluidity
// only python_statec.c (that does init) should allow import
#ifndef ALLOW_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#include "numpy/arrayobject.h"
#endif

void python_init_(void);  // Initialize
void python_end_(void);   // Finalize
void python_reset_(void); // Clear the dictionary
void init_vars(void);

void python_add_statec_(char *name,int *len); // Add a new state object to the Python environment, if a state with the same name already exists it will overwrite that state; also the last added state will be accessible as 'state', all others in the 'states' dictionary

void python_add_csr_matrix_real_(int *valSize, double val[], int *col_indSize, int col_ind [], int *row_ptrSize,
                                 int row_ptr [], char *name, int *namelen, char *state, int *statelen, int *numCols);
void python_add_csr_matrix_integer_(int *valSize, int ival[], int *col_indSize, int col_ind [], int *row_ptrSize,
                                    int row_ptr [], char *name, int *namelen, char *state, int *statelen, int *numCols);

void python_add_scalar_(int *sx,double x[],char *name,int *nlen, int *field_type, char *option_path, int *oplen, char *state,int *slen,char*,int*,int*);  // Add a new scalar field to the state with name *state
void python_add_vector_(int *num_dim, int *s,
  double x[],
  char *name,int *nlen, int *field_type, char *option_path, int *oplen, char *state,int *slen,char*,int*,int*); // Add a new vector field to the state with name *state
void python_add_tensor_(int *sx,int *sy,int *sz, double *x, int *num_dim,
  char *name,int *nlen, int *field_type, char *option_path, int *oplen, char *state,int *slen,char*,int*,int*); // Add a new tensor field to the state with name *state

void python_add_halo_(char *name, int *name_len, int *nprocs, int *unn_offset,
                      char *state_name, int *state_name_len, int *comm,
                      int *uid);
void python_add_mesh_(int*,int*,int*, int*, int*,int*,
  char*,int*,char*,int*,char*,int*,int*,int*,int*,char*,int*,
  char *, int *, char *, int *,int*);

// Procedures to add an element
void python_add_element_(int *dim, int *loc, int *ngi, int *degree,
  char *state_name, int *state_name_len, char *mesh_name, int *mesh_name_len,
  double *n,int *nx, int *ny, double *dn, int *dnx, int *dny, int *dnz,
  int *size_spoly_x,int *size_spoly_y,int *size_dspoly_x,int *size_dspoly_y,
  char* family_name, int* family_name_len,
  char* type_name, int* type_name_len,
  double* coords, int* size_coords_x, int* size_coords_y);
void python_add_quadrature_(int *dim,int *degree,int *loc, int *ngi,
  double *weight, int *weight_size, double *locations, int *l_size, int *is_surfacequadr);
void python_add_superconvergence_(int *nsp, double *l, int *lx, int *ly,
  double *n, int *nx, int *ny,
  double *dn, int *dnx, int *dny, int *dnz);
void python_add_polynomial_(double*,int*,int*,int*,int*,int*);

char* fix_string(char*,int);
void python_run_stringc_(char* s, int *slen, int *stat); // Run an arbitrary string in the Python interpreter
void python_run_filec_(char *f,int *flen, int *stat);  // Run a file in the Python interpreter

void python_add_array_double_1d(double *arr, int *size, char *name);
void python_add_array_double_2d(double *arr, int *sizex, int *sizey, char *name);
void python_add_array_double_3d(double *arr, int *sizex, int *sizey, int *sizez, char *name);

void python_add_array_integer_1d(int *arr, int *size, char *name);
void python_add_array_integer_2d(int *arr, int *sizex, int *sizey, char *name);
void python_add_array_integer_3d(int *arr, int *sizex, int *sizey, int *sizez, char *name);

int get_global_debug_level_(void);
