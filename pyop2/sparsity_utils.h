#ifndef _SPARSITY_UTILS_H
#define _SPARSITY_UTILS_H

typedef struct
{
  int  from_size,
       from_exec_size,
       to_size,
       to_exec_size,
       arity,    /* dimension of pointer */
      *values; /* array defining pointer */
} cmap;

#ifdef __cplusplus
extern "C" {
#endif

void build_sparsity_pattern_seq ( int rmult, int cmult, int nrows, int nmaps,
                                  cmap * rowmaps, cmap * colmaps,
                                  int ** nnz, int ** rowptr, int ** colidx,
                                  int * nz );

void build_sparsity_pattern_mpi ( int rmult, int cmult, int nrows, int nmaps,
                                  cmap * rowmaps, cmap * colmaps,
                                  int ** d_nnz, int ** o_nnz,
                                  int * d_nz, int * o_nz );

#ifdef __cplusplus
}
#endif

#endif // _SPARSITY_UTILS_H
