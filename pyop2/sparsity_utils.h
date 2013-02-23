#ifndef _SPARSITY_UTILS_H
#define _SPARSITY_UTILS_H

#include "op_lib_core.h"

#ifdef __cplusplus
extern "C" {
#endif

void build_sparsity_pattern_seq ( int rmult, int cmult, int nrows, int nmaps,
                                  op_map * rowmaps, op_map * colmaps,
                                  int ** nnz, int ** rowptr, int ** colidx,
                                  int * nz );

void build_sparsity_pattern_mpi ( int rmult, int cmult, int nrows, int nmaps,
                                  op_map * rowmaps, op_map * colmaps,
                                  int ** d_nnz, int ** o_nnz,
                                  int * d_nz, int * o_nz );

#ifdef __cplusplus
}
#endif

#endif // _SPARSITY_UTILS_H
