#ifndef _SPARSITY_UTILS_H
#define _SPARSITY_UTILS_H

#include "op_lib_core.h"

#ifdef __cplusplus
extern "C" {
#endif

void build_sparsity_pattern ( int rmult, int cmult, int nrows, int nmaps,
                              op_map * rowmaps, op_map * colmaps,
                              int ** d_nnz, int ** o_nnz,
                              int ** rowptr, int ** colidx );

#ifdef __cplusplus
}
#endif

#endif // _SPARSITY_UTILS_H
