#ifndef _MAT_UTILS_H
#define _MAT_UTILS_H

#include "op_lib_core.h"

void addto_scalar(op_mat mat, const void *value, int row, int col);
void addto_vector(op_mat mat, const void* values, int nrows,
                  const int *irows, int ncols, const int *icols);
void assemble_mat(op_mat mat);

#endif // _MAT_UTILS_H
