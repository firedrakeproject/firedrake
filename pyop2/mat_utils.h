#ifndef _MAT_UTILS_H
#define _MAT_UTILS_H

#include <petscmat.h>

#include "op_lib_core.h"

void addto_scalar(Mat mat, const void *value, int row, int col);
void addto_vector(Mat mat, const void* values, int nrows,
                  const int *irows, int ncols, const int *icols);
void assemble_mat(Mat mat);

#endif // _MAT_UTILS_H
