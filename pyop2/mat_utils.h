#ifndef _MAT_UTILS_H
#define _MAT_UTILS_H

#include <petscmat.h>

void addto_scalar(Mat mat, const void *value, int row, int col, int insert);
void addto_vector(Mat mat, const void* values, int nrows,
                  const int *irows, int ncols, const int *icols, int insert);

#endif // _MAT_UTILS_H
