#include "op_lib_mat.h"
#include "mat_utils.h"

void addto_scalar(op_mat mat, const void *value, int row, int col)
{
  op_mat_addto_scalar(mat, value, row, col);
}

void addto_vector(op_mat mat, const void *values,
                  int nrows, const int *irows,
                  int ncols, const int *icols)
{
    op_mat_addto(mat, values, nrows, irows, ncols, icols);
}

void assemble_mat(op_mat mat)
{
    op_mat_assemble(mat);
}
