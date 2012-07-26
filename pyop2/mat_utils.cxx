#include "op_lib_mat.h"
#include "mat_utils.h"

void addto_scalar(op_mat mat, const void *value, int row, int col)
{
  op_mat_addto_scalar(mat, value, row, col);
}

void assemble_mat(op_mat mat)
{
    op_mat_assemble(mat);
}
