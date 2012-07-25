#include "op_lib_mat.h"
#include "mat_utils.h"
#include <Python.h>

typedef struct {
  PyObject_HEAD;
  op_mat _handle;
} cython_op_mat;

op_mat get_mat_from_pyobj(void *o)
{
  return ((cython_op_mat*)o)->_handle;
}

void addto_scalar(op_mat mat, const void *value, int row, int col)
{
  op_mat_addto_scalar(mat, value, row, col);
}

void assemble_mat(op_mat mat)
{
    op_mat_assemble(mat);
}
