#ifndef _MAT_UTILS_H
#define _MAT_UTILS_H

#include "op_lib_core.h"

op_mat get_mat_from_pyobj(void *o);
void addto_scalar(op_mat mat, const void *value, int row, int col);
void assemble_mat(op_mat mat);

#endif // _MAT_UTILS_H
