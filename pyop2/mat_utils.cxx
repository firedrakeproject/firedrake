#include <assert.h>
#include "mat_utils.h"

void addto_scalar(Mat mat, const void *value, int row, int col)
{
  assert( mat && value);
  // FIMXE: this assumes we're getting a double
  const PetscScalar * v = (const PetscScalar *)value;

  if ( v[0] == 0.0 ) return;
  MatSetValues( mat,
                1, (const PetscInt *)&row,
                1, (const PetscInt *)&col,
                v, ADD_VALUES );
}

void addto_vector(Mat mat, const void *values,
                  int nrows, const int *irows,
                  int ncols, const int *icols)
{
  assert( mat && values && irows && icols );
  // FIMXE: this assumes we're getting a double
  MatSetValues( mat,
                nrows, (const PetscInt *)irows,
                ncols, (const PetscInt *)icols,
                (const PetscScalar *)values, ADD_VALUES);
}

void assemble_mat(Mat mat)
{
  assert( mat );
  MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
}
