#ifndef _MAT_UTILS_H
#define _MAT_UTILS_H

#include <petscmat.h>
#include <assert.h>

static inline void addto_scalar(Mat mat, const void *value, int row, int col, int insert)
{
  assert( mat && value);
  // FIMXE: this assumes we're getting a PetscScalar
  const PetscScalar * v = (const PetscScalar *)value;

  if ( v[0] == 0.0 && !insert ) return;
  MatSetValuesLocal( mat,
                1, (const PetscInt *)&row,
                1, (const PetscInt *)&col,
                v, insert ? INSERT_VALUES : ADD_VALUES );
}

static inline void addto_vector(Mat mat, const void *values,
                  int nrows, const int *irows,
                  int ncols, const int *icols, int insert)
{
  assert( mat && values && irows && icols );
  // FIMXE: this assumes we're getting a PetscScalar
  MatSetValuesLocal( mat,
                nrows, (const PetscInt *)irows,
                ncols, (const PetscInt *)icols,
                (const PetscScalar *)values,
                insert ? INSERT_VALUES : ADD_VALUES );
}

#endif // _MAT_UTILS_H
