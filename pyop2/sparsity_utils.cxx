#include <vector>
#include <set>
#include <stdlib.h>

#include "sparsity_utils.h"

void build_sparsity_pattern_seq ( int rmult, int cmult, int nrows, int nmaps,
                                  cmap * rowmaps, cmap * colmaps,
                                  int ** _nnz, int ** _rowptr, int ** _colidx,
                                  int * _nz )
{
  // Create and populate auxiliary data structure: for each element of
  // the from set, for each row pointed to by the row map, add all
  // columns pointed to by the col map
  int lsize = nrows*rmult;
  std::vector< std::set< int > > s_diag(lsize);

  for ( int m = 0; m < nmaps; m++ ) {
    cmap rowmap = rowmaps[m];
    cmap colmap = colmaps[m];
    int rsize = rowmap.from_size;
    for ( int e = 0; e < rsize; ++e ) {
      for ( int i = 0; i < rowmap.arity; ++i ) {
        for ( int r = 0; r < rmult; r++ ) {
          int row = rmult * rowmap.values[i + e*rowmap.arity] + r;
          for ( int d = 0; d < colmap.arity; d++ ) {
            for ( int c = 0; c < cmult; c++ ) {
              s_diag[row].insert(cmult * colmap.values[d + e * colmap.arity] + c);
            }
          }
        }
      }
    }
  }

  // Create final sparsity structure
  int * nnz = (int*)malloc(lsize * sizeof(int));
  int * rowptr = (int*)malloc((lsize+1) * sizeof(int));
  rowptr[0] = 0;
  for ( int row = 0; row < lsize; ++row ) {
    nnz[row] = s_diag[row].size();
    rowptr[row+1] = rowptr[row] + nnz[row];
  }
  int * colidx = (int*)malloc(rowptr[lsize] * sizeof(int));
  // Note: elements in a set are always sorted, so no need to sort colidx
  for ( int row = 0; row < lsize; ++row ) {
    std::copy(s_diag[row].begin(), s_diag[row].end(), colidx + rowptr[row]);
  }
  *_nz = rowptr[lsize];
  *_nnz = nnz;
  *_rowptr = rowptr;
  *_colidx = colidx;
}

void build_sparsity_pattern_mpi ( int rmult, int cmult, int nrows, int nmaps,
                                  cmap * rowmaps, cmap * colmaps,
                                  int ** _d_nnz, int ** _o_nnz,
                                  int * _d_nz, int * _o_nz )
{
  // Create and populate auxiliary data structure: for each element of
  // the from set, for each row pointed to by the row map, add all
  // columns pointed to by the col map
  int lsize = nrows*rmult;
  std::vector< std::set< int > > s_diag(lsize);
  std::vector< std::set< int > > s_odiag(lsize);

  for ( int m = 0; m < nmaps; m++ ) {
    cmap rowmap = rowmaps[m];
    cmap colmap = colmaps[m];
    int rsize = rowmap.from_exec_size;
    for ( int e = 0; e < rsize; ++e ) {
      for ( int i = 0; i < rowmap.arity; ++i ) {
        for ( int r = 0; r < rmult; r++ ) {
          int row = rmult * rowmap.values[i + e*rowmap.arity] + r;
          // NOTE: this hides errors due to invalid map entries
          if ( row < lsize ) { // ignore values inside the MPI halo region
            for ( int d = 0; d < colmap.arity; d++ ) {
              for ( int c = 0; c < cmult; c++ ) {
                int entry = cmult * colmap.values[d + e * colmap.arity] + c;
                if ( entry < lsize ) {
                  s_diag[row].insert(entry);
                } else {
                  s_odiag[row].insert(entry);
                }
              }
            }
          }
        }
      }
    }
  }

  // Create final sparsity structure
  int * d_nnz = (int*)malloc(lsize * sizeof(int));
  int * o_nnz = (int *)malloc(lsize * sizeof(int));
  int d_nz = 0, o_nz = 0;
  for ( int row = 0; row < lsize; ++row ) {
    d_nnz[row] = s_diag[row].size();
    d_nz += d_nnz[row];
    o_nnz[row] = s_odiag[row].size();
    o_nz += o_nnz[row];
  }
  *_d_nnz = d_nnz;
  *_o_nnz = o_nnz;
  *_d_nz = d_nz;
  *_o_nz = o_nz;
}
