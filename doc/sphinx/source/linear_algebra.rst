.. _linear_algebra:

PyOP2 Linear Algebra Interface
==============================

PyOP2 supports linear algebra operations on sparse matrices using a thin
wrapper around the PETSc_ library harnessed via its petsc4py_ interface.

As described in :doc:`concepts`, a sparse matrix is a linear operator that
maps a :class:`~pyop2.DataSet` representing its row space to a
:class:`~pyop2.DataSet` representing its column space and vice versa. These
two spaces are commonly the same, in which case the resulting matrix is
square. A sparse matrix is represented by a :class:`~pyop2.Mat`, which is
declared on a :class:`~pyop2.Sparsity`, representing its non-zero structure.

.. _matrix_storage:

Sparse Matrix Storage Formats
-----------------------------

PETSc_ uses the popular Compressed Sparse Row (CSR) format to only store the
non-zero entries of a sparse matrix. In CSR, a matrix is stored as three
one-dimensional arrays of *row pointers*, *column indices* and *values*, where
the two former are of integer type and the latter of float type, usually
double. As the name suggests, non-zero entries are stored per row, where each
non-zero is defined by a pair of column index and corresponding value. The
column indices and values arrays therefore have a length equal to the total
number of non-zero entries. Row indices are given implicitly by the row
pointer array, which contains the starting index in the column index and
values arrays for the non-zero entries of each row. In other words, the
non-zeros for row ``i`` are at positions ``row_ptr[i]`` up to but not
including ``row_ptr[i+1]`` in the column index and values arrays. For each
row, entries are sorted by column index to allow for faster lookups using a
binary search.

.. figure:: images/csr.svg

  *A sparse matrix and its corresponding CSR row pointer, column indices and
  values arrays*

For distributed parallel storage with MPI, the rows of the matrix are
distribued evenly among the processors. Each row is then again divided into a
*diagonal* and an *off-diagonal* part, where the diagonal part comprises
columns ``i`` to ``j`` if ``i`` and ``j`` are the first and last row owned by
a given processor, and the off-diagonal part all other rows.

.. _matrix_assembly:

Matrix assembly
---------------

Sparse matrices are assembled by adding up local contributions which are
mapped to global matrix entries via a local-to-global mapping represented by a
pair of :class:`Maps <pyop2.Map>` for the row and column space. 

For each :func:`~pyop2.par_loop` that assembles a matrix, PyOP2 generates a
call to PETSc_'s MatSetValues_ function for each element of the iteration set,
adding the local contributions computed by the user kernel to the global
matrix using the given :class:`Maps <pyop2.Map>`. At the end of the
:func:`~pyop2.par_loop` PyOP2 automatically calls MatAssemblyBegin_ and
MatAssemblyEnd_ to finalise matrix assembly.

.. _sparsity_pattern:

Building a sparsity pattern
---------------------------

The sparsity pattern of a matrix is uniquely defined by the dimensions of the
:class:`DataSets <pyop2.DataSet>` forming its row and column space, and one or
more pairs of :class:`Maps <pyop2.Map>` defining its non-zero structure. This
is exploited in PyOP2 by caching sparsity patterns with these unique
attributes as the cache key to save expensive recomputation. Whenever a
:class:`Sparsity` is initialised, an already computed pattern with the same
unique key is returned if it exists.

For a valid sparsity, each row :class:`~pyop2.Map` must map to the set of the
row :class:`~pyop2.DataSet`, each column :class:`~pyop2.Map` to that of the
column :class:`~pyop2.DataSet` and the from sets of each pair must match. A
matrix on a sparsity pattern built from more than one pair of maps is
assembled by multiple parallel loops iterating over the corresponding
iteration set for each pair.

Sparsity construction proceeds by iterating each :class:`~pyop2.Map` pair and
building a set of indices of the non-zero columns for each row. Each pair of
entries in the row and column maps gives the row and column index of a
non-zero entry in the matrix and therefore the column index is added to the
set of non-zero entries for that particular row. The array of non-zero entries
per row is then determined as the size of the set for each row and its
exclusive scan yields the row pointer array. The column index array is the
concatenation of all the sets. An algorithm for the sequential case is given
below: ::

  for rowmap, colmap in maps:
    for e in range(rowmap.from_size):
      for i in range(rowmap.arity):
        for r in range(rowdim):
          row = rowdim * rowmap.values[i + e*rowmap.arity] + r
          for d in range(colmap.arity):
            for c in range(coldim):
              diag[row].insert(coldim * colmap.values[d + e * colmap.arity] + c)

For the MPI parallel case a minor modification is required, since for each row
a set of diagonal and off-diagonal column indices needs to be built as
described in :ref:`matrix_storage`: ::

  for rowmap, colmap in maps:
    for e in range(rowmap.from_size):
      for i in range(rowmap.arity):
        for r in range(rowdim):
          row = rowdim * rowmap.values[i + e*rowmap.arity] + r
          if row < nrows * rowdim:
            for d in range(colmap.arity):
              for c in range(coldim):
                col = coldim * (colmap.values[d + e*colmap.arity]) + c
                if col < ncols * coldim:
                    diag[row].insert(col)
                else:
                    odiag[row].insert(col)

.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _petsc4py: http://pythonhosted.org/petsc4py/
.. _MatSetValues: http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MatSetValues.html
.. _MatAssemblyBegin: http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MatAssemblyBegin.html
.. _MatAssemblyEnd: http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MatAssemblyEnd.html
