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

.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _petsc4py: http://pythonhosted.org/petsc4py/
.. _MatSetValues: http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MatSetValues.html
.. _MatAssemblyBegin: http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MatAssemblyBegin.html
.. _MatAssemblyEnd: http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MatAssemblyEnd.html
