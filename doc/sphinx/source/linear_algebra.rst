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

.. _PETSc: http://www.mcs.anl.gov/petsc/
.. _petsc4py: http://pythonhosted.org/petsc4py/
