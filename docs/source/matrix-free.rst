.. only:: html

  .. contents::

=======================
 Matrix-free operators
=======================

In addition to supporting computation with the workhorse of sparse
linear algebra, an assembled sparse matrix, Firedrake also supports
computing "matrix-free".  In this case, the matrix returned from
:func:`.assemble` implements matrix-vector multiplication by the
assembly of a 1-form subject to boundary conditions rather than direct
construction of a sparse matrix ("aij" format) followed by traditional
CSR algorithms.  This functionality is documented in more detail in
:cite:`Kirby2017`.

There are two ways of accessing this functionality.  One can either
request a matrix-free operator by passing ``mat_type="matfree"`` to
:func:`.assemble`.  In this case, the returned object is an
:class:`.ImplicitMatrix`.  This object can be used in the normal way
with a :class:`.LinearSolver`.  Alternately, when solving a
variational problem, an :class:`.ImplicitMatrix` is requested through
the ``solver_parameters`` dict, by setting the option ``mat_type`` to
``matfree``.  The type of the preconditioning matrix can be controlled
separately by setting ``pmat_type``.

Generically, one can expect such a matrix to be cheaper to "assemble"
and to use less memory, especially for high-order
discretizations or complicated systems.  The downside is that
traditional algebraic preconditioners will not work with such
unassembled matrices.  To take advantage of these features, we need to
configure our solvers correctly.  To expedite this, the matrix-free
operator, implemented as a PETSc shell matrix, contains an application
context of type :class:`.ImplicitMatrixContext`.  This context
provides some important features to enabled advanced solver
configuration.

Splitting unassembled matrices
==============================

For the purposes of fieldsplit preconditioners, the PETSc matrix
object must be able to extract submatrices.  For unassembled matrices,
this is achieved through a specialized
:meth:`.ImplicitMatrixContext.getSubMatrix` method that partitions
the UFL form defining the operator into pieces corresponding to the
integer labels of the unknown fields.  This is in
contrast to the normal splitting of assembled matrices which operates
at a purely algebraic level.  With unassembled operators, the PDE
description is available in the matrix, and is therefore propagated
down to the split operators.

Preconditioning unassembled matrices
====================================

As well as providing symbolic field splitting, the
:class:`.ImplicitMatrixContext` object is available to
preconditioners.  Since it contains a complete UFL
description of the bilinear form, preconditioners can query or
manipulate it as desired.  As a particularly simple example, the class
:class:`.AssembledPC` simply passes the UFL into :func:`.assemble`
to produce an explicit matrix during set up.  It also sets up a new
PETSc PC context acting on this assembled matrix so that the user can
configure it at run-time via the options database.  This allows the
use of matrix-free actions in the Krylov solve, preconditioned using
an assembled matrix.

Providing application context to preconditioners
------------------------------------------------

Frequently, such custom preconditioners require some additional
information that will not be fully available from the UFL description
in :class:`.ImplicitMatrixContext`.  For example, it is not possible
to extract physical parameters such as the Reynolds number from a UFL
bilinear form.  In this case, the solver accepts a dictionary
``"appctx"`` as an optional keyword argument, the same argument may
also be passed to :func:`~.assemble` in the case of preassembled
solves.  Firedrake passes that down into the
:class:`.ImplicitMatrixContext` so that it is accessible to
preconditioners.

Example usage
=============

To demonstrate basic usage of matrix-free operators and
preconditioners, we show a simple Poisson problem, introducing some of
the additional solver options.

.. include:: demos/poisson.py.rst
