==================================================================
 Support for matrix-free operator application and preconditioning
==================================================================

A new matrix type and modification of certain Firedrake internals now
makes it possible to perform "matrix-free" calculations.  In this
case, :function:`.assemble` can be told to produce a PETSc matrix of
"python" type that implements matrix-vector multiplication by the
assembly of a 1-form subject to boundary conditions rather than direct
construction of a sparse matrix ("aij" format) followed by traditional CSR
algorithms.

This is implemented internally by :class:`.ImplicitMatrix`, which is
a sibling class to Firedrake's :class:`.Matrix`, but which contains
a reference to the PETSc Python matrix rather than an assembled
CSR matrix by way of a PyOP2 object.

The :class:`.NonlinearVariationalSolver` now can take a parameter
type "mat_type", which is set either to "aij" (the default) or
"matfree".  This is passed into the :class:`_SNESContext` and on into
:function:`.assemble`.

Generically, one can expect such a matrix to be cheaper to "assemble"
and to use less memory, especially for high-order
discretization or complicated systems.  At the same time, until TSFC
can produce optimal sum-factored algorithms for 1-form assembly, one
expects that matrix-vector product to perform somewhat more slowly
than the PETSc aij matrix-vector product.  Moreover, algebraic
preconditioners like ILU or AMG will not work with these unassembled
matrices.

The Python context, implemented in the class
:class:`.ImplicitMatrixContext`, provides some important
features to enable advanced solver configuration.

First, it implements a specialized
:method:`.ImplicitMatrixContext.getSubMatrix` method that allows PETSc
to construct FieldSplit preconditioners on fully unassembled matrices
by partitioning UFL into pieces corresponding to the integer labels of
the unknown fields.  This is distinct from the existing options of
splitting either a monolithic matrix (which requires a certain
overhead to build a new matrix and extract the values) or MatNest,
which is more efficient but extracts subblocks into a monolithic
format so that some efficiency is lost on multi-level field splits.

Second, the :class:`.ImplicitMatrixContext` object becomes available
to custom preconditioners.  Since it contains a complete UFL
description of the bilinear form, preconditioners can query or
manipulate it as desired.  As a particularly simple example, the class
:class:`.AssembledPC` simply passes the UFL into :function:`.assemble`
to produce an explicit matrix during set up.  It also sets up a new
PETSc PC context acting on this assembled matrix so that the user can
configure it at run-time via the options database.  The
:method:`AssembledPC.apply` simply invokes the stored PETSc
preconditioner.  Other examples include :class:`.MassInvPC`, which is
useful for Stokes problems since the Schur complement is spectrally
equivalent to the mass matrix.  It extracts the pressure function
space, assembles a mass matrix and sets up a user-configurable KSP
context.  Similarly, the :class:`.PCDPC` implements the PCD
preconditioner for the Navier-Stokes Schur complement, which
approximates the inverse of the Schur complement by a mass matrix
solve, the application of a scalar convection-diffusion operator using
the current velocity of the Newton step, and a Poisson solve.

Frequently, such custom preconditioners require some additional
information that will not be fully available from the UFL description
in :class:`.ImplicitMatrixContext`.  For example, it is not possible
to extract physical parameters such as the Reynolds number from a UFL
bilinear form.  In this case, the :class:`.NonlinearVariationalSolver`
takes a dictionary "extra_ctx" as an optional keyword argument.
Firedrake passes that down into the :class:`.ImplicitMatrixContext` so
that it is accessible to preconditioners.

In this directory, we have several demonstration files, together with
PETSc options files, showing some basic usage and configuration,
together with a more advanced example of Rayleigh-Benard convection.
