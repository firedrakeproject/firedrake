.. only:: html

   .. contents::


=================================
 Matrix-free operators and PETSc
=================================

Introduction
============

Sometimes, the system we wish to solve can not be described purely in
terms of a sum of weak forms that we can then assemble.  Or else, it
might be, but the resulting assembled operator would be dense.  In
this chapter, we will see how to solve such problems in a
"matrix-free" manner, using Firedrake to assemble the pieces and then
providing a matrix object to PETSc which is unassembled.

To take a concrete example, let us consider a linear system obtained
from a normal variational problem, augmented with a rank-1
perturbation:

.. math::

   B := A + \vec{u} \vec{v}^T

The matrix :math:`B` is dense, however its action on a vector may be
computed in only marginally more work than computing the action of
:math:`A` since

.. math::

   B \vec{x} \equiv A \vec{x} + \vec{u} (\vec{v} \cdot \vec{x})


Accessing PETSc objects
=======================

Firedrake builds on top of PETSc for its linear algebra, and therefore
all assembled forms provide access to the underlying PETSc object.
For assembled bilinear forms, the PETSc object is a ``Mat``; for
assembled linear forms, it is a ``Vec``.  The ways we access these are
different.  For a bilinear form, the matrix is obtained with:

.. code-block:: python

   petsc_mat = assemble(bilinear_form).M.handle

For a linear form, we need to use a context manager.  There are two
options available here, depending on whether we want read-only or
read-write access to the PETSc object.  For read-only access, we use:

.. code-block:: python

   with assemble(linear_form).dat.vec_ro as v:
       petsc_vec_ro = v

For read-write access, use:

.. code-block:: python

   with assemble(linear_form).dat.vec as v:
       petsc_vec = v


Building an operator
====================

To solve the linear system :math:`Bx = b` we need to define the
operator :math:`B` such that PETSc can use it.  To do this, we build a
Python class that provides a ``mult`` method:

.. code-block:: python

   class MatrixFreeB(object):

       def __init__(self, A, u, v):
           self.A = A
           self.u = u
           self.v = v

       def mult(self, mat, x, y):
           # y <- A x
           self.A.mult(x, y)

           # alpha <- v^T x
           alpha = self.v.dot(x)

           # y <- y + alpha*u
           y.axpy(alpha, self.u)


Now we must build a PETSc ``Mat`` and indicate that it should use this
newly defined class to compute the matrix action:

.. code-block:: python

   # Import petsc4py namespace
   from firedrake.petsc import PETSc

   B = PETSc.Mat().create()

   # Assemble the bilinear form that defines A and get the concrete
   # PETSc matrix
   A = assemble(bilinear_form).M.handle

   # Now do the same for the linear forms for u and v, making a copy

   with assemble(u_form).dat.vec_ro as u_vec:
       u = u_vec.copy()

   with assemble(v_form).dat.vec_ro as v_vec:
       v = v_vec.copy()


   # Build the matrix "context"

   Bctx = MatrixFreeB(A, u, v)

   # Set up B
   # B is the same size as A
   B.setSizes(*A.getSizes())

   B.setType(B.Type.PYTHON)
   B.setPythonContext(Bctx)
   B.setUp()


The next step is to build a linear solver object to solve the system.
For this we need a PETSc ``KSP``:

.. code-block:: python

   ksp = PETSc.KSP().create()

   ksp.setOperators(B)

   ksp.setFromOptions()


Now we can solve a system using this ``ksp`` object:

.. code-block:: python

   solution = Function(V)

   rhs = assemble(rhs_form)

   with rhs.dat.vec_ro as b:
       with solution.dat.vec as x:
           ksp.solve(b, x)
