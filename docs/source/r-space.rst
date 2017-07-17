.. default-role:: math

The `R` space
=============

The function space `R` (for "Real") is the space of functions which
are constant over the whole domain. It is employed to model concepts
such as global constraints.

An example:
-----------

.. warning::

   This section illustrates the use of the Real space using the
   simplest example. This is usually not the optimal approach for
   removing the nullspace of an operator. If that is your only goal
   then you are probably better placed removing the null space in the
   linear solver using the facilities documented in the section
   :ref:`singular_systems`.

Consider a Poisson equation in weak form, find `u\in V` such that:

.. math::

  \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x  = -\int_{\Gamma(3)} v\,\mathrm{d}s + \int_{\Gamma(4)} v\,\mathrm{d}s \qquad\forall v\in V

where `\Gamma(3)` and `\Gamma(4)` are domain boundaries over which the
boundary conditions `\nabla u \cdot n = -1` and `\nabla u \cdot n = 1`
are applied respectively. This system has a null space composed of the
constant functions. One way to remove this is to add a Lagrange
multiplier from the space `R` and use the resulting constraint
equation to enforce that the integral of `u` is zero. The resulting
system is find `u\in V`, `r\in R` such that:

.. math::

  \int_\Omega \nabla u \cdot \nabla v + rv\,\mathrm{d}x  = -\int_{\Gamma(3)} v\,\mathrm{d}s + \int_{\Gamma(4)} v\,\mathrm{d}s \qquad\forall v\in V

  \int_\Omega us \,\mathrm{d}x = 0 \qquad \forall s\in R

The corresponding Python code is:

.. code-block:: python

  from firedrake import *

  m = UnitSquareMesh(25, 25)
  V = FunctionSpace(m, 'CG', 1)
  R = FunctionSpace(m, 'R', 0)
  W = V * R
  u, r = TrialFunctions(W)
  v, s = TestFunctions(W)

  a = inner(grad(u), grad(v))*dx + u*s*dx + v*r*dx
  L = -v*ds(3) + v*ds(4)

  w = Function(W)
  solve(a == L, w)
  u, s = split(w)
  exact = Function(V)
  exact.interpolate(Expression('x[1] - 0.5'))
  print sqrt(assemble((u - exact)*(u - exact)*dx))


Representing matrices involving `R`
-----------------------------------

Functions in the space `R` are different from other finite element
functions in that their support extends to the whole domain. To
illustrate the consequences of this, we can represent the matrix in
the Poisson problem above as:

.. math::

  A= \begin{bmatrix} L & K \\
  K^T & 0
  \end{bmatrix}

where:

.. math::

  L_{ij} = \int_\Omega \nabla \phi_i \phi_j \,\mathrm{d}x

  K_{ij} = \int_\Omega \phi_i \psi_j \,\mathrm{d}x

where `\{\phi_i\}` is the basis for `V` and `\{\psi_i\}` is the basis
for `R`. Note that there is only a single basis function for `R` and `\psi_i \equiv 1` hence:

.. math::

  K_{ij} = \int_\Omega \phi_i \,\mathrm{d}x

with the result that `K` is a single dense matrix column. Similiarly,
`K^T` is a single dense matrix row.

Using the CSR matrix format typically employed by Firedrake, each
matrix row is stored on a single processor. Were this carried through to `K^T`, both the assembly and
action of this row would require the entire system state to be gathered
onto one MPI process. This is clearly a horribly non-performant
option.

Instead, we observe that a dense matrix row (or column) is isomorphic
to a :class:`~firedrake.function.Function` and implement these matrix
blocks accordingly.

.. figure:: images/real_distribution.png
   :figwidth: 60%
   :alt: Parallel distribution of a matrix
   :align: center

   Example parallel distribution of the matrix `A`. Colours indicate
   the processor on which the data is stored. Notice the dense row and
   column, and that the dense row is distributed across the
   processors.


Assembling matrices involving `R`
---------------------------------

Assembling the column block is implemented by replacing the trial
function with the constant 1, thereby transforming a 2-form into a
1-form, and assembling. Similarly, assembling the row block simply
requires the replacement of the test function with the constant 1, and
assembling.

The one by one block in the corner is assembled by replacing both
the test and trial functions of the corresponding form with 1 and
assembling. The remaining block does not involve `R` and is assembled
as usual.
