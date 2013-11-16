Simple Helmholtz equation
=========================

Let's start by considering the Helmholtz equation on a unit square,
:math:`\Omega`, with boundary :math:`\Gamma`:

.. math::

   -\nabla^2 u + u = f

   \nabla u \cdot \vec{n} = 0 \ \textrm{on}\ \Gamma

for some known function :math:`f`. The solution to this equation will
be some function :math:`u\in V` for some suitable function space
:math:`V` such that satisfies these equations. We transform the
equation into weak form by multiplying by an arbitrary test function
in :math:`V`, integrating over the domain and then integrating by
parts. The variational problem so derived reads: find :math:`u\in V` such that:

.. math::

   \int_\Omega \nabla u\cdot\nabla v \mathrm{d}x = \int_\Omega vf\mathrm{d}x

.. if I could do strikout I would put - \sout{\int_\Gamma v\nabla u \cdot \vec{n}} \mathrm{d}s in

Note that the boundary condition has been enforced weakly by removing
the surface term resulting from the integration by parts.

We can choose the function :math:`f`, so we take:

.. math::

   f = (1.0 + 8.0\pi^2)\cos(2\pi x)\cos(2\pi y)

which conveniently yields the analytic solution:

.. math::

   u = \cos(2\pi x)\cos(2\pi y)

However we wish to employ this as an example for the finite element
method, so lets go ahead and produce numerical solution.

First, we always need a mesh. Let's have a :math:`10\times10` element unit square::

  from firedrake import *
  mesh = UnitSquareMesh(10, 10)

Now we need to decide which function space we'd like to solve the
problem. Let's use piecewise quadratic functions continuous between
elements::

  V = FunctionSpace(mesh, "CG", degree)
