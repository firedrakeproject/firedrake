Simple Helmholtz equation
=========================

Let's start by considering the modified Helmholtz equation on a unit square,
:math:`\Omega`, with boundary :math:`\Gamma`:

.. math::

   -\nabla^2 u + u &= f

   \nabla u \cdot \vec{n} &= 0 \quad \textrm{on}\ \Gamma

for some known function :math:`f`. The solution to this equation will
be some function :math:`u\in V`, for some suitable function space
:math:`V`, that satisfies these equations. Note that this is the
Helmholtz equation that appears in meteorology, rather than the
indefinite Helmholtz equation :math:`\nabla^2 u + u = f` that arises
in wave problems.

We transform the equation into weak form by multiplying by an arbitrary
test function in :math:`V`, integrating over the domain and then
integrating by parts. The variational problem so derived reads: find
:math:`u \in V` such that:

.. math::

   \require{cancel}
   \int_\Omega \nabla u\cdot\nabla v  + uv\ \mathrm{d}x = \int_\Omega
   vf\ \mathrm{d}x + \cancel{\int_\Gamma v \nabla u \cdot \vec{n} \mathrm{d}s}

Note that the boundary condition has been enforced weakly by removing
the surface term resulting from the integration by parts.

We can choose the function :math:`f`, so we take:

.. math::

   f = (1.0 + 8.0\pi^2)\cos(2\pi x)\cos(2\pi y)

which conveniently yields the analytic solution:

.. math::

   u = \cos(2\pi x)\cos(2\pi y)

However we wish to employ this as an example for the finite element
method, so lets go ahead and produce a numerical solution.

First, we always need a mesh. Let's have a :math:`10\times10` element unit square::

  from firedrake import *
  mesh = UnitSquareMesh(10, 10)

We need to decide on the function space in which we'd like to solve the
problem. Let's use piecewise linear functions continuous between
elements::

  V = FunctionSpace(mesh, "CG", 1)

We'll also need the test and trial functions corresponding to this
function space::

  u = TrialFunction(V)
  v = TestFunction(V)

We declare a function over our function space and give it the
value of our right hand side function::

  f = Function(V)
  x, y = SpatialCoordinate(mesh)
  f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

We can now define the bilinear and linear forms for the left and right
hand sides of our equation respectively::

  a = (dot(grad(v), grad(u)) + v * u) * dx
  L = f * v * dx

Finally we solve the equation. We redefine `u` to be a function
holding the solution:: 

  u = Function(V)

Since we know that the Helmholtz equation is
symmetric, we instruct PETSc to employ the conjugate gradient method::

  solve(a == L, u, solver_parameters={'ksp_type': 'cg'})

For more details on how to specify solver parameters, see the section
of the manual on :doc:`solving PDEs <../solving-interface>`.

Next, we might want to look at the result, so we output our solution
to a file::

  File("helmholtz.pvd").write(u)

This file can be visualised using `paraview <http://www.paraview.org/>`__.

We could use the built in plot function of firedrake by calling 
:func:`plot <firedrake.plot.plot>` to plot a surface graph. Before that,
matplotlib.pyplot should be installed and imported::

  try:
    import matplotlib.pyplot as plt
  except:
    warning("Matplotlib not imported")

  try:
    plot(u)
  except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e.message)

For a contour plot, it could be plotted by adding an additional key word
argument::

  try:
    plot(u, contour=True)
  except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e.message)

Don't forget to show the image::

  try:
    plt.show()
  except Exception as e:
    warning("Cannot show figure. Error msg: '%s'" % e.message)

Alternatively, since we have an analytic solution, we can check the
:math:`L_2` norm of the error in the solution::

  f.interpolate(cos(x*pi*2)*cos(y*pi*2))
  print sqrt(assemble(dot(u - f, u - f) * dx))

A python script version of this demo can be found `here <helmholtz.py>`__.
