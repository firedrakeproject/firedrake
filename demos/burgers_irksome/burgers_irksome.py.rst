Burgers equation
================

The Burgers equation is a non-linear equation for the advection and
diffusion of momentum. Here we choose to write the Burgers equation in
two dimensions to demonstrate the use of vector function spaces:

.. math::

   \frac{\partial u}{\partial t} + (u\cdot\nabla) u - \nu\nabla^2 u = 0

   (n\cdot \nabla) u = 0 \ \textrm{on}\ \Gamma

where :math:`\Gamma` is the domain boundary and :math:`\nu` is a
constant scalar viscosity. The solution :math:`u` is sought in some
suitable vector-valued function space :math:`V`. We take the inner
product with an arbitrary test function :math:`v\in V` and integrate
the viscosity term by parts:

.. math::

   \int_\Omega\frac{\partial u}{\partial t}\cdot v +
   ((u\cdot\nabla) u)\cdot v + \nu\nabla u\cdot\nabla v \ \mathrm d x = 0.

The boundary condition has been used to discard the surface
integral. We can now proceed to set up the problem. We choose a resolution
and set up a square mesh::


  from firedrake import (UnitSquareMesh, VectorFunctionSpace, Function,
                      TestFunction, SpatialCoordinate, as_vector, sin,
                      pi, Constant, inner, dx, dot, grad, VTKFile,
                      project, warning)

  try:
      from irksome import TimeStepper, RadauIIA, Dt, MeshConstant
  except ImportError:
      import sys
      warning("This demo requires Irksome to be installed.")
      sys.exit(0)

We will create the Butcher tableau for the Radau IIA method. Note that Radau IIA is backward
Euler and we can get higher order methods by changing our argument::

  butcher_tableau = RadauIIA(1)
  ns = butcher_tableau.num_stages

  n = 30
  mesh = UnitSquareMesh(n, n)

We then define our function space :math:`V`, our function :math:`u`, and
our test function :math:`v`. We choose degree 2 continuous Lagrange polynomials and also need 
a piecewise linear space for output purposes::

  V = VectorFunctionSpace(mesh, "CG", 2)
  V_out = VectorFunctionSpace(mesh, "CG", 1)

  u = Function(V, name="Velocity")

  v = TestFunction(V)

  x = SpatialCoordinate(mesh)
  u.project(as_vector([sin(pi*x[0]), 0]))

:math:`\nu` is set to a (fairly arbitrary) small constant value::

  nu = Constant(0.0001)

  F = inner(Dt(u), v)*dx + inner(dot(u, grad(u)), v)*dx + nu*inner(grad(u),
                                                                   grad(v))*dx

We now create an object for output visualisation::

  outfile = VTKFile("burgers_irksome.pvd")

  outfile.write(project(u, V_out, name="Velocity"))

We define variables to store the time step and current time value::

  MC = MeshConstant(mesh)
  dt = MC.Constant(1.0 / n)
  t = MC.Constant(0.0)

  luparams = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

Most of Irksome's magic happens in the :class:`.TimeStepper`.  It
transforms our semidiscrete form `F` into a fully discrete form for
the stage unknowns and sets up a variational problem to solve for the
stages at each time step.::

  stepper = TimeStepper(F, butcher_tableau, t, dt, u)

This logic is pretty self-explanatory.  We use the
:class:`.TimeStepper`'s :meth:`~.TimeStepper.advance` method, which solves the variational
problem to compute the Runge-Kutta stage values and then updates the solution.::

  tfinal = 0.5
  while (tfinal - float(t) > 1.e-8):
      stepper.advance()
      print(float(t))
      t.assign(float(t) + float(dt))
      outfile.write(project(u, V_out, name="Velocity"))

A python script version of this demo can be found :demo:`here <burgers_irksome.py>`.
