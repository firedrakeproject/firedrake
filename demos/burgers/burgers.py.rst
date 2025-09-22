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
integral. Next, we need to discretise in time. For simplicity and
stability we elect to use a backward Euler discretisation:

.. math::

   \int_\Omega\frac{u^{n+1}-u^n}{dt}\cdot v +
   ((u^{n+1}\cdot\nabla) u^{n+1})\cdot v + \nu\nabla u^{n+1}\cdot\nabla v \ \mathrm d x = 0.

We can now proceed to set up the problem. We choose a resolution and set up a square mesh::

  from firedrake import *
  n = 30
  mesh = UnitSquareMesh(n, n)

We choose degree 2 continuous Lagrange polynomials. We also need a
piecewise linear space for output purposes::

  V = VectorFunctionSpace(mesh, "CG", 2)
  V_out = VectorFunctionSpace(mesh, "CG", 1)

We also need solution functions for the current and the next
timestep. Note that, since this is a nonlinear problem, we don't
define trial functions::

  u_ = Function(V, name="Velocity")
  u = Function(V, name="VelocityNext")

  v = TestFunction(V)

For this problem we need an initial condition::

  x = SpatialCoordinate(mesh)
  ic = project(as_vector([sin(pi*x[0]), 0]), V)

We start with current value of u set to the initial condition, but we
also use the initial condition as our starting guess for the next
value of u::

  u_.assign(ic)
  u.assign(ic)

:math:`\nu` is set to a (fairly arbitrary) small constant value::

  nu = 0.0001

The timestep is set to produce an advective Courant number of
around 1. Since we are employing backward Euler, this is stricter than
is required for stability, but ensures good temporal resolution of the
system's evolution::

  timestep = 1.0/n

Here we finally get to define the residual of the equation. In the advection
term we need to contract the test function :math:`v` with
:math:`(u\cdot\nabla)u`, which is the derivative of the velocity in the
direction :math:`u`. This directional derivative can be written as
``dot(u,nabla_grad(u))`` since ``nabla_grad(u)[i,j]``:math:`=\partial_i u_j`.
Note once again that for a nonlinear problem, there are no trial functions in
the formulation. These will be created automatically when the residual
is differentiated by the nonlinear solver::

  F = (inner((u - u_)/timestep, v)
       + inner(dot(u,nabla_grad(u)), v) + nu*inner(grad(u), grad(v)))*dx

We now create an object for output visualisation::

  outfile = VTKFile("burgers.pvd")

Output only supports visualisation of linear fields (either P1, or
P1DG).  In this example we project to a linear space by hand.  Another
option is to let the :class:`~.vtk_output.VTKFile` object manage the
decimation.  It supports both interpolation to linears (the default) or
projection (by passing ``project_output=True`` when creating the
:class:`~.vtk_output.VTKFile`). Outputting data is carried out using
the :meth:`~.vtk_output.VTKFile.write` method of
:class:`~.vtk_output.VTKFile` objects::

  outfile.write(project(u, V_out, name="Velocity"))

Finally, we loop over the timesteps solving the equation each time and
outputting each result. Firedrake's default solver parameters are used,
which amount to applying a full LU decomposition as a preconditioner. ::

  t = 0.0
  end = 0.5
  while (t <= end):
      solve(F == 0, u)
      u_.assign(u)
      t += timestep
      outfile.write(project(u, V_out, name="Velocity"))

A python script version of this demo can be found :demo:`here <burgers.py>`.
