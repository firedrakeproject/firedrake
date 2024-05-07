Double slit experiment
======================

Here we solve a linear wave equation using an explicit timestepping
scheme. This example demonstrates the use of an externally generated
mesh, pointwise operations on Functions, and a time varying boundary
condition. The strong form of the equation we set out to solve is:

.. math::

   \frac{\partial^2\phi}{\partial t^2} - \nabla^2 \phi = 0

   \nabla \phi \cdot n = 0 \ \textrm{on}\ \Gamma_N

   \phi = \frac{1}{10\pi}\cos(10\pi t)  \ \textrm{on}\ \Gamma_D

To facilitate our choice of time integrator, we make the substitution:

.. math::

   \frac{\partial\phi}{\partial t} = - p

   \frac{\partial p}{\partial t} + \nabla^2 \phi = 0

   \nabla \phi \cdot n = 0 \ \textrm{on}\ \Gamma_N

   p = \sin(10\pi t)  \ \textrm{on}\ \Gamma_D

We then form the weak form of the equation for :math:`p`. Find
:math:`p \in V` such that:

.. math::

   \int_\Omega \frac{\partial p}{\partial t} v\,\mathrm d x = \int_\Omega \nabla\phi\cdot\nabla v\,\mathrm d x
   \quad \forall v \in V

For a suitable function space V. Note that the absence of spatial
derivatives in the equation for :math:`\phi` makes the weak form of
this equation equivalent to the strong form so we will solve it pointwise.

In time we use a simple symplectic method in which we offset :math:`p`
and :math:`\phi` by a half timestep.

This time we created the mesh with `Gmsh <http://gmsh.info/>`_:

.. code-block:: bash

   gmsh -2 wave_tank.geo

We can then start our Python script and load this mesh::

  from firedrake import *
  mesh = Mesh("wave_tank.msh")

We choose a degree 1 continuous function space, and set up the
function space and functions. Setting the `name` parameter when
constructing :class:`.Function` objects will set the name used in the
output file::

  V = FunctionSpace(mesh, 'Lagrange', 1)
  p = Function(V, name="p")
  phi = Function(V, name="phi")

  u = TrialFunction(V)
  v = TestFunction(V)

Output the initial conditions::

  outfile = VTKFile("out.pvd")
  outfile.write(phi)

We next establish a boundary condition object. Since we have time-dependent
boundary conditions, we first create a :class:`.Constant` to hold the
value and use that::

  bcval = Constant(0.0)
  bc = DirichletBC(V, bcval, 1)

Now we set the timestepping variables::

  T = 10.
  dt = 0.001
  t = 0
  step = 0

Finally we set a flag indicating whether we wish to perform
mass-lumping in the timestepping scheme::

  lump_mass = True

Now we are ready to start the timestepping loop::

  while t <= T:
      step += 1

Update the boundary condition value for this timestep::

      bcval.assign(sin(2*pi*5*t))

Step forward :math:`\phi` by half a timestep. Since this does not involve a matrix inversion, this is implemented as a pointwise operation::

      phi -= dt / 2 * p

Now step forward :math:`p`. This is an explicit timestepping scheme
which only requires the inversion of a mass matrix.  We have two
options at this point, we may either `lump` the mass, which reduces
the inversion to a pointwise division::

      if lump_mass:
          p.dat.data[:] += assemble(dt * inner(nabla_grad(v), nabla_grad(phi))*dx).dat.data_ro / assemble(v*dx).dat.data_ro

In the mass lumped case, we must now ensure that the resulting
solution for :math:`p` satisfies the boundary conditions::

          bc.apply(p)

Alternatively, we can invert the mass matrix using a linear solver::

      else:
          solve(u * v * dx == v * p * dx + dt * inner(grad(v), grad(phi)) * dx,
                p, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                              'pc_type': 'sor',
                                              'pc_sor_symmetric': True})


Step forward :math:`\phi` by the second half timestep::

      phi -= dt / 2 * p

Advance time and output as appropriate, note how we pass the current
timestep value into the :meth:`~.VTKFile.write` method, so that when
visualising the results Paraview will use it::

      t += dt
      if step % 10 == 0:
          outfile.write(phi, time=t)

.. only:: html

   The following animation, produced in Paraview, illustrates the output of this simulation:

   .. container:: youtube

      .. youtube:: xhxvM1N8mDQ?modestbranding=1;controls=0;rel=0
         :width: 600px

.. only:: latex

   An animation, produced in Paraview, illustrating the output of this simulation can be found `on youtube <https://www.youtube.com/watch?v=xhxvM1N8mDQ>`_.


A python script version of this demo can be found :demo:`here <linear_wave_equation.py>`. The gmsh input file is :demo:`here <wave_tank.geo>`.
