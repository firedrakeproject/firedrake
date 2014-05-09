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

This time we created the mesh with `Gmsh <http://geuz.org/gmsh/>`_::

  from firedrake import *
  mesh = Mesh("wave_tank.msh")

We choose a degree 2 continuous function space, and set up the
function space and functions::

  V = FunctionSpace(mesh, 'Lagrange', 2)
  p = Function(V)
  phi = Function(V)

  u = TrialFunction(V)
  v = TestFunction(V)

We also need a first order space to make output work properly::

  outfs = FunctionSpace(mesh, 'Lagrange', 1)

Output the initial conditions::

  outfile = File("out.pvd")
  outfile << project(phi, outfs, name="phi")

We establish a boundary condition object::

  bc = DirichletBC(V, 0.0, 1)

Now we set the timestepping variables::

  T = 10.
  dt = 0.001
  t = 0
  step = 0

  while t <= T:
      step += 1

Update the boundary condition value for this timestep::

      bc.set_value(sin(2*pi*5*t))

Step forward :math:`\phi` by half a timestep. Since this does not involve a matrix inversion, this is implemented as a pointwise operation::

      phi -= dt / 2 * p

Now step forward :math:`p`. This is an explicit timestepping scheme
which only requires hte inversion of a mass matrix::

      solve(u * v * dx == v * p * dx + dt * inner(grad(v), grad(phi)) * dx,
            p, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                          'pc_type': 'sor',
                                          'pc_sor_symmetric': True})

Step forward :math:`\phi` by the second half timestep::

      phi -= dt / 2 * p

Advance time and output as appropriate::

      t += dt
      if step % 10 == 0:
          outfile << project(phi, outfs, name="phi")

.. only:: html

   The following animation, produced in Paraview, illustrates the output of this simulation:

   .. container:: youtube

      .. youtube:: xhxvM1N8mDQ?modestbranding=1;controls=0;rel=0
         :width: 600px

.. only:: latex

   An animation, produced in Paraview, illustrating the output of this simulation can be found `on youtube <https://www.youtube.com/watch?v=xhxvM1N8mDQ>`_.
  

A python script version of this demo can be found `here <linear_wave_equation.py>`__.
