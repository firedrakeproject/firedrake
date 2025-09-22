Quasi-Geostrophic Model
=======================

.. rst-class:: emphasis

   This tutorial was contributed by `Francis Poulin
   <mailto:fpoulin@uwaterloo.ca>`__, based on code from `Colin Cotter
   <mailto:colin.cotter@imperial.ac.uk>`__.

The Quasi-Geostrophic (QG) model is very important in geophysical fluid
dynamics as it describes some aspects of large-scale flows in the oceans
and atmosphere very well. The interested reader can find derivations in
:cite:`QG-Pedlosky:1992` and :cite:`QG-Vallis:2006`.

In these notes we present the nonlinear equations for the one-layer QG
model with a free-surface. Then, the weak form will be derived as is
needed for Firedrake.

Governing Equations
-------------------

The Quasi-Geostrophic (QG) model is very similar to the 2D vorticity
equation. Since the leading order geostrophic velocity is incompressible
in the horizontal, the governing equations can be written as

.. math::

   \begin{aligned}
   \partial_t q + \vec \nabla \cdot \left( \vec u q \right)  + \beta v &= 0, \\
   \vec u & = \vec\nabla^\perp \psi, \\
   \nabla^2 \psi - \frac{1}{L_d^2} \psi &= q. \end{aligned}

where the :math:`\psi` and :math:`q` are the streamfunction and
Potential Vorticity (PV). The Laplacian is 2D since we are only in the
horizontal plane and we defined

.. math:: \vec\nabla^\perp =  \hat e_z \times \vec\nabla.

The first equation above states that the PV is conserved following the
flow. The second equation forces the leading order velocity to be
geostrophic and the third equation is the definition for the QG PV for
this barotropic model. To solve this using Finite Elements it is
necessary to establish the weak form of the model, which is done in the
next subsection.

Weak Form
---------

Evolving the nonlinear equations consists of two steps. First, the
elliptic problem must be solved to compute the streamfunction given the
PV. Second, the PV equation must be integrated forward in time. This is
done using a strong stability preserving Runge Kutta 3 (SSPRK3) method.

Elliptic Equation
~~~~~~~~~~~~~~~~~

First, we focus on the elliptic inversion in the case of a flat bottom.
If we compute the inner product of the equation with the test function
:math:`\phi` we obtain,

.. math::

   \begin{aligned}
   \langle \nabla^2 \psi, \phi \rangle - \frac{1}{L_d^2} \langle \psi, \phi \rangle  &= \langle q, \phi \rangle, \\
   \langle \nabla \psi, \nabla \phi \rangle +  \frac{1}{L_d^2} \langle \psi, \phi \rangle &= -\langle q, \phi \rangle,\end{aligned}

where in the second equation we used the divergence theorem and the
homogeneous Dirichlet boundary conditions on the test function.

Evolution Equation
~~~~~~~~~~~~~~~~~~

The SSPRK3 method used as explained in :cite:`QG-Gottlieb:2005` can be written as

.. math::

   \begin{aligned}
   q^{(1)} &= q^n - \Delta t \left[ \vec \nabla \cdot \left( \vec u^n q^n \right) +  \beta v^n \right] , \\
   q^{(2)} &= \frac34 q^n + \frac14 \left[ q^{(1)} - \Delta t  \vec \nabla \cdot \left( \vec u^{(1)} q^{(1)} \right)
   - \Delta t \beta v^{(1)}\right], \\
   q^{n+1} &= \frac13 q^n + \frac23 \left[ q^{(2)} - \Delta t \vec \nabla \cdot \left( \vec u^{(2)} q^{(2)} \right) - \Delta t \beta v^{(1)} \right].\end{aligned}

To get the weak form we need to introduce a test function, :math:`p`,
and take the inner product of the first equation with :math:`p`.

.. math::

   \begin{aligned}
   \langle q^{(1)}, p \rangle &= \langle q^n, p \rangle  - \Delta t \langle \vec \nabla \cdot \left( \vec u^n q^n \right), p \rangle
   - \Delta t \langle \beta  v, q \rangle, \\
   \langle q^{(1)}, p \rangle - \Delta t \langle \vec u^n q^n, \vec\nabla p \rangle  +  \Delta t \langle \beta  v, q \rangle
   &= \langle q^n, p \rangle  - \Delta t \langle \vec u^n q^n, p \rangle_{bdry}\end{aligned}

The first and second terms on the left hand side are referred to as
:math:`a_{mass}` and :math:`a_{int}` in the code. The first term on the
right-hand side is referred to as :math:`a_{mass}` in the code. The
second term on the right-hand side is the extra term due to the DG
framework, which does not exist in the CG version of the problem and it
is referred to as :math:`a_{flux}`. This above problem must be solved
for :math:`q^{(1)}` and then :math:`q^{(2)}` and then these are used to
compute the numerical approximation to the PV at the new time
:math:`q^{n+1}`.

We now move on to the implementation of the QG model for the case of a
freely propagating Rossby wave.  As ever, we begin by importing the
Firedrake library. ::

  from firedrake import *

Next we define the domain we will solve the equations on, square
domain with 50 cells in each direction that is periodic along the
x-axis. ::

  Lx = 2.0 * pi  # Zonal length
  Ly = 2.0 * pi  # Meridonal length
  n0 = 50  # Spatial resolution
  mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly, direction="x", quadrilateral=True)

We define function spaces::

  Vdg = FunctionSpace(mesh, "DQ", 1)  # DQ elements for Potential Vorticity (PV)
  Vcg = FunctionSpace(mesh, "CG", 1)  # CG elements for Streamfunction
  Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity

and initial conditions for the potential vorticity, here we use
Firedrake's ability to :doc:`interpolate UFL expressions <../interpolation>`. ::

  x = SpatialCoordinate(mesh)
  q0 = Function(Vdg).interpolate(0.1 * sin(x[0]) * sin(x[1]))

We define some :class:`~.Function`\s to store the fields::

  dq1 = Function(Vdg)  # PV fields for different time steps
  qh = Function(Vdg)
  q1 = Function(Vdg)

  psi0 = Function(Vcg)  # Streamfunctions for different time steps
  psi1 = Function(Vcg)

along with the physical parameters of the model. ::

  F = Constant(1.0)  # Rotational Froude number
  beta = Constant(0.1)  # beta plane coefficient
  Dt = 0.1  # Time step
  dt = Constant(Dt)

Next, we define the variational problems.  First the elliptic problem
for the stream function. ::

  psi = TrialFunction(Vcg)
  phi = TestFunction(Vcg)

  # Build the weak form for the inversion
  Apsi = (inner(grad(psi), grad(phi)) + F * psi * phi) * dx
  Lpsi = -q1 * phi * dx

We impose homogeneous dirichlet boundary conditions on the stream
function at the top and bottom of the domain. ::

  bc1 = DirichletBC(Vcg, 0.0, (1, 2))

  psi_problem = LinearVariationalProblem(Apsi, Lpsi, psi0, bcs=bc1, constant_jacobian=True)
  psi_solver = LinearVariationalSolver(psi_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})

Next we'll set up the advection equation, for which we need an
operator :math:`\vec\nabla^\perp`, defined as a python anonymouus
function::

  gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

For upwinding, we'll need a representation of the normal to a facet,
and a way of selecting the upwind side::

  n = FacetNormal(mesh)
  un = 0.5 * (dot(gradperp(psi0), n) + abs(dot(gradperp(psi0), n)))

Now the variational problem for the advection equation itself. ::

  q = TrialFunction(Vdg)
  p = TestFunction(Vdg)
  a_mass = p * q * dx
  a_int = (dot(grad(p), -gradperp(psi0) * q) + beta * p * psi0.dx(0)) * dx
  a_flux = (dot(jump(p), un("+") * q("+") - un("-") * q("-"))) * dS
  arhs = a_mass - dt * (a_int + a_flux)

  q_problem = LinearVariationalProblem(a_mass, action(arhs, q1), dq1)

Since the operator is a mass matrix in a discontinuous space, it can
be inverted exactly using an incomplete LU factorisation with zero
fill. ::

  q_solver = LinearVariationalSolver(q_problem,
                                     solver_parameters={"ksp_type": "preonly",
                                                        "pc_type": "bjacobi",
                                                        "sub_pc_type": "ilu"})

To visualise the output of the simulation, we create a
:class:`~.vtk_output.VTKFile` object.  To which we can store multiple
:class:`~.Function`\s.  So that we can distinguish between them we will
give them descriptive names. ::

  q0.rename("Potential vorticity")
  psi0.rename("Stream function")
  v = Function(Vu, name="gradperp(stream function)")
  v.project(gradperp(psi0))

  output = VTKFile("output.pvd")

  output.write(q0, psi0, v)

Now all that is left is to define the timestepping parameters and
execute the time loop. ::

  t = 0.0
  T = 10.0
  dumpfreq = 5
  tdump = 0

  while t < (T - Dt / 2):
      # Compute the streamfunction for the known value of q0
      q1.assign(q0)
      psi_solver.solve()
      q_solver.solve()

      # Find intermediate solution q^(1)
      q1.assign(dq1)
      psi_solver.solve()
      q_solver.solve()

      # Find intermediate solution q^(2)
      q1.assign(0.75 * q0 + 0.25 * dq1)
      psi_solver.solve()
      q_solver.solve()

      # Find new solution q^(n+1)
      q0.assign(q0 / 3 + 2 * dq1 / 3)

      # Store solutions to xml and pvd
      t += Dt
      print(t)

      tdump += 1
      if tdump == dumpfreq:
          tdump -= dumpfreq
          v.project(gradperp(psi0))
          output.write(q0, psi0, v, time=t)

A python script version of this demo can be found :demo:`here <qg_1layer_wave.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
   :keyprefix: QG-
   :labelprefix: QG-
