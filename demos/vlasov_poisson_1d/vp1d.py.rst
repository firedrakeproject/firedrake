1D Vlasov-Poisson Equation
===========================

This tutorial was contributed by `Colin Cotter
<mailto:colin.cotter@imperial.ac.uk>`__ and Werner Bauer.

A plasma is a continuum of moving particles with nonunique velocity
at each point in space. In :math:`d` dimensions, the plasma is
described by a density :math:`f(x,v,t)` where :math:`x\in \mathbb{R}^d`
are the physical coordinates and :math:`v \in \mathbb{R}^d` are velocity
coordinates. Hence, in :math:`d` dimensions, a :math:`2d`
dimensional mesh is required. To deal with this curse of
dimensionality, particle-in-cell methods are usually used. However,
in 1 dimension, it is tractable to simulate the plasma on a 2
dimensional mesh.

The Vlasov equation models the (collisionless) conservation of plasma
particles, according to 

.. math::
   f_t + \nabla_{\vec{x}} \cdot (\vec{v}f) + \nabla_{\vec{v}} \cdot (\vec{a}f) = 0,

where

.. math::
   \nabla_{\vec{x}} = (\partial_{x_1},\ldots, \partial_{x_d}), \quad
   \nabla_{\vec{v}} = (\partial_{v_1},\ldots, \partial_{v_d}).

To close the system, we need a formula for the acceleration :math:`\vec{a}`.
In the (single species) Vlasov-Poisson model, the acceleration is
determined by the electrostatic force,

.. math::
   \vec{a} = -\frac{1}{m}\nabla\phi,

where :math:`m`
is the mass per plasma particle, and :math:`\phi` is the electrostatic
potential determined by the Poisson equation,

.. math::
   -\nabla^2\phi = q\int_{\mathbb{R}^d} f(\vec{x},\vec{v},t)\mathrm{d} v,

where :math:`q` is the electric charge per plasma particle.

In this demo we specialise to :math:`d=1`, and the equations become

.. math::
   f_t + (fv)_x + (-f\phi_x/m)_v = 0, \quad
   -\phi_{xx} = q\int f(x,v,t)\mathrm{d} v,

with coordinates :math:`(x,v)\in \mathbb{R}^2`. From now on we will
relabel these coordinates :math:`(x,v)\mapsto (x_1,x_2)`, obtaining
the equivalent form,

.. math::
   f_t + \nabla\cdot(\vec{u}f) = 0, \quad \vec{u} = (v,-\phi_x/m), \quad
   -\phi_{x_1x_1} = q\int f(x_1,x_2,t)\mathrm{d} x_2,

where :math:`\nabla=(\partial_{x_1},\partial{x_2})`. From now we will
choose units such that :math:`q,m` are absorbed into the definition of
:math:`f`.

To proceed, we need to develop variational formulations of these
equations.

For the density we will use a discontinuous Galerkin formulation,
and the continuity equation becomes 

.. math::

   \int_\Omega \! q \frac{\partial f}{\partial t} \, \mathrm{d} x
   &= \int_\Omega \! f \nabla \cdot (q \vec{u}) \, \mathrm{d} x\\
   &\quad- \int_{\Gamma_\mathrm{int}} \! \widetilde{f}(q_+ \vec{u} \cdot \vec{n}_+
     + q_- \vec{u} \cdot \vec{n}_-) \, \mathrm{d} S\\
   &\quad- \int_{\Gamma_{\mathrlap{\mathrm{ext, inflow}}}} q f_\mathrm{in} \vec{u} \cdot
   \vec{n} \, \mathrm{d} s\\
   &\quad- \int_{\Gamma_{\mathrlap{\mathrm{ext, outflow}}}} q f \vec{u} \cdot
   \vec{n} \, \mathrm{d} s
   \qquad \forall q \in V,

where :math:`\Omega` is the computational domain in :math:`(x,v)`
space, :math:`V` is the discontinuous finite element space,
:math:`\Gamma_\mathrm{int}` is the set of interior cell edges,
:math:`\Gamma_{\mathrlap{\mathrm{ext, inflow}}}` is the part of
exterior boundary where :math:`\vec{u}\cdot\vec{n}<0`,
:math:`\Gamma_{\mathrlap{\mathrm{ext, outflow}}}` is the part of
exterior boundary where :math:`\vec{u}\cdot\vec{n}>0`, :math:`n` is
the normal to each edge, :math:`\tilde{f}` is the upwind value of
:math:`f`, and :math:`f_{\mathrm{in}}` is the inflow boundary value
for :math:`f`. See the Discontinuous Galerkin advection
:doc:`demo<DG_advection.py>` for more details. The unapproximated
problem should have :math:`-\infty < x_2 < \infty`, but we approximate
the problem by solving in the domain :math:`\Omega=I_1\times [-H/2, H/2]`,
where :math:`I` is some chosen interval.

For the Poisson equation, we will use a regular Galerkin formulation.
The difficulty in the formulation is the integral over :math:`x_2`. We
deal with this by considering a space :math:`\bar{W}` which is restricted
to functions that are constant in the vertical. Multiplying by a
test function :math:\psi\in \bar{W}` and integrating by parts gives

.. math::

   \int \psi_{x_1}\phi_{x_1} \mathrm{d} x_1
   = \int \int f(x_1,x_2,t) \psi \mathrm{d} x_1\mathrm{d} x_2, \quad
   \forall \psi \in \bar{W}.

Since the left hand side integrand is independent of :math:`v`, we
can integrate over :math:`v` and divide by :math:`H`, to obtain

.. math::

   \int_\Omega \psi_{x_1}\phi_{x_1} \mathrm{d} x
   = \int Hf \psi \mathrm{d} x, \quad
   \forall \psi \in \bar{W},

which is now in a form which we can implement easily in Firedrake. One
final issue is that this problem only has a solution up to an additive
constant, so we further restrict :math:`\phi \in \mathring{\bar{W}}`,
where

.. math::
   \mathring{\bar{W}} = \{ w\in \bar{W}: \bar{w}=0\},

where

.. math::

   \bar{w} = \frac{\int_{\Omega} w \mathrm{d} x}{\int_{\Omega} 1 \mathrm{d} x}.
   
Then we seek the solution of 

.. math::

   \int_\Omega \psi_{x_1}\phi_{x_1}\mathrm{d} x
   = \int H(f-\bar{f}) \psi \mathrm{d} x, \quad
   \forall \psi \in \mathring{\bar{W}}.

To discretise in time, we will use an SSPRK3 time discretisation.  At
each Runge-Kutta stage, we must solve for the electrostatic potential,
and then use it to compute :math:`\vec{u}`, in order to compute
:math:`\partial f/\partial t`.
   
As usual, to implement this problem, we start by importing the
Firedrake namespace. ::

  from firedrake import *

We build the mesh by constructing a 1D mesh, which will be extruded in
the vertical. Here we will use periodic boundary conditions in the
:math:`x_1` direction, ::
  
  ncells = 50
  L = 4*pi
  base_mesh = PeriodicIntervalMesh(ncells, L)

    H = 10.0
  nlayers = 50

  # extruded mesh in x-v coordinates
  mesh = ExtrudedMesh(base_mesh, layers=nlayers,
                      layer_height=H/nlayers)

  # move the mesh in the vertical so v=0 is in the middle
  Vc = mesh.coordinates.function_space()
  x, v = SpatialCoordinate(mesh)
  X = Function(Vc).interpolate(as_vector([x, v-H/2]))
  mesh.coordinates.assign(X)

  # Space for the number density
  V = FunctionSpace(mesh, 'DG', 1)

  # Space for the electric field (independent of v)
  Vbar = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

  x, v = SpatialCoordinate(mesh)

  # initial condition
  A = Constant(0.05)
  k = Constant(0.5)
  fn = Function(V).interpolate(
     v**2*exp(-v**2/2)
     *(1 + A*cos(k*x))/(2*pi)**0.5
  )

  # remove the mean
  One = Function(V).assign(1.0)
  fbar = assemble(fn*dx)/assemble(One*dx)
  
  # electrostatic potential
  phi = Function(Vbar)
  
  # input for electrostatic solver
  f_in = Function(V)
  # Solver for electrostatic potential
  psi = TestFunction(Vbar)
  dphi = TrialFunction(Vbar)
  phi_eqn = dphi.dx(0)*psi.dx(0)*dx - H*(f_in-fbar)*psi*dx
  shift_eqn = dphi.dx(0)*psi.dx(0)*dx + dphi*psi*dx
  nullspace = VectorSpaceBasis(constant=True)
  phi_problem = LinearVariationalProblem(lhs(phi_eqn), rhs(phi_eqn),
  phi, aP=shift_eqn)
  params = {
     'ksp_type': 'gmres',
     'pc_type': 'lu',
     'ksp_rtol': 1.0e-8,
     }

  phi_solver = LinearVariationalSolver(phi_problem,
                                       nullspace=nullspace,
				       solver_parameters=params)
  dtc = Constant(0)

  # Solver for DG advection
  df_out = Function(V)
  q = TestFunction(V)
  u = as_vector([v, -phi.dx(0)])
  n = FacetNormal(mesh)
  un = 0.5*(dot(u, n) + abs(dot(u, n)))
  df = TrialFunction(V)
  df_a = q*df*dx
  dS = dS_h + dS_v
  f_bc = Function(V).assign(0.)
  df_L = dtc*(div(u*q)*f_in*dx
     - (q('+') - q('-'))*(un('+')*f_in('+') - un('-')*f_in('-'))*dS
     - conditional(dot(u, n) > 0, q*dot(u, n)*f_in, 0.)*ds_tb
      )
  df_problem = LinearVariationalProblem(df_a, df_L, df_out)
  df_solver = LinearVariationalSolver(df_problem)

  T = 50.0 # maximum timestep
  t = 0. # model time
  ndump = 100
  dumpn = 0
  nsteps = 5000
  dt = T/nsteps
  dtc.assign(dt)

  # RK stage variables
  f1 = Function(V)
  f2 = Function(V)

  outfile = VTKFile("vlasov.pvd")
  f_in.assign(fn)
  phi_solver.solve()
  outfile.write(fn, phi)
  phi.assign(.0)
  
  for step in ProgressBar("Timestep").iter(range(nsteps)):
    f_in.assign(fn)
    phi_solver.solve()
    df_solver.solve()
    f1.assign(fn + df_out)

    f_in.assign(f1)
    phi_solver.solve()
    df_solver.solve()
    f2.assign(3*fn/4 + (f1 + df_out)/4)

    f_in.assign(f2)
    phi_solver.solve()
    df_solver.solve()
    fn.assign(fn/3 + 2*(f2 + df_out)/3)

    t += dt
    dumpn += 1
    if dumpn % ndump == 0:
        dumpn = 0
        outfile.write(fn, phi)

A Python script version of this demo can be found :demo:`here <vp1d.py>`.
