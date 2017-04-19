DG advection equation with upwinding
====================================

We next consider the advection equation

.. math::

  \frac{\partial q}{\partial t} + (\vec{u}\cdot\nabla)q = 0

in a domain :math:`\Omega`, where :math:`\vec{u}` is a prescribed vector field,
and :math:`q(\vec{x}, t)` is an unknown scalar field. The value of :math:`q` is
known initially:

.. math::

  q(\vec{x}, 0) = q_0(\vec{x}),

and the value of :math:`q` is known for all time on the subset of the boundary
:math:`\Gamma` in which :math:`\vec{u}` is directed towards the interior of the
domain:

.. math::

  q(\vec{x}, t) = q_\mathrm{in}(\vec{x}, t) \quad \text{on} \ \Gamma_\mathrm{inflow}

where :math:`\Gamma_\mathrm{inflow}` is defined appropriately.

We will look for a solution :math:`q` in a space of *discontinuous* functions
:math:`V`.  A weak form of the continuous equation in each element :math:`e` is

.. math::

   \int_e \! \phi_e \frac{\partial q}{\partial t} \, \mathrm{d} x
   + \int_e \! \phi_e (\vec{u}\cdot\nabla)q \, \mathrm{d} x = 0, \qquad
   \forall \phi_e \in V_e,

where we explicitly introduce the subscript :math:`e` since the test functions
:math:`\phi_e` are local to each element.  Using integration by parts on the
second term, we get

.. math::

   \int_e \! \phi_e \frac{\partial q}{\partial t} \, \mathrm{d} x
   = \int_e \! q \nabla \cdot (\phi_e \vec{u}) \, \mathrm{d} x
   - \int_{\partial e} \! \phi_e q \vec{u} \cdot \vec{n}_e \, \mathrm{d} S,
   \qquad \forall \phi_e \in V_e,

where :math:`\vec{n}_e` is an outward-pointing unit normal.

Since :math:`q` is discontinuous, we have to make a choice about how to define
:math:`q` on facets when we assemble the equations globally.  We will use
upwinding: we choose the *upstream* value of :math:`q` on facets, with respect
to the velocity field :math:`\vec{u}`.  We note that there are three types of
facets that we may encounter:

1. Interior facets. Here, the value of :math:`q` from the upstream side, denoted
   :math:`\widetilde{q}`, is used.
2. Inflow boundary facets, where :math:`\vec{u}` points towards the interior.
   Here, the upstream value is the prescribed boundary value :math:`q_\mathrm{in}`.
3. Outflow boundary facets, where :math:`\vec{u}` points towards the outside.
   Here, the upstream value is the interior solution value :math:`q`.

We must now express our problem in terms of integrals over the entire mesh and
over the sets of interior and exterior facets.  This is done by summing our
earlier expression over all elements :math:`e`.  The cell integrals are easy to
handle, since :math:`\sum_e \int_e \cdot  \,\mathrm{d}x = \int_\Omega \cdot \,\mathrm{d}x`.
The interior facet integrals are more difficult to express, since each facet
in the set of interior facets :math:`\Gamma_\mathrm{int}` appears twice in the
:math:`\sum_e \int_{\partial e}`.  In other words, contributions arise from both
of the neighbouring cells.

In Firedrake, the separate quantities in the two cells neighbouring an interior
facet are denoted by + and -.  These markings are arbitrary -- there is no
built-in concept of upwinding, for example -- and the user is responsible for
providing a form that works in all cases.  We will give an example shortly.  The
exterior facet integrals are easier to handle, since each facet in the set of
exterior facets :math:`\Gamma_\mathrm{ext}` appears exactly once in
:math:`\sum_e \int_{\partial e}`. The full equations are then

.. math::

   \int_\Omega \! \phi \frac{\partial q}{\partial t} \, \mathrm{d} x
   = \int_\Omega \! q \nabla \cdot (\phi \vec{u}) \, \mathrm{d} x
   - \int_{\Gamma_\mathrm{int}} \! \widetilde{q}(\phi_+ \vec{u} \cdot \vec{n}_+
     + \phi_- \vec{u} \cdot \vec{n}_-) \, \mathrm{d} S
   - \int_{\Gamma_\rlap{\mathrm{ext, inflow}}} \phi q_\mathrm{in} \vec{u} \cdot
   \vec{n} \, \mathrm{d} s
   - \int_{\Gamma_\rlap{\mathrm{ext, outflow}}} \phi q \vec{u} \cdot
   \vec{n} \, \mathrm{d} s
   \qquad \forall \phi \in V.

As a timestepping scheme, we use the three-stage strong-stability-preserving
Runge-Kutta (SSPRK) scheme from :cite:`Shu:1988`: to discretise
:math:`\frac{\partial q}{\partial t} = \mathcal{L}(q)`, we set

.. math::

   q^{(1)} &= q^n + \Delta t \mathcal{L}(q^n)\\
   q^{(2)} &= \frac{3}{4}q^n + \frac{1}{4}(q^{(1)} + \Delta t \mathcal{L}(q^{(1)}))\\
   q^{n+1} &= \frac{1}{3}q^n + \frac{2}{3}(q^{(2)} + \Delta t \mathcal{L}(q^{(2)}))\\

In this worked example, we reproduce the classic
cosine-bell--cone--slotted-cylinder advection test case of :cite:`LeVeque:1996`.
The domain :math:`\Omega` is the unit square :math:`\Omega = [0,1] \times [0,1]`,
and the velocity field corresponds to solid body rotation
:math:`\vec{u} = (0.5 - y, x - 0.5)`. Each side of the domain has a section of
inflow and a section of outflow boundary.  We therefore perform both the inflow
and outflow integrals over the entire boundary, but construct them so that they
only contribute in the correct places.

As usual, we start by importing Firedrake.  We also import the math library to
give us access to the value of pi.  We use a 40-by-40 mesh of squares. ::

  from firedrake import *
  import math

  mesh = UnitSquareMesh(40, 40, quadrilateral=True)

We set up a function space of discontinous bilinear elements for :math:`q`, and
a vector-valued continuous function space for our velocity field. ::

  V = FunctionSpace(mesh, "DQ", 1)
  W = VectorFunctionSpace(mesh, "CG", 1)

We set up the initial velocity field using a simple analytic expression. ::

  x, y = SpatialCoordinate(mesh)

  velocity = as_vector((0.5 - y, x - 0.5))
  u = Function(W).interpolate(velocity)

Now, we set up the cosine-bell--cone--slotted-cylinder initial coniditon. The
first four lines declare various parameters relating to the positions of these
objects, while the analytic expressions appear in the last three lines. ::

  bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
  cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
  cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
  slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

  bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
  cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
  slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
               conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                 0.0, 1.0), 0.0)

We then declare the inital condition of :math:`q` to be the sum of these fields.
Furthermore, we add 1 to this, so that the initial field lies between 1 and 2,
rather than between 0 and 1.  This ensures that we can't get away with
neglecting the inflow boundary condition.  We also save the initial state so
that we can check the :math:`L^2`-norm error at the end. ::

  q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
  q_init = Function(V).assign(q)

We declare the output filename, and write out the initial condition. ::

  outfile = File("DGadv.pvd")
  outfile.write(q)

We will run for time :math:`2\pi`, a full rotation.  We take 500 steps, giving
a timestep close to the CFL limit.  We declare an extra variable ``dtc``; for
technical reasons, this means that Firedrake does not have to compile new C code
if the user tries different timesteps.  Finally, we define the inflow boundary
condition, :math:`q_\mathrm{in}`.  In general, this would be a ``Function``, but
here we just use a ``Constant`` value. ::

  T = 2*math.pi
  dt = T/500.0
  dtc = Constant(dt)
  q_in = Constant(1.0)

Now we declare our variational forms.  Solving for :math:`\Delta q` at each
stage, the explicit timestepping scheme means that the left hand side is just a
mass matrix. ::

  dq_trial = TrialFunction(V)
  phi = TestFunction(V)
  a = phi*dq_trial*dx

The right-hand-side is more interesting.  We define ``n`` to be the built-in
``FacetNormal`` object; a unit normal vector that can be used in integrals over
exterior and interior facets.  We next define ``un`` to be an object which is
equal to :math:`\vec{u}\cdot\vec{n}` if this is positive, and zero if this is
negative.  This will be useful in the upwind terms.  We now define our
right-hand-side form ``L1`` as :math:`\Delta t` times the sum of four integrals.

The first integral is a straightforward cell integral of
:math:`q\nabla\cdot(\phi\vec{u})`.  The second integral represents the inflow
boundary condition.  We only want this to contribute on the inflow part of the
boundary, where :math:`\vec{u}\cdot\vec{n} < 0` (recall that :math:`\vec{n}` is
an outward-pointing normal).  Where this is true, the condition gives the
desired expression :math:`\phi q_\mathrm{in}\vec{u}\cdot\vec{n}`, otherwise the
condition gives zero.  The third integral operates in a similar way to give
the outflow boundary condition.  The last integral represents the integral
:math:`\widetilde{q}(\phi_+ \vec{u} \cdot \vec{n}_+ + \phi_- \vec{u} \cdot \vec{n}_-)`
over interior facets.  We could again use a conditional in order to represent
the upwind value :math:`\widetilde{q}` by the correct choice of :math:`q_+` or
:math:`q_-`, depending on the sign of :math:`\vec{u}\cdot\vec{n_+}`, say.
Instead, we make use of the quantity ``un``, which is either
:math:`\vec{u}\cdot\vec{n}` or zero, in order to avoid writing explicit
conditionals. Although it is not obvious at first sight, the expression given in
code is equivalent to the desired expression, assuming
:math:`\vec{n}_- = -\vec{n}_+`. ::

  n = FacetNormal(mesh)
  un = 0.5*(dot(u, n) + abs(dot(u, n)))

  L1 = dtc*(q*div(phi*u)*dx
            - conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
            - conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds
            - (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)

In our Runge-Kutta scheme, the first step uses :math:`q^n` to obtain
:math:`q^{(1)}`.  We therefore declare similar forms that use :math:`q^{(1)}`
to obtain :math:`q^{(2)}`, and :math:`q^{(2)}` to obtain :math:`q^{n+1}`. We
make use of UFL's ``replace`` feature to avoid writing out the form repeatedly. ::

  q1 = Function(V); q2 = Function(V)
  L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2})

We now declare a variable to hold the temporary increments at each stage. ::

  dq = Function(V)

Since we want to perform hundreds of timesteps, ideally we should avoid
reassembling the left-hand-side mass matrix each step, as this does not change.
We therefore make use of the ``LinearVariationalProblem`` and
``LinearVariationalSolver`` objects for each of our Runge-Kutta stages. These
cache and reuse the assembled left-hand-side matrix.  Since the DG mass matrices
are block-diagonal, we use the 'preconditioner' ILU(0) to solve the linear
systems. ::

  params = {'ksp_type': 'preonly', 'pc_type': 'ilu'}
  prob1 = LinearVariationalProblem(a, L1, dq)
  solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
  prob2 = LinearVariationalProblem(a, L2, dq)
  solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
  prob3 = LinearVariationalProblem(a, L3, dq)
  solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

We now run the time loop.  This consists of three Runge-Kutta stages, and every
20 steps we write out the solution to file and print the current time to the
terminal. ::

  t = 0.0
  step = 0
  while t < T - 0.5*dt:
      solv1.solve()
      q1.assign(q + dq)

      solv2.solve()
      q2.assign(0.75*q + 0.25*(q1 + dq))

      solv3.solve()
      q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))

      step += 1
      t += dt

      if step % 20 == 0:
          outfile.write(q)
          print "t=", t

Finally, we display the normalised :math:`L^2` error, by comparing to the
initial condition. ::

  L2_err = sqrt(assemble((q - q_init)*(q - q_init)*dx))
  L2_init = sqrt(assemble(q_init*q_init*dx))
  print L2_err/L2_init

This demo can be found as a script in
`DG_advection.py <DG_advection.py>`__.


.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
