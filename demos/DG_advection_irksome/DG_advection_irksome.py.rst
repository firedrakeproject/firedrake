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
   &= \int_\Omega \! q \nabla \cdot (\phi \vec{u}) \, \mathrm{d} x\\
   &\quad- \int_{\Gamma_\mathrm{int}} \! \widetilde{q}(\phi_+ \vec{u} \cdot \vec{n}_+
     + \phi_- \vec{u} \cdot \vec{n}_-) \, \mathrm{d} S\\
   &\quad- \int_{\Gamma_{\mathrlap{\mathrm{ext, inflow}}}} \phi q_\mathrm{in} \vec{u} \cdot
   \vec{n} \, \mathrm{d} s\\
   &\quad- \int_{\Gamma_{\mathrlap{\mathrm{ext, outflow}}}} \phi q \vec{u} \cdot
   \vec{n} \, \mathrm{d} s
   \qquad \forall \phi \in V.

For timestepping we use Irksome, specifcally the explicit Strong Stability 
Preserving Runge-Kutta method. 

In this worked example, we reproduce the classic
cosine-bell--cone--slotted-cylinder advection test case of :cite:`LeVeque:1996`.
The domain :math:`\Omega` is the unit square :math:`\Omega = [0,1] \times [0,1]`,
and the velocity field corresponds to solid body rotation
:math:`\vec{u} = (0.5 - y, x - 0.5)`. Each side of the domain has a section of
inflow and a section of outflow boundary.  We therefore perform both the inflow
and outflow integrals over the entire boundary, but construct them so that they
only contribute in the correct places.

As usual, we start by importing Firedrake.  We also import the math library to
give us access to the value of pi. ::

    from firedrake import (UnitSquareMesh, FunctionSpace, as_vector, sqrt,
                        VectorFunctionSpace, SpatialCoordinate, And,
                        Function, min_value, conditional, Constant,
                        TestFunction, FacetNormal, dot, div, ds, dS,
                        dx, inner, cos, assemble, errornorm, warning)

    from firedrake.pyplot import FunctionPlotter, tripcolor
    import math
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

We then imoprt Irksome which gives us access to our time stepper and use our
explicit Strong Stability Preserving Runge-Kutta method. ::

    try:
        from irksome import TimeStepper, Dt, SSPButcherTableau
    except ImportError:
        import sys
        warning("This demo requires Irksome to be installed.")
        sys.exit(0)

    butcher_tableau = SSPButcherTableau(3, 3)

    ns = butcher_tableau.num_stages

We use a 40-by-40 mesh of squares. ::

    n = 40
    mesh = UnitSquareMesh(n, n, quadrilateral=True)

We set up a function space of discontinuous bilinear elements for :math:`q`, and
a vector-valued continuous function space for our velocity field. ::    

    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)

We set up the initial velocity field using a simple analytic expression. ::

    x, y = SpatialCoordinate(mesh)

    velocity = as_vector((0.5 - y, x - 0.5))
    u = Function(W).interpolate(velocity)

Now, we set up the cosine-bell--cone--slotted-cylinder initial condition. The
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

We then declare the initial condition of :math:`q` to be the sum of these fields.
Furthermore, we add 1 to this, so that the initial field lies between 1 and 2,
rather than between 0 and 1.  This ensures that we can't get away with
neglecting the inflow boundary condition.  We also save the initial state so
that we can check the :math:`L^2`-norm error at the end. ::

    q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
    q_init = Function(V).assign(q)

Next we'll create a list to store the function values at every timestep so that
we can make a movie of them later. ::

    qs = []

We will run for time :math:`2\pi`, a full rotation.  We take 600 steps, giving
a timestep close to the CFL limit.  We declare an extra variable ``dtc``; for
technical reasons, this means that Firedrake does not have to compile new C code
if the user tries different timesteps.  Finally, we define the inflow boundary
condition, :math:`q_\mathrm{in}`.  In general, this would be a ``Function``, but
here we just use a ``Constant`` value. ::

    T = 2*math.pi
    dt = T/600.0
    dtc = Constant(dt)
    q_in = Constant(1.0)

Now we declare our variational forms.  Solving for :math:`\Delta q` at each
stage, the explicit timestepping scheme means that the left hand side is just a
mass matrix. ::

    phi = TestFunction(V)

The right-hand-side is more interesting.  We define ``n`` to be the built-in
``FacetNormal`` object; a unit normal vector that can be used in integrals over
exterior and interior facets.  We next define ``un`` to be an object which is
equal to :math:`\vec{u}\cdot\vec{n}` if this is positive, and zero if this is
negative.  This will be useful in the upwind terms. ::

    n = FacetNormal(mesh)
    un = 0.5*(dot(u, n) + abs(dot(u, n)))

We now define our right-hand-side form ``F`` as :math:`\Delta t` times the
sum of four integrals.

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

    F = inner(Dt(q), phi)*dx - inner(q, div(phi*u))*dx + (
            conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
            + conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds
            + (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)

We then set our parameters. Since the DG mass matrices
are block-diagonal, we use the 'preconditioner' ILU(0) to solve the linear
systems. As a minor technical point, we in fact use an outer block Jacobi
preconditioner. This allows the code to be executed in parallel without any
further changes being necessary. ::

    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

We now use our time stepper with the stage type explicit using the parameters
we just set. ::

    t = Constant(0)
    stepper = TimeStepper(F, butcher_tableau, t, dtc, q, stage_type="explicit",
                        solver_parameters=params)

We now run the time loop and every 20 steps we write out the solution to file and
print the current time to the terminal. ::

    step = 0
    output_freq = 20
    while float(t) < T - 0.5*dt:
        stepper.advance()

        step += 1
        t.assign(float(t)+dt)

        print(float(t))

        if step % output_freq == 0:
            qs.append(q.copy(deepcopy=True))
            # print("t=", t)

To check our solution, we display the normalised :math:`L^2` error, by comparing
to the initial condition. ::

    L2_err = errornorm(q_init, q)
    L2_init = sqrt(assemble(q_init*q_init*dx))
    print(L2_err/L2_init)

Finally, we'll animate our solution using matplotlib. We'll need to evaluate
the solution at many points in every frame of the animation, so we'll employ a
helper class that pre-computes some relevant data in order to speed up the
evaluation. ::

    nsp = 16
    fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)

We first set up a figure and axes and draw the first frame. ::

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(q_init, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
    fig.colorbar(colors)

Now we'll create a function to call in each frame. This function will use the
helper object we created before. ::

    def animate(q):
        colors.set_array(fn_plotter(q))

The last step is to make the animation and save it to a file. ::

    interval = 1e3 * output_freq * dt
    animation = FuncAnimation(fig, animate, frames=qs, interval=interval)
    try:
        animation.save("DG_advection_irksome.mp4", writer="ffmpeg")
    except:
        print("Failed to write movie! Try installing `ffmpeg`.")
