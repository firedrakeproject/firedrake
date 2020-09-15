
Scalar wave equation with higher-order mass lumping
===================================================


    **Here we focus on solving the scalar wave equation with a
    fully-explicit, higher-order (e.g., :math:`p < 5`) mass
    lumping technique. This scalar wave equation is widely used
    in seismology to model seismic waves and is especially popular
    in algorithms for geophysical exploration such as Full Waveform
    Inversion and Reverse Time Migration. This tutorial demonstrates how to
    use the mass-lumped triangular elements originally discovered in
    :cite:`Kong:1999` and later improved in :cite:`Geevers` into the
    Firedrake environment.**

    **The tutorial was prepared by `Keith J. Roberts
    <mailto:krober@usp.br>`__**


The scalar wave equation solved here is:

.. math::

    \rho \partial_{t}^2 u = \nabla \cdot c \nabla u + f

    u = 0

    u \vert_{t=0} = u_0

    \partial_{t} u \vert_{t=0} = v_0

The weak formulation is finding :math:`u \in V` such that:

.. math::

    <\partial_t(\rho \partial_t u), v> + a(u,v) = (f,w)

Here :math:`<\cdot, \cdot>` denotes the pairing between :math:`H^{-1}(\Omega)` and :math:`H^{1}_{0}(\Omega)`, :math:`(\cdot, \cdot)` denotes the :math:`L^{2}(\Omega)` inner product, and :math:`a(\cdot, \cdot) : H^{1}_{0}(\Omega) \times H^{1}_{0}(\Omega)\rightarrow ℝ` is the elliptical operator given by:

.. math::

    a(u, v) := \int_{\Omega} c \nabla u \cdot \nabla v  \mathrm d x

We solve the above weak formulation using the finite element method.

In time, we use simple central scheme that is formally 2nd order accurate.

.. note::
    Mass lumping is a common technique in finite elements to produce a diagonal mass matrix that can be trivially inverted and thus result in very efficient explicit time integration scheme. It's usually done with nodal basis functions and an inexact quadrature rule for the mass matrix. A diagonal matrix is obtained when the integration points coincide with the nodes of the basis function. However, when using elements of :math:`p \ge 2`, this technique does not result in a stable and accurate finite element scheme.

In the work of :cite:`Kong:1999` and later :cite:`Geevers:2018`, several triangular and tetrahedral elements were discovered that could produce convergent and stable mass lumping for :math:`p \ge 2`. Here we implemented eight of them for use within the Firedrake computing environment (e.g., five triangular elements :math:`p \le 5` and three tetrahdrals up to :math:`p \le 3`). We honorifically refer to the elements as `KMV` elements (i.e., Kong-Mulder-Veldhuizen).

We can then start our Python script loading Firedrake *and* fiat/finat, which is used to build the special quadrature rules::

    from firedrake import *
    import FIAT
    import finat
    import math

A simple triangular mesh is created::

    mesh = UnitSquareMesh(50, 50)

We choose a degree 2 `KMV` continuous function space, and set up the
function space, the function to hold the seismic velocity, and some functions used in timestepping::

    V = FunctionSpace(mesh, "KMV", 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    c = Function(V) # Acoustic wave velocity

    u_np1 = Function(V)  # timestep n+1
    u_n = Function(V)    # timestep n
    u_nm1 = Function(V)  # timestep n-1

.. note::
    The user can select orders up to P=5 for triangles and up to P=3 for tetrahedral.

Output file to hold the simulation results::

    outfile = File("out.pvd")

Now we set the time-stepping variables::

    T = 1.0
    dt = 0.001
    t = 0
    step = 0

In seismology, often [Ricker wavelets](https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet) are used to force the domain, which have only one parameter: a peak frequency :math:`freq`. Here we inject a Ricker forcing function into the domain. We also specify the seismic velocity in the domain :math:`c` to be a constant however this is arbitrary::

    freq = 6
    c = Function(V).assign(1.5)

The following two functions are used to inject the Ricker wavelet source into the domain::

    # Source function
    def RickerWavelet(t, freq, amp=1.0):
        # shift so the entire waveform is injected
        t = t - (math.sqrt(6.0) / (math.pi * freq))
        return amp * (
            1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )


    # Kernel function to apply the source function
    def delta_expr(x0, x, y, sigma_x=2000.0):
        sigma_x = Constant(sigma_x)
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))

In order to achieve a diagonal mass matrix, a custom quadrature rule must be used (note we specify the degree here too)::

    Tria = FIAT.reference_element.UFCTriangle()
    qr_rule = finat.quadrature.make_quadrature(Tria, 2, "KMV")

Here we set up the mass matrix and specify the special quadrature rule to render the matrix diagonal::

    m = (1.0 / (c * c)) * (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dx(rule=qr_rule)

The stiffness matrix is treated explictly::

    a = dot(grad(u_n), grad(v)) * dx

The source term is injected into the central of the unit square::

    x, y = SpatialCoordinate(mesh)
    source = Constant([0.5, 0.5])
    delta = Interpolator(delta_expr(source, x, y), V)
    ricker = Constant(0.0)
    expr = Function(delta.interpolate()) * ricker
    ricker.assign(RickerWavelet(t, freq))
    f = Function(V).assign(expr)

Finally, we define our variational form :math:`F`, assemble it, and then create a cached PETSc solver object to efficiently timestep with::

    F = m + a - f * v * dx
    a, r = lhs(F), rhs(F)
    A = assemble(a)
    solver = LinearSolver(A, P=None, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

.. note::
    We inform PETSc to not solve anything by passing an dictionary of options! These options tell PETSc to only do a simple Jacobi pre-conditioning step, which for our case solves our diagonal system exactly.

Now we are ready to start the time-stepping loop::

    step = 0
    while t < T:
        step += 1

        # Update the RHS vector accordingly

        ricker.assign(RickerWavelet(t, freq))
        f.assign(expr)

        R = assemble(r)

        # Call the solver object to do pointwise division and solve the system.

        solver.solve(u_np1, R)

        # Exchange the solution at the timestepping levels.

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        # Write the solution to the file for visualization in ParaView.

        t += dt
        if step % 10 == 0:
            print("Elapsed time is: "+str(t))
            outfile.write(u_n, time=t)

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
