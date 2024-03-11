Scalar wave equation with higher-order mass lumping
===================================================

Introduction
************

In this demo, we solve the scalar wave equation with a fully explicit, higher-order
(up to degree 5) mass lumping technique for triangular and tetrahedral meshes.
This scalar wave equation is widely used in seismology to model seismic waves and is especially popular
in algorithms for geophysical exploration such as Full Waveform
Inversion and Reverse Time Migration. This tutorial demonstrates how to
use the mass-lumped triangular elements originally discovered in
:cite:`Chin:1999` and later improved upon in :cite:`Geevers:2018` in the
Firedrake computing environment.**

*The short tutorial was prepared by `Keith J. Roberts <mailto:krober@usp.br>`__*


The scalar wave equation is:

.. math::

    \rho \partial_{t}^2 u = \nabla \cdot c^2 \nabla u + f

    u = 0

    u \vert_{t=0} = u_0

    \partial_{t} u \vert_{t=0} = v_0

where :math:`c` is the scalar wave speed and :math:`rho` is the density (assumed to be 1 for simplicity).

The weak formulation is finding :math:`u \in V` such that:

.. math::

    <\partial_t(\rho \partial_t u), v> + a(u,v) = (f,w)

where :math:`<\cdot, \cdot>` denotes the pairing between :math:`H^{-1}(\Omega)` and :math:`H^{1}_{0}(\Omega)`, :math:`(\cdot, \cdot)` denotes the :math:`L^{2}(\Omega)` inner product, and :math:`a(\cdot, \cdot) : H^{1}_{0}(\Omega) \times H^{1}_{0}(\Omega)\rightarrow \mathbb{R}` is the elliptic operator given by:

.. math::

    a(u, v) := \int_{\Omega} c^2 \nabla u \cdot \nabla v  \mathrm d x

We solve the above weak formulation using the finite element method.

In the work of :cite:`Chin:1999` and later :cite:`Geevers:2018`, several triangular and tetrahedral elements were discovered that could produce convergent and stable mass lumping for :math:`p \ge 2`. These elements have enriched function spaces in the interior of the element that lead to more degree-of-freedom per element than the standard Lagrange element. However, this additional computational cost is offset by the fact that these elements produce diagonal matrices that are comparatively quick to solve, which improve simulation throughput especially at scale. Firedrake supports (through FInAT) these elements up to degree 5 on triangular, and degree 3 on tetrahedral meshes. They can be selected by choosing the "KMV" finite element.

In addition to importing firedrake as usual, we will need to construct the correct quadrature rules for the mass-lumping by hand. FInAT is responsible for providing these quadrature rules, so we import it here too.::

    from firedrake import *
    from firedrake.output import VTKFile
    import finat

    import math

A simple uniform triangular mesh is created::

    mesh = UnitSquareMesh(50, 50)

We choose a degree 2 `KMV` continuous function space, set it up and then create some functions used in time-stepping::

    V = FunctionSpace(mesh, "KMV", 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    u_np1 = Function(V)  # timestep n+1
    u_n = Function(V)    # timestep n
    u_nm1 = Function(V)  # timestep n-1

.. note::
    The user can select orders up to p=5 for triangles and up to p=3 for tetrahedra.

We create an output file to hold the simulation results::

    outfile = VTKFile("out.pvd")

Now we set the time-stepping variables performing a simulation for 1 second with a timestep of 0.001 seconds::

    T = 1.0
    dt = 0.001
    t = 0
    step = 0

Ricker wavelets are often used to excite the domain in seismology. They have one free parameter: a peak frequency :math:`\text{peak}`.

Here we inject a Ricker wavelet into the domain with a frequency of 6 Hz. For simplicity, we set the seismic velocity in the domain to be a constant::

    freq = 6
    c = Constant(1.5)

The following two functions are used to inject the Ricker wavelet source into the domain. We
create a time-varying function to model the time evolution of the Ricker wavelet::

    def RickerWavelet(t, freq, amp=1.0):
        # Shift in time so the entire wavelet is injected
        t = t - (math.sqrt(6.0) / (math.pi * freq))
        return amp * (
            1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )

The spatial distribution of the source function is a Guassian kernel with a standard deviation
of 2,000 so that it's sufficiently localized to emulate a Dirac delta function::

    def delta_expr(x0, x, y, sigma_x=2000.0):
        sigma_x = Constant(sigma_x)
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))

To assemble the diagonal mass matrix, we need to create the matching colocated quadrature rule.
FInAT implements custom "KMV" quadrature rules to do this. We obtain the appropriate cell from the function
space, along with the degree of the element and construct the quadrature rule::

    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

Then we make a new Measure object that uses this rule::

    dxlump=dx(scheme=quad_rule)

To discretize :math:`\partial_{t}^2 u` we use a central scheme

.. math::

    \partial_{t}^2 u = \frac{u^{n+1} - 2*u^{n} + u^{n-1}}{\Delta t^2}

Substituting the above into the time derivative term in the variational form leads to

.. math::

    \frac{u^{n+1} - 2*u^{n} + u^{n-1}}{\Delta t^2}), v> + a(u,v) = (f,w)

Using Firedrake, we specify the mass matrix using the special quadrature rule with the Measure object we created above like so::

    m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dxlump

.. note::
    Mass lumping is a common technique in finite elements to produce a diagonal mass matrix that can be trivially inverted resulting in a in very efficient explicit time integration scheme. It's usually done with nodal basis functions and an inexact quadrature rule for the mass matrix. A diagonal matrix is obtained when the integration points coincide with the nodes of the basis function. However, when using elements of :math:`p \ge 2`, this technique does not result in a stable and accurate finite element scheme and new elements must be found such as those detailed in :cite:Chin:1999 .

The stiffness matrix :math:`a(u,v)` is formed using a standard quadrature rule and is treated explicitly::

    a = c*c*dot(grad(u_n), grad(v)) * dx

The source is injected at the center of the unit square::

    x, y = SpatialCoordinate(mesh)
    source = Constant([0.5, 0.5])
    ricker = Constant(0.0)
    ricker.assign(RickerWavelet(t, freq))

We also create a cofunction `R` to save the assembled RHS vector::

    R = Cofunction(V.dual())

Finally, we define the whole variational form :math:`F`, assemble it, and then create a cached PETSc `LinearSolver` object to efficiently timestep with::

    F = m + a -  delta_expr(source, x, y)*ricker * v * dx
    a, r = lhs(F), rhs(F)
    A = assemble(a)
    solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

.. note::
    Since we have arranged that the matrix A is diagonal, we can invert it with a single application of Jacobi iteration. We select this here using    appropriate solver parameters, which tell PETSc to construct a solver which just applies a single step of Jacobi preconditioning.

Now we are ready to start the time-stepping loop::

    step = 0
    while t < T:
        step += 1

        # Update the RHS vector according to the current simulation time `t`

        ricker.assign(RickerWavelet(t, freq))

        R = assemble(r, tensor=R)

        # Call the solver object to do point-wise division to solve the system.

        solver.solve(u_np1, R)

        # Exchange the solution at the two time-stepping levels.

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        # Increment the time and write the solution to the file for visualization in ParaView.

        t += dt
        if step % 10 == 0:
            print("Elapsed time is: "+str(t))
            outfile.write(u_n, time=t)

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
