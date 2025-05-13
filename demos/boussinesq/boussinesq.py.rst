Steady Boussinesq problem with integral constraints
=========================

.. rst-class:: emphasis

    This demo demonstrates 

    The demo was contributed by `Aaron Baier-Reinio
    <mailto:baierreinio@maths.ox.ac.uk>`__ and `Kars Knook
    <mailto:knook@maths.ox.ac.uk>`__.

We consider a steady Boussinesq problem in nondimensionalised form.
The two-dimensional domain is denoted by :math:`\Omega \subset \mathbb{R}^2`
with boundary :math:`\Gamma`. The boundary value problem is:

.. math::

    -\nabla \cdot (\mu(T) \epsilon (u)) + \nabla p &= f T \quad \textrm{in}\ \Omega,

    \nabla \cdot u &= 0 \quad \textrm{in}\ \Omega,

    u \cdot \nabla T - \nabla^2 T = 0 \quad \textrm{in}\ \Omega,

    u &= 0 \quad \textrm{on}\ \Gamma,

    \nabla T \cdot \vec{n} &= g \quad \textrm{on}\ \Gamma.

The unknowns are the :math:`\mathbb{R}^2`-valued velocity field :math:`u`,
scalar pressure field :math:`p` and temperature field :math:`T`.
The viscosity is :math:`\mu(T)` and is assumed to be a known function of :math:`T`.
Moreover :math:`\epsilon (u)` denotes the symmetric gradient of :math:`u`
and :math:`f = (0, -1)^T` the acceleration due to gravity.
Inhomogeneous Neumann boundary conditions are imposed on :math:`T` through :math:`g` which
must satisfy a compatibility condition

.. math::

    \int_{\Gamma} g \ {\rm d} s = 0.

Evidently the pressure :math:`p` is only determined up to a constant, since it only appears in
the problem through its gradient. This choice of constant is arbitrary and does not affect the model.
The situation regarding the temperature :math:`T` is however more subtle.
For the sake of discussion let us first assume that :math:`\mu(T) = \mu_0` is a constant that does
not depend on :math:`T`. It is then clear that, just like the pressure, the temperature :math:`T`
is undetermined up to a constant. We pin down this constant by enforcing that the mean of :math:`T`
is a user-supplied constant :math:`T_0`:

.. math::

    \int_{\Omega} (T - T_0) \ {\rm d} x.

In the Boussinesq approximation the density is assumed to vary linearly with the temperature;
hence this integral constraint can be viewed as an imposition of the total mass of fluid in :math:`Omega`.

Now suppose that :math:`\mu(T)` does depend on :math:`T`.
For simplicity we use a crude power-law scaling :math:`\mu(T) = T**{1/2}`
(the precise functional form of :math:`\mu(T)` is unimportant in this demo).
We must still impose the integral constraint on :math:`T` to obtain a unique solution,
but the value of :math:`T_0` now affects the model in a non-trivial way since :math:`\mu(T)` and 
:math:`T` are coupled. This is demonstrated in the numerical experiments below.

We begin by setting up the mesh using :doc:`netgen <netgen_mesh.py>`.
We choose a trapezoidal geometry as this will prevent hydrostatic equilibrium and ensure
that the velocity field is non-zero.
::

    from firedrake import *
    import netgen.occ as ngocc

    wp = ngocc.WorkPlane()
    wp.MoveTo(0, 0)
    wp.LineTo(2, 0)
    wp.LineTo(2, 1)
    wp.LineTo(0, 2)
    wp.LineTo(0, 0)

    shape = wp.Face()
    shape.edges.Min(ngocc.X).name = "left"
    shape.edges.Max(ngocc.X).name = "right"

    ngmesh = ngocc.OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.1)

    left_id = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "left"]
    right_id = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "right"]

    mesh = Mesh(ngmesh)
    x, y = SpatialCoordinate(mesh)

Next we set up the discrete function spaces.
We use lowest-order Taylor--Hood elements for the velocity and pressure,
and continuous piecewise-linear elements for the temperature.
We also introduce a Lagrange multiplier as this will enable us to enforce
the integral constraint on the temperature.::

    U = VectorFunctionSpace(mesh, "CG", degree=2)
    V = FunctionSpace(mesh, "CG", degree=1)
    W = FunctionSpace(mesh, "CG", degree=1)
    R = FunctionSpace(mesh, "R", 0)

    Z = U * V * W * R

The trial and test functions are::

    z = Function(Z)
    (u, p, T_aux, l) = split(z)
    (v, q, w, s) = split(TestFunction(Z))

    T = T_aux + l

The test Lagrange multiplier :math:`s` will allow us to impose the integral constraint on the temperature.
We use the trial Lagrange multiplier :math:`l` by decomposing the discretized temperature field :math:`T`
as a sum :math:`T = T_{\textrm{aux}} + l` where `T_{\textrm{aux}}` is the trial function from :math:`W`.
The value of :math:`l` will then be determined by the integral constraint on :math:`T`.

The remaining problem data to be specified is the Neumann data,
viscosity, acceleration due to gravity and :math:`T_0`.
For the Neumann data we choose parabolic data on the left and right edges, zero data on the top and bottom.
