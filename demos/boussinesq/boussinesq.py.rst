Steady Boussinesq problem with integral constraints
===================================================

.. rst-class:: emphasis

    This demo demonstrates how integral constraints can be imposed on a field
    that appears nonlinearly in the governing equations.

    The demo was contributed by `Aaron Baier-Reinio
    <mailto:baierreinio@maths.ox.ac.uk>`__ and `Kars Knook
    <mailto:knook@maths.ox.ac.uk>`__.

We consider a nondimensionalised steady Boussinesq problem.
The domain is :math:`\Omega \subset \mathbb{R}^2`
with boundary :math:`\Gamma`. The boundary value problem is:

.. math::

    -\nabla \cdot (\mu(T) \epsilon (u)) + \nabla p - f T &= 0 \quad \textrm{in}\ \Omega,

    \nabla \cdot u &= 0 \quad \textrm{in}\ \Omega,

    u \cdot \nabla T - \nabla^2 T &= 0 \quad \textrm{in}\ \Omega,

    u &= 0 \quad \textrm{on}\ \Gamma,

    \nabla T \cdot n &= g \quad \textrm{on}\ \Gamma.

The unknowns are the :math:`\mathbb{R}^2`-valued velocity :math:`u`,
scalar pressure :math:`p` and temperature :math:`T`.
The viscosity :math:`\mu(T)` is assumed to be a known function of :math:`T`.
Moreover :math:`\epsilon (u)` denotes the symmetric gradient of :math:`u`
and :math:`f = (0, -1)^T` the acceleration due to gravity.
Inhomogeneous Neumann boundary conditions are imposed on :math:`\nabla T \cdot n` through 
given data :math:`g` which must satisfy a compatibility condition

.. math::

    \int_{\Gamma} g \ {\rm d} s = 0.

Evidently the pressure :math:`p` is only determined up to a constant, since it only appears in
the problem through its gradient. This choice of constant is arbitrary and does not affect the model.
The situation regarding the temperature :math:`T` is, however, more subtle.
For the sake of discussion let us first assume that :math:`\mu(T) = \mu_0` is a constant that does
not depend on :math:`T`. It is then clear that, just like the pressure, the temperature :math:`T`
is undetermined up to a constant. We shall pin this down by enforcing that the mean of :math:`T`
is a user-supplied constant :math:`T_0`,

.. math::

    \int_{\Omega} (T - T_0) \ {\rm d} x = 0.

The Boussinesq approximation assumes that density varies linearly with temperature.
Hence, this constraint can be viewed as an indirect imposition on the total mass of fluid in :math:`\Omega`.

Now suppose that :math:`\mu(T)` does depend on :math:`T`.
For simplicity we use a basic power law :math:`\mu(T) = 10^{-4} \cdot T^{1/2}`
but emphasise that the precise functional form of :math:`\mu(T)` is unimportant to this demo.
We must still impose the integral constraint on :math:`T` to obtain a unique solution,
but the value of :math:`T_0` now affects the model in a non-trivial way since :math:`\mu(T)` and 
:math:`T` are coupled (c.f. the figures at the bottom of the demo).
**In particular, this is not a "trivial" situation like the incompressible
Stokes problem where the discretised Jacobians have a nullspace corresponding to the constant pressures.
Instead, we have an integral constraint on** :math:`T` **even though
the discretised Jacobians do not have a nullspace corresponding to the constant temperatures.**

We build the mesh using :doc:`netgen <netgen_mesh.py>`, choosing a trapezoidal geometry
to prevent hydrostatic equilibrium and allow for a non-trivial velocity solution. ::

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
We introduce a Lagrange multiplier to enforce the integral constraint on :math:`T`::

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

The test Lagrange multiplier :code:`s` will allow us to impose the integral constraint on the temperature.
We use the trial Lagrange multiplier :code:`l` by decomposing the discretised temperature field :code:`T`
as :code:`T = T_aux + l` where :code:`T_aux` is the trial function from :code:`W`.
The value of :code:`l` will then be determined by the integral constraint on :code:`T`.

The remaining problem data to be specified is the Neumann data,
viscosity, acceleration due to gravity and :math:`T_0`.
For the Neumann data we choose a parabolic profile on the left and right edges,
and zero data on the top and bottom. ::

    g_left = y*(y-2)                # Neumann data on the left
    g_right = -8*y*(y-1)            # Neumann data on the right
    mu = (1e-4) * T ** (1/2)        # Viscosity
    f = as_vector([0, -1])          # Acceleration due to gravity
    T0 = Constant(1)                # Mean of the temperature

The nonlinear form for the problem is::

    F = (mu * inner(sym(grad(u)), sym(grad(v))) * dx    # Viscous terms
     - inner(p, div(v)) * dx                            # Pressure gradient
     - inner(f, T*v) * dx                               # Buoyancy term
     - inner(div(u), q) * dx                            # Incompressibility constraint
     + inner(dot(u, grad(T)), w) * dx                   # Temperature advection term
     + inner(grad(T), grad(w)) * dx                     # Temperature diffusion term
     - inner(g_left, w) * ds(tuple(left_id))            # Weakly imposed Neumann BC terms
     - inner(g_right, w) * ds(tuple(right_id))          # Weakly imposed Neumann BC terms
     + inner(T - T0, s) * dx                            # Integral constraint on T
     )

and the (strongly enforced) Dirichlet boundary conditions on :math:`u` are enforced by::

    bc_u = DirichletBC(Z.sub(0), 0, "on_boundary")

At this point we could form and solve a :class:`~.NonlinearVariationalProblem`
using :code:`F` and :code:`bc_u`. However, the resultant problem has a nullspace of
dimension 2, corresponding to (i) shifting :math:`p` by a constant :math:`C_1`
and (ii) shifting :math:`l` by a constant :math:`C_2` while simultaneuosly shifting
:math:`T_{\textrm{aux}}` by :math:`-C_2`.

One way of dealing with nullspaces in Firedrake is to pass a :code:`nullspace` and
:code:`transpose_nullspace` to :class:`~.NonlinearVariationalSolver`. However, sometimes 
this approach may not be practical. First, for nonlinear problems with Jacobians that 
are not symmetric, it may not obvious what the :code:`transpose_nullspace` is. A second 
reason is that, when using customised PETSc linear solvers, it may be desirable
to directly eliminate the nullspace from the assembled Jacobian matrix, since one
cannot always be sure that the linear solver at hand is correctly utilising the provided
:code:`nullspace` and :code:`transpose_nullspace`.

To directly eliminate the nullspace we introduce a class :code:`FixAtPointBC` which
implements a boundary condition that fixes a field at a single point. ::

    import functools

    class FixAtPointBC(DirichletBC):
       r'''A special BC object for pinning a function at a point.

       :arg V: the :class:`.FunctionSpace` on which the boundary condition should be applied.
       :arg g: the boundary condition value.
       :arg bc_point: the point at which to pin the function.
           The location of the finite element DOF nearest to bc_point is actually used.
       '''
       def __init__(self, V, g, bc_point):
           super().__init__(V, g, bc_point)

       @functools.cached_property
       def nodes(self):
           V = self.function_space()
           if V.mesh().ufl_coordinate_element().degree() != 1:
               # Ensure a P1 mesh
               coordinates = V.mesh().coordinates
               P1 = coordinates.function_space().reconstruct(degree=1)
               P1_mesh = Mesh(Function(P1).interpolate(coordinates))
               V = V.reconstruct(mesh=P1_mesh)

           point = [tuple(self.sub_domain)]
           vom = VertexOnlyMesh(V.mesh(), point)
           P0 = FunctionSpace(vom, "DG", 0)
           Fvom = Cofunction(P0.dual()).assign(1)

           # Take the basis function with the largest abs value at bc_point
           v = TestFunction(V)
           F = assemble(Interpolate(inner(v, v), Fvom))
           with F.dat.vec as Fvec:
               max_index, _ = Fvec.max()
           nodes = V.dof_dset.lgmap.applyInverse([max_index])
           nodes = nodes[nodes >= 0]
           return nodes
 
We use this to fix the pressure and auxiliary temperature at the origin::

    aux_bcs = [FixAtPointBC(Z.sub(1), 0, as_vector([0, 0])), 
               FixAtPointBC(Z.sub(2), 0, as_vector([0, 0]))]

:code:`FixAtPointBC` takes three arguments: the function space to fix, the value with which it
will be fixed, and the location at which to fix. Generally :code:`FixAtPointBC` will not fix
the function at exactly the supplied point; it will fix it at the finite element DOF closest
to that point. By default CG elements have DOFs on all mesh vertices, so if the supplied
point if a mesh vertex then CG fields will be fixed at exactly the supplied point.

.. warning::

    A :code:`FixAtPointBC` does more than just fix the corresponding trial function
    at the chosen DOF. It also ensures that the corresponding test function
    (which is equal to one at that DOF and zero at all others)
    will no longer be used in the discretised variational problem.
    One must be sure that it is mathematically acceptable to do this.

    In the present setting this is acceptable owing to the homogeneous Dirichlet
    boundary conditions on :math:`u` and compatibility condition :math:`\int_{\Gamma} g \ {\rm d} s = 0`
    on the Neumann data. The former ensures that the rows in the discretised Jacobian
    corresponding to the incompressibility constraint are linearly dependent
    (if there are :math:`M` rows, only :math:`M-1` of them are linearly independent, since
    the boundary conditions on :math:`u` ensure that 
    :math:`\int_{\Omega} \nabla \cdot u \ {\rm d} x = 0` automatically).
    Similarily the rows in the Jacobian corresponding to the temperature advection-diffusion
    equation are linearly independent (again, if there are :math:`M` rows, 
    only :math:`M-1` of them are linearly independent).
    The effect of :code:`FixAtPointBC` will be to remove one of the rows corresponding
    to the incompressibility constraint and one corresponding to the temperature advection-diffusion
    equation. Which row ends up getting removed is determined by the location of :code:`bc_point`,
    but in the present setting removing a given row is mathematically equivalent to removing any one of the others.

    One could envisage a more complicated scenario than the one in this demo, wherein the Neumann data
    depends nonlinearly on some other problem unknowns, and it only satisfies the compatibility condition
    approximately (e.g. up to some discretization error).
    In this case one would have to be very careful when using :code:`FixAtPointBC` --
    although similar cautionary behaviour would also have to be taken if using 
    :code:`nullspace` and :code:`transpose_nullspace` instead.
    
Finally, we form and solve the nonlinear variational problem for :math:`T_0 \in \{1, 10, 100, 1000, 10000 \}`::

    NLVP = NonlinearVariationalProblem(F, z, bcs=[bc_u]+aux_bcs)
    NLVS = NonlinearVariationalSolver(NLVP)
    
    (u, p, T_aux, l) = z.subfunctions
    File = VTKFile(f"output/boussinesq.pvd")
    
    for i in range(0, 5):
        T0.assign(10**(i))
        l.assign(Constant(T0))
        NLVS.solve()

        u_out = assemble(project(u, Z.sub(0)))
        p_out = assemble(project(p, Z.sub(1)))
        T_out = assemble(project(T, Z.sub(2)))

        u_out.rename("u")
        p_out.rename("p")
        T_out.rename("T")

        File.write(u_out, p_out, T_out, time=i)

The temperature and stream lines for :math:`T_0=1` and :math:`T_0=10000` are displayed below on the left and right respectively.

+-------------------------+-------------------------+
| .. image:: T0_1.png     | .. image:: T0_10000.png |
|    :width: 100%         |    :width: 100%         |
+-------------------------+-------------------------+

A Python script version of this demo can be found :demo:`here
<boussinesq.py>`.
