Periodic meshes in Firedrake
============================

This tutorial was contributed by `Thomas Higham <mailto:Thomas.Higham@maths.ox.ac.uk>`__ and `Umberto Zerbinati <mailto:umberto.zerbinati@oriel.ox.ac.uk>`__.

The purpose of this demo is to summarise the support for periodic meshes in Firedrake.
Firedrake can build periodic meshes for simple one- and two-dimensional geometries and can also periodically extrude two-dimensional meshes into three-dimensional prism meshes.
Firedrake also has support for periodic tetrahedral meshes generated in Netgen.

We begin by importing the necessary libraries: ::
    
    from firedrake import *

1D Periodic Poisson Problem
---------------------------
We solve the 1D periodic Poisson problem

.. math::

    - \Delta u = \sin(x), \quad u(0) = u(2 \pi),

on :math:`\Omega = [0, 2 \pi]`. This problem has a trivial nullspace of constants; if we fix the constant to be zero then this problem has an analytical solution of :math:`u(x) = \sin(x)`.
We use the Firedrake function ``PeriodicIntervalMesh`` to build the mesh and implement the usual weak formulation of the Poisson equation: ::

    mesh = PeriodicIntervalMesh(32, 2*pi)
    x, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    uh = Function(V, name="Numerical")
    u_exact = sin(x)

    a = dot(grad(u), grad(v)) * dx
    L = inner(sin(x), v) * dx

    problem = LinearVariationalProblem(a, L, uh)

We use the ``VectorSpaceBasis`` function to tell PETSc that the problem has a
constant nullspace::

    nullspace = VectorSpaceBasis(
        constant=True,
        comm=mesh.comm,
    )

    solver = LinearVariationalSolver(
        problem,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
    )

    solver.solve()

    print(f"L2 error = {errornorm(u_exact, uh, norm_type='L2'):.3e}")

We plot the solution below.


.. figure:: Example1.png
   :align: center
   :alt: Finite element solution to 1D periodic Poisson problem.

   Finite element solution to 1D periodic Poisson problem.

2D Periodic Poisson Problem
---------------------------

In two dimensions, we can choose which boundaries are periodic. We consider
an :math:`x`-periodic Poisson problem, 

.. math::

    - \Delta u = (1 + \pi^2) \sin(x) \sin(\pi y), \quad u(0,y) = u(2 \pi, y), \quad u(x, 0) = u(x, 1) = 0,

on :math:`\Omega = [0, 2 \pi] \times [0,1]`. The nullspace is fixed for this problem. The exact solution is :math:`u(x,y) = \sin(x)\sin(\pi y)`.
We use the Firedrake function ``PeriodicRectangleMesh`` to build the mesh: if you call this function without calling the argument ``PeriodicRectangleMesh`` then the rectangle will be periodic in both :math:`x` and :math:`y`.
In our case we only want periodicity in :math:`x`. ::

    nx = 32
    ny = 16

    mesh = PeriodicRectangleMesh(nx, ny, 2*pi, 1.0, direction="x")
    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    uh = Function(V, name="Numerical")
    u_exact = sin(x) * sin(pi*y)

    f = (1 + pi**2)*sin(x)*sin(pi*y)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    # Homogeneous Dirichlet BCs on y = 0 and y = 1.
    bcs = DirichletBC(V, 0.0, (3, 4))

    solve(a == L, uh, bcs=bcs)

    print(f"L2 error = {errornorm(u_exact, uh, norm_type='L2'):.3e}")
    VTKFile("output/rectangle_poisson.pvd").write(uh)


We plot the solution below.


.. figure:: Example2.png
   :align: center
   :alt: Finite element solution to 2D periodic Poisson problem.

   Finite element solution to 2D periodic Poisson problem.

3D Periodic Helmholtz Problem
-----------------------------

For the simplest 3D shapes (cube and cuboid) Firedrake is able to generate tetrahedral periodic meshes using the commands ``PeriodicUnitCubeMesh`` and ``PeriodicBoxMesh`` respectively.
For cylindrical objects we have to construct a 2D cross-section and extrude, forming a mesh with prismatic elements. To construct any periodic mesh with tetrahedral elements we need to use Netgen, which is discussed in the next section.

For now we solve the 3D :math:`z`-periodic Helmholtz problem

.. math::

    (I - \Delta) u = f , \quad u(x,y,0) = u(x, y, 2 \pi), \quad u\vert_{\text{walls}} = 0,

on the cylindrical domain

.. math::

    \Omega =
    \left\{
    (x,y,z)\in\mathbb{R}^3 :
    x^2+y^2\le 1,\;
    0\le z\le 2\pi
    \right\}.

We can manufacture a right-hand side by choosing the solution
:math:`u_{\text{exact}}(x,y,z) = (1-x^2-y^2)\cos(z)` which gives :math:`f = (I - \Delta) u_{\text{exact}}`.


We build a 2D mesh for the desired cross section of the cylinder and then we use the command ``ExtrudedMesh`` with flag ``periodic=True`` to generate a mesh of prisms. ::

    refinements = 2
    base = UnitDiskMesh(refinements)

    mesh = ExtrudedMesh(
        base,
        layers=32,
        layer_height=2*pi/32,
        periodic=True,
    )

The two important arguments are ``layers`` and ``layer_height``, which tell Firedrake how far to extrude the mesh. By default ``layer_height`` is 1/``layers``. ::

    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    uh = Function(V, name="Numerical")
    u_exact = (1 - x*x - y*y) * cos(z)

    f = u_exact - div(grad(u_exact))

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(u, v) + inner(grad(u), grad(v))) * dx
    L = inner(f, v) * dx


    bc = DirichletBC(V, 0.0, "on_boundary")

    solve(
        a == L,
        uh,
        bcs=bc,
    )

    print(f"L2 error = {errornorm(u_exact, uh, norm_type='L2'):.3e}")
    VTKFile("output/cylinder_fd_Helmholtz.pvd").write(uh)

We plot the solution below.


.. figure:: Example3.png
   :align: center
   :alt: Cross-section of the finite element solution to a 3D periodic Helmholtz problem.

   Cross-section of the finite element solution to a 3D periodic Helmholtz problem using prismatic elements.

Periodic Meshes From Netgen
---------------------------

Netgen can help us generate a periodic mesh of tetrahedral elements.

Netgen can identify pairs of vertices lying on opposite boundaries of a geometry as being *the same* point.
When such a mesh is imported into Firedrake, the identified vertices are merged in the mesh topology, so that
a continuous (CG) function space automatically shares its degrees of freedom across the seam: the mesh is
genuinely **periodic**. This is exactly the representation Firedrake uses for its built-in ``PeriodicBoxMesh``, and it is also available for any Netgen geometry carrying
periodic identifications.

Identifications are declared on the geometry, before meshing, with the OCC
``Identify`` method:

.. code-block:: python

   shape_a.Identify(
       shape_b,
       name,
       IdentificationType.PERIODIC,
       transformation,
   )

where ``transformation`` is the rigid motion (typically a translation) that maps ``shape_a`` onto ``shape_b``.
Netgen then meshes the two boundaries compatibly and records the vertex pairs; Firedrake consumes them
automatically -- no extra flag on the ``Mesh`` constructor is required.

To construct a periodic cylinder of length :math:`2\pi` we identify the two end caps of the cylinder by a translation of :math:`2\pi` along ``z``. :: 

    from netgen.occ import Cylinder, OCCGeometry, Pnt, Z, gp_Trsf, gp_Vec
    from netgen.meshing import IdentificationType

    height = 2 * pi 
    cyl = Cylinder(Pnt(0, 0, 0), Z, r=1.0, h=height)

    # Label the lateral wall, then the two end caps that we will identify.
    for face in cyl.faces:
        face.name = "wall"
    cyl.faces.Min(Z).name = "bottom"
    cyl.faces.Max(Z).name = "top"

    # Identify the bottom cap with the top cap: a translation of 2*pi along Z
    cyl.faces.Min(Z).Identify(cyl.faces.Max(Z), "toroidal",
                              IdentificationType.PERIODIC,
                              gp_Trsf.Translation(gp_Vec(0, 0, height)))
    ngmsh = OCCGeometry(cyl).GenerateMesh(maxh=0.4)
    msh = Mesh(ngmsh)

.. warning::

   The mesh must contain at least a handful of cells along each periodic direction. If a single cell spans a
   whole period, its two ends are identified and the cell collapses; Firedrake then raises a ``ValueError``
   asking you to refine along the periodic direction. Here the axis has length :math:`2\pi` and ``maxh=0.4``
   gives roughly sixteen cells along it, which is ample. Only ``degree == 1`` periodic meshes are supported
   for now.

Because the two end caps have been identified, no boundary markers survive on them: the seam has become an
*interior* set of facets, and the only labelled boundary that remains is the lateral wall. This is what makes
a continuous field wrap around continuously in the ``z``-direction. We can verify the geometry survived the
merge intact --- the volume of the cylinder is :math:`\pi r^2 h = 2\pi^2`: ::

    volume = assemble(Constant(1.0) * dx(domain=msh))
    print(
        f"cylinder volume: {volume:.4f}  "
        f"(exact 2*pi**2 = {2*pi**2:.4f})"
    )

We solve again the same Helmholtz problem as in the previous section. ::

    V = FunctionSpace(msh, "CG", 2)
    x, y, z = SpatialCoordinate(msh)
    u_exact = cos(z) * (1 - x**2 - y**2)    
    f = u_exact - div(grad(u_exact))

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(u, v) + inner(grad(u), grad(v))) * dx
    L = inner(f, v) * dx
    
We look up the id of the ``"wall"`` boundary with ``GetRegionNames`` ::

    labels = [i + 1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name == "wall"]
    bc = DirichletBC(V, 0, labels)

    uh = Function(V)
    solve(a == L, uh, bcs=bc)

    print(f"L2 error = {errornorm(u_exact, uh, norm_type='L2'):.3e}")
    VTKFile("output/cylinder_helmholtz.pvd").write(uh)


We plot the solution below with a "crinkle cut" cross-section to inspect the solution.


.. figure:: Example4.png
   :align: center
   :alt: Finite element solution to 3D periodic Helmholtz problem using tetrahedral elements.

   Finite element solution to 3D periodic Helmholtz problem.

Netgen Tokamak Example
---------------------------

Helmholtz problems on periodic cylinders are used in simplified models of magnetic confinement fusion devices, where the straight cylinder acts as a large-aspect-ratio approximation of a torus. 
To obtain a more realistic geometry, we construct the domain in cylindrical coordinates and choose a cross-section representative of a tokamak plasma. After solving we transform the solution to Cartesian coordiantes. 

We describe the tokamak in cylindrical coordinates :math:`(R,\phi,Z)` where :math:`\phi` is the toroidal angle.
The cross-section is constructed in the :math:`(R,Z)` plane. Rather than extruding by :math:`\phi` we extrude by the corresponding toroidal arc-length 

.. math::

  s = R_0 \phi,

where :math:`R_0` is called the major radius. This makes one complete revolution of the tokamak have length :math:`2 \pi R_0`.
We parametrize the tokamak cross-section using the formula 

.. math::

   R(\theta) = R0 + a \cos(\theta + \sin^{-1}(\delta) \sin(\theta)), \quad Z(\theta) = a \kappa \sin(\theta).

where :math:`\delta` is the triangularity, :math:`\kappa` is the elongation, :math:`a` is the tokamak minor radius, and :math:`\theta \in [0, 2\pi]`.  ::

    from netgen.occ import (
        OCCGeometry, WorkPlane, Axes, Pnt, Z, X,
        gp_Trsf, gp_Vec
    )
    from netgen.meshing import IdentificationType
    import math

    # Geometry parameters - large aspect ratio tokamak
    R0 = 3.0
    a = 1.0
    kappa = 2.0
    delta = 0.3

    n_boundary = 40
    maxh = 0.5

    # Build a tokamak cross section and extrude periodically

    alpha = math.asin(delta)

    boundary_points = []
    for i in range(n_boundary):
        theta = 2.0 * pi * i / n_boundary
        R = R0 + a * math.cos(theta + alpha * math.sin(theta))
        Zc = kappa * a * math.sin(theta)
        boundary_points.append((R, Zc))

    wp = WorkPlane(Axes((0, 0, 0), n=Z, h=X))

    # Start at the first boundary point, then draw a closed polyline
    R0p, Z0p = boundary_points[0]
    wp.MoveTo(R0p, Z0p)
    for R, Zc in boundary_points[1:]:
        wp.LineTo(R, Zc)
    wp.Close()

    face = wp.Face()

    # Extrude in periodic phi direction and identify end caps.

    height = 2 * pi * R0
    solid = face.Extrude(height * Z)

    # Side wall(s)
    for f in solid.faces:
        f.name = "wall"

    bottom = solid.faces.Min(Z)
    top = solid.faces.Max(Z)
    bottom.name = "bottom"
    top.name = "top"

    # Periodic identification of the end caps
    bottom.Identify(
        top,
        "periodic_phi",
        IdentificationType.PERIODIC,
        gp_Trsf.Translation(gp_Vec(0, 0, height)),
    )

    # Mesh and convert to Firedrake
    ngmsh = OCCGeometry(solid).GenerateMesh(maxh=maxh)
    msh = Mesh(ngmsh, name="TokamakPeriodic")

The physical cylindrical coordinate convention is :math:`(R,\phi,Z)`.
However, the Netgen mesh stores its coordinate components in the order
:math:`(R,Z,s)`: the first two components are the coordinates of the
cross-section and the third is the toroidal arc-length. We therefore unpack
``SpatialCoordinate`` as ``R, Zc, s``.

We solve the Helmholtz problem

.. math::

    (I-\Delta)u=f, \quad u(r, 0, z) = u(r, 2 \pi R_0, z), \quad u|_{\text{walls}} = 0,

with source term
:math:`f(R,s,Z)=RZ\cos(s)`. ::

    V = FunctionSpace(msh, "CG", 2)
    R, Zc, s = SpatialCoordinate(msh)

    f = R * Zc * cos(s)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(u, v) + inner(grad(u), grad(v))) * dx
    L = inner(f, v) * dx

    labels = [
        i + 1
        for i, name in enumerate(ngmsh.GetRegionNames(codim=1))
        if name == "wall"
    ]
    bc = DirichletBC(V, 0, labels)

    uh = Function(V, name="Solution")
    solve(a == L, uh, bcs=bc)
    VTKFile("output/tokamak_helmholtz.pvd").write(uh)

We plot the output in :math:`(R, s, Z)`, a cylindrical representation:


.. figure:: Example5.png
   :align: center
   :width: 85%
   :alt: Finite element solution to 3D periodic Helmholtz problem in cylindrical tokamak geometry with tetrahedral elements.

   Finite element solution to 3D periodic Helmholtz problem in cylindrical tokamak geometry.

Since the solution is represented as a function on the mesh, we can obtain a
Cartesian visualisation by interpolating the cylindrical-to-Cartesian map

.. math::

    (R,s,Z) \longmapsto
    \bigl( R\cos(s / R_0), R\sin(s/ R_0), Z \bigr)

into the mesh coordinate field. The mesh components must again be unpacked in
their stored order, :math:`(R,Z,s)`::

    R, Zc, s = SpatialCoordinate(msh)
    msh.coordinates.interpolate(as_vector((
        R*cos(s/R0),
        R*sin(s/R0),
        Zc,
    )))
    VTKFile("TokamakCartesianSolution.pvd").write(uh)

We plot a cross-section "crinkle cut" of the solution in Cartesian coordinates:

.. figure:: Example6.png
   :align: center
   :width: 105%
   :alt: Finite element solution to 3D periodic Helmholtz problem in Cartesian tokamak geometry.

   Cross-section of a finite element solution to a 3D periodic Helmholtz problem in Cartesian tokamak geometry with tetrahedral elements.

