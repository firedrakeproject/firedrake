from firedrake import *
import numpy as np
import pytest


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_mesh_high_order():
    from netgen.geom2d import Circle, CSG2d
    geo = CSG2d()
    geo.Add(Circle(center=(0, 0), radius=1.0, mat="mat1", bc="circle"))
    ngmesh = geo.GenerateMesh(maxh=0.75)

    # Test that setting the degree in netgen_flags produces a high-order mesh
    order = 3
    mesh1 = Mesh(ngmesh, netgen_flags={"degree": order})
    assert mesh1.coordinates.function_space().ufl_element().degree() == order
    dim = mesh1.topological_dimension
    DG0 = FunctionSpace(mesh1, "DG", 0)
    markers = Function(DG0)

    # Test mesh refinement: 1 refinement
    markers.assign(1)
    mesh2 = mesh1.refine_marked_elements(markers)
    assert FunctionSpace(mesh1, "DG", 0).dim() * 2**dim == FunctionSpace(mesh2, "DG", 0).dim()
    # Test that refining a high-order mesh gives a high-order mesh
    assert mesh2.coordinates.function_space().ufl_element().degree() == order

    # Test mesh refinement: 2 refinements
    markers.assign(2)
    mesh3 = mesh1.refine_marked_elements(markers)
    assert FunctionSpace(mesh1, "DG", 0).dim() * 4**dim == FunctionSpace(mesh3, "DG", 0).dim()
    # Test that refining a high-order mesh gives a high-order mesh
    assert mesh3.coordinates.function_space().ufl_element().degree() == order


def square_geometry(h):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (np.pi, np.pi), bc="rect")
    ngmesh = geo.GenerateMesh(maxh=h)
    return ngmesh


def poisson(h, degree=2):
    import netgen
    comm = COMM_WORLD
    # Setting up Netgen geometry and mesh
    if comm.Get_rank() == 0:
        ngmesh = square_geometry(h)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "rect"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
        labels = None

    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)
    # Setting up the problem
    V = FunctionSpace(msh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(msh)
    f = assemble(interpolate(2*sin(x)*sin(y), V))
    a = inner(grad(u), grad(v))*dx
    l = inner(f, v) * dx
    u = Function(V)
    bc = DirichletBC(V, 0.0, labels)

    # Assembling matrix
    A = assemble(a, bcs=bc)
    b = assemble(l, bcs=bc)

    # Solving the problem
    solve(A, u, b, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Computing the error
    f.interpolate(sin(x)*sin(y))
    return sqrt(assemble(inner(u - f, u - f) * dx)), u, f


def poisson3D(h, degree=2):
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    import netgen

    comm = COMM_WORLD
    # Setting up Netgen geometry and mesh
    if comm.Get_rank() == 0:
        box = OrthoBrick(Pnt(0, 0, 0), Pnt(np.pi, np.pi, np.pi))
        box.bc("bcs")
        geo = CSGeometry()
        geo.Add(box)
        ngmesh = geo.GenerateMesh(maxh=h)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bcs"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
        labels = None

    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)

    # Setting up the problem
    V = FunctionSpace(msh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y, z = SpatialCoordinate(msh)
    f = assemble(interpolate(3*sin(x)*sin(y)*sin(z), V))
    a = inner(grad(u), grad(v))*dx
    l = inner(f, v) * dx
    u = Function(V)
    bc = DirichletBC(V, 0.0, labels)

    # Assembling matrix
    A = assemble(a, bcs=bc)
    b = assemble(l, bcs=bc)

    # Solving the problem
    solve(A, u, b, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Computing the error
    f.interpolate(sin(x)*sin(y)*sin(z))
    S = sqrt(assemble(inner(u - f, u - f) * dx))
    return S


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_poisson_2d():
    diff = np.array([poisson(h)[0] for h in [1/2, 1/4, 1/8]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_poisson_3d():
    diff = np.array([poisson3D(h) for h in [1, 1/2, 1/4]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


@pytest.mark.skipnetgen
def test_netgen_csg_2d_integral():
    from netgen.geom2d import SplineGeometry
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0, 0), (1, 1), bc="rect")
        ngmesh = geo.GenerateMesh(maxh=0.1)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "rect"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
        labels = None
    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)
    V = FunctionSpace(msh, "CG", 3)
    x, y = SpatialCoordinate(msh)
    f = assemble(interpolate(x*x+y*y*y+x*y, V))
    assert abs(assemble(f * dx) - (5/6)) < 1.e-10


@pytest.mark.skipnetgen
def test_netgen_csg_3d_integral():
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        box = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1))
        box.bc("bcs")
        geo = CSGeometry()
        geo.Add(box)
        ngmesh = geo.GenerateMesh(maxh=0.25)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bcs"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
        labels = None

    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)
    V = FunctionSpace(msh, "CG", 3)
    x, y, z = SpatialCoordinate(msh)
    f = assemble(interpolate(2 * x + 3 * y * y + 4 * z * z * z, V))
    assert abs(assemble(f * ds) - (2 + 4 + 2 + 5 + 2 + 6)) < 1.e-10


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_manifold():
    from netgen.csg import CSGeometry, Pnt, Sphere
    from netgen.meshing import MeshingParameters
    from netgen.meshing import MeshingStep
    import netgen

    comm = COMM_WORLD
    if comm.rank == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        mp = MeshingParameters(maxh=0.05, perfstepsend=MeshingStep.MESHSURFACE)
        ngmesh = geo.GenerateMesh(mp=mp)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    msh = Mesh(ngmesh)
    assert msh.topological_dimension == 2
    assert msh.geometric_dimension == 3

    V = FunctionSpace(msh, "CG", 3)
    f = assemble(interpolate(Constant(1), V))
    assert abs(assemble(f * dx) - 4*np.pi) < 1.e-2


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_occ_manifold():
    from netgen.occ import Pnt, SplineApproximation, Face, Wire, Axis, OCCGeometry, Z
    from netgen.meshing import MeshingStep
    R = 3.0
    r = 1.5
    surface_area = R*r*(2*pi)**2

    def Curve(t):
        return Pnt(0, R+r*np.cos(t), r*np.sin(t))

    n = 100
    pnts = [Curve(2*np.pi*t/n) for t in range(n+1)]

    spline = SplineApproximation(pnts)
    f = Face(Wire(spline))

    torus = f.Revolve(Axis((0, 0, 0), Z), 360)
    geo = OCCGeometry(torus, dim=3)
    ngmesh = geo.GenerateMesh(maxh=0.5, perfstepsend=MeshingStep.MESHSURFACE)

    msh = Mesh(ngmesh)
    assert msh.topological_dimension == 2
    assert msh.geometric_dimension == 3

    V = FunctionSpace(msh, "CG", 3)
    f = assemble(interpolate(Constant(1), V))
    assert abs(assemble(f * dx) - surface_area)/surface_area < 5.e-3


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_high_order_integral():
    from netgen.csg import CSGeometry, Pnt, Sphere
    import netgen

    comm = COMM_WORLD
    if comm.rank == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        ngmesh = geo.GenerateMesh(maxh=0.7)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    homsh = Mesh(ngmesh, netgen_flags={"degree": 2})
    V = FunctionSpace(homsh, "CG", 2)
    f = assemble(interpolate(Constant(1), V))
    assert abs(assemble(f * dx) - (4/3)*np.pi) < 1.e-2


@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_occ_adaptivity():
    from netgen.occ import WorkPlane, OCCGeometry, Axes
    from netgen.occ import X, Z

    def solve_poisson(mesh):
        V = FunctionSpace(mesh, "CG", 1)
        uh = Function(V, name="Solution")
        v = TestFunction(V)
        bc = DirichletBC(V, 0, "on_boundary")
        f = Constant(1)
        F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
        solve(F == 0, uh, bc)
        return uh

    def estimate_error(mesh, uh):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
        f = Constant(1)
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        v = CellVolume(mesh)

        # Compute error indicator cellwise
        G = inner(eta_sq / v, w)*dx
        G = G - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
        G = G - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS

        # Each cell is an independent 1x1 solve, so Jacobi is exact
        sp = {"mat_type": "matfree",
              "ksp_type": "richardson",
              "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(W)
        eta.interpolate(sqrt(eta_sq))  # the above computed eta^2

        with eta.dat.vec_ro as eta_:
            error_est = sqrt(eta_.dot(eta_))
        return (eta, error_est)

    def adapt(mesh, eta):
        W = FunctionSpace(mesh, "DG", 0)
        markers = Function(W)
        with eta.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        theta = 0.5
        should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
        markers.interpolate(should_refine)

        refined_mesh = mesh.refine_marked_elements(markers)
        return refined_mesh

    rect1 = WorkPlane(Axes((0, 0, 0), n=Z, h=X)).Rectangle(1, 2).Face()
    rect2 = WorkPlane(Axes((0, 1, 0), n=Z, h=X)).Rectangle(2, 1).Face()
    L = rect1 + rect2

    geo = OCCGeometry(L, dim=2)
    ngmsh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmsh)

    max_iterations = 10
    error_estimators = []
    dofs = []
    for i in range(max_iterations):
        uh = solve_poisson(mesh)
        (eta, error_est) = estimate_error(mesh, uh)
        error_estimators.append(error_est)
        dofs.append(uh.function_space().dim())
        if error_est < 0.05:
            break
        mesh = adapt(mesh, eta)
    assert error_estimators[-1] < 0.06


def _occ_periodic_square(maxh, directions="x"):
    from netgen.occ import Rectangle, OCCGeometry, X, Y, gp_Trsf, gp_Vec
    from netgen.meshing import IdentificationType
    shape = Rectangle(1, 1).Face()
    shape.edges.Min(X).name, shape.edges.Max(X).name = "left", "right"
    shape.edges.Min(Y).name, shape.edges.Max(Y).name = "bottom", "top"
    if "x" in directions:
        shape.edges.Min(X).Identify(shape.edges.Max(X), "px",
                                    IdentificationType.PERIODIC,
                                    gp_Trsf.Translation(gp_Vec(1, 0, 0)))
    if "y" in directions:
        shape.edges.Min(Y).Identify(shape.edges.Max(Y), "py",
                                    IdentificationType.PERIODIC,
                                    gp_Trsf.Translation(gp_Vec(0, 1, 0)))
    return OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh)


def _solve_periodic_helmholtz(mesh, uex):
    # Solve (I - div grad) u = f, where f is manufactured so the exact solution
    # is the periodic field uex; return the L2 error.
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    f = uex - div(grad(uex))
    a = (inner(u, v) + inner(grad(u), grad(v))) * dx
    L = inner(f, v) * dx
    uh = Function(V)
    solve(a == L, uh)
    return sqrt(assemble(inner(uh - uex, uh - uex) * dx))


@pytest.mark.skipnetgen
def test_netgen_periodic_square():
    # A periodic netgen mesh identifies the seam DOFs, so the periodic boundary
    # markers are absent and the geometry is recovered exactly.
    mesh = Mesh(_occ_periodic_square(0.1, directions="x"))
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - 1.0) < 1e-10

    x, y = SpatialCoordinate(mesh)
    # exact solution periodic in x; the seam continuity is essential to recover it.
    err = _solve_periodic_helmholtz(mesh, sin(2 * pi * x))
    assert err < 5e-2


@pytest.mark.skipnetgen
def test_netgen_periodic_square_both_directions():
    mesh = Mesh(_occ_periodic_square(0.1, directions="xy"))
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - 1.0) < 1e-10
    # both pairs identified: no exterior facets remain (a torus).
    assert mesh.exterior_facets.set.total_size == 0
    x, y = SpatialCoordinate(mesh)
    err = _solve_periodic_helmholtz(mesh, sin(2 * pi * x) * cos(2 * pi * y))
    assert err < 5e-2


@pytest.mark.skipnetgen
def test_netgen_periodic_cylinder():
    # Periodic along the axis of a cylinder: a curved-boundary periodic mesh.
    from netgen.occ import Cylinder, OCCGeometry, Pnt, gp_Trsf, gp_Vec, Z
    from netgen.meshing import IdentificationType
    cyl = Cylinder(Pnt(0, 0, 0), Z, r=1.0, h=1.0)
    cyl.faces.Min(Z).Identify(cyl.faces.Max(Z), "pz",
                              IdentificationType.PERIODIC,
                              gp_Trsf.Translation(gp_Vec(0, 0, 1)))
    mesh = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=0.15))
    # Volume approaches pi as the polygonal boundary is refined.
    assert abs(assemble(Constant(1.0) * dx(domain=mesh)) - pi) < 2e-2
    x, y, z = SpatialCoordinate(mesh)
    err = _solve_periodic_helmholtz(mesh, sin(2 * pi * z))
    assert err < 2e-1


@pytest.mark.skipnetgen
def test_netgen_periodic_too_coarse():
    # A mesh too coarse along the periodic direction produces seam-spanning cells
    # that collapse on merging; this must raise a clear error rather than build a
    # broken mesh.
    from netgen.occ import Box, OCCGeometry, X, gp_Trsf, gp_Vec, Pnt
    from netgen.meshing import IdentificationType
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    box.faces.Min(X).Identify(box.faces.Max(X), "px",
                              IdentificationType.PERIODIC,
                              gp_Trsf.Translation(gp_Vec(1, 0, 0)))
    ngmesh = OCCGeometry(box).GenerateMesh(maxh=0.4)
    with pytest.raises(ValueError, match="degenerate"):
        Mesh(ngmesh)


